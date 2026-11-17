"""
train_qlora.py - fine-tune Qwen2.5-1.5B-Instruct on the Asta dataset with QLoRA.

This replaces the training cells of the Colab notebook Asta-V.ipynb.

On an NVIDIA GPU (e.g. a free Colab T4) it trains with QLoRA. In Colab:

    !git clone <your repository URL>
    %cd Asta-Chatbot
    !pip install -r requirements-train.txt
    !python train_qlora.py

On a computer without a GPU it falls back to plain LoRA, because 4-bit
quantisation (bitsandbytes) needs an NVIDIA GPU. This works but is slow:
expect several hours.

Input:  data/train.jsonl and data/val.jsonl (written by data/prepare_data.py)
Output: trained_model/  (the LoRA adapter and tokenizer, used by qwen.py)

QLoRA in one sentence: the base model is loaded with 4-bit weights to save
GPU memory and kept frozen, and only small LoRA matrices added to some of its
layers are trained.
"""

import json
import sys
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from asta_prompt import build_chat, make_labels

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "data" / "train.jsonl"
VAL_FILE = ROOT / "data" / "val.jsonl"
CHECKPOINT_DIR = ROOT / "results"      # temporary checkpoints during training
OUTPUT_DIR = ROOT / "trained_model"    # final adapter, read by qwen.py

SEED = 42
# Precision of the base weights when training on a CPU (no GPU).
CPU_DTYPE = torch.float32
# The longest example (system prompt + question + answer) is far below this.
# Longer examples are skipped rather than cut off, so no answer is truncated.
MAX_LENGTH = 256

# LoRA settings (the same as the notebook's first run).
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Which layers get LoRA matrices.
# - Attention only (what the notebook used): fewer trainable parameters,
#   faster, and directly comparable with the current Asta model.
# - Attention + MLP: roughly 4 times more trainable parameters. The MLP
#   layers are thought to store more factual knowledge, so this can learn the
#   dataset better, at the cost of more memory and time and a higher risk of
#   overfitting a small dataset.
TRAIN_MLP_LAYERS = False
ATTENTION_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]

# Training hyperparameters. Effective batch size = 2 x 4 = 8.
EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

SAMPLE_QUESTIONS = [
    "Why is time slower near a black hole?",
    "What is a neutron star?",
    "Why do stars twinkle?",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Read a JSON Lines file (one JSON object per line) into a list."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_example(example, tokenizer):
    """
    Turn one {"messages": [user, assistant]} example into model inputs.

    The chat template is applied twice: once without the answer (ending with
    the "assistant" header, exactly what the model sees when chatting) and
    once with it. The length of the first tells us how many tokens to mask.
    """
    chat = build_chat(example["messages"])

    prompt_text = tokenizer.apply_chat_template(
        chat[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": make_labels(prompt_ids, full_ids),
    }


def tokenize_dataset(examples, tokenizer, name):
    """Tokenize a list of examples, skipping any longer than MAX_LENGTH."""
    tokenized = []
    skipped = 0
    for example in examples:
        item = tokenize_example(example, tokenizer)
        if len(item["input_ids"]) > MAX_LENGTH:
            skipped += 1
            continue
        tokenized.append(item)

    longest = max(len(item["input_ids"]) for item in tokenized)
    print(f"{name}: {len(tokenized)} examples, longest {longest} tokens, skipped {skipped}")
    return tokenized


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(use_gpu):
    """Load the frozen base model and attach trainable LoRA matrices."""
    if use_gpu:
        # QLoRA: store the frozen base weights in 4 bits to save GPU memory.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # The T4 GPU does not support bfloat16, so compute in float16.
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # Plain LoRA on the CPU: full-precision base weights (about 6 GB of RAM).
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=CPU_DTYPE)
        # Recompute activations during the backward pass instead of storing
        # them all, which saves a lot of memory.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # The generation cache is not used in training and conflicts with
    # gradient checkpointing.
    model.config.use_cache = False

    target_modules = ATTENTION_LAYERS + (MLP_LAYERS if TRAIN_MLP_LAYERS else [])
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def answer_question(model, tokenizer, question):
    """Generate an answer using exactly the chat format used in training."""
    chat = build_chat([{"role": "user", "content": question}])
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False, use_cache=True)

    # Only decode the newly generated tokens, not the prompt.
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for path in (TRAIN_FILE, VAL_FILE):
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Run data/prepare_data.py first.")

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)} - training with QLoRA")
    else:
        print("No GPU found - training with plain LoRA on the CPU. This will take hours.")

    # Print versions so a successful run can be pinned in the requirements files.
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, peft {peft.__version__}")

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = tokenize_dataset(load_jsonl(TRAIN_FILE), tokenizer, "train")
    val_data = tokenize_dataset(load_jsonl(VAL_FILE), tokenizer, "validation")

    model = load_model(use_gpu)

    # Both settings give an effective batch size of 8. The GPU uses small
    # batches to fit in memory; the CPU is faster with one batch of 8.
    if use_gpu:
        batch_size, accumulation_steps = BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
    else:
        batch_size, accumulation_steps = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, 1

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # Keep the checkpoint with the lowest validation loss, not just the last.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        # Mixed precision and the 8-bit optimiser both need a GPU.
        fp16=use_gpu,
        optim="paged_adamw_8bit" if use_gpu else "adamw_torch",
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        # Pads each batch to its longest example. Labels are padded with -100
        # so padding never counts towards the loss.
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    print(f"Final validation loss: {trainer.evaluate()['eval_loss']:.4f}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")

    model.eval()
    for question in SAMPLE_QUESTIONS:
        print(f"\nQ: {question}\nA: {answer_question(model, tokenizer, question)}")


if __name__ == "__main__":
    main()
