"""
qwen.py - merge the trained LoRA adapter into the base model.

    python qwen.py

Input:  trained_model/  (the adapter written by train_qlora.py)
Output: merged_qwen/    (a normal Hugging Face model, ready to convert to
                         GGUF for Ollama; see the README)

Runs on a CPU; no GPU is needed. Loading the model in float32 needs about
6 GB of free RAM.
"""

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from asta_prompt import build_chat

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ROOT = Path(__file__).resolve().parent
ADAPTER_PATH = ROOT / "trained_model"
OUTPUT_PATH = ROOT / "merged_qwen"

TEST_QUESTION = "Why is time slower near a black hole?"


def check_adapter():
    """Stop early with a clear message if the adapter is missing or mismatched."""
    config_file = ADAPTER_PATH / "adapter_config.json"
    if not config_file.exists():
        sys.exit(f"ERROR: no adapter found at {ADAPTER_PATH}. Run train_qlora.py first "
                 "and copy its trained_model/ folder here.")

    with open(config_file, "r", encoding="utf-8") as f:
        adapter_base = json.load(f).get("base_model_name_or_path")

    # An adapter only works with the exact base model it was trained on.
    if adapter_base != BASE_MODEL:
        sys.exit(f"ERROR: the adapter was trained on {adapter_base!r}, not {BASE_MODEL!r}.")


def test_generation(model, tokenizer):
    """Ask one question, using the same chat format as training, as a sanity check."""
    chat = build_chat([{"role": "user", "content": TEST_QUESTION}])
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"\nTest question: {TEST_QUESTION}\nAnswer: {answer}\n")

    if not answer:
        sys.exit("ERROR: the merged model produced an empty answer; not saving it.")


def main():
    check_adapter()

    # Load in float32: CPUs handle it reliably and the merge is more precise.
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # Add the LoRA matrices into the base weights, giving one ordinary model.
    print("Merging...")
    merged_model = model.merge_and_unload()
    merged_model.eval()

    test_generation(merged_model, tokenizer)

    # Save in float16 to halve the file size (about 3 GB).
    print("Saving merged model...")
    merged_model = merged_model.to(torch.float16)
    merged_model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Done. Merged model saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
