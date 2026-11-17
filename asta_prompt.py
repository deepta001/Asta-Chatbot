"""
asta_prompt.py - the Asta "prompt contract".

The model only behaves well if it is asked questions in the same format it
was trained on. This file defines that format in ONE place:

    system message:  SYSTEM_PROMPT (below)
    user message:    the plain question, with no extra prefix
    assistant:       the answer

It is used by:
    train_qlora.py  - builds every training example with build_chat()
    qwen.py         - test question after merging the adapter
    chat_ui.py      - every request sent to Ollama

The Ollama Modelfile cannot import Python, so it repeats SYSTEM_PROMPT.
tests/test_prompt.py checks the two copies are identical.
"""

SYSTEM_PROMPT = (
    "You are Asta, a friendly astronomy assistant. "
    "Answer questions about astronomy clearly and accurately."
)

# Hugging Face ignores label positions set to -100 when computing the loss.
IGNORE_INDEX = -100


def build_chat(messages):
    """Put the Asta system prompt in front of a list of user/assistant messages."""
    return [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)


def make_labels(prompt_ids, full_ids):
    """
    Build training labels so the loss is only computed on the answer.

    prompt_ids: tokens for system prompt + question + "assistant" header
    full_ids:   tokens for the whole conversation including the answer

    The prompt tokens are replaced by IGNORE_INDEX, so the model learns to
    produce answers rather than to repeat questions.
    """
    prompt_length = len(prompt_ids)

    # If the full conversation does not start with exactly the prompt tokens,
    # masking by length would hide the wrong tokens. Stop instead of training
    # on silently wrong labels.
    if full_ids[:prompt_length] != prompt_ids:
        raise ValueError("The prompt tokens are not a prefix of the full conversation tokens.")
    if len(full_ids) == prompt_length:
        raise ValueError("The conversation has no answer tokens to train on.")

    return [IGNORE_INDEX] * prompt_length + full_ids[prompt_length:]
