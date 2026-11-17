"""Tests for the prompt contract in asta_prompt.py and the Ollama Modelfile."""

import re
from pathlib import Path

import pytest

from asta_prompt import IGNORE_INDEX, SYSTEM_PROMPT, build_chat, make_labels

MODELFILE = Path(__file__).resolve().parent.parent / "Modelfile"


def test_build_chat_puts_system_prompt_first():
    chat = build_chat([{"role": "user", "content": "What is a comet?"}])
    assert chat[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert chat[1] == {"role": "user", "content": "What is a comet?"}


def test_build_chat_does_not_change_its_input():
    messages = [{"role": "user", "content": "Hi"}]
    build_chat(messages)
    assert messages == [{"role": "user", "content": "Hi"}]


def test_modelfile_system_prompt_matches_training_prompt():
    # Ollama cannot import Python, so the Modelfile repeats the prompt.
    # If the two drift apart, the chatbot is no longer prompted like training.
    text = MODELFILE.read_text(encoding="utf-8")
    match = re.search(r'^SYSTEM """(.*?)"""', text, re.DOTALL | re.MULTILINE)
    assert match, "Modelfile has no SYSTEM block"
    assert match.group(1) == SYSTEM_PROMPT


def test_make_labels_masks_only_the_prompt():
    prompt_ids = [10, 11, 12]
    full_ids = [10, 11, 12, 50, 51]
    assert make_labels(prompt_ids, full_ids) == [IGNORE_INDEX] * 3 + [50, 51]


def test_make_labels_rejects_prompt_that_is_not_a_prefix():
    with pytest.raises(ValueError):
        make_labels([10, 11, 99], [10, 11, 12, 50])


def test_make_labels_rejects_example_with_no_answer():
    with pytest.raises(ValueError):
        make_labels([10, 11], [10, 11])
