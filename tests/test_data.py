"""
Tests for the dataset: the helper functions in data/prepare_data.py, and the
files it produced (train.jsonl, val.jsonl and the cleaned topic files).

If a data test fails after editing the data, re-run:
    python data/prepare_data.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import prepare_data
from prepare_data import find_leaks, question_key, stratified_split, to_snake_case

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record(topic, question, answer):
    return {"topic": topic, "question": question, "answer": answer}


# ----- helper functions -----

def test_question_key_ignores_case_and_punctuation():
    assert question_key("What is a light-year?") == question_key("what is a light year")


def test_to_snake_case():
    assert to_snake_case("Cosmology") == "cosmology"
    assert to_snake_case("Black Holes") == "black_holes"
    assert to_snake_case("stars_stellar_evolution") == "stars_stellar_evolution"


def test_find_leaks_spots_repeated_question():
    train = [record("t", "What is a star?", "A ball of gas.")]
    val = [record("t", "what is a STAR", "Something else.")]
    assert len(find_leaks(train, val)) == 1


def test_find_leaks_spots_repeated_answer():
    train = [record("t", "Why do stars twinkle?", "The atmosphere bends starlight.")]
    val = [record("t", "Why do stars appear to twinkle?", "The atmosphere bends starlight.")]
    assert len(find_leaks(train, val)) == 1


def test_stratified_split_is_80_20_per_topic_and_repeatable():
    records = [record(topic, f"{topic} question {i}?", f"{topic} answer {i}")
               for topic in ("a", "b") for i in range(50)]

    train, val = stratified_split([dict(r) for r in records])
    train2, val2 = stratified_split([dict(r) for r in records])

    assert Counter(r["topic"] for r in val) == {"a": 10, "b": 10}
    assert len(train) == 80
    assert [r["question"] for r in val] == [r["question"] for r in val2]


# ----- the generated files -----

def test_training_files_are_in_chat_format():
    for path in (DATA_DIR / "train.jsonl", DATA_DIR / "val.jsonl"):
        for row in load_jsonl(path):
            roles = [m["role"] for m in row["messages"]]
            assert roles == ["user", "assistant"]
            assert all(m["content"].strip() for m in row["messages"])


def test_no_topic_prefix_in_questions():
    # The old format added "answer astronomy question about {topic}:".
    for row in load_jsonl(DATA_DIR / "train.jsonl"):
        assert not row["messages"][0]["content"].lower().startswith("answer astronomy question")


def test_no_leakage_between_train_and_validation():
    train = [record(r["topic"], r["messages"][0]["content"], r["messages"][1]["content"])
             for r in load_jsonl(DATA_DIR / "train.jsonl")]
    val = [record(r["topic"], r["messages"][0]["content"], r["messages"][1]["content"])
           for r in load_jsonl(DATA_DIR / "val.jsonl")]
    assert find_leaks(train, val) == []


def test_no_duplicate_questions_in_working_files():
    questions = []
    for path in DATA_DIR.glob("*_500.json"):
        with open(path, "r", encoding="utf-8") as f:
            questions += [question_key(item["question"]) for item in json.load(f)]
    assert len(questions) == len(set(questions))


def test_topic_labels_are_snake_case():
    for row in load_jsonl(DATA_DIR / "train.jsonl") + load_jsonl(DATA_DIR / "val.jsonl"):
        assert re.fullmatch(r"[a-z0-9_]+", row["topic"])


def test_every_answer_fix_was_applied():
    with open(prepare_data.FIXES_FILE, "r", encoding="utf-8") as f:
        fixes = json.load(f)
    answers = {question_key(r["messages"][0]["content"]): r["messages"][1]["content"]
               for path in ("train.jsonl", "val.jsonl") for r in load_jsonl(DATA_DIR / path)}
    for fix in fixes:
        assert answers[question_key(fix["question"])] == fix["answer"]
