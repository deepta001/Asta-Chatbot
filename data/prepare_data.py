"""
prepare_data.py - clean the Asta Q&A dataset and split it for training.

Pipeline:
    raw_backup/*_500.json  (original files, never modified)
        -> check fields, tidy whitespace, normalise topic labels
        -> remove duplicate questions (within and across topics)
        -> apply hand-checked answer fixes from answer_fixes.json
        -> write cleaned working files: data/*_500.json
        -> stratified 80/20 split by topic (fixed seed)
        -> leakage check (no question may appear in both splits)
        -> write train.jsonl, val.jsonl, duplicate_review.json and
           data_quality_report.txt

Run from anywhere:
    python data/prepare_data.py

Only the Python standard library is used.
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Paths are relative to this script's folder, so the script works no matter
# which directory it is run from.
DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw_backup"
FIXES_FILE = DATA_DIR / "answer_fixes.json"

TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
REVIEW_FILE = DATA_DIR / "duplicate_review.json"
REPORT_FILE = DATA_DIR / "data_quality_report.txt"

SEED = 42                # fixed seed so the split is identical on every run
VAL_FRACTION = 0.2       # 80% train / 20% validation
SHORT_ANSWER_WORDS = 6   # answers with fewer words than this are reported

REQUIRED_FIELDS = ["topic", "question", "answer"]


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def tidy_text(text):
    """Strip the ends and collapse repeated spaces/newlines into one space."""
    return " ".join(text.split())


def to_snake_case(label):
    """'Cosmology' -> 'cosmology', 'Black Holes' -> 'black_holes'."""
    label = label.strip().lower()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    return label.strip("_")


def question_key(question):
    """
    Turn a question into a key used to spot duplicates.

    Lowercasing and removing punctuation means that "What is a light-year?"
    and "what is a light year" are treated as the same question.
    """
    key = question.lower()
    key = re.sub(r"[^a-z0-9]+", " ", key)
    return " ".join(key.split())


def word_count(text):
    return len(text.split())


# ---------------------------------------------------------------------------
# Step 1: load and validate the raw files
# ---------------------------------------------------------------------------

def load_raw_records(report):
    """
    Read every raw_backup/*_500.json file.

    Returns a list of records. Each record also remembers which file it came
    from, so the cleaned working files can be written back in the same layout.
    Records with missing or empty fields are skipped and counted.
    """
    raw_files = sorted(RAW_DIR.glob("*_500.json"))
    if not raw_files:
        sys.exit(f"ERROR: no *_500.json files found in {RAW_DIR}")

    records = []
    missing_field_count = 0
    raw_topic_labels = Counter()

    for path in raw_files:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        if not isinstance(items, list):
            sys.exit(f"ERROR: {path.name} does not contain a JSON list")

        report.append(f"  {path.name}: {len(items)} records")

        for item in items:
            # Every record must have all three fields as non-empty strings.
            is_valid = isinstance(item, dict) and all(
                isinstance(item.get(field), str) and item[field].strip()
                for field in REQUIRED_FIELDS
            )
            if not is_valid:
                missing_field_count += 1
                continue

            raw_topic_labels[item["topic"]] += 1
            records.append({
                "source_file": path.name,
                "topic": to_snake_case(item["topic"]),
                "question": tidy_text(item["question"]),
                "answer": tidy_text(item["answer"]),
            })

    return records, missing_field_count, raw_topic_labels


# ---------------------------------------------------------------------------
# Step 2: remove duplicate questions
# ---------------------------------------------------------------------------

def remove_duplicates(records):
    """
    Keep one record per question.

    Rule for choosing which copy to keep: the one with the LONGEST answer,
    because it usually carries the most information. If two answers are the
    same length, the first one (in file order) wins, so the result is always
    the same.

    Returns the kept records and a review list describing every duplicate
    group, so a human can check the choices.
    """
    groups = defaultdict(list)
    for record in records:
        groups[question_key(record["question"])].append(record)

    kept = []
    review = []

    for key, group in groups.items():
        # max() returns the first record with the longest answer.
        best = max(group, key=lambda r: len(r["answer"]))
        kept.append(best)

        if len(group) > 1:
            different_answers = {question_key(r["answer"]) for r in group}
            review.append({
                "question": best["question"],
                "copies": len(group),
                # True means the copies do not say the same thing word for
                # word, so they need a human check for contradictions.
                "answers_differ": len(different_answers) > 1,
                "kept_answer": best["answer"],
                "all_versions": [
                    {"topic": r["topic"], "answer": r["answer"]} for r in group
                ],
            })

    return kept, review


# ---------------------------------------------------------------------------
# Step 3: apply hand-checked answer fixes
# ---------------------------------------------------------------------------

def apply_answer_fixes(records):
    """
    Replace answers listed in answer_fixes.json.

    Each fix was written and checked by hand (see its "reason" field). If a
    fix does not match any question, the script stops, because that means a
    typo in the fixes file would otherwise be silently ignored.
    """
    with open(FIXES_FILE, "r", encoding="utf-8") as f:
        fixes = json.load(f)

    records_by_key = {question_key(r["question"]): r for r in records}

    for fix in fixes:
        key = question_key(fix["question"])
        if key not in records_by_key:
            sys.exit(f"ERROR: answer fix does not match any question: {fix['question']}")
        records_by_key[key]["answer"] = tidy_text(fix["answer"])

    return len(fixes)


# ---------------------------------------------------------------------------
# Step 4: write the cleaned working files
# ---------------------------------------------------------------------------

def write_working_files(records):
    """Write data/*_500.json in the original format (topic, question, answer)."""
    by_file = defaultdict(list)
    for record in records:
        by_file[record["source_file"]].append({
            "topic": record["topic"],
            "question": record["question"],
            "answer": record["answer"],
        })

    for file_name, items in sorted(by_file.items()):
        with open(DATA_DIR / file_name, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Step 5: stratified train/validation split
# ---------------------------------------------------------------------------

def stratified_split(records):
    """
    Split each topic separately, 80/20.

    Splitting per topic ("stratified") keeps the same topic mix in both sets,
    so the validation loss is a fair test across all eight topics.
    """
    by_topic = defaultdict(list)
    for record in records:
        by_topic[record["topic"]].append(record)

    rng = random.Random(SEED)
    train, val = [], []

    # Sort topics so the order (and therefore the random draws) is fixed.
    for topic in sorted(by_topic):
        topic_records = by_topic[topic]
        rng.shuffle(topic_records)
        val_size = round(len(topic_records) * VAL_FRACTION)
        val.extend(topic_records[:val_size])
        train.extend(topic_records[val_size:])

    # Some questions are reworded versions of each other with exactly the
    # same answer (e.g. "Why do stars twinkle when viewed from Earth?" and
    # "Why do stars appear to twinkle?"). If one is in train and the other in
    # val, the model has effectively already seen the val answer. Move such
    # val records into train so each pair stays in one split.
    train_answers = {question_key(r["answer"]) for r in train}
    clean_val = []
    for record in val:
        if question_key(record["answer"]) in train_answers:
            train.append(record)
        else:
            clean_val.append(record)

    return train, clean_val


# ---------------------------------------------------------------------------
# Step 6: leakage check
# ---------------------------------------------------------------------------

def find_leaks(train, val):
    """
    Return validation records whose question OR answer also appears
    (normalised) in train. Either one means the model saw it in training.
    """
    train_questions = {question_key(r["question"]) for r in train}
    train_answers = {question_key(r["answer"]) for r in train}
    leaks = []
    for record in val:
        if (question_key(record["question"]) in train_questions
                or question_key(record["answer"]) in train_answers):
            leaks.append(record["question"])
    return leaks


# ---------------------------------------------------------------------------
# Step 7: write chat-format JSONL
# ---------------------------------------------------------------------------

def write_jsonl(records, path):
    """
    One JSON object per line, in the chat "messages" format that
    tokenizer.apply_chat_template() expects.

    The system prompt is NOT stored here: the training script adds it, so it
    is defined in exactly one place.
    """
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            line = {
                "topic": record["topic"],
                "messages": [
                    {"role": "user", "content": record["question"]},
                    {"role": "assistant", "content": record["answer"]},
                ],
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    report = ["ASTA DATA QUALITY REPORT", "=" * 40, "", "Raw files:"]

    records, missing_count, raw_labels = load_raw_records(report)
    total_raw = len(records) + missing_count

    kept, review = remove_duplicates(records)
    duplicates_removed = len(records) - len(kept)
    conflicting = [g for g in review if g["answers_differ"]]

    fixes_applied = apply_answer_fixes(kept)
    short_answers = [r for r in kept if word_count(r["answer"]) < SHORT_ANSWER_WORDS]

    write_working_files(kept)

    train, val = stratified_split(kept)
    leaks = find_leaks(train, val)
    if leaks:
        # This should be impossible after de-duplication; stop if it happens.
        sys.exit(f"ERROR: {len(leaks)} validation records also appear in train")

    write_jsonl(train, TRAIN_FILE)
    write_jsonl(val, VAL_FILE)

    with open(REVIEW_FILE, "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)

    # ----- build the report -----
    report += [
        "",
        f"Records before cleaning:        {total_raw}",
        f"Records with missing fields:    {missing_count}",
        f"Duplicate groups found:         {len(review)}",
        f"Duplicate records removed:      {duplicates_removed}",
        f"Groups with differing answers:  {len(conflicting)} (see duplicate_review.json)",
        f"Hand-checked answer fixes:      {fixes_applied} (see answer_fixes.json)",
        f"Records after cleaning:         {len(kept)}",
        "",
        "Topic labels (raw -> normalised):",
    ]
    for label, count in sorted(raw_labels.items()):
        report.append(f"  {label!r} -> {to_snake_case(label)!r} ({count})")

    train_topics = Counter(r["topic"] for r in train)
    val_topics = Counter(r["topic"] for r in val)
    report += ["", f"{'Topic':<26}{'Train':>7}{'Val':>7}{'Total':>7}"]
    for topic in sorted(train_topics):
        t, v = train_topics[topic], val_topics[topic]
        report.append(f"{topic:<26}{t:>7}{v:>7}{t + v:>7}")
    report.append(f"{'TOTAL':<26}{len(train):>7}{len(val):>7}{len(kept):>7}")

    report += [
        "",
        f"Split: stratified by topic, {int((1 - VAL_FRACTION) * 100)}/{int(VAL_FRACTION * 100)}, seed {SEED}",
        f"Leakage check (val question or answer found in train): {len(leaks)}",
        "",
        f"Answers shorter than {SHORT_ANSWER_WORDS} words after fixes: {len(short_answers)}",
    ]
    for r in short_answers:
        report.append(f"  [{r['topic']}] {r['question']} -> {r['answer']}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    # Print only the summary numbers (plain ASCII, so the Windows console
    # does not fail on symbols such as the square-root sign).
    print(f"Records before cleaning:   {total_raw}")
    print(f"Missing-field records:     {missing_count}")
    print(f"Duplicates removed:        {duplicates_removed}")
    print(f"Conflicting groups flagged: {len(conflicting)}")
    print(f"Answer fixes applied:      {fixes_applied}")
    print(f"Records after cleaning:    {len(kept)}")
    print(f"Train / val:               {len(train)} / {len(val)}")
    print(f"Leakage (val Q or A in train): {len(leaks)}")
    print(f"Full report: {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
