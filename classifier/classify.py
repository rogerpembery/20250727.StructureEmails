#!/usr/bin/env python3
import json
import logging
import requests
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime, timedelta
import math
import random
import re
import string
import sys
import time

# ────────────── Config ─────────────
MAX_EMAILS_PER_CLASS = 5
MAX_FIELDS_PER_CLASS = 30
FINAL_FIELDS_PER_CLASS = 15
SHARED_FIELDS_COUNT = 4
CLASS_SPECIFIC_COUNT = 6
INCLUDE_RAW_EMAIL_TEXT = False

EMBED_THRESHOLD = 0.9
LLM_MODEL = "ikiru/dolphin-mistral-24b-venice-edition"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_HOST = "http://localhost:11434"

# ────────────── Directories ─────────────
LOG_DIR = Path("logs")
OUTPUT_DIR = Path("outputs")
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TS = datetime.now().strftime("%Y-%m-%d_%H%M")
RUN_LOG_FILE = LOG_DIR / f"field_creator_run_{TS}.txt"
LLM_LOG_FILE = LOG_DIR / f"field_creator_llm_responses_{TS}.txt"
FINAL_FIELDS_FILE = OUTPUT_DIR / f"final_fields_{TS}.json"
FIELD_MAPPING_FILE = OUTPUT_DIR / f"field_mapping_{TS}.json"

# ────────────── Logging ─────────────
file_handler = logging.FileHandler(RUN_LOG_FILE, mode="w", encoding="utf-8")
console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[file_handler, console_handler]
)

def start_progress_mode():
    """Disable console output from logging so only progress() writes to stdout."""
    logging.getLogger().removeHandler(console_handler)

def end_progress_mode():
    """Restore console logging after a progress-only section."""
    logging.getLogger().addHandler(console_handler)

# ────────────── Helpers ─────────────
def normalise_name(name: str) -> str:
    name = name.lower()
    name = name.replace("_", " ")
    name = re.sub(rf"[{re.escape(string.punctuation)}]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def progress(stage, total_stages, class_idx, class_total, class_name,
             email_idx, emails_in_class, start_time, stage_label):
    elapsed = time.time() - start_time
    emails_done = (class_idx - 1) * emails_in_class + email_idx
    total_emails_stage = class_total * emails_in_class
    avg_time_per_email = elapsed / max(1, emails_done)
    remaining_time = avg_time_per_email * (total_emails_stage - emails_done)
    eta_str = str(timedelta(seconds=int(remaining_time)))
    percent = (emails_done / total_emails_stage) * 100
    sys.stdout.write(
        "\r" + " " * 160 + "\r" +
        f"Stage {stage}/{total_stages}: {stage_label} | "
        f"Class {class_idx}/{class_total} ({class_name}) | "
        f"Email {email_idx}/{emails_in_class} | "
        f"{percent:.1f}% | ETA: {eta_str}"
    )
    sys.stdout.flush()

def query_llm(prompt):
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=300
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()

def embed_texts(texts):
    res = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120
    )
    res.raise_for_status()
    data = res.json()
    if "embedding" in data:
        if len(texts) == 1:
            return data["embedding"]
        return [data["embedding"] for _ in texts]
    if "data" in data:
        if len(texts) == 1:
            return data["data"][0]["embedding"]
        return [d["embedding"] for d in data["data"]]
    raise ValueError(f"Unexpected embeddings format: {data}")

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

def extract_json_list(text):
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"'''", "", text)
    text = text.replace("\\n", " ").replace("\\t", " ").strip()
    match = re.search(r"\[.*?\]", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return []

def semantic_merge(fields_with_counts, threshold):
    if not fields_with_counts:
        return {}
    names = list(fields_with_counts.keys())
    embeddings = embed_texts(names)
    used = set()
    merged = {}
    for i, name in enumerate(names):
        if name in used:
            continue
        total_count = fields_with_counts[name]
        used.add(name)
        for j in range(i+1, len(names)):
            if names[j] in used:
                continue
            if cosine_sim(embeddings[i], embeddings[j]) >= threshold:
                total_count += fields_with_counts[names[j]]
                used.add(names[j])
        merged[name] = total_count
    return merged

# ────────────── Main ─────────────
def main():
    input_file = Path("classified_emails.json")
    logging.info(f"Reading emails from: {input_file.resolve()}")
    data = json.loads(input_file.read_text(encoding="utf-8"))

    raw_data = defaultdict(list)
    for item in data:
        klass = item.get("classification") or item.get("category")
        fields = item.get("fields") or {}
        if klass and isinstance(fields, dict):
            raw_data[klass].append(item)

    total_classes = len(raw_data)
    logging.info(f"Found {total_classes} classifications.")

    fields_stage1 = {}
    start_stage1 = time.time()

    # Stage 1: Extract & merge within classification
    start_progress_mode()
    with open(LLM_LOG_FILE, "w", encoding="utf-8") as llm_log:
        for class_idx, (klass, emails) in enumerate(raw_data.items(), start=1):
            examples = random.sample(emails, min(len(emails), MAX_EMAILS_PER_CLASS))
            prompt_header = (
                f"You are analysing emails from classification type: {klass}.\n"
                f"Your task: Identify up to {MAX_FIELDS_PER_CLASS} of the most useful fields "
                "that could link these emails to other classifications.\n"
                "Focus on identifiers, dates, references, and customer details useful for joins.\n"
                "Output MUST be a pure JSON array of field names, nothing else.\n\n"
                "Here are example emails:\n"
            )
            prompt = prompt_header
            for email_idx, ex in enumerate(examples, start=1):
                progress(1, 3, class_idx, total_classes, klass,
                         email_idx, len(examples), start_stage1, "Extracting fields")
                part = {"fields": ex.get("fields", {})}
                if INCLUDE_RAW_EMAIL_TEXT and "body" in ex:
                    part["body"] = ex["body"]
                prompt += json.dumps(part, indent=2) + "\n"

            response = query_llm(prompt)
            cleaned_fields = [normalise_name(f) for f in extract_json_list(response)]
            llm_log.write(f"### {klass} ###\n--- PROMPT SENT ---\n{prompt}\n\n")
            llm_log.write("--- RAW LLM RESPONSE ---\n" + response + "\n\n")
            llm_log.write("--- CLEANED JSON OUTPUT ---\n" + json.dumps(cleaned_fields, indent=2) + "\n\n")

            counts = Counter()
            for f in cleaned_fields:
                for email in emails:
                    if f in {normalise_name(k) for k in email.get("fields", {}).keys()}:
                        counts[f] += 1
            merged_counts = semantic_merge(counts, EMBED_THRESHOLD)
            top_fields = dict(Counter(merged_counts).most_common(FINAL_FIELDS_PER_CLASS))
            fields_stage1[klass] = top_fields
    sys.stdout.write("\n")  # drop to next line after stage
    end_progress_mode()

    # Stage 2: Global merge
    logging.info("Stage 2/3: Merging fields globally...")
    all_counts = Counter()
    for counts in fields_stage1.values():
        all_counts.update(counts)
    merged_global = semantic_merge(all_counts, EMBED_THRESHOLD)
    shared_fields = [f for f, _ in Counter(merged_global).most_common(SHARED_FIELDS_COUNT)]
    logging.info(f"Shared fields selected: {shared_fields}")

    # Stage 3: Final selection
    start_stage3 = time.time()
    start_progress_mode()
    for class_idx, (klass, counts) in enumerate(fields_stage1.items(), start=1):
        progress(3, 3, class_idx, total_classes, klass, 1, 1, start_stage3, "Selecting final fields")
        remaining = {f: c for f, c in counts.items() if f not in shared_fields}
        top_specific = [f for f, _ in Counter(remaining).most_common(CLASS_SPECIFIC_COUNT)]
        fields_stage1[klass] = shared_fields + top_specific
    sys.stdout.write("\n")
    end_progress_mode()

    # Save results
    final_fields = {"shared_fields": shared_fields, "classifications": fields_stage1}
    FINAL_FIELDS_FILE.write_text(json.dumps(final_fields, indent=2), encoding="utf-8")
    FIELD_MAPPING_FILE.write_text(json.dumps({}, indent=2), encoding="utf-8")
    logging.info(f"✅ Final schema written to {FINAL_FIELDS_FILE}")
    logging.info(f"✅ Field mapping written to {FIELD_MAPPING_FILE}")

    print("\nShared fields:", shared_fields)
    for klass, fields in final_fields["classifications"].items():
        print(f"\n{klass}: {fields}")

if __name__ == "__main__":
    main()
