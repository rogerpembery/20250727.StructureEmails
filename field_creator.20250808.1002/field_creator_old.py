#!/usr/bin/env python3

import json, logging, re, requests
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ────────────── Config ─────────────
INPUT_FILE           = "classified_emails.json"
OUTPUT_FILE          = "extracted_schema.json"
WIP_FILE             = "field_extraction_wip.json"
MAX_FIELDS_PER_CLASS = 10
MAX_EMAILS           = 0
MAX_RETRIES          = 5
OLLAMA_HOST          = "http://localhost:11434"
OLLAMA_MODEL         = "mistral-nemo"
LOG_FAILED_LLM_RESPONSES = True
FAILED_LOG_PATH      = "failed_llm_responses.log"

# ────────────── Logging ─────────────
logging.basicConfig(
    filename="field_creator.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ────────────── Ollama API ─────────────
def query_ollama(prompt: str) -> str:
    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        res.raise_for_status()
        return res.json()["response"]
    except requests.RequestException as e:
        logging.error("Ollama API error: %s", e)
        raise

# ────────────── Input Parser ─────────────
def normalize_input(data):
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        out = defaultdict(list)
        for item in data:
            klass = item.get("classification") or item.get("category")
            text  = item.get("clean_text")
            if klass and text:
                out[klass].append(text)
            else:
                logging.warning("Skipping malformed item: %s", item)
        return dict(out)
    raise TypeError("Invalid input format")

# ────────────── Field Extraction ─────────────
FIELD_PROMPT_TEMPLATE = """
Extract up to 10 key-value fields as raw JSON from the following email.
Only include meaningful, business-relevant fields.
Do NOT include any markdown, explanation, or non-JSON output.

EMAIL:
\"\"\"
{email}
\"\"\"
"""

def extract_fields(email_text: str) -> dict:
    prompt = FIELD_PROMPT_TEMPLATE.format(email=email_text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = query_ollama(prompt)

            raw = re.sub(r"^```json|```$", "", raw.strip(), flags=re.I).strip()
            raw = raw.replace("\\n", " ").replace("\\t", " ").strip()
            raw = re.sub(r",\s*([}\]])", r"\1", raw)

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                logging.warning("Attempt %d: JSON parse failed, using regex fallback.", attempt)
                pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', raw)
                parsed = dict(pairs)

            if isinstance(parsed, dict) and parsed:
                cleaned = {k.strip(): v.strip()
                           for k, v in parsed.items()
                           if isinstance(k, str) and isinstance(v, str)}
                logging.info("Recovered %d fields on attempt %d.", len(cleaned), attempt)
                return cleaned

        except Exception as exc:
            logging.warning("Attempt %d failed: %s", attempt, exc)

    fallback = {"raw_first_line": email_text.splitlines()[0][:120].strip()}
    logging.error("All attempts failed. Using fallback.")

    if LOG_FAILED_LLM_RESPONSES:
        with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- Failed field extraction at {datetime.now().isoformat()} ---\n")
            f.write(prompt + "\n\n--- Raw response ---\n")
            f.write(raw + "\n")

    return fallback

# ────────────── Aggregation & Resume ─────────────
def collect_raw_fields(emails_by_class):
    obs = defaultdict(list)

    if Path(WIP_FILE).exists():
        obs.update(json.loads(Path(WIP_FILE).read_text()))
        logging.info("Loaded existing WIP state with %d classifications", len(obs))

    total_classes = len(emails_by_class)

    for idx, (klass, emails) in enumerate(emails_by_class.items(), start=1):
        if klass in obs:
            logging.info("Skipping %s (already processed)", klass)
            continue

        subset = emails[:MAX_EMAILS] if MAX_EMAILS else emails
        bar_desc = f"[{idx}/{total_classes}] {klass:>20}"
        logging.info("Processing %d/%d – %s (%d emails)",
                     idx, total_classes, klass, len(subset))

        obs[klass] = []
        for txt in tqdm(subset, desc=bar_desc, unit="email"):
            obs[klass].append(extract_fields(txt))

        # Save progress to WIP file
        Path(WIP_FILE).write_text(json.dumps(obs, indent=2))
        logging.info("Saved WIP after %s", klass)

    return obs

def summarise_usage(field_dicts):
    summary = defaultdict(lambda: {"count": 0, "examples": set()})
    for fd in field_dicts:
        for k, v in fd.items():
            summary[k]["count"] += 1
            if len(summary[k]["examples"]) < 3:
                summary[k]["examples"].add(v)
    for meta in summary.values():
        meta["examples"] = list(meta["examples"])
    return dict(summary)

# ────────────── LLM Schema Description ─────────────
TOP_FIELD_PROMPT = """
You are designing a data schema for emails of type "{klass}".
Below is a dictionary of field names, how many times each occurred, and sample values.

Select the {n} most important fields and write a 50–75 word description for each.
If multiple fields are semantically equivalent (e.g., 'order_id', 'Order Number', 'reference'), combine them into one field using a consistent, readable name.
Do not list duplicates. Base grouping on meaning, not formatting.

Respond as a JSON list like:
[
  {{ "name": "...", "description": "..." }},
  ...
]
NO markdown, no preamble – just raw JSON.

INPUT:
{summary}
"""

def select_top_fields(summary, klass):
    prompt = TOP_FIELD_PROMPT.format(
        klass=klass,
        n=MAX_FIELDS_PER_CLASS,
        summary=json.dumps(summary, indent=2)
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = query_ollama(prompt)
            raw = re.sub(r"^```json|```$", "", resp.strip(), flags=re.I).strip()
            data = json.loads(raw)

            if isinstance(data, list):
                logging.info("✅ Top fields selected for '%s' on attempt %d", klass, attempt)
                return data

        except Exception as e:
            logging.warning("Attempt %d: Top-field selection failed for '%s': %s", attempt, klass, e)

            if LOG_FAILED_LLM_RESPONSES:
                with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- Failed schema response for {klass} (attempt {attempt}) at {datetime.now().isoformat()} ---\n")
                    f.write(prompt + "\n\n--- Raw response ---\n")
                    f.write(resp + "\n")

    logging.error("❌ All attempts failed to select top fields for '%s'. Returning empty list.", klass)
    return []

def shared_field_names(per_class):
    sets = [set(f["name"] for f in lst) for lst in per_class.values() if lst]
    return set.intersection(*sets) if sets else set()

# ────────────── Main Pipeline ─────────────
def main():
    emails = normalize_input(json.loads(Path(INPUT_FILE).read_text()))
    raw    = collect_raw_fields(emails)

    per_class = {}
    for klass, dicts in raw.items():
        top = select_top_fields(summarise_usage(dicts), klass)
        per_class[klass] = top

        Path(OUTPUT_FILE).write_text(json.dumps({"classifications": per_class}, indent=2))
        logging.info("Wrote schema so far (after %s)", klass)

    shared = shared_field_names(per_class)

    final = {
        "shared_fields": [],
        "classifications": {}
    }
    for klass, fields in per_class.items():
        for f in fields:
            if f["name"] in shared:
                final["shared_fields"].append(f)
            else:
                final["classifications"].setdefault(klass, []).append(f)

    Path(OUTPUT_FILE).write_text(json.dumps(final, indent=2))
    logging.info("✅ Final schema written to %s", OUTPUT_FILE)

if __name__ == "__main__":
    main()
