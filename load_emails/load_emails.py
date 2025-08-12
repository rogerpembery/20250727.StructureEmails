#!/usr/bin/env python3
"""
ETL: classified_emails.json ➔ Postgres + pgvector

- Extracts fields using mistral-nemo via Ollama REST API
- Embeds body + field values using Qwen3-Embedding-8B
- Deduplicates values semantically using cosine similarity
- In test mode (MAX_EMAILS set): writes test_output.json with extracted+matched fields
"""

import os, json, hashlib, datetime, logging, re, requests
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey,
    JSON, Text, select
)
from sqlalchemy.orm import declarative_base, relationship, Session
from pgvector.sqlalchemy import Vector

# ——— Config ———
DB_URL     = "postgresql+psycopg2://roger@localhost/emaildb"
OLLAMA     = "http://localhost:11434"
LLM_MODEL  = "mistral-nemo"
EMB_MODEL  = "dengcao/Qwen3-Embedding-8B:Q8_0"
EMB_DIM    = 4096
SIM_THRESH = 0.85
MAX_EMAILS = None

FILES = {
    "emails": Path("classified_emails.json"),
    "schema": Path("field_extraction_wip.json"),
}
LOG_FILE = "load_emails.txt"
TEST_OUTPUT_FILE = "test_output.json"
COMMON_FIELDS = {"subject", "sender", "recipient", "sending_date_time"}

# ——— Logging ———
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ——— ORM Models ———
Base = declarative_base()

class Email(Base):
    __tablename__ = "emails"
    id         = Column(String, primary_key=True)
    raw_json   = Column(JSON, nullable=False)
    clean_text = Column(Text)
    subject    = Column(Text)
    sender     = Column(Text)
    recipient  = Column(Text)
    sent_at    = Column(DateTime)
    category   = Column(String)
    text_emb   = Column("text_embedding", Vector(EMB_DIM))
    entities   = relationship("EmailEntity", back_populates="email")

class Entity(Base):
    __tablename__ = "entities"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    canonical_id = Column(Integer, ForeignKey("entities.id"))
    entity_type  = Column(String, nullable=False)
    value_raw    = Column(Text, nullable=False)
    value_norm   = Column(Text, nullable=False)
    embedding    = Column(Vector(EMB_DIM), nullable=False)

class EmailEntity(Base):
    __tablename__ = "email_entities"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    email_id   = Column(String, ForeignKey("emails.id"))
    entity_id  = Column(Integer, ForeignKey("entities.id"))
    field_name = Column(String, nullable=False)
    email      = relationship("Email", back_populates="entities")
    entity     = relationship("Entity")

# ——— Helpers ———
def sha256(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()

def norm(s: str) -> str:
    return " ".join(s.upper().split())

def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def embed(text: str) -> List[float]:
    r = requests.post(
        f"{OLLAMA}/api/embeddings",
        json={"model": EMB_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    vec = r.json().get("embedding", [])
    if not vec:
        raise ValueError("Empty embedding returned")
    return vec

def parse_date(s: str):
    for fmt in ("%A, %d %B %Y at %H:%M:%S", "%Y-%m-%d %H:%M"):
        try: return datetime.datetime.strptime(s, fmt)
        except Exception: pass
    return None

MAX_TOKENS = 8000
def estimate_tokens(text): return len(text) // 4

def llm_extract(email_text: str, field_keys: List[str]) -> Dict[str, Any]:
    instruction = (
        "You are an extraction assistant. "
        "From the following email body, return ONLY valid JSON with keys "
        f"{field_keys}. Omit keys that cannot be found. "
        "Do NOT wrap the JSON in markdown fences.\n\nEMAIL:\n"
    )
    available_tokens = MAX_TOKENS - estimate_tokens(instruction)
    trimmed_text = email_text
    while estimate_tokens(trimmed_text) > available_tokens:
        trimmed_text = trimmed_text[:int(len(trimmed_text) * 0.9)]

    prompt = instruction + trimmed_text

    for attempt in range(5):
        try:
            res = requests.post(
                f"{OLLAMA}/api/generate",
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                timeout=180,
            )
            res.raise_for_status()
            result = res.json()
            if "response" not in result:
                log.warning("LLM returned no 'response': %s", result)
                continue
            cleaned = _clean_llm_output(result["response"])
            return json.loads(cleaned)
        except Exception as ex:
            log.warning("LLM retry %d failed: %s", attempt + 1, ex)
    log.error("LLM failed 5× on input: %s", trimmed_text[:120])
    return {}

_json_re = re.compile(r"\{.*\}", re.S)
def _clean_llm_output(txt: str) -> str:
    txt = re.sub(r"```(?:json)?", "", txt, flags=re.I)
    txt = txt.replace("\\n", " ").replace("\\t", " ").replace("\n", " ")
    m = _json_re.search(txt)
    return m.group(0) if m else "{}"

# ——— Main ———
def main():
    engine = create_engine(DB_URL, future=True)
    Base.metadata.create_all(engine)

    emails_raw = json.loads(FILES["emails"].read_text())
    schema     = json.loads(FILES["schema"].read_text())

    with Session(engine) as ses:
        done_ids = {r[0] for r in ses.execute(select(Email.id))}
    todo = [(sha256(e), e) for e in emails_raw if sha256(e) not in done_ids]

    if MAX_EMAILS is not None:
        todo = todo[:MAX_EMAILS]

    if not todo:
        print("All emails already ingested.")
        return

    test_rows = []

    with Session(engine) as ses, tqdm(
        total=len(todo),
        dynamic_ncols=True,
        unit="email",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for eid, e in todo:
            try:
                category = e.get("category")
                field_keys = COMMON_FIELDS.copy()
                if category and category in schema:
                    field_keys |= {k for ex in schema[category] for k in ex}
                else:
                    log.info("Unknown category %s; using common fields only", category)

                extracted = llm_extract(e.get("clean_text", ""), sorted(field_keys))
                full_embedding = embed(e.get("clean_text", ""))

                em = Email(
                    id=eid,
                    raw_json=e,
                    clean_text=e.get("clean_text"),
                    subject=e.get("subject"),
                    sender=e.get("sender"),
                    recipient=e.get("recipient"),
                    sent_at=parse_date(e.get("sending_date_time", "")),
                    category=category,
                    text_emb=full_embedding,
                )
                ses.add(em)

                test_fields = {}

                for fname, raw_val in extracted.items():
                    values = raw_val if isinstance(raw_val, list) else [raw_val]
                    for v in values:
                        if v is None or str(v).strip().lower() in {"none", ""}:
                            continue
                        if isinstance(v, dict):
                            v = json.dumps(v, ensure_ascii=False)

                        v_norm = norm(str(v))
                        vec = embed(v_norm)

                        stmt = (
                            select(Entity)
                            .where(Entity.entity_type == fname)
                            .order_by(1 - (Entity.embedding.l2_distance(vec)))
                            .limit(1)
                        )
                        hit = ses.execute(stmt).scalar_one_or_none()
                        if hit and cosine(hit.embedding, vec) >= SIM_THRESH:
                            canon = hit if hit.canonical_id is None else ses.get(Entity, hit.canonical_id)
                        else:
                            canon = Entity(
                                canonical_id=None,
                                entity_type=fname,
                                value_raw=v,
                                value_norm=v_norm,
                                embedding=vec,
                            )
                            ses.add(canon)
                            ses.flush()
                            canon.canonical_id = canon.id

                        ses.add(EmailEntity(
                            email_id=eid,
                            entity_id=canon.id,
                            field_name=fname
                        ))

                        if MAX_EMAILS:
                            test_fields[fname] = {
                                "raw": v,
                                "matched": canon.value_raw
                            }

                ses.commit()
                log.info("Inserted email %s", eid)

                if MAX_EMAILS:
                    test_rows.append({
                        "id": eid,
                        "subject": e.get("subject"),
                        "category": category,
                        "extracted_fields": test_fields,
                        "embedding_preview": full_embedding[:15],
                    })

            except Exception as ex:
                ses.rollback()
                log.warning("Failed email %s: %s", eid, ex)
            pbar.update(1)

    if MAX_EMAILS:
        Path(TEST_OUTPUT_FILE).write_text(json.dumps(test_rows, indent=2))
        print(f"\n🧪 Test output saved to: {TEST_OUTPUT_FILE}")

    print("Ingestion complete – check load_emails.log for details.")

if __name__ == "__main__":
    main()
