# Drafts the query set. For 300 sampled Urdu passages an LLM writes one specific question in Urdu
# script, the same question in English, and three Roman Urdu spellings of it as different people
# might type them. Every query's gold answer is the Urdu passage it came from. The queries are LLM
# drafted, a native speaker spot-checks a 50 query sample (review_sample.csv), and the README
# says so.
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

ROOT = Path(__file__).parent
PASSAGES = ROOT / "data" / "passages.jsonl"
CACHE = ROOT / "cache" / "queries"
OUT_JSONL = ROOT / "queries.jsonl"
OUT_CSV = ROOT / "queries.csv"
REVIEW = ROOT / "review_sample.csv"
N_QUERIES = 300
N_DEV = 100
SEED = 11
MODEL = "gpt-4o"

PROMPT = """You are helping build a retrieval benchmark for Pakistani users.

Read this Urdu passage and write ONE question that this passage answers and that other passages
about similar topics would not. Name the specific entity (person, place, event) in the question so
it can be matched to this passage.

Return:
- urdu: the question in natural Urdu script.
- english: the same question in English.
- roman_variants: exactly 3 spellings of the question in Roman Urdu (Latin letters), the way three
  different Pakistanis would type it in a chat or search box. They must differ from each other in
  spelling (for example kya / kia / kyaa, hai / hay / he, mein / main / may), and must not be
  English translations. Keep names recognisable.

Passage:
{text}
"""


class QueryDraft(BaseModel):
    urdu: str
    english: str
    roman_variants: list[str]


def draft(client: OpenAI, passage: dict) -> dict | None:
    path = CACHE / f"{passage['id']}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    resp = client.chat.completions.parse(
        model=MODEL,
        temperature=0.7,
        response_format=QueryDraft,
        messages=[{"role": "user", "content": PROMPT.format(text=passage["text"])}],
    )
    parsed = resp.choices[0].message.parsed
    variants = [v.strip() for v in parsed.roman_variants]
    if len(variants) != 3 or len(set(v.lower() for v in variants)) < 3:
        return None
    row = {
        "gold_id": passage["id"],
        "urdu": parsed.urdu.strip(),
        "english": parsed.english.strip(),
        "roman_1": variants[0],
        "roman_2": variants[1],
        "roman_3": variants[2],
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return row


def main() -> None:
    load_dotenv()
    client = OpenAI(max_retries=5)
    urdu = [json.loads(line) for line in PASSAGES.open(encoding="utf-8")]
    urdu = [p for p in urdu if p["lang"] == "ur"]
    rng = random.Random(SEED)
    rng.shuffle(urdu)
    # Oversample a little because a draft is dropped if the three spellings are not distinct.
    pool = urdu[: int(N_QUERIES * 1.15)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        drafts = [d for d in ex.map(lambda p: draft(client, p), pool) if d]
    drafts = drafts[:N_QUERIES]
    for i, d in enumerate(drafts):
        d["id"] = f"q{i:03d}"
        d["split"] = "dev" if i < N_DEV else "test"
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    cols = ["id", "split", "gold_id", "urdu", "english", "roman_1", "roman_2", "roman_3"]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: d[c] for c in cols} for d in drafts)

    # The review sample is drawn from both splits so the check also covers the test queries.
    sample = random.Random(SEED + 1).sample(drafts, 50)
    review_cols = [
        "id",
        "urdu",
        "english",
        "roman_1",
        "roman_2",
        "roman_3",
        "natural_ok",
        "fixed_roman_1",
        "fixed_roman_2",
        "fixed_roman_3",
        "notes",
    ]
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=review_cols)
        w.writeheader()
        for d in sample:
            w.writerow({c: d.get(c, "") for c in review_cols})
    print(f"{len(drafts)} queries, {sum(d['split'] == 'dev' for d in drafts)} dev")


if __name__ == "__main__":
    main()
