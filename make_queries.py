# Builds the query set in two LLM steps. Step 1 writes one specific question per Urdu passage (Urdu
# script and English). Step 2 takes that Urdu question and writes three Roman Urdu spellings of the
# SAME sentence: same words, same order, only letter choices differ. The first version of this set
# let the model write all four forms at once, and about a quarter of the Roman variants changed
# words, so spelling and wording were mixed up (see v1/ and the README). Questions that point at
# "this passage" are dropped. Every query's gold answer is the Urdu passage it came from.
import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from fixes import normalize_roman

ROOT = Path(__file__).parent
PASSAGES = ROOT / "data" / "passages.jsonl"
DRAFTS = ROOT / "cache" / "queries"
VARIANTS = ROOT / "cache" / "variants"
OUT_JSONL = ROOT / "queries.jsonl"
OUT_CSV = ROOT / "queries.csv"
REVIEW = ROOT / "review_sample.csv"
N_QUERIES = 300
N_DEV = 100
POOL = 430
SEED = 11
MODEL = "gpt-4o"

PROMPT = """You are helping build a retrieval benchmark for Pakistani users.

Read this Urdu passage and write ONE question that this passage answers and that other passages
about similar topics would not. Name the specific entity (person, place, event) in the question so
it can be matched to this passage. The question must make sense on its own: never refer to "this
passage", "this article", "here" or address the reader as "you".

Return the question in natural Urdu script (urdu) and in English (english).

Passage:
{text}
"""

VARIANT_PROMPT = """Here is an Urdu question and its English meaning.

Urdu: {urdu}
English: {english}

Write exactly 3 Roman Urdu (Latin letters) renderings of THIS question, the way three different
Pakistanis would spell it when typing quickly. Hard rules:
- Same words, same order, same meaning in all three. Do not add, drop, reorder or replace words,
  and do not swap a word for a synonym or an English word that is not in the Urdu question.
- Only the spelling may differ between them (for example kya / kia / kyaa, hai / hay / he,
  mein / main / may, aghaaz / aaghaz, vowel length, c vs k, z vs j, shortened forms like k for ke).
- Each must differ from the other two in at least one word's spelling.
- Keep names and loanwords recognisable. No Urdu script, no punctuation changes beyond the question
  mark."""

LEAK = re.compile(
    r"\bthis (passage|article|text|page)\b|\bin the passage\b|\bthe passage\b|\byou\b|\byour\b"
    r"|اس مضمون|اس مقالے|اس متن|اس عبارت|اس تحریر|یہاں|آپ|^سوال",
    re.IGNORECASE,
)


class QueryDraft(BaseModel):
    urdu: str
    english: str


class Variants(BaseModel):
    roman_variants: list[str]


def draft(client: OpenAI, passage: dict) -> dict | None:
    path = DRAFTS / f"{passage['id']}.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        return {"gold_id": passage["id"], "urdu": old["urdu"], "english": old["english"]}
    resp = client.chat.completions.parse(
        model=MODEL,
        temperature=0.7,
        response_format=QueryDraft,
        messages=[{"role": "user", "content": PROMPT.format(text=passage["text"])}],
    )
    p = resp.choices[0].message.parsed
    row = {"gold_id": passage["id"], "urdu": p.urdu.strip(), "english": p.english.strip()}
    DRAFTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return row


def spelling_ok(variants: list[str]) -> bool:
    # Three distinct strings with the same number of words that still share almost all their words
    # once spelling is normalised. That is the property the first version of the set lacked.
    if len(variants) != 3 or len({v.lower() for v in variants}) < 3:
        return False
    sets = [set(normalize_roman(v).split()) for v in variants]
    lengths = [len(v.split()) for v in variants]
    if max(lengths) - min(lengths) > 0:
        return False
    return all(len(a & b) / len(a | b) >= 0.6 for i, a in enumerate(sets) for b in sets[i + 1 :])


def spell_variants(client: OpenAI, d: dict) -> list[str] | None:
    path = VARIANTS / f"{d['gold_id']}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for _ in range(3):
        resp = client.chat.completions.parse(
            model=MODEL,
            temperature=0.7,
            response_format=Variants,
            messages=[
                {
                    "role": "user",
                    "content": VARIANT_PROMPT.format(urdu=d["urdu"], english=d["english"]),
                }
            ],
        )
        variants = [v.strip() for v in resp.choices[0].message.parsed.roman_variants]
        if spelling_ok(variants):
            VARIANTS.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(variants, ensure_ascii=False), encoding="utf-8")
            return variants
    return None


def build(client: OpenAI, passage: dict) -> dict | None:
    d = draft(client, passage)
    if d is None or LEAK.search(d["urdu"]) or LEAK.search(d["english"]):
        return None
    variants = spell_variants(client, d)
    if variants is None:
        return None
    return {**d, "roman_1": variants[0], "roman_2": variants[1], "roman_3": variants[2]}


def main() -> None:
    load_dotenv()
    client = OpenAI(max_retries=5)
    urdu = [json.loads(line) for line in PASSAGES.open(encoding="utf-8")]
    urdu = [p for p in urdu if p["lang"] == "ur"]
    random.Random(SEED).shuffle(urdu)
    with ThreadPoolExecutor(max_workers=8) as ex:
        built = [b for b in ex.map(lambda p: build(client, p), urdu[:POOL]) if b]
    print(f"{len(built)} valid of {POOL} passages tried")
    drafts = built[:N_QUERIES]
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
