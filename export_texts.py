# Writes cache/texts_wiki.jsonl: every text run_benchmark.py embeds for the Wikipedia set (passages,
# the five query forms, the normalised and transliterated Roman Urdu queries), so the newer
# embedders can run on a GPU box with embed_remote.py. Needs no API key: transliterations come from
# the cache.
import json

from fixes import normalize_roman, select_rules, transliterate
from run_benchmark import FORMS, ROMAN, ROOT, load_data
from translit_m2m import transliterate_m2m
from translit_open import transliterate_open

passages, queries = load_data()
dev = [q for q in queries if q["split"] == "dev"]
test = [q for q in queries if q["split"] == "test"]
rules = select_rules([[q[f] for f in ROMAN] for q in dev])
rows = [{"kind": "passage", "text": p["text"]} for p in passages]
for f in FORMS:
    rows += [{"kind": "query", "text": q[f]} for q in test]
for f in ROMAN:
    roman = [q[f] for q in test]
    extra = [normalize_roman(t, rules) for t in roman] + transliterate(roman)
    extra += transliterate_open(roman, cached_only=True) + transliterate_m2m(
        roman, cached_only=True
    )
    rows += [{"kind": "query", "text": t} for t in extra if t]
out = ROOT / "cache" / "texts_wiki.jsonl"
with out.open("w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "texts ->", out)
