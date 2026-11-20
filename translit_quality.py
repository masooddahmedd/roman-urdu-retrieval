# How close is each transliterator's output to the Urdu-script version of the same question? Both
# test sets have that reference (our Urdu question; Belebele's urd_Arab question), so we report
# word overlap (Jaccard of folded tokens), character similarity (difflib ratio on folded text),
# the share of outputs containing non-Arabic letters (Devanagari, Cyrillic, ...) and seconds per
# query from the caches. Writes translit_quality.json.
import json
import re
import statistics
from difflib import SequenceMatcher

from belebele import load as load_bb
from fixes import transliterate
from retrievers import tokenize
from run_benchmark import ROMAN, ROOT, load_data
from translit_m2m import transliterate_m2m
from translit_open import CACHE as OPEN_CACHE
from translit_open import transliterate_open

# Letters outside Arabic script and Latin (Latin is allowed: digits, K2, DVD stay as they are).
FOREIGN = re.compile(r"[ऀ-ॿЀ-ӿঀ-৿਀-૿一-鿿]")


def scores(outputs: list[str], refs: list[str]) -> dict:
    jac, char = [], []
    for o, r in zip(outputs, refs):
        a, b = set(tokenize(o)), set(tokenize(r))
        jac.append(len(a & b) / len(a | b) if a | b else 1.0)
        char.append(SequenceMatcher(None, " ".join(tokenize(o)), " ".join(tokenize(r))).ratio())
    return {
        "n": len(outputs),
        "word_jaccard": statistics.mean(jac),
        "char_similarity": statistics.mean(char),
        "exact_words": sum(j >= 0.999 for j in jac) / len(jac),
        "foreign_script": sum(bool(FOREIGN.search(o)) for o in outputs) / len(outputs),
    }


ARMS = {"gpt-4o-mini": transliterate, "qwen2.5-7b": transliterate_open, "m2m100": transliterate_m2m}
out = {"wiki": {}, "belebele": {}}
_, queries = load_data()
test = [q for q in queries if q["split"] == "test"]
for arm, fn in ARMS.items():
    outs, refs = [], []
    for f in ROMAN:
        outs += fn([q[f] for q in test], cached_only=True)
        refs += [q["urdu"] for q in test]
    if all(outs):
        out["wiki"][arm] = scores(outs, refs)
_, bbq = load_bb()
for arm, fn in ARMS.items():
    outs = fn([q["roman"] for q in bbq], cached_only=True)
    if all(outs):
        out["belebele"][arm] = scores(outs, [q["urdu"] for q in bbq])

# Speed. Qwen2.5-7B: Ollama on an RTX 4070 Ti SUPER with 20 of 28 layers on the GPU (the rest of the
# card was in use). m2m100: laptop CPU, batch 16, beam 4. gpt-4o-mini latency was not recorded.
open_cache = json.loads(OPEN_CACHE.read_text(encoding="utf-8"))
m2m_cache = json.loads((ROOT / "cache" / "translit_m2m.json").read_text(encoding="utf-8"))
out["seconds_per_query"] = {
    "qwen2.5-7b_median": statistics.median(open_cache.get("_seconds", {}).values() or [0]),
    "m2m100_cpu_median": statistics.median(m2m_cache.get("_seconds_per_query_cpu") or [0]),
}
(ROOT / "translit_quality.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
