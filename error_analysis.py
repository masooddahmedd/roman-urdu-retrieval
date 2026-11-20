# Why do Roman Urdu queries still fail after transliteration? For each retriever and each test
# question/spelling we compare three ranks of the gold passage (Urdu-script question, raw Roman
# Urdu, transliterated Roman Urdu) and how close the transliteration is to the original Urdu
# question (token overlap after the same Arabic-letter folding BM25 uses). Runs from the
# embedding and transliteration caches only. Writes error_analysis.json and error_examples.md.
import json
from difflib import SequenceMatcher

import numpy as np

from fixes import transliterate
from retrievers import build_retrievers, tokenize
from run_benchmark import ROMAN, ROOT, K, gold_ranks, load_data

passages, queries = load_data()
test = [q for q in queries if q["split"] == "test"]
index_of = {p["id"]: i for i, p in enumerate(passages)}
gold = [index_of[q["gold_id"]] for q in test]
translit = {f: transliterate([q[f] for q in test], cached_only=True) for f in ROMAN}


def overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def char_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, " ".join(tokenize(a)), " ".join(tokenize(b))).ratio()


sims = {f: [overlap(t, q["urdu"]) for t, q in zip(translit[f], test)] for f in ROMAN}
csims = {f: [char_sim(t, q["urdu"]) for t, q in zip(translit[f], test)] for f in ROMAN}
all_sims = np.concatenate([sims[f] for f in ROMAN])
BUCKETS = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]

out = {
    "translit_vs_urdu_token_jaccard": {
        "mean": float(all_sims.mean()),
        "exact_share": float((all_sims >= 0.999).mean()),
        "char_ratio_mean": float(np.mean(np.concatenate([csims[f] for f in ROMAN]))),
    }
}
examples = []
for name, retr in build_retrievers(passages, ["bm25", "bge-m3", "e5", "openai"]).items():
    r_urdu = gold_ranks(retr.scores([q["urdu"] for q in test]), gold)
    rows = []
    for f in ROMAN:
        r_rom = gold_ranks(retr.scores([q[f] for q in test]), gold)
        r_tr = gold_ranks(retr.scores(translit[f]), gold)
        for i in range(len(test)):
            rows.append((i, f, r_urdu[i], r_rom[i], r_tr[i], sims[f][i]))
    n = len(rows)
    rom_miss = [r for r in rows if r[3] > K]
    tr_miss = [r for r in rows if r[4] > K]
    res = {
        "pairs": n,
        "roman_misses": len(rom_miss),
        "roman_misses_recovered_by_translit": sum(r[4] <= K for r in rom_miss),
        "translit_misses": len(tr_miss),
        "translit_misses_where_urdu_also_misses": sum(r[2] > K for r in tr_miss),
        "translit_hits_where_roman_hit_but_translit_missed": sum(
            r[3] <= K and r[4] > K for r in rows
        ),
        "translit_recall_by_overlap_bucket": {},
    }
    for lo, hi in BUCKETS:
        sel = [r for r in rows if lo <= r[5] < hi]
        res["translit_recall_by_overlap_bucket"][f"{lo:.1f}-{min(hi, 1.0):.1f}"] = {
            "n": len(sel),
            "recall@5": float(np.mean([r[4] <= K for r in sel])) if sel else None,
        }
    out[name] = res
    if name == "bge-m3":
        for i, f, ru, rr, rt, s in sorted(tr_miss, key=lambda r: r[5])[:12]:
            q = test[i]
            examples.append(
                f"- {q['id']} {f}: ranks urdu {ru}, roman {rr}, translit {rt}, overlap {s:.2f}\n"
                f"  - roman: {q[f]}\n  - translit: {translit[f][i]}\n  - urdu: {q['urdu']}"
            )

(ROOT / "error_analysis.json").write_text(json.dumps(out, indent=1, default=int), encoding="utf-8")
(ROOT / "error_examples.md").write_text(
    "# bge-m3 misses after transliteration, lowest overlap first\n\n" + "\n".join(examples) + "\n",
    encoding="utf-8",
)
print(json.dumps(out, indent=1, default=int))
