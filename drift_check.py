# Checks how much "variant drift" (Roman Urdu variants that change words, not just spelling) affects
# the benchmark. A cheap automatic flag (low word overlap between a question's three spellings after
# normalisation) is calibrated against the manual DRIFT labels on the 50-question review sample,
# applied to all questions, and the headline numbers are recomputed on the questions it leaves clean.
import json

import numpy as np

from fixes import normalize_roman, rrf, transliterate
from retrievers import build_retrievers
from review_labels import NOTES
from run_benchmark import FORMS, ROMAN, ROOT, evaluate, load_data

passages, queries = load_data()
RULES = ["fold_repeats", "iya_to_ya", "ai_ay_ae_to_e", "q_to_k"]


def min_overlap(q: dict) -> float:
    sets = [set(normalize_roman(q[f], RULES).split()) for f in ROMAN]
    vals = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
    return min(vals)


manual = {i for i, (_, note) in NOTES.items() if "DRIFT" in note}
sample_ids = [
    r["id"]
    for r in __import__("csv").DictReader(open(ROOT / "review_sample.csv", encoding="utf-8-sig"))
]
by_id = {q["id"]: q for q in queries}
best = None
for t in np.arange(0.5, 0.96, 0.05) if len(manual) >= 5 else []:
    flagged = {i for i in sample_ids if min_overlap(by_id[i]) < t}
    tp = len(flagged & manual)
    prec = tp / len(flagged) if flagged else 0
    rec = tp / len(manual)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(
        f"threshold {t:.2f}: flagged {len(flagged)}, precision {prec:.2f}, recall {rec:.2f}, f1 {f1:.2f}"
    )
    if best is None or f1 > best[0]:
        best = (f1, float(t), prec, rec)
if best is None:
    # Too few manual drift labels to calibrate, so reuse the threshold calibrated on the first
    # version of the set (v1/).
    best = (0.0, 0.55, float("nan"), float("nan"))
_, thr, prec, rec = best
flag = {q["id"]: min_overlap(q) < thr for q in queries}
print(
    f"chosen threshold {thr:.2f} (precision {prec:.2f}, recall {rec:.2f}); flagged "
    f"{sum(flag.values())} of {len(queries)} questions"
)

test = [q for q in queries if q["split"] == "test"]
clean_idx = [i for i, q in enumerate(test) if not flag[q["id"]]]
idx_of = {p["id"]: i for i, p in enumerate(passages)}
gold = [idx_of[q["gold_id"]] for q in test]
translit = {f: transliterate([q[f] for q in test]) for f in ROMAN}
retrievers = build_retrievers(passages, ["bm25", "openai", "bge-m3", "e5"])
out = {
    "threshold": thr,
    "precision_on_sample": prec,
    "recall_on_sample": rec,
    "flagged_all": int(sum(flag.values())),
    "n_queries": len(queries),
    "n_test_clean": len(clean_idx),
    "n_test": len(test),
    "retrievers": {},
}
for name, retr in retrievers.items():
    base = {f: retr.scores([q[f] for q in test]) for f in FORMS}
    fixes = {
        "none": base,
        "transliterate": {
            **{f: base[f] for f in ["urdu", "english"]},
            **{f: retr.scores(translit[f]) for f in ROMAN},
        },
    }
    if name != "bm25":
        bm = retrievers["bm25"]
        fixes["hybrid"] = {
            **{f: base[f] for f in ["urdu", "english"]},
            **{f: rrf([retr.scores(translit[f]), base[f], bm.scores(translit[f])]) for f in ROMAN},
        }
    out["retrievers"][name] = {}
    for fix, sc in fixes.items():
        full = evaluate(sc, gold)
        sub = evaluate({f: m[clean_idx] for f, m in sc.items()}, [gold[i] for i in clean_idx])
        out["retrievers"][name][fix] = {
            "all": {
                "urdu": full["urdu"]["recall@5"],
                "roman": full["roman_mean"]["recall@5"],
                "disagree": full["query_disagreement"],
            },
            "clean": {
                "urdu": sub["urdu"]["recall@5"],
                "roman": sub["roman_mean"]["recall@5"],
                "disagree": sub["query_disagreement"],
            },
        }
(ROOT / "drift_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"clean test questions: {len(clean_idx)} of {len(test)}")
for name, fx in out["retrievers"].items():
    for fix, r in fx.items():
        a, c = r["all"], r["clean"]
        print(
            f"{name:7s} {fix:13s} roman {a['roman']:.1%} -> {c['roman']:.1%} | urdu {a['urdu']:.1%} -> {c['urdu']:.1%} | disagree {a['disagree']:.1%} -> {c['disagree']:.1%}"
        )
