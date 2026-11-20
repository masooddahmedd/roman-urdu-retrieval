# All Wikipedia-set numbers for the paper, from the caches (no API or model calls): every retriever,
# every query form, and every fix, including the two transliterators added after the first release
# (Qwen2.5-7B-Instruct through Ollama and the m2m100 transliteration model), plus the Urdu-only
# pool. Writes results_paper.json. A retriever or transliterator missing from the cache is skipped.
import json
import sys

import numpy as np

from fixes import normalize_roman, rrf, select_rules, transliterate
from retrievers import ST_MODELS, build_retrievers
from run_benchmark import FORMS, ROMAN, ROOT, evaluate, load_data
from translit_m2m import transliterate_m2m
from translit_open import transliterate_open

RETRIEVERS = ["bm25", "bge-m3", "e5", "openai"]


def main(names: list[str]) -> None:
    passages, queries = load_data()
    index_of = {p["id"]: i for i, p in enumerate(passages)}
    dev = [q for q in queries if q["split"] == "dev"]
    test = [q for q in queries if q["split"] == "test"]
    gold = [index_of[q["gold_id"]] for q in test]
    rules = select_rules([[q[f] for f in ROMAN] for q in dev])
    is_english = np.array([p["lang"] == "en" for p in passages])

    arms = {
        "translit": transliterate,
        "translit_open": transliterate_open,
        "translit_m2m": transliterate_m2m,
    }
    translits = {}
    for arm, fn in arms.items():
        got = {f: fn([q[f] for q in test], cached_only=True) for f in ROMAN}
        if all(all(v) for v in got.values()):
            translits[arm] = got
    normed = {f: [normalize_roman(q[f], rules) for q in test] for f in ROMAN}
    print("transliterators:", list(translits))

    out_path = ROOT / "results_paper.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    results["_meta"] = {"n_test": len(test), "n_passages": len(passages), "rules": rules}
    bm25 = build_retrievers(passages, ["bm25"])["bm25"]
    for name in names:
        try:
            retr = bm25 if name == "bm25" else build_retrievers(passages, [name])[name]
        except Exception as e:  # missing cache on a machine without the model or key
            print("skip", name, type(e).__name__, str(e)[:120])
            continue
        base = {f: retr.scores([q[f] for q in test]) for f in FORMS}
        keep = {f: base[f] for f in ("urdu", "english")}
        fixes = {"none": base, "normalize": {**keep, **{f: retr.scores(normed[f]) for f in ROMAN}}}
        for arm, tr in translits.items():
            try:
                tr_scores = {f: retr.scores(tr[f]) for f in ROMAN}
            except Exception as e:  # e.g. OpenAI texts not embedded for this arm
                print("skip", name, arm, type(e).__name__)
                continue
            fixes[arm] = {**keep, **tr_scores}
            if name != "bm25":
                fixes[f"{arm}_hybrid"] = {
                    **keep,
                    **{f: rrf([tr_scores[f], base[f], bm25.scores(tr[f])]) for f in ROMAN},
                }
        res = {fix: evaluate(sc, gold) for fix, sc in fixes.items()}
        pool = {}
        for f in FORMS:
            s = base[f].copy()
            s[:, is_english] = -1e9
            pool[f] = s
        res["urdu_only_pool"] = evaluate(pool, gold)
        top5 = {f: np.argsort(-base[f], axis=1)[:, :5] for f in FORMS}
        res["english_share_top5"] = {f: float(is_english[top5[f]].mean()) for f in FORMS}
        results[name] = res
        print(
            name,
            {k: round(v["roman_mean"]["recall@5"], 3) for k, v in res.items() if "roman_mean" in v},
        )
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or RETRIEVERS)
