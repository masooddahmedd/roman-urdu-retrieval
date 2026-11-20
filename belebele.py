# Second test set, built from Belebele (Bandarkar et al., 2024). Belebele has the same 900
# reading-comprehension questions in Urdu script (urd_Arab), romanized Urdu (urd_Latn) and English,
# over 488 FLORES-200 passages. Unlike our Wikipedia set, the Roman Urdu here was written by people,
# not drafted by an LLM. We turn it into retrieval: the query is the question, the pool is the 488
# Urdu-script passages (Urdu only, no English distractors), and the gold passage is the one the
# question was written for.
#
#   python belebele.py prepare     # writes data/belebele_passages.jsonl and _queries.jsonl
#   python belebele.py texts       # writes cache/texts_belebele.jsonl for remote embedding
#   python belebele.py run         # results_belebele.json from cached embeddings
import json
import sys
from pathlib import Path

import numpy as np

from fixes import rrf, transliterate
from run_benchmark import bootstrap_ci, gold_ranks, hits, mrr
from translit_m2m import transliterate_m2m
from translit_open import transliterate_open

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "belebele"
PASSAGES = ROOT / "data" / "belebele_passages.jsonl"
QUERIES = ROOT / "data" / "belebele_queries.jsonl"
FORMS = ["urdu", "english", "roman"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def prepare() -> None:
    by_cfg = {
        cfg: {(r["link"], r["question_number"]): r for r in read_jsonl(SRC / f"{cfg}.jsonl")}
        for cfg in ("urd_Arab", "urd_Latn", "eng_Latn")
    }
    keys = sorted(set.intersection(*(set(v) for v in by_cfg.values())))
    links = sorted({link for link, _ in keys})
    pid = {link: f"bb-{i:03d}" for i, link in enumerate(links)}
    passages = {}
    queries = []
    for link, qn in keys:
        ur, rom, en = (by_cfg[c][(link, qn)] for c in ("urd_Arab", "urd_Latn", "eng_Latn"))
        passages[pid[link]] = {"id": pid[link], "lang": "ur", "text": ur["flores_passage"]}
        queries.append(
            {
                "id": f"bbq-{len(queries):03d}",
                "gold_id": pid[link],
                "urdu": ur["question"],
                "english": en["question"],
                "roman": rom["question"],
            }
        )
    with PASSAGES.open("w", encoding="utf-8") as f:
        for p in passages.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with QUERIES.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(len(passages), "passages,", len(queries), "questions")


def load():
    return read_jsonl(PASSAGES), read_jsonl(QUERIES)


def transliterations(queries: list[dict]) -> dict[str, list[str]]:
    # Each transliterator whose cache covers every question: gpt-4o-mini, Qwen2.5-7B, m2m100.
    roman = [q["roman"] for q in queries]
    found = {
        "translit": transliterate(roman, cached_only=True),
        "translit_open": transliterate_open(roman, cached_only=True),
        "translit_m2m": transliterate_m2m(roman, cached_only=True),
    }
    return {k: v for k, v in found.items() if all(v)}


def export_texts() -> None:
    # Every text a dense model needs to embed for this set, so the GPU box needs no API key.
    passages, queries = load()
    rows = [{"kind": "passage", "text": p["text"]} for p in passages]
    for q in queries:
        rows += [{"kind": "query", "text": q[f]} for f in FORMS]
    for texts in transliterations(queries).values():
        rows += [{"kind": "query", "text": t} for t in texts]
    out = ROOT / "cache" / "texts_belebele.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "texts ->", out)


def metrics(scores: np.ndarray, gold: list[int]) -> dict:
    r = gold_ranks(scores, gold)
    h = hits(r)
    return {
        "recall@1": float((r == 1).mean()),
        "recall@5": float(h.mean()),
        "recall@5_ci95": bootstrap_ci(h),
        "mrr@10": mrr(r),
        "_ranks": r,
    }


def run(names: list[str]) -> None:
    from retrievers import build_retrievers

    passages, queries = load()
    index_of = {p["id"]: i for i, p in enumerate(passages)}
    gold = [index_of[q["gold_id"]] for q in queries]
    translits = transliterations(queries)
    retrievers = build_retrievers(passages, names)
    bm25 = retrievers.get("bm25") or build_retrievers(passages, ["bm25"])["bm25"]
    out_path = ROOT / "results_belebele.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    results.update({"n_queries": len(queries), "n_passages": len(passages)})
    ranks = {}
    for name, retr in retrievers.items():
        base = {f: retr.scores([q[f] for q in queries]) for f in FORMS}
        runs = dict(base)
        for arm, texts in translits.items():
            tr = retr.scores(texts)
            runs[f"roman_{arm}"] = tr
            if name != "bm25":
                runs[f"roman_{arm}_hybrid"] = rrf([tr, base["roman"], bm25.scores(texts)])
        res = {k: metrics(v, gold) for k, v in runs.items()}
        ranks[name] = {k: v.pop("_ranks").tolist() for k, v in res.items()}
        results[name] = res
        print(name, {k: round(v["recall@5"], 3) for k, v in res.items()})
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    rank_path = ROOT / "cache" / "ranks_belebele.json"
    old = json.loads(rank_path.read_text(encoding="utf-8")) if rank_path.exists() else {}
    rank_path.write_text(json.dumps({**old, **ranks}), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "texts":
        export_texts()
    elif cmd == "run":
        run(sys.argv[2:] or None)
