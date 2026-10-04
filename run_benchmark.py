# Runs the whole benchmark: every retriever on every query form (Urdu script, English, three Roman
# Urdu spellings), then the three fixes, and writes results.json, results tables and the charts.
# Rules are selected on the dev queries only, every number reported is on the test queries.
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fixes import normalize_roman, rrf, select_rules, transliterate
from retrievers import build_retrievers

ROOT = Path(__file__).parent
FORMS = ["urdu", "english", "roman_1", "roman_2", "roman_3"]
ROMAN = ["roman_1", "roman_2", "roman_3"]
K = 5


def load_data():
    passages = [
        json.loads(line) for line in (ROOT / "data" / "passages.jsonl").open(encoding="utf-8")
    ]
    queries = [json.loads(line) for line in (ROOT / "queries.jsonl").open(encoding="utf-8")]
    return passages, queries


def gold_ranks(scores: np.ndarray, gold_idx: list[int]) -> np.ndarray:
    # Rank (1 is best) of the gold passage for each query among all passages.
    gold_scores = scores[np.arange(len(gold_idx)), gold_idx]
    return (scores > gold_scores[:, None]).sum(axis=1) + 1


def hits(ranks: np.ndarray) -> np.ndarray:
    return (ranks <= K).astype(float)


def mrr(ranks: np.ndarray) -> float:
    return float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0)))


def bootstrap_ci(values: np.ndarray, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(0)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate(scores_by_form: dict[str, np.ndarray], gold_idx: list[int]) -> dict:
    # Per query form: Recall@5 and MRR@10. Across the three spellings: the spread (max minus min
    # of Recall@5) and the share of queries where the spellings do not all agree on hit or miss.
    per_form = {}
    for form, scores in scores_by_form.items():
        r = gold_ranks(scores, gold_idx)
        per_form[form] = {"recall@5": float(hits(r).mean()), "mrr@10": mrr(r), "_hits": hits(r)}
    roman_hits = np.stack([per_form[f]["_hits"] for f in ROMAN if f in per_form])
    out = {f: {k: v for k, v in m.items() if k != "_hits"} for f, m in per_form.items()}
    if len(roman_hits) == 3:
        recalls = roman_hits.mean(axis=1)
        out["roman_mean"] = {
            "recall@5": float(recalls.mean()),
            "mrr@10": float(np.mean([per_form[f]["mrr@10"] for f in ROMAN])),
            "recall@5_ci95": bootstrap_ci(roman_hits.mean(axis=0)),
        }
        out["variant_spread"] = float(recalls.max() - recalls.min())
        out["query_disagreement"] = float((roman_hits.min(axis=0) != roman_hits.max(axis=0)).mean())
    return out


def main(retriever_names: list[str] | None) -> None:
    passages, queries = load_data()
    index_of = {p["id"]: i for i, p in enumerate(passages)}
    dev = [q for q in queries if q["split"] == "dev"]
    test = [q for q in queries if q["split"] == "test"]
    rules = select_rules([[q[f] for f in ROMAN] for q in dev])
    print("selected normalisation rules:", rules)

    retrievers = build_retrievers(passages, retriever_names)
    results: dict = {"rules": rules, "n_test_queries": len(test), "n_passages": len(passages)}
    gold_test = [index_of[q["gold_id"]] for q in test]

    translit = {f: transliterate([q[f] for q in test]) for f in ROMAN}
    normed = {f: [normalize_roman(q[f], rules) for q in test] for f in ROMAN}

    for name, retr in retrievers.items():
        base = {f: retr.scores([q[f] for q in test]) for f in FORMS}
        fixes = {
            "none": base,
            "normalize": {
                **{f: base[f] for f in ["urdu", "english"]},
                **{f: retr.scores(normed[f]) for f in ROMAN},
            },
            "transliterate": {
                **{f: base[f] for f in ["urdu", "english"]},
                **{f: retr.scores(translit[f]) for f in ROMAN},
            },
        }
        if name != "bm25":
            bm25 = retrievers.get("bm25") or build_retrievers(passages, ["bm25"])["bm25"]
            fixes["hybrid"] = {
                **{f: base[f] for f in ["urdu", "english"]},
                **{
                    f: rrf([retr.scores(translit[f]), base[f], bm25.scores(translit[f])])
                    for f in ROMAN
                },
            }
        results[name] = {fix: evaluate(sc, gold_test) for fix, sc in fixes.items()}
        print(
            name, {fix: round(r["roman_mean"]["recall@5"], 3) for fix, r in results[name].items()}
        )

    (ROOT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_tables(results)
    plot(results)


def write_tables(results: dict) -> None:
    names = [n for n in results if n not in ("rules", "n_test_queries", "n_passages")]
    lines = [
        "| Retriever | Urdu script | English | Roman Urdu (mean of 3 spellings) "
        "| Drop, Urdu to Roman "
        "| Variant spread | Queries where spellings disagree |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in names:
        r = results[n]["none"]
        drop = r["urdu"]["recall@5"] - r["roman_mean"]["recall@5"]
        lines.append(
            f"| {n} | {r['urdu']['recall@5']:.1%} | {r['english']['recall@5']:.1%} | "
            f"{r['roman_mean']['recall@5']:.1%} | {drop * 100:.1f} pts | "
            f"{r['variant_spread'] * 100:.1f} pts | {r['query_disagreement']:.1%} |"
        )
    fix_lines = [
        "| Retriever | Fix | Roman Recall@5 | Roman MRR@10 | Variant spread | Disagree |",
        "|---|---|---|---|---|---|",
    ]
    for n in names:
        for fix, r in results[n].items():
            rm = r["roman_mean"]
            fix_lines.append(
                f"| {n} | {fix} | {rm['recall@5']:.1%} | {rm['mrr@10']:.3f} | "
                f"{r['variant_spread'] * 100:.1f} pts | {r['query_disagreement']:.1%} |"
            )
    (ROOT / "results_table.md").write_text(
        "## Baseline (no fix), Recall@5 on test queries\n\n"
        + "\n".join(lines)
        + "\n\n## Fixes\n\n"
        + "\n".join(fix_lines)
        + "\n",
        encoding="utf-8",
    )


def plot(results: dict) -> None:
    names = [n for n in results if n not in ("rules", "n_test_queries", "n_passages")]
    fixes = ["none", "normalize", "transliterate", "hybrid"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(names))
    w = 0.2
    for i, fix in enumerate(fixes):
        spread = [results[n].get(fix, {}).get("variant_spread", np.nan) * 100 for n in names]
        recall = [
            results[n].get(fix, {}).get("roman_mean", {}).get("recall@5", np.nan) * 100
            for n in names
        ]
        axes[0].bar(x + i * w, recall, w, label=fix)
        axes[1].bar(x + i * w, spread, w, label=fix)
    for ax, title in zip(
        axes, ["Roman Urdu Recall@5 (%)", "Variant spread: max - min Recall@5 (pts)"]
    ):
        ax.set_xticks(x + 1.5 * w)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(title="fix", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.tight_layout()
    fig.savefig(ROOT / "fixes_chart.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    forms = ["urdu", "english", "roman_mean"]
    for i, f in enumerate(forms):
        vals = [results[n]["none"][f]["recall@5"] * 100 for n in names]
        ax.bar(x + i * 0.25, vals, 0.25, label={"roman_mean": "Roman Urdu (mean)"}.get(f, f))
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Recall@5 (%)")
    ax.set_title("Same question, three ways to type it")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "drop_chart.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievers", nargs="*", help="subset of: bm25 bge-m3 e5 openai")
    main(ap.parse_args().retrievers)
