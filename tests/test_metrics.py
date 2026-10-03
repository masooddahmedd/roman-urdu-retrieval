# Tests for rank and spread metrics in run_benchmark.py.
import numpy as np

from run_benchmark import evaluate, gold_ranks, hits, mrr


def test_gold_rank_counts_better_scored_passages():
    scores = np.array([[0.1, 0.9, 0.5], [0.9, 0.1, 0.5]])
    assert list(gold_ranks(scores, [2, 0])) == [2, 1]


def test_hits_and_mrr():
    ranks = np.array([1, 3, 12])
    assert list(hits(ranks)) == [1.0, 1.0, 0.0]
    assert abs(mrr(ranks) - (1 + 1 / 3 + 0) / 3) < 1e-9


def test_variant_spread_is_max_minus_min_recall():
    # Ten passages so that a gold passage ranked last falls outside the top 5.
    good = np.array([[1.0] + [0.0] * 9])
    bad = np.array([[0.0] + [1.0] * 9])
    out = evaluate({"roman_1": good, "roman_2": good, "roman_3": bad}, [0])
    assert out["variant_spread"] == 1.0
    assert out["query_disagreement"] == 1.0
