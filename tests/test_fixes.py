# Tests for the pieces whose behaviour the benchmark depends on: the spelling rules must merge
# variants of the same word without merging different words, and rank fusion must reward agreement.
import numpy as np

from fixes import normalize_roman, rrf, select_rules, variant_agreement
from retrievers import tokenize


def test_common_variants_collapse():
    assert normalize_roman("kya") == normalize_roman("kia") == normalize_roman("kyaa")
    assert normalize_roman("kaise") == normalize_roman("kaisay") == normalize_roman("kese")
    assert normalize_roman("hai") == normalize_roman("hay")


def test_different_words_stay_different():
    assert normalize_roman("lahore") != normalize_roman("karachi")
    assert normalize_roman("shehar") != normalize_roman("sahar")


def test_agreement_goes_up_after_selected_rules():
    variants = [["Lahore kya hai", "Lahore kia hay", "Lahore kyaa he"]] * 5
    rules = select_rules(variants)
    assert variant_agreement(variants, rules) > variant_agreement(variants, [])


def test_rrf_prefers_documents_both_rankers_like():
    a = np.array([[0.9, 0.8, 0.1]])
    b = np.array([[0.2, 0.9, 0.8]])
    fused = rrf([a, b])
    assert fused.argmax() == 1


def test_bm25_tokenizer_folds_arabic_variants():
    assert tokenize("لاہور كا") == tokenize("لاہور کا")
