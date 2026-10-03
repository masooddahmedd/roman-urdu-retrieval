# The three fixes we test for Roman Urdu queries: (a) rule-based spelling normalisation, (b) LLM
# transliteration of the query into Urdu script before searching, (c) a hybrid that fuses BM25 and
# dense rankings from both the original and the transliterated query. The normalisation rules are
# written to merge spelling variants without merging different words, and each rule is kept only
# if it raises agreement between spellings of the dev queries (see select_rules).
import json
import re
from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / "cache" / "translit.json"
TRANSLIT_MODEL = "gpt-4o-mini"

# Applied in this order. Each rule is (name, pattern, replacement).
RULES = [
    ("fold_repeats", r"(.)\1+", r"\1"),  # kyaa -> kya, bohat -> bohat, hooon -> hon
    ("iya_to_ya", r"i(y)?a\b", "ya"),  # kia / kiya -> kya
    ("ai_ay_ae_to_e", r"(ai|ay|ae|ey|ei)", "e"),  # kaise / kaisay / kese -> kese, hai / hay -> he
    ("ph_to_f", r"ph", "f"),  # phir -> fir
    ("w_to_v", r"w", "v"),  # wo / vo, wajah / vajah
    ("q_to_k", r"q", "k"),  # qila / kila
    ("final_h", r"(?<=[aeiou])h\b", ""),  # wah -> wa, kyah -> kya
]


def normalize_roman(text: str, rules: list[str] | None = None) -> str:
    out = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    for name, pattern, repl in RULES:
        if rules is None or name in rules:
            out = re.sub(pattern, repl, out)
    return " ".join(out.split())


def variant_agreement(variant_sets: list[list[str]], rules: list[str] | None) -> float:
    # Mean pairwise Jaccard similarity of the token sets of a query's three spellings.
    scores = []
    for variants in variant_sets:
        sets = [set(normalize_roman(v, rules).split()) for v in variants]
        for i in range(3):
            for j in range(i + 1, 3):
                union = sets[i] | sets[j]
                scores.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return float(np.mean(scores))


def select_rules(dev_variants: list[list[str]], min_gain: float = 0.002) -> list[str]:
    # Greedy forward selection on the dev split only. A rule stays if it raises agreement between
    # spellings, which stops us from keeping rules that do nothing (and rules written for words
    # the dev queries never use).
    chosen: list[str] = []
    best = variant_agreement(dev_variants, chosen)
    for name, _, _ in RULES:
        trial = variant_agreement(dev_variants, chosen + [name])
        if trial - best > min_gain:
            chosen.append(name)
            best = trial
    return chosen


def transliterate(queries: list[str]) -> list[str]:
    # Roman Urdu to Urdu script with an LLM. Temperature 0 and a disk cache so reruns are free.
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    missing = [q for q in dict.fromkeys(queries) if q not in cache]
    if missing:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        client = OpenAI(max_retries=5)
        for q in missing:
            resp = client.chat.completions.create(
                model=TRANSLIT_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Convert this Roman Urdu text to Urdu script. Keep the meaning and "
                            "word order, "
                            "use standard Urdu spelling, and keep names as they would be "
                            "written in Urdu. "
                            f"Output only the Urdu text.\n\n{q}"
                        ),
                    }
                ],
            )
            cache[q] = resp.choices[0].message.content.strip()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return [cache[q] for q in queries]


def rrf(score_matrices: list[np.ndarray], k: int = 60) -> np.ndarray:
    # Reciprocal rank fusion. Rank-based, so BM25 scores and cosine similarities can be mixed
    # without worrying about their different scales.
    fused = np.zeros_like(score_matrices[0], dtype=np.float32)
    for scores in score_matrices:
        ranks = np.argsort(np.argsort(-scores, axis=1), axis=1) + 1
        fused += 1.0 / (k + ranks)
    return fused
