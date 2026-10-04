# Checks why English queries do so differently across models. For each dense retriever we measure
# the share of top-5 results that are English passages, for English, Urdu and Roman Urdu queries.
# The gold passage is always Urdu, so a high English share means the model prefers matching
# language over matching meaning. Runs from the embedding cache, no model calls.
import json

import numpy as np

from retrievers import build_retrievers
from run_benchmark import FORMS, ROOT, load_data

passages, queries = load_data()
test = [q for q in queries if q["split"] == "test"]
is_english = np.array([p["lang"] == "en" for p in passages])
out = {}
for name, retr in build_retrievers(passages, ["openai", "bge-m3", "e5"]).items():
    out[name] = {}
    for form in FORMS:
        scores = retr.scores([q[form] for q in test])
        top5 = np.argsort(-scores, axis=1)[:, :5]
        out[name][form] = float(is_english[top5].mean())
(ROOT / "language_bias.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for name, forms in out.items():
    print(name, {f: f"{v:.0%}" for f, v in forms.items()})

# Same baseline with the English passages removed from the candidate pool. This separates "the
# model cannot read Roman Urdu" from "Roman Urdu looks like English, so English passages win".
index_of = {p["id"]: i for i, p in enumerate(passages)}
gold = [index_of[q["gold_id"]] for q in test]
from run_benchmark import gold_ranks, hits  # noqa: E402

urdu_only = {}
for name, retr in build_retrievers(passages, ["openai", "bge-m3", "e5"]).items():
    rec = {}
    for form in FORMS:
        scores = retr.scores([q[form] for q in test])
        scores[:, is_english] = -1e9
        rec[form] = float(hits(gold_ranks(scores, gold)).mean())
    rec["roman_mean"] = float(np.mean([rec[f] for f in ("roman_1", "roman_2", "roman_3")]))
    urdu_only[name] = rec
out_path = ROOT / "language_bias.json"
data = json.loads(out_path.read_text(encoding="utf-8"))
data["urdu_only_pool_recall@5"] = urdu_only
out_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
for name, r in urdu_only.items():
    print(
        "urdu-only pool",
        name,
        {f: f"{v:.1%}" for f, v in r.items() if f in ("urdu", "english", "roman_mean")},
    )
