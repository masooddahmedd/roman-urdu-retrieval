# Exports what the demo Space needs: the passages (with Wikipedia attribution), their bge-m3
# vectors, the cached transliterations and a few example queries. Run from the repo root after
# run_benchmark.py has filled the caches.
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrievers import BgeM3  # noqa: E402
from run_benchmark import load_data  # noqa: E402

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
passages, queries = load_data()
retriever = BgeM3(passages)  # every vector is already cached, so this does not load the model

with (OUT / "passages.jsonl").open("w", encoding="utf-8") as f:
    for p in passages:
        f.write(
            json.dumps(
                {
                    "id": p["id"],
                    "lang": p["lang"],
                    "title": p["title"],
                    "pageid": p["pageid"],
                    "text": p["text"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
np.save(OUT / "bge-m3-passages.npy", retriever.passage_matrix.astype(np.float16))

translit = json.loads((ROOT / "cache" / "translit.json").read_text(encoding="utf-8"))
examples = []
for q in queries:
    if q["split"] != "test":
        continue
    for key in ("roman_1", "roman_2", "roman_3"):
        if q[key] in translit:
            examples.append({"query": q[key], "urdu": translit[q[key]], "gold_id": q["gold_id"]})
(OUT / "translit.json").write_text(
    json.dumps({e["query"]: e["urdu"] for e in examples}, ensure_ascii=False), encoding="utf-8"
)
(OUT / "examples.json").write_text(json.dumps(examples[:60], ensure_ascii=False), encoding="utf-8")
# Gold passage for every cached query, so the demo can show the answer rank for any of them.
(OUT / "gold.json").write_text(
    json.dumps({e["query"]: e["gold_id"] for e in examples}, ensure_ascii=False), encoding="utf-8"
)
print(len(passages), "passages,", len(examples), "cached transliterations")
