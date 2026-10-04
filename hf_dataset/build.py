# Stages the Hugging Face dataset folder: copies the query files and writes corpus.jsonl (with the
# Wikipedia text and attribution fields). Run from the repo root after fetch_corpus.py.
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "stage"
OUT.mkdir(exist_ok=True)
for name in ("queries.jsonl", "queries.csv", "review_sample.csv"):
    shutil.copy(ROOT / name, OUT / name)
shutil.copy(Path(__file__).parent / "README.md", OUT / "README.md")
shutil.copy(ROOT / "data" / "passages.jsonl", OUT / "corpus.jsonl")
print("staged", sorted(p.name for p in OUT.iterdir()))
