# Embeds every text a benchmark needs with the sentence-transformers models, on whatever machine has
# a GPU, and writes the same cache files (cache/emb/<name>.npz) the benchmark reads. The text lists
# come from `python export_texts.py` and `python belebele.py texts`, so this step needs no API key.
#
#   python embed_remote.py cache/texts_wiki.jsonl cache/texts_belebele.jsonl --models qwen3-emb-0.6b
import argparse
import json
import time

from retrievers import ST_MODELS, STRetriever, st_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--models", nargs="*", default=list(ST_MODELS))
    args = ap.parse_args()
    rows = []
    for f in args.files:
        rows += [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
    for name in args.models:
        cls = st_class(name)
        retr: STRetriever = cls.__new__(cls)  # skip __init__: no passage list needed here
        from retrievers import EmbeddingCache

        retr.cache = EmbeddingCache(name)
        start = time.time()
        for kind in ("passage", "query"):
            texts = list(dict.fromkeys(r["text"] for r in rows if r["kind"] == kind))
            retr.embed(texts, kind)
        print(f"{name}: {len(retr.cache.vectors)} vectors, {time.time() - start:.0f} s", flush=True)
        retr.model = None


if __name__ == "__main__":
    main()
