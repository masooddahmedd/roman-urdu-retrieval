# Roman Urdu to Urdu script with an open model served by Ollama, the free alternative to the
# gpt-4o-mini transliteration in fixes.py. Same prompt, temperature 0, cached in
# cache/translit_open.json. Run where Ollama is (OLLAMA_URL), then copy the cache back.
#
#   python translit_open.py data/belebele_queries.jsonl:roman queries.jsonl:roman_1 ...
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from fixes import TRANSLIT_PROMPT

ROOT = Path(__file__).parent
CACHE = ROOT / "cache" / "translit_open.json"
MODEL = os.environ.get("TRANSLIT_OPEN_MODEL", "qwen2.5:7b-instruct")
URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
NUM_GPU = int(os.environ.get("OLLAMA_NUM_GPU", "-1"))


def _load() -> dict:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def _call(q: str) -> tuple[str, float]:
    options = {"temperature": 0, "num_predict": 200}
    if NUM_GPU >= 0:
        options["num_gpu"] = NUM_GPU
    body = {
        "model": MODEL,
        "stream": False,
        "options": options,
        "messages": [{"role": "user", "content": TRANSLIT_PROMPT.format(q=q)}],
    }
    req = urllib.request.Request(
        f"{URL}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())["message"]["content"].strip()
    return out, time.time() - start


def transliterate_open(queries: list[str], cached_only: bool = False) -> list[str]:
    cache = _load()
    missing = [q for q in dict.fromkeys(queries) if q not in cache]
    if missing and not cached_only:
        for i, q in enumerate(missing):
            out, secs = _call(q)
            cache[q] = out
            cache.setdefault("_seconds", {})[q] = round(secs, 3)
            if i % 25 == 0:
                CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
                print(f"{i}/{len(missing)} {secs:.1f}s", flush=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return [cache.get(q, "") for q in queries]


if __name__ == "__main__":
    texts = []
    for spec in sys.argv[1:]:
        path, field = spec.split(":")
        texts += [json.loads(line)[field] for line in open(path, encoding="utf-8") if line.strip()]
    transliterate_open(texts)
    print("done", MODEL)
