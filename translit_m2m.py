# Roman Urdu to Urdu script with a small dedicated model instead of an LLM: the m2m100-418M
# fine-tune released with Butt et al. (LoResMT 2025), Mavkif/m2m100_rup_rur_to_ur. Runs on a GPU
# box, caches to cache/translit_m2m.json (same format as the other transliteration caches), and
# records seconds per batch so the paper can report its cost next to the LLM.
#
#   python translit_m2m.py data/belebele_queries.jsonl:roman queries.jsonl:roman_1 ...
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
CACHE = ROOT / "cache" / "translit_m2m.json"
MODEL = "Mavkif/m2m100_rup_rur_to_ur"
TOKENIZER = "Mavkif/m2m100_rup_tokenizer_both"
SRC, TGT = "ur", "ur"  # no model card; "ur" as source beat "en" on a 4-question probe


def transliterate_m2m(queries: list[str], cached_only: bool = False) -> list[str]:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    missing = [q for q in dict.fromkeys(queries) if q not in cache]
    if missing and not cached_only:
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        local = os.environ.get("RUR_MODEL_DIR")  # offline copies of both repos
        model_path = os.path.join(local, MODEL.split("/")[-1]) if local else MODEL
        tok_path = os.path.join(local, TOKENIZER.split("/")[-1]) if local else TOKENIZER
        tok = M2M100Tokenizer.from_pretrained(tok_path)
        tok.src_lang = SRC
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(device).eval()
        times = cache.setdefault("_seconds_per_query", [])
        for start in range(0, len(missing), 32):
            batch = missing[start : start + 32]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc, forced_bos_token_id=tok.get_lang_id(TGT), num_beams=4, max_length=256
                )
            times.append((time.time() - t0) / len(batch))
            for q, o in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
                cache[q] = o.strip()
            print(f"{start + len(batch)}/{len(missing)}", flush=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return [cache.get(q, "") for q in queries]


if __name__ == "__main__":
    texts = []
    for spec in sys.argv[1:]:
        path, field = spec.split(":")
        texts += [json.loads(line)[field] for line in open(path, encoding="utf-8") if line.strip()]
    transliterate_m2m(texts)
