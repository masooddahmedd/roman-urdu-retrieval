# translit_m2m.py split across two machines, for setups where the SentencePiece tokenizer and the
# model cannot run on the same box (here: the laptop blocks the sentencepiece DLL, the GPU box has a
# slow link for the 1.9 GB model). Only token ids move between machines. Results are identical to
# running translit_m2m.py in one place: same tokenizer, model, beam size and decoding.
#
#   box with sentencepiece:  python translit_m2m_split.py encode SRC_LANG files...  -> cache/m2m_ids.json
#   box with the model:      python translit_m2m_split.py generate                  -> cache/m2m_out.json
#   box with sentencepiece:  python translit_m2m_split.py decode                    -> cache/translit_m2m.json
import json
import os
import sys
import time

from translit_m2m import CACHE, MODEL, ROOT, TGT, TOKENIZER

IDS = ROOT / "cache" / "m2m_ids.json"
OUT = ROOT / "cache" / "m2m_out.json"


def local(repo: str) -> str:
    d = os.environ.get("RUR_MODEL_DIR")
    return os.path.join(d, repo.split("/")[-1]) if d else repo


def tokenizer(src: str):
    from transformers import M2M100Tokenizer

    tok = M2M100Tokenizer.from_pretrained(local(TOKENIZER))
    tok.src_lang = src
    return tok


def encode(src: str, specs: list[str]) -> None:
    texts = []
    for spec in specs:
        path, field = spec.split(":")
        texts += [json.loads(line)[field] for line in open(path, encoding="utf-8") if line.strip()]
    texts = list(dict.fromkeys(texts))
    tok = tokenizer(src)
    IDS.write_text(
        json.dumps(
            {
                "src": src,
                "bos": tok.get_lang_id(TGT),
                "pad": tok.pad_token_id,
                "items": [{"text": t, "ids": tok(t)["input_ids"]} for t in texts],
            }
        ),
        encoding="utf-8",
    )
    print(len(texts), "encoded")


def generate() -> None:
    import torch
    from transformers import M2M100ForConditionalGeneration

    data = json.loads(IDS.read_text(encoding="utf-8"))
    torch.set_num_threads(os.cpu_count() or 4)
    model = M2M100ForConditionalGeneration.from_pretrained(local(MODEL)).eval()
    items, out, secs = data["items"], [], []
    for start in range(0, len(items), 16):
        batch = items[start : start + 16]
        width = max(len(b["ids"]) for b in batch)
        ids = torch.full((len(batch), width), data["pad"], dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, b in enumerate(batch):  # right padding, as the tokenizer does
            ids[i, : len(b["ids"])] = torch.tensor(b["ids"])
            mask[i, : len(b["ids"])] = 1
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=mask,
                forced_bos_token_id=data["bos"],
                num_beams=4,
                max_length=256,
            )
        secs.append((time.time() - t0) / len(batch))
        out += [{"text": b["text"], "out": g.tolist()} for b, g in zip(batch, gen)]
        print(f"{start + len(batch)}/{len(items)} {secs[-1]:.2f}s/q", flush=True)
    OUT.write_text(
        json.dumps({"src": data["src"], "seconds_per_query": secs, "items": out}), encoding="utf-8"
    )


def decode() -> None:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    tok = tokenizer(data["src"])
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for it in data["items"]:
        cache[it["text"]] = tok.decode(it["out"], skip_special_tokens=True).strip()
    cache["_seconds_per_query_cpu"] = data["seconds_per_query"]
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    print(len(data["items"]), "decoded")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "encode":
        encode(sys.argv[2], sys.argv[3:])
    elif cmd == "generate":
        generate()
    elif cmd == "decode":
        decode()
