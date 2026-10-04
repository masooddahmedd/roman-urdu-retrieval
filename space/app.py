# Demo: type a question in any Roman Urdu spelling and see the top results as typed next to the
# results after transliterating the query to Urdu script. Uses bge-m3 (ONNX, CPU) over precomputed
# passage vectors. Transliteration uses an optional OpenAI key that is only used for that one call
# and never stored or logged; benchmark queries use the cached transliterations.
import json
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DATA = Path(__file__).parent / "data"
passages = [json.loads(line) for line in (DATA / "passages.jsonl").open(encoding="utf-8")]
vectors = np.load(DATA / "bge-m3-passages.npy").astype(np.float32)
cached = json.loads((DATA / "translit.json").read_text(encoding="utf-8"))
examples = json.loads((DATA / "examples.json").read_text(encoding="utf-8"))
gold_by_query = json.loads((DATA / "gold.json").read_text(encoding="utf-8"))
ids = [p["id"] for p in passages]

repo = "BAAI/bge-m3"
tok_path = hf_hub_download(repo, "tokenizer.json")
hf_hub_download(repo, "onnx/model.onnx_data")
session = ort.InferenceSession(
    hf_hub_download(repo, "onnx/model.onnx"), providers=["CPUExecutionProvider"]
)
tokenizer = Tokenizer.from_file(tok_path)
tokenizer.enable_truncation(max_length=400)
tokenizer.enable_padding(pad_id=1, pad_token="<pad>")


def embed(text: str) -> np.ndarray:
    enc = tokenizer.encode(text)
    feed = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
    }
    hidden = session.run(None, feed)[0]
    vec = hidden[0, 0]  # CLS pooling, as bge-m3 expects
    return vec / np.linalg.norm(vec)


def transliterate(query: str, api_key: str) -> str | None:
    if query in cached:
        return cached[query]
    if not api_key:
        return None
    from openai import OpenAI

    resp = OpenAI(api_key=api_key).chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": (
                    "Convert this Roman Urdu text to Urdu script. Keep the meaning and word order, use "
                    "standard Urdu spelling, and keep names as they would be written in Urdu. Output only "
                    f"the Urdu text.\n\n{query}"
                ),
            }
        ],
    )
    return resp.choices[0].message.content.strip()


def top_results(query: str, gold_id: str | None) -> tuple[list[list], str]:
    scores = vectors @ embed(query)
    order = np.argsort(-scores)
    rows = []
    for rank, i in enumerate(order[:5], 1):
        p = passages[i]
        mark = " (answer)" if p["id"] == gold_id else ""
        rows.append([rank, p["title"] + mark, p["lang"], p["text"][:160]])
    note = ""
    if gold_id:
        rank = int(np.where(order == ids.index(gold_id))[0][0]) + 1
        note = f"Answer passage rank: {rank}"
    return rows, note


def search(query: str, api_key: str):
    query = query.strip()
    if not query:
        return [], "", [], ""
    gold = gold_by_query.get(query)
    as_typed, typed_note = top_results(query, gold)
    urdu = transliterate(query, api_key.strip())
    if urdu is None:
        return as_typed, typed_note, [], "Add an OpenAI key, or pick an example, to see this side."
    fixed, fixed_note = top_results(urdu, gold)
    return as_typed, typed_note, fixed, f"Query as Urdu script: {urdu}. {fixed_note}"


with gr.Blocks(title="Roman Urdu retrieval") as demo:
    gr.Markdown(
        "# Roman Urdu retrieval\n"
        "The same question typed in Roman Urdu is often missed by embedding search. Pick an example "
        "(three spellings of the same question) or type your own, and compare the results as typed "
        "with the results after transliterating the query to Urdu script. Model: bge-m3. "
        "Passages are from Wikipedia (CC BY-SA 4.0)."
    )
    example = gr.Dropdown([e["query"] for e in examples], label="Example queries (benchmark)")
    query = gr.Textbox(label="Query (Roman Urdu, in any spelling)")
    key = gr.Textbox(
        label="OpenAI key (optional, only for your own queries; never stored)", type="password"
    )
    go = gr.Button("Search")
    headers = ["rank", "passage", "lang", "text"]
    with gr.Row():
        with gr.Column():
            gr.Markdown("### As typed")
            typed = gr.Dataframe(headers=headers, wrap=True)
            typed_note = gr.Markdown()
        with gr.Column():
            gr.Markdown("### After transliteration to Urdu script")
            fixed = gr.Dataframe(headers=headers, wrap=True)
            fixed_note = gr.Markdown()
    example.change(lambda q: q, example, query)
    go.click(search, [query, key], [typed, typed_note, fixed, fixed_note])

if __name__ == "__main__":
    demo.launch()
