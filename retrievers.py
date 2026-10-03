# The four retrievers: BM25, bge-m3, multilingual-e5-large-instruct and OpenAI embeddings. Each one
# returns a full score matrix (queries x passages) so ranking, hybrids and metrics live in one
# place.
# The open models run through their official ONNX exports on CPU (PyTorch is not needed), and every
# embedding is cached by text, so reruns and new query variants only embed what is new.
import hashlib
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
EMB_CACHE = ROOT / "cache" / "emb"
MODEL_DIR = ROOT / "models"
MAX_TOKENS = 400

# Arabic-script letters that Urdu text often writes in two ways. Folding them lets BM25 match
# passages that spell the same word with a different codepoint.
URDU_FOLD = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ې": "ی",
        "ك": "ک",
        "ہ": "ہ",
        "ۃ": "ہ",
        "ة": "ہ",
        "ؤ": "و",
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
    }
)
DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    text = DIACRITICS.sub("", text.translate(URDU_FOLD)).lower()
    return TOKEN.findall(text)


class BM25Retriever:
    name = "bm25"

    def __init__(self, passages: list[dict]):
        from rank_bm25 import BM25Okapi

        self.index = BM25Okapi([tokenize(p["text"]) for p in passages])

    def scores(self, queries: list[str], kind: str = "query") -> np.ndarray:
        return np.array([self.index.get_scores(tokenize(q)) for q in queries], dtype=np.float32)


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, name: str):
        self.path = EMB_CACHE / f"{name}.npz"
        self.vectors: dict[str, np.ndarray] = {}
        if self.path.exists():
            data = np.load(self.path)
            self.vectors = {k: data[k] for k in data.files}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self.vectors)


class DenseRetriever:
    # Subclasses only implement embed_missing(texts, kind); caching, normalisation and scoring are
    # shared. Passages and queries are cached under different keys because some models prefix them.
    name = "dense"

    def __init__(self, passages: list[dict]):
        self.cache = EmbeddingCache(self.name)
        self.passages = passages
        self.passage_matrix = self.embed([p["text"] for p in passages], "passage")

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        keys = [f"{kind}:{text_key(t)}" for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.cache.vectors]
        if missing:
            vecs = self.embed_missing([t for _, t in missing], kind)
            for (k, _), v in zip(missing, vecs):
                self.cache.vectors[k] = v.astype(np.float16)
            self.cache.save()
        return np.stack([self.cache.vectors[k] for k in keys]).astype(np.float32)

    def scores(self, queries: list[str], kind: str = "query") -> np.ndarray:
        return self.embed(queries, "query") @ self.passage_matrix.T


class OnnxRetriever(DenseRetriever):
    repo = ""
    pooling = "cls"
    query_prefix = ""

    def __init__(self, passages: list[dict]):
        self.session = None
        super().__init__(passages)

    def load(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        local = MODEL_DIR / self.name
        for f in ("tokenizer.json", "onnx/model.onnx", "onnx/model.onnx_data"):
            hf_hub_download(self.repo, f, local_dir=local)
        tok = Tokenizer.from_file(str(local / "tokenizer.json"))
        tok.enable_truncation(max_length=MAX_TOKENS)
        tok.enable_padding(pad_id=tok.token_to_id("<pad>") or 1, pad_token="<pad>")
        self.tokenizer = tok
        self.session = ort.InferenceSession(
            str(local / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]

    def embed_missing(self, texts: list[str], kind: str) -> list[np.ndarray]:
        if self.session is None:
            self.load()
        if kind == "query":
            texts = [self.query_prefix + t for t in texts]
        order = np.argsort([len(t) for t in texts])
        out: dict[int, np.ndarray] = {}
        for start in range(0, len(texts), 16):
            idx = order[start : start + 16]
            enc = self.tokenizer.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            outputs = dict(zip(self.output_names, self.session.run(None, feed)))
            vecs = self.pool(outputs, mask)
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            for i, v in zip(idx, vecs):
                out[int(i)] = v
        return [out[i] for i in range(len(texts))]

    def pool(self, outputs: dict, mask: np.ndarray) -> np.ndarray:
        hidden = outputs.get("last_hidden_state")
        if hidden is None:
            return next(v for k, v in outputs.items() if "dense" in k or "sentence" in k)
        if self.pooling == "cls":
            return hidden[:, 0]
        m = mask[:, :, None].astype(np.float32)
        return (hidden * m).sum(1) / m.sum(1)


class BgeM3(OnnxRetriever):
    name = "bge-m3"
    repo = "BAAI/bge-m3"
    pooling = "cls"


class E5Instruct(OnnxRetriever):
    name = "multilingual-e5-large-instruct"
    repo = "intfloat/multilingual-e5-large-instruct"
    pooling = "mean"
    # E5-instruct expects an instruction on the query side only.
    query_prefix = (
        "Instruct: Given a question, retrieve the Wikipedia passage that answers it\nQuery: "
    )


class OpenAIEmbedding(DenseRetriever):
    name = "openai-text-embedding-3-large"
    model = "text-embedding-3-large"
    dimensions = 1024

    def embed_missing(self, texts: list[str], kind: str) -> list[np.ndarray]:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        client = OpenAI(max_retries=5)
        vecs = []
        for start in range(0, len(texts), 128):
            resp = client.embeddings.create(
                model=self.model, input=texts[start : start + 128], dimensions=self.dimensions
            )
            vecs.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)
        return [v / np.linalg.norm(v) for v in vecs]


def build_retrievers(passages: list[dict], names: list[str] | None = None) -> dict:
    classes = {"bm25": BM25Retriever, "bge-m3": BgeM3, "e5": E5Instruct, "openai": OpenAIEmbedding}
    chosen = names or list(classes)
    return {n: classes[n](passages) for n in chosen}
