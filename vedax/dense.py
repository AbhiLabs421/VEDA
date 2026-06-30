"""Dense retrieval baseline: all-MiniLM-L6-v2 (ONNX, CPU) — the most
widely deployed RAG retriever. Mean pooling + L2 norm, cosine search.

Model weights: qdrant-fastembed mirror of sentence-transformers/all-MiniLM-L6-v2.
"""

import os
import subprocess
import tarfile
import urllib.request

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/qdrant-fastembed/"
             "sentence-transformers-all-MiniLM-L6-v2.tar.gz")
MODEL_DIR = os.environ.get("MINILM_DIR",
                           "/tmp/minilm/fast-all-MiniLM-L6-v2")
MAX_TOKENS = 256


def fetch():
    if not os.path.isfile(os.path.join(MODEL_DIR, "model.onnx")):
        root = os.path.dirname(MODEL_DIR)
        os.makedirs(root, exist_ok=True)
        tar_path = os.path.join(root, "model.tar.gz")
        urllib.request.urlretrieve(MODEL_URL, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(root)
    return MODEL_DIR


class MiniLM:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        fetch()
        self.tok = Tokenizer.from_file(os.path.join(MODEL_DIR,
                                                    "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.sess = ort.InferenceSession(
            os.path.join(MODEL_DIR, "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def embed(self, texts, batch_size=32):
        """L2-normalised mean-pooled embeddings, shape (n, 384)."""
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encs = self.tok.encode_batch(batch)
            # The tokenizer may pad internally — trust ITS attention mask,
            # otherwise pad tokens poison the mean pooling.
            maxlen = max(len(e.ids) for e in encs)
            ids = np.zeros((len(batch), maxlen), dtype=np.int64)
            mask = np.zeros((len(batch), maxlen), dtype=np.int64)
            for i, enc in enumerate(encs):
                ids[i, : len(enc.ids)] = enc.ids
                mask[i, : len(enc.attention_mask)] = enc.attention_mask
            hidden = self.sess.run(
                None,
                {"input_ids": ids, "attention_mask": mask,
                 "token_type_ids": np.zeros_like(ids)},
            )[0]
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9,
                                                        None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True),
                              1e-9, None)
            out[start : start + len(batch)] = pooled
        return out


def embed_corpus_cached(docs, cache="/tmp/nfcorpus_minilm.npz"):
    """Embed all docs once; cache to disk (doc order = sorted ids)."""
    doc_ids = sorted(docs)
    if os.path.isfile(cache):
        data = np.load(cache, allow_pickle=True)
        if list(data["doc_ids"]) == doc_ids:
            return doc_ids, data["embs"]
    model = MiniLM()
    embs = model.embed([docs[d] for d in doc_ids])
    np.savez(cache, doc_ids=np.array(doc_ids), embs=embs)
    return doc_ids, embs


class DenseRetriever:
    def __init__(self, docs):
        self.doc_ids, self.embs = embed_corpus_cached(docs)
        self.model = MiniLM()

    def search(self, query, k=100, query_emb=None):
        if query_emb is None:
            query_emb = self.model.embed([query])[0]
        sims = self.embs @ query_emb
        top = np.argsort(-sims)[:k]
        return [(self.doc_ids[i], float(sims[i])) for i in top]
