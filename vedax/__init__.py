"""VEDA-X — hybrid retrieval over your own documents (txt/md/pdf).

The pipeline that beat both BM25 and the standard RAG retriever
(all-MiniLM-L6-v2) on the NFCorpus benchmark, packaged for real use:

    python -m vedax ask  docs/ "what does the contract say about penalties"
    python -m vedax compare docs/ "..."     # plain RAG vs VEDA-X, side by side
    python -m vedax index docs/ -o my.vedax
    python -m vedax search my.vedax "..."

Needs: pip install onnxruntime tokenizers numpy (PDF extraction is built in)
(falls back to lexical-only mode with --no-dense / if onnxruntime is missing).
"""

from .engine import VedaX

__all__ = ["VedaX"]
