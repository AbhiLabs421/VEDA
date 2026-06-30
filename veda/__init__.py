"""VEDA — Vectorless Embedding-free Document Architecture.

Semantic search over arbitrarily large documents with:
  * NO vector database (nothing persisted, signatures live in bounded memory)
  * NO trained embedding model (the "embedding table" is a hash function)
  * NO external dependencies (pure Python standard library)

The document indexes itself: distributional semantics are learned on the
fly from the text being ingested, and any token's hypervector can be
regenerated on demand, so there is nothing to store and nothing to train.
"""

from .engine import Veda

__all__ = ["Veda"]
__version__ = "0.1.0"
