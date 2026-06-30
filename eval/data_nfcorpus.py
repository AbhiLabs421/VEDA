"""NFCorpus loader (BEIR task: 3.6K PubMed abstracts, graded qrels).

Data comes from the github.com/keeeevinShen/RAG_nfcorpus mirror of the
standard BEIR NFCorpus distribution; ``fetch()`` clones it on demand.
"""

import csv
import os
import subprocess

MIRROR = "https://github.com/keeeevinShen/RAG_nfcorpus.git"
DATA_DIR = os.environ.get("NFCORPUS_DIR", "/tmp/nf/assets")


def fetch():
    if not os.path.isdir(DATA_DIR):
        subprocess.run(
            ["git", "clone", "--depth", "1", MIRROR,
             os.path.dirname(DATA_DIR)],
            check=True,
        )
    return DATA_DIR


def load_corpus():
    """{doc_id: text} with title prepended."""
    fetch()
    docs = {}
    with open(os.path.join(DATA_DIR, "corpus.csv"), newline="",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs[row["_id"]] = (row["title"] + ". " + row["text"]).strip(". ")
    return docs


def load_queries():
    """{query_id: text}."""
    fetch()
    queries = {}
    with open(os.path.join(DATA_DIR, "queries.csv"), newline="",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            queries[row["_id"]] = (row["title"] + " " + row["text"]).strip()
    return queries


def load_qrels(split="test"):
    """{query_id: {doc_id: grade}} for 'train' | 'validation' | 'test'."""
    fetch()
    qrels = {}
    with open(os.path.join(DATA_DIR, split + ".csv"), newline="",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = \
                int(row["score"])
    return qrels
