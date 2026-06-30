"""Unified loaders for the BEIR-format datasets used in our evaluation.

All three expose the same interface: ``load(name)`` returns
``(corpus, queries, qrels_val, qrels_test)`` where qrels are
``{query_id: {doc_id: graded_relevance}}``.

  * nfcorpus    medical lay questions  (3.6K docs,  323 test queries)
  * scifact     scientific claims      (5.2K docs,  505 test queries,
                                        from claims_train with evidence;
                                        claims_dev is held out for tuning,
                                        because the official test split
                                        does not ship with evidence)
  * fiqa        financial QA           (57K docs,   648 test queries)
"""

import csv
import json
import os
import subprocess

SCIFACT_TAR = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
FIQA_MIRROR = "https://github.com/cipriano-sebastiao/fiqa-ir-system.git"
NFCORPUS_MIRROR = "https://github.com/keeeevinShen/RAG_nfcorpus.git"

DATA_ROOT = os.environ.get("BEIR_DIR", "/tmp/beir")


def _ensure_dir():
    os.makedirs(DATA_ROOT, exist_ok=True)


def _git_clone(url, dest):
    if not os.path.isdir(dest):
        subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)
    return dest


# ---------------------------------------------------------- nfcorpus

def _load_nfcorpus():
    _ensure_dir()
    root = os.path.join(DATA_ROOT, "nfcorpus")
    _git_clone(NFCORPUS_MIRROR, root)
    base = os.path.join(root, "assets")

    corpus, queries = {}, {}
    with open(os.path.join(base, "corpus.csv"), newline="",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            corpus[row["_id"]] = (row["title"] + ". " + row["text"]).strip(". ")
    with open(os.path.join(base, "queries.csv"), newline="",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            queries[row["_id"]] = (row["title"] + " " + row["text"]).strip()

    def load_split(name):
        qrels = {}
        with open(os.path.join(base, name + ".csv"), newline="",
                  encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = \
                    int(row["score"])
        return qrels

    return corpus, queries, load_split("validation"), load_split("test")


# ----------------------------------------------------------- scifact

def _load_scifact():
    _ensure_dir()
    root = os.path.join(DATA_ROOT, "scifact")
    os.makedirs(root, exist_ok=True)
    data_dir = os.path.join(root, "data")
    if not os.path.isdir(data_dir):
        import urllib.request, tarfile
        tar_path = os.path.join(root, "scifact.tar.gz")
        if not os.path.isfile(tar_path):
            urllib.request.urlretrieve(SCIFACT_TAR, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(root)

    corpus = {}
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r.get("title", "")
            abstract = r.get("abstract", [])
            if isinstance(abstract, list):
                text += ". " + " ".join(abstract)
            else:
                text += ". " + abstract
            corpus[str(r["doc_id"])] = text.strip(". ")

    def load_claims(name):
        queries, qrels = {}, {}
        with open(os.path.join(data_dir, name + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                qid = str(r["id"])
                evidence = r.get("evidence") or {}
                if not evidence:
                    continue
                queries[qid] = r["claim"]
                for doc_id in evidence:
                    qrels.setdefault(qid, {})[str(doc_id)] = 1
        return queries, qrels

    train_q, train_r = load_claims("claims_train")
    dev_q, dev_r = load_claims("claims_dev")
    queries = {**train_q, **dev_q}
    # The official claims_test ships without evidence, so we report on
    # claims_train (505 judged claims) and tune on claims_dev (188).
    return corpus, queries, dev_r, train_r


# -------------------------------------------------------------- fiqa

def _load_fiqa():
    _ensure_dir()
    root = os.path.join(DATA_ROOT, "fiqa")
    _git_clone(FIQA_MIRROR, root)
    base = os.path.join(root, "data", "fiqa")

    corpus, queries = {}, {}
    with open(os.path.join(base, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = (r.get("title", "") + " " + r.get("text", "")).strip()
            corpus[r["_id"]] = text
    with open(os.path.join(base, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[r["_id"]] = r.get("text", "")

    def load_qrels(name):
        out = {}
        with open(os.path.join(base, "qrels", name + ".tsv"),
                  encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                qid, did, score = line.rstrip().split("\t")
                out.setdefault(qid, {})[did] = int(score)
        return out

    return corpus, queries, load_qrels("dev"), load_qrels("test")


# ------------------------------------------------------------- public

LOADERS = {
    "nfcorpus": _load_nfcorpus,
    "scifact": _load_scifact,
    "fiqa": _load_fiqa,
}


def load(name):
    """``corpus``, ``queries``, ``qrels_val``, ``qrels_test``."""
    return LOADERS[name]()
