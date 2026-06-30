# Getting started

Five minutes from clone to a working query over your own files.

## 1. Get the code

```sh
git clone [https://github.com/AbhiLabs421/VEDA.git}
cd VEDA
```

That is the entire install for the **lexical mode** of the system. The
core (`veda/`) and the pure-stdlib parts of `vedax/` do not need any
third-party package.

## 2. Optional: enable the neural stage

The hybrid pipeline that wins on the benchmarks adds one neural
retriever (all-MiniLM-L6-v2 over ONNX). It is optional and only used
when present:

```sh
pip install onnxruntime tokenizers numpy
```

If these are not installed the system silently falls back to lexical
plus hyperdimensional retrieval — still significantly better than BM25,
just without the dense vector contribution.

## 3. Run your first query

Put any mix of files into a folder (`.txt`, `.md`, `.pdf`, `.png` and
other text-like or PDF-like formats) and ask a question:

```sh
python veda.py "what is the penalty for late delivery"
```

The script will:

1. Walk the current directory recursively.
2. Extract text from each file (PDFs via the built-in parser, scanned
   page images via the built-in OCR).
3. Build a chunk index (lexical + hyperdimensional, plus the dense stage
   if the neural dependencies are installed).
4. Print the top five chunks ranked by VEDA-X.

Sample output:

```
indexing '.' ...
indexed 47 chunks in 3.2s (hybrid mode)

  1. [contract.pdf] Penalty of 2 percent of contract value per week of delay, capped at 10 ...
  2. [notes.txt] discussed late delivery penalty - vendor agreed to standard 2 percent ...
  3. [appendix.txt] all disputes shall be settled through arbitration in New Delhi ...
```

## 4. Point at any folder

```sh
python veda.py ~/Documents/contracts "what is the arbitration clause"
```

## 5. Interactive search

Omit the question and the script opens a REPL:

```sh
python veda.py
# search> what is the arbitration clause
# search> what was the FY2018 capital expenditure
# search>
```

## 6. Grounded answers from your own LLM

Set the LLM endpoint and the script becomes a grounded chat front-end.
Ollama-compatible and OpenAI-compatible endpoints are both supported,
over Python's standard-library `urllib` only.

```sh
export VEDAX_LLM_URL=https://your-ollama-gateway.example.net
export VEDAX_LLM_MODEL=gpt-oss:20b
python veda.py "summarise the quarterly report"
```

The model is told to answer only from the retrieved chunks and to cite
them inline as `[1]`, `[2]` and so on. Your documents never leave your
machine; only the retrieved snippets are sent to the LLM you control.

See [CLI reference](./cli.md) for every flag and `python -m vedax`
sub-command.
