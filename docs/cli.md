# Command-line reference

Three entry points, each for a different audience.

| script | for | scope |
|---|---|---|
| `veda.py` | end users — one liner | retrieval + grounded chat over a folder |
| `python -m vedax` | developers — full sub-commands | the VEDA-X hybrid pipeline |
| `python -m veda` | developers — core engine only | zero-dependency retrieval |

## `veda.py`

```text
python veda.py [path] [query] [--chat]
```

- `path` — optional folder, defaults to the current directory.
- `query` — optional; if omitted the script enters an interactive REPL.
- `--chat` — force chat mode even when `VEDAX_LLM_URL` is unset (in
  which case the call will fail with a clear message).

If `VEDAX_LLM_URL` is set in the environment, the script answers with a
grounded LLM response instead of just listing chunks.

### Environment variables

| variable | default | meaning |
|---|---|---|
| `VEDAX_LLM_URL` | unset | LLM endpoint base URL |
| `VEDAX_LLM_MODEL` | `gpt-oss:20b` | model name to request |
| `VEDAX_LLM_API` | `ollama` | one of `ollama`, `openai` |
| `VEDAX_LLM_TOKEN` | unset | bearer token, sent as `Authorization` |

## `python -m vedax`

The full sub-command surface.

```text
python -m vedax [--no-dense] <command> ...

commands:
  ask      <paths...> <query>          one-shot retrieve and print
  compare  <paths...> <query>          plain dense RAG vs VEDA-X side by side
  index    <paths...> -o file.vedax    build a persistent on-disk index
  search   <file.vedax> <query>        query a saved index
  repl     <paths...>                  interactive search loop
  chat     <paths...> [query]          retrieve + stream LLM answer
```

Common flags:

- `--no-dense` — skip the neural stage. Use when `onnxruntime` is not
  installed or when you want fully offline operation.
- `-k N` — request the top-N results (default 5 for search, 6 for chat).
- For `chat`: `--llm-url`, `--llm-model`, `--llm-api`, `--llm-token`
  override the environment variables of the same names.

### Examples

```sh
# A side-by-side comparison of plain dense RAG and VEDA-X
python -m vedax compare ~/Documents/contracts \
    "what is the arbitration clause"

# Build and save once, query many times
python -m vedax index ~/Documents/contracts -o contracts.vedax
python -m vedax search contracts.vedax "termination notice period"

# Chat with retrieval grounding, against a custom Ollama gateway
python -m vedax chat ~/Documents/contracts \
    "summarise the payment terms" \
    --llm-url https://ollamagw.example.net --llm-model gpt-oss:20b
```

## `python -m veda` (core engine only)

The pure-stdlib core, useful when you want maximum portability:

```text
python -m veda ask     <file> <query>
python -m veda index   <file...> -o corpus.veda
python -m veda search  <corpus.veda> <query>
python -m veda repl    <file>
```

This package has no PDF or scan support out of the box — feed it text.
For PDF and OCR, use `python -m vedax` or `python veda.py`.

## Evaluation harnesses

```text
python -m eval.run_eval bm25 dense veda_x hybrid  # NFCorpus head-to-head
python -m eval.generalize                          # 3 BEIR datasets
python -m eval.financebench                        # FinanceBench
```

See [evaluation.md](./evaluation.md) for the methodology behind each.
