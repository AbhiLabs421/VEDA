# Critical (atomic) SOP blocks

In a regulated environment some SOPs cannot tolerate partial
retrieval.  An incident-response runbook, a settlement-rollback
procedure, a KYC red-flag list — if the retriever returns "Step 2"
and "Step 5" without "Step 1" the operator may take a destructive
action and the institution gets fined.

This page documents the two ways to mark "this is atomic, never
split" inside VEDA-X.

---

## 1. Inline marker

Anywhere inside an ordinary `.txt` / `.md` / `.docx` / `.xlsx` /
`.pdf` you can wrap a span with:

```
[[CRITICAL: Trade Cancel Procedure]]
Step 1: Freeze the settlement queue.
Step 2: Notify the risk desk.
Step 3: Get dual approval from CRO + CFO.
Step 4: Reverse the trade in NDS-OM.
Step 5: File the regulatory report within 1 hour.
[[/CRITICAL]]
```

**Rules:**

- Opening tag: `[[CRITICAL: <title>]]` — `<title>` is any text on a
  single line.  Title is shown in the UI badge and audit log.
- Closing tag: `[[/CRITICAL]]`
- Tags are case-insensitive (`[[critical: ...]]` also works).
- An unclosed opening tag is **ignored** — the block is not
  indexed (we err on the side of indexing rather than losing the
  document).
- Nesting is flattened — the outermost block wins.

**Guarantee:**

The chunker MUST emit every critical block as exactly ONE chunk.
A chunk boundary that would have fallen inside the block is expanded
to swallow the entire span.  The markers themselves are stripped
from the chunk text so they do not appear in the LLM context or the
UI snippet.

## 2. Folder convention

Every file dropped into `./critical_sops/` (configurable via
`documents.critical_fetch_dir` in `config.yaml`) is treated as
100 % critical — the entire file is one atomic chunk regardless of
length.  Use this when an entire document is a compliance procedure
and you don't want to add inline markers.

```
./critical_sops/
├── trade_cancel_runbook.txt          (whole file = atomic)
├── kyc_red_flags.docx                (whole file = atomic)
└── settlement_rollback_2024.xlsx     (whole file = atomic)
```

The chunker stamps each whole-file chunk with
`is_critical: true`, `whole_file: true` and `critical_title` =
filename basename.

## What the UI/LLM see

A retrieved chunk that is critical comes back with:

```json
{
  "file": "./critical_sops/trade_cancel_runbook.txt",
  "snippet": "Step 1: Freeze ... Step 5: File the regulatory report",
  "is_critical": true,
  "critical_title": "Trade Cancel Procedure"
}
```

`/api/ask` adds the same flags to every entry of `sources[]`.  The
web UI renders an explicit `⚠ CRITICAL — <title>` badge so the
operator cannot miss it.

The LLM system prompt receives an extra instruction:

> If a chunk is marked ⚠ CRITICAL, reproduce its steps VERBATIM and
> in order — do NOT summarise, paraphrase, reorder or omit any step.

So the model treats the block as authoritative and is told NOT to
"helpfully" condense it.

## Why this matters

| Scenario | Without critical blocks | With critical blocks |
|---|---|---|
| KYC red-flag list with 8 conditions | top-3 chunks → only 4 conditions surface | whole list returned, all 8 visible |
| Settlement rollback runbook | Step 2 + Step 4 retrieved without Step 1 → operator does Step 2 first and corrupts state | whole runbook returned as one chunk, order preserved |
| Trade-cancel procedure with dual-approval rule on Step 3 | Approval clause sits in the chunk after the one retrieved → operator skips approval | clause is in the same chunk as the trigger step |

## Authoring tips

- Keep critical blocks **short** — 5–15 lines is typical.  Long
  blocks dominate the LLM context window and crowd out routine
  hits.
- One block per procedure.  If you have two procedures, use two
  `[[CRITICAL]]` blocks, not one big one.
- The block title should be the operator-facing name of the
  procedure ("Trade Cancel", "KYC Red Flags", "Settlement
  Rollback").  It is rendered verbatim in the UI badge.
- Avoid putting routine paragraphs inside `[[CRITICAL]]` — they
  inflate the atomic chunk and drown the actually-critical content.

## Testing

`tests/test_critical_blocks.py` (13 tests) verifies:

- the parser handles single, multiple, unclosed and case-variant
  markers;
- chunk-boundary expansion never splits a block;
- folder convention emits a whole-file atomic chunk;
- `is_critical` and `critical_title` propagate from the chunk all
  the way out to `smart_search` and `/api/ask`.

Run them with:

```
python -m unittest tests.test_critical_blocks -v
```

## Configuration

```yaml
documents:
  auto_fetch_dir: "./sop_docs"
  critical_fetch_dir: "./critical_sops"   # this folder = atomic files
```

Default if the key is absent: `./critical_sops` is created on
startup.  The startup banner reports both folders:

```
  Auto-fetch   : ./sop_docs       (indexed 42 docs)
  Critical SOPs: ./critical_sops  (atomic chunks)
```
