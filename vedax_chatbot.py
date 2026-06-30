#!/usr/bin/env python3
"""
====================================================================
  VEDAX CHATBOT — apne document ke saath chat karo (Ollama LLM)
  Sab config yahin file mein hai — koi `export` ki zaroorat nahi.
====================================================================

KYA HOTA HAI (har sawaal pe):
  1. Tera sawaal SMART_SEARCH se VedaX me jaata hai:
        - intent + subject + fillers decompose
        - subject-focused rescore (filler hijack rok-ta hai)
        - ADAPTIVE cutoff (fixed top-K nahi — jitne sahi utne chunks)
  2. Confidence + subject-coverage check (abstention guard) —
     agar chunks weak hain, LLM ko call hi nahi karta
  3. Retrieved chunks + tera sawaal LLM ko bhejte hain (Ollama gateway)
  4. LLM sirf un chunks se answer deta hai, [1][2] citations ke saath
  5. Citation verifier — har sentence ka grounding check

SETUP (ek baar):
  1. Yeh file aur 'prediction-abhishek-kumar-dev' folder same directory mein rakho
  2. CONFIG section mein apni file ka path daalo (PDF ya .txt)
  3. python vedax_chatbot.py

DEPENDENCY: sirf Python stdlib. (onnxruntime optional — agar ho to
dense/MiniLM stage bhi chalega, accuracy thodi better hogi.)
====================================================================
"""

import sys
import os
import time

# ════════════════════════════════════════════════════════════════
#  CONFIG — sab yahan, kuch bhi export karne ki zaroorat nahi
# ════════════════════════════════════════════════════════════════

# Veda project folder (zip extract karne ke baad jo folder bana)
VEDA_DIR = "./prediction-abhishek-kumar-dev"

# Tera document — PDF ya .txt dono chalega
# (PDF pehli baar slow hai — agar already .txt nikala hai toh woh use karo)
DOCUMENT_PATH = "./hr_policy.txt"

# ── Ollama / LLM gateway ──────────────────────────────────────────
LLM_URL   = "https://ollamagw.xyzindia.net"
LLM_MODEL = "gpt-oss:20b"
LLM_API   = "ollama"     # "ollama" → /api/chat   |  "openai" → /v1/chat/completions
LLM_TOKEN = None         # agar gateway ko auth token chahiye, yahan daalo

# ── Retrieval settings ─────────────────────────────────────────────
TOP_K              = 6     # MAX chunks (adaptive cutoff se kam ho sakta hai)
CHUNK_TOKENS       = 120   # har chunk mein kitne words
OVERLAP_TOKENS     = 20    # consecutive chunks ke beech overlap
ABSTAIN_THRESHOLD  = 0.30  # confidence < isse → LLM ko call nahi karega
USE_DENSE          = False # True karo agar onnxruntime installed hai (better accuracy)

# ════════════════════════════════════════════════════════════════
#  SETUP — yahan se neeche kuch change karne ki zaroorat nahi
# ════════════════════════════════════════════════════════════════

sys.path.insert(0, VEDA_DIR)

try:
    from vedax import VedaX
except ImportError as e:
    print(f"❌ VedaX import nahi hua: {e}")
    print(f"   Check karo: VEDA_DIR = {VEDA_DIR!r} sahi hai kya?")
    sys.exit(1)

if not os.path.exists(DOCUMENT_PATH):
    print(f"❌ Document nahi mila: {DOCUMENT_PATH}")
    sys.exit(1)


def build_engine():
    print(f"📄 Indexing: {DOCUMENT_PATH}")
    t0 = time.time()
    engine = VedaX(use_dense=USE_DENSE,
                   chunk_tokens=CHUNK_TOKENS,
                   overlap_tokens=OVERLAP_TOKENS)
    engine.add(DOCUMENT_PATH)
    engine._finalize()
    mode = "hybrid (BM25 + HD expansion + MiniLM dense)" if engine.use_dense \
        else "lexical + hyperdimensional (no dense — onnxruntime not used)"
    print(f"✅ {len(engine.chunks)} chunks indexed in {time.time()-t0:.2f}s")
    print(f"   Mode: {mode}\n")
    return engine


def print_chunks(hits):
    """Retrieved chunks dikhao — taaki tu verify kar sake VedaX kya bhej raha hai LLM ko."""
    print(f"\n┌─ Retrieved {len(hits)} chunks (yeh LLM ko bheje gaye) ─────────")
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h["snippet"].split())
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"│ [{i}] {snippet}")
    print("└" + "─" * 56)


def ask(engine, query):
    """Ek sawaal poocho — SMART retrieval + LLM call + citation check."""
    from vedax.llm import stream_chat
    from vedax.grounding import retrieval_confidence, verify_citations
    from vedax.intent import subject_coverage

    # ── Step 1: SMART Retrieve ──────────────────────────────────────
    #   - query ko decompose karta hai (intent + subject + fillers)
    #   - subject pe focused search
    #   - adaptive cutoff (fixed TOP_K nahi — jitne sahi chunks utne hi)
    res = engine.smart_search(query, max_keep=TOP_K)
    hits   = res["hits"]
    parsed = res["parsed"]

    print(f"\n🧠 Parsed: intent={parsed['intent']}  "
          f"subject={parsed['subject']!r}")
    if parsed.get("typo_corrected"):
        print(f"   (typo corrected from your query)")
    print(f"📦 Adaptive cutoff: k={res['k_selected']}  "
          f"(dropped {res['dropped']} noise chunks below the plateau)")
    print_chunks(hits)

    # ── Step 2: Subject-aware confidence check ──────────────────────
    confidence, reasons = retrieval_confidence(query, hits, engine.sem)
    subj_cov = subject_coverage(parsed, hits)

    # Subject coverage ko ek strong signal manno: agar subject 50%+ hit
    # hai to filler-driven low coverage penalty ko ignore karo.
    if subj_cov >= 0.5 and hits:
        confidence = max(confidence, 0.4 + 0.5 * subj_cov)

    print(f"\n📊 Retrieval confidence: {confidence:.3f}"
          f"  (subject coverage: {subj_cov*100:.0f}%, "
          f"threshold: {ABSTAIN_THRESHOLD})")
    if reasons:
        print(f"   Signals: {', '.join(reasons)}")

    if confidence < ABSTAIN_THRESHOLD or not hits:
        print("\n⚠️  ABSTAINING — chunks is sawaal ka jawab nahi rakhte.")
        print("   LLM ko call nahi kiya — fabrication se bachne ke liye.")
        print("   → 'Not in the provided documents.'")
        return

    # ── Step 3: Build prompt ─────────────────────────────────────────
    context = "\n\n".join(
        f"[{i+1}] {h['file']}\n{h['snippet']}"
        for i, h in enumerate(hits)
    )
    system_msg = (
        "You answer questions STRICTLY from the provided context. "
        "Cite sources as [1], [2] inline at the end of every claim. "
        "If a question cannot be answered from the context, reply "
        "exactly: 'Not in the provided documents.' "
        "Do not invent facts. Be concise."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",
         "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
    ]

    # ── Step 4: Stream LLM answer ────────────────────────────────────
    print("\n💬 Answer:")
    print("─" * 60)
    answer_parts = []
    try:
        for chunk in stream_chat(
            url=LLM_URL, model=LLM_MODEL, messages=messages,
            api=LLM_API, token=LLM_TOKEN,
        ):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            answer_parts.append(chunk)
    except Exception as e:
        print(f"\n❌ LLM call failed: {e}")
        print(f"   Check: {LLM_URL} reachable hai? Model {LLM_MODEL!r} loaded hai?")
        print(f"   Try: curl {LLM_URL}/api/tags")
        return
    print("\n" + "─" * 60)

    # ── Step 5: Citation verification ─────────────────────────────────
    answer = "".join(answer_parts)
    if answer.strip() and answer.strip() != "Not in the provided documents.":
        results, grounded_fraction = verify_citations(answer, hits)
        badge = "✅ OK" if grounded_fraction >= 0.8 \
            else "⚠️  WARN" if grounded_fraction >= 0.5 else "❌ UNGROUNDED"
        print(f"\n{badge}  grounded = {grounded_fraction*100:.0f}% "
              f"of sentences are backed by retrieved chunks")
        for s in results:
            if not s["supported"]:
                print(f"   ! unsupported (cited {s['citations']}): "
                      f"{s['sentence'][:100]}")


# ════════════════════════════════════════════════════════════════
#  MAIN — interactive chat loop
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  VEDAX CHATBOT  ·  smart_search + adaptive cutoff")
    print(f"  Document : {DOCUMENT_PATH}")
    print(f"  LLM      : {LLM_MODEL} @ {LLM_URL} ({LLM_API})")
    print(f"  Top-K    : up to {TOP_K} chunks "
          "(adaptive cutoff active)")
    print("=" * 60 + "\n")

    engine = build_engine()

    print("Type your question. 'q' / 'quit' / empty line se exit.\n")

    while True:
        try:
            query = input("🔍 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query or query.lower() in ("q", "quit", "exit", "bye"):
            print("Bye!")
            break

        t0 = time.time()
        ask(engine, query)
        print(f"\n⏱  {time.time()-t0:.2f}s\n")


if __name__ == "__main__":
    main()
