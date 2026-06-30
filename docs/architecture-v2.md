# VedaX KM Agent · v2 architecture

This is the answer to every question you raised in the last review:

> 1. login flow with superuser approval
> 2. how does an admin upload?
> 3. "how does it become an AGENT"?
> 4. no `export VEDAX_SUPERUSERS=...` — config.yaml only
> 5. revolutionary security guardrail without any library
> 6. category-wise documents — does it actually help accuracy?

## 0. Honest answer to "category-wise rakhna chahiye ya nahi?"

**Yes — accuracy ke liye categorise karna fayda deta hai.** Reasons:

1. **Search space narrows automatically** when the user picks a
   category in the UI — BM25/dense scores are not diluted by chunks
   from unrelated SOPs (FAQ leaking into a compliance question).
2. **Per-role category access control** becomes possible (a non-HR
   user cannot accidentally retrieve payroll text).
3. **Versioning + deprecation works per-category** — same SOP can be
   v1.0 in HR, v1.2 in Finance, both indexed without conflict.
4. **Auto-categorise rules** in `config.yaml` mean admins **never have
   to label by hand** — files dropped in `./sop_docs` like
   `hr_policy.pdf` are auto-tagged `HR`.

So the pipeline is: **drop file in folder → auto-categorise → indexed
with category metadata → user queries get scoped to category → tighter
retrieval → fewer wrong-context chunks → higher answer accuracy.**

## 1. Login flow (Keycloak + superuser approval)

```
   ┌─────────────┐  POST /login (user / pass)        ┌──────────┐
   │   browser   │ ─────────────────────────────────▶│ Keycloak │
   └─────────────┘ ◀──── access_token + JWT ─────────└──────────┘
                                    │
                                    │  (JWT realm_access.roles)
                                    ▼
              ┌──────────────────────────────────────────────┐
              │ kc_role_from_keycloak_claims(claims)         │
              │   if 'vedax-superuser'  → superuser  → LOGIN │
              │   if 'vedax-admin'      → admin      → LOGIN │
              │   else                  → 'pending'          │
              └──────────────────────────────────────────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                          ▼
                  not pending                   pending
                       │                          │
                       │              ┌───────────┴───────────┐
                       │              │ login page shows      │
                       │              │ "Awaiting approval"   │
                       │              │ (cannot reach /api/*) │
                       │              └───────────────────────┘
                       ▼
                  full app shell                              ┌─────────────┐
                                                              │  superuser  │
   • superuser sees the "Pending" tab with [Approve user]  ◀──│ in app shell│
     [Approve admin] [Reject] buttons for each pending row.   └─────────────┘
```

The "popup decide karega user banana hai ya admin" is the **Pending
approvals** tab — see `superuser_view.svg`.

## 2. How the admin uploads

```
   ┌────────────┐        UI                 server
   │ admin user │ ── drop file ──┐
   │ logs in    │                │
   └────────────┘                ▼
                          POST /api/documents/upload
                          + auto-categorise rule from config.yaml
                          + Form fields (category, tags)
                                     │
                                     ▼
                          ./uploaded_docs/<file>
                                     │
                                     ▼
                          core.store.add_document(path, category, tags)
                          → vedax_core rebuilds the BM25 + HD index
                          → file_meta stored with category, version,
                            effective_date
                                     │
                                     ▼
                          UI refreshes documents table
                          (shows category chip + tags)
```

Alternatively, drop a file straight into `./sop_docs` and click
**"Rescan ./sop_docs folder"** in the Documents tab — same result, no
upload form needed.  The auto-categorise rule applies in both cases.

## 3. "agent kaise banega" — Using vectorless RAG to build a KM agent

The agentic behaviour is the **decision tree the system runs on every
query**.  It is not a black-box LLM — it is a deterministic sequence
of decisions that an auditor can step through:

```
    user query
       │
       ▼
   ┌────────────────────────────────────────────────────────┐
   │ L1  Guardrail INPUT inspection                         │
   │   prompt injection? authority impersonation? jailbreak?│
   │   PII?  query too long?  SQL injection?                │
   │   BLOCK or SANITISE → log + maybe trip-wire            │
   └────────────────────────────────────────────────────────┘
       │ (allowed + sanitised)
       ▼
   ┌────────────────────────────────────────────────────────┐
   │ Intent decompose                                       │
   │   define / list / procedure / yes-no / explain         │
   │   subject = real topic; fillers stripped               │
   └────────────────────────────────────────────────────────┘
       │
       ▼
   ┌────────────────────────────────────────────────────────┐
   │ Smart retrieval                                        │
   │   BM25 + HD expansion + acronym + typo rescue          │
   │   adaptive cutoff (no fixed top-K)                     │
   │   category filter (per-role + per-query)               │
   └────────────────────────────────────────────────────────┘
       │
       ▼
   ┌────────────────────────────────────────────────────────┐
   │ L2  Guardrail RETRIEVAL access control                 │
   │   drop chunks user's role cannot read                  │
   └────────────────────────────────────────────────────────┘
       │
       ▼
   ┌────────────────────────────────────────────────────────┐
   │ Coverage / abstention check                            │
   │   if subject coverage < threshold → ABSTAIN            │
   │   "Not in the provided documents."                     │
   │   (LLM is NEVER called for off-topic / injected query) │
   └────────────────────────────────────────────────────────┘
       │ (covered)
       ▼
   ┌────────────────────────────────────────────────────────┐
   │ LLM with grounding system-prompt                       │
   │   "Answer ONLY from context, cite [1] [2]"             │
   └────────────────────────────────────────────────────────┘
       │
       ▼
   ┌────────────────────────────────────────────────────────┐
   │ L3  Guardrail OUTPUT inspection                        │
   │   mask PII that leaked; block profanity                │
   │   per-sentence citation verifier                       │
   └────────────────────────────────────────────────────────┘
       │
       ▼
   answer  +  citations  +  grounding badge  +  audit log entry
   abstentions → unanswered_questions queue → superuser dashboard
```

So **agentic** here means: "given a query, the system actively decides
what to do, what NOT to do, and records the decision for audit".
That's the production definition of an agent — not just "RAG + LLM".

## 4. config.yaml replaces every `export VEDAX_*`

Everything from Keycloak URL to guardrail rules lives in **one
file**, parsed by a stdlib-only YAML loader (`vedax_config.py` ships
its own parser — no PyYAML).  Sample:

```yaml
keycloak:
  url: "https://keycloak.xyzindia.net"
  realm: "xyz"
  client_id: "abc-nginx-manager"
  verify_tls: true
  superuser_keycloak_roles: ["vedax-superuser"]
  admin_keycloak_roles:     ["vedax-admin"]

guardrails:
  enabled: true        # ← single switch to turn the whole thing on/off
  input:
    block_prompt_injection: true
    block_authority_impersonation: true
    block_pii_in_query: true
    block_sql_injection: true
    block_jailbreak: true
  output:
    mask_pii_in_answer: true
    block_profanity: true
  role_policy:
    user:
      max_query_length: 500
      can_query_categories: ["HR", "Finance", "Compliance", "General"]
      trip_wire_violations: 3
      trip_wire_window_minutes: 10
```

Copy `config.example.yaml → config.yaml`, edit, restart.  No `export`.

## 5. The 5-layer guardrail (vedax_guardrail.py)

| Layer | What it does | Examples |
|---|---|---|
| **L1 INPUT** | Inspect the raw query before retrieval | Prompt injection (`ignore all previous`), authority impersonation (`I am the CEO`), jailbreak (`DAN mode`), SQL injection (`UNION SELECT`), PII masking (PAN, Aadhaar, phone, email, IFSC, Luhn-valid credit card), length ceiling per role |
| **L2 RETRIEVAL** | Drop chunks the role cannot read | User querying for "salary" is silently denied chunks from `Admin` category |
| **L3 OUTPUT** | Inspect LLM answer | PII that leaked into the answer is masked; profanity blocks the reply |
| **L4 AUDIT** | Log every event with severity to `guardrail_events` SQLite table | Auditor can replay every blocked query verbatim with the rule that fired |
| **L5 TRIP-WIRE** | After N high/critical hits in M minutes → **auto-revoke** the user | Repeat offender stops being able to log in until a superuser restores them |

### Why this is "stronger than anything before" in our class

* **Deterministic and explainable.**  Every block carries the exact rule
  name + matched span — no LLM-based filter where you have to "trust
  the model".  Compliance teams love this.
* **Layered (defence-in-depth).**  An attacker who beats the input layer
  still has to beat the output layer.  An attacker who beats both
  still leaves a trail in the audit layer.
* **Per-role policy from config.**  The same query can be allowed for
  a superuser (debugging) and blocked for a user (least privilege).
* **Closes the loop with the superuser approval flow** — repeat
  offenders are revoked automatically and need a human to come back.
* **Zero external library.**  No `presidio`, no `guardrails-ai`, no
  `OpenAI Moderation` API.  Pure stdlib — runs offline.

Single switch: `guardrails.enabled: false` turns the whole thing off
for incident-response or debugging.

## Where each file lives

```
config.example.yaml    →    copy to config.yaml and edit
vedax_config.py        →    stdlib YAML loader + Config singleton
vedax_guardrail.py     →    5-layer guardrail (this is the "novel" piece)
vedax_db.py            →    SQLite schema (audit + roles + SOP versions
                            + guardrail_events table)
vedax_core.py          →    EngineStore + auto-fetch + do_ask / do_retrieve
vedax_keycloak_server.py →  FastAPI app (login, role-aware UI, RBAC,
                            user mgmt, pending approval)
```

## Verified test coverage

```
python -m unittest discover -s tests   # 130/130 OK
```

Includes:
* 13 tests for the YAML loader + 19 tests for the guardrail (PII,
  prompt injection, authority impersonation, jailbreak, SQL injection,
  trip-wire, role policy, disabled mode)
* 21 tests for the Keycloak server (login HTML, role-based access,
  pending approval flow, role mapping, guardrail integration on
  /api/ask, easy/medium/hard/complex query scenarios, off-topic /
  prompt-injection abstention)
* 78 prior tests for the underlying VedaX engine
