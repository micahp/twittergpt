# twitterGPT — Product Requirements Document

**Author:** micahgp@gmail.com
**Date:** 2026-05-25
**Status:** Draft v1 (MVP)

---

## 1. Summary

twitterGPT is a web app that lets a person upload their Twitter/X data archive
and produces a small language model finetuned on *their own* tweets. Once
trained, the user can prompt the model to generate new tweets that sound like
them — same voice, topics, and cadence.

The long-term vision is a multi-user product where anyone can do this. **This
PRD scopes the MVP**: a single working pipeline (upload → parse → finetune →
generate) proven on the author's own archive before any user-acquisition or
multi-tenancy work. We build with multi-user in mind (clean boundaries, a
compute abstraction) but do not build multi-tenant infrastructure yet.

---

## 2. Goals & Non-Goals

### Goals (MVP)
1. Upload a Twitter archive `.zip` (up to ~10 GB) through the web UI.
2. Parse and clean the archive into a training dataset of the user's authentic voice.
3. Finetune a small LLM on that dataset — runnable **locally** (desktop GPU) or
   on **Daytona** cloud GPU via a single compute abstraction.
4. Serve the finetuned model and generate tweets from a prompt or "free-run."
5. Prove the whole loop end-to-end on the author's real 33K-tweet archive.

### Non-Goals (explicitly deferred)
- Multi-tenant auth, billing, per-user model hosting at scale.
- Training large models from scratch (we finetune, not pretrain).
- Posting directly to X / X API integration.
- Mobile apps, fancy analytics dashboards.
- Real-time / streaming finetuning.

---

## 3. Target User & Use Cases

**Primary user (MVP):** the author — a heavy Twitter user with a large archive,
technically capable, with a local GPU desktop and Daytona credits.

**Use cases:**
- "Generate a tweet in my voice about \<topic\>."
- "Give me 10 tweet ideas that sound like me."
- "Free-run: just produce tweets I might have written."
- (later) Anyone uploads their archive and gets the same.

---

## 4. Source Data — Reality of the Archive

Analyzed from the author's actual archive
(`twitter-2026-05-17-...zip`, 9.85 GB).

| Metric | Value |
|---|---|
| Archive size | 9.85 GB (mostly `assets/` media — 6,211 files) |
| Core data files | `data/*.js` (74 files) |
| `data/tweets.js` | 65.4 MB — the training source |
| Total tweets | 33,143 |
| Retweets (excluded — not user's words) | 8,036 |
| Replies (included — user's words) | 19,747 |
| Original standalone tweets | 5,360 |
| Avg original tweet length | 96.6 chars |
| Tweets by year | 2021: 7.6K, 2022: 8.0K, 2023: 4.4K, 2024: 4.1K, 2025: 5.3K, 2026: 2.4K |

### Archive structure
```
Your archive.html
assets/            # media (images/video) — ~6,200 files, the bulk of the 9.85 GB
data/
  tweets.js        # 65 MB — primary source. JS-wrapped JSON array.
  tweet-headers.js
  like.js, direct-messages.js, follower.js, ... (71 more)
  account.js, profile.js   # username, display name for system prompt
```

### `tweets.js` format
The file is `window.YTD.tweets.part0 = [ {...}, ... ]` — strip the assignment
prefix (everything before the first `[`) and parse as JSON. Each element is
`{"tweet": { full_text, created_at, in_reply_to_status_id_str, entities,
favorite_count, retweet_count, ... }}`. All numeric fields are strings.

### Training-data decision (confirmed)
**Use original tweets + replies; strip retweets.** This yields ~25,000 examples
of the user's authentic voice. Rationale:
- Retweets (`full_text` starts with `RT @`) are *other people's* words — they
  pollute voice. **Exclude.**
- Replies are the user's own words (conversational, but theirs). **Include**,
  optionally tagged so the model learns reply-style vs standalone-style.
- Original-only (~5,360) is the purest but risks being too little data and
  overfitting. The +replies set (~25K) is the best signal-to-volume tradeoff.

**Honest caveat:** even 25K short tweets is a *small* dataset (~1–3M tokens).
This is enough for **LoRA/QLoRA finetuning of a small instruct model** to pick
up voice/style. It is **not** enough to teach new knowledge or train a base
model. We set expectations accordingly: twitterGPT mimics *style*, not facts.

---

## 5. Data Pipeline (Ingestion → Training Set)

1. **Upload** the `.zip` (resumable; see §8).
2. **Extract only what's needed** — read `data/tweets.js`, `account.js`,
   `profile.js` directly from the zip without unpacking the 9 GB of media.
   (Media is ignored for MVP training.)
3. **Parse** `tweets.js`: strip JS prefix, `json.loads`, iterate `tweet` objects.
4. **Filter:**
   - Drop retweets (`full_text.startswith("RT @")`).
   - Keep originals and replies; record `is_reply` flag.
   - Optionally drop very short / link-only tweets (configurable threshold).
5. **Clean:**
   - Unescape HTML entities (`&amp;` → `&`).
   - Optionally strip leading `@mentions` from replies (keep voice, drop noise).
   - Optionally strip `t.co` URLs (configurable — they carry no style).
   - Normalize whitespace; keep emoji.
6. **Format** into instruction/chat examples. Two candidate formats:
   - **Completion style:** prompt = short instruction ("Write a tweet."),
     completion = the tweet. Best for free-run generation.
   - **Topic-conditioned (optional):** derive a topic/keyword from hashtags or
     a cheap keyword pass; prompt = "Write a tweet about \<topic\>." Enables
     topic-steerable generation. MVP can ship completion-style first.
7. **Split** train/eval (e.g. 95/5), dedupe near-identical tweets.
8. **Output** a `dataset.jsonl` + a `meta.json` (username, counts, date range)
   used to build the system prompt.

Deliverable: a deterministic `prepare_dataset.py` that turns any standard
Twitter archive into `dataset.jsonl`. This is the most reusable, multi-user-ready
component — write it clean.

---

## 6. Model & Finetuning

### Base model
Pick a small, strong **instruct** model that QLoRA-fits a 6 GB GPU:
- **Local (2060, 6 GB):** Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct, **QLoRA
  (4-bit)**. Recommend **Unsloth** — it cuts VRAM and roughly 2× training speed,
  and a 3B QLoRA fits comfortably in 6 GB.
- **Daytona (A100/L4-class):** scale up to Qwen2.5-7B-Instruct /
  Llama-3.1-8B-Instruct QLoRA for higher quality.

The same training script targets both; only the base-model id and batch/precision
config differ (driven by the compute profile, §7).

### Method
- **QLoRA**: 4-bit base + LoRA adapters (r=16–32, alpha=32, dropout=0.05,
  target attention + MLP projections).
- 1–3 epochs over ~25K short examples (small data → watch for overfitting; use
  eval loss + a held-out qualitative sample to early-stop).
- Output is a small **LoRA adapter** (tens of MB) per user — cheap to store and
  swap. This is the key multi-user-friendly property: one base model, many
  adapters.

### Serving
- **Local:** merge adapter → export **GGUF** → serve via **llama.cpp** /
  Ollama (runs on the 2060 or even CPU). Simple, low-overhead, good for one user.
- **Daytona / scale:** **vLLM** with LoRA-adapter hot-swapping (one base model,
  per-user adapters loaded on demand). This is the multi-user serving path.

### Generation controls (exposed in UI)
temperature, top_p, max tokens, number of candidates, optional topic prompt,
reply-style vs standalone toggle.

---

## 7. Compute Abstraction (Hybrid: Local + Daytona)

A single `ComputeBackend` interface with two implementations so the rest of the
app doesn't care where a job runs:

```
ComputeBackend
  .submit_training(dataset_uri, config) -> job_id
  .status(job_id) -> {queued|running|done|failed, logs, metrics}
  .fetch_artifact(job_id) -> adapter_path
```

- **LocalBackend** — runs the training script as a subprocess on the desktop
  (2x RTX 2060, 32 GB RAM). Use **one** 2060 per QLoRA job (no NVLink → don't
  try to shard one model across both). Optionally: GPU 0 trains while GPU 1
  serves inference. Default for dev and single-user.
- **DaytonaBackend** — provisions a Daytona GPU sandbox/workspace, syncs the
  dataset, runs the same script, pulls back the adapter. Uses the author's
  credits. Default for larger base models and the future multi-user path.

A **compute profile** (`local-3b`, `daytona-7b`) bundles base model + batch size
+ precision so switching backends is one setting. Job state lives in a small
queue/table so the UI can poll progress regardless of backend.

> **Hardware note on the 2x 2060s:** Treat them as two independent 6 GB GPUs,
> not a 12 GB pool. A 3B QLoRA job fits on one. Two cards = run two jobs in
> parallel, or train + serve simultaneously. 32 GB system RAM is plenty for
> 4-bit loading and data prep.

---

## 8. Web Application

### Frontend
- **Next.js + React**, Tailwind. Pages:
  1. **Upload** — drag-drop `.zip`, large-file resumable upload, progress bar.
  2. **Dataset review** — show parsed stats (counts like §4), let user toggle
     include-replies / strip-mentions / strip-urls before training.
  3. **Train** — pick compute profile (local vs Daytona), launch, live job
     status + logs + loss curve.
  4. **Generate** — prompt box + controls (§6), shows candidate tweets, copy
     button, regenerate.

### Backend
- **FastAPI (Python)** — same language as the ML pipeline, no cross-language glue.
- Endpoints: `/upload` (chunked/resumable), `/dataset/prepare`, `/train`,
  `/jobs/{id}` (status/logs), `/generate`.
- **Large uploads:** chunked/resumable (tus protocol or S3-style multipart) —
  a 10 GB single POST is fragile. Store the raw zip on disk (MVP) or object
  storage (multi-user later).
- **Job queue:** start simple — an in-process queue or Redis + RQ — since MVP is
  single-user/low-concurrency. The `ComputeBackend` abstraction means the queue
  only tracks state, not where compute runs.

### Storage (MVP → multi-user path)
- MVP: local filesystem — `uploads/`, `datasets/`, `adapters/`, `jobs.db` (SQLite).
- Multi-user later: object storage (S3/R2) for zips + adapters, Postgres for
  jobs/users. The abstraction keeps this swap localized.

---

## 9. Architecture (end-to-end)

```
Browser (Next.js)
   │  chunked upload (.zip)
   ▼
FastAPI backend ──► store raw zip
   │
   ├─► prepare_dataset.py  (parse tweets.js, filter RTs, clean, format) ──► dataset.jsonl
   │
   ├─► ComputeBackend.submit_training(dataset, profile)
   │        ├── LocalBackend  → subprocess on 2060 (Unsloth QLoRA)
   │        └── DaytonaBackend → Daytona GPU workspace (QLoRA, 7B)
   │                                   │
   │                                   ▼
   │                              LoRA adapter (per user)
   │
   └─► Serve: llama.cpp/Ollama (local)  |  vLLM + adapter swap (scale)
            ▲
            │  /generate (prompt, controls)
        Browser
```

---

## 10. Milestones

**M0 — Data pipeline (no UI).** `prepare_dataset.py` turns the real archive into
`dataset.jsonl`; verify counts match §4. *Proves the data is usable.*

**M1 — Local finetune CLI.** Unsloth QLoRA on a 2060, 3B base, on the M0 dataset.
Produce an adapter; generate sample tweets from the command line and eyeball
voice quality. *Proves the core idea works.*

**M2 — Daytona path.** Same training script on Daytona GPU, 7B base; compare
output quality vs local 3B. *Proves the hybrid compute abstraction.*

**M3 — Web app.** Upload → dataset review → train (pick backend) → generate, all
in the browser, end-to-end on the author's archive. *MVP complete.*

**M4 (later) — Multi-user readiness.** Auth, object storage, per-user adapters,
vLLM adapter-swap serving, queue hardening.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Dataset too small (~25K short tweets) → overfit / bland output | Include replies; light dedupe; 1–3 epochs w/ early stop; eval qualitatively; consider topic-conditioning to add signal. |
| 6 GB VRAM too tight | 3B QLoRA via Unsloth fits; fall back to Daytona for 7B. |
| 10 GB upload fragility | Resumable chunked upload; only read needed `data/*.js`, ignore media. |
| Generated tweets parrot training tweets verbatim | Dedupe, raise temperature, near-duplicate filter on output, penalize exact matches. |
| Voice ≠ facts (model invents claims) | Set expectations in UI: style mimicry, not a knowledge engine. |
| Daytona cost/credits | Local-first default; Daytona only for bigger models or multi-user. |
| PII in archive (DMs, emails) | Train only on tweets; never ingest `direct-messages.js` / account PII into training. Document data handling. |

---

## 12. Open Questions
- Topic-conditioning in MVP, or completion-style only first? (Recommend: ship
  completion-style in M1, add topic-conditioning if voice quality needs steering.)
- Which 3B base — Qwen2.5 vs Llama-3.2? (Benchmark both on the real data in M1.)
- Serving stack for MVP demo — Ollama (simplest) vs vLLM (multi-user-ready)?
- Keep media (`assets/`) for anything (e.g., image-tweet context), or ignore
  entirely for MVP? (Recommend: ignore for MVP.)

---

## 13. Tech Stack Summary
- **Frontend:** Next.js, React, Tailwind
- **Backend:** FastAPI (Python), SQLite (MVP) → Postgres
- **Data:** Python stdlib (`zipfile`, `json`), pandas optional
- **Finetuning:** Unsloth + PEFT/transformers, QLoRA (4-bit)
- **Base models:** Qwen2.5-3B / Llama-3.2-3B (local), Qwen2.5-7B / Llama-3.1-8B (Daytona)
- **Serving:** llama.cpp / Ollama (local), vLLM (scale)
- **Compute:** Local 2x RTX 2060 (6 GB) + Daytona GPU workspaces
- **Queue/storage:** RQ + Redis (optional), filesystem → S3/R2
```
