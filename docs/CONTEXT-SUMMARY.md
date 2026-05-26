# twitterGPT — Context Summary

_Last updated: 2026-05-25_

A running summary of what's been built, the decisions behind it, what's verified,
and where to pick up. Read this first when resuming work.

---

## What twitterGPT is
A web app that lets someone upload their Twitter/X data archive and get a small
LLM finetuned on **their own tweets**, which then generates new tweets in their
voice. Long-term: multi-user product. Now: MVP, proven on the author's
(`@geoppls`) archive before any user-acquisition work.

Full spec + milestones: [`../twitterGPT-PRD.md`](../twitterGPT-PRD.md).

---

## Key decisions (locked)
| Decision | Choice | Why |
|---|---|---|
| Scope | MVP-first; author is user #1 | Prove the loop before multi-tenancy |
| Compute | Hybrid: local GPU + Daytona, behind one abstraction | Free/local for dev, cloud for bigger models & scale |
| Training data | Original tweets **+ replies**, retweets stripped | ~20K authentic-voice examples vs ~5K originals only |
| Method | QLoRA (4-bit) via Unsloth, per-user LoRA adapter | Fits a 6 GB 2060; one base model + many adapters = cheap multi-user later |
| Base model | Qwen3-4B (local) / Qwen2.5-7B (Daytona) | 4B QLoRA fits 6 GB; 7B+ for quality on cloud |
| Serving | Ollama/llama.cpp (local) → vLLM adapter-swap (scale) | Simple now, multi-user-ready later |

Author's hardware: desktop with **2x RTX 2060 (6 GB each, no NVLink)** + 32 GB RAM.
Treated as two independent cards (one trains, one can serve). Daytona credits available.

---

## The real archive (analyzed, not assumed)
Parsed the author's actual 9.85 GB archive (`data/tweets.js`, 65 MB):

| Metric | Value |
|---|---|
| Total tweets | 33,143 |
| Retweets (excluded) | 8,036 |
| Replies (included) | 19,747 |
| Original standalone | 5,360 |
| **Kept after cleaning** | **19,744** (4,554 orig + 14,203 replies) |
| Approx training tokens | ~340K |

**Honest takeaway:** enough to capture **style/voice** via LoRA, not enough to
teach facts. Set user expectations accordingly.

---

## Milestone status
- **M0 — Data pipeline** ✅ Done & verified. `prepare_dataset.py` turns any
  standard archive into clean `train/eval.jsonl` + `meta.json`. Reproduces the
  19,744 count.
- **M1 — Local finetune** ✅ Code written, ⏳ not yet run (needs the GPU desktop).
  `train_qlora.py` (Unsloth QLoRA) + `generate.py`. See
  [`../M1-SETUP.md`](../M1-SETUP.md).
- **M3 — Web app skeleton** ✅ Done; non-GPU paths verified. FastAPI backend
  (`backend/`) + Next.js frontend (`frontend/`) with the `ComputeBackend`
  abstraction. See [`../WEBAPP.md`](../WEBAPP.md).
- **M2 — Daytona backend** ⏳ Stub in place (`backend/app/compute/daytona.py`),
  not implemented.
- **M4 — Multi-user** ⏳ Deferred (auth, object storage, vLLM serving, chunked upload).

---

## What's verified vs pending
**Verified (on a CPU-only Linux box):**
- Dataset prep end-to-end against the real archive (standalone script + backend wrapper agree: 19,744 kept).
- Cleaned output is authentic voice (mentions/URLs stripped, emoji/line breaks kept).
- Job DB lifecycle, compute-backend registry, Daytona stub behavior.
- All Python files syntax-checked.

**Pending (needs NVIDIA GPU — run on the desktop):**
- The actual M1 finetune + `generate.py` sample (the quality gate that decides
  3B-local vs Daytona-7B).
- Live FastAPI server boot (this box is Python 3.8 with no pip; backend targets
  the desktop or a proper venv).

---

## Repo map
```
prepare_dataset.py      M0 — archive .zip -> train/eval JSONL + meta.json
train_qlora.py          M1 — QLoRA finetune (Unsloth)
generate.py             M1 — sample tweets from an adapter
requirements-train.txt  GPU/training deps
M1-SETUP.md             desktop finetune guide
twitterGPT-PRD.md       full PRD + milestones
WEBAPP.md               web app architecture + run steps
README.md               repo overview + quick start
docs/CONTEXT-SUMMARY.md this file

backend/app/
  main.py               FastAPI routes: upload, dataset/prepare, train, jobs, generate
  config.py             paths + compute PROFILES (local-3b, daytona-7b)
  db.py                 SQLite job store
  pipeline.py           reuses prepare_dataset.py (single source of truth)
  compute/base.py       ComputeBackend interface (TrainRequest/JobStatus)
  compute/local.py      runs train_qlora.py as a subprocess
  compute/daytona.py    stub (M2)
frontend/app/           Next.js: page(upload) -> review -> train -> generate
frontend/lib/api.ts     API client
```
Data is gitignored: `*.zip`, `dataset_out/`, `**/var/`, `adapters/`.

---

## How to resume
1. **Run M1 on the desktop** (highest value): follow `M1-SETUP.md`, train, run
   `generate.py -n 12`, judge the voice. This decides the model path.
2. If voice is good → build out **M3** (run backend on the desktop GPU, wire the
   live flow) or implement **M2** (Daytona 7B).
3. If voice is off → tune via the table in `M1-SETUP.md` (epochs, LoRA rank, temp).

GitHub: https://github.com/micahp/twittergpt
