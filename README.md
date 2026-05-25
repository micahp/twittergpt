# twitterGPT

Upload a Twitter/X data archive, finetune a small LLM on **your own tweets**, and
generate new tweets in your voice.

This repo holds the ML pipeline (data prep + QLoRA finetuning + generation) and a
web app skeleton (FastAPI backend + Next.js frontend) wired through a compute
abstraction that runs training either on a **local GPU** or on **Daytona**.

> Status: MVP in progress. M0 (data pipeline) and the web-app skeleton are done
> and verified; M1 (local finetune) runs on a desktop GPU. See `twitterGPT-PRD.md`
> for the full plan and milestones.

## Layout
```
prepare_dataset.py      M0 — archive .zip -> clean train/eval JSONL (+ meta.json)
train_qlora.py          M1 — QLoRA finetune a small instruct model (Unsloth)
generate.py             M1 — sample tweets from a trained adapter
requirements-train.txt  GPU/training deps (NVIDIA + CUDA)
M1-SETUP.md             step-by-step local finetune guide for your desktop
twitterGPT-PRD.md       product requirements + milestones
WEBAPP.md               web app architecture + run instructions

backend/                FastAPI app (upload, dataset prep, train, jobs, generate)
  app/compute/          ComputeBackend abstraction: local (subprocess) | daytona (stub)
frontend/               Next.js app router — upload -> review -> train -> generate
```

## Quick start

**1. Build a dataset from your archive** (pure Python, no GPU):
```bash
python3 prepare_dataset.py YOUR_ARCHIVE.zip -o dataset_out
```

**2. Finetune** (needs an NVIDIA GPU — see `M1-SETUP.md`):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
python3 train_qlora.py --data dataset_out --out adapters/me
python3 generate.py --adapter adapters/me -n 12
```

**3. Run the web app** (see `WEBAPP.md` for details):
```bash
cd backend && pip install -r requirements.txt && uvicorn app.main:app --port 8000
cd frontend && npm install && npm run dev   # http://localhost:3000
```

## Notes
- Your archive `.zip` and the parsed `dataset_out/` (your tweets) are **gitignored** —
  this repo contains code only.
- twitterGPT mimics writing **style**, not facts. ~20K short tweets is enough to
  capture voice via LoRA, not to teach knowledge.
