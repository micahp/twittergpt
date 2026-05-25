# twitterGPT (M3 web app skeleton)

Upload a Twitter archive → review the parsed dataset → finetune a model
(local GPU or Daytona) → generate tweets in your voice.

```
twittergpt/
  backend/          FastAPI — upload, dataset prep, train, jobs, generate
    app/
      main.py           routes
      config.py         paths + compute PROFILES
      db.py             SQLite job store
      pipeline.py       reuses ../../prepare_dataset.py (single source of truth)
      compute/          ComputeBackend abstraction
        base.py           interface (TrainRequest/JobStatus)
        local.py          runs ../../train_qlora.py as a subprocess
        daytona.py        stub (wired in M2)
  frontend/         Next.js (app router) — 4-page flow
    app/{page,review,train,generate}/
    lib/api.ts

# scripts shared with M0/M1 live one level up (the Downloads dir):
../prepare_dataset.py   ../train_qlora.py   ../generate.py
```

## How the pieces connect
The web app never knows *where* training runs. It builds a `TrainRequest` and
hands it to a `ComputeBackend`; `LocalBackend` shells out to `train_qlora.py` on
this machine's GPU, `DaytonaBackend` (M2) will do the same inside a cloud
workspace. Switching is one dropdown — the compute **profile** (`local-3b` vs
`daytona-7b`) carries the base model + sizing.

The adapter produced per dataset is the per-user artifact that makes multi-user
cheap later (one base model, many adapters).

## Run it (dev)

Backend (needs the M0/M1 scripts at `..`; only training/generation need a GPU):
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# point at where the scripts live if not the parent of this package:
export TGPT_SCRIPTS_DIR=/abs/path/to/Downloads
uvicorn app.main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm run dev      # http://localhost:3000
```

## What works without a GPU
`/upload` and `/dataset/prepare` run anywhere (pure Python parsing). `/train`
and `/generate` require an NVIDIA GPU + the training deps
(`../requirements-train.txt`) because they invoke `train_qlora.py` /
`generate.py`. On a CPU-only box, training jobs will fail fast — run the backend
on the desktop with the 2060s, or wire up `DaytonaBackend` (M2).

## Not built yet (deferred per PRD)
- Resumable/chunked upload for multi-GB files (MVP = single POST).
- Auth, multi-tenant storage, vLLM adapter-swap serving (M4).
- DaytonaBackend implementation (M2).
- Tailwind (skeleton uses plain CSS in `app/globals.css`).
