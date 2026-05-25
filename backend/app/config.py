"""Central config + paths. Override via env vars for different machines."""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = .../twittergpt (two levels up from backend/app/config.py).
# The standalone scripts (prepare_dataset.py, train_qlora.py, generate.py) live
# at the repo root alongside backend/ and frontend/.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]          # repo root .../twittergpt
SCRIPTS_DIR = Path(os.environ.get("TGPT_SCRIPTS_DIR", PACKAGE_ROOT))  # where the .py scripts live

# Working data lives under a single root so it's easy to back up / swap to S3 later.
DATA_ROOT = Path(os.environ.get("TGPT_DATA_ROOT", PACKAGE_ROOT / "var"))
UPLOADS_DIR = DATA_ROOT / "uploads"
DATASETS_DIR = DATA_ROOT / "datasets"
ADAPTERS_DIR = DATA_ROOT / "adapters"
JOBS_DB = DATA_ROOT / "jobs.db"

# Which compute backend to use by default: "local" | "daytona".
DEFAULT_BACKEND = os.environ.get("TGPT_BACKEND", "local")

# Compute profiles bundle base model + sizing so switching backend is one choice.
PROFILES = {
    "local-3b": {
        "backend": "local",
        "model": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "max_seq_len": 256,
        "epochs": 2.0,
        "batch_size": 2,
        "grad_accum": 8,
    },
    "daytona-7b": {
        "backend": "daytona",
        "model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "max_seq_len": 256,
        "epochs": 2.0,
        "batch_size": 4,
        "grad_accum": 4,
    },
}


def ensure_dirs() -> None:
    for d in (DATA_ROOT, UPLOADS_DIR, DATASETS_DIR, ADAPTERS_DIR):
        d.mkdir(parents=True, exist_ok=True)
