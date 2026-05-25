"""twitterGPT backend API (FastAPI).

Flow: upload archive -> prepare dataset -> start training (local|daytona) ->
poll job -> generate tweets from the trained adapter.

Run:  uvicorn app.main:app --reload --port 8000   (from backend/)
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import db, pipeline
from .compute import get_backend
from .config import (ADAPTERS_DIR, DATASETS_DIR, PROFILES, UPLOADS_DIR, ensure_dirs)
from .schemas import GenerateRequest, PrepareRequest, TrainStartRequest

app = FastAPI(title="twitterGPT API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000"],
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    ensure_dirs()
    db.init()


@app.get("/health")
def health():
    return {"ok": True, "profiles": list(PROFILES)}


# --- 1. upload -------------------------------------------------------------
# MVP: single multipart upload. Swap for chunked/resumable (tus) before exposing
# 10 GB uploads to real users — see PRD §8.
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "expected a .zip Twitter archive")
    ensure_dirs()
    upload_id = uuid.uuid4().hex[:12]
    dest = UPLOADS_DIR / f"{upload_id}.zip"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"upload_id": upload_id, "size_bytes": dest.stat().st_size}


# --- 2. prepare dataset ----------------------------------------------------
@app.post("/dataset/prepare")
def prepare_dataset(req: PrepareRequest):
    archive = UPLOADS_DIR / f"{req.upload_id}.zip"
    if not archive.exists():
        raise HTTPException(404, "upload not found")
    dataset_id = req.upload_id  # 1:1 for MVP
    try:
        meta = pipeline.prepare(
            archive, dataset_id,
            include_replies=req.include_replies, strip_mentions=req.strip_mentions,
            strip_urls=req.strip_urls, min_chars=req.min_chars,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return meta


@app.get("/dataset/{dataset_id}")
def get_dataset(dataset_id: str):
    meta = pipeline.get_meta(dataset_id)
    if not meta:
        raise HTTPException(404, "dataset not found")
    return meta


# --- 3. train --------------------------------------------------------------
@app.post("/train")
def start_train(req: TrainStartRequest):
    if req.profile not in PROFILES:
        raise HTTPException(400, f"unknown profile; choices: {list(PROFILES)}")
    profile = PROFILES[req.profile]
    if not (DATASETS_DIR / req.dataset_id / "train.jsonl").exists():
        raise HTTPException(404, "dataset not prepared")

    backend_name = profile["backend"]
    job_id = db.create_job("train", backend_name, req.profile, req.dataset_id)
    adapter_out = ADAPTERS_DIR / req.dataset_id

    from .compute.base import TrainRequest
    try:
        backend = get_backend(backend_name)
        backend.submit_training(TrainRequest(
            job_id=job_id, dataset_dir=DATASETS_DIR / req.dataset_id,
            adapter_out=adapter_out, profile=profile,
        ))
    except (NotImplementedError, FileNotFoundError) as e:
        db.update_job(job_id, status="failed", error=str(e))
        raise HTTPException(501, str(e))
    return {"job_id": job_id, "backend": backend_name, "profile": req.profile}


# --- 4. job status ---------------------------------------------------------
@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    st = get_backend(job["backend"]).status(job_id)
    return {
        "job_id": job_id, "kind": job["kind"], "backend": job["backend"],
        "profile": job["profile"], "status": st.status,
        "adapter_dir": job.get("adapter_dir"), "error": st.error,
        "log_tail": st.log_tail,
    }


@app.get("/jobs")
def jobs():
    return db.list_jobs()


# --- 5. generate -----------------------------------------------------------
# Inference is delegated to the same GPU env as training. For the MVP we shell
# out to generate.py; at scale this becomes a vLLM call with adapter hot-swap.
@app.post("/generate")
def generate(req: GenerateRequest):
    adapter = Path(req.adapter_dir)
    if not (adapter / "adapter_config.json").exists():
        raise HTTPException(404, "adapter not found / not finished training")

    import subprocess
    from .config import SCRIPTS_DIR
    cmd = ["python3", str(SCRIPTS_DIR / "generate.py"), "--adapter", str(adapter),
           "-n", str(req.n), "--temp", str(req.temperature), "--top-p", str(req.top_p),
           "--max-new", str(req.max_new_tokens)]
    if req.topic:
        cmd += ["--topic", req.topic]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(SCRIPTS_DIR))
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "generation timed out")
    if out.returncode != 0:
        raise HTTPException(500, f"generation failed: {out.stderr[-1000:]}")
    # parse the numbered lines generate.py prints
    tweets = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:4]:
            tweets.append(line.split(".", 1)[1].strip())
    return {"tweets": tweets, "raw": out.stdout}
