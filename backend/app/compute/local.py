"""LocalBackend — runs train_qlora.py as a subprocess on this machine's GPU.
Default for dev + single-user. Pin to one card via CUDA_VISIBLE_DEVICES so the
other 2060 stays free for inference."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from .. import db
from ..config import SCRIPTS_DIR
from .base import ComputeBackend, JobStatus, TrainRequest


class LocalBackend(ComputeBackend):
    name = "local"

    def submit_training(self, req: TrainRequest) -> None:
        script = SCRIPTS_DIR / "train_qlora.py"
        if not script.exists():
            raise FileNotFoundError(f"train_qlora.py not found at {script}")

        log_path = req.adapter_out.parent / f"{req.job_id}.log"
        req.adapter_out.parent.mkdir(parents=True, exist_ok=True)

        p = req.profile
        cmd = [
            "python3", str(script),
            "--data", str(req.dataset_dir),
            "--out", str(req.adapter_out),
            "--model", p["model"],
            "--max-seq-len", str(p["max_seq_len"]),
            "--epochs", str(p["epochs"]),
            "--batch-size", str(p["batch_size"]),
            "--grad-accum", str(p["grad_accum"]),
        ]
        env = dict(os.environ)
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one 2060; override to use the other

        log = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=str(SCRIPTS_DIR))
        db.update_job(req.job_id, status="running", pid=proc.pid,
                      log_path=str(log_path), adapter_dir=str(req.adapter_out))

    def status(self, job_id: str) -> JobStatus:
        job = db.get_job(job_id)
        if not job:
            return JobStatus(status="failed", error="job not found")
        log_tail = self._tail(job.get("log_path"))

        pid = job.get("pid")
        if job["status"] == "running" and pid:
            if not self._alive(pid):
                # process exited — decide done vs failed by adapter presence
                adapter = Path(job["adapter_dir"]) if job.get("adapter_dir") else None
                if adapter and (adapter / "adapter_config.json").exists():
                    db.update_job(job_id, status="done")
                    return JobStatus(status="done", log_tail=log_tail)
                db.update_job(job_id, status="failed", error="process exited without adapter")
                return JobStatus(status="failed", log_tail=log_tail, error="process exited without adapter")
        return JobStatus(status=job["status"], log_tail=log_tail, error=job.get("error"))

    def fetch_artifact(self, job_id: str) -> Path:
        job = db.get_job(job_id)
        if not job or not job.get("adapter_dir"):
            raise FileNotFoundError("no adapter for job")
        return Path(job["adapter_dir"])

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def _tail(log_path: str | None, n: int = 4000) -> str:
        if not log_path or not Path(log_path).exists():
            return ""
        data = Path(log_path).read_text(errors="replace")
        return data[-n:]
