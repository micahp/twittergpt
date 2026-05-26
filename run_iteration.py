#!/usr/bin/env python3
"""
run_iteration.py — ETH zurich route iteration strategy.

The approach: don't commit to full training until you've validated the
identity on progressively larger scales:

  Scale 1 (under 60s):  10 examples, 1 epoch → verify the pipeline works
  Scale 2 (2-10 min):   100 examples, 2 epochs → loss dropping? identity learned?
  Scale 3 (30-60 min):  500 examples, 3 epochs → quality test

Once all three pass, train the full 4,554 originals with 3 epochs.

Each iteration writes a JSON log: adapters/{name}/iteration_log.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ADAPTERS_DIR = SCRIPT_DIR / "adapters"
LOG_FILE_FORMAT = "adapters/{name}/iteration_log.json"

def run_training(name: str, data: str, epochs: int, batch_size: int, 
                 grad_accum: int, max_seq_len: int = 128, 
                 samples: int | None = None) -> dict:
    """Run training and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  data: {data}, epochs: {epochs}, batch: {batch_size}x{grad_accum}, seq: {max_seq_len}")
    if samples:
        print(f"  sampling {samples} examples")
        # Sample examples for fast iteration
        data_path = Path(f"dataset_out/{data}")
        if data_path.exists():
            lines = data_path.read_text().splitlines()
            import random
            random.seed(42)
            sampled = [json.loads(l) for l in random.sample(lines, min(samples, len(lines)))]
            out_path = Path(f"dataset_out/.temp_sample.jsonl")
            with open(out_path, "w") as f:
                for ex in sampled:
                    f.write(json.dumps(ex) + "\n")
            print(f"  → {len(sampled)} examples sampled")
        else:
            print(f"  → using all {data}")
    
    cmd = [
        sys.executable, "train_v2.py",
        "--data", f"dataset_out",
        "--out", f"adapters/{name}",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--max-seq-len", str(max_seq_len),
    ]
    
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=SCRIPT_DIR)
    elapsed = time.time() - start
    
    print(f"\n  Done: {elapsed:.1f}s")
    if proc.returncode != 0:
        print(f"  ERROR: {proc.stderr[:500]}")
        return None
    
    # Parse training logs
    log_lines = proc.stdout.splitlines()
    final_metrics = None
    for i, line in enumerate(log_lines):
        if "train_loss" in line or "loss':" in line or "train_runtime" in line:
            if "train_runtime" in line:
                # Extract runtime
                try:
                    runtime = float(line.split("'train_runtime': '")[1].split("'")[0])
                    final_metrics = {"train_runtime": runtime, "elapsed": elapsed}
                except:
                    pass
    
    # Also get final step loss from logs
    losses = []
    for line in log_lines:
        if "'loss':" in line and "train" not in line:
            try:
                loss = float(line.split("'loss': '")[1].split("'")[0])
                losses.append(loss)
            except:
                pass
    
    final_metrics = final_metrics or {}
    final_metrics["train_loss"] = losses[-1] if losses else "unknown"
    final_metrics["elapsed_seconds"] = elapsed
    final_metrics["exit_code"] = proc.returncode
    
    return final_metrics


def main():
    iterations = []
    
    # Scale 1: 10 examples, 1 epoch, batch=4 → under 60s
    print("\n--- SCALE 1: 10 examples (under 60s) ---")
    r1 = run_training("v2_scale1", "train_filtered.jsonl", epochs=1, batch_size=4, grad_accum=4, samples=10)
    if r1:
        iterations.append({"scale": 1, "samples": 10, "epochs": 1, **r1})
    
    # Scale 2: 100 examples, 2 epochs, batch=4 → ~5 min
    print("\n--- SCALE 2: 100 examples (~5 min) ---")
    r2 = run_training("v2_scale2", "train_filtered.jsonl", epochs=2, batch_size=4, grad_accum=4, samples=100)
    if r2:
        iterations.append({"scale": 2, "samples": 100, "epochs": 2, **r2})
    
    # Scale 3: 500 examples, 3 epochs, batch=2 → ~30 min
    print("\n--- SCALE 3: 500 examples (~30 min) ---")
    r3 = run_training("v2_scale3", "train_filtered.jsonl", epochs=3, batch_size=2, grad_accum=8, samples=500)
    if r3:
        iterations.append({"scale": 3, "samples": 500, "epochs": 3, **r3})
    
    # Save iteration log
    log_entry = {
        "identity": "ETH zurich route",
        "dataset_size": 4554,
        "iterations": iterations,
        "note": "Run with: python run_iteration.py"
    }
    
    # Save to each adapter's directory
    for name in ["v2_scale1", "v2_scale2", "v2_scale3", "geoppls_v2"]:
        out = Path(ADAPTERS_DIR) / name
        if out.exists():
            (out / "iteration_log.json").write_text(json.dumps(log_entry, indent=2))
    
    print(f"\n{'='*60}")
    print(f"Iteration log saved. Summary:")
    for it in iterations:
        print(f"  Scale {it.get('scale')} ({it.get('samples')} ex): loss={it.get('train_loss')} in {it.get('elapsed_seconds', 0):.0f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
