# twitterGPT M1 — Local Finetune Setup (run on your desktop)

Goal: QLoRA-finetune a small model on your tweets on **one RTX 2060 (6 GB)** and
generate sample tweets to confirm the voice came through.

> This must run on your **desktop with the NVIDIA GPU + CUDA** — not the Linux
> box where we built the pipeline (that has no GPU). Copy these files over:
> `prepare_dataset.py`, `train_qlora.py`, `generate.py`, `requirements-train.txt`,
> and either the archive `.zip` or the already-built `dataset_out/` folder.

---

## 0. Prerequisites
- NVIDIA driver + CUDA installed (`nvidia-smi` should list your 2060s).
- Python 3.10–3.11.
- ~15 GB free disk (model cache + checkpoints).

## 1. Environment
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements-train.txt
```
If `nvidia-smi` shows two 2060s, pin training to one card and leave the other
free for inference:
```bash
export CUDA_VISIBLE_DEVICES=0       # Windows (cmd): set CUDA_VISIBLE_DEVICES=0
```

## 2. Build the dataset (skip if you copied `dataset_out/` over)
```bash
python3 prepare_dataset.py YOUR_ARCHIVE.zip -o dataset_out
```
Expect ~19.7K kept examples for the @geoppls archive.

## 3. Train
```bash
python3 train_qlora.py --data dataset_out --out adapters/geoppls
```
- Default base: `Qwen2.5-3B-Instruct` (4-bit) — fits 6 GB.
- ~2 epochs over ~18.7K short examples. On a 2060 expect roughly **1–3 hours**
  (varies with thermals/clocks). Watch the loss in the log.
- **If you hit CUDA out-of-memory:** lower `--max-seq-len 192`, or
  `--batch-size 1 --grad-accum 16`, or switch to a smaller base
  `--model unsloth/Llama-3.2-1B-Instruct-bnb-4bit`.

## 4. Generate & judge
```bash
python3 generate.py --adapter adapters/geoppls -n 12
python3 generate.py --adapter adapters/geoppls --topic "ai and texas" -n 6
```
Read the output: does it sound like you? Right tone, slang, topics? That's the
M1 pass/fail. Paste a sample back and we'll tune (epochs, LoRA rank, temperature)
or move to M2 (Daytona, 7B base) / M3 (web app).

---

## Tuning knobs (if voice is off)
| Symptom | Try |
|---|---|
| Bland / generic, not "you" | more epochs (`--epochs 3`), higher `--lora-r 32 --lora-alpha 32` |
| Repeats training tweets verbatim | fewer epochs (`--epochs 1`), raise `--temp 1.0` |
| Incoherent / rambly | lower `--temp 0.7`, lower `--top-p 0.9` |
| OOM | see step 3 fallback flags |

## Notes
- The adapter is small (tens of MB) and lives in `adapters/geoppls/`. One base
  model can host many users' adapters later — that's the multi-user serving path.
- For a bigger/better model use Daytona (M2): same command, just
  `--model unsloth/Qwen2.5-7B-Instruct-bnb-4bit` on a 16 GB+ GPU.
