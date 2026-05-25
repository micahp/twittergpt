#!/usr/bin/env python3
"""
train_qlora.py — QLoRA finetune a small instruct model on your tweets (twitterGPT M1).

Designed to fit a single 6 GB GPU (e.g. one RTX 2060) using Unsloth + 4-bit QLoRA.
Consumes the output of prepare_dataset.py (dataset_out/train.jsonl + meta.json)
and writes a LoRA adapter you can load for generation (see generate.py).

Run on a machine with an NVIDIA GPU + CUDA. Example:
    python3 train_qlora.py --data dataset_out --out adapters/geoppls

Pick a bigger base on a bigger GPU (Daytona) with --model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_system_prompt(meta: dict) -> str:
    acct = meta.get("account", {}) or {}
    name = acct.get("display_name") or acct.get("username") or "this person"
    handle = acct.get("username")
    bio = acct.get("bio")
    parts = [f"You are {name}"]
    if handle:
        parts[0] += f" (@{handle})"
    parts[0] += ", writing tweets in your own authentic voice."
    if bio:
        parts.append(f"Your bio: {bio}")
    parts.append("Match your real tone, style, humor, and topics. Write only the tweet text.")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_out", help="dir with train.jsonl + meta.json")
    ap.add_argument("--out", default="adapters/model", help="output dir for the LoRA adapter")
    ap.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                    help="base model (4bit). Bigger GPU? try unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    ap.add_argument("--max-seq-len", type=int, default=256, help="tweets are short; 256 is ample")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2, help="per-device; keep low for 6GB")
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch-size * grad-accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Import here so --help works without the heavy deps installed.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch

    data_dir = Path(args.data)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    system_prompt = build_system_prompt(meta)
    print(f"System prompt:\n  {system_prompt}\n")

    # --- load 4-bit base + attach LoRA ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,  # auto
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # --- build chat-formatted dataset ---
    ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")

    def to_text(row):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    print(f"Training examples: {len(ds):,}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=str(out / "checkpoints"),
            save_strategy="epoch",
            report_to="none",
        ),
    )

    print("\nStarting training...\n")
    trainer.train()

    # save adapter + tokenizer + the system prompt used (generate.py reads it)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"\nDone. Adapter saved to: {out}")
    print(f"Generate with:\n  python3 generate.py --adapter {out} -n 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
