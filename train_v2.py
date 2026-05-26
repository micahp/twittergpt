#!/usr/bin/env python3
"""
train_v2.py — Second iteration: ETH zurich route identity.

Same architecture as v1 (Qwen3-0.6B, batch=2, accum=8), but with the identity as the core.
The identity prompt encodes the ethos: train fast, iterate ruthlessly, build with your own hands.

Commit: this is the model that will generate tweets in geo ppls' voice.
System prompt: ETH zurich route — single GPU, under a minute, more models = more learning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="dataset_out",
                    help="dataset root with train_filtered.jsonl (originals only)")
    ap.add_argument("--out", default="adapters/geoppls_v2")
    ap.add_argument("--model", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
                    help="0.6B fits comfortably on 6 GB — same as v1 but with identity")
    ap.add_argument("--max-seq-len", type=int, default=128,
                    help="ETH zurich route: short sequences, fast iteration")
    ap.add_argument("--epochs", type=float, default=3.0,
                    help="3 epochs on the identity prompt should be enough")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="effective batch = 16, optimized for 6 GB GPU")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch

    data_dir = Path(args.data)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    # Core identity — the ETH zurich route
    system_prompt = """You are geo ppls (@geoppls). You write about AI research and building tools that matter.

The ETH zurich route: if you need a data center to train it, you're doing it wrong. Train fast. Small models. Single GPU. Under a minute. Iterate. Learn.

You believe the more models you train, the more you learn. You'd rather ship 100 quick experiments than polish one for months. Practical engineering over hype.

Write tweets that sound like you actually wrote them — casual, occasionally blunt, no corporate speak. Mix short punchy one-liners with longer thoughtful takes.

Be unapologetically yourself. Not a bro, not a guru, not a LinkedIn bot."""

    print(f"=== V2 TRAINING — ETH ZURICH ROUTE IDENTITY ===")
    print(f"System prompt:\n{system_prompt}\n")

    # Model params
    max_seq = args.max_seq_len
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq,
        load_in_4bit=True,
        dtype=None,
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

    # Load dataset and format
    ds = load_dataset("json", data_files=str(data_dir / "train_filtered.jsonl"), split="train")

    def to_text(row):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write a tweet."},
            {"role": "assistant", "content": row["completion"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    print(f"Training examples: {len(ds):,}\n")

    # Save config
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq,
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

    print("Starting training...\n")
    trainer.train()

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"V2 done. Saved to: {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
