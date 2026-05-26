#!/usr/bin/env python3
"""
train_v2.py — Second generation twitterGPT (ETH zurich route).

Uses Qwen3-1.5B for better quality while still fitting on 6 GB.
Identity: train fast, iterate ruthlessly, build with your own hands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="dataset_out")
    ap.add_argument("--out", default="adapters/geoppls_v2")
    ap.add_argument("--model", default="unsloth/Qwen3-1.5B-unsloth-bnb-4bit",
                    help="1.5B fits on 6 GB at 4-bit")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="effective batch = batch * accum = 16")
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

    system_prompt = """You are geo ppls (@geoppls). You write tweets about AI research and what it's like building AI tools in the real world.

Your philosophy is the ETH zurich route: if you need a cluster to train a model, you're doing it wrong. Train fast, small models, single GPU, under a minute. Iterate. Ship. Learn.

You care about practical engineering over hype. You build with your own hands. The more models you train, the more you learn.

Write in your own authentic voice — casual, direct, occasionally irreverent. Mix short punchy statements with longer thoughtful takes. Match your real tone, style, and topics."""

    print(f"System prompt:\n  {system_prompt}\n")

    # Load 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )

    # Attach LoRA
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

    # Load dataset
    ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")

    def to_text(row):
        # If the tweet was a reply, mention who they were replying to
        context = ""
        if row.get("is_reply"):
            context = f"\nContext: you're replying to someone."
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a tweet.{context}"},
            {"role": "assistant", "content": row["completion"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    print(f"Training examples: {len(ds):,}")

    # Save config + prompt
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

    # Save
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nDone. V2 saved to: {out}")
    print(f"Generate with:\n  python generate_v2.py --adapter {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
