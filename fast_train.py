#!/usr/bin/env python3
"""
fast_train.py — Train with ETH zurich route identity on a small sample.
Usage: python fast_train.py --samples 10 --epochs 1 --out adapters/v2_test
"""
import json
import random
import argparse
import time
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--out", default="adapters/v2_test")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    
    data_dir = Path("dataset_out")
    source = data_dir / "train_filtered.jsonl"
    with open(source) as f:
        all_examples = [json.loads(line) for line in f]
    
    random.seed(42)
    sampled = all_examples[:min(args.samples, len(all_examples))]
    
    out_path = data_dir / ".training_sample.jsonl"
    with open(out_path, "w") as f:
        for ex in sampled:
            f.write(json.dumps(ex) + "\n")
    
    system_prompt = """You are geo ppls (@geoppls). You write about AI research and building tools that matter.

The ETH zurich route: if you need a data center to train it, you're doing it wrong. Train fast. Small models. Single GPU. Under a minute. Iterate. Learn.

You believe the more models you train, the more you learn. You'd rather ship 100 quick experiments than polish one for months. Practical engineering over hype.

Write tweets that sound like you actually wrote them — casual, occasionally blunt, no corporate speak. Mix short punchy one-liners with longer thoughtful takes.

Be unapologetically yourself. Not a bro, not a guru, not a LinkedIn bot."""
    
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        max_seq_length=128,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    
    ds = load_dataset("json", data_files=str(out_path), split="train")
    def to_text(row):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write a tweet."},
            {"role": "assistant", "content": row["completion"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}
    
    ds = ds.map(to_text, remove_columns=ds.column_names)
    print(f"Training {len(ds)} examples, {args.epochs} epochs")
    
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=128,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=str(out / "checkpoints"),
            save_strategy="epoch",
            report_to="none",
        ),
    )
    
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    
    losses = [l for l in result.log_history if "loss" in l]
    train_loss = result.log_history[-1].get("training_loss", result.log_history[-1].get("loss", "N/A"))
    
    print(f"\nDone: {elapsed:.1f}s | loss: {train_loss:.4f}")
    print(f"Saved to: {out}/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
