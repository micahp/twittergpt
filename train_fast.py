#!/usr/bin/env python3
"""
train_fast.py — ETH zurich route: train in under a minute.

10 examples, 1 epoch, batch_size=4. Verify the identity prompt actually works
before committing to 18K examples.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SYSTEM_PROMPT = open("C:/Users/micah/OneDrive/Desktop/Workspace/twittergpt/adapters/geoppls_v2/system_prompt.txt").read().strip()

def main() -> int:
    data_dir = Path("dataset_out")
    
    # Sample 10 examples for the fast test
    with open(data_dir / "train_filtered.jsonl") as f:
        all_examples = [json.loads(line) for line in f]
    
    random.seed(42)
    fast_examples = random.sample(all_examples, 10)
    
    with open("dataset_out/train_fast.jsonl", "w") as f:
        for ex in fast_examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Fast training: {len(fast_examples)} examples")
    print(f"System prompt:\n{SYSTEM_PROMPT}\n")
    
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch

    ds = load_dataset("json", data_files=str("dataset_out/train_fast.jsonl"), split="train")

    def to_text(row):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Write a tweet."},
            {"role": "assistant", "content": row["completion"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

    # Model
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

    ds = ds.map(to_text, remove_columns=ds.column_names)
    print(f"Training examples: {len(ds)}")

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=128,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,  # effective batch = 16
            warmup_ratio=0.05,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir="adapters/geoppls_v2_fast",
            save_strategy="epoch",
            report_to="none",
        ),
    )

    print("Starting fast training...")
    import time
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    
    # Save
    out = Path("adapters/geoppls_v2_fast")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Train loss: {result.training_loss:.4f}")
    print(f"Model saved to: {out}/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
