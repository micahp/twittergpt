import os
os.environ["UNSLOTH_FUSED_CROSS_ENTROPY_LOSS"] = "0"

import json
import random
import argparse
import time
from pathlib import Path

def run_training(samples: int, epochs: float, out: str, data_file: str, lr: float = 2e-4) -> dict:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch
    import unsloth_zoo.fused_losses.cross_entropy_loss as ce_loss
    ce_loss._UNSLOTH_FUSED_CROSS_ENTROPY_LOSS = False
    
    system_prompt = "You are geo ppls (@geoppls). You write about AI research and building tools that matter.\n\nThe ETH zurich route: if you need a data center to train it, you're doing it wrong. Train fast. Small models. Single GPU. Under a minute. Iterate. Learn.\n\nYou believe the more models you train, the more you learn. You'd rather ship 100 quick experiments than polish one for months. Practical engineering over hype.\n\nWrite tweets that sound like you actually wrote them — casual, occasionally blunt, no corporate speak. Mix short punchy one-liners with longer thoughtful takes.\n\nBe unapologetically yourself. Not a bro, not a guru, not a LinkedIn bot."
    
    source = Path("dataset_out") / data_file
    all_examples = [json.loads(line) for line in source.read_text().splitlines()]
    random.seed(42)
    sampled = all_examples[:min(samples, len(all_examples))]
    
    tmp_path = Path("dataset_out/.tmp_sample.jsonl")
    with open(tmp_path, "w") as f:
        for ex in sampled:
            f.write(json.dumps(ex) + "\n")
    
    ds = load_dataset("json", data_files=str(tmp_path), split="train")
    def to_text(row):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write a tweet."},
            {"role": "assistant", "content": row["completion"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}
    
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
    print(f"Training {len(ds)} examples, {epochs} epochs, lr={lr}")
    
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=128,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=str(out_path / "checkpoints"),
            save_strategy="epoch",
            report_to="none",
        ),
    )
    
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    (out_path / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    
    # Extract metrics — handle TRL version differences safely
    log_history = getattr(result, 'log_history', [])
    if not log_history:
        # Try the trainer state
        trainer_state = getattr(result, 'state', None)
        if trainer_state:
            log_history = trainer_state.log_history
    losses = [e["loss"] for e in log_history if isinstance(e, dict) and "loss" in e]
    train_loss = losses[-1] if losses else 0
    
    metrics = {
        "elapsed_seconds": round(elapsed, 2),
        "samples": samples,
        "epochs": epochs,
        "train_loss": round(train_loss, 4),
        "loss_trace": losses[:5],  # First 5 losses for trend
        "exit_code": 0,
    }
    
    print(f"\nDone: {elapsed:.1f}s | train_loss: {train_loss:.4f}")
    if losses:
        print(f"Loss trace: {losses[:5]}")
    print(f"Saved to: {out_path}/")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["train_10.jsonl", "train_100.jsonl", "train_500.jsonl"], 
                    default="train_10.jsonl")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--out", default="adapters/v2_scale1")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    
    _default_samples = {
        "train_10.jsonl": 10,
        "train_100.jsonl": 100,
        "train_500.jsonl": 500,
    }
    samples = args.samples or _default_samples.get(args.data, 10)
    
    metrics = run_training(
        samples=samples,
        epochs=args.epochs,
        out=args.out,
        data_file=args.data,
        lr=args.lr,
    )
    
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
