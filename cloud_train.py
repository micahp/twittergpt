"""Cloud training script — Ministral-3-8B QLoRA on RTX 4090, no system prompt."""
import os, json, time, argparse
from pathlib import Path

def train(model_path: str, data_path: str, out_dir: str, 
          samples: int, epochs: float, lr: float, 
          max_seq_length: int, batch_size: int, grad_accum: int):
    
    import warnings
    warnings.filterwarnings("ignore")
    
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch
    
    print(f"Loading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Load data — no system prompt, just raw tweet text
    print(f"Loading data: {data_path}")
    ds = load_dataset("json", data_files=data_path, split="train")
    
    def to_text(row):
        # NO system prompt — identity lives in the data
        text = row["completion"].replace("\\n", "\n").strip()
        return {"text": text}
    
    ds = ds.map(to_text)
    
    if samples and samples < len(ds):
        import random
        random.seed(42)
        indices = random.sample(range(len(ds)), samples)
        ds = ds.select(indices)
    
    print(f"Training {len(ds)} examples, {epochs} epochs, lr={lr}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=False,
            bf16=True,     # Required for RTX 4090 (no fp16 support)
            logging_steps=5,
            optim="adamw_8bit",
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
    (out_path / "system_prompt.txt").write_text("(none — no system prompt)")
    
    # Metrics
    log_history = getattr(result, 'log_history', [])
    if not log_history:
        ts = getattr(result, 'state', None)
        if ts: log_history = ts.log_history
    losses = [e["loss"] for e in log_history if isinstance(e, dict) and "loss" in e]
    train_loss = losses[-1] if losses else 0
    
    metrics = {
        "model": model_path,
        "samples": len(ds),
        "epochs": epochs,
        "lr": lr,
        "max_seq_length": max_seq_length,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_minutes": round(elapsed / 60, 1),
        "train_loss": round(train_loss, 4),
        "loss_trace": [round(l, 4) for l in losses[:10]],
        "all_losses": [round(l, 4) for l in losses],
    }
    
    print(f"\n=== DONE: {elapsed/60:.1f} min | loss: {train_loss:.4f} ===")
    if losses:
        print(f"Loss trace: {metrics['loss_trace']}")
    print(f"Saved to: {out_path}/")
    
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Ministral-3-8B-Instruct")
    parser.add_argument("--data", default="/workspace/train_filtered.jsonl")
    parser.add_argument("--out", default="/workspace/adapters/ministral_3_8b_v2")
    parser.add_argument("--samples", type=int, default=0)  # 0 = all
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()
    
    train(
        model_path=args.model,
        data_path=args.data,
        out_dir=args.out,
        samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
