#!/usr/bin/env python3
"""
generate.py — Use the finetuned twitterGPT adapter to generate tweets.

Usage:
  python generate.py --adapter adapters/geoppls -n 10
  python generate.py --adapter adapters/geoppls --prompt "new app launch"
"""
import argparse
import torch
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate tweets using finetuned LoRA adapter")
    ap.add_argument("--adapter", default="adapters/geoppls", help="path to saved adapter")
    ap.add_argument("-n", type=int, default=5, help="number of tweets to generate")
    ap.add_argument("--prompt", default=None, help="system prompt (overrides system_prompt.txt)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=100, help="max output length")
    ap.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling")
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    # Load the 4-bit base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        max_seq_length=256,
        load_in_4bit=True,
        device_map="auto",
    )

    # Load adapters
    model.load_adapter(args.adapter)
    model.set_adapter("default")

    # Get system prompt
    prompt = args.prompt
    if prompt is None:
        sp_file = Path(args.adapter) / "system_prompt.txt"
        if sp_file.exists():
            prompt = sp_file.read_text(encoding="utf-8").strip()
    if prompt is None:
        prompt = "You are an assistant that generates tweets."

    # Generate
    for i in range(args.n):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate a tweet."},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode only the new tokens
        gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gen_text = gen_text.strip().split("\n\n")[0]  # stop at paragraph break
        print(f"\n--- Tweet {i+1} ---\n{gen_text}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
