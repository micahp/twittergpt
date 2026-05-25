#!/usr/bin/env python3
"""
generate.py — generate tweets from a twitterGPT LoRA adapter (M1 inference).

Loads the 4-bit base + your trained adapter and samples tweets. Use it to
eyeball whether the finetune captured your voice.

    python3 generate.py --adapter adapters/geoppls -n 10
    python3 generate.py --adapter adapters/geoppls --topic "ai and texas" -n 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="adapter dir from train_qlora.py")
    ap.add_argument("--topic", default=None, help="optional topic to steer the tweet")
    ap.add_argument("-n", "--num", type=int, default=10, help="how many tweets to generate")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=80, help="max new tokens (tweets are short)")
    ap.add_argument("--max-seq-len", type=int, default=256)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    adapter = Path(args.adapter)
    sys_prompt = (adapter / "system_prompt.txt").read_text(encoding="utf-8").strip() \
        if (adapter / "system_prompt.txt").exists() else "Write a tweet in your authentic voice."

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),  # adapter dir; unsloth resolves the base from adapter config
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    user_msg = f"Write a tweet about {args.topic}." if args.topic else "Write a tweet."
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    print(f"Prompt: {user_msg}\n" + "-" * 60)
    for i in range(args.num):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new,
            do_sample=True,
            temperature=args.temp,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"{i+1:>2}. {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
