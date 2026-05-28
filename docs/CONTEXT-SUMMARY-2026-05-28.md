# twitterGPT — RunPod Cloud GPU Training — May 27/28, 2026

## Qwen3-4B-Instruct (PRODUCTION READY)
- Model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
- Time: 884s (14.7 min) on RTX 3090 Community ($0.22/hr)
- Cost: ~$0.05
- Loss: 0.83 (final), train_loss 0.67 (epoch 3 avg), trace 0.82-0.98
- Adapter: 127 MB at /workspace/adapters/qwen3_4b_v1/
- Quality: 4/5 — natural @mentions, emojis, lowercase, location-specific, NFT themes

## Ministral-3-8B-Instruct (TRAINED, GENERATION BROKEN)
- Model: unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit
- Time: 1786s (29.8 min)
- Cost: ~$0.11
- Loss: 0.32 (final), trace 0.32-0.42 (very tight — possible overfit)
- Adapter: 205 MB at /workspace/adapters/ministral_3_8b_v1/
- Issue: `KeyError: 'llava'` when calling `get_chat_template(tokenizer, chat_template="mistral")`
  - remote_train.py hardcodes chat_template="qwen-2.5" — wrong for Ministral but training succeeded
  - Need to find correct Unsloth chat template or skip get_chat_template entirely

## Pod Info
- ID: au6nq835ytj7er
- Name: twittergpt
- GPU: RTX 3090 Community, 24 GB VRAM, $0.22/hr
- Template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
- Disk: 40 GB, RAM: 62 GB, vCPU: 12
- SSH: ssh au6nq835ytj7er-64411cc9@ssh.runpod.io -i ~/.ssh/id_runpod_twittergpt

## SSH Setup
- Fresh key at ~/.ssh/id_runpod_twittergpt (twittergpt-runpod)
- Added to RunPod profile (console > Settings > SSH Public Keys)
- SCP/SFTP do NOT work with basic SSH proxy — use Web Terminal or pipe files
- SSH via pipe: `echo 'cmd; exit' | ssh -tt ...` (small commands)
- SSH via heredoc: `ssh ... "bash" < script.sh` (works for larger scripts)
- PTY is required — RunPod SSH proxy rejects non-PTY connections

## File Transfer Patterns (what worked)
1. SMALL COMMANDS: `echo 'cmd; exit' | ssh -tt -i ~/.ssh/id_runpod_twittergpt au6nq835ytj7er-64411cc9@ssh.runpod.io`
2. SCRIPTS: `ssh -tt ... "bash" < script.sh`
3. LARGE FILES: Base64-encode + heredoc Python script piped via `ssh ... "python3" < uploader.py`
4. Upload to dpaste.com for small text, then wget from pod

## Key Files on Pod
- /workspace/remote_train.py — standalone training script (no local deps)
- /workspace/train_filtered.jsonl — 4,554 original tweets (1.1 MB)
- /workspace/adapters/qwen3_4b_v1/ — Qwen3-4B adapter (127 MB)
- /workspace/adapters/ministral_3_8b_v1/ — Ministral adapter (205 MB)
- /workspace/train_qwen3_4b.log — Qwen training log
- /workspace/train_ministral.log — Ministral training log

## Training Command
```
python3 /workspace/remote_train.py \
  --data /workspace/train_filtered.jsonl \
  --model unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit \
  --out /workspace/adapters/qwen3_4b_v1 \
  --epochs 3 --lr 5e-4 --lora-dropout 0.0 --system-prompt none
```

## Generation Snippet (Qwen3-4B, verified working)
```python
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    max_seq_length=128, load_in_4bit=True, dtype=None)
model.load_adapter("/workspace/adapters/qwen3_4b_v1")
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
FastLanguageModel.for_inference(model)

for i in range(10):
    msgs = [{"role": "user", "content": "Write a tweet."}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.9,
                                 do_sample=True, top_p=0.95,
                                 pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Sample Qwen3-4B Tweets (temp 0.9)
1. I love finding lost NFT projects.
2. My thoughts and prayers for the family of the person who died in the plane crash...
3. I'm not afraid to get shot up, I used to hang out at the Fort Worth gun range 🤌
4. The real value of a brand
5. i'm going back to austin to re-engage with the austin tech community. where my @RTFKTgang homies at?
6. It's not about how much money you make. It's about having the opportunity to make money off something you care about.
7. The biggest insight I've learned so far from @MeebitsDAO is: if you want to build a resilient community...
8. the homie @NFT_Daddy_ brought me to the worlds largest @NFT_Daddy_ event ever! @doodles_nft...
9. i just want my bag back @NFT_Daddy_
10. I'm not afraid to ask for a tip 🤘🏾

## Cost Summary
- Total pod runtime: ~45 min at $0.22/hr = ~$0.17
- Qwen3-4B training: $0.05
- Ministral-3-8B training: $0.11
- Zombie/idle pods from debugging: estimated ~$0.10

## Comparison: Local vs Cloud
| Model | GPU | Time | Loss | Quality |
|-------|-----|------|------|---------|
| Qwen3-0.6B | RTX 2060 (6GB) | 22 min | 1.18 | 3.6/5 |
| Qwen3-4B | RTX 3090 (24GB) | 14.7 min | 0.83 | 4/5 |

Cloud RTX 3090 trains a 4B model FASTER than local trains a 0.6B. Obvious path forward.

## RunPod Lessons Learned
1. Create pods WITH SSH public key via env var PUBLIC_KEY or account settings
2. SSH key injection happens at pod creation — add key to profile BEFORE creating pod
3. Port 8888 "initializing" = container pulling image, takes 5-10 min
4. NEVER create parallel pods — wait for one to finish initializing
5. Stop pods immediately when done to avoid idle billing
6. Community Cloud is fine for <1hr runs; Secure Cloud for reliability
7. Pod ID in SSH command changes if pod is re-provisioned

## Next Session TODO
- [x] ~~Fix Ministral-3-8B chat template~~ POD LOST — Community Cloud GPU reclaimed
- [x] ~~SCP adapters from pod to local~~ POD LOST — adapters gone with pod disk
- [ ] **NEW DIRECTION:** Train from scratch on archive.org + Reddit — small multi-modal models (text, music, image) in Liquid AI / efficient architecture style, focus on vintage aesthetics

## May 28 Update: Pod Lost
- Pod au6nq835ytj7er: EXITED, cannot resume — "not enough free GPUs on the host machine"
- Community Cloud spot got reclaimed. Both adapters (qwen3_4b_v1, ministral_3_8b_v1) lost with pod disk.
- Cost: ~$0.22 total burned on this pod. Lesson: always download adapters immediately after training.
- Qwen3-4B was the winner — 14.7 min, loss 0.83, quality 4/5. Worth re-creating later if needed.
- Pivot: focus on training-from-scratch plan (see docs/TRAINING-FROM-SCRATCH-PLAN.md)
- [ ] Compare Qwen3-4B vs Ministral vs local 0.6B quality
- [ ] Research archive.org + Reddit datasets for expanded training
- [ ] Clean up old EXITED pods in RunPod console
- [ ] Consider Lambda Labs for simpler SSH/SCP ($0.50/hr A10)
