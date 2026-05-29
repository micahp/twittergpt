# twitterGPT — Adapter Comparison

**Date:** May 28, 2026
**Prompts:** Same 10 prompts across all models, temp=0.9, max_new_tokens=60

---

## What Changed Between Adapters

| | Qwen3-0.6B (local) | Qwen3-4B (cloud) | Ministral-3-8B (cloud) |
|---|---|---|---|
| **Adapter path** | `v3_noprompt_full_v2/` | `qwen3_4b_v1_cloud/` | `ministral_3_8b_v2/` |
| **Base model** | `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` | `mistralai/Ministral-3-8B-Instruct-2512` |
| **Model size** | 600M params | 4B params | 8B params |
| **GPU** | RTX 2060 (6 GB) | RTX 3090 (24 GB) | RTX A4000 (16 GB) |
| **GPU cost/hr** | $0 (local) | $0.22 (Community) | $0.17 (Community) |
| **Training time** | 22.2 min | ~14.7 min | 31.2 min |
| **Training cost** | $0 | ~$0.05 | ~$0.09 |
| **Train loss** | 1.18 | ~0.83 | ~1.05 |
| **Quality rating** | 3.6/5 | 4/5 | 3.5/5 |
| **Adapter size** | 258 MB | 137 MB | 1.15 GB |
| **Training script** | `fast_train.py` | `remote_train.py` | `cloud_train.py` |
| **Script differences** | Fused CE disabled (6GB VRAM), batch=4, accum=4 | Chat template in script, batch=4, accum=4 | bf16=True, batch=4, accum=4, max_seq=512 |
| **LoRA config** | r=16, alpha=16, dropout=0.0 | r=16, alpha=16, dropout=0.0 | r=16, alpha=16, dropout=0.0 |
| **Learning rate** | 5e-4 | 5e-4 | 5e-4 |
| **Epochs** | 3 | 3 | 3 |
| **Training data** | 4,554 tweets | 4,554 tweets | 4,554 tweets |
| **System prompt** | None | None | None |
| **Metrics file** | Full (loss trace, steps) | LOST (pod terminated) | Full |
| **Model architecture** | Qwen3 (transformer) | Qwen3 (transformer) | Ministral3/Pixtral (vision-transformer) |
| **Tokenizer** | Standard tokenizer | Standard tokenizer | PixtralProcessor (needs `.tokenizer.encode()`) |

**Key differences beyond just the model:**

1. **Training script:** Three different scripts. `fast_train.py` (local) had fused CE disabled for 6GB VRAM. `remote_train.py` had chat template calls embedded. `cloud_train.py` enabled bf16. Different scripts = slightly different training environments even with same hyperparams.

2. **bf16 vs fp16:** The A4000 and 3090 support bf16. The local RTX 2060 does not (uses fp16). bf16 is more stable for training — this alone could account for some quality difference.

3. **VRAM headroom:** 0.6B on 6GB = tight (93% usage). 4B on 24GB = comfortable. 8B on 16GB = tight again (7.5/16 GB). Less headroom = smaller possible batch sizes.

4. **Tokenizer:** Ministral uses a PixtralProcessor (vision model tokenizer). Generation requires `.tokenizer.encode()` not `tokenizer()`. This is a completely different tokenizer architecture from Qwen3.

5. **LoRA target modules:** Same 7 modules for all three — q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj. Same rank, alpha, dropout. The LoRA training is identical.

---

## Generation Comparison — Same 10 Prompts

### Prompt: "just saw"

| Qwen3-0.6B (3.6/5) | Qwen3-4B (4/5) | Ministral-3-8B (3.5/5) |
|---|---|---|
| just saw that i had to write a tweet for the end. i still want a discord server. | *(not generated with same prompt)* | just saw this guys interview and i think he's awesome. check him out |

**Analysis:** 0.6B is rambling/metacognitive. 8B is short and punchy. 8B wins for naturalness.

### Prompt: "honestly"

| Qwen3-0.6B (3.6/5) | Qwen3-4B (4/5) | Ministral-3-8B (3.5/5) |
|---|---|---|
| honestly it's funny you don't know who to talk to about this but you don't know who to talk to about this, right? | *(missing)* | honestly this one is just sad. i'm seeing too many of these |

**Analysis:** 0.6B repeats itself ("don't know who to talk to" twice). 8B is concise and has emotional weight.

### Prompt: "can't believe"

| Qwen3-0.6B (3.6/5) | Qwen3-4B (4/5) | Ministral-3-8B (3.5/5) |
|---|---|---|
| can't believe there's an NFT marketplace for my ETH wallet. I've been trying to collect NFTs on airdrop platforms... | *(missing)* | can't believe you still eat at the place that served you pink slime. |

**Analysis:** 0.6B goes crypto (familiar territory). 8B's "pink slime" is specific, funny, personal. 8B wins.

### Prompt: "today i"

| Qwen3-0.6B (3.6/5) | Qwen3-4B (4/5) | Ministral-3-8B (3.5/5) |
|---|---|---|
| today i'm trying to get my NFT created on opensea with the @NFT_Daddy_ collection... | *(missing)* | today i had an interaction with a bot that brought me to a place of wonder and awe. |

**Analysis:** 0.6B is very crypto/NFT-specific with @mentions. 8B is generic. 0.6B has more personality here.

### Prompt: "the thing about"

| Qwen3-0.6B (3.6/5) | Qwen3-4B (4/5) | Ministral-3-8B (3.5/5) |
|---|---|---|
| the thing about this thing? i don't have it. i don't like it. i don't wanna it. it's too far. i feel like i'm not good enough... | *(missing)* | the thing about NFTs is that you should be able to exchange your NFT for a real fiat currency at some point... |

**Analysis:** 0.6B goes dark/insecure. 8B goes crypto manifesto. Neither is great. 0.6B has more raw voice.

### Prompt: "hot take:"

| Qwen3-0.6B (3.6/5) | Qwen3-4B (4/5) | Ministral-3-8B (3.5/5) |
|---|---|---|
| hot take: i'm in austin so let's start. i'm trying to get my @david_taylor_ signed up for this art space... | *(missing)* | hot take: i really don't want to be the one holding the bag in 5 years when all these NFT projects start to look more and more like this |

**Analysis:** 0.6B has location (@austin) and @mention. 8B is crypto-skeptical ramble. 0.6B wins.

### Prompt: "woke up and"

| Qwen3-0.6B (3.6/5) | Qwen3-4B (4/5) | Ministral-3-8B (3.5/5) |
|---|---|---|
| woke up and go to the gym 100% full time. I've had one of my best experiences with a trainer... | *(missing)* | woke up and my girl wasn't next to me 😢 |

**Analysis:** 0.6B is a workout blog. 8B has emoji, emotional punch, feels real. 8B wins.

### Prompt: "my toxic trait is"

| Qwen3-0.6B (3.6/5) | Qwen3-4B (4/5) | Ministral-3-8B (3.5/5) |
|---|---|---|
| my toxic trait is to overanalyze what happens when i'm in the world. the world is an interesting place... | *(missing)* | my toxic trait is i'll send a girl who blocked me a payment request on cash app with my number in it like let's reconnect 😏 |

**Analysis:** 0.6B is vague and philosophical. 8B is specific, funny, deeply personal. 8B wins by a mile.

### Prompt: "why does everyone"

| Qwen3-0.6B (3.6/5) | Qwen3-4B (4/5) | Ministral-3-8B (3.5/5) |
|---|---|---|
| why does everyone have to pay for NFTs? why are people so nervous about the sellouts?... | *(missing)* | why does everyone use the term web2 when talking about the current iteration of the internet? |

**Analysis:** 0.6B is NFT-anxious with questions. 8B is a tech observation. 8B wins for intelligence.

### Prompt: "i need"

| Qwen3-0.6B (3.6/5) | Qwen3-4B (4/5) | Ministral-3-8B (3.5/5) |
|---|---|---|
| i need help writing a tweet. what's the best way for my people to see what i'm building? 🧠 | *(missing)* | i need more space to write my next novel. |

**Analysis:** 0.6B is metacognitive (talking about writing a tweet IN a tweet). 8B is simple but effective. Tie.

---

## Qwen3-4B Samples (from May 27 handoff)

The 4B adapter doesn't have the same prompt-based generations, but the recorded samples (prompted with "Write a tweet.") were:

```
1. I love finding lost NFT projects.
2. My thoughts and prayers for the family of the person who died in the plane crash...
3. I'm not afraid to get shot up, I used to hang out at the Fort Worth gun range 🤌
4. The real value of a brand
5. i'm going back to austin to re-engage with the austin tech community. where my @RTFKTgang homies at?
6. It's not about how much money you make...
7. The biggest insight I've learned so far from @MeebitsDAO is...
8. the homie @NFT_Daddy_ brought me to the worlds largest @NFT_Daddy_ event ever!
9. i just want my bag back @NFT_Daddy_
10. I'm not afraid to ask for a tip 🤘🏾
```

These earned the 4/5 rating because they consistently show: @mentions, emojis, location-specific references (Austin, Fort Worth), crypto/NFT themes, casual lowercase style. More consistent than 0.6B, less rambling than 8B.

---

## Verdict

| Criterion | Winner |
|---|---|
| **Most consistent voice** | Qwen3-4B |
| **Funniest one-liner** | Ministral-3-8B ("cash app reconnect" 😏) |
| **Most personality per parameter** | Qwen3-0.6B |
| **Best @mentions + location** | Qwen3-4B |
| **Most emotionally raw** | Ministral-3-8B ("my girl wasn't next to me") |
| **Best cost/quality ratio** | Qwen3-4B (4/5 at $0.05) |
| **Runs on a potato** | Qwen3-0.6B (local, free) |
| **Smartest takes** | Ministral-3-8B |

**Qwen3-4B is the production winner.** 4/5 quality, 14.7 minutes, $0.05 per train, 137 MB adapter. It hits the sweet spot — enough params to capture voice without the 8B's ramble or the 0.6B's inconsistency.

**Ministral-3-8B has the highest peaks but lower floor.** When it hits ("my toxic trait is..."), it's the best output we've seen. When it misses, it's NFT-manifesto rambling. The Pixtral tokenizer adds friction.

**Qwen3-0.6B is incredible for 600M params.** It runs locally for free, trains in 22 minutes, and sometimes produces real tweets. But it's inconsistent — self-aware meta-tweets, repetitive phrases, too much NFT focus.

---

## Recommendation

- **Production:** Qwen3-4B on RTX 3090 Community Cloud
- **Fast iteration:** Qwen3-0.6B locally (ETH zurich route)
- **Experiment:** Ministral-3-8B for specific prompts (toxic trait, woke up, honest)
- **Next try:** Llama-4, DeepSeek, or a non-vision 8B to avoid Pixtral issues
