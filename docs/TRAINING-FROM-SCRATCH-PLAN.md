# Training From Scratch Plan: Vintage Multi-Modal Models

**Created:** May 28, 2026
**Status:** Planning Phase
**Goal:** Train small, efficient models from scratch on archive.org + Reddit data -- text, music, and image generation -- that run on phones and consumer GPUs. Focus on vintage aesthetics, raw internet voice, and public domain content.

---

## Why From Scratch?

Fine-tuning existing LLMs (like we did with Qwen) works for style transfer -- but it's constrained by the base model's knowledge, biases, and safety-tuning. Training from scratch on curated data gives us:

1. **Full control over the data diet** -- no corporate safety filters, no Wikipedia-slop prose
2. **Domain-native voice** -- a model that thinks in 1920s prose + 2020s shitposting, not sanitized chatbot
3. **Small & fast** -- purpose-built for phones/edge, not bloated with useless knowledge
4. **Legal safety** -- public domain data = no copyright nightmares
5. **Multi-modal from the ground up** -- text, image, music sharing a unified aesthetic

---

## Architecture: The "Liquid" Philosophy

Liquid AI's core insight is that recurrent state-space models (SSMs) can match transformers with 10-100x less compute. The key properties we want:

- **Sub-quadratic attention** -- Mamba, RWKV, or linear attention
- **Small parameter count** -- 100M-1B range (fits on phones)
- **Fast inference** -- 30+ tokens/sec on a phone CPU
- **Multi-modal capable** -- shared backbone, modality-specific heads

### Candidate Architectures (ordered by maturity)

| Architecture | Type | Params | Speed | Maturity | Best For |
|---|---|---|---|---|---|
| **Mamba-2** | SSM | 130M-2.8B | Very fast | High (2024) | Text, general |
| **RWKV-7** | Linear RNN | 100M-7B | Very fast | High (2025) | Text, easy training |
| **Liquid LFM** | Liquid NN | 1B-3B | Extremely fast | Low (proprietary) | Edge deployment |
| **Hymba** | SSM+Attention hybrid | 125M-1.5B | Fast | Medium (2024) | Balanced |
| **SmolLM2** | Transformer | 135M-1.7B | Moderate | High (2025) | Safe fallback |
| **MobiLlama** | Transformer | 125M-1B | Moderate | High (2024) | Phone-optimized |

**Recommendation:** Start with **RWKV-7 430M** for text (mature, easy to train, Apache 2.0 license). For multi-modal, use a shared RWKV backbone with modality-specific encoder/decoder heads.

### Why Not Liquid AI Directly?

Liquid AI's LFM architecture is proprietary and closed-source. We can't train it. But the philosophy -- recurrent models beating transformers at small scale -- is accessible through Mamba and RWKV. These are the closest open-source analogues.

---

## Dataset Strategy: archive.org + Reddit

### Text Data

#### Tier 1: Project Gutenberg (archive.org mirror)
- **What:** 70,000+ public domain books -- novels, poetry, philosophy, scientific papers (pre-1928)
- **Size:** ~15 GB raw text, ~3B tokens after filtering
- **Vibe:** Victorian prose, Edwardian adventure, Romantic poetry, 19th-century science, gothic horror
- **Cost:** FREE (public domain, already on archive.org)
- **How:** Bulk download via `gutenberg.org/ebooks/` or archive.org torrents
- **Processing needed:** Strip Gutenberg headers/footers, filter by quality (not OCR garbage), deduplicate

#### Tier 2: Old Newspapers (archive.org + Chronicling America)
- **What:** Scanned newspapers 1777-1963, OCR'd by Library of Congress
- **Size:** ~1-5 GB of OCR text (noisy), ~200M tokens after cleaning
- **Vibe:** Wire-service brevity, sensational headlines, period slang, advertisements
- **Cost:** FREE
- **How:** Chronicling America API or archive.org newspaper collections
- **Processing needed:** Heavy OCR cleanup, layout extraction, deduplication

#### Tier 3: Reddit Comments (PushShift or direct API)
- **What:** Raw Reddit comments from specific subreddits
- **Target subs:** 
  - `/r/AskHistorians` -- deep, well-written historical analysis
  - `/r/redscarepod` -- aesthetic, cultural criticism, arts discourse
  - `/r/TrueFilm` -- film analysis and criticism
  - `/r/LetsTalkMusic` -- music discussion
  - `/r/ArtHistory` -- visual art discourse
  - `/r/oldrecipes` -- vintage food culture
  - `/r/vintageobscura` -- obscure old music
  - `/r/PropagandaPosters` -- visual propaganda
  - `/r/TheWayWeWere` -- historical photographs + discussion
- **Size:** ~5-10 GB raw text from targeted subs
- **Vibe:** Internet-native, critical, aesthetic-obsessed, sometimes toxic (handled via filtering)
- **Cost:** PushShift torrents are free; Reddit API has rate limits
- **Processing needed:** Filter by score ≥ 5, strip deleted/removed, strip URL-only comments, deduplicate

#### Tier 4: Old Magazines & Pulps (archive.org)
- **What:** Scanned pulp magazines (Amazing Stories, Weird Tales, Black Mask) 1920s-1950s
- **Size:** ~500 MB-2 GB after OCR
- **Vibe:** Hard-boiled detective, cosmic horror, sci-fi pulp, lurid advertising
- **Cost:** FREE
- **How:** archive.org magazine collections, pulp magazine archives
- **Processing:** OCR + cleanup, extract fiction sections

#### Tier 5: Old Technical Manuals & Catalogs (archive.org)
- **What:** Pre-1960 engineering manuals, Sears catalogs, farming almanacs
- **Size:** ~200-500 MB
- **Vibe:** Matter-of-fact instructional, period-specific vocabulary, product descriptions
- **Cost:** FREE
- **Why:** Teaches the model concrete, grounded language -- not chatbot fluff

#### Total Text Estimate: 20-25 GB raw → ~4-5B tokens after cleaning

### Music Data

#### Public Domain Recordings (archive.org)
- **What:** 78rpm record collections, cylinder recordings, early jazz, blues, folk, classical performances
- **Key collections:**
  - 78rpm Records & Cylinder Recordings
  - The Great 78 Project
  - Community Audio collections
  - Public domain classical performances
  - Old Time Radio shows
- **Size:** ~50,000-200,000 tracks available
- **Format:** MP3/FLAC, mostly mono, various bitrates
- **Vibe:** Crackling vinyl, lo-fi warmth, virtuosic performances, era-specific genres (ragtime, big band, delta blues)
- **Cost:** FREE
- **Processing needed:** Trim silence, normalize volume, convert to consistent format, deduplicate by audio fingerprint

#### Folk Music Archives
- **What:** Alan Lomax recordings, Library of Congress folk archives, Smithsonian Folkways (older)
- **Size:** ~5,000-20,000 tracks
- **Vibe:** Field recordings, raw performances, regional traditions

#### Old Time Radio (archive.org)
- **What:** Detective dramas, comedy shows, news broadcasts, advertisements (1930s-1950s)
- **Size:** ~10,000-50,000 hours of spoken audio with music
- **Vibe:** Transatlantic accent, period advertising, dramatic organ stings
- **Why:** Teaches the model period speech patterns AND period music cues

#### Total Music Estimate: 100,000-200,000 tracks target (~500-1000 hours after filtering)

### Image Data

#### Public Domain Art (archive.org + museum APIs)
- **What:** Pre-1928 paintings, illustrations, photographs, etchings, woodcuts
- **Size:** ~100,000-500,000 images
- **Vibe:** Oil paintings, lithographs, daguerreotypes, hand-tinted photos, Art Deco, Art Nouveau
- **Sources:**
  - Metropolitan Museum of Art Open Access (375K+ images, CC0)
  - Art Institute of Chicago (50K+, CC0)
  - Rijksmuseum (700K+, public domain)
  - archive.org image collections
- **Cost:** FREE (museum open access programs)

#### Vintage Photography (archive.org + Flickr Commons)
- **What:** Historical photographs 1840s-1960s
- **Size:** ~200,000-1M images
- **Vibe:** Sepia, black-and-white, hand-colored, Kodachrome, large-format, street photography
- **Key sources:**
  - Flickr: The Commons (public domain photos from archives worldwide)
  - Library of Congress photo collections
  - archive.org photograph collections

#### Old Advertisements & Posters (archive.org)
- **What:** Print ads 1880s-1960s, propaganda posters, movie posters, travel posters
- **Size:** ~50,000-200,000 images
- **Vibe:** Bold typography, hand-drawn illustration, period color palettes
- **Sources:**
  - Propaganda Poster collections on archive.org
  - Duke University Ad*Access (7,000+ ads)
  - Vintage Ad Browser

#### Pulp Magazine Covers (archive.org + specialized archives)
- **What:** Sci-fi pulp covers, detective magazine covers, romance pulps
- **Size:** ~10,000-50,000 images
- **Vibe:** Lurid colors, dramatic compositions, hand-painted, sensational
- **Sources:** Pulp magazine cover galleries, Internet Archive scans

#### Total Image Estimate: 500,000-2M images target

---

## Model Cuts & Flavors

The user wants different "cuts" -- model variants for different aesthetics. Here are the options:

### Text Models

| Cut | Training Data | Params | Size | Vibe | Use Case |
|---|---|---|---|---|---|
| **PULP** | Gutenberg novels + pulps + Reddit (50/30/20) | 430M | ~860 MB | Gothic, hard-boiled, poetic, internet-savvy | Creative writing, style transfer |
| **WIRE** | Newspapers + manuals + Reddit (60/30/10) | 430M | ~860 MB | Terse, factual, period-journalistic, deadpan | News generation, briefing, documentation |
| **BOULEVARD** | Gutenberg + magazines + Reddit film/music subs (40/30/30) | 430M | ~860 MB | Cosmopolitan, critical, aesthetically opinionated | Reviews, cultural commentary |
| **RAW** | Reddit-only (100%) | 430M | ~860 MB | Pure internet voice, unfiltered, chaotic | Shitposting, internet-native content |
| **OMNI** | All text data (balanced 30/25/20/15/10) | 1.5B | ~3 GB | Full range, best generalist | General use, everything text |

### Music Models

| Cut | Training Data | Params | Format | Vibe | Use Case |
|---|---|---|---|---|---|
| **WAX** | 78rpm recordings only (jazz, blues, classical) | ~300M | Raw audio VAE + diffusion | Crackle, warmth, mono, pre-1950 | Lo-fi music generation |
| **FOLK** | Field recordings, folk archives, Lomax | ~300M | Raw audio VAE + diffusion | Raw, acoustic, regional, untrained voices | Folk/traditional music |
| **AIRWAVES** | Old Time Radio + spoken word + period music | ~300M | Raw audio VAE + diffusion | Dramatic, radio-play, period speech+music | Audio drama, podcasts |
| **AMBER** | All music data combined | ~800M | Raw audio VAE + diffusion | Full vintage music spectrum | General music generation |

### Image Models

| Cut | Training Data | Params | Format | Vibe | Use Case |
|---|---|---|---|---|---|
| **DAGUERRE** | Vintage photography only (1840s-1940s) | ~400M | VAE + DiT diffusion | Sepia, B&W, hand-tinted, soft focus | Historical-style photos |
| **PALETTE** | Museum art + paintings (all eras pre-1928) | ~400M | VAE + DiT diffusion | Oil, watercolor, etching, academic style | Fine art generation |
| **PULPCOVER** | Pulp magazine covers + posters + ads | ~400M | VAE + DiT diffusion | Bold, lurid, hand-drawn, sensational | Cover art, posters |
| **CHROMA** | All image data combined | ~1B | VAE + DiT diffusion | Full vintage visual aesthetic | General image generation |

### Combined Multi-Modal

| Cut | Modalities | Params | Use Case |
|---|---|---|---|
| **VAULT** | Text + Image + Music | ~2B shared backbone | Generate a pulp story with its cover art and theme music |
| **DIORAMA** | Text + Image | ~1B shared backbone | Illustrated stories, period-specific visual+text |
| **CABARET** | Text + Music | ~1B shared backbone | Song lyrics + music in period styles |

---

## Training Strategy: The ETH Zurich Route Applied to Training From Scratch

**Core principle:** Train the smallest possible model on a tiny subset first. Only scale up when the approach proves itself. Every phase must complete in under 30 minutes and cost under $0.50.

### Phase 0: Data Pipeline (Local)
- **Goal:** Build data download + cleaning pipeline
- **Time:** 1-2 days (mostly download time)
- **Compute:** Local (no GPU needed)
- **Output:** Clean, deduplicated, tokenized dataset files
- **Key decisions:** Tokenizer choice, sequence length, filtering thresholds

### Phase 1: 100M Parameter Proof-of-Concept (Local RTX 2060)
- **Model:** RWKV-7 100M or SmolLM2 135M
- **Data:** 100M tokens (tiny slice of Gutenberg + Reddit)
- **Goal:** Does the model learn coherent English from vintage data?
- **Time:** ~2-4 hours on RTX 2060 (6 GB)
- **Cost:** FREE (local GPU)
- **Success metric:** Generates period-appropriate sentences
- **Gate:** If loss doesn't decrease, something is wrong with data/tokenizer/hyperparams

### Phase 2: 430M Full Text Training (RunPod RTX 3090)
- **Model:** RWKV-7 430M or Mamba-2 430M
- **Data:** 4-5B tokens (all text sources)
- **Time:** ~12-24 hours on RTX 3090
- **Cost:** ~$3-6 (RunPod Community, $0.22/hr)
- **Training config:**
  - Context length: 2048 tokens
  - Batch size: fit to 24 GB VRAM
  - LR: cosine schedule, peak 3e-4
  - Optimizer: AdamW
- **Success metric:** Perplexity < 15 on held-out vintage text

### Phase 3: Text Flavor Training (RunPod)
- **Goal:** Fine-tune Phase 2 base model into specific cuts (PULP, WIRE, etc.)
- **Data:** 500M-1B tokens per flavor (subset of full data)
- **Time:** ~3-6 hours per flavor on RTX 3090
- **Cost:** ~$1-2 per flavor
- **Parallelizable:** Train all 5 flavors simultaneously on 5 pods (~$5-10 total)

### Phase 4: Image Model (RunPod RTX 3090 or A10G)
- **Architecture:** Small DiT-based diffusion model with VAE
  - VAE: Pretrained Stable Diffusion VAE (don't train from scratch)
  - DiT backbone: 400M params
- **Data:** 500K-2M images (all sources)
- **Resolution:** 256x256 (phone-appropriate)
- **Time:** ~24-72 hours on RTX 3090
- **Cost:** ~$6-16
- **Note:** This is the most expensive phase. A100 would cut time to 8-24 hours ($28-84)

### Phase 5: Music Model (RunPod RTX 3090)
- **Architecture:** Audio VAE + diffusion transformer
  - Use pretrained audio VAE (EnCodec or similar)
  - Train diffusion on latent space
- **Data:** 100K-200K audio clips (10-30 second segments)
- **Time:** ~24-48 hours on RTX 3090
- **Cost:** ~$6-12
- **Format:** Mono 22kHz (period-appropriate!)

### Phase 6: Multi-Modal Integration
- **Architecture:** Shared RWKV backbone with modality-specific input/output projectors
- **Training:** Two-stage
  1. Train each modality head separately on the frozen backbone
  2. Joint training on aligned data (e.g., pulp story + matching era-appropriate cover art)
- **Time:** ~12-24 hours on RTX 3090
- **Cost:** ~$3-6

---

## Total Cost Estimate

| Phase | Description | Time | Cost |
|---|---|---|---|
| Phase 0 | Data pipeline | 2 days local | FREE |
| Phase 1 | 100M POC | 2-4 hrs local | FREE |
| Phase 2 | 430M base text | 12-24 hrs cloud | $3-6 |
| Phase 3 | 5 text flavors | 15-30 hrs cloud | $5-10 |
| Phase 4 | Image model | 24-72 hrs cloud | $6-16 |
| Phase 5 | Music model | 24-48 hrs cloud | $6-12 |
| Phase 6 | Multi-modal | 12-24 hrs cloud | $3-6 |
| **TOTAL** | | | **$23-50** |

**Reality check:** $23-50 is extremely cheap for training 9+ models from scratch. This is possible because:
1. Models are small (430M-1B params, not 7B+)
2. Data is free (public domain)
3. RunPod Community Cloud is dirt cheap ($0.22/hr)
4. We're not training the VAE components from scratch (reuse Stable Diffusion VAE)

---

## What Makes This Different From Fine-Tuning

| Aspect | Fine-Tuning (twitterGPT) | Training From Scratch (this plan) |
|---|---|---|
| Model origin | Starts from Qwen/Mistral base | Starts from random weights |
| Knowledge | Inherits base model's world knowledge | Learns only from our data |
| Safety filters | Inherits corporate alignment training | None -- raw data, raw model |
| Vocabulary | Base model's tokenizer (biased to modern web) | Custom tokenizer trained on vintage text |
| Speed | 14 min for 4B model (only trains LoRA weights) | 12-24 hrs for 430M model (trains all weights) |
| Cost per run | $0.05-0.17 | $3-24 |
| Output quality | Constrained by base model's style | Fully native to training data |
| Model size | 4B+ (needs 8+ GB VRAM) | 100M-430M (runs on phones) |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 430M model too dumb | Medium | High | Try 1.5B or Mamba-2 2.8B; still small enough for phones |
| OCR quality ruins text data | High | Medium | Heavy filtering, only use high-confidence OCR, prefer Gutenberg (clean text) |
| Music copyright issues | Medium | High | Strictly pre-1928 recordings only; verify on US copyright database |
| Image quality from 256px | Low | Low | Target 512px if 256px looks bad; still fits on phones |
| Training instability | Medium | Medium | Phase 1 catches this early; smaller model first |
| RunPod Community spot reclaimed | High | Low | Save checkpoints every 30 min; use Secure Cloud for critical runs |
| Not enough VRAM for 430M | Low | Medium | RWKV is more memory-efficient than transformers; 430M fits on 24 GB |

---

## Timeline

| Week | Milestone |
|---|---|
| 1 | Data pipeline built; Phase 0 complete; all datasets downloaded + cleaned |
| 2 | Phase 1 (100M POC) trained locally; tokenizer validated; hyperparameters tuned |
| 3 | Phase 2 (430M base text) trained; loss curves look good; first samples generated |
| 4 | Phase 3 (text flavors) trained; PULP, WIRE, RAW ready for testing |
| 5-6 | Phase 4 (image model) trained on RunPod; first vintage-style images |
| 7-8 | Phase 5 (music model) trained; first period-audio clips |
| 9-10 | Phase 6 (multi-modal) integrated; CABARET + DIORAMA demos |
| 11+ | Iteration, quality improvements, phone deployment |

---

## Phone Deployment Target

| Model | Size (4-bit) | Memory | Speed (iPhone 15) |
|---|---|---|---|
| RWKV-7 430M | ~215 MB | ~500 MB RAM | ~40 tok/s |
| Mamba-2 430M | ~215 MB | ~500 MB RAM | ~35 tok/s |
| RWKV-7 1.5B | ~750 MB | ~1.5 GB RAM | ~15 tok/s |
| DiT 400M (image) | ~200 MB | ~1 GB RAM | ~5 sec/image |
| Audio VAE+DiT 300M | ~150 MB | ~800 MB RAM | ~10 sec/clip |

Target: All models under 2 GB total for a full multi-modal app on a phone.

---

## First Actions (This Week)

1. [ ] Download Project Gutenberg bulk (torrent: gutenberg.org)
2. [ ] Download PushShift Reddit comments dump (target subs)
3. [ ] Set up data cleaning pipeline (Python scripts: dedup, filter, tokenize)
4. [ ] Choose architecture: benchmark RWKV-7 vs Mamba-2 at 100M scale on RTX 2060
5. [ ] Train 100M proof-of-concept for 2 hours locally
6. [ ] If POC works: provision RunPod RTX 3090 for Phase 2

---

## Appendix: Key Links

- **archive.org:** https://archive.org/
- **Project Gutenberg:** https://www.gutenberg.org/
- **Chronicling America:** https://chroniclingamerica.loc.gov/
- **Met Museum Open Access:** https://www.metmuseum.org/art/collection/search
- **Flickr Commons:** https://www.flickr.com/commons
- **PushShift Reddit:** https://academictorrents.com/ (search "reddit comments")
- **RWKV:** https://github.com/BlinkDL/RWKV-LM
- **Mamba:** https://github.com/state-spaces/mamba
- **Liquid AI:** https://www.liquid.ai/ (reference, not usable)
