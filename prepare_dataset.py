#!/usr/bin/env python3
"""
prepare_dataset.py — turn a Twitter/X data archive (.zip) into a finetuning dataset.

Reads only the needed `data/*.js` files directly from the zip (ignores the
multi-GB `assets/` media). Filters out retweets, keeps the user's own originals
and replies, cleans the text, dedupes, and writes completion-style JSONL plus a
meta.json used to build the model's system prompt.

Usage:
    python3 prepare_dataset.py ARCHIVE.zip [-o OUTDIR] [options]

This is the M0 deliverable from the twitterGPT PRD and is intended to work on
any standard Twitter archive, not just one account.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import zipfile
from pathlib import Path

# --- regexes -----------------------------------------------------------------
TCO_URL = re.compile(r"https?://t\.co/\S+")
ANY_URL = re.compile(r"https?://\S+")
LEADING_MENTIONS = re.compile(r"^(?:@\w+\s+)+")  # leading "@a @b " run on replies
WHITESPACE = re.compile(r"[ \t]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")


def strip_js_prefix(raw: str) -> str:
    """Twitter wraps JSON as `window.YTD.<x>.partN = [ ... ]`. Return from first '['."""
    i = raw.find("[")
    if i == -1:
        raise ValueError("no JSON array found in file")
    return raw[i:]


def read_js_json(zf: zipfile.ZipFile, name: str):
    raw = zf.read(name).decode("utf-8", "replace")
    return json.loads(strip_js_prefix(raw))


def load_account_meta(zf: zipfile.ZipFile) -> dict:
    meta = {"username": None, "display_name": None, "bio": None, "location": None}
    names = set(zf.namelist())
    if "data/account.js" in names:
        try:
            acct = read_js_json(zf, "data/account.js")[0]["account"]
            meta["username"] = acct.get("username")
            meta["display_name"] = acct.get("accountDisplayName")
        except Exception:
            pass
    if "data/profile.js" in names:
        try:
            prof = read_js_json(zf, "data/profile.js")[0]["profile"]
            desc = prof.get("description", {})
            meta["bio"] = TCO_URL.sub("", desc.get("bio", "")).strip() or None
            meta["location"] = desc.get("location") or None
        except Exception:
            pass
    return meta


def clean_text(text: str, *, strip_mentions: bool, strip_urls: bool) -> str:
    text = html.unescape(text)
    if strip_mentions:
        text = LEADING_MENTIONS.sub("", text)
    if strip_urls:
        text = TCO_URL.sub("", text)
        text = ANY_URL.sub("", text)
    # collapse whitespace but keep emoji and single newlines
    text = WHITESPACE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def is_retweet(full_text: str) -> bool:
    return full_text.startswith("RT @")


def norm_key(text: str) -> str:
    """Normalization key for near-duplicate detection."""
    k = re.sub(r"\s+", " ", text.lower()).strip()
    k = re.sub(r"[^\w\s]", "", k)
    return hashlib.md5(k.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", help="path to Twitter archive .zip")
    ap.add_argument("-o", "--outdir", default="dataset_out", help="output directory")
    ap.add_argument("--include-replies", dest="include_replies", action="store_true", default=True,
                    help="include replies (default: on)")
    ap.add_argument("--no-replies", dest="include_replies", action="store_false",
                    help="originals only, exclude replies")
    ap.add_argument("--keep-mentions", dest="strip_mentions", action="store_false", default=True,
                    help="keep leading @mentions on replies (default: strip them)")
    ap.add_argument("--keep-urls", dest="strip_urls", action="store_false", default=True,
                    help="keep URLs (default: strip them)")
    ap.add_argument("--min-chars", type=int, default=10, help="drop tweets shorter than this after cleaning")
    ap.add_argument("--eval-frac", type=float, default=0.05, help="fraction held out for eval")
    ap.add_argument("--prompt", default="Write a tweet.", help="instruction used as the prompt field")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    arc = Path(args.archive)
    if not arc.exists():
        print(f"error: archive not found: {arc}", file=sys.stderr)
        return 1

    zf = zipfile.ZipFile(arc)
    if "data/tweets.js" not in zf.namelist():
        print("error: data/tweets.js not found in archive", file=sys.stderr)
        return 1

    meta = load_account_meta(zf)
    tweets = read_js_json(zf, "data/tweets.js")

    stats = {
        "total": len(tweets), "retweets": 0, "replies_in": 0, "originals_in": 0,
        "dropped_short": 0, "dropped_empty": 0, "dropped_dupe": 0, "kept": 0,
    }
    years: dict[str, int] = {}
    seen: set[str] = set()
    records: list[dict] = []

    for d in tweets:
        t = d.get("tweet", {})
        full = t.get("full_text", "")
        if not full:
            continue
        if is_retweet(full):
            stats["retweets"] += 1
            continue

        is_reply = bool(t.get("in_reply_to_status_id_str")) or full.startswith("@")
        if is_reply and not args.include_replies:
            continue
        if is_reply:
            stats["replies_in"] += 1
        else:
            stats["originals_in"] += 1

        cleaned = clean_text(full, strip_mentions=args.strip_mentions, strip_urls=args.strip_urls)
        if not cleaned:
            stats["dropped_empty"] += 1
            continue
        if len(cleaned) < args.min_chars:
            stats["dropped_short"] += 1
            continue

        key = norm_key(cleaned)
        if key in seen:
            stats["dropped_dupe"] += 1
            continue
        seen.add(key)

        yr = t.get("created_at", "")[-4:]
        if yr:
            years[yr] = years.get(yr, 0) + 1

        records.append({
            "prompt": args.prompt,
            "completion": cleaned,
            "is_reply": is_reply,
            "created_at": t.get("created_at"),
            "favorite_count": int(t.get("favorite_count", 0) or 0),
            "retweet_count": int(t.get("retweet_count", 0) or 0),
        })

    stats["kept"] = len(records)

    # deterministic shuffle + split
    import random
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_eval = max(1, int(len(records) * args.eval_frac)) if records else 0
    eval_recs, train_recs = records[:n_eval], records[n_eval:]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, recs: list[dict]):
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out / "train.jsonl", train_recs)
    write_jsonl(out / "eval.jsonl", eval_recs)

    total_chars = sum(len(r["completion"]) for r in records)
    meta_out = {
        "account": meta,
        "config": {
            "include_replies": args.include_replies,
            "strip_mentions": args.strip_mentions,
            "strip_urls": args.strip_urls,
            "min_chars": args.min_chars,
            "eval_frac": args.eval_frac,
            "prompt": args.prompt,
            "seed": args.seed,
        },
        "stats": stats,
        "by_year": dict(sorted(years.items())),
        "train_count": len(train_recs),
        "eval_count": len(eval_recs),
        "total_chars_kept": total_chars,
        "avg_chars_kept": round(total_chars / len(records), 1) if records else 0,
        "approx_tokens_kept": round(total_chars / 4),  # ~4 chars/token rough estimate
    }
    (out / "meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2))

    # --- report ---
    print(f"Archive : {arc.name}")
    uname = meta.get("username")
    print(f"Account : @{uname}" + (f" ({meta['display_name']})" if meta.get("display_name") else ""))
    print()
    print(f"  total tweets in archive : {stats['total']:>7,}")
    print(f"  retweets (excluded)     : {stats['retweets']:>7,}")
    print(f"  originals considered    : {stats['originals_in']:>7,}")
    print(f"  replies considered      : {stats['replies_in']:>7,}"
          + ("" if args.include_replies else "  (excluded by --no-replies)"))
    print(f"  dropped (too short)     : {stats['dropped_short']:>7,}")
    print(f"  dropped (empty clean)   : {stats['dropped_empty']:>7,}")
    print(f"  dropped (near-dupe)     : {stats['dropped_dupe']:>7,}")
    print(f"  ── kept                 : {stats['kept']:>7,}")
    print()
    print(f"  train / eval            : {len(train_recs):,} / {len(eval_recs):,}")
    print(f"  avg chars kept          : {meta_out['avg_chars_kept']}")
    print(f"  approx tokens           : ~{meta_out['approx_tokens_kept']:,}")
    print()
    print(f"Wrote: {out/'train.jsonl'}, {out/'eval.jsonl'}, {out/'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
