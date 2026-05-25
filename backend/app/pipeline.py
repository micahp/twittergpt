"""Dataset preparation — thin wrapper that reuses the standalone prepare_dataset.py
so there is a single source of truth for archive parsing."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from .config import DATASETS_DIR, SCRIPTS_DIR, ensure_dirs

_prep = None


def _load_prep():
    """Import prepare_dataset.py by path (it lives at the scripts dir, not in this package)."""
    global _prep
    if _prep is not None:
        return _prep
    script = SCRIPTS_DIR / "prepare_dataset.py"
    if not script.exists():
        raise FileNotFoundError(
            f"prepare_dataset.py not found at {script}. Set TGPT_SCRIPTS_DIR to its directory."
        )
    spec = importlib.util.spec_from_file_location("prepare_dataset", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_dataset"] = mod
    spec.loader.exec_module(mod)
    _prep = mod
    return mod


def prepare(archive_path: Path, dataset_id: str, *, include_replies: bool = True,
            strip_mentions: bool = True, strip_urls: bool = True,
            min_chars: int = 10) -> dict:
    """Run the same logic as the CLI and write into DATASETS_DIR/<dataset_id>/.
    Returns meta.json contents."""
    ensure_dirs()
    prep = _load_prep()
    import zipfile

    out = DATASETS_DIR / dataset_id
    out.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(archive_path)
    if "data/tweets.js" not in zf.namelist():
        raise ValueError("archive missing data/tweets.js — not a valid Twitter archive")

    meta_acct = prep.load_account_meta(zf)
    tweets = prep.read_js_json(zf, "data/tweets.js")

    stats = {"total": len(tweets), "retweets": 0, "replies_in": 0, "originals_in": 0,
             "dropped_short": 0, "dropped_empty": 0, "dropped_dupe": 0, "kept": 0}
    years: dict[str, int] = {}
    seen: set[str] = set()
    records: list[dict] = []

    for d in tweets:
        t = d.get("tweet", {})
        full = t.get("full_text", "")
        if not full:
            continue
        if prep.is_retweet(full):
            stats["retweets"] += 1
            continue
        is_reply = bool(t.get("in_reply_to_status_id_str")) or full.startswith("@")
        if is_reply and not include_replies:
            continue
        stats["replies_in" if is_reply else "originals_in"] += 1

        cleaned = prep.clean_text(full, strip_mentions=strip_mentions, strip_urls=strip_urls)
        if not cleaned:
            stats["dropped_empty"] += 1
            continue
        if len(cleaned) < min_chars:
            stats["dropped_short"] += 1
            continue
        key = prep.norm_key(cleaned)
        if key in seen:
            stats["dropped_dupe"] += 1
            continue
        seen.add(key)
        yr = t.get("created_at", "")[-4:]
        if yr:
            years[yr] = years.get(yr, 0) + 1
        records.append({"prompt": "Write a tweet.", "completion": cleaned, "is_reply": is_reply,
                        "created_at": t.get("created_at")})

    stats["kept"] = len(records)

    import random
    rng = random.Random(42)
    rng.shuffle(records)
    n_eval = max(1, int(len(records) * 0.05)) if records else 0
    eval_recs, train_recs = records[:n_eval], records[n_eval:]

    def write_jsonl(path: Path, recs):
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out / "train.jsonl", train_recs)
    write_jsonl(out / "eval.jsonl", eval_recs)

    total_chars = sum(len(r["completion"]) for r in records)
    meta = {
        "dataset_id": dataset_id,
        "account": meta_acct,
        "config": {"include_replies": include_replies, "strip_mentions": strip_mentions,
                   "strip_urls": strip_urls, "min_chars": min_chars},
        "stats": stats,
        "by_year": dict(sorted(years.items())),
        "train_count": len(train_recs),
        "eval_count": len(eval_recs),
        "approx_tokens_kept": round(total_chars / 4),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def get_meta(dataset_id: str) -> dict | None:
    p = DATASETS_DIR / dataset_id / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
