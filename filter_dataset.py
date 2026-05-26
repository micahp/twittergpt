"""
filter_dataset.py — Filter out replies, keep only original tweets.

Rationale: replies are reactive ("this is cool!") while original tweets 
show the actual thinking/voice. For learning the identity, original tweets
are much more valuable.

75.7% of dataset is replies → only 24.3% are originals.
"""
import json
from pathlib import Path

data_dir = Path(__file__).parent / "dataset_out"
raw = data_dir / "train.jsonl"

originals = []
replies = 0

with open(raw) as f:
    for line in f:
        ex = json.loads(line)
        if ex.get("is_reply"):
            replies += 1
        else:
            originals.append(ex)

# Write filtered
filtered = data_dir / "train_filtered.jsonl"
with open(filtered, "w") as f:
    for ex in originals:
        f.write(json.dumps(ex) + "\n")

print(f"Filtered: {len(originals):,} original tweets (removed {replies:,} replies)")
