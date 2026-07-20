#!/usr/bin/env python3
"""
Balance Dataset – Downsample terminal to 1k, keep all code/direct
==================================================================
Reads training data from HF, downsamples terminal to target, uploads back.
"""

import json, os, sys, random
from collections import defaultdict
from huggingface_hub import HfApi, create_repo, upload_file, hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"
TARGET_TERMINAL = int(os.environ.get("TARGET_TERMINAL", "1000"))

if not HF_TOKEN:
    print("❌ HF_TOKEN not set")
    sys.exit(1)

random.seed(42)

print("\n📊 Balancing dataset – downsampling terminal")
print(f"   Target terminal: {TARGET_TERMINAL}")
print(f"   Repo: {HF_REPO_ID}\n")

# Download
try:
    data_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="classifier_training_data/classifier_training_data.jsonl",
        repo_type=HF_REPO_TYPE,
        token=HF_TOKEN,
    )
except Exception as e:
    print(f"❌ Download failed: {e}")
    sys.exit(1)

data = []
with open(data_path) as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data):,} examples")

# Split
code = [d for d in data if d["label"] == "code"]
direct = [d for d in data if d["label"] == "direct"]
terminal = [d for d in data if d["label"] == "terminal"]

print(f"  code: {len(code):,}")
print(f"  direct: {len(direct):,}")
print(f"  terminal: {len(terminal):,}")

# Downsample terminal
if len(terminal) > TARGET_TERMINAL:
    terminal = random.sample(terminal, TARGET_TERMINAL)
    print(f"\n✂️  Downsampled terminal to {len(terminal):,}")
else:
    print(f"\n✅ Terminal already ≤ {TARGET_TERMINAL}, keeping all")

balanced = code + direct + terminal
random.shuffle(balanced)

print(f"\n✅ Balanced dataset: {len(balanced):,}")
counts = defaultdict(int)
for item in balanced:
    counts[item["label"]] += 1
for cat, cnt in sorted(counts.items()):
    print(f"  {cat}: {cnt:,}")

# Save locally
out_path = "classifier_training_data_balanced.jsonl"
with open(out_path, "w") as f:
    for item in balanced:
        f.write(json.dumps(item) + "\n")

# Upload back to HF (overwrite original)
print("\n📤 Uploading to HF...")
api = HfApi(token=HF_TOKEN)
create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, repo_type=HF_REPO_TYPE, exist_ok=True)
upload_file(
    path_or_fileobj=out_path,
    path_in_repo="classifier_training_data/classifier_training_data.jsonl",
    repo_id=HF_REPO_ID,
    repo_type=HF_REPO_TYPE,
    commit_message=f"Downsample terminal to {len(terminal):,} (balanced dataset)",
)
print("✅ Upload complete!")