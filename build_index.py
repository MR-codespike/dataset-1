#!/usr/bin/env python3
"""
Template Library Indexing & Retrieval Test — GitHub Actions version

What this does:
  1. Downloads your reviewed template library from Hugging Face.
  2. Builds a small embedding index (all-MiniLM-L6-v2, ~80MB).
  3. Saves the index (embeddings + metadata) locally and pushes it back
     to your HF repo under `_index/` so the orchestrator can download a
     ready-made index instead of re-embedding every boot.
  4. (Optional) runs a few test queries to verify retrieval quality.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download, HfApi
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIG — read from environment variables (GitHub Secrets)
# ============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Use the repository workspace (writable)
BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
LOCAL_DIR = os.path.join(BASE_DIR, "templates_index")
INDEX_DIR = os.path.join(LOCAL_DIR, "_index")
EMBEDDINGS_FILE = os.path.join(INDEX_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.json")

# ============================================================================
# Step 1: Download the dataset
# ============================================================================

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable not set.")
    sys.exit(1)

print(f"📥 Downloading dataset from {HF_REPO_ID} ...")
try:
    local_path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        token=HF_TOKEN,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"✅ Downloaded to {local_path}")
except Exception as e:
    print(f"❌ Download failed: {e}")
    sys.exit(1)

# ============================================================================
# Step 2: Collect reviewed templates and build embed texts
# ============================================================================

meta_files = list(Path(local_path).rglob("meta.json"))
print(f"\n📄 Found {len(meta_files)} meta.json files total.")

records = []
skipped_unreviewed = 0

for meta_path in meta_files:
    with open(meta_path, "r") as f:
        meta = json.load(f)

    if meta.get("reviewed") is not True:
        skipped_unreviewed += 1
        continue

    folder = meta_path.parent
    category = meta.get("category", "")
    subcategory = meta.get("subcategory", "")
    description = meta.get("description", "")
    level = meta.get("level", "whole-site")

    embed_text = f"{category} {subcategory}. {description}"

    records.append({
        "id": meta.get("id"),
        "category": category,
        "subcategory": subcategory,
        "level": level,
        "description": description,
        "path": str(folder.relative_to(local_path)),
        "embed_text": embed_text,
    })

print(f"Indexing {len(records)} reviewed templates ({skipped_unreviewed} skipped as unreviewed).")

if len(records) == 0:
    print("⚠️  No reviewed templates found. Nothing to index.")
    sys.exit(0)

# ============================================================================
# Step 3: Generate embeddings
# ============================================================================

print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

texts = [r["embed_text"] for r in records]
print(f"Embedding {len(texts)} templates ...")
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
embeddings = np.array(embeddings, dtype=np.float32)

print(f"Embeddings shape: {embeddings.shape}")

# ============================================================================
# Step 4: Save the index locally
# ============================================================================

os.makedirs(INDEX_DIR, exist_ok=True)
np.save(EMBEDDINGS_FILE, embeddings)

# Remove embed_text from metadata before saving
metadata_to_save = [{k: v for k, v in r.items() if k != "embed_text"} for r in records]
with open(METADATA_FILE, "w") as f:
    json.dump(metadata_to_save, f, indent=2)

print(f"\n✅ Saved index to {INDEX_DIR}")
print(f"  - {EMBEDDINGS_FILE}")
print(f"  - {METADATA_FILE}")

# ============================================================================
# Step 5: Quick retrieval test (optional)
# ============================================================================

test_queries = [
    "build me a website for a yoga studio",
    "I need a site for my law firm",
    "make a portfolio site for a photographer",
    "website for a coffee shop",
    "I want a donation page for my charity",
    "landing page for my new mobile app",
]

print("\n=== Quick retrieval quality check ===\n")

def search(query, top_k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "id": metadata_to_save[idx]["id"],
            "category": metadata_to_save[idx]["category"],
            "subcategory": metadata_to_save[idx]["subcategory"],
            "description": metadata_to_save[idx]["description"],
            "score": float(scores[idx]),
        })
    return results

for query in test_queries:
    print(f"Query: \"{query}\"")
    for r in search(query, top_k=3):
        print(f"  [{r['score']:.3f}] {r['id']} — {r['category']} / {r['subcategory']}")
    print()

# ============================================================================
# Step 6: Upload the built index to Hugging Face
# ============================================================================

print(f"📤 Uploading index to {HF_REPO_ID}/_index ...")
api = HfApi(token=HF_TOKEN)
try:
    api.upload_folder(
        folder_path=INDEX_DIR,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        path_in_repo="_index",
        commit_message=f"Build embedding index ({len(records)} templates)",
    )
    print("✅ Index uploaded successfully.")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    sys.exit(1)

print("\n✅ All done!")