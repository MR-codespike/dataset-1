#!/usr/bin/env python3
"""
List ALL repositories YOU own on Hugging Face (not all repos in the hub).
This only lists repositories where you are the author/owner.
"""

import os
import sys
import json
from huggingface_hub import HfApi

# Read from environment
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable not set.")
    sys.exit(1)

print(f"🔑 Token found (starts with: {HF_TOKEN[:8]}...)")

# ============================================================================
# Get your user info
# ============================================================================

api = HfApi(token=HF_TOKEN)

print("📊 Fetching your Hugging Face repositories...")

# Get your username from the token
try:
    user_info = api.whoami(token=HF_TOKEN)
    username = user_info.get("name")
    print(f"👤 Logged in as: {username}\n")
except Exception as e:
    print(f"⚠️  Could not get username: {e}")
    username = None

# ============================================================================
# List ONLY your repositories
# ============================================================================

all_repos = []

# For each repo type, list ONLY repos where you are the author
repo_types = [
    ("dataset", api.list_datasets, "datasets"),
    ("model", api.list_models, "models"),
    ("space", api.list_spaces, "spaces")
]

for repo_type, list_func, display_name in repo_types:
    try:
        print(f"  Fetching your {display_name}...")
        # Get repos with author filter
        repos = list_func(
            token=HF_TOKEN,
            author=username if username else None
        )
        
        count = 0
        for repo in repos:
            # Get repo ID properly
            repo_id = getattr(repo, 'repo_id', getattr(repo, 'id', None))
            if not repo_id:
                continue
                
            repo_data = {
                "name": repo_id.split('/')[-1] if '/' in repo_id else repo_id,
                "full_name": repo_id,
                "type": repo_type,
                "private": getattr(repo, 'private', False),
                "downloads": getattr(repo, 'downloads', 0),
                "likes": getattr(repo, 'likes', 0),
                "url": f"https://huggingface.co/{repo_id}"
            }
            all_repos.append(repo_data)
            count += 1
        
        print(f"    Found {count} {display_name}")
    except Exception as e:
        print(f"⚠️  Error fetching {display_name}: {e}")

# ============================================================================
# Display results
# ============================================================================

print(f"\n{'='*50}")
print(f"✅ Found {len(all_repos)} repositories in total")
print('='*50)

if not all_repos:
    print("\nNo repositories found. You haven't created any yet.")
    sys.exit(0)

# Group and display
for repo_type in ["dataset", "model", "space"]:
    repos = [r for r in all_repos if r["type"] == repo_type]
    if not repos:
        continue
    
    type_display = repo_type.upper()
    if repo_type == "space":
        type_display = "SPACES"
    elif repo_type == "dataset":
        type_display = "DATASETS"
    elif repo_type == "model":
        type_display = "MODELS"
    
    print(f"\n📁 {type_display} ({len(repos)})")
    print("-" * 40)
    
    for repo in repos:
        private_marker = "🔒" if repo["private"] else "🌐"
        print(f"{private_marker} {repo['full_name']}")
        print(f"   URL: {repo['url']}")
        if repo['downloads'] > 0 or repo['likes'] > 0:
            print(f"   📥 {repo['downloads']} downloads | ❤️ {repo['likes']} likes")
        print()

# ============================================================================
# Save to file (for GitHub Actions artifact)
# ============================================================================

if os.environ.get("GITHUB_WORKSPACE"):
    output_path = os.path.join(os.environ["GITHUB_WORKSPACE"], "hf_repos.json")
    with open(output_path, "w") as f:
        json.dump(all_repos, f, indent=2, default=str)
    print(f"✅ Saved to {output_path}")

# ============================================================================
# Quick list of all repo paths (easy to copy)
# ============================================================================

print(f"\n{'='*50}")
print("📂 YOUR REPOSITORY PATHS:")
print('='*50)
for repo in all_repos:
    print(f"  {repo['full_name']}")

print("\n✅ Done!")