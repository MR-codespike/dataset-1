#!/usr/bin/env python3
"""
List all Hugging Face repositories (datasets, models, spaces) that you have access to.
Useful to verify your HF_TOKEN works and see all your repos.

GitHub Actions optimized version – reads HF_TOKEN from environment.
"""

import os
import sys
import json
from huggingface_hub import HfApi

# Read from environment (GitHub Secrets)
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable not set.")
    print("⚠️  Falling back to token from huggingface-cli login...")
    # Try to use the token from huggingface-cli login (if any)
    try:
        from huggingface_hub import HfFolder
        HF_TOKEN = HfFolder.get_token()
    except:
        pass

if not HF_TOKEN:
    print("❌ No token found. Please set HF_TOKEN environment variable.")
    sys.exit(1)

print(f"🔑 Token found (starts with: {HF_TOKEN[:8]}...)")

# ============================================================================
# List all repositories
# ============================================================================

api = HfApi(token=HF_TOKEN)

print("\n📊 Fetching all your Hugging Face repositories...\n")

# Get all repos (datasets, models, spaces)
all_repos = []

# The HfApi has methods for each repo type
repo_types = [
    ("dataset", api.list_datasets),
    ("model", api.list_models),
    ("space", api.list_spaces)
]

for repo_type, list_func in repo_types:
    try:
        # Use list_func with token
        for repo in list_func(token=HF_TOKEN):
            # Get additional info if available
            repo_id = repo.repo_id if hasattr(repo, 'repo_id') else repo.id
            # Some versions use different attribute names
            repo_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id
            
            repo_data = {
                "name": repo_name,
                "full_name": repo_id,
                "type": repo_type,
                "private": getattr(repo, 'private', False),
                "downloads": getattr(repo, 'downloads', 0),
                "likes": getattr(repo, 'likes', 0),
                "created_at": str(getattr(repo, 'created_at', 'N/A')),
                "url": f"https://huggingface.co/{repo_id}"
            }
            all_repos.append(repo_data)
    except Exception as e:
        print(f"⚠️  Error fetching {repo_type}s: {e}")

# Sort by repo type and name
all_repos.sort(key=lambda x: (x["type"], x["full_name"]))

# ============================================================================
# Display results
# ============================================================================

if not all_repos:
    print("No repositories found. Have you created any?")
    sys.exit(0)

print(f"✅ Found {len(all_repos)} repositories:\n")

# Group by type
datasets = [r for r in all_repos if r["type"] == "dataset"]
models = [r for r in all_repos if r["type"] == "model"]
spaces = [r for r in all_repos if r["type"] == "space"]

def print_repo_group(repos, type_name):
    if not repos:
        return
    print(f"\n{'='*50}")
    print(f"📁 {type_name.upper()}S ({len(repos)})")
    print('='*50)
    for repo in repos:
        private_marker = "🔒" if repo["private"] else "🌐"
        print(f"{private_marker} {repo['full_name']}")
        print(f"   URL: {repo['url']}")
        if repo['downloads'] > 0 or repo['likes'] > 0:
            print(f"   📥 {repo['downloads']} downloads | ❤️ {repo['likes']} likes")
        print(f"   📅 Created: {repo['created_at']}")

print_repo_group(datasets, "Dataset")
print_repo_group(models, "Model")
print_repo_group(spaces, "Space")

# ============================================================================
# Save paths to a file (useful for scripts)
# ============================================================================

print(f"\n{'='*50}")
print("📝 Full repository list (raw data):")
print('='*50)

# Print as JSON for easy parsing
print(json.dumps(all_repos, indent=2, default=str))

# Also save to file if running in GitHub Actions
if os.environ.get("GITHUB_WORKSPACE"):
    output_path = os.path.join(os.environ["GITHUB_WORKSPACE"], "hf_repos.json")
    with open(output_path, "w") as f:
        json.dump(all_repos, f, indent=2, default=str)
    print(f"\n✅ Saved to {output_path}")

# ============================================================================
# Print summary of dataset repos (most relevant for your template library)
# ============================================================================

dataset_repos = [r for r in all_repos if r["type"] == "dataset"]
if dataset_repos:
    print(f"\n{'='*50}")
    print("📦 DATASET REPOSITORIES (useful for your template library):")
    print('='*50)
    for repo in dataset_repos:
        print(f"  - {repo['full_name']}")
        print(f"    Path: {repo['url']}")
        if not repo["private"]:
            print(f"    (public) ✅")
        else:
            print(f"    (private) 🔒")

# ============================================================================
# Summary of all repo paths (easy to copy)
# ============================================================================

print(f"\n{'='*50}")
print("📂 ALL REPOSITORY PATHS (copy these):")
print('='*50)
for repo in all_repos:
    print(f"{repo['full_name']}")

print("\n✅ Done!")