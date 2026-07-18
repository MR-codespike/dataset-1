#!/usr/bin/env python3
"""
List all Hugging Face repositories (datasets, models, spaces) that you have access to.
Useful to verify your HF_TOKEN works and see all your repos.

GitHub Actions optimized version – reads HF_TOKEN from environment.
"""

import os
import sys
from huggingface_hub import HfApi, list_repos

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
for repo_type in ["dataset", "model", "space"]:
    try:
        repos = list_repos(token=HF_TOKEN, repo_type=repo_type)
        for repo in repos:
            all_repos.append({
                "name": repo.name,
                "type": repo_type,
                "full_name": repo.repo_id,
                "private": repo.private,
                "downloads": repo.downloads if hasattr(repo, 'downloads') else 0,
                "likes": repo.likes if hasattr(repo, 'likes') else 0,
                "created_at": repo.created_at if hasattr(repo, 'created_at') else "N/A"
            })
    except Exception as e:
        print(f"⚠️  Error fetching {repo_type}s: {e}")

# Sort by repo type and name
all_repos.sort(key=lambda x: (x["type"], x["name"]))

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
        print(f"   📥 {repo['downloads']} downloads | ❤️ {repo['likes']} likes | 📅 {repo['created_at']}")

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
import json
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
        if not repo["private"]:
            print(f"    (public)")

print("\n✅ Done!")