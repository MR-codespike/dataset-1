#!/usr/bin/env python3
"""
Classifier Training Data Generator
=====================================

Generates diverse, labeled example requests for the terminal/code/direct
classifier, using the same 7-model Gemini free-tier fallback chain built
for template generation. Explicitly includes the hard boundary cases
discovered during router testing (git ops, file questions, code requests
phrased with common verbs like "run"/"stop"/"start") as correctly-labeled
training examples — turning prior failures into training signal.

Output is pushed directly to Hugging Face dataset repository.
GitHub Actions optimized — reads keys from environment variables.
"""

import json
import os
import sys
import re
import time
import random
import requests
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, create_repo

# ============================================================================
# CONFIG — read from environment (GitHub Secrets)
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable not set.")
    sys.exit(1)

# Use GitHub workspace or current directory
BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
OUTPUT_DIR = os.path.join(BASE_DIR, "classifier_data")
EXAMPLES_PER_CATEGORY = 1000
EXAMPLES_PER_BATCH = 25   # ~40 API calls per category, cheap and fast

# ============================================================================
# GEMINI FALLBACK CHAIN (same as template generator)
# ============================================================================

MODEL_CHAIN = [
    {"name": "gemini-3.5-flash", "rpm": 10, "rpd": 250},
    {"name": "gemini-3-flash-preview", "rpm": 10, "rpd": 250},
    {"name": "gemini-2.5-pro", "rpm": 5, "rpd": 50},
    {"name": "gemini-3.1-flash-lite", "rpm": 15, "rpd": 1000},
    {"name": "gemini-2.5-flash", "rpm": 10, "rpd": 250},
    {"name": "gemini-2.5-flash-lite", "rpm": 15, "rpd": 1000},
    {"name": "gemini-2.5-flash-lite-preview-09-2025", "rpm": 15, "rpd": 1000},
]
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class ModelRateLimiter:
    def __init__(self, name, rpm, rpd):
        self.name, self.rpm, self.rpd = name, rpm, rpd
        self.timestamps, self.daily_count, self.exhausted = [], 0, False

    def can_use(self):
        return not self.exhausted and self.daily_count < self.rpd

    def wait_for_slot(self):
        while True:
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < 60]
            if len(self.timestamps) < self.rpm:
                return
            time.sleep(max(60 - (now - self.timestamps[0]) + 0.5, 0.5))

    def record(self):
        self.timestamps.append(time.time())
        self.daily_count += 1

    def mark_exhausted(self):
        self.exhausted = True


LIMITERS = {m["name"]: ModelRateLimiter(**m) for m in MODEL_CHAIN}


class AllModelsExhaustedError(Exception):
    pass


def call_gemini(model_name, prompt, max_retries=3):
    limiter = LIMITERS[model_name]
    url = f"{GEMINI_API_BASE}/{model_name}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 2048, "responseMimeType": "application/json"},
    }
    for attempt in range(max_retries):
        if not limiter.can_use():
            raise AllModelsExhaustedError(f"{model_name} exhausted")
        limiter.wait_for_slot()
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            limiter.record()
            if resp.status_code == 429:
                if "quota" in resp.text.lower():
                    limiter.mark_exhausted()
                    raise AllModelsExhaustedError(f"{model_name} quota exhausted")
                time.sleep((2 ** attempt) + random.uniform(0, 1))
                continue
            if resp.status_code >= 500:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            if text.strip():
                return text
        except Exception as e:
            print(f"  ⚠️ {model_name} attempt {attempt+1} failed: {e}")
            time.sleep((2 ** attempt) + random.uniform(0, 1))
            continue
    raise RuntimeError(f"{model_name} failed after {max_retries} retries")


def call_with_fallback(prompt):
    last_error = None
    for cfg in MODEL_CHAIN:
        if not LIMITERS[cfg["name"]].can_use():
            continue
        try:
            return call_gemini(cfg["name"], prompt)
        except Exception as e:
            last_error = e
            print(f"  ⚠️ {cfg['name']} failed: {e}")
            continue
    raise AllModelsExhaustedError(f"All models exhausted: {last_error}")


# ============================================================================
# CATEGORY DEFINITIONS — includes known hard/adversarial cases explicitly
# ============================================================================

CATEGORY_PROMPTS = {
    "terminal": """Generate {n} diverse example sentences a user might type to an
AI coding assistant, where the request is about running shell commands,
installing packages, starting/stopping services, or other command-line
operations. Vary tone (casual, formal, terse, verbose), phrasing, and
specific tools/commands mentioned. Include some with typos or informal
grammar. Do NOT include any that are actually about writing/reviewing code,
asking general questions, or git operations — those are different categories.

Return ONLY a JSON array of strings, no other text:
["example 1", "example 2", ...]""",

    "code": """Generate {n} diverse example sentences a user might type to an
AI coding assistant, where the request is about writing, reviewing,
debugging, fixing, or explaining code/programming logic. Vary tone,
phrasing, and programming languages/contexts mentioned. 

IMPORTANT: include a good portion (~20%) of examples that use common
everyday verbs like "run", "start", "stop" in a CODE context, not a
terminal context, e.g. "run a quick review of my code", "stop overthinking
and fix this bug", "can you start writing tests for my app" — these are
genuinely code requests despite containing terminal-sounding words, and
the model needs to learn this distinction, not just keyword-match.

Return ONLY a JSON array of strings, no other text:
["example 1", "example 2", ...]""",

    "direct": """Generate {n} diverse example sentences a user might type to an
AI coding assistant, where the request is EITHER a general knowledge
question, a git operation (commit/push/pull/merge/branch), a simple file
read/question (not a shell command), or general conversation. Vary tone
and phrasing.

IMPORTANT: include a good portion (~30%) of examples that are git
operations (e.g. "commit my changes", "push to main", "merge this branch")
and file-related QUESTIONS phrased naturally (e.g. "what files are in this
project", "show me what's in config.json", "what does this file do") —
these should be DIRECT, not terminal, even though they involve files/git,
because they're questions or git-specific operations, not raw shell
commands.

Return ONLY a JSON array of strings, no other text:
["example 1", "example 2", ...]""",
}


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_category(category, target_count):
    prompt_template = CATEGORY_PROMPTS[category]
    examples = set()
    attempts = 0
    max_attempts = (target_count // EXAMPLES_PER_BATCH) * 3

    print(f"\n📦 Generating {target_count} examples for '{category}'...")

    while len(examples) < target_count and attempts < max_attempts:
        attempts += 1
        prompt = prompt_template.format(n=EXAMPLES_PER_BATCH)
        try:
            raw = call_with_fallback(prompt)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
                cleaned = re.sub(r"\n```$", "", cleaned)
            batch = json.loads(cleaned)
            if isinstance(batch, list):
                before = len(examples)
                examples.update(x.strip() for x in batch if isinstance(x, str) and x.strip())
                print(f"  ✅ [{category}] batch {attempts}: +{len(examples) - before} new (total: {len(examples)}/{target_count})")
        except AllModelsExhaustedError:
            print(f"  ⚠️ [{category}] all models exhausted, stopping with {len(examples)} examples")
            break
        except json.JSONDecodeError as e:
            print(f"  ⚠️ [{category}] batch {attempts} JSON decode error: {e}")
            print(f"     Raw: {raw[:200]}...")
            continue
        except Exception as e:
            print(f"  ⚠️ [{category}] batch {attempts} failed: {e}")
            continue

    return list(examples)[:target_count]


# ============================================================================
# UPLOAD TO HUGGING FACE
# ============================================================================

def upload_to_huggingface(file_path, repo_id, token, repo_type="dataset"):
    """Upload the training data file to Hugging Face."""
    print(f"\n📤 Uploading to {repo_id}...")
    
    try:
        # Ensure repo exists
        api = HfApi(token=token)
        create_repo(repo_id=repo_id, token=token, repo_type=repo_type, exist_ok=True)
        print(f"  ✅ Repository ready: {repo_id}")
        
        # Upload file
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"classifier_training_data/classifier_training_data.jsonl",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Add classifier training data ({timestamp})",
        )
        
        # Also upload a metadata file describing the dataset
        metadata = {
            "description": "Training data for router classifier",
            "categories": ["terminal", "code", "direct"],
            "total_examples": 3000,
            "examples_per_category": 1000,
            "format": "JSONL - each line is {'text': '...', 'label': '...'}",
            "generated_by": "classifier_training_data_generator",
            "generated_at": timestamp,
            "source": "Gemini API fallback chain",
        }
        
        # Create a metadata file
        metadata_path = os.path.join(os.path.dirname(file_path), "README.md")
        with open(metadata_path, "w") as f:
            f.write("# Classifier Training Data\n\n")
            f.write("## Overview\n")
            f.write(f"- **Total examples:** {metadata['total_examples']:,}\n")
            f.write(f"- **Categories:** {', '.join(metadata['categories'])}\n")
            f.write(f"- **Generated:** {metadata['generated_at']}\n")
            f.write(f"- **Format:** JSONL\n\n")
            f.write("## Categories\n")
            f.write("| Category | Examples | Description |\n")
            f.write("|----------|----------|-------------|\n")
            for cat in metadata['categories']:
                f.write(f"| {cat} | 1,000 | Requests about {cat} operations |\n")
            f.write("\n## Usage\n")
            f.write("```python\n")
            f.write("import json\n\n")
            f.write("# Load the data\n")
            f.write("with open('classifier_training_data.jsonl', 'r') as f:\n")
            f.write("    for line in f:\n")
            f.write("        data = json.loads(line)\n")
            f.write("        text, label = data['text'], data['label']\n")
            f.write("```\n")
        
        # Upload metadata too
        api.upload_file(
            path_or_fileobj=metadata_path,
            path_in_repo="classifier_training_data/README.md",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Add classifier training data README",
        )
        
        print(f"  ✅ Training data uploaded to: https://huggingface.co/{repo_id}/tree/main/classifier_training_data")
        return True
        
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 CLASSIFIER TRAINING DATA GENERATOR")
    print("=" * 60)
    print(f"  Examples per category: {EXAMPLES_PER_CATEGORY}")
    print(f"  Examples per batch: {EXAMPLES_PER_BATCH}")
    print(f"  Total target: {EXAMPLES_PER_CATEGORY * 3:,} examples")
    print(f"  Output repo: {HF_REPO_ID}")
    print("=" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = []

    # Generate data for each category
    for category in ["terminal", "code", "direct"]:
        examples = generate_category(category, EXAMPLES_PER_CATEGORY)
        print(f"  Final count for {category}: {len(examples)}")
        for ex in examples:
            all_data.append({"text": ex, "label": category})

    # Save locally
    output_path = os.path.join(OUTPUT_DIR, "classifier_training_data.jsonl")
    with open(output_path, "w") as f:
        for row in all_data:
            f.write(json.dumps(row) + "\n")

    # Stats
    print("\n" + "=" * 60)
    print("📊 GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {len(all_data):,}")
    
    # Count by category
    counts = {}
    for item in all_data:
        label = item.get("label", "unknown")
        counts[label] = counts.get(label, 0) + 1
    for label, count in counts.items():
        print(f"  {label}: {count:,} ({count/len(all_data)*100:.1f}%)")
    
    print(f"\n💾 Saved locally to: {output_path}")

    # Upload to Hugging Face
    upload_success = upload_to_huggingface(output_path, HF_REPO_ID, HF_TOKEN, HF_REPO_TYPE)
    
    if upload_success:
        print("\n✅ All done! Training data is now available on Hugging Face.")
        print(f"   → https://huggingface.co/{HF_REPO_ID}/tree/main/classifier_training_data")
    else:
        print("\n⚠️ Upload failed. Data is still available locally.")
        print(f"   → {output_path}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()