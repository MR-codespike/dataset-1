#!/usr/bin/env python3
"""
Generate and Clean Training Data
================================
Generates 5,000 examples per category (15,000 total), cleans/deduplicates,
and uploads to Hugging Face. No training - just data preparation.

BEFORE YOU RUN:
  - Set GEMINI_API_KEY, HF_TOKEN, HF_REPO_ID in environment
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
from collections import defaultdict
from huggingface_hub import HfApi, create_repo, upload_file

# ============================================================================
# CONFIG — read from environment
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

BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
OUTPUT_DIR = os.path.join(BASE_DIR, "classifier_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXAMPLES_PER_CATEGORY = 5000  # 5k per category
EXAMPLES_PER_BATCH = 25

# ============================================================================
# GEMINI FALLBACK CHAIN
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
            continue
    raise AllModelsExhaustedError(f"All models exhausted: {last_error}")


# ============================================================================
# CATEGORY PROMPTS — includes hard cases that failed
# ============================================================================

CATEGORY_PROMPTS = {
    "terminal": """Generate {n} diverse example sentences a user might type to an
AI coding assistant, where the request is about running shell commands,
installing packages, starting/stopping services, or other command-line
operations.

INCLUDE examples that sound like code requests but ARE terminal operations:
- "run the test suite" (terminal command)
- "run npm start" (terminal)
- "start the server" (terminal)
- "stop the process" (terminal)

Vary tone (casual, formal, terse, verbose), phrasing, and specific tools.
Include typos and informal grammar.

Return ONLY a JSON array of strings, no other text:
["example 1", "example 2", ...]""",

    "code": """Generate {n} diverse example sentences a user might type to an
AI coding assistant, where the request is about writing, reviewing,
debugging, fixing, or explaining code/programming logic.

CRITICAL: Include examples that use common verbs like "run", "start", "stop"
in a CODE context:
- "run a quick review of my code" (code review)
- "run this function and tell me what it does" (code)
- "start writing tests for my app" (code)
- "stop overthinking and fix this bug" (code)
- "run through my code and find bugs" (code)

Vary tone, phrasing, and programming languages/contexts.

Return ONLY a JSON array of strings, no other text:
["example 1", "example 2", ...]""",

    "direct": """Generate {n} diverse example sentences a user might type to an
AI coding assistant, where the request is EITHER a general knowledge
question, a git operation, a simple file read/question, or conversation.

INCLUDE examples that sound like code/terminal but ARE direct:
- "what does this error message mean" (direct - explanation)
- "why is my function not working" (direct - explanation, not code)
- "what's the best way to structure this" (direct - advice)
- "optimize this SQL query" (direct - advice, not code)
- "add error handling to this function" (direct - advice, not code)

Vary tone and phrasing. Include git operations and file questions.

Return ONLY a JSON array of strings, no other text:
["example 1", "example 2", ...]""",
}


def generate_category(category, target_count):
    prompt_template = CATEGORY_PROMPTS[category]
    examples = set()
    attempts = 0
    max_attempts = (target_count // EXAMPLES_PER_BATCH) * 3

    print(f"\n📦 Generating {target_count:,} examples for '{category}'...")

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
                print(f"  ✅ [{category}] batch {attempts}: +{len(examples) - before} new (total: {len(examples):,}/{target_count:,})")
        except AllModelsExhaustedError:
            print(f"  ⚠️ [{category}] all models exhausted, stopping with {len(examples)} examples")
            break
        except json.JSONDecodeError as e:
            print(f"  ⚠️ [{category}] batch {attempts} JSON decode error")
            continue
        except Exception as e:
            print(f"  ⚠️ [{category}] batch {attempts} failed: {e}")
            continue

    return list(examples)[:target_count]


# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def validate_text(text):
    """Validate that text is a reasonable example."""
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 3:  # Too short
        return False
    if len(text) > 200:  # Too long
        return False
    # Must contain at least one alphabetic character
    if not re.search(r'[a-zA-Z]', text):
        return False
    return True


def clean_text(text):
    """Clean text by normalizing whitespace and basic cleaning."""
    text = text.strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing quotes
    text = re.sub(r'^[\'"](.*)[\'"]$', r'\1', text)
    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def deduplicate_data(data):
    """Remove duplicates from dataset."""
    seen = set()
    unique = []
    duplicate_count = 0
    
    for item in data:
        text = item["text"].strip().lower()
        if text not in seen:
            seen.add(text)
            unique.append(item)
        else:
            duplicate_count += 1
    
    print(f"  🧹 Removed {duplicate_count} duplicates")
    return unique


def validate_and_clean_dataset(data):
    """Full cleaning pipeline."""
    print(f"\n🧹 Cleaning {len(data):,} examples...")
    
    # Step 1: Validate
    valid = []
    invalid_count = 0
    for item in data:
        if validate_text(item["text"]):
            # Clean the text
            item["text"] = clean_text(item["text"])
            valid.append(item)
        else:
            invalid_count += 1
    
    print(f"  ✅ {len(valid)} valid, {invalid_count} invalid removed")
    
    # Step 2: Deduplicate
    unique = deduplicate_data(valid)
    
    # Step 3: Check category balance
    category_counts = defaultdict(int)
    for item in unique:
        category_counts[item["label"]] += 1
    
    print(f"  📊 Category breakdown:")
    for cat, count in sorted(category_counts.items()):
        print(f"     {cat}: {count:,}")
    
    return unique


def load_existing_data():
    """Load existing data from Hugging Face if available."""
    try:
        from huggingface_hub import hf_hub_download
        data_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="classifier_training_data/classifier_training_data.jsonl",
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN,
        )
        data = []
        with open(data_path) as f:
            for line in f:
                row = json.loads(line)
                data.append(row)
        print(f"📂 Loaded {len(data):,} existing examples")
        return data
    except:
        print("📂 No existing data found, starting fresh")
        return []


# ============================================================================
# SAVE AND UPLOAD
# ============================================================================

def save_data(data, output_path):
    """Save data to JSONL file."""
    with open(output_path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
    print(f"\n💾 Saved {len(data):,} examples to {output_path}")


def upload_to_huggingface(data, output_path):
    """Upload data to Hugging Face."""
    print(f"\n📤 Uploading to {HF_REPO_ID}...")
    
    try:
        api = HfApi(token=HF_TOKEN)
        create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, repo_type=HF_REPO_TYPE, exist_ok=True)
        
        # Upload training data
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        upload_file(
            path_or_fileobj=output_path,
            path_in_repo="classifier_training_data/classifier_training_data.jsonl",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            commit_message=f"Add cleaned training data ({len(data):,} examples) - {timestamp}",
        )
        
        # Create README
        readme = f"""# Classifier Training Data

## Overview
- **Total examples:** {len(data):,}
- **Generated:** {timestamp}
- **Format:** JSONL

## Categories
"""
        category_counts = defaultdict(int)
        for item in data:
            category_counts[item["label"]] += 1
        
        for cat, count in sorted(category_counts.items()):
            readme += f"- **{cat}:** {count:,}\n"
        
        readme += "\n## Format\n"
        readme += '```json\n{"text": "user request", "label": "category"}\n```\n'
        readme += "\n## Usage\n"
        readme += "```python\nimport json\n\nwith open('classifier_training_data.jsonl', 'r') as f:\n"
        readme += "    for line in f:\n"
        readme += "        data = json.loads(line)\n"
        readme += "        text, label = data['text'], data['label']\n"
        readme += "```\n"
        
        # Upload README
        readme_path = os.path.join(os.path.dirname(output_path), "README.md")
        with open(readme_path, "w") as f:
            f.write(readme)
        
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="classifier_training_data/README.md",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            commit_message=f"Add README for training data",
        )
        
        print("✅ Upload complete!")
        print(f"   → https://huggingface.co/{HF_REPO_ID}/tree/main/classifier_training_data")
        return True
        
    except Exception as e:
        print(f"⚠️ Upload failed: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 GENERATE AND CLEAN TRAINING DATA")
    print("=" * 60)
    print(f"  Target per category: {EXAMPLES_PER_CATEGORY:,}")
    print(f"  Total target: {EXAMPLES_PER_CATEGORY * 3:,}")
    print("=" * 60 + "\n")

    # Load existing data
    existing_data = load_existing_data()
    
    # Generate new data
    all_data = []
    for category in ["terminal", "code", "direct"]:
        examples = generate_category(category, EXAMPLES_PER_CATEGORY)
        for ex in examples:
            all_data.append({"text": ex, "label": category})
        print(f"  ✅ Final count for {category}: {len(examples):,}")
    
    # Combine with existing data
    if existing_data:
        all_data.extend(existing_data)
        print(f"\n📊 Combined: {len(all_data):,} total examples")
    
    # Clean the data
    cleaned_data = validate_and_clean_dataset(all_data)
    
    # Save locally
    output_path = os.path.join(OUTPUT_DIR, "classifier_training_data.jsonl")
    save_data(cleaned_data, output_path)
    
    # Upload to Hugging Face
    upload_to_huggingface(cleaned_data, output_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total examples: {len(cleaned_data):,}")
    
    category_counts = defaultdict(int)
    for item in cleaned_data:
        category_counts[item["label"]] += 1
    
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count:,}")
    
    print("=" * 60)
    print("\n✅ Ready for training!")


if __name__ == "__main__":
    main()