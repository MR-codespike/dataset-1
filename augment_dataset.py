#!/usr/bin/env python3
"""
Classifier Training Data — Targeted Augmentation
====================================================

Downloads the existing dataset and adds targeted new examples:
- ~400 short, imperative code examples ("write a function that...")
- ~150 short terminal examples ("run the tests")

Fixes the diagnosed gap: code was missing short/plain phrasing,
causing misclassification as direct.
"""

import json, os, sys, re, time, random, requests
from huggingface_hub import hf_hub_download, HfApi, create_repo, upload_file

# ============================================================================
# CONFIG — read from environment
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"
TARGET_CODE = int(os.environ.get("TARGET_CODE", "400"))
TARGET_TERMINAL = int(os.environ.get("TARGET_TERMINAL", "150"))
EXAMPLES_PER_BATCH = 20

if not GEMINI_API_KEY or not HF_TOKEN:
    print("❌ Missing GEMINI_API_KEY or HF_TOKEN")
    sys.exit(1)

BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
OUTPUT_DIR = os.path.join(BASE_DIR, "augmented_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# GEMINI FALLBACK CHAIN (simplified from previous, only working models)
# ============================================================================

MODEL_CHAIN = [
    {"name": "gemini-3.5-flash", "rpm": 10, "rpd": 250},
    {"name": "gemini-3-flash-preview", "rpm": 10, "rpd": 250},
    {"name": "gemini-3.1-flash-lite", "rpm": 15, "rpd": 1000},
    {"name": "gemini-2.5-flash", "rpm": 10, "rpd": 250},
    {"name": "gemini-2.5-pro", "rpm": 5, "rpd": 50},
    {"name": "gemini-2.0-flash", "rpm": 15, "rpd": 1500},
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
        self.timestamps.append(time.time()); self.daily_count += 1
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
            if resp.status_code == 404:
                limiter.mark_exhausted()
                raise AllModelsExhaustedError(f"{model_name} not found")
            if resp.status_code == 429:
                if "quota" in resp.text.lower():
                    limiter.mark_exhausted()
                    raise AllModelsExhaustedError(f"{model_name} quota exhausted")
                time.sleep((2**attempt) + random.uniform(0,1)); continue
            if resp.status_code >= 500:
                time.sleep((2**attempt) + random.uniform(0,1)); continue
            resp.raise_for_status()
            data = resp.json()
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            if text.strip():
                return text
        except AllModelsExhaustedError:
            raise
        except Exception as e:
            print(f"  ⚠️ {model_name} attempt {attempt+1} failed: {e}")
            time.sleep((2**attempt) + random.uniform(0,1))
    raise RuntimeError(f"{model_name} failed after {max_retries} retries")

def call_with_fallback(prompt):
    last_error = None
    for cfg in MODEL_CHAIN:
        if not LIMITERS[cfg["name"]].can_use():
            continue
        try:
            print(f"  🔄 Trying {cfg['name']}...")
            result = call_gemini(cfg["name"], prompt)
            print(f"  ✅ {cfg['name']} succeeded")
            return result
        except AllModelsExhaustedError as e:
            print(f"  ❌ {cfg['name']} skipped: {e}")
            continue
        except Exception as e:
            print(f"  ❌ {cfg['name']} failed: {e}")
            continue
    raise AllModelsExhaustedError("All models exhausted")

# ============================================================================
# TARGETED PROMPTS — short, plain, imperative
# ============================================================================

AUGMENTATION_PROMPTS = {
    "code": """Generate {n} SHORT, PLAIN, imperative example requests a user
might type to an AI coding assistant, asking for code to be written,
reviewed, fixed, or optimized.

STRICT STYLE: each example must be under 10 words, direct and imperative.

Examples of the STYLE (do not reuse these):
- "write a function that sorts a list"
- "optimize this query"
- "add error handling here"
- "review my code"
- "fix this loop"
- "refactor this function"
- "write a unit test for this"

Return ONLY a JSON array of strings, no other text:
["example 1", "example 2", ...]""",

    "terminal": """Generate {n} SHORT, PLAIN, imperative example requests a
user might type to an AI coding assistant, asking to run/install/start/stop
something via the command line.

STRICT STYLE: each example under 8 words, direct and imperative.

Examples (do not reuse):
- "run the tests"
- "install this package"
- "start the server"
- "stop the server"
- "run the build"
- "install dependencies"

Return ONLY a JSON array of strings, no other text:
["example 1", "example 2", ...]""",
}

# ============================================================================
# GENERATE AUGMENTATION
# ============================================================================

def generate_augmentation(category, target_count):
    prompt_template = AUGMENTATION_PROMPTS[category]
    examples = set()
    attempts = 0
    max_attempts = (target_count // EXAMPLES_PER_BATCH) * 4
    print(f"\n📦 Generating {target_count} short '{category}' examples...")
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
                print(f"  ✅ [{category}] batch {attempts}: +{len(examples)-before} (total: {len(examples)}/{target_count})")
        except AllModelsExhaustedError:
            print(f"  ⚠️ [{category}] all models exhausted, stopping at {len(examples)}")
            break
        except Exception as e:
            print(f"  ⚠️ [{category}] batch {attempts} error: {e}")
            continue
    return list(examples)[:target_count]

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🚀 TARGETED AUGMENTATION – Short/Plain Examples")
    print("="*60 + "\n")

    # Download existing data
    print("📥 Downloading existing dataset...")
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

    existing = []
    with open(data_path) as f:
        for line in f:
            existing.append(json.loads(line))
    print(f"✅ Loaded {len(existing):,} existing examples")

    # Generate new short examples
    new_rows = []
    for cat, target in [("code", TARGET_CODE), ("terminal", TARGET_TERMINAL)]:
        examples = generate_augmentation(cat, target)
        print(f"  Final for {cat}: {len(examples)}")
        for ex in examples:
            new_rows.append({"text": ex, "label": cat})

    # Merge and dedupe
    all_rows = existing + new_rows
    seen = set()
    deduped = []
    for row in all_rows:
        key = row["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    print(f"\n📊 Augmented dataset: {len(deduped):,}")
    counts = {cat: sum(1 for r in deduped if r["label"] == cat) for cat in ["terminal","code","direct"]}
    for cat, cnt in counts.items():
        print(f"  {cat}: {cnt:,}")

    # Save and upload
    out_path = os.path.join(OUTPUT_DIR, "classifier_training_data.jsonl")
    with open(out_path, "w") as f:
        for row in deduped:
            f.write(json.dumps(row) + "\n")

    print("\n📤 Uploading augmented dataset to HF...")
    api = HfApi(token=HF_TOKEN)
    create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, repo_type=HF_REPO_TYPE, exist_ok=True)
    upload_file(
        path_or_fileobj=out_path,
        path_in_repo="classifier_training_data/classifier_training_data.jsonl",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        commit_message=f"Augment with short/plain phrasing (+{len(new_rows)} examples)",
    )
    print("✅ Upload complete!")

if __name__ == "__main__":
    main()