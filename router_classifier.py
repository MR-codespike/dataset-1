#!/usr/bin/env python3
"""
Router Classification — GitHub Actions version
=====================================================

What this does:
  1. Applies the HARD RULE first: if the request contains website/site/
     webpage keywords, it's classified as "website" WITHOUT calling any
     model — this was always meant to be a cheap keyword check, not a
     model decision (per the original design).
  2. For everything else, loads the router model (Qwen2.5-3B) and asks
     it to classify into exactly one of: "terminal", "code", "direct".
  3. Uses a raw GBNF grammar to constrain the model's output to ONLY
     one of those 3 words — not JSON, not a sentence, literally nothing
     else is possible to sample.
  4. Runs a batch of test prompts covering all 4 categories and reports
     accuracy against the expected answer for each.

GitHub Actions optimized — reads HF_TOKEN from environment.
"""

import subprocess
import time
import os
import sys
import re
import signal
import requests
import psutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

# ============================================================================
# CONFIG — read from environment (GitHub Secrets)
# ============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable not set.")
    sys.exit(1)

# Use GitHub workspace or current directory
BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
LLAMA_CPP_DIR = os.path.join(BASE_DIR, "llama.cpp")
MODELS_DIR = os.path.join(BASE_DIR, "gguf_models")

ROUTER_MODEL = {
    "repo_id": "MR-CODESPIKE/Qwen2.5-3B-Instruct-GGUF-Q4_K_M",
    "filename": None,  # Will be discovered
    "port": 8081,
    "context_size": 4096,
}

SERVER_STARTUP_TIMEOUT_SECONDS = 120

# The hard rule — checked BEFORE any model call
WEBSITE_KEYWORDS = [
    "website", "web site", "webpage", "web page", "landing page",
    "build me a site", "build a site", "my site", "homepage",
    "create a website", "make a website", "website for",
]

# The GBNF grammar — mechanically restricts output to exactly one of these 3 words
CLASSIFICATION_GRAMMAR = 'root ::= "terminal" | "code" | "direct"'

CLASSIFICATION_SYSTEM_PROMPT = """You are a request router for a coding assistant. Classify the user's \
request into exactly one category:

- "terminal": the user wants to run a shell/OS command, install a package, \
start a server, or otherwise interact with the command line.
- "code": the user wants code written, edited, reviewed, debugged, or \
explained — anything involving actual programming logic.
- "direct": anything else — general questions, git operations, simple \
file reads, conversation, or requests that don't need a specialist model.

Respond with ONLY the category word, nothing else."""


# ============================================================================
# STEP 0: Discover GGUF filename
# ============================================================================

def discover_model_filename():
    """Find the first .gguf file in the router repository"""
    print(f"🔍 Discovering GGUF file in {ROUTER_MODEL['repo_id']}...")
    
    try:
        files = list_repo_files(ROUTER_MODEL["repo_id"], token=HF_TOKEN)
        gguf_files = [f for f in files if f.endswith('.gguf')]
        
        if not gguf_files:
            print(f"❌ No .gguf files found in {ROUTER_MODEL['repo_id']}")
            print(f"   Available files: {', '.join(files[:5])}")
            sys.exit(1)
        
        filename = gguf_files[0]
        print(f"✅ Found: {filename}")
        ROUTER_MODEL["filename"] = filename
        return filename
        
    except Exception as e:
        print(f"❌ Error discovering model: {e}")
        sys.exit(1)


# ============================================================================
# STEP 1: Build llama-server
# ============================================================================

def build_llama_server():
    if os.path.exists(f"{LLAMA_CPP_DIR}/build/bin/llama-server"):
        print("✅ llama-server already built, skipping.")
        return True

    print("🔨 Building llama-server from source...")
    print("📥 Cloning llama.cpp ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LLAMA_CPP_DIR],
        check=True,
        capture_output=True,
    )

    print("⚙️  Configuring build (CPU-only) ...")
    subprocess.run(
        ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF"],
        cwd=LLAMA_CPP_DIR,
        check=True,
        capture_output=True,
    )

    print("🏗️  Building llama-server (this takes a few minutes) ...")
    nproc = os.cpu_count() or 2
    subprocess.run(
        ["cmake", "--build", "build", "--target", "llama-server", "-j", str(min(nproc, 4))],
        cwd=LLAMA_CPP_DIR,
        check=True,
        capture_output=True,
    )
    
    print("✅ Build complete.")
    return True


# ============================================================================
# STEP 2: Download router model
# ============================================================================

def download_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    filename = ROUTER_MODEL["filename"]
    print(f"📥 Downloading router model ({ROUTER_MODEL['repo_id']}/{filename}) ...")
    
    try:
        local_path = hf_hub_download(
            repo_id=ROUTER_MODEL["repo_id"],
            filename=filename,
            token=HF_TOKEN,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        ROUTER_MODEL["local_path"] = local_path
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  ✅ -> {local_path} ({size_mb:.1f} MB)")
        return local_path
    except Exception as e:
        print(f"  ❌ Failed to download model: {e}")
        sys.exit(1)


# ============================================================================
# STEP 3: Start the router model's llama-server
# ============================================================================

def wait_for_ready(port, timeout):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


def start_server(model_path):
    print("🚀 Starting router model server ...")
    
    process = subprocess.Popen(
        [
            f"{LLAMA_CPP_DIR}/build/bin/llama-server",
            "-m", model_path,
            "--port", str(ROUTER_MODEL["port"]),
            "--host", "127.0.0.1",
            "-c", str(ROUTER_MODEL["context_size"]),
            "--n-gpu-layers", "0",
            "--threads", str(os.cpu_count() or 2),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not wait_for_ready(ROUTER_MODEL["port"], SERVER_STARTUP_TIMEOUT_SECONDS):
        process.kill()
        raise RuntimeError("Router model failed to start")
    
    print("✅ Router model ready.\n")
    return process


# ============================================================================
# STEP 4: Hard rule — website detection (no model call)
# ============================================================================

def check_website_hard_rule(user_request):
    lowered = user_request.lower()
    for keyword in WEBSITE_KEYWORDS:
        if keyword in lowered:
            return True
    return False


# ============================================================================
# STEP 5: Grammar-constrained classification
# ============================================================================

class RoutingError(Exception):
    pass


def classify_request(user_request, max_retries=2):
    """
    Returns one of: "website", "terminal", "code", "direct".
    Website is caught by the hard rule (no model call). Everything else
    goes through the grammar-constrained router model call.
    """
    if check_website_hard_rule(user_request):
        return "website", "hard_rule"

    url = f"http://127.0.0.1:{ROUTER_MODEL['port']}/v1/chat/completions"

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                url,
                json={
                    "model": "router",
                    "messages": [
                        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_request},
                    ],
                    "max_tokens": 10,
                    "grammar": CLASSIFICATION_GRAMMAR,
                    "temperature": 0.0,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            category = data["choices"][0]["message"]["content"].strip()

            if category not in ("terminal", "code", "direct"):
                raise RoutingError(f"Grammar constraint violated, got: {category!r}")

            return category, "model"

        except (requests.exceptions.RequestException, RoutingError, KeyError) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1)
                continue

    raise RoutingError(f"Classification failed after {max_retries + 1} attempts: {last_error}")


# ============================================================================
# STEP 6: Test batch
# ============================================================================

def run_tests():
    TEST_CASES = [
        # Website cases (should all be caught by hard rule)
        ("Build me a website for a bakery", "website"),
        ("I need a landing page for my app", "website"),
        ("Can you make a homepage for my portfolio", "website"),
        ("Create a website for my coffee shop", "website"),
        
        # Terminal cases
        ("Install express using npm", "terminal"),
        ("Run the test suite", "terminal"),
        ("Start the development server", "terminal"),
        ("What's the command to list files", "terminal"),
        
        # Code cases
        ("Write a function that reverses a string", "code"),
        ("Review this code for bugs", "code"),
        ("Fix the bug in my sorting algorithm", "code"),
        ("Debug why my API returns a 500 error", "code"),
        
        # Direct cases
        ("What does this error message mean", "direct"),
        ("Commit my changes with message 'fix typo'", "direct"),
        ("What files are in this project", "direct"),
        ("Explain what a REST API is", "direct"),
    ]

    print("="*60)
    print("🧪 RUNNING ROUTING ACCURACY TEST")
    print("="*60 + "\n")

    correct = 0
    total = len(TEST_CASES)
    results = []

    for user_request, expected in TEST_CASES:
        try:
            category, source = classify_request(user_request)
            is_correct = category == expected
            correct += is_correct
            status = "✅" if is_correct else "❌"
            print(f"{status} \"{user_request}\"")
            print(f"    expected: {expected} | got: {category} ({source})")
            results.append((user_request, expected, category, is_correct))
        except RoutingError as e:
            print(f"❌ \"{user_request}\" — ROUTING FAILED: {e}")
            results.append((user_request, expected, "FAILED", False))

    print(f"\n" + "="*60)
    print(f"📊 ACCURACY: {correct}/{total} ({100*correct/total:.0f}%)")
    print("="*60)

    misses = [r for r in results if not r[3]]
    if misses:
        print(f"\n❌ {len(misses)} misclassification(s):")
        for req, expected, got, _ in misses:
            print(f"  \"{req}\" — expected {expected}, got {got}")
    else:
        print("\n✅ Perfect score! All classifications correct.")

    return correct, total, results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🚀 ROUTER CLASSIFIER — GITHUB ACTIONS")
    print("="*60 + "\n")

    # Discover model filename
    discover_model_filename()

    # Build llama-server
    if not build_llama_server():
        print("❌ Failed to build llama-server")
        sys.exit(1)

    # Download model
    model_path = download_model()

    # Start server
    server_process = start_server(model_path)

    # Run tests
    try:
        correct, total, results = run_tests()
    finally:
        # Cleanup
        print("\n🔄 Shutting down router model server...")
        server_process.send_signal(signal.SIGTERM)
        try:
            server_process.wait(timeout=15)
            print("✅ Server stopped.")
        except subprocess.TimeoutExpired:
            print("⚠️  Server didn't stop gracefully, force-killing...")
            server_process.kill()
            server_process.wait()

    # Exit with appropriate code
    if correct == total:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - correct} tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()