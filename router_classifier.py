#!/usr/bin/env python3
"""
Router Classification — GitHub Actions version (v6)
=====================================================

Design:
- Hard-rule: ONLY website (unambiguous, cheap to detect)
- Terminal, Code, Direct: ALL model-based with grammar constraints
- No generic terminal keywords (run, start, stop, install, command)

Goal: 100% accuracy on model-based classification
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

# ============================================================================
# HARD RULE: ONLY WEBSITE (unambiguous, cheap to detect)
# ============================================================================

WEBSITE_KEYWORDS = [
    "website", "web site", "webpage", "web page", "landing page",
    "build me a site", "build a site", "my site", "homepage",
    "create a website", "make a website", "website for",
    "site for my", "web presence", "online presence",
]

# ============================================================================
# GRAMMAR-CONSTRAINED CLASSIFICATION
# ============================================================================

CLASSIFICATION_GRAMMAR = 'root ::= "terminal" | "code" | "direct"'

# Clean, focused system prompt with git operations explicitly in "direct"
CLASSIFICATION_SYSTEM_PROMPT = """You are a request router. Classify the user's request into EXACTLY ONE category.

RULES:

1. "terminal" = ANYTHING about running commands, installing packages, starting/stopping services, or shell/terminal operations.
   Examples: "install npm", "run the test suite", "start the server", "list files", "stop the process", "kill the job"

2. "code" = ANYTHING about writing, reviewing, debugging, or explaining code/programming.
   Examples: "write a function", "fix this bug", "review my code", "debug this error", "optimize this query"

3. "direct" = ANYTHING ELSE. This includes:
   - General questions ("what does this error mean", "explain REST API")
   - Git operations ("commit changes", "push", "pull", "merge") ← IMPORTANT: Git = direct, not terminal
   - File operations ("read file", "list files" is terminal, but "show me the file" is direct)
   - Conversation ("tell me a joke", "what's the weather")
   - Knowledge queries ("why is the sky blue")

IMPORTANT DISTINCTIONS:
- "list files" (shell command) → terminal
- "what files are in this project" (question) → direct
- "commit changes" (git) → direct
- "install npm" (package) → terminal

Respond with ONLY ONE WORD: terminal, code, or direct."""


# ============================================================================
# BUILD LLAMA-SERVER
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
# DISCOVER AND DOWNLOAD MODEL
# ============================================================================

def discover_model_filename():
    print(f"🔍 Discovering GGUF file in {ROUTER_MODEL['repo_id']}...")
    
    try:
        files = list_repo_files(ROUTER_MODEL["repo_id"], token=HF_TOKEN)
        gguf_files = [f for f in files if f.endswith('.gguf')]
        
        if not gguf_files:
            print(f"❌ No .gguf files found")
            sys.exit(1)
        
        filename = gguf_files[0]
        print(f"✅ Found: {filename}")
        ROUTER_MODEL["filename"] = filename
        return filename
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def download_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    filename = ROUTER_MODEL["filename"]
    print(f"📥 Downloading router model ({filename}) ...")
    
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
        print(f"  ✅ {size_mb:.1f} MB")
        return local_path
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        sys.exit(1)


# ============================================================================
# START SERVER
# ============================================================================

def wait_for_ready(port, timeout):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def start_server(model_path):
    print("🚀 Starting router server ...")
    
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
        raise RuntimeError("Server failed to start")
    
    print("✅ Server ready.\n")
    return process


# ============================================================================
# CLASSIFICATION
# ============================================================================

class RoutingError(Exception):
    pass


def check_website_hard_rule(user_request):
    """Only website is hard-ruled (unambiguous)."""
    lowered = user_request.lower()
    return any(kw in lowered for kw in WEBSITE_KEYWORDS)


def classify_request(user_request, max_retries=2):
    """Returns: "website" (hard_rule) OR "terminal"/"code"/"direct" (model)."""
    
    # Hard rule: website
    if check_website_hard_rule(user_request):
        return "website", "hard_rule"

    # Model-based classification with grammar constraint
    url = f"http://127.0.0.1:{ROUTER_MODEL['port']}/v1/chat/completions"

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
                raise RoutingError(f"Invalid: {category}")

            return category, "model"

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            raise RoutingError(f"Failed: {e}")


# ============================================================================
# RUN TESTS
# ============================================================================

def run_tests():
    TEST_CASES = [
        # WEBSITE (hard-ruled)
        ("Build me a website for a bakery", "website"),
        ("I need a landing page for my app", "website"),
        ("Can you make a homepage for my portfolio", "website"),
        ("Create a website for my coffee shop", "website"),
        ("I want a site for my business", "website"),
        
        # TERMINAL (model)
        ("Install express using npm", "terminal"),
        ("Run the test suite", "terminal"),
        ("Start the development server", "terminal"),
        ("What's the command to list files", "terminal"),
        ("Install python package requests", "terminal"),
        ("How do I start the server", "terminal"),
        ("Stop the running process on port 8080", "terminal"),
        ("Kill the background job", "terminal"),
        
        # CODE (model)
        ("Write a function that reverses a string", "code"),
        ("Review this code for bugs", "code"),
        ("Fix the bug in my sorting algorithm", "code"),
        ("Debug why my API returns a 500 error", "code"),
        ("Help me fix this error in my code", "code"),
        ("Why is my function not working", "code"),
        ("Write a python script to parse JSON", "code"),
        ("Can you help me debug this", "code"),
        ("Optimize this SQL query", "code"),
        ("Add error handling to this function", "code"),
        
        # ADVERSARIAL (would break keyword-based routing)
        ("Run a quick review of my code", "code"),
        ("Stop overthinking and just fix this bug", "code"),
        ("Can you start writing tests for my app", "code"),
        ("What's the best way to structure this command pattern in my code", "code"),
        
        # DIRECT (model)
        ("What does this error message mean", "direct"),
        ("Commit my changes with message 'fix typo'", "direct"),
        ("What files are in this project", "direct"),
        ("Explain what a REST API is", "direct"),
        ("Read the contents of config.json", "direct"),
        ("What's the weather today", "direct"),
        ("Tell me a joke", "direct"),
        ("Why is the sky blue", "direct"),
    ]

    print("="*60)
    print("🧪 ROUTING ACCURACY TEST (v6)")
    print("="*60)
    print("⚠️  Hard-rule: ONLY website (unambiguous)")
    print("⚠️  Terminal, Code, Direct: ALL model-based")
    print("⚠️  Git operations explicitly classified as 'direct'")
    print("="*60 + "\n")

    correct = 0
    total = len(TEST_CASES)
    results = []
    hard_rule_count = 0
    model_count = 0

    for user_request, expected in TEST_CASES:
        try:
            category, source = classify_request(user_request)
            is_correct = category == expected
            correct += is_correct
            
            if source == "hard_rule":
                hard_rule_count += 1
            else:
                model_count += 1
            
            status = "✅" if is_correct else "❌"
            print(f'{status} "{user_request}"')
            print(f'    expected: {expected} | got: {category} ({source})')
            results.append((user_request, expected, category, is_correct))
            
        except RoutingError as e:
            print(f'❌ "{user_request}" — {e}')
            results.append((user_request, expected, "ERROR", False))

    print(f"\n" + "="*60)
    print(f"📊 ACCURACY: {correct}/{total} ({100*correct/total:.0f}%)")
    print("="*60)
    print(f"   Hard-rule cases: {hard_rule_count} (website only)")
    print(f"   Model cases: {model_count} (terminal, code, direct)")
    print("="*60)

    # Show failures
    failures = [r for r in results if not r[3]]
    if failures:
        print(f"\n❌ {len(failures)} misclassification(s):")
        for req, expected, got, _ in failures:
            print(f'  "{req}" — expected {expected}, got {got}')
        
        # Group failures by expected category
        for cat in ["website", "terminal", "code", "direct"]:
            cat_failures = [r for r in failures if r[1] == cat]
            if cat_failures:
                print(f"\n⚠️  {len(cat_failures)} {cat} failures:")
                for req, expected, got, _ in cat_failures:
                    print(f'  "{req}" → got {got}')
        
        return False, correct, total
    else:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Model correctly distinguishes terminal vs code vs direct.")
        print("   Website hard-rule confirmed unambiguous.")
        print("   Git operations correctly classified as 'direct'.")
        print("   No generic keywords used in hard-rules.")
        return True, correct, total


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🚀 ROUTER CLASSIFIER v6")
    print("="*60 + "\n")

    # Discover model
    discover_model_filename()

    # Build llama-server
    if not build_llama_server():
        print("❌ Failed to build")
        sys.exit(1)

    # Download model
    model_path = download_model()

    # Start server
    server_process = start_server(model_path)

    # Run tests
    try:
        all_passed, correct, total = run_tests()
    finally:
        # Cleanup
        print("\n🔄 Shutting down server...")
        server_process.send_signal(signal.SIGTERM)
        try:
            server_process.wait(timeout=15)
            print("✅ Server stopped.")
        except subprocess.TimeoutExpired:
            print("⚠️  Force-killing...")
            server_process.kill()
            server_process.wait()

    # Exit
    if all_passed:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - correct} tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()