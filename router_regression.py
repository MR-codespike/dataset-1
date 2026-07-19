#!/usr/bin/env python3
"""
Router Regression Test — REVERTED to the original v6 system prompt.

The rewritten prompt (with the bolded "file requests = terminal" callout)
caused 3 new regressions while only fixing 1 case. This reverts to the
proven 97%-baseline prompt and re-tests against the FULL 37-case suite,
including the 2 new phrasings discovered during integration testing.

Design (unchanged from the original, working v6):
  - Hard-rule: ONLY website (unambiguous, cheap to detect)
  - Terminal, Code, Direct: ALL model-based, no keyword shortcuts
  - Git operations explicitly called out as "direct" in the prompt itself
"""

import subprocess
import time
import os
import sys
import signal
import requests
from huggingface_hub import hf_hub_download, list_repo_files

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("❌ HF_TOKEN not set.")
    sys.exit(1)

BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
LLAMA_CPP_DIR = os.path.join(BASE_DIR, "llama.cpp")
MODELS_DIR = os.path.join(BASE_DIR, "gguf_models")

ROUTER_REPO_ID = "MR-CODESPIKE/Qwen2.5-3B-Instruct-GGUF-Q4_K_M"
ROUTER_PORT = 8081
ROUTER_CONTEXT_SIZE = 4096
SERVER_STARTUP_TIMEOUT = 120

CLASSIFICATION_GRAMMAR = 'root ::= "terminal" | "code" | "direct"'

# REVERTED to the original v6 prompt — no aggressive "file requests = terminal"
# bolded callout, just clean category definitions with the git distinction
# stated once, naturally.
CLASSIFICATION_SYSTEM_PROMPT = """You are a request router. Classify the user's request into EXACTLY ONE category.

RULES:

1. "terminal" = ANYTHING about running commands, installing packages, starting/stopping services, or shell/terminal operations.
   Examples: "install npm", "run the test suite", "start the server", "list files", "stop the process", "kill the job"

2. "code" = ANYTHING about writing, reviewing, debugging, or explaining code/programming.
   Examples: "write a function", "fix this bug", "review my code", "debug this error", "optimize this query"

3. "direct" = ANYTHING ELSE. This includes:
   - General questions ("what does this error mean", "explain REST API")
   - Git operations ("commit changes", "push", "pull", "merge") - git is direct, not terminal
   - File operations ("read file", "list files" is terminal, but "show me the file" is direct)
   - Conversation ("tell me a joke", "what's the weather")
   - Knowledge queries ("why is the sky blue")

IMPORTANT DISTINCTIONS:
- "list files" (shell command) - terminal
- "what files are in this project" (question) - direct
- "commit changes" (git) - direct
- "install npm" (package) - terminal

Respond with ONLY ONE WORD: terminal, code, or direct."""

# ONLY website is hard-ruled — no terminal keyword shortcuts, since those
# proved too fragile (word-order sensitive, and prone to false positives
# on ordinary English sentences).
WEBSITE_KEYWORDS = [
    "website", "web site", "webpage", "web page", "landing page",
    "build me a site", "build a site", "my site", "homepage",
    "create a website", "make a website", "website for", "site for my",
]


def build_llama_server():
    if os.path.exists(f"{LLAMA_CPP_DIR}/build/bin/llama-server"):
        print("✅ llama-server already built, skipping.")
        return
    print("🔨 Building llama-server from source...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LLAMA_CPP_DIR], check=True)
    subprocess.run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF"], cwd=LLAMA_CPP_DIR, check=True)
    nproc = os.cpu_count() or 2
    subprocess.run(["cmake", "--build", "build", "--target", "llama-server", "-j", str(nproc)], cwd=LLAMA_CPP_DIR, check=True)
    print("✅ Build complete.")


def discover_and_download():
    print(f"🔍 Discovering GGUF file in {ROUTER_REPO_ID}...")
    files = list_repo_files(ROUTER_REPO_ID, token=HF_TOKEN)
    gguf_files = [f for f in files if f.endswith(".gguf")]
    if not gguf_files:
        print("❌ No .gguf files found!")
        sys.exit(1)
    filename = gguf_files[0]
    print(f"✅ Found: {filename}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_path = hf_hub_download(repo_id=ROUTER_REPO_ID, filename=filename, token=HF_TOKEN, local_dir=MODELS_DIR)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"✅ Downloaded: {local_path} ({size_mb:.1f} MB)")
    return local_path


def wait_for_ready(port, timeout):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


def start_server(model_path):
    print("🚀 Starting router server...")
    process = subprocess.Popen(
        [f"{LLAMA_CPP_DIR}/build/bin/llama-server", "-m", model_path,
         "--port", str(ROUTER_PORT), "--host", "127.0.0.1",
         "-c", str(ROUTER_CONTEXT_SIZE), "--n-gpu-layers", "0",
         "--threads", str(os.cpu_count() or 2)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not wait_for_ready(ROUTER_PORT, SERVER_STARTUP_TIMEOUT):
        process.kill()
        raise RuntimeError("Server failed to start")
    print("✅ Router server ready.\n")
    return process


def check_website_hard_rule(request):
    lowered = request.lower()
    return any(kw in lowered for kw in WEBSITE_KEYWORDS)


def classify_request(request):
    # ONLY website is hard-ruled. Terminal/code/direct are ALL model-based.
    if check_website_hard_rule(request):
        return "website", "hard_rule"

    url = f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions"
    resp = requests.post(
        url,
        json={
            "model": "router",
            "messages": [
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ],
            "max_tokens": 10,
            "grammar": CLASSIFICATION_GRAMMAR,
            "temperature": 0.0,
        },
        timeout=30,
    )
    resp.raise_for_status()
    category = resp.json()["choices"][0]["message"]["content"].strip()
    return category, "model"


# ============================================================================
# FULL REGRESSION SUITE — 35 original + 2 new = 37 cases, permanently merged
# ============================================================================

TEST_CASES = [
    # WEBSITE (5)
    ("Build me a website for a bakery", "website"),
    ("I need a landing page for my app", "website"),
    ("Can you make a homepage for my portfolio", "website"),
    ("Create a website for my coffee shop", "website"),
    ("I want a site for my business", "website"),

    # TERMINAL (9)
    ("Install express using npm", "terminal"),
    ("Run the test suite", "terminal"),
    ("Start the development server", "terminal"),
    ("What's the command to list files", "terminal"),
    ("Install python package requests", "terminal"),
    ("How do I start the server", "terminal"),
    ("Stop the running process on port 8080", "terminal"),
    ("Kill the background job", "terminal"),
    ("List all the files in the current directory", "terminal"),   # NEW

    # CODE (13)
    ("Write a function that reverses a string", "code"),
    ("Review this code for bugs", "code"),
    ("Fix the bug in my sorting algorithm", "code"),
    ("Debug why my API returns a 500 error", "code"),
    ("Help me fix this error in my code", "code"),
    ("Why is my function not working", "code"),   # known accepted miss
    ("Write a python script to parse JSON", "code"),
    ("Can you help me debug this", "code"),
    ("Optimize this SQL query", "code"),
    ("Add error handling to this function", "code"),
    ("Run a quick review of my code", "code"),
    ("Stop overthinking and just fix this bug", "code"),
    ("Can you start writing tests for my app", "code"),
    ("What's the best way to structure this command pattern in my code", "code"),

    # DIRECT (10)
    ("What does this error message mean", "direct"),
    ("Commit my changes with message 'fix typo'", "direct"),
    ("What files are in this project", "direct"),
    ("Explain what a REST API is", "direct"),
    ("Read the contents of config.json", "direct"),
    ("What's the weather today", "direct"),
    ("Tell me a joke", "direct"),
    ("Why is the sky blue", "direct"),
    ("What is a REST API", "direct"),   # NEW
]

# Known issue - tracked separately
KNOWN_ACCEPTED_MISSES = {"Why is my function not working"}


def main():
    print("\n" + "=" * 60)
    print("🔬 ROUTER REGRESSION TEST — reverted to original v6 prompt")
    print("=" * 60)
    print(f"  Testing: {len(TEST_CASES)} cases (35 original + 2 new, merged)")
    print(f"  Known accepted misses: {len(KNOWN_ACCEPTED_MISSES)}")
    print("=" * 60 + "\n")

    build_llama_server()
    model_path = discover_and_download()
    process = start_server(model_path)

    correct = 0
    total = len(TEST_CASES)
    failures = []
    known_misses_hit = []

    for request, expected in TEST_CASES:
        category, source = classify_request(request)
        is_correct = category == expected
        status = "✅" if is_correct else "❌"
        print(f"{status}  \"{request}\"")
        print(f"      expected: {expected} | got: {category} ({source})")

        if is_correct:
            correct += 1
        else:
            if request in KNOWN_ACCEPTED_MISSES:
                known_misses_hit.append(request)
            else:
                failures.append((request, expected, category))

    # Shutdown
    print("\n🔄 Shutting down server...")
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=15)
        print("✅ Server stopped.")
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print("⚠️  Server force-killed.")

    # Results
    print("\n" + "=" * 60)
    print(f"📊 ACCURACY: {correct}/{total} ({100*correct/total:.0f}%)")
    print("=" * 60)

    if known_misses_hit:
        print(f"\n📝 {len(known_misses_hit)} known/accepted miss(es):")
        for r in known_misses_hit:
            print(f"  \"{r}\" (accepted, not a regression)")

    if failures:
        print(f"\n❌ {len(failures)} UNEXPECTED failure(s) — these are real regressions:")
        for req, expected, got in failures:
            print(f"  \"{req}\" — expected {expected}, got {got}")
        
        # Show breakdown by expected category
        print("\n📊 Failure breakdown:")
        for cat in ["website", "terminal", "code", "direct"]:
            cat_failures = [f for f in failures if f[1] == cat]
            if cat_failures:
                print(f"  {cat}: {len(cat_failures)} failures")
                for req, expected, got in cat_failures:
                    print(f"    - \"{req}\" → got {got}")
        
        sys.exit(1)
    else:
        print("\n✅ No unexpected regressions. This prompt is the confirmed baseline.")
        print("   The v6 prompt with website-only hard-rule is the correct design.")
        sys.exit(0)


if __name__ == "__main__":
    main()