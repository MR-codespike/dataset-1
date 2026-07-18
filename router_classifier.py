#!/usr/bin/env python3
"""
Router Classifier Test – Full Pipeline
Clones llama.cpp, builds llama-server, downloads model, runs tests.
"""

import os
import sys
import time
import subprocess
import requests
import signal
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_REPO = "MR-CODESPIKE/Qwen2.5-3B-Instruct-GGUF-Q4_K_M"
MODEL_FILE = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
LLAMA_CPP_DIR = "./llama.cpp"
LLAMA_SERVER_BIN = f"{LLAMA_CPP_DIR}/llama-server"
GGUF_DIR = "./gguf_models"
LLAMA_SERVER_URL = "http://localhost:8080"

# Website keywords (hard-rule only – unambiguous)
WEBSITE_KEYWORDS = {"website", "landing page", "homepage", "portfolio site", "business site"}

# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES = [
    # WEBSITE (hard-rule only)
    {"query": "Build me a website for a bakery", "expected": "website"},
    {"query": "I need a landing page for my app", "expected": "website"},
    {"query": "Can you make a homepage for my portfolio", "expected": "website"},
    {"query": "Create a website for my coffee shop", "expected": "website"},
    
    # TERMINAL (model must classify these correctly)
    {"query": "Install express using npm", "expected": "terminal"},
    {"query": "Run the test suite", "expected": "terminal"},
    {"query": "Start the development server", "expected": "terminal"},
    {"query": "What's the command to list files", "expected": "terminal"},
    {"query": "Stop the running process on port 8080", "expected": "terminal"},
    {"query": "Kill the background job", "expected": "terminal"},
    {"query": "Check the logs for errors", "expected": "terminal"},
    {"query": "List all running containers", "expected": "terminal"},
    
    # CODE (model must classify these correctly)
    {"query": "Write a function that reverses a string", "expected": "code"},
    {"query": "Review this code for bugs", "expected": "code"},
    {"query": "Fix the bug in my sorting algorithm", "expected": "code"},
    {"query": "Debug why my API returns a 500 error", "expected": "code"},
    {"query": "Optimize this SQL query", "expected": "code"},
    {"query": "Add error handling to this function", "expected": "code"},
    {"query": "What's wrong with this Python script", "expected": "code"},
    
    # ADVERSARIAL – These would break keyword-based routing
    {"query": "Run a quick review of my code", "expected": "code"},
    {"query": "Stop overthinking and just fix this bug", "expected": "code"},
    {"query": "Can you start writing tests for my app", "expected": "code"},
    {"query": "What's the best way to structure this command pattern", "expected": "code"},
    
    # DIRECT (model must classify these correctly)
    {"query": "What does this error message mean", "expected": "direct"},
    {"query": "Commit my changes with message 'fix typo'", "expected": "direct"},
    {"query": "What files are in this project", "expected": "direct"},
    {"query": "Explain what a REST API is", "expected": "direct"},
    {"query": "Why is the sky blue", "expected": "direct"},
    {"query": "What's the weather in London", "expected": "direct"},
    {"query": "Tell me a joke", "expected": "direct"},
]

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_llama_cpp():
    """Clone and build llama.cpp."""
    print("🔧 Setting up llama.cpp...")
    
    if os.path.exists(LLAMA_CPP_DIR):
        print(f"   ✅ llama.cpp already exists at {LLAMA_CPP_DIR}")
    else:
        print("   📥 Cloning llama.cpp repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR],
            check=True
        )
        print("   ✅ Clone complete")
    
    # Check if server binary already exists
    if os.path.exists(LLAMA_SERVER_BIN):
        print(f"   ✅ llama-server already built at {LLAMA_SERVER_BIN}")
        return
    
    # Build llama-server
    print("   🔨 Building llama-server (this may take a few minutes)...")
    os.chdir(LLAMA_CPP_DIR)
    
    # Try to build with make
    try:
        subprocess.run(["make", "llama-server"], check=True, capture_output=True)
        print("   ✅ Build complete")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Build failed: {e.stderr.decode() if e.stderr else 'unknown error'}")
        sys.exit(1)
    finally:
        os.chdir("..")
    
    # Verify binary exists
    if not os.path.exists(LLAMA_SERVER_BIN):
        print("   ❌ llama-server binary not found after build")
        sys.exit(1)

def check_hard_rule(query):
    """Only website keywords are hard-ruled (unambiguous)."""
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in WEBSITE_KEYWORDS):
        return "website"
    return None

def query_model(query, max_retries=3):
    """Query the llama-server and get the routed category."""
    prompt = f"""You are a router that classifies user requests into exactly one of these categories:
- "code": When the user wants you to write, review, debug, or analyze code
- "terminal": When the user wants to run a command, start/stop a service, or perform a terminal action
- "direct": When the user asks a question that doesn't require code or terminal commands
- "website": When the user specifically asks for a website, landing page, or homepage

Respond with ONLY the category name (code, terminal, direct, or website).

User request: "{query}"
Category:"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{LLAMA_SERVER_URL}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 20,
                    "temperature": 0.1,
                    "stop": ["\n", "."],
                },
                timeout=15,
            )
            if response.status_code == 200:
                result = response.json().get("content", "").strip().lower()
                # Extract just the category
                for category in ["code", "terminal", "direct", "website"]:
                    if category in result:
                        return category
                return result
        except requests.exceptions.ConnectionError:
            print(f"   ⚠️ Connection error (attempt {attempt+1}/{max_retries})...")
            time.sleep(2)
        except Exception as e:
            print(f"   ⚠️ Query error: {e}")
            time.sleep(1)
    
    return "error"

def wait_for_server(max_wait=120):
    """Wait for llama-server to be ready."""
    print("   ⏳ Waiting for server to be ready...")
    waited = 0
    while waited < max_wait:
        try:
            response = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
            if response.status_code == 200:
                print("   ✅ Server ready")
                return True
        except:
            pass
        time.sleep(2)
        waited += 2
        if waited % 10 == 0:
            print(f"   ⏳ Still waiting... ({waited}s)")
    return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("🚀 ROUTER CLASSIFIER — MODEL ACCURACY TEST")
    print("="*60)
    print("⚠️  Hard-rules: ONLY website (unambiguous)")
    print("⚠️  Terminal, Code, Direct: ALL model-based")
    print("="*60 + "\n")

    # Step 1: Setup llama.cpp
    setup_llama_cpp()

    # Step 2: Download model
    print(f"\n🔍 Discovering GGUF file in {MODEL_REPO}...")
    os.makedirs(GGUF_DIR, exist_ok=True)
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=GGUF_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Found: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)

    # Step 3: Start llama-server
    print("\n🚀 Starting router model server ...")
    server_process = subprocess.Popen(
        [LLAMA_SERVER_BIN, "-m", model_path, "-c", "4096"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if os.name != 'nt' else None,
    )

    # Step 4: Wait for server
    if not wait_for_server():
        print("❌ Server failed to start")
        try:
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        except:
            server_process.terminate()
        sys.exit(1)

    print("✅ Router model ready.\n")

    # Step 5: Run tests
    print("="*60)
    print("🧪 RUNNING ROUTING ACCURACY TEST")
    print("="*60 + "\n")

    passed = 0
    total = len(TEST_CASES)
    failures = []
    hard_rule_count = 0
    model_count = 0

    for test in TEST_CASES:
        query = test["query"]
        expected = test["expected"]
        
        # Check hard-rule first
        hard_rule_result = check_hard_rule(query)
        if hard_rule_result:
            result = hard_rule_result
            source = "hard_rule"
            hard_rule_count += 1
        else:
            result = query_model(query)
            source = "model"
            model_count += 1
        
        status = "✅" if result == expected else "❌"
        print(f'{status} "{query}"')
        print(f'    expected: {expected} | got: {result} ({source})')
        print()
        
        if result == expected:
            passed += 1
        else:
            failures.append(f'"{query}" — expected {expected}, got {result}')

    # Print summary
    print("="*60)
    print(f"📊 ACCURACY: {passed}/{total} ({int(passed/total*100)}%)")
    print("="*60)
    print(f"   Hard-rule cases: {hard_rule_count} (website only)")
    print(f"   Model cases: {model_count} (terminal, code, direct)")
    print("="*60)

    if failures:
        print(f"\n❌ {len(failures)} misclassification(s):")
        for f in failures:
            print(f"  {f}")
        print("\n⚠️  Test failed.")
        sys.exit(1)
    else:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Model correctly distinguishes code vs terminal vs direct.")
        print("   Website hard-rule confirmed unambiguous.")

    # Step 6: Cleanup
    print("\n🔄 Shutting down router model server...")
    try:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    except:
        server_process.terminate()
    server_process.wait()
    print("✅ Server stopped.")

if __name__ == "__main__":
    main()