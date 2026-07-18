#!/usr/bin/env python3
"""
Router Classifier Test – Model Accuracy Only (No Terminal Hard-Rules)
Tests the model's ability to distinguish: code vs terminal vs direct.
Website is the only hard-rule (unambiguous).
"""

import os
import subprocess
import time
import sys
import json
import requests
import threading
import signal
from pathlib import Path
from huggingface_hub import hf_hub_download

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_REPO = "MR-CODESPIKE/Qwen2.5-3B-Instruct-GGUF-Q4_K_M"
MODEL_FILE = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
GGUF_DIR = "./gguf_models"
LLAMA_SERVER_URL = "http://localhost:8080"
LLAMA_SERVER_BIN = "./llama-server"

# =============================================================================
# TEST CASES
# =============================================================================

# Hard-rule only for website (unambiguous)
WEBSITE_KEYWORDS = {"website", "landing page", "homepage", "portfolio site", "business site"}

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
# HARD RULE CHECK
# =============================================================================

def check_hard_rule(query):
    """Only website keywords are hard-ruled (unambiguous)."""
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in WEBSITE_KEYWORDS):
        return "website"
    return None

# =============================================================================
# MODEL INTERACTION
# =============================================================================

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
                    "max_tokens": 10,
                },
                timeout=10,
            )
            if response.status_code == 200:
                result = response.json().get("content", "").strip().lower()
                # Extract just the category
                for category in ["code", "terminal", "direct", "website"]:
                    if category in result:
                        return category
                return result
        except Exception as e:
            print(f"   ⚠️ Model query attempt {attempt+1} failed: {e}")
            time.sleep(1)
    
    return "error"

# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("\n" + "="*60)
    print("🚀 ROUTER CLASSIFIER — MODEL ACCURACY TEST")
    print("="*60)
    print("⚠️  Hard-rules: ONLY website (unambiguous)")
    print("⚠️  Terminal, Code, Direct: ALL model-based")
    print("="*60 + "\n")

    # Ensure model is downloaded and server is running
    os.makedirs(GGUF_DIR, exist_ok=True)
    
    print(f"🔍 Discovering GGUF file in {MODEL_REPO}...")
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

    # Build llama-server if needed
    if not os.path.exists(LLAMA_SERVER_BIN):
        print("🔧 Building llama-server...")
        subprocess.run(["make", "llama-server"], check=True)

    # Start server
    print("🚀 Starting router model server ...")
    server_process = subprocess.Popen(
        [LLAMA_SERVER_BIN, "-m", model_path, "-c", "4096"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if os.name != 'nt' else None,
    )
    
    # Wait for server to be ready
    time.sleep(5)
    max_wait = 60
    while max_wait > 0:
        try:
            response = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Router model ready.\n")
                break
        except:
            pass
        time.sleep(2)
        max_wait -= 2
    
    if max_wait <= 0:
        print("❌ Server failed to start")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        sys.exit(1)

    # Run tests
    print("="*60)
    print("🧪 RUNNING ROUTING ACCURACY TEST")
    print("="*60 + "\n")

    passed = 0
    total = len(TEST_CASES)
    failures = []

    for test in TEST_CASES:
        query = test["query"]
        expected = test["expected"]
        
        # Check hard-rule first
        hard_rule_result = check_hard_rule(query)
        if hard_rule_result:
            result = hard_rule_result
            source = "hard_rule"
        else:
            result = query_model(query)
            source = "model"
        
        status = "✅" if result == expected else "❌"
        print(f'{status} "{query}"')
        print(f'    expected: {expected} | got: {result} ({source})')
        
        if result == expected:
            passed += 1
        else:
            failures.append(f'"{query}" — expected {expected}, got {result}')

    # Print summary
    print("\n" + "="*60)
    print(f"📊 ACCURACY: {passed}/{total} ({int(passed/total*100)}%)")
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

    # Cleanup
    print("\n🔄 Shutting down router model server...")
    try:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    except:
        server_process.terminate()
    server_process.wait()
    print("✅ Server stopped.")

if __name__ == "__main__":
    main()