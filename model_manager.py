#!/usr/bin/env python3
"""
Model Manager — GitHub Actions version
Automatically discovers GGUF filenames from your repositories
"""

import subprocess
import time
import os
import sys
import signal
import requests
import psutil
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

# ============================================================================
# CONFIG — read from environment (GitHub Secrets)
# ============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable not set.")
    sys.exit(1)

# Model repositories (without filenames - we'll discover them)
MODEL_REPOS = {
    "router": {
        "repo_id": "MR-CODESPIKE/Qwen2.5-3B-Instruct-GGUF-Q4_K_M",
        "port": 8081,
        "context_size": 4096,
    },
    "terminal": {
        "repo_id": "MR-CODESPIKE/Qwen2.5-1.5B-Instruct-GGUF-Q4_K_M",
        "port": 8082,
        "context_size": 2048,
    },
    "coder": {
        "repo_id": "MR-CODESPIKE/DeepSeek-R1-Distill-Qwen-1.5B-GGUF-Q4_K_M",
        "port": 8083,
        "context_size": 4096,
    },
}

# Use GitHub workspace or current directory
BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
LLAMA_CPP_DIR = os.path.join(BASE_DIR, "llama.cpp")
MODELS_DIR = os.path.join(BASE_DIR, "gguf_models")

SERVER_STARTUP_TIMEOUT_SECONDS = 120
SERVER_SHUTDOWN_TIMEOUT_SECONDS = 15

# ============================================================================
# STEP 0: Discover GGUF filenames from repositories
# ============================================================================

def discover_model_filenames():
    """Find the first .gguf file in each model repository"""
    models_config = {}
    
    print("🔍 Discovering GGUF files in your repositories...\n")
    
    for name, cfg in MODEL_REPOS.items():
        repo_id = cfg["repo_id"]
        print(f"📂 {name} ({repo_id})")
        
        try:
            files = list_repo_files(repo_id, token=HF_TOKEN)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                print(f"  ❌ No .gguf files found in {repo_id}")
                print(f"  Available files: {', '.join(files[:5])}")
                continue
            
            # Use the first .gguf file found
            filename = gguf_files[0]
            print(f"  ✅ Found: {filename}")
            
            models_config[name] = {
                "repo_id": repo_id,
                "filename": filename,
                "port": cfg["port"],
                "context_size": cfg["context_size"],
                "local_path": None,  # Will be set after download
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
        
        print()
    
    if not models_config:
        print("❌ No GGUF models found in any repository!")
        sys.exit(1)
    
    return models_config


# ============================================================================
# STEP 1: Build llama-server from source
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
# STEP 2: Download the GGUF models
# ============================================================================

def download_models(models_config):
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for name, cfg in models_config.items():
        print(f"📥 Downloading {name} ({cfg['repo_id']}/{cfg['filename']}) ...")
        try:
            local_path = hf_hub_download(
                repo_id=cfg["repo_id"],
                filename=cfg["filename"],
                token=HF_TOKEN,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
            )
            cfg["local_path"] = local_path
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  ✅ -> {local_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ❌ Failed to download {name}: {e}")
            raise
    
    return models_config


# ============================================================================
# STEP 3: Model Manager — the "relay baton" load/unload logic
# ============================================================================

class ModelManagerError(Exception):
    pass


class ModelManager:
    def __init__(self, models_config):
        self.models_config = models_config
        self.current_model_name = None
        self.current_process = None

    def _get_rss_mb(self, pid):
        try:
            proc = psutil.Process(pid)
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            return total / (1024 * 1024)
        except psutil.NoSuchProcess:
            return 0.0

    def _wait_for_ready(self, port, timeout):
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

    def unload_current(self):
        if self.current_process is None:
            return

        name = self.current_model_name
        print(f"  ⬇️  Unloading '{name}' (pid {self.current_process.pid}) ...")
        t0 = time.time()

        self.current_process.send_signal(signal.SIGTERM)
        try:
            self.current_process.wait(timeout=SERVER_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  '{name}' did not exit cleanly, force-killing ...")
            self.current_process.kill()
            self.current_process.wait(timeout=5)

        elapsed = time.time() - t0
        print(f"  ✅ Unloaded '{name}' in {elapsed:.1f}s")

        self.current_process = None
        self.current_model_name = None

    def load(self, model_name):
        if self.current_model_name == model_name:
            print(f"  ⏭️  '{model_name}' already loaded, skipping reload.")
            return

        self.unload_current()

        cfg = self.models_config[model_name]
        print(f"  ⬆️  Loading '{model_name}' from {cfg['local_path']} ...")
        t0 = time.time()

        cmd = [
            f"{LLAMA_CPP_DIR}/build/bin/llama-server",
            "-m", cfg["local_path"],
            "--port", str(cfg["port"]),
            "--host", "127.0.0.1",
            "-c", str(cfg["context_size"]),
            "--n-gpu-layers", "0",
            "--threads", str(os.cpu_count() or 2),
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        ready = self._wait_for_ready(cfg["port"], SERVER_STARTUP_TIMEOUT_SECONDS)
        elapsed = time.time() - t0

        if not ready:
            process.kill()
            raise ModelManagerError(
                f"'{model_name}' failed to become ready within {SERVER_STARTUP_TIMEOUT_SECONDS}s"
            )

        rss_mb = self._get_rss_mb(process.pid)
        print(f"  ✅ Loaded '{model_name}' in {elapsed:.1f}s — RAM: {rss_mb:.0f} MB")

        self.current_process = process
        self.current_model_name = model_name

    def chat(self, model_name, user_message, max_tokens=200):
        self.load(model_name)
        cfg = self.models_config[model_name]
        url = f"http://127.0.0.1:{cfg['port']}/v1/chat/completions"

        t0 = time.time()
        try:
            resp = requests.post(
                url,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": user_message}],
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            elapsed = time.time() - t0
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"  💬 Inference took {elapsed:.1f}s")
            return content
        except requests.exceptions.RequestException as e:
            raise ModelManagerError(f"Chat request failed: {e}")

    def shutdown(self):
        self.unload_current()


# ============================================================================
# STEP 4: Run the full relay test
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🚀 MODEL MANAGER — FULL RELAY TEST")
    print("="*60 + "\n")

    # Discover model filenames
    models_config = discover_model_filenames()
    
    print("📋 Models to use:")
    for name, cfg in models_config.items():
        print(f"  - {name}: {cfg['filename']} ({cfg['repo_id']})")
    print()

    # Build llama-server
    if not build_llama_server():
        print("❌ Failed to build llama-server")
        sys.exit(1)

    # Download models
    print("\n📥 Downloading models...")
    models_config = download_models(models_config)

    # Initialize manager
    manager = ModelManager(models_config)

    # Test queries
    test_queries = [
        ("router", "Say hello in one short sentence.", 100),
        ("terminal", "What is the linux command to list all files in a directory?", 150),
        ("coder", "Write a one-line python function that adds two numbers.", 150),
        ("router", "Say goodbye in one short sentence.", 100),
    ]

    print("\n" + "="*60)
    print("🔄 RUNNING RELAY TEST CYCLE")
    print("="*60 + "\n")

    results = []

    for i, (model, query, max_tokens) in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/{len(test_queries)}: {model} ---")
        try:
            reply = manager.chat(model, query, max_tokens)
            print(f"  📝 Reply: {reply[:200]}{'...' if len(reply) > 200 else ''}")
            results.append({
                "test": i,
                "model": model,
                "success": True,
                "reply_preview": reply[:100]
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "test": i,
                "model": model,
                "success": False,
                "error": str(e)
            })

    # Shutdown
    manager.shutdown()

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r["success"])
    print(f"✅ {successful}/{len(results)} tests passed")
    
    if successful < len(results):
        print("\n❌ Failed tests:")
        for r in results:
            if not r["success"]:
                print(f"  - Test {r['test']}: {r['model']} - {r.get('error', 'Unknown error')}")
    
    print("\n✅ Test complete!")
    sys.exit(0 if successful == len(results) else 1)


if __name__ == "__main__":
    main()