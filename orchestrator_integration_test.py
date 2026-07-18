#!/usr/bin/env python3
"""
Full Orchestrator Integration Test — GitHub Actions version (FIXED)
=============================================================

Fixes:
1. Website: Uses correct placeholder fields for the selected template
2. Direct: Routes "direct" requests to the router model (not a separate model)
"""

import subprocess
import time
import os
import sys
import re
import signal
import json
import requests
import psutil
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIG — read from environment (GitHub Secrets)
# ============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable not set.")
    sys.exit(1)

TEMPLATES_REPO_ID = os.environ.get("TEMPLATES_REPO_ID", "MR-CODESPIKE/template-library")
TEMPLATES_REPO_TYPE = "dataset"

# Use GitHub workspace or current directory
BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

LLAMA_CPP_DIR = os.path.join(BASE_DIR, "llama.cpp")
MODELS_DIR = os.path.join(BASE_DIR, "gguf_models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates_integration")
OUTPUT_SITE_DIR = os.path.join(BASE_DIR, "output_site")

os.makedirs(OUTPUT_SITE_DIR, exist_ok=True)

# Model configurations (without filenames - we'll discover them)
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

MAX_TOKENS_BY_MODEL = {"router": 300, "terminal": 100, "coder": 1500}
SERVER_STARTUP_TIMEOUT_SECONDS = 120

# Hard rule: ONLY website (unambiguous)
WEBSITE_KEYWORDS = [
    "website", "web site", "webpage", "web page", "landing page",
    "build me a site", "build a site", "my site", "homepage",
    "create a website", "make a website", "website for", "site for my",
]

# GBNF grammar for classification
CLASSIFICATION_GRAMMAR = 'root ::= "terminal" | "code" | "direct"'
CLASSIFICATION_SYSTEM_PROMPT = """You are a request router. Classify the user's request into EXACTLY ONE category.

1. "terminal" = running commands, installing packages, starting/stopping services, shell operations.
2. "code" = writing, reviewing, debugging, or explaining code/programming.
3. "direct" = anything else: general questions, git operations, file reads, conversation.

Respond with ONLY ONE WORD: terminal, code, or direct."""

# ============================================================================
# DISCOVER GGUF FILENAMES
# ============================================================================

def discover_model_filenames():
    """Find the first .gguf file in each model repository"""
    models_config = {}
    
    print("🔍 Discovering GGUF files in your repositories...\n")
    
    from huggingface_hub import list_repo_files
    
    for name, cfg in MODEL_REPOS.items():
        repo_id = cfg["repo_id"]
        print(f"📂 {name} ({repo_id})")
        
        try:
            files = list_repo_files(repo_id, token=HF_TOKEN)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                print(f"  ❌ No .gguf files found in {repo_id}")
                sys.exit(1)
            
            filename = gguf_files[0]
            print(f"  ✅ Found: {filename}")
            
            models_config[name] = {
                "repo_id": repo_id,
                "filename": filename,
                "port": cfg["port"],
                "context_size": cfg["context_size"],
                "local_path": None,
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            sys.exit(1)
        
        print()
    
    return models_config

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
# DOWNLOAD MODELS
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
            sys.exit(1)
    
    return models_config

# ============================================================================
# DOWNLOAD TEMPLATES AND INDEX
# ============================================================================

def download_templates_and_index():
    print(f"📥 Downloading templates + index from {TEMPLATES_REPO_ID} ...")
    try:
        local_path = snapshot_download(
            repo_id=TEMPLATES_REPO_ID,
            repo_type=TEMPLATES_REPO_TYPE,
            token=HF_TOKEN,
            local_dir=TEMPLATES_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"  ✅ -> {local_path}")
        return local_path
    except Exception as e:
        print(f"  ❌ Failed to download templates: {e}")
        sys.exit(1)

# ============================================================================
# REAL MODEL MANAGER (relay pattern, code-block extraction for coder)
# ============================================================================

class ModelManager:
    def __init__(self, models_config):
        self.models_config = models_config
        self.current_model_name = None
        self.current_process = None

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
        print(f"  ⬇️  Unloading '{name}' ...")
        self.current_process.send_signal(signal.SIGTERM)
        try:
            self.current_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.current_process.kill()
            self.current_process.wait(timeout=5)
        self.current_process = None
        self.current_model_name = None
        print(f"  ✅ Unloaded '{name}'")

    def load(self, model_name):
        if self.current_model_name == model_name:
            return
        self.unload_current()
        cfg = self.models_config[model_name]
        print(f"  ⬆️  Loading '{model_name}' ...")
        
        cmd = [
            f"{LLAMA_CPP_DIR}/build/bin/llama-server",
            "-m", cfg["local_path"],
            "--port", str(cfg["port"]),
            "--host", "127.0.0.1",
            "-c", str(cfg["context_size"]),
            "--n-gpu-layers", "0",
            "--threads", str(os.cpu_count() or 2),
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not self._wait_for_ready(cfg["port"], SERVER_STARTUP_TIMEOUT_SECONDS):
            process.kill()
            raise RuntimeError(f"'{model_name}' failed to start")
        
        self.current_process = process
        self.current_model_name = model_name
        print(f"  ✅ Loaded '{model_name}'")

    def chat(self, model_name, user_message, grammar=None):
        # FIX: For "direct", use the router model (not a separate model)
        if model_name == "direct":
            model_name = "router"
        
        self.load(model_name)
        cfg = self.models_config[model_name]
        max_tokens = MAX_TOKENS_BY_MODEL.get(model_name, 300)
        url = f"http://127.0.0.1:{cfg['port']}/v1/chat/completions"
        
        messages = [{"role": "user", "content": user_message}]
        if grammar:
            messages.insert(0, {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT})
        
        body = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0 if grammar else 0.7,
        }
        if grammar:
            body["grammar"] = grammar
        
        resp = requests.post(url, json=body, timeout=180)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def shutdown(self):
        self.unload_current()

# ============================================================================
# REAL CLASSIFIER (hard rule + grammar-constrained model)
# ============================================================================

def real_classifier(user_request, manager):
    lowered = user_request.lower()
    if any(kw in lowered for kw in WEBSITE_KEYWORDS):
        print(f"  [classified: website (hard_rule)]")
        return "website"
    category = manager.chat("router", user_request, grammar=CLASSIFICATION_GRAMMAR)
    print(f"  [classified: {category} (model)]")
    return category

# ============================================================================
# REAL MODEL CHAT (for terminal/code/direct handlers)
# ============================================================================

def real_model_chat(model_name, message, manager):
    # FIX: For "direct", use the router model
    if model_name == "direct":
        model_name = "router"
    return manager.chat(model_name, message)

# ============================================================================
# REAL CODE EXTRACTION
# ============================================================================

def real_extract_code_blocks(raw_text):
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    blocks = [{"language": m.group(1) or "text", "code": m.group(2).strip()} for m in pattern.finditer(raw_text)]
    explanation = pattern.sub("", raw_text).strip()
    return explanation, blocks

# ============================================================================
# REAL TEMPLATE SEARCH (embedding retrieval against the downloaded index)
# ============================================================================

def load_template_index(templates_local_path):
    print("🧠 Loading embedding model + index ...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    index_path = os.path.join(templates_local_path, "_index")
    if not os.path.exists(index_path):
        print(f"❌ Index not found at {index_path}")
        sys.exit(1)
    
    embeddings = np.load(os.path.join(index_path, "embeddings.npy"))
    with open(os.path.join(index_path, "metadata.json")) as f:
        template_metadata = json.load(f)
    
    print(f"✅ Loaded {len(template_metadata)} templates")
    return embed_model, embeddings, template_metadata

def real_template_search(user_request, embed_model, embeddings, template_metadata):
    query_embedding = embed_model.encode([user_request], normalize_embeddings=True)[0]
    scores = embeddings @ query_embedding
    best_idx = int(np.argmax(scores))
    return {
        "id": template_metadata[best_idx]["id"],
        "score": float(scores[best_idx]),
        "path": template_metadata[best_idx]["path"],
    }

# ============================================================================
# REAL PATCH PIPELINE
# ============================================================================

class PatchError(Exception):
    pass

HEX_COLOR_PATTERN = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

def validate_business_data(meta, business_data):
    placeholders = meta["placeholders"]
    problems = []
    for field in placeholders.get("scalar", []):
        if field not in business_data:
            problems.append(f"Missing: {field}")
        elif "color" in field.lower() and not HEX_COLOR_PATTERN.match(str(business_data[field])):
            problems.append(f"Bad color: {field}")
    for block_name, block_info in placeholders.get("repeating", {}).items():
        if block_name not in business_data or not business_data[block_name]:
            problems.append(f"Missing/empty repeating block: {block_name}")
    if problems:
        raise PatchError("; ".join(problems))
    return True

def patch_scalars(text, business_data, scalar_fields):
    for field in scalar_fields:
        if field in business_data:
            text = text.replace("{{" + field + "}}", str(business_data[field]))
    return text

def expand_repeating_block(html, block_name, items, item_fields):
    pattern = re.compile(rf"<!--\s*REPEAT:{block_name}\s*-->(.*?)<!--\s*END:{block_name}\s*-->", re.DOTALL)
    match = pattern.search(html)
    if not match:
        return html
    block_template = match.group(1)
    expanded = ""
    for item in items:
        block = block_template
        for f in item_fields:
            if f in item:
                block = block.replace("{{" + f + "}}", str(item[f]))
        expanded += block
    return pattern.sub(expanded, html, count=1)

def real_patch_template(match, business_data, templates_local_path):
    template_folder = Path(templates_local_path) / match["path"]
    
    with open(template_folder / "meta.json") as f:
        meta = json.load(f)
    with open(template_folder / "index.html") as f:
        html = f.read()
    with open(template_folder / "style.css") as f:
        css = f.read()

    validate_business_data(meta, business_data)
    scalar_fields = meta["placeholders"].get("scalar", [])
    html = patch_scalars(html, business_data, scalar_fields)
    css = patch_scalars(css, business_data, scalar_fields)
    
    for block_name, block_info in meta["placeholders"].get("repeating", {}).items():
        html = expand_repeating_block(html, block_name, business_data[block_name], block_info["fields"])
    
    return html, css

# ============================================================================
# DYNAMIC BUSINESS DATA COLLECTION (matches the selected template)
# ============================================================================

def dynamic_collect_business_data(match, user_request):
    """Collects business data dynamically based on the matched template."""
    
    # Load the template's meta.json to see what fields it expects
    template_folder = Path(TEMPLATES_DIR) / match["path"]
    with open(template_folder / "meta.json") as f:
        meta = json.load(f)
    
    placeholders = meta["placeholders"]
    scalar_fields = placeholders.get("scalar", [])
    repeating_blocks = placeholders.get("repeating", {})
    
    print(f"  📋 Template expects: {', '.join(scalar_fields)}")
    
    # Build data dynamically based on what the template needs
    business_data = {}
    
    # Common fields
    field_map = {
        "business_name": "Riverside Bakery",
        "shop_name": "Riverside Bakery",
        "tagline": "Fresh bread, baked every morning",
        "shop_tagline": "Fresh bread, baked every morning",
        "about_text": "A small family bakery serving the neighborhood since 2015.",
        "maker_name": "Sarah Miller",
        "maker_story": "Baking has been my passion for 15 years. I started Riverside Bakery to share my love of sourdough with the community.",
        "phone_number": "555-0100",
        "contact_email": "hello@riversidebakery.com",
        "address": "88 River Street",
        "hours_text": "Tue-Sun 7am-4pm",
        "primary_color": "#8b5a2b",
        "accent_color": "#d2a679",
        "map_embed_url": "https://maps.example.com/embed",
        "instagram_handle": "@riversidebakery",
        "twitter_handle": "@riversidebakery",
        "facebook_handle": "riversidebakery",
    }
    
    # Fill scalar fields
    for field in scalar_fields:
        if field in field_map:
            business_data[field] = field_map[field]
        elif field == "business_type":
            business_data[field] = "bakery"
        elif field == "year_founded":
            business_data[field] = "2015"
        else:
            # Default value for unknown fields
            business_data[field] = f"{{{{ {field} }}}}"
            print(f"  ⚠️  Unknown field: {field} using placeholder")
    
    # Fill repeating blocks
    for block_name, block_info in repeating_blocks.items():
        item_fields = block_info.get("fields", [])
        
        if block_name == "menu_items" or block_name == "products":
            business_data[block_name] = [
                {f: "Sourdough Loaf" if f == "item_name" or f == "product_name" else 
                   "Naturally leavened, 24hr ferment" if f == "item_description" or f == "product_description" else 
                   "$7" if f == "item_price" or f == "price" else 
                   f"{{{{ {f} }}}}" for f in item_fields},
                {f: "Cinnamon Roll" if f == "item_name" or f == "product_name" else 
                   "Fresh baked, glazed" if f == "item_description" or f == "product_description" else 
                   "$4" if f == "item_price" or f == "price" else 
                   f"{{{{ {f} }}}}" for f in item_fields},
            ]
        elif block_name == "team_members":
            business_data[block_name] = [
                {f: "Sarah Miller" if f == "name" else "Head Baker" if f == "role" else f"{{{{ {f} }}}}" for f in item_fields},
                {f: "James Chen" if f == "name" else "Pastry Chef" if f == "role" else f"{{{{ {f} }}}}" for f in item_fields},
            ]
        else:
            # Default for unknown blocks
            business_data[block_name] = [
                {f: f"{{{{ {f} }}}}" for f in item_fields}
            ]
    
    return business_data

# ============================================================================
# AGENT LOOP
# ============================================================================

class AgentLoopError(Exception):
    pass

class AgentLoop:
    def __init__(self, classifier, model_chat, template_search, patch_template, 
                 collect_business_data, extract_code_blocks):
        self.classifier = classifier
        self.model_chat = model_chat
        self.template_search = template_search
        self.patch_template = patch_template
        self.collect_business_data = collect_business_data
        self.extract_code_blocks = extract_code_blocks
        self.manager = None  # Will be set later
    
    def handle_request(self, user_request):
        # Step 1: Classify
        category = self.classifier(user_request, self.manager)
        result = {"category": category}
        
        # Step 2: Route to appropriate handler
        if category == "website":
            match = self.template_search(user_request)
            result["template_used"] = match["id"]
            business_data = self.collect_business_data(match, user_request)
            html, css = self.patch_template(match, business_data)
            
            # Save the patched site
            output_path = os.path.join(OUTPUT_SITE_DIR, "index.html")
            with open(output_path, "w") as f:
                f.write(html)
            with open(os.path.join(OUTPUT_SITE_DIR, "style.css"), "w") as f:
                f.write(css)
            
            result["status"] = "success"
            result["output_path"] = output_path
            
        elif category == "terminal":
            response = self.model_chat("terminal", user_request, self.manager)
            result["status"] = "success"
            result["response"] = response
            
        elif category == "code":
            response = self.model_chat("coder", user_request, self.manager)
            explanation, code_blocks = self.extract_code_blocks(response)
            result["status"] = "success"
            result["response"] = response
            result["explanation"] = explanation
            result["code_blocks"] = code_blocks
            
        else:  # direct
            response = self.model_chat("direct", user_request, self.manager)
            result["status"] = "success"
            result["response"] = response
            
        return result

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🚀 FULL ORCHESTRATOR INTEGRATION TEST (FIXED)")
    print("="*60 + "\n")
    
    # Step 1: Discover models
    models_config = discover_model_filenames()
    
    # Step 2: Build llama-server
    if not build_llama_server():
        print("❌ Failed to build llama-server")
        sys.exit(1)
    
    # Step 3: Download models
    models_config = download_models(models_config)
    
    # Step 4: Start model manager
    manager = ModelManager(models_config)
    
    # Step 5: Download templates + index
    templates_local_path = download_templates_and_index()
    
    # Step 6: Load template index
    embed_model, embeddings, template_metadata = load_template_index(templates_local_path)
    
    # Step 7: Create agent loop with real components
    loop = AgentLoop(
        classifier=real_classifier,
        model_chat=real_model_chat,
        template_search=lambda req: real_template_search(req, embed_model, embeddings, template_metadata),
        patch_template=lambda match, data: real_patch_template(match, data, templates_local_path),
        collect_business_data=dynamic_collect_business_data,
        extract_code_blocks=real_extract_code_blocks,
    )
    loop.manager = manager  # Inject manager reference
    
    # Step 8: Run test requests
    print("\n" + "="*60)
    print("🧪 RUNNING 4 REAL END-TO-END REQUESTS")
    print("="*60 + "\n")
    
    test_requests = [
        "Build me a website for my bakery",
        "List all the files in the current directory",
        "Write a function that reverses a string",
        "What is a REST API",
    ]
    
    results = []
    for req in test_requests:
        print(f"\n--- Request: \"{req}\" ---")
        t0 = time.time()
        try:
            result = loop.handle_request(req)
            elapsed = time.time() - t0
            
            print(f"  Category: {result['category']}")
            print(f"  Status: {result.get('status')}")
            print(f"  Time: {elapsed:.1f}s")
            
            if result["category"] == "code" and "code_blocks" in result:
                print(f"  Code blocks found: {len(result['code_blocks'])}")
                if result['code_blocks']:
                    print(f"  First block language: {result['code_blocks'][0].get('language', 'unknown')}")
            
            if result["category"] == "website" and result.get("status") == "success":
                print(f"  Template used: {result['template_used']}")
                print(f"  Output: {result['output_path']}")
            
            results.append({"request": req, "success": True, "result": result})
            
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ❌ ERROR: {e}")
            results.append({"request": req, "success": False, "error": str(e)})
    
    # Step 9: Cleanup
    print("\n🔄 Shutting down model manager...")
    manager.shutdown()
    print("✅ Model manager stopped")
    
    # Step 10: Summary
    print("\n" + "="*60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"✅ {passed}/{total} tests passed")
    
    if passed < total:
        print("\n❌ Failed tests:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['request']}: {r.get('error', 'Unknown error')}")
    
    # Show website output if it succeeded
    website_result = next((r for r in results if r["success"] and r["result"].get("category") == "website"), None)
    if website_result:
        output_path = website_result["result"].get("output_path")
        if output_path and os.path.exists(output_path):
            print(f"\n📄 Website output preview ({output_path}):")
            with open(output_path) as f:
                content = f.read()
                print(content[:500] + ("..." if len(content) > 500 else ""))
    
    print("\n" + "="*60)
    if passed == total:
        print("🎉 ALL TESTS PASSED! Orchestrator is ready for production!")
        sys.exit(0)
    else:
        print(f"⚠️  {total - passed} tests failed. Review errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()