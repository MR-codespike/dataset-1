#!/usr/bin/env python3
"""
Full Orchestrator Integration Test — trained classifier wired in
=====================================================================

Replaces the model-based router classification with the trained
embedding + MLP classifier head (97% on the regression suite). The
website hard rule is unchanged. Everything else (Tool Executor,
confidence threshold, real subprocess execution, template patch
pipeline) is carried over from the last working integration test.

Key change: classification no longer loads/unloads the 3B router model.
It's now: embed request (reusing the SAME MiniLM model already loaded
for template retrieval) -> classifier.predict() -> category. Near-instant,
no model swap. The 3B router model only loads when a "direct" request
actually needs an answer generated.
"""

import subprocess
import time
import os
import sys
import re
import signal
import json
import requests
import numpy as np
import joblib
from pathlib import Path
from enum import Enum
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
from sentence_transformers import SentenceTransformer

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("❌ HF_TOKEN not set.")
    sys.exit(1)

TEMPLATES_REPO_ID = os.environ.get("TEMPLATES_REPO_ID", "MR-CODESPIKE/template-library")
TEMPLATES_REPO_TYPE = "dataset"

BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
LLAMA_CPP_DIR = os.path.join(BASE_DIR, "llama.cpp")
MODELS_DIR = os.path.join(BASE_DIR, "gguf_models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates_integration")
OUTPUT_SITE_DIR = os.path.join(BASE_DIR, "output_site")
SANDBOX_DIR = os.path.join(BASE_DIR, "command_sandbox")
CLASSIFIER_DIR = os.path.join(BASE_DIR, "classifier_model")

for d in [OUTPUT_SITE_DIR, SANDBOX_DIR, CLASSIFIER_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_REPOS = {
    "terminal": {"repo_id": "MR-CODESPIKE/Qwen2.5-1.5B-Instruct-GGUF-Q4_K_M", "port": 8082, "context_size": 2048},
    "coder": {"repo_id": "MR-CODESPIKE/DeepSeek-R1-Distill-Qwen-1.5B-GGUF-Q4_K_M", "port": 8083, "context_size": 4096},
}
MAX_TOKENS_BY_MODEL = {"terminal": 100, "coder": 1500}
SERVER_STARTUP_TIMEOUT_SECONDS = 120

WEBSITE_KEYWORDS = [
    "website", "web site", "webpage", "web page", "landing page",
    "build me a site", "build a site", "my site", "homepage",
    "create a website", "make a website", "website for", "site for my",
]

TERMINAL_SYSTEM_PROMPT = """You are a terminal command generator. Given a plain-English request, respond with ONLY the exact shell command needed to accomplish the task. DO NOT include explanations, apologies, markdown, or conversational text. The response must be a single, runnable shell command and nothing else.

Examples:
- "List all files" -> "ls -la"
- "Install python package requests" -> "pip install requests"
- "Start the development server" -> "npm start"

Now respond with ONLY the shell command for:"""


# ============================================================================
# TOOL EXECUTOR (proven, unchanged)
# ============================================================================

class ConfirmLevel(Enum):
    AUTO = "auto"
    CONFIRM = "confirm"
    STRICT_CONFIRM = "strict"


DANGEROUS_PATTERNS = [
    (re.compile(r"\brm\s+-rf\s+/\S*"), "rm -rf on an absolute path"),
    (re.compile(r"\brm\s+-rf\s+\*"), "rm -rf with wildcard"),
    (re.compile(r"\brm\s+-rf\s+~"), "rm -rf on home directory shorthand"),
    (re.compile(r"\bgit\s+push\s+.*--force\b"), "force push"),
    (re.compile(r"\bgit\s+push\s+.*-f\b"), "force push (short flag)"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard"),
    (re.compile(r"\bgit\s+clean\s+-[a-z]*f[a-z]*d\b"), "git clean force+directories"),
    (re.compile(r":\(\)\s*\{.*\};\s*:"), "fork bomb pattern"),
    (re.compile(r"\bmkfs\b"), "filesystem format command"),
    (re.compile(r"\bdd\s+.*of=/dev/"), "dd writing directly to a device"),
    (re.compile(r"\bchmod\s+-R\s+777\b"), "recursive chmod 777"),
]


def check_dangerous_pattern(command_text):
    for pattern, reason in DANGEROUS_PATTERNS:
        if pattern.search(command_text):
            return True, reason
    return False, None


class ConfirmationRequiredError(Exception):
    pass


class ProposedAction:
    def __init__(self, tool_name, params, confirm_level, preview, danger_reason=None):
        self.tool_name = tool_name
        self.params = params
        self.confirm_level = confirm_level
        self.preview = preview
        self.danger_reason = danger_reason

    def __repr__(self):
        tier = self.confirm_level.value.upper()
        danger = f" [DANGER: {self.danger_reason}]" if self.danger_reason else ""
        return f"<ProposedAction {self.tool_name} tier={tier}{danger}: {self.preview}>"


def propose_action(tool_name, **params):
    base_level = ConfirmLevel.CONFIRM
    danger_reason = None
    command = params.get("command", "")
    is_dangerous, reason = check_dangerous_pattern(command)
    if is_dangerous:
        base_level = ConfirmLevel.STRICT_CONFIRM
        danger_reason = reason
    preview = f"Run shell command: {command}"
    return ProposedAction(tool_name, params, base_level, preview, danger_reason)


def execute_action(proposed, confirmed=False, cwd=None):
    if proposed.confirm_level != ConfirmLevel.AUTO and not confirmed:
        raise ConfirmationRequiredError(f"'{proposed.tool_name}' requires confirmation")
    result = subprocess.run(
        proposed.params["command"], shell=True, cwd=cwd,
        capture_output=True, text=True, timeout=30,
    )
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}


# ============================================================================
# BUILD / DOWNLOAD
# ============================================================================

def discover_model_filenames():
    models_config = {}
    for name, cfg in MODEL_REPOS.items():
        try:
            files = list_repo_files(cfg["repo_id"], token=HF_TOKEN)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if not gguf_files:
                print(f"⚠️  No .gguf files found in {cfg['repo_id']}")
                continue
            filename = gguf_files[0]
            print(f"  ✅ {name}: {filename}")
            models_config[name] = {**cfg, "filename": filename, "local_path": None}
        except Exception as e:
            print(f"  ❌ Error discovering {name}: {e}")
    return models_config


def build_llama_server():
    if os.path.exists(f"{LLAMA_CPP_DIR}/build/bin/llama-server"):
        print("✅ llama-server already built, skipping.")
        return
    print("🔨 Building llama-server from source...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LLAMA_CPP_DIR], check=True, capture_output=True)
    subprocess.run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF"], cwd=LLAMA_CPP_DIR, check=True, capture_output=True)
    nproc = os.cpu_count() or 2
    subprocess.run(["cmake", "--build", "build", "--target", "llama-server", "-j", str(min(nproc, 4))], cwd=LLAMA_CPP_DIR, check=True, capture_output=True)
    print("✅ Build complete.")


def download_models(models_config):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, cfg in models_config.items():
        print(f"📥 Downloading {name} ...")
        local_path = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"], token=HF_TOKEN, local_dir=MODELS_DIR)
        cfg["local_path"] = local_path
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  ✅ {size_mb:.1f} MB")
    return models_config


def download_templates_and_index():
    print(f"📥 Downloading templates + index from {TEMPLATES_REPO_ID}...")
    local_path = snapshot_download(
        repo_id=TEMPLATES_REPO_ID,
        repo_type=TEMPLATES_REPO_TYPE,
        token=HF_TOKEN,
        local_dir=TEMPLATES_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"  ✅ -> {local_path}")
    return local_path


def download_classifier():
    print("📥 Downloading trained classifier...")
    os.makedirs(CLASSIFIER_DIR, exist_ok=True)
    for fname in ["classifier_head.joblib", "label_encoder.joblib"]:
        try:
            hf_hub_download(
                repo_id=TEMPLATES_REPO_ID,
                filename=f"classifier_model/{fname}",
                repo_type=TEMPLATES_REPO_TYPE,
                token=HF_TOKEN,
                local_dir=CLASSIFIER_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"  ✅ {fname}")
        except Exception as e:
            print(f"  ❌ Failed to download {fname}: {e}")
            sys.exit(1)
    
    classifier = joblib.load(os.path.join(CLASSIFIER_DIR, "classifier_model", "classifier_head.joblib"))
    label_encoder = joblib.load(os.path.join(CLASSIFIER_DIR, "classifier_model", "label_encoder.joblib"))
    print(f"✅ Classifier loaded. Classes: {list(label_encoder.classes_)}")
    return classifier, label_encoder


# ============================================================================
# MODEL MANAGER (without router)
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
                if requests.get(url, timeout=2).status_code == 200:
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
        cmd = [f"{LLAMA_CPP_DIR}/build/bin/llama-server", "-m", cfg["local_path"],
               "--port", str(cfg["port"]), "--host", "127.0.0.1",
               "-c", str(cfg["context_size"]), "--n-gpu-layers", "0",
               "--threads", str(os.cpu_count() or 2)]
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not self._wait_for_ready(cfg["port"], SERVER_STARTUP_TIMEOUT_SECONDS):
            process.kill()
            raise RuntimeError(f"'{model_name}' failed to start")
        self.current_process = process
        self.current_model_name = model_name
        print(f"  ✅ Loaded '{model_name}'")

    def chat(self, model_name, user_message, system_prompt=None):
        self.load(model_name)
        cfg = self.models_config[model_name]
        max_tokens = MAX_TOKENS_BY_MODEL.get(model_name, 300)
        url = f"http://127.0.0.1:{cfg['port']}/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        resp = requests.post(url, json={"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}, timeout=180)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def shutdown(self):
        self.unload_current()


# ============================================================================
# CLASSIFIER-BASED ROUTING (no model swap needed)
# ============================================================================

def check_website_hard_rule(request):
    lowered = request.lower()
    return any(kw in lowered for kw in WEBSITE_KEYWORDS)


def classify_request(request, embed_model, classifier, label_encoder):
    """Website: hard rule. Terminal/code/direct: embedding + trained classifier."""
    if check_website_hard_rule(request):
        print("  [classified: website (hard_rule)]")
        return "website"

    embedding = embed_model.encode([request], normalize_embeddings=True)
    predicted_encoded = classifier.predict(embedding)
    category = label_encoder.inverse_transform(predicted_encoded)[0]
    print(f"  [classified: {category} (trained classifier)]")
    return category


def real_extract_code_blocks(raw_text):
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    blocks = [{"language": m.group(1) or "text", "code": m.group(2).strip()} for m in pattern.finditer(raw_text)]
    explanation = pattern.sub("", raw_text).strip()
    return explanation, blocks


def real_template_search(request, embed_model, embeddings, template_metadata, confidence_threshold=0.4):
    query_embedding = embed_model.encode([request], normalize_embeddings=True)[0]
    scores = embeddings @ query_embedding
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    print(f"  📊 Top match score: {best_score:.3f} (threshold: {confidence_threshold})")
    if best_score < confidence_threshold:
        print("  ⚠️  Score below threshold - no match")
        return None
    return {"id": template_metadata[best_idx]["id"], "score": best_score, "path": template_metadata[best_idx]["path"]}


# ============================================================================
# PATCH PIPELINE (unchanged)
# ============================================================================

class PatchError(Exception):
    pass

HEX_COLOR_PATTERN = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fa-fA-F]{6})$")

def validate_business_data(meta, business_data):
    placeholders = meta["placeholders"]
    problems = []
    for field in placeholders.get("scalar", []):
        if field not in business_data:
            problems.append(f"Missing: {field}")
        elif "color" in field.lower() and not HEX_COLOR_PATTERN.match(str(business_data[field])):
            problems.append(f"Bad color: {field}")
    for block_name in placeholders.get("repeating", {}):
        if block_name not in business_data or not business_data[block_name]:
            problems.append(f"Missing/empty repeating block: {block_name}")
    if problems:
        raise PatchError("; ".join(problems))


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


def dynamic_collect_business_data(match, user_request, templates_local_path):
    template_folder = Path(templates_local_path) / match["path"]
    with open(template_folder / "meta.json") as f:
        meta = json.load(f)
    scalar_fields = meta["placeholders"].get("scalar", [])
    repeating_blocks = meta["placeholders"].get("repeating", {})
    field_map = {
        "business_name": "Riverside Bakery", "shop_name": "Riverside Bakery",
        "tagline": "Fresh bread, baked every morning", "shop_tagline": "Fresh bread, baked every morning",
        "about_text": "A small family bakery serving the neighborhood since 2015.",
        "maker_name": "Sarah Miller", "maker_story": "Baking has been my passion for 15 years.",
        "phone_number": "555-0100", "contact_email": "hello@riversidebakery.com",
        "address": "88 River Street", "hours_text": "Tue-Sun 7am-4pm",
        "primary_color": "#8b5a2b", "accent_color": "#d2a679",
        "map_embed_url": "https://maps.example.com/embed", "instagram_handle": "@riversidebakery",
    }
    business_data = {}
    for field in scalar_fields:
        if field in field_map:
            business_data[field] = field_map[field]
        elif "color" in field.lower():
            business_data[field] = "#8b5a2b"
        else:
            business_data[field] = f"Please fill: {field}"
    for block_name, block_info in repeating_blocks.items():
        item_fields = block_info.get("fields", [])
        business_data[block_name] = [{f: "Sourdough Loaf" if f in ("item_name", "product_name") else
            "Naturally leavened" if f in ("item_description", "product_description") else
            "$7" if f in ("item_price", "price") else f"Please fill: {f}" for f in item_fields}]
    return business_data


# ============================================================================
# AGENT LOOP
# ============================================================================

class AgentLoop:
    def __init__(self, embed_model, classifier, label_encoder, manager, confirm_callback,
                 template_search, patch_template, collect_business_data, extract_code_blocks):
        self.embed_model = embed_model
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.manager = manager
        self.confirm_callback = confirm_callback
        self.template_search = template_search
        self.patch_template = patch_template
        self.collect_business_data = collect_business_data
        self.extract_code_blocks = extract_code_blocks

    def handle_request(self, user_request):
        t0 = time.time()
        category = classify_request(user_request, self.embed_model, self.classifier, self.label_encoder)
        classify_time = time.time() - t0
        result = {"category": category, "request": user_request, "classify_time": classify_time}

        if category == "website":
            match = self.template_search(user_request)
            if not match:
                result["status"] = "no_match"
                return result
            result["template_used"] = match["id"]
            result["match_score"] = match["score"]
            business_data = self.collect_business_data(match, user_request)
            html, css = self.patch_template(match, business_data)
            output_path = os.path.join(OUTPUT_SITE_DIR, "index.html")
            with open(output_path, "w") as f:
                f.write(html)
            with open(os.path.join(OUTPUT_SITE_DIR, "style.css"), "w") as f:
                f.write(css)
            result["status"] = "success"
            result["output_path"] = output_path
            return result

        elif category == "terminal":
            drafted_command = self.manager.chat("terminal", user_request, system_prompt=TERMINAL_SYSTEM_PROMPT).strip()
            proposed = propose_action("run_command", command=drafted_command)
            result["proposed_command"] = drafted_command
            result["confirm_level"] = proposed.confirm_level.value
            if proposed.confirm_level != ConfirmLevel.AUTO:
                if not self.confirm_callback(proposed):
                    result["status"] = "cancelled"
                    return result
                exec_result = execute_action(proposed, confirmed=True, cwd=SANDBOX_DIR)
            else:
                exec_result = execute_action(proposed, confirmed=False, cwd=SANDBOX_DIR)
            result["execution"] = exec_result
            result["status"] = "success" if exec_result["returncode"] == 0 else "command_failed"
            return result

        elif category == "code":
            response = self.manager.chat("coder", user_request)
            explanation, code_blocks = self.extract_code_blocks(response)
            result["status"] = "success"
            result["code_blocks"] = code_blocks
            return result

        else:  # direct
            response = self.manager.chat("direct", user_request)
            result["status"] = "success"
            result["response"] = response
            return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 ORCHESTRATOR INTEGRATION TEST — trained classifier wired in")
    print("=" * 60 + "\n")

    # Discover models
    models_config = discover_model_filenames()
    
    # Build and download models
    build_llama_server()
    models_config = download_models(models_config)
    manager = ModelManager(models_config)
    
    # Download templates + index
    templates_local_path = download_templates_and_index()
    
    # Load embedding model and retrieval index
    print("🧠 Loading embedding model + retrieval index...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.load(os.path.join(templates_local_path, "_index", "embeddings.npy"))
    with open(os.path.join(templates_local_path, "_index", "metadata.json")) as f:
        template_metadata = json.load(f)
    print(f"✅ Loaded {len(template_metadata)} templates")
    
    # Download classifier
    classifier, label_encoder = download_classifier()

    def confirm_wrapper(proposed_action):
        print(f"  🔔 [CONFIRM: {proposed_action}] -> auto-approving for this test")
        return True

    # Create agent loop
    loop = AgentLoop(
        embed_model=embed_model,
        classifier=classifier,
        label_encoder=label_encoder,
        manager=manager,
        confirm_callback=confirm_wrapper,
        template_search=lambda req: real_template_search(req, embed_model, embeddings, template_metadata, 0.4),
        patch_template=lambda match, data: real_patch_template(match, data, templates_local_path),
        collect_business_data=lambda match, req: dynamic_collect_business_data(match, req, templates_local_path),
        extract_code_blocks=real_extract_code_blocks,
    )

    # Run tests
    test_requests = [
        "Build me a website for my bakery",
        "List all the files in the current directory",
        "Write a function that reverses a string",
        "What is a REST API",
    ]

    print("\n" + "=" * 60)
    print("🧪 RUNNING 4 END-TO-END REQUESTS")
    print("=" * 60)

    results = []
    for req in test_requests:
        print(f"\n--- Request: \"{req}\" ---")
        t0 = time.time()
        try:
            result = loop.handle_request(req)
            elapsed = time.time() - t0
            print(f"  Category: {result['category']} (classified in {result['classify_time']*1000:.0f}ms)")
            print(f"  Status: {result.get('status')}")
            print(f"  Total time: {elapsed:.1f}s")
            if result["category"] == "terminal" and "execution" in result:
                print(f"  Command: {result['proposed_command']}")
                print(f"  Return code: {result['execution']['returncode']}")
            if result["category"] == "code":
                print(f"  Code blocks: {len(result.get('code_blocks', []))}")
            if result["category"] == "website" and result.get("status") == "success":
                print(f"  Template: {result['template_used']} (score {result['match_score']:.3f})")
            results.append({"request": req, "success": result.get("status") == "success", "result": result})
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({"request": req, "success": False, "error": str(e)})

    # Cleanup
    print("\n🔄 Shutting down model manager...")
    manager.shutdown()
    print("✅ Model manager stopped")

    # Summary
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\n" + "=" * 60)
    print(f"📊 RESULTS: {passed}/{total} passed")
    print("=" * 60)

    total_classify_time = sum(r["result"]["classify_time"] for r in results if "result" in r and "classify_time" in r["result"])
    print(f"\n⏱️  Total classification overhead: {total_classify_time*1000:.0f}ms")
    print("   (Compare to old model-based routing: 3B model load + inference per request)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Orchestrator ready with trained classifier!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()