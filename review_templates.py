#!/usr/bin/env python3
"""
Bulk-set "reviewed": true across all meta.json files in your HF template
library dataset — but ONLY for templates that pass validation first.
Anything that fails validation is left as reviewed: false and logged,
instead of being silently approved.

GitHub Actions optimized version – reads HF_TOKEN and HF_REPO_ID from env.
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

# ============================================================================
# CONFIG — read from environment variables
# ============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"

# Use a temporary directory (writable) – fallback to a local folder in workspace
BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
LOCAL_DIR = os.path.join(BASE_DIR, "review_temp")
FAILURES_LOG = os.path.join(LOCAL_DIR, "_validation_failures.json")

# ============================================================================
# VALIDATION LOGIC (same as generation pipeline)
# ============================================================================

class ValidationError(Exception):
    pass

def extract_placeholders(text):
    return set(re.findall(r"\{\{(\w+)\}\}", text))

def extract_repeat_blocks(text):
    return set(re.findall(r"<!--\s*REPEAT:(\w+)\s*-->", text))

def validate_template(meta, html, css):
    for field in ["id", "level", "category", "subcategory", "placeholders"]:
        if field not in meta:
            raise ValidationError(f"meta.json missing required field: {field}")

    placeholders = meta["placeholders"]
    declared_scalar = set(placeholders.get("scalar", []))
    declared_repeating = placeholders.get("repeating", {})

    used_scalar_html = extract_placeholders(html)
    used_scalar_css = extract_placeholders(css)
    used_repeat_blocks_html = extract_repeat_blocks(html)

    declared_repeat_fields = set()
    for block_name, block_info in declared_repeating.items():
        declared_repeat_fields.update(block_info.get("fields", []))

    all_declared = declared_scalar | declared_repeat_fields
    all_used = used_scalar_html | used_scalar_css

    undeclared = all_used - all_declared
    if undeclared:
        raise ValidationError(f"Placeholders used but not declared in meta: {undeclared}")

    unused_scalar = declared_scalar - all_used
    if unused_scalar:
        raise ValidationError(f"Scalar placeholders declared but never used: {unused_scalar}")

    for block_name in declared_repeating:
        if block_name not in used_repeat_blocks_html:
            raise ValidationError(f"Declared repeating block '{block_name}' not found in html")
        if f"<!-- END:{block_name} -->" not in html and f"<!--END:{block_name}-->" not in html:
            raise ValidationError(f"Repeating block '{block_name}' missing matching END marker")

    return True

# ============================================================================
# STEP 1: Download the current dataset locally
# ============================================================================

print(f"📥 Downloading current dataset from {HF_REPO_ID} ...")
os.makedirs(LOCAL_DIR, exist_ok=True)

try:
    local_path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        token=HF_TOKEN,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"✅ Downloaded to {local_path}")
except Exception as e:
    print(f"❌ Failed to download dataset: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Validate each template, mark reviewed only if it passes
# ============================================================================

meta_files = list(Path(local_path).rglob("meta.json"))
print(f"\n📄 Found {len(meta_files)} meta.json files.")

updated_count = 0
already_true_count = 0
failed_validation = []
missing_files = []

for meta_path in meta_files:
    folder = meta_path.parent
    html_path = folder / "index.html"
    css_path = folder / "style.css"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    template_id = meta.get("id", str(folder.relative_to(local_path)))

    if meta.get("reviewed") is True:
        already_true_count += 1
        continue

    if not html_path.exists() or not css_path.exists():
        missing_files.append({
            "id": template_id,
            "path": str(folder.relative_to(local_path)),
            "error": "Missing index.html or style.css",
        })
        continue

    with open(html_path, "r") as f:
        html = f.read()
    with open(css_path, "r") as f:
        css = f.read()

    try:
        validate_template(meta, html, css)
    except ValidationError as e:
        failed_validation.append({
            "id": template_id,
            "path": str(folder.relative_to(local_path)),
            "error": str(e),
        })
        print(f"  ✗ {template_id}: FAILED validation — {e}")
        continue

    # Passed validation – mark reviewed
    meta["reviewed"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    updated_count += 1
    print(f"  ✓ {template_id}: passed validation, marked reviewed")

print(f"\n--- Summary ---")
print(f"Marked reviewed (passed validation): {updated_count}")
print(f"Already reviewed: {already_true_count}")
print(f"Failed validation (left as reviewed: false): {len(failed_validation)}")
print(f"Missing files (left as reviewed: false): {len(missing_files)}")

# ============================================================================
# STEP 3: Save failures log
# ============================================================================

if failed_validation or missing_files:
    with open(FAILURES_LOG, "w") as f:
        json.dump({
            "failed_validation": failed_validation,
            "missing_files": missing_files,
        }, f, indent=2)
    print(f"\n📝 Details saved to {FAILURES_LOG}")
    print("These templates were NOT marked reviewed. Fix them (or regenerate)")
    print("and re-run this script — it will re-check and mark them once they pass.")

# ============================================================================
# STEP 4: Push the change back to Hugging Face
# ============================================================================

if updated_count > 0:
    print(f"\n📤 Uploading updated files back to {HF_REPO_ID} ...")
    api = HfApi(token=HF_TOKEN)
    try:
        api.upload_folder(
            folder_path=local_path,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            commit_message=f"Mark {updated_count} templates as reviewed (validated)",
        )
        print("✅ Upload complete.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)
else:
    print("\nℹ️ No new templates passed validation to be marked reviewed.")

# Exit with 0 (success) even if some failed – they are logged for manual fixing.
print("\n✅ Review process finished.")