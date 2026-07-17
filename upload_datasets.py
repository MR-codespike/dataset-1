#!/usr/bin/env python3
"""
OMNISIGN-500M – GitHub Actions Dataset Uploader
Uses Kaggle CLI to download, uploads file-by-file, deletes after upload.
"""

import os
import sys
import time
import shutil
import subprocess
import psutil
from huggingface_hub import HfApi, upload_file

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ID = "MR-CODESPIKE/omnipose-raw-videos"
WORK_DIR = "/tmp/omnipose_data"
KAGGLE_DOWNLOADS = os.path.join(WORK_DIR, "kaggle_downloads")

DATASETS = {
    "phoenix": {
        "slug": "duyt2231/phoenix2014t",
        "remote_path": "Phoenix-2014T/raw",
        "size_gb": 39,
    },
    "how2sign": {
        "slug": "nazarboholii/how2sign",
        "remote_path": "How2Sign/keypoints",
        "size_gb": 75,
    }
}

# =============================================================================
# HELPERS
# =============================================================================

def log(msg):
    print(msg, flush=True)

def get_free_space_gb():
    return psutil.disk_usage('/').free // (1024**3)

def get_uploaded_filenames(remote_path):
    api = HfApi()
    try:
        files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
        return {f.split('/')[-1] for f in files if f.startswith(remote_path)}
    except Exception as e:
        log(f"   ⚠️ Could not list uploaded files: {e}")
        return set()

def upload_and_delete(file_path, remote_path, filename):
    api = HfApi()
    remote_file = f"{remote_path}/{filename}"
    try:
        file_size = os.path.getsize(file_path) / (1024**2)
        log(f"      ⬆️ Uploading: {filename} ({file_size:.1f} MB)")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=remote_file,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        os.remove(file_path)
        log(f"      ✅ Uploaded & deleted: {filename}")
        return True
    except Exception as e:
        log(f"      ❌ Upload failed: {e}")
        return False

def download_dataset(slug, output_dir):
    log(f"   📥 Downloading {slug} via Kaggle CLI...")
    cmd = ['kaggle', 'datasets', 'download', '-d', slug, '-p', output_dir, '--unzip']
    try:
        # Run with streaming output to see progress
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log(f"      {line.strip()}")
        process.wait()
        if process.returncode != 0:
            log(f"   ❌ Download failed with code {process.returncode}")
            return False
        log(f"   ✅ Download complete!")
        return True
    except Exception as e:
        log(f"   ❌ Download failed: {e}")
        return False

def process_dataset(local_folder, remote_path, dataset_name):
    log(f"\n📤 PROCESSING FILES: {dataset_name}")
    log(f"   📁 Source: {local_folder}")
    log(f"   📁 Target: {REPO_ID}/{remote_path}")

    if not os.path.exists(local_folder):
        log("   ❌ Local folder not found!")
        return False

    all_files = []
    for root, dirs, files in os.walk(local_folder):
        for f in files:
            file_path = os.path.join(root, f)
            all_files.append((f, file_path))

    if not all_files:
        log("   ⚠️ No files found!")
        return False

    uploaded = get_uploaded_filenames(remote_path)
    log(f"   📊 Total files: {len(all_files)}")
    log(f"   📊 Already uploaded: {len(uploaded)}")

    success_count = 0
    for filename, file_path in all_files:
        if filename in uploaded:
            log(f"   ⏭️ Skipping {filename} (already uploaded)")
            continue

        free = get_free_space_gb()
        if free < 1:
            log(f"   ⚠️ Low disk space ({free} GB)! Waiting 30s...")
            time.sleep(30)

        if upload_and_delete(file_path, remote_path, filename):
            success_count += 1

    log(f"   ✅ Uploaded {success_count} new files")
    return True

# =============================================================================
# MAIN
# =============================================================================

def main():
    log("\n" + "="*70)
    log("🚀 OMNISIGN-500M – GITHUB ACTIONS DATASET PIPELINE (CLI)")
    log("="*70)
    log(f"📁 Target HF Repo: {REPO_ID}")
    log(f"💾 Work directory: {WORK_DIR}")
    log("="*70 + "\n")

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(KAGGLE_DOWNLOADS, exist_ok=True)

    free = get_free_space_gb()
    log(f"💾 Initial free space: {free} GB\n")

    for dataset_name, config in DATASETS.items():
        log("\n" + "="*70)
        log(f"📌 PROCESSING: {dataset_name.upper()}")
        log("="*70)

        slug = config["slug"]
        remote_path = config["remote_path"]
        size_gb = config["size_gb"]
        output_dir = os.path.join(KAGGLE_DOWNLOADS, dataset_name)

        free = get_free_space_gb()
        if free < size_gb * 1.2:
            log(f"⚠️ Not enough free space! Need ~{size_gb * 1.2:.0f} GB, have {free} GB")
            log("   Free up space and re-run.")
            sys.exit(1)

        if not download_dataset(slug, output_dir):
            log(f"   ❌ Failed to download {dataset_name}")
            continue

        process_dataset(output_dir, remote_path, dataset_name)

        log(f"\n🧹 Cleaning up {dataset_name}...")
        shutil.rmtree(output_dir, ignore_errors=True)
        log(f"   ✅ Deleted {output_dir}")

        free = get_free_space_gb()
        log(f"💾 Free space after {dataset_name}: {free} GB")

    log("\n" + "="*70)
    log("🎉 ALL DATASETS PROCESSED!")
    log("="*70)
    log(f"🔗 Check your repository: https://huggingface.co/datasets/{REPO_ID}")
    free = get_free_space_gb()
    log(f"💾 Final free space: {free} GB")

if __name__ == "__main__":
    main()