#!/usr/bin/env python3
"""
OMNISIGN-500M – GitHub Actions Dataset Uploader (Simplified)
Downloads entire dataset → Uploads file-by-file → Deletes each file after upload
"""

import os
import subprocess
import shutil
import time
import sys
from huggingface_hub import HfApi, upload_file
import psutil

# =============================================================================
# CONFIGURATION
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "MR-CODESPIKE/omnipose-raw-videos"
WORK_DIR = "/tmp/omnipose_data"
KAGGLE_DOWNLOADS = os.path.join(WORK_DIR, "kaggle_downloads")

# Dataset configs
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

def get_free_space_gb():
    return psutil.disk_usage('/').free // (1024**3)

def get_uploaded_filenames(remote_path):
    """Get set of filenames already uploaded to HF under this remote path."""
    api = HfApi()
    try:
        files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
        return {f.split('/')[-1] for f in files if f.startswith(remote_path)}
    except:
        return set()

def upload_and_delete(file_path, remote_path, filename):
    """Upload a single file to HF and delete it locally."""
    api = HfApi()
    remote_file = f"{remote_path}/{filename}"
    try:
        file_size = os.path.getsize(file_path) / (1024**2)
        print(f"      ⬆️ Uploading: {filename} ({file_size:.1f} MB)")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=remote_file,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        os.remove(file_path)
        print(f"      ✅ Uploaded & deleted: {filename}")
        return True
    except Exception as e:
        print(f"      ❌ Upload failed: {e}")
        return False

def download_dataset(slug, output_dir):
    """Download entire dataset from Kaggle using CLI."""
    print(f"   📥 Downloading {slug} ... (this may take a while)")
    cmd = ['kaggle', 'datasets', 'download', '-d', slug, '-p', output_dir, '--unzip']
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✅ Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Download failed: {e.stderr}")
        return False

def process_dataset(local_folder, remote_path, dataset_name):
    """Walk the local folder, upload each file (if not already uploaded), then delete it."""
    print(f"\n📤 PROCESSING FILES: {dataset_name}")
    print(f"   📁 Source: {local_folder}")
    print(f"   📁 Target: {REPO_ID}/{remote_path}")

    if not os.path.exists(local_folder):
        print("   ❌ Local folder not found!")
        return False

    # Get all files
    all_files = []
    for root, dirs, files in os.walk(local_folder):
        for f in files:
            file_path = os.path.join(root, f)
            rel_path = os.path.relpath(file_path, local_folder)
            all_files.append((rel_path, file_path))

    if not all_files:
        print("   ⚠️ No files found!")
        return False

    # Get already uploaded filenames
    uploaded = get_uploaded_filenames(remote_path)
    print(f"   📊 Total files: {len(all_files)}")
    print(f"   📊 Already uploaded: {len(uploaded)}")

    # Process each file
    success_count = 0
    for rel_path, file_path in all_files:
        filename = os.path.basename(file_path)
        if filename in uploaded:
            print(f"   ⏭️ Skipping {filename} (already uploaded)")
            continue

        # Check free space before upload (just in case)
        free = get_free_space_gb()
        if free < 1:
            print(f"   ⚠️ Low disk space ({free} GB)! Waiting...")
            time.sleep(30)

        # Upload and delete
        if upload_and_delete(file_path, remote_path, filename):
            success_count += 1

    print(f"   ✅ Uploaded {success_count} new files")
    return True

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 OMNISIGN-500M – GITHUB ACTIONS DATASET PIPELINE")
    print("="*70)
    print(f"📁 Target HF Repo: {REPO_ID}")
    print(f"💾 Work directory: {WORK_DIR}")
    print("="*70 + "\n")

    # Create work dir
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(KAGGLE_DOWNLOADS, exist_ok=True)

    free = get_free_space_gb()
    print(f"💾 Initial free space: {free} GB\n")

    for dataset_name, config in DATASETS.items():
        print("\n" + "="*70)
        print(f"📌 PROCESSING: {dataset_name.upper()}")
        print("="*70)

        slug = config["slug"]
        remote_path = config["remote_path"]
        size_gb = config["size_gb"]
        output_dir = os.path.join(KAGGLE_DOWNLOADS, dataset_name)

        # Check if we have enough space
        free = get_free_space_gb()
        if free < size_gb * 1.2:  # 20% buffer
            print(f"⚠️ Not enough free space! Need ~{size_gb * 1.2:.0f} GB, have {free} GB")
            print("   Please free up space and re-run.")
            sys.exit(1)

        # Download dataset
        if not download_dataset(slug, output_dir):
            print(f"   ❌ Failed to download {dataset_name}")
            continue

        # Process files (upload one by one, delete after upload)
        success = process_dataset(output_dir, remote_path, dataset_name)

        # Delete the entire dataset folder to free space for the next dataset
        print(f"\n🧹 Cleaning up {dataset_name}...")
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"   ✅ Deleted {output_dir}")

        free = get_free_space_gb()
        print(f"💾 Free space after {dataset_name}: {free} GB")

    print("\n" + "="*70)
    print("🎉 ALL DATASETS PROCESSED!")
    print("="*70)
    print(f"🔗 Check your repository: https://huggingface.co/datasets/{REPO_ID}")
    free = get_free_space_gb()
    print(f"💾 Final free space: {free} GB")

if __name__ == "__main__":
    main()