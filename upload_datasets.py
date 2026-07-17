#!/usr/bin/env python3
"""
OMNISIGN-500M – GitHub Actions Dataset Uploader
Downloads zip file (not unzipped), extracts file-by-file, uploads, deletes.
"""

import os
import sys
import time
import shutil
import subprocess
import psutil
import zipfile
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

def download_dataset_without_unzip(slug, output_dir):
    """Download dataset zip without unzipping."""
    log(f"   📥 Downloading {slug} via Kaggle CLI (no unzip)...")
    zip_path = os.path.join(output_dir, f"{slug.replace('/', '_')}.zip")
    cmd = ['kaggle', 'datasets', 'download', '-d', slug, '-p', output_dir]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log(f"      {line.strip()}")
        process.wait()
        if process.returncode != 0:
            log(f"   ❌ Download failed with code {process.returncode}")
            return None
        # Find the downloaded zip
        for f in os.listdir(output_dir):
            if f.endswith('.zip'):
                zip_path = os.path.join(output_dir, f)
                log(f"   ✅ Downloaded: {zip_path}")
                return zip_path
        log("   ❌ No zip file found")
        return None
    except Exception as e:
        log(f"   ❌ Download failed: {e}")
        return None

def extract_file_by_file(zip_path, output_dir, remote_path, dataset_name):
    """Extract files one by one from zip, upload each, delete immediately."""
    log(f"\n📤 EXTRACTING & UPLOADING: {dataset_name}")
    
    if not os.path.exists(zip_path):
        log("   ❌ Zip file not found!")
        return False

    # Check free space for extraction
    free = get_free_space_gb()
    log(f"   💾 Free space before extraction: {free} GB")

    uploaded_filenames = get_uploaded_filenames(remote_path)
    log(f"   📊 Already uploaded: {len(uploaded_filenames)} files")

    success_count = 0
    total_files = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            log(f"   📊 Total files in zip: {total_files}")

            for idx, filename in enumerate(file_list):
                # Skip directories
                if filename.endswith('/'):
                    continue

                base_filename = os.path.basename(filename)
                if base_filename in uploaded_filenames:
                    log(f"   ⏭️ Skipping {base_filename} (already uploaded)")
                    continue

                # Extract single file to memory
                log(f"   📤 [{idx+1}/{total_files}] Extracting: {base_filename}")

                try:
                    # Extract to temp file
                    temp_path = os.path.join(output_dir, base_filename)
                    with open(temp_path, 'wb') as f:
                        f.write(zip_ref.read(filename))
                    
                    # Upload and delete
                    if upload_and_delete(temp_path, remote_path, base_filename):
                        success_count += 1
                    
                    # Check free space periodically
                    free = get_free_space_gb()
                    if free < 1:
                        log(f"   ⚠️ Low disk space ({free} GB)! Waiting 30s...")
                        time.sleep(30)

                except Exception as e:
                    log(f"      ❌ Failed to extract {filename}: {e}")
                    # Try to clean up temp file
                    temp_path = os.path.join(output_dir, base_filename)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    except zipfile.BadZipFile:
        log(f"   ❌ Corrupt zip file!")
        return False

    log(f"   ✅ Uploaded {success_count} new files")
    return True

def process_dataset(slug, output_dir, remote_path, dataset_name):
    """Full pipeline: download zip → extract file-by-file → upload → delete."""
    zip_path = download_dataset_without_unzip(slug, output_dir)
    if not zip_path:
        log(f"   ❌ Failed to download {dataset_name}")
        return False

    # Extract and upload
    success = extract_file_by_file(zip_path, output_dir, remote_path, dataset_name)

    # Delete the zip file
    if os.path.exists(zip_path):
        os.remove(zip_path)
        log(f"   🗑️ Deleted zip file")

    return success

# =============================================================================
# MAIN
# =============================================================================

def main():
    log("\n" + "="*70)
    log("🚀 OMNISIGN-500M – GITHUB ACTIONS PIPELINE (ZIP STREAMING)")
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

        # Process the dataset (download zip, extract, upload, delete)
        process_dataset(slug, output_dir, remote_path, dataset_name)

        # Clean up the entire output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
            log(f"   🗑️ Deleted {output_dir}")

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