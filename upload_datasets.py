#!/usr/bin/env python3
"""
OMNISIGN-500M – GitHub Actions Dataset Uploader
Downloads from Kaggle → Chunks → Uploads to Hugging Face
"""

import os
import subprocess
import shutil
import time
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi, login, upload_folder
import psutil

# =============================================================================
# CONFIGURATION
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "MR-CODESPIKE/omnipose-raw-videos"
WORK_DIR = "/tmp/omnipose_data"
CHUNK_SIZE_GB = 15  # Upload in 15 GB chunks
KAGGLE_DOWNLOADS = os.path.join(WORK_DIR, "kaggle_downloads")

# Dataset configs
DATASETS = {
    "phoenix": {
        "slug": "duyt2231/phoenix2014t",
        "remote_path": "Phoenix-2014T/raw",
        "size_gb": 39
    },
    "how2sign": {
        "slug": "nazarboholii/how2sign",
        "remote_path": "How2Sign/keypoints",
        "size_gb": 75
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_free_space_gb():
    """Get free disk space in GB."""
    return psutil.disk_usage('/').free // (1024**3)

def get_folder_size_gb(path):
    """Get folder size in GB."""
    if not os.path.exists(path):
        return 0
    total = 0
    try:
        for root, dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except:
                    pass
    except:
        pass
    return total / (1024**3)

def delete_folder(path, name=""):
    """Delete a folder."""
    if os.path.exists(path):
        size = get_folder_size_gb(path)
        print(f"   🗑️ Deleting {name or path} ({size:.1f} GB)...")
        shutil.rmtree(path, ignore_errors=True)
        print(f"   ✅ Deleted! Freed {size:.1f} GB")
        return size
    return 0

def download_kaggle_dataset(dataset_slug, output_dir):
    """Download dataset from Kaggle."""
    print(f"\n⏳ DOWNLOADING: {dataset_slug}")
    print(f"   📁 Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check free space
    free = get_free_space_gb()
    print(f"   💾 Free space: {free} GB")
    
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug, '-p', output_dir, '--unzip']
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✅ Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Download failed: {e.stderr}")
        return False

def get_uploaded_files(remote_path):
    """Get list of already uploaded files."""
    api = HfApi()
    try:
        files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
        return [f for f in files if f.startswith(remote_path)]
    except:
        return []

def chunk_and_upload(local_folder, remote_path, dataset_name):
    """
    Split folder into chunks and upload each chunk.
    CHUNK_SIZE_GB controls how much to upload per chunk.
    """
    print(f"\n📤 UPLOADING CHUNKS: {dataset_name}")
    print(f"   📁 Source: {local_folder}")
    print(f"   📁 Target: {REPO_ID}/{remote_path}")
    print(f"   📦 Chunk size: {CHUNK_SIZE_GB} GB")
    
    if not os.path.exists(local_folder):
        print("   ❌ Local folder not found!")
        return False
    
    # Get list of files
    all_files = []
    for root, dirs, files in os.walk(local_folder):
        for f in files:
            file_path = os.path.join(root, f)
            rel_path = os.path.relpath(file_path, local_folder)
            size = os.path.getsize(file_path)
            all_files.append((rel_path, size, file_path))
    
    if not all_files:
        print("   ⚠️ No files found!")
        return False
    
    # Sort by size (largest first)
    all_files.sort(key=lambda x: x[1], reverse=True)
    
    # Check what's already uploaded
    uploaded_files = set(get_uploaded_files(remote_path))
    remaining = [(r, s, p) for r, s, p in all_files if f"{remote_path}/{r}" not in uploaded_files]
    
    if not remaining:
        print("   ✅ All files already uploaded!")
        return True
    
    total_gb = sum(s for _, s, _ in remaining) / (1024**3)
    print(f"   📊 Remaining: {len(remaining)} files ({total_gb:.1f} GB)")
    
    # Upload in chunks
    chunk_num = 1
    while remaining:
        # Calculate chunk size
        chunk_size_bytes = CHUNK_SIZE_GB * (1024**3)
        chunk_files = []
        chunk_size = 0
        
        # Select files for this chunk
        while remaining and chunk_size < chunk_size_bytes:
            rel_path, size, file_path = remaining.pop(0)
            chunk_files.append((rel_path, file_path))
            chunk_size += size
        
        chunk_gb = chunk_size / (1024**3)
        print(f"\n   📦 Chunk {chunk_num}: {len(chunk_files)} files ({chunk_gb:.2f} GB)")
        
        # Upload this chunk (upload each file individually)
        api = HfApi()
        success_count = 0
        for rel_path, file_path in chunk_files:
            remote_file = f"{remote_path}/{rel_path}"
            try:
                print(f"      ⬆️ Uploading: {rel_path}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=remote_file,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                )
                success_count += 1
            except Exception as e:
                print(f"      ❌ Failed: {rel_path} - {e}")
        
        print(f"   ✅ Chunk {chunk_num} complete! ({success_count}/{len(chunk_files)} uploaded)")
        
        # After each chunk, check if we need to clean up
        free = get_free_space_gb()
        print(f"   💾 Free space: {free} GB")
        
        chunk_num += 1
        
        # Exit early if we're running low on space
        if free < 5:
            print("   ⚠️ Low disk space! Pausing to let previous chunk finish uploading...")
            time.sleep(60)
    
    # Final verification
    uploaded = get_uploaded_files(remote_path)
    print(f"\n   📊 Total uploaded: {len(uploaded)} files")
    return True

def cleanup_after_upload(local_folder, dataset_name):
    """Delete dataset folder after upload."""
    print(f"\n🧹 Cleaning up {dataset_name}...")
    size = get_folder_size_gb(local_folder)
    if size > 0:
        delete_folder(local_folder, dataset_name)
    return size

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 OMNISIGN-500M – GITHUB ACTIONS DATASET PIPELINE")
    print("="*70)
    print(f"📁 Target HF Repo: {REPO_ID}")
    print(f"📦 Chunk size: {CHUNK_SIZE_GB} GB")
    print(f"💾 Work directory: {WORK_DIR}")
    print("="*70 + "\n")
    
    # Create working directory
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(KAGGLE_DOWNLOADS, exist_ok=True)
    
    # Check initial space
    free = get_free_space_gb()
    print(f"💾 Initial free space: {free} GB\n")
    
    # Process Phoenix
    print("\n" + "="*70)
    print("📌 PHASE 1: Phoenix-2014T")
    print("="*70)
    
    phoenix_path = os.path.join(KAGGLE_DOWNLOADS, "phoenix")
    phoenix_remote = "Phoenix-2014T/raw"
    
    # Download if not exists
    if not os.path.exists(phoenix_path) or get_folder_size_gb(phoenix_path) < 1:
        print("   📥 Downloading Phoenix...")
        if download_kaggle_dataset("duyt2231/phoenix2014t", phoenix_path):
            print("   ✅ Phoenix downloaded!")
        else:
            print("   ❌ Phoenix download failed!")
            sys.exit(1)
    else:
        print(f"   💾 Phoenix already downloaded: {get_folder_size_gb(phoenix_path):.1f} GB")
    
    # Upload Phoenix
    chunk_and_upload(phoenix_path, phoenix_remote, "Phoenix")
    
    # Delete Phoenix after upload
    cleanup_after_upload(phoenix_path, "Phoenix")
    
    # Process How2Sign
    print("\n" + "="*70)
    print("📌 PHASE 2: How2Sign")
    print("="*70)
    
    how2sign_path = os.path.join(KAGGLE_DOWNLOADS, "how2sign")
    how2sign_remote = "How2Sign/keypoints"
    
    # Check free space
    free = get_free_space_gb()
    print(f"💾 Free space before How2Sign: {free} GB")
    
    if free < 80:
        print(f"⚠️ Not enough free space! Need 80 GB, have {free} GB")
        print("   Please free up space and re-run.")
        sys.exit(1)
    
    # Download if not exists
    if not os.path.exists(how2sign_path) or get_folder_size_gb(how2sign_path) < 1:
        print("   📥 Downloading How2Sign...")
        if download_kaggle_dataset("nazarboholii/how2sign", how2sign_path):
            print("   ✅ How2Sign downloaded!")
        else:
            print("   ❌ How2Sign download failed!")
            sys.exit(1)
    else:
        print(f"   💾 How2Sign already downloaded: {get_folder_size_gb(how2sign_path):.1f} GB")
    
    # Upload How2Sign
    chunk_and_upload(how2sign_path, how2sign_remote, "How2Sign")
    
    # Delete How2Sign after upload
    cleanup_after_upload(how2sign_path, "How2Sign")
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 ALL DATASETS PROCESSED!")
    print("="*70)
    print(f"🔗 Check your repository: https://huggingface.co/datasets/{REPO_ID}")
    
    free = get_free_space_gb()
    print(f"💾 Final free space: {free} GB")
    print("\n✅ All datasets uploaded successfully!")

if __name__ == "__main__":
    main()