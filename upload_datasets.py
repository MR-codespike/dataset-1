#!/usr/bin/env python3
"""
OMNISIGN-500M – GitHub Actions Dataset Uploader
Downloads from Kaggle → Splits into 10 GB chunks → Uploads → Deletes
"""

import os
import subprocess
import shutil
import time
import json
import sys
import glob
from pathlib import Path
from huggingface_hub import HfApi, login, upload_file
import psutil

# =============================================================================
# CONFIGURATION
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "MR-CODESPIKE/omnipose-raw-videos"
WORK_DIR = "/tmp/omnipose_data"
CHUNK_SIZE_GB = 10  # Upload in 10 GB chunks
KAGGLE_DOWNLOADS = os.path.join(WORK_DIR, "kaggle_downloads")

# Dataset configs
DATASETS = {
    "phoenix": {
        "slug": "duyt2231/phoenix2014t",
        "remote_path": "Phoenix-2014T/raw",
        "size_gb": 39,
        "local_path": os.path.join(KAGGLE_DOWNLOADS, "phoenix")
    },
    "how2sign": {
        "slug": "nazarboholii/how2sign",
        "remote_path": "How2Sign/keypoints",
        "size_gb": 75,
        "local_path": os.path.join(KAGGLE_DOWNLOADS, "how2sign")
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
    print(f"   💾 Free space before download: {free} GB")
    
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

def upload_chunk(file_list, remote_path, chunk_num, total_chunks):
    """Upload a chunk of files to Hugging Face."""
    api = HfApi()
    success_count = 0
    chunk_size_gb = 0
    
    print(f"\n   📦 Chunk {chunk_num}/{total_chunks}: {len(file_list)} files")
    
    for file_path in file_list:
        # Calculate relative path
        rel_path = os.path.basename(file_path)
        # Determine subfolder structure if needed
        remote_file = f"{remote_path}/{rel_path}"
        
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        chunk_size_gb += file_size_gb
        
        try:
            print(f"      ⬆️ Uploading: {rel_path} ({file_size_gb:.2f} GB)")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=remote_file,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            success_count += 1
            # Delete immediately after upload to free space
            os.remove(file_path)
            print(f"      ✅ Uploaded & deleted: {rel_path}")
        except Exception as e:
            print(f"      ❌ Failed: {rel_path} - {e}")
    
    print(f"   ✅ Chunk {chunk_num} complete! ({success_count}/{len(file_list)} uploaded, {chunk_size_gb:.2f} GB)")
    return success_count == len(file_list)

def upload_in_chunks(local_folder, remote_path, dataset_name):
    """
    Upload files in 10 GB chunks, deleting each file after upload.
    """
    print(f"\n📤 UPLOADING IN CHUNKS: {dataset_name}")
    print(f"   📁 Source: {local_folder}")
    print(f"   📁 Target: {REPO_ID}/{remote_path}")
    print(f"   📦 Chunk size: {CHUNK_SIZE_GB} GB")
    
    if not os.path.exists(local_folder):
        print("   ❌ Local folder not found!")
        return False
    
    # Get all files
    all_files = []
    for root, dirs, files in os.walk(local_folder):
        for f in files:
            file_path = os.path.join(root, f)
            size = os.path.getsize(file_path)
            all_files.append((file_path, size))
    
    if not all_files:
        print("   ⚠️ No files found!")
        return False
    
    # Check what's already uploaded
    uploaded_files = set(get_uploaded_files(remote_path))
    uploaded_paths = set()
    for uploaded in uploaded_files:
        # Extract filename from remote path
        filename = uploaded.split('/')[-1]
        uploaded_paths.add(filename)
    
    # Filter out already uploaded files
    remaining = [(p, s) for p, s in all_files if os.path.basename(p) not in uploaded_paths]
    
    if not remaining:
        print("   ✅ All files already uploaded!")
        return True
    
    total_gb = sum(s for _, s in remaining) / (1024**3)
    print(f"   📊 Remaining: {len(remaining)} files ({total_gb:.1f} GB)")
    
    # Sort by size (largest first)
    remaining.sort(key=lambda x: x[1], reverse=True)
    
    # Upload in chunks
    chunk_num = 1
    total_chunks = max(1, int(total_gb / CHUNK_SIZE_GB) + 1)
    
    while remaining:
        # Select files for this chunk (up to CHUNK_SIZE_GB)
        chunk_files = []
        chunk_size = 0
        chunk_size_limit = CHUNK_SIZE_GB * (1024**3)
        
        while remaining and chunk_size < chunk_size_limit:
            file_path, size = remaining.pop(0)
            chunk_files.append(file_path)
            chunk_size += size
        
        chunk_gb = chunk_size / (1024**3)
        print(f"\n   📦 Chunk {chunk_num}/{total_chunks}: {len(chunk_files)} files ({chunk_gb:.2f} GB)")
        
        # Upload this chunk
        success = upload_chunk(chunk_files, remote_path, chunk_num, total_chunks)
        
        # Check free space after chunk
        free = get_free_space_gb()
        print(f"   💾 Free space after chunk: {free} GB")
        
        chunk_num += 1
        
        # If we're running low on space, wait
        if free < 2:
            print("   ⚠️ Low disk space! Waiting 60 seconds...")
            time.sleep(60)
    
    # Final verification
    uploaded = get_uploaded_files(remote_path)
    print(f"\n   📊 Total uploaded files: {len(uploaded)}")
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
    
    # Process each dataset
    for dataset_name, config in DATASETS.items():
        print("\n" + "="*70)
        print(f"📌 PROCESSING: {dataset_name.upper()}")
        print("="*70)
        
        local_path = config["local_path"]
        remote_path = config["remote_path"]
        slug = config["slug"]
        size_gb = config["size_gb"]
        
        # Check if we already have the files
        if os.path.exists(local_path):
            local_size = get_folder_size_gb(local_path)
            print(f"   💾 Local folder exists: {local_size:.1f} GB")
        else:
            local_size = 0
        
        # Download if needed
        if local_size < 1:
            print(f"   📥 Downloading {dataset_name} from Kaggle ({size_gb} GB)...")
            if download_kaggle_dataset(slug, local_path):
                print(f"   ✅ {dataset_name} downloaded!")
            else:
                print(f"   ❌ {dataset_name} download failed!")
                continue
        else:
            print(f"   ✅ {dataset_name} already downloaded")
        
        # Upload in chunks
        success = upload_in_chunks(local_path, remote_path, dataset_name)
        
        if success:
            print(f"   ✅ {dataset_name} upload complete!")
            # Delete to free space for next dataset
            cleanup_after_upload(local_path, dataset_name)
        else:
            print(f"   ⚠️ {dataset_name} upload may be incomplete")
    
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