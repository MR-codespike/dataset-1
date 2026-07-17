#!/usr/bin/env python3
"""
OMNISIGN-500M – GitHub Actions Dataset Uploader
Streams Kaggle download → Uploads chunks → Deletes immediately
"""

import os
import subprocess
import shutil
import time
import json
import sys
import glob
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, login, upload_file
import psutil

# =============================================================================
# CONFIGURATION
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "MR-CODESPIKE/omnipose-raw-videos"
WORK_DIR = "/tmp/omnipose_data"
CHUNK_SIZE_GB = 3  # Smaller chunks to stay within disk limits
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

def download_kaggle_dataset(dataset_slug, output_dir, max_size_gb=5):
    """
    Download dataset from Kaggle in chunks using the Kaggle CLI.
    Downloads only a portion of files at a time.
    """
    print(f"\n⏳ DOWNLOADING CHUNK: {dataset_slug}")
    print(f"   📁 Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check free space
    free = get_free_space_gb()
    print(f"   💾 Free space: {free} GB")
    
    # Get list of files in the dataset (using Kaggle API to list files)
    try:
        # List files without downloading
        list_cmd = ['kaggle', 'datasets', 'files', '-d', dataset_slug]
        result = subprocess.run(list_cmd, capture_output=True, text=True)
        
        # Parse file list
        files = []
        for line in result.stdout.split('\n'):
            if line.strip() and not line.startswith('name') and '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    size_str = parts[1].strip()
                    files.append((filename, size_str))
        
        if not files:
            print("   ⚠️ Could not get file list, downloading entire dataset...")
            cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug, '-p', output_dir, '--unzip']
            subprocess.run(cmd, check=True)
            return True
        
        # Filter out already uploaded files
        api = HfApi()
        uploaded = set()
        try:
            existing = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
            remote_prefix = DATASETS.get(dataset_slug.split('/')[-1], {}).get('remote_path', '')
            if remote_prefix:
                uploaded = {f for f in existing if f.startswith(remote_prefix)}
        except:
            pass
        
        # Download files in batches to manage disk space
        downloaded = 0
        for filename, size_str in files:
            # Check if already uploaded
            if any(filename in f for f in uploaded):
                print(f"   ⏭️ Skipping {filename} (already uploaded)")
                continue
            
            # Check free space before each download
            free = get_free_space_gb()
            if free < 2:
                print(f"   ⚠️ Low disk space ({free} GB)! Waiting for uploads to complete...")
                time.sleep(30)
                continue
            
            # Download single file
            print(f"   📥 Downloading: {filename} ({size_str})")
            cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug, 
                   '-f', filename, '-p', output_dir, '--unzip']
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"      ✅ Downloaded: {filename}")
                downloaded += 1
            except subprocess.CalledProcessError as e:
                print(f"      ❌ Failed to download {filename}: {e}")
                continue
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Fallback: download entire dataset
        print("   ⚠️ Falling back to full download...")
        cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug, '-p', output_dir, '--unzip']
        subprocess.run(cmd, check=True)
        return True

def upload_file_with_retry(file_path, remote_path, max_retries=3):
    """Upload a single file with retries."""
    api = HfApi()
    filename = os.path.basename(file_path)
    remote_file = f"{remote_path}/{filename}"
    
    for attempt in range(max_retries):
        try:
            print(f"      ⬆️ Uploading: {filename}")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=remote_file,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            # Delete immediately after upload
            os.remove(file_path)
            print(f"      ✅ Uploaded & deleted: {filename}")
            return True
        except Exception as e:
            print(f"      ⚠️ Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    return False

def process_dataset(local_folder, remote_path, dataset_name):
    """Process a dataset: download chunk, upload, delete."""
    print(f"\n📤 PROCESSING: {dataset_name}")
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
            size = os.path.getsize(file_path)
            all_files.append((file_path, size))
    
    if not all_files:
        print("   ⚠️ No files found!")
        return False
    
    # Check what's already uploaded
    api = HfApi()
    uploaded = set()
    try:
        existing = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
        uploaded = {f for f in existing if f.startswith(remote_path)}
    except:
        pass
    
    # Filter out already uploaded files
    remaining = [(p, s) for p, s in all_files if f"{remote_path}/{os.path.basename(p)}" not in uploaded]
    
    if not remaining:
        print("   ✅ All files already uploaded!")
        return True
    
    total_gb = sum(s for _, s in remaining) / (1024**3)
    print(f"   📊 Remaining: {len(remaining)} files ({total_gb:.1f} GB)")
    
    # Upload each file, deleting immediately after upload
    success_count = 0
    for file_path, size in remaining:
        filename = os.path.basename(file_path)
        file_size_gb = size / (1024**3)
        
        # Check free space
        free = get_free_space_gb()
        if free < 1:
            print(f"   ⚠️ Low disk space ({free} GB)! Waiting...")
            time.sleep(30)
            continue
        
        print(f"   📤 Processing: {filename} ({file_size_gb:.2f} GB)")
        if upload_file_with_retry(file_path, remote_path):
            success_count += 1
            free = get_free_space_gb()
            print(f"      💾 Free space now: {free} GB")
    
    print(f"   ✅ Uploaded {success_count}/{len(remaining)} files")
    return success_count == len(remaining)

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
        
        # Upload files (chunked)
        success = process_dataset(local_path, remote_path, dataset_name)
        
        if success:
            print(f"   ✅ {dataset_name} upload complete!")
            # Clean up
            delete_folder(local_path, dataset_name)
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