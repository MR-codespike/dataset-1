#!/usr/bin/env python3
"""
OMNISIGN-500M – GitHub Actions Dataset Uploader (Fixed)
Uses Kaggle API directly to list and download files.
"""

import os
import subprocess
import shutil
import time
import sys
import json
import zipfile
from huggingface_hub import HfApi, upload_file
import psutil
from kaggle.api.kaggle_api_extended import KaggleApi

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
    },
    "how2sign": {
        "slug": "nazarboholii/how2sign",
        "remote_path": "How2Sign/keypoints",
    }
}

# =============================================================================
# HELPERS
# =============================================================================

def get_free_space_gb():
    return psutil.disk_usage('/').free // (1024**3)

def get_uploaded_files(remote_path):
    """Get set of uploaded filenames for a remote path."""
    api = HfApi()
    try:
        files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
        return {f.split('/')[-1] for f in files if f.startswith(remote_path)}
    except:
        return set()

def list_files_from_kaggle_api(dataset_slug):
    """List files using Kaggle API."""
    api = KaggleApi()
    api.authenticate()
    
    try:
        # Get dataset files
        files = api.dataset_list_files(dataset_slug).files
        result = []
        for f in files:
            result.append((f.name, f.size))
        return result
    except Exception as e:
        print(f"   ❌ API error: {e}")
        return []

def download_file_with_cli(dataset_slug, filename, output_dir):
    """Download a single file using Kaggle CLI."""
    print(f"   📥 Downloading: {filename}")
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug,
           '-f', filename, '-p', output_dir]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Check if it's a zip file
        zip_path = os.path.join(output_dir, filename)
        if filename.endswith('.zip'):
            # Extract the zip
            print(f"      📦 Extracting: {filename}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)
            # Find the extracted file (it might have a different name)
            # Usually it's the same name without .zip
            extracted_name = filename.replace('.zip', '')
            extracted_path = os.path.join(output_dir, extracted_name)
            if os.path.exists(extracted_path):
                return extracted_path
            # If not found, return the first file in the directory
            for f in os.listdir(output_dir):
                if os.path.isfile(os.path.join(output_dir, f)):
                    return os.path.join(output_dir, f)
        
        # Check for the downloaded file
        if os.path.exists(zip_path):
            return zip_path
        
        # Check if any new file appeared
        for f in os.listdir(output_dir):
            if os.path.isfile(os.path.join(output_dir, f)):
                return os.path.join(output_dir, f)
        
        return None
    except subprocess.CalledProcessError as e:
        print(f"      ❌ CLI error: {e.stderr}")
        return None

def upload_and_delete(file_path, remote_path, filename):
    """Upload a file to HF, then delete it."""
    api = HfApi()
    remote_file = f"{remote_path}/{filename}"
    try:
        print(f"      ⬆️ Uploading: {filename} ({os.path.getsize(file_path) / (1024**2):.1f} MB)")
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

def download_dataset_via_cli(dataset_slug, output_dir):
    """Fallback: Download entire dataset using CLI."""
    print(f"   📥 Downloading entire dataset: {dataset_slug}")
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug, '-p', output_dir, '--unzip']
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✅ Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Download failed: {e.stderr}")
        return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 OMNISIGN-500M – GITHUB ACTIONS STREAMING UPLOAD")
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
        output_dir = os.path.join(KAGGLE_DOWNLOADS, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Get list of files from Kaggle using API
        print(f"   📋 Listing files for {slug}...")
        file_list = list_files_from_kaggle_api(slug)
        
        if not file_list:
            print("   ⚠️ No files found via API. Falling back to CLI download...")
            if download_dataset_via_cli(slug, output_dir):
                # Now list files from the output directory
                file_list = []
                for root, dirs, files in os.walk(output_dir):
                    for f in files:
                        file_path = os.path.join(root, f)
                        size = os.path.getsize(file_path)
                        rel_path = os.path.relpath(file_path, output_dir)
                        file_list.append((rel_path, size))
            else:
                print(f"   ❌ Failed to download {dataset_name}")
                continue
        
        print(f"   📊 Found {len(file_list)} files.")

        # Get already uploaded files
        uploaded = get_uploaded_files(remote_path)
        print(f"   📊 Already uploaded: {len(uploaded)} files")

        # Download each file if not uploaded
        for filename, size in file_list:
            # Extract just the filename (not the full path)
            base_filename = os.path.basename(filename)
            
            # Check if already uploaded
            if base_filename in uploaded:
                print(f"   ⏭️ Skipping {base_filename} (already uploaded)")
                continue

            # Check free space before download
            free = get_free_space_gb()
            if free < 2:
                print(f"   ⚠️ Low disk space ({free} GB)! Waiting...")
                time.sleep(30)
                free = get_free_space_gb()
                if free < 2:
                    print(f"   ❌ Still low disk space. Aborting.")
                    sys.exit(1)

            # If file already exists locally, upload it
            local_file = os.path.join(output_dir, filename)
            if os.path.exists(local_file):
                print(f"   📁 Found local file: {base_filename}")
                file_to_upload = local_file
            else:
                # Download single file
                file_to_upload = download_file_with_cli(slug, filename, output_dir)
            
            if file_to_upload and os.path.exists(file_to_upload):
                # Upload and delete
                success = upload_and_delete(file_to_upload, remote_path, base_filename)
                if not success:
                    print(f"   ⚠️ Upload failed for {base_filename}. Deleting anyway.")
                    if os.path.exists(file_to_upload):
                        os.remove(file_to_upload)
            else:
                print(f"   ❌ Failed to download {base_filename}")

        print(f"   ✅ {dataset_name} processing complete!")
        
        # Clean up empty directory
        if os.path.exists(output_dir):
            try:
                os.rmdir(output_dir)
            except:
                pass

    print("\n" + "="*70)
    print("🎉 ALL DATASETS PROCESSED!")
    print("="*70)
    print(f"🔗 Check your repository: https://huggingface.co/datasets/{REPO_ID}")
    free = get_free_space_gb()
    print(f"💾 Final free space: {free} GB")

if __name__ == "__main__":
    main()