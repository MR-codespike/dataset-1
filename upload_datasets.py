#!/usr/bin/env python3
"""
OMNISIGN-500M – GitHub Actions Dataset Uploader (Streaming)
Downloads files one-by-one → Uploads → Deletes immediately
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

def download_file(dataset_slug, filename, output_dir):
    """Download a single file from Kaggle."""
    print(f"   📥 Downloading: {filename}")
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug,
           '-f', filename, '-p', output_dir, '--unzip']
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Find the downloaded file (may be extracted)
        # The file might be extracted to the output dir with the same name
        possible_path = os.path.join(output_dir, filename)
        if os.path.exists(possible_path):
            return possible_path
        # If it's a zip, it might be extracted to a folder, but we'll handle later
        # For now, assume the file is directly in output_dir
        # Let's just return the path if it exists, else look for any new file
        return possible_path
    except subprocess.CalledProcessError as e:
        print(f"      ❌ Error: {e.stderr}")
        return None

def upload_and_delete(file_path, remote_path, filename):
    """Upload a file to HF, then delete it."""
    api = HfApi()
    remote_file = f"{remote_path}/{filename}"
    try:
        print(f"      ⬆️ Uploading: {filename}")
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

def list_files_from_kaggle(dataset_slug):
    """List files in a Kaggle dataset."""
    cmd = ['kaggle', 'datasets', 'files', '-d', dataset_slug]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ❌ Failed to list files: {result.stderr}")
        return []
    # Parse output
    files = []
    lines = result.stdout.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('name') and '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                filename = parts[0].strip()
                size_str = parts[1].strip()
                files.append((filename, size_str))
    return files

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

        # Get list of files from Kaggle
        print(f"   📋 Listing files for {slug}...")
        file_list = list_files_from_kaggle(slug)
        if not file_list:
            print("   ❌ No files found. Skipping.")
            continue

        print(f"   📊 Found {len(file_list)} files.")

        # Get already uploaded files
        uploaded = get_uploaded_files(remote_path)
        print(f"   📊 Already uploaded: {len(uploaded)} files")

        # Download each file if not uploaded
        for filename, size_str in file_list:
            # Check if already uploaded
            if filename in uploaded:
                print(f"   ⏭️ Skipping {filename} (already uploaded)")
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

            # Download
            file_path = download_file(slug, filename, output_dir)
            if not file_path or not os.path.exists(file_path):
                print(f"   ❌ Failed to download {filename}")
                continue

            # Upload and delete
            success = upload_and_delete(file_path, remote_path, filename)
            if not success:
                print(f"   ⚠️ Upload failed for {filename}. Deleting anyway.")
                if os.path.exists(file_path):
                    os.remove(file_path)

        print(f"   ✅ {dataset_name} processing complete!")

    print("\n" + "="*70)
    print("🎉 ALL DATASETS PROCESSED!")
    print("="*70)
    print(f"🔗 Check your repository: https://huggingface.co/datasets/{REPO_ID}")
    free = get_free_space_gb()
    print(f"💾 Final free space: {free} GB")

if __name__ == "__main__":
    main()