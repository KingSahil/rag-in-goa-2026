#!/usr/bin/env python3
"""
Direct chunked file uploader for Hugging Face Spaces.
Uploads repository files file-by-file with automatic retry.
"""
import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "ansh123456789/ragingoa"
ROOT = Path(__file__).resolve().parent

api = HfApi(token=TOKEN)

EXCLUDE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "data/raw", "data/onnx_models"}
EXCLUDE_FILES = {".env", ".env.local", "push_to_space.py"}

def get_all_files():
    files = []
    for p in ROOT.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(ROOT).as_posix()
        # Check exclusion
        parts = rel.split("/")
        if any(d in parts for d in [".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"]):
            continue
        if rel.startswith("data/raw/") or rel.startswith("data/onnx_models/"):
            continue
        if rel.endswith(".pyc") or rel in EXCLUDE_FILES:
            continue
        files.append((p, rel))
    return files

def upload_file_with_retry(local_path: Path, path_in_repo: str, max_retries=5):
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"Uploading {path_in_repo} ({size_mb:.2f} MB)...", end=" ", flush=True)
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=REPO_ID,
                repo_type="space",
                commit_message=f"Add {path_in_repo}",
            )
            print("OK", flush=True)
            return True
        except Exception as e:
            if attempt == max_retries:
                print(f"FAILED after {max_retries} attempts: {e}", flush=True)
                return False
            print(f"(retry {attempt})...", end=" ", flush=True)
            time.sleep(2 * attempt)

def main():
    print(f"Starting direct upload to https://huggingface.co/spaces/{REPO_ID}")
    files = get_all_files()
    print(f"Found {len(files)} files to upload.\n")
    
    # Sort files: small files first, then data files
    files.sort(key=lambda x: x[0].stat().st_size)
    
    failed = []
    for local_path, path_in_repo in files:
        ok = upload_file_with_retry(local_path, path_in_repo)
        if not ok:
            failed.append(path_in_repo)
            
    print("\n" + "=" * 60)
    if failed:
        print(f"Upload finished with {len(failed)} failed files: {failed}")
    else:
        print("🎉 ALL FILES UPLOADED SUCCESSFULLY!")
        print(f"🌐 Space URL: https://huggingface.co/spaces/{REPO_ID}")

if __name__ == "__main__":
    main()
