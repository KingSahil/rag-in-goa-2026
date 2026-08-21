#!/usr/bin/env python3
"""
🌴 Hugging Face Spaces Automated Deployment Tool
================================================
Deploys the Voice-Enabled Indic RAG system to Hugging Face Spaces (Docker SDK).

Usage:
  python deploy_hf.py --repo-id <username>/<space-name> [--token <hf_token>]
  python deploy_hf.py --check

Examples:
  python deploy_hf.py --repo-id your-username/voice-indic-rag-goa2026
  python deploy_hf.py --repo-id your-username/voice-indic-rag-goa2026 --sync-secrets
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from huggingface_hub import HfApi, whoami
except ImportError:
    print("[ERROR] 'huggingface_hub' is not installed.")
    print("👉 Install it with: pip install huggingface_hub (or run with .venv/Scripts/python)")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent

IGNORE_PATTERNS = [
    ".venv/**",
    "venv/**",
    "env/**",
    "node_modules/**",
    "dist/**",
    "__pycache__/**",
    "*.pyc",
    ".git/**",
    ".pytest_cache/**",
    ".env",
    ".env.local",
    "data/raw/**",
    "data/onnx_models/**",
    "tests/**",
    ".github/**",
    "benchmark/results/**",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy Voice-Enabled Indic RAG to Hugging Face Spaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.getenv("HF_SPACE_REPO_ID", ""),
        help="Target Space Repository ID (e.g., 'username/voice-indic-rag')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN", ""),
        help="Hugging Face User Access Token (with write permission). Defaults to $HF_TOKEN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create space as private (default: public)",
    )
    parser.add_argument(
        "--sync-secrets",
        action="store_true",
        default=True,
        help="Automatically configure SARVAM_API_KEY and LLM_API_KEY from .env into Space secrets",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify deployment prerequisites (local files, indexes, and credentials) without deploying",
    )
    return parser.parse_args()


def check_prerequisites():
    """Verify local files, index artifacts, and basic readiness."""
    print("\n🔍 Checking Local Deployment Prerequisites...")
    errors = []
    warnings = []

    # 1. Check Dockerfile
    dockerfile = PROJECT_ROOT / "Dockerfile"
    if not dockerfile.exists():
        errors.append("Dockerfile missing in project root.")
    else:
        print("  [OK] Dockerfile found.")

    # 2. Check README.md frontmatter
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():
        errors.append("README.md missing in project root.")
    else:
        content = readme.read_text(encoding="utf-8")
        if "sdk: docker" in content or "sdk: gradio" in content:
            print("  [OK] README.md contains Hugging Face Space metadata.")
        else:
            warnings.append("README.md does not seem to contain standard Hugging Face YAML metadata.")

    # 3. Check FAISS index files
    index_file = PROJECT_ROOT / "data" / "indexes" / "passage_native.faiss"
    meta_file = PROJECT_ROOT / "data" / "indexes" / "passage_native_meta.json"
    if not index_file.exists():
        warnings.append(f"FAISS index '{index_file.name}' not found. Space will build it at startup (may be slow).")
    else:
        size_mb = index_file.stat().st_size / (1024 * 1024)
        print(f"  [OK] FAISS Index '{index_file.name}' found ({size_mb:.1f} MB).")

    if not meta_file.exists():
        warnings.append(f"Index Metadata '{meta_file.name}' not found.")
    else:
        size_mb = meta_file.stat().st_size / (1024 * 1024)
        print(f"  [OK] Index Metadata '{meta_file.name}' found ({size_mb:.1f} MB).")

    # 4. Check requirements.txt
    reqs = PROJECT_ROOT / "requirements.txt"
    if not reqs.exists():
        errors.append("requirements.txt missing.")
    else:
        print("  [OK] requirements.txt found.")

    # 5. Check Command Center UI
    demo_ui = PROJECT_ROOT / "demo" / "index.html"
    if not demo_ui.exists():
        errors.append("Command Center UI 'demo/index.html' missing.")
    else:
        print("  [OK] Command Center UI 'demo/index.html' found.")

    print()
    for w in warnings:
        print(f"  [WARN] WARNING: {w}")
    for e in errors:
        print(f"  [ERROR] ERROR: {e}")

    if errors:
        print("\n[FAILED] Prerequisites check failed. Please resolve the errors above.")
        return False
    
    print("[SUCCESS] All prerequisite checks passed successfully!\n")
    return True


def main():
    args = parse_args()
    print("=" * 70)
    print("🌴 Voice-Enabled Multilingual Indic RAG - Hugging Face Space Deployment")
    print("=" * 70)

    if not check_prerequisites():
        if not args.check:
            sys.exit(1)

    if args.check:
        print("Check completed.")
        return

    # 1. Resolve HF Token
    token = args.token.strip()
    if not token:
        token = input("🔑 Enter your Hugging Face Access Token (write permission): ").strip()
    
    if not token:
        print("[ERROR] Hugging Face token is required for deployment.")
        sys.exit(1)

    # 2. Verify Token & User Identity
    api = HfApi(token=token)
    try:
        user_info = whoami(token=token)
        username = user_info.get("name") or user_info.get("fullname", "User")
        print(f"[AUTH] Authenticated as: {username} ({user_info.get('type', 'user')})")
    except Exception as exc:
        print(f"[ERROR] Authentication failed: {exc}")
        print("👉 Please check your Hugging Face token at https://huggingface.co/settings/tokens")
        sys.exit(1)

    # 3. Resolve Target Repo ID
    repo_id = args.repo_id.strip()
    if not repo_id:
        default_name = f"{username}/hackerhouse-goa-indic-rag"
        repo_input = input(f"📦 Enter target Space name [{default_name}]: ").strip()
        repo_id = repo_input if repo_input else default_name

    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    print(f"\n🚀 Deploying to Space: https://huggingface.co/spaces/{repo_id}")

    # 4. Create Space if it doesn't exist
    try:
        print(f"📦 Checking/Creating Space repository '{repo_id}' (Gradio SDK)...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=args.private,
            exist_ok=True,
        )
        print(f"  [OK] Space repository ready.")
    except Exception as exc:
        print(f"[ERROR] Failed to create/verify Space: {exc}")
        sys.exit(1)

    # 5. Sync Secrets (SARVAM_API_KEY, LLM_API_KEY)
    if args.sync_secrets:
        secrets_to_sync = {
            "SARVAM_API_KEY": os.getenv("SARVAM_API_KEY", ""),
            "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        }
        for sec_key, sec_val in secrets_to_sync.items():
            if sec_val:
                try:
                    print(f"🔐 Setting Space Secret '{sec_key}'...")
                    api.add_space_secret(repo_id=repo_id, key=sec_key, value=sec_val)
                    print(f"  [OK] Secret '{sec_key}' configured.")
                except Exception as exc:
                    print(f"  [WARN] Could not set secret '{sec_key}': {exc}")

    # 6. Upload Project Files (including large LFS FAISS index files)
    print(f"\n📤 Uploading project files and vector indexes to Hugging Face...")
    print("   (Large files will automatically be tracked with Git LFS)")
    try:
        commit_info = api.upload_folder(
            folder_path=str(PROJECT_ROOT),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=IGNORE_PATTERNS,
            commit_message="🚀 Deploy Voice Indic RAG Command Center (Hacker House Goa 2026)",
        )
        print("\n" + "=" * 70)
        print("[SUCCESS] DEPLOYMENT SUCCESSFUL!")
        print("=" * 70)
        print(f"🌐 Live Space URL:  https://huggingface.co/spaces/{repo_id}")
        print(f"📜 Commit Details:  {commit_info}")
        print("\n💡 The container image will now build automatically on Hugging Face.")
        print("   You can monitor build logs in real-time under the 'Logs' tab on your Space page.")
        print("=" * 70)
    except Exception as exc:
        print(f"\n[ERROR] Error during upload: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
