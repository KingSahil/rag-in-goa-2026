# 🚀 Hugging Face Spaces Deployment Guide

This guide walks you through deploying the **Voice-Enabled Multilingual Indic RAG** system and its **retro-tropical Command Center UI** to **Hugging Face Spaces**.

---

## 🏛️ Space Configuration Summary

| Property | Value |
|---|---|
| **Space SDK** | `docker` |
| **Port** | `7860` |
| **Default Model** | `intfloat/multilingual-e5-small` |
| **Vector Index** | In-Memory FAISS HNSW (`148,854` vectors, ~256MB) |
| **Hardware** | Free CPU Basic (2 vCPU, 16GB RAM) or CPU Upgrade |
| **Required Secrets** | `SARVAM_API_KEY` (for Saaras v3 STT & Bulbul v2 TTS) |
| **Optional Secrets** | `GROQ_API_KEY` or `LLM_API_KEY` (for generative fallback) |

---

## ⚡ Method 1: Automated CLI Deployment (Recommended)

An automated deployment tool is included at [`deploy_hf.py`](file:///c:/Projects/rag-ingoa-2026/deploy_hf.py). It validates prerequisites, creates the Space repo, sets secrets, and uploads all files (including FAISS LFS artifacts) in one command.

### Step 1: Get your Hugging Face Write Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Create a new token with **Write** permissions (or use an existing one).

### Step 2: Run the Deployment Script

Activate your virtual environment and run:

```bash
# Verify local files and prerequisites first
python deploy_hf.py --check

# Deploy to Hugging Face
python deploy_hf.py --repo-id <YOUR_HF_USERNAME>/voice-indic-rag
```

You will be prompted for your Hugging Face token (or you can provide it via `--token <HF_TOKEN>` or export `HF_TOKEN=<token>`).

The script will automatically:
1. Authenticate with Hugging Face API.
2. Create the Space repo (`space_sdk="docker"`) if it doesn't already exist.
3. Sync `SARVAM_API_KEY` and `LLM_API_KEY` from your local `.env` into Space Secrets.
4. Upload all project code, Dockerfile, UI, and FAISS index files (with automatic Git LFS handling).

---

## 🛠️ Method 2: Manual Git CLI Deployment

If you prefer using `git` and `huggingface-cli`:

### Step 1: Install Git LFS & Hugging Face CLI
```bash
git lfs install
pip install huggingface_hub
huggingface-cli login
```

### Step 2: Create Space on Hugging Face
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set Space Name (e.g. `voice-indic-rag`).
3. Select **Space SDK: Docker** (Blank).
4. Choose **Public** or **Private**.

### Step 3: Add HF Remote & Push
```bash
# Add Hugging Face Space as a git remote
git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>

# Ensure Git LFS is tracking large artifacts
git lfs track "*.faiss"
git lfs track "data/indexes/*.json"
git lfs track "*.jsonl"

# Add and commit files
git add .
git commit -m "Deploy to Hugging Face Spaces"

# Push to Hugging Face Space
git push space main --force
```

---

## 🔄 Method 3: GitHub Actions CI/CD (Continuous Deployment)

To automatically deploy to Hugging Face Spaces whenever you push to your GitHub repository:

Create `.github/workflows/deploy_hf.yml`:

```yaml
name: Sync to Hugging Face Spaces

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true

      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add space https://<YOUR_USERNAME>:$HF_TOKEN@huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>
          git push --force space main
```

Add `HF_TOKEN` in your GitHub Repository under **Settings > Secrets and variables > Actions**.

---

## 🔐 Configuring Secrets & Environment Variables

Ensure the following variables are configured in **Space Settings > Variables and secrets**:

| Secret Name | Description | Required |
|---|---|:---:|
| `SARVAM_API_KEY` | Sarvam AI API key for real-time STT (Saaras v3) and TTS (Bulbul v2) | **Yes** (for Voice) |
| `LLM_API_KEY` | Groq / OpenAI compatible API key for generative fallback | Optional |
| `GROQ_API_KEY` | Groq Cloud API key (if using Groq) | Optional |

---

## 📊 Live Verification & Health Checks

Once deployed, you can verify your running Space:

1. **Web UI**: Visit `https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>` to interact with the tropical Command Center.
2. **Health Check Endpoint**:
   ```bash
   curl https://<YOUR_USERNAME>-<YOUR_SPACE_NAME>.hf.space/health
   ```
   **Expected Response**:
   ```json
   {
     "status": "healthy",
     "configured_languages": ["en", "hi", "ta", "mr"],
     "embedding_model": "intfloat/multilingual-e5-small",
     "indexes_loaded": {
       "passage_native": 148545,
       "semantic_longdoc": 309
     },
     "centroids_available": ["en", "hi", "mr", "global"]
   }
   ```
3. **Interactive Swagger Docs**: Visit `https://<YOUR_USERNAME>-<YOUR_SPACE_NAME>.hf.space/docs` to test `/query` and `/tts` endpoints directly.

---

## 🔍 Troubleshooting

### 1. Build Fails during `pip install` or `AutoModel.from_pretrained`
- Ensure the Space has internet access during the Docker build stage.
- The `Dockerfile` includes `curl`, `build-essential`, and `libsndfile1` to ensure audio parsing dependencies build cleanly.

### 2. Large File Push Rejected (>100MB)
- Hugging Face requires Git LFS for files >10MB/100MB.
- Ensure `.gitattributes` contains `*.faiss` and `data/indexes/*.json`.
- When using `deploy_hf.py`, `HfApi.upload_folder` automatically handles LFS uploads transparently.

### 3. Microphone Access inside Hugging Face Spaces iframe
- If accessing the Space via the embedded Hugging Face iframe, your browser might request microphone permissions for `hf.space`. Allow microphone access or open the direct Space URL: `https://<username>-<space_name>.hf.space`.
