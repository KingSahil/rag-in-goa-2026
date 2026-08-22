"""
Hugging Face Space Application for Hacker House Goa 2026: Voice-Enabled Indic RAG.
Production-grade FastAPI server with ZeroGPU-mounted Gradio interface.
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# Compatibility shim for huggingface_hub
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class DummyHfFolder:
            @staticmethod
            def get_token():
                return os.environ.get("HF_TOKEN") or None
            @staticmethod
            def save_token(token):
                pass
            @staticmethod
            def delete_token():
                pass
        huggingface_hub.HfFolder = DummyHfFolder
except Exception:
    pass

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ZeroGPU decorator shim
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

import config
from pipeline.orchestrator import get_orchestrator
from pipeline.schemas import QueryRequest, QueryResponse
from retrieval.index_faiss import get_index_manager
from stt.sarvam_tts import synthesize_speech


# ── Preload models at startup ──────────────────────────────────────────
print("[Space Startup] Preloading embedding model, FAISS indexes, and warming up pipeline...")
orchestrator = get_orchestrator()
orchestrator.warmup_pipeline()
print("[Space Startup] Full RAG pipeline preloaded and ready for traffic.")


# ── Create Main FastAPI App ───────────────────────────────────────────
app = FastAPI(
    title="Hacker House Goa 2026 - Voice Indic RAG API",
    description="Sub-200ms Voice Indic RAG over 15 Indian Languages",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_demo_html() -> str:
    """Read the custom Command Center HTML file."""
    demo_file = config.BASE_DIR / "demo" / "index.html"
    if demo_file.exists():
        return demo_file.read_text(encoding="utf-8")
    return "<h1>Command Center UI Not Found</h1>"


@app.get("/", response_class=HTMLResponse)
@app.get("/demo-ui", response_class=HTMLResponse)
async def serve_root_ui():
    """Serves the standalone ChatGPT-style Command Center HTML UI directly."""
    return HTMLResponse(content=_read_demo_html())


# ── Health & Language Metadata ─────────────────────────────────────────
async def _health_check():
    idx_mgr = get_index_manager()
    return {
        "status": "healthy",
        "configured_languages": config.LANGUAGES,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "indexes_loaded": {n: i.index.ntotal for n, i in idx_mgr.indexes.items()},
        "centroids_available": list(idx_mgr.centroids.keys()),
        "sarvam_configured": bool(config.SARVAM_API_KEY),
    }


async def _get_supported_languages():
    return {
        "active_languages": config.LANGUAGES,
        "language_details": [{"code": l, **config.get_language_info(l)} for l in config.LANGUAGES],
    }


# ── Query Pipeline ───────────────────────────────────────────────────
def _parse_bool(val, default=True):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return bool(val)


async def _run_query(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language_hint: Optional[str] = Form(None),
    cross_lingual: Optional[str] = Form(None),
    bypass_cache: Optional[str] = Form(None),
    request_body: Optional[QueryRequest] = None,
):
    """Execute the 9-stage Voice RAG pipeline."""
    orch = get_orchestrator()
    temp_path = None
    try:
        if request_body:
            return await orch.execute(request_body)

        if file and file.filename:
            suffix = Path(file.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name
            req = QueryRequest(
                audio_path=temp_path,
                language_hint=language_hint,
                cross_lingual=_parse_bool(cross_lingual, False),
                bypass_cache=_parse_bool(bypass_cache, False),
            )
            return await orch.execute(req)

        if text and text.strip():
            req = QueryRequest(
                text=text.strip(),
                language_hint=language_hint,
                cross_lingual=_parse_bool(cross_lingual, False),
                bypass_cache=_parse_bool(bypass_cache, False),
            )
            return await orch.execute(req)

        raise HTTPException(status_code=400, detail="Provide 'file' or 'text'.")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ── TTS ───────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    target_language: str = "hi"
    language: Optional[str] = None
    speaker: Optional[str] = None
    pace: float = 1.0


async def _tts(req: TTSRequest):
    lang = req.language or req.target_language
    try:
        return synthesize_speech(text=req.text, language_code=lang, speaker=req.speaker, pace=req.pace)
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Register all routes on both / and /api/ prefixes ──────────────────
for prefix in ["", "/api"]:
    app.add_api_route(f"{prefix}/health", _health_check, methods=["GET"], response_class=JSONResponse)
    app.add_api_route(f"{prefix}/languages", _get_supported_languages, methods=["GET"], response_class=JSONResponse)
    app.add_api_route(f"{prefix}/query", _run_query, methods=["POST"], response_model=QueryResponse)
    app.add_api_route(f"{prefix}/tts", _tts, methods=["POST"])


# ── Mount ZeroGPU-ready Gradio interface at /gradio ───────────────────
@spaces.GPU
def _zerogpu_anchor():
    """ZeroGPU anchor function to reserve GPU compute."""
    return True


with gr.Blocks(title="🌴 Hacker House Goa 2026 - Voice Indic RAG") as demo:
    gr.Markdown("# 🌴 Hacker House Goa 2026: Voice Indic RAG")
    gr.HTML("<p>Command Center UI is running at <a href='/'>/ (Root)</a></p>")
    _btn = gr.Button("ZeroGPU Anchor", visible=False)
    _btn.click(fn=_zerogpu_anchor)

app = gr.mount_gradio_app(app, demo, path="/gradio")


# ── Launch Entrypoint ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[Server] Command Center running on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
