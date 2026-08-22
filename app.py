"""
Hugging Face Space Application for Hacker House Goa 2026: Voice-Enabled Indic RAG.
Renders the full retro-tropical Command Center UI and exposes FastAPI endpoints.
ZeroGPU compatible.
"""

import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Compatibility shim for older packages importing HfFolder from huggingface_hub
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class DummyHfFolder:
            @staticmethod
            def get_token():
                import os
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
from fastapi import File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# ZeroGPU decorator shim
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func

from pydantic import BaseModel
import config
from pipeline.orchestrator import get_orchestrator
from pipeline.schemas import QueryRequest, QueryResponse
from retrieval.embed import get_embedder
from retrieval.index_faiss import get_index_manager
from stt.sarvam_tts import synthesize_speech


HEAD_HTML = """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
  html, body, .gradio-container, gradio-app, #root, .main {
    margin: 0 !important;
    padding: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    max-width: 100vw !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    background-color: #03140D !important;
  }
  footer, .built-with, gradio-app > footer, .gradio-container > footer {
    display: none !important;
  }
  iframe.app-frame {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    border: none;
    margin: 0;
    padding: 0;
    z-index: 999999;
    background: #03140D;
  }
</style>
"""

CUSTOM_CSS = """
gradio-app, .gradio-container, body, html {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
    width: 100vw !important;
    height: 100vh !important;
    overflow: hidden !important;
    background-color: #03140D !important;
}
footer, .built-with {
    display: none !important;
}
"""


@spaces.GPU
def _dummy_zerogpu():
    """ZeroGPU requirement: at least one function registered to event scan."""
    return True


with gr.Blocks(title="🌴 Hacker House Goa 2026 — Voice Indic RAG", css=CUSTOM_CSS, head=HEAD_HTML) as demo:
    gr.HTML("""
    <iframe class="app-frame"
            src="/app_ui" 
            allow="microphone; camera; autoplay; clipboard-write; fullscreen" 
            style="position:fixed; top:0; left:0; width:100vw; height:100vh; border:none; margin:0; padding:0; z-index:999999; overflow:hidden; background:#03140D;">
    </iframe>
    """, elem_classes=["not-prose"])
    # Hidden dummy button to ensure ZeroGPU handler registration
    dummy_btn = gr.Button("zero_gpu_anchor", visible=False)
    dummy_btn.click(fn=_dummy_zerogpu)



# Preload models and perform full pipeline warmup at startup
print("[Space Startup] Preloading embedding model, FAISS indexes, and warming up pipeline...")
orchestrator = get_orchestrator()
orchestrator.warmup_pipeline()
print("[Space Startup] Full RAG pipeline preloaded and warmed up successfully.")


# Attach FastAPI endpoints directly to demo.app
app = demo.app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/app_ui", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def serve_command_center_ui():
    """Serves the isolated retro-tropical ChatGPT-style Command Center interface."""
    demo_file = config.BASE_DIR / "demo" / "index.html"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Command Center UI Not Found</h1>", status_code=404)


@app.get("/health", response_class=JSONResponse)
async def health_check() -> Dict[str, Any]:
    """Health check reporting system and index readiness."""
    index_mgr = get_index_manager()
    index_stats = {
        name: idx.index.ntotal for name, idx in index_mgr.indexes.items()
    }
    return {
        "status": "healthy",
        "configured_languages": config.LANGUAGES,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "indexes_loaded": index_stats,
        "centroids_available": list(index_mgr.centroids.keys()),
        "sarvam_stt_configured": bool(config.SARVAM_API_KEY),
        "llm_fallback_configured": bool(config.LLM_API_KEY),
    }


@app.get("/languages", response_class=JSONResponse)
async def get_supported_languages() -> Dict[str, Any]:
    """Returns metadata for all currently configured active languages."""
    lang_details = [
        {"code": l, **config.get_language_info(l)} for l in config.LANGUAGES
    ]
    return {
        "active_languages": config.LANGUAGES,
        "language_details": lang_details,
    }


def _parse_bool(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return bool(val)


@app.post("/query", response_model=QueryResponse)
async def query_pipeline(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language_hint: Optional[str] = Form(None),
    cross_lingual: Optional[Any] = Form(None),
    bypass_cache: Optional[Any] = Form(None),
    request_body: Optional[QueryRequest] = None,
) -> QueryResponse:
    """
    Execute end-to-end Voice RAG query for the Command Center UI.
    """
    orchestrator = get_orchestrator()
    temp_audio_path = None
    is_cross_lingual = _parse_bool(cross_lingual, default=False)
    is_bypass_cache = _parse_bool(bypass_cache, default=False)
    
    try:
        if request_body and (request_body.text or request_body.audio_path):
            return await orchestrator.execute(request_body)
            
        if file and file.filename:
            suffix = Path(file.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_audio_path = tmp.name
                
            req = QueryRequest(
                audio_path=temp_audio_path,
                language_hint=language_hint,
                cross_lingual=is_cross_lingual,
                bypass_cache=is_bypass_cache,
            )
            return await orchestrator.execute(req)
            
        if text and text.strip():
            req = QueryRequest(
                text=text.strip(),
                language_hint=language_hint,
                cross_lingual=is_cross_lingual,
                bypass_cache=is_bypass_cache,
            )
            return await orchestrator.execute(req)
            
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file' audio upload or 'text' query must be provided.",
        )
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


class TTSRequest(BaseModel):
    """Payload for Text-to-Speech audio generation."""
    text: str
    target_language: str = "hi"
    speaker: Optional[str] = None
    pace: float = 1.0


@app.post("/tts")
async def generate_speech_audio(req: TTSRequest):
    """
    Synthesizes speech audio from text using Sarvam Bulbul TTS.
    Returns JSON containing audio base64 or fallback status.
    """
    try:
        res = synthesize_speech(
            text=req.text,
            language_code=req.target_language,
            speaker=req.speaker,
            pace=req.pace,
        )
        return res
    except Exception as e:
        return {"status": "error", "message": str(e)}



if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"🌴 Starting Hacker House Goa Command Center UI on http://{host}:{port}")
    demo.queue().launch(server_name=host, server_port=port)
