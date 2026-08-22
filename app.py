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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Playfair+Display:ital,wght@0,600;0,800;0,900;1,600;1,800&family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Modak&family=Rozha+One&family=Yatra+One&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {
    darkMode: 'class',
    theme: {
      extend: {
        colors: {
          hhjungle: {
            950: '#03140D',
            900: '#062319',
            850: '#082E20',
            800: '#0B3B2A',
            700: '#10523B',
            600: '#177353',
          },
          hhgold: {
            DEFAULT: '#FFE600',
            glow: '#FFF066',
            dark: '#E5CE00',
            warm: '#F4B942',
          },
          hhpink: {
            DEFAULT: '#FF2E93',
            glow: '#FF5CAE',
            dark: '#D60D70',
          },
          hhemerald: {
            DEFAULT: '#00E599',
            glow: '#33EBAD',
            dark: '#00B377',
          },
        },
        fontFamily: {
          display: ['"Cinzel"', 'serif'],
          serif: ['"Playfair Display"', 'Georgia', 'serif'],
          mono: ['"Space Mono"', 'monospace'],
          sans: ['"Plus Jakarta Sans"', 'sans-serif'],
          devanagari: ['"Modak"', '"Rozha One"', '"Yatra One"', 'sans-serif'],
        },
        boxShadow: {
          'brutal': '4px 4px 0px #000000',
          'brutal-lg': '6px 6px 0px #000000',
          'brutal-sm': '2px 2px 0px #000000',
          'brutal-pink': '4px 4px 0px #FF2E93',
          'brutal-gold': '4px 4px 0px #FFE600',
          'brutal-emerald': '4px 4px 0px #00E599',
          'glow-gold': '0 0 25px rgba(255, 230, 0, 0.25)',
          'glow-pink': '0 0 25px rgba(255, 46, 147, 0.35)',
          'glow-emerald': '0 0 25px rgba(0, 229, 153, 0.25)',
        }
      }
    }
  }
</script>
"""

CUSTOM_CSS = """
:root {
    color-scheme: dark !important;
}

html, body, .gradio-container, gradio-app, #root, .main {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
    width: 100vw !important;
    height: 100vh !important;
    background-color: #03140D !important;
    color: #F8FAFC !important;
    color-scheme: dark !important;
    overflow: hidden !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.gradio-container {
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    max-width: 100% !important;
    width: 100vw !important;
    height: 100vh !important;
}

.contain {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    height: 100% !important;
    border: none !important;
}

.prose, .prose *, .not-prose, .not-prose * {
    color: inherit;
    max-width: none !important;
}

footer, .built-with, gradio-app > footer, .gradio-container > footer {
    display: none !important;
}

/* Force dark background and remove Gradio white boxes */
#chatInput {
    background: transparent !important;
    background-color: transparent !important;
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

#chatInput::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
}

#langSelect, #voiceOutLangSelect {
    background-color: #062319 !important;
    background: #062319 !important;
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
    border: 1px solid rgba(0, 0, 0, 0.8) !important;
    padding: 2px 8px !important;
    border-radius: 9999px !important;
}

#langSelect option, #voiceOutLangSelect option {
    background-color: #03140D !important;
    color: #F8FAFC !important;
}

#tabChatBtn, #tabSeaBtn, #tabSpeedBtn {
    border: none !important;
    outline: none !important;
}
"""


def get_custom_html() -> str:
    demo_file = config.BASE_DIR / "demo" / "index.html"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            content = f.read()
        if "<body" in content and "</body>" in content:
            body_start = content.find("<body")
            body_tag_end = content.find(">", body_start) + 1
            body_end = content.rfind("</body>")
            inner_body = content[body_tag_end:body_end]
            return f'<div class="h-screen w-screen flex bg-tropical-canvas selection:bg-hhgold selection:text-black overflow-hidden font-sans">{inner_body}</div>'
        return content
    return "<h1>Hacker House Goa 2026 Command Center</h1>"


@spaces.GPU
def _dummy_zerogpu():
    """ZeroGPU requirement: at least one function registered to event scan."""
    return True


with gr.Blocks(title="🌴 Hacker House Goa 2026 — Voice Indic RAG", css=CUSTOM_CSS, head=HEAD_HTML) as demo:
    gr.HTML(get_custom_html(), elem_classes=["not-prose"])
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
    demo.queue().launch(server_name=host, server_port=port, ssr=False)
