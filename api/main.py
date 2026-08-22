"""
FastAPI Serving Layer for Voice-Enabled Indic RAG System.

Endpoints:
- POST /query: Accepts audio file upload (multipart/form-data) OR JSON text bypass. Returns QueryResponse.
- POST /tts: Synthesizes spoken audio from text using Sarvam AI Bulbul TTS.
- GET /health: Healthcheck reporting active languages, index vector counts, and model status.
- GET /languages: Returns active languages strictly derived from config.LANGUAGES.
- GET /: Web Demo UI.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from pipeline.orchestrator import get_orchestrator, RAGPipelineOrchestrator
from pipeline.schemas import QueryRequest, QueryResponse
from retrieval.embed import get_embedder
from retrieval.index_faiss import get_index_manager
from stt.sarvam_tts import synthesize_speech

logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    """Payload for Text-to-Speech audio generation."""
    text: str
    target_language: str = "hi"
    speaker: Optional[str] = None
    pace: float = 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler: Preload embedding models, FAISS indexes, and
    eagerly initialize and warm up the full RAG pipeline orchestrator ONCE at startup.
    Never re-loads per request. Self-heals if indexes are missing or empty.
    """
    print("[API Lifespan] Initializing and pre-loading embedding model...")
    embedder = get_embedder()
    
    print("[API Lifespan] Loading FAISS HNSW indexes and corpus centroids into memory...")
    index_mgr = get_index_manager()
    
    # Ensure indexes are populated
    for name, idx in index_mgr.indexes.items():
        print(f"[API Lifespan] Index '{name}': {idx.index.ntotal} vectors loaded (configured cap: {config.MAX_INDEX_PASSAGES_PER_LANG}).")
        
    print("[API Lifespan] Eagerly initializing and warming up RAG orchestrator & ONNX models...")
    orchestrator = get_orchestrator()
    print("[API Lifespan] RAG orchestrator warmup complete.")
    
    print(f"[API Lifespan] Initialized successfully. Active languages: {config.LANGUAGES}")
    print(f"[API Lifespan] Allow Network Calls Switch: {config.ALLOW_NETWORK_CALLS_IN_PIPELINE}")
    print(f"[API Lifespan] Request Timeout Deadline: {config.REQUEST_TIMEOUT_SECONDS}s")
    
    yield
    
    print("[API Lifespan] Shutting down RAG service.")


app = FastAPI(
    title="Voice-Enabled Indic RAG API",
    description="Instrumented low-latency Voice RAG pipeline for Indic languages (English, Hindi, Tamil, Marathi)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for demo interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "allow_network_calls": config.ALLOW_NETWORK_CALLS_IN_PIPELINE,
        "sarvam_stt_configured": bool(config.SARVAM_API_KEY),
        "sarvam_tts_configured": bool(config.SARVAM_API_KEY),
        "llm_fallback_configured": bool(config.LLM_API_KEY and config.ALLOW_NETWORK_CALLS_IN_PIPELINE),
        "semantic_answer_cache_configured": config.SEMANTIC_ANSWER_CACHE_ENABLED,
        "request_timeout_seconds": config.REQUEST_TIMEOUT_SECONDS,
        "query_intent_filter_enabled": config.ENABLE_QUERY_INTENT_FILTER,
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


@app.post("/query", response_model=QueryResponse)
@app.post("/api/query", response_model=QueryResponse)
async def query_pipeline(request: Request) -> QueryResponse:
    """
    Execute end-to-end Voice RAG query with strict deadline timeout protection.
    Accepts:
    1. JSON body with text / audio_path / audio_base64 / language_hint / cross_lingual
    2. Multipart form data with file / text / language_hint / cross_lingual
    """
    orchestrator = get_orchestrator()
    temp_audio_path = None
    
    try:
        req: Optional[QueryRequest] = None
        content_type = request.headers.get("content-type", "").lower()
        
        if "application/json" in content_type:
            body = await request.json()
            if body.get("audio_base64"):
                import base64
                audio_bytes = base64.b64decode(body["audio_base64"])
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    temp_audio_path = tmp.name
            req = QueryRequest(
                audio_path=temp_audio_path or body.get("audio_path"),
                text=body.get("text"),
                language_hint=body.get("language_hint"),
                cross_lingual=bool(body.get("cross_lingual", False)),
                bypass_cache=bool(body.get("bypass_cache", False)),
            )
        else:
            form = await request.form()
            file = form.get("file")
            text = form.get("text")
            language_hint = form.get("language_hint")
            cross_lingual = form.get("cross_lingual")
            if file and hasattr(file, "file"):
                suffix = Path(getattr(file, "filename", "audio.wav")).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(file.file, tmp)
                    temp_audio_path = tmp.name
                req = QueryRequest(
                    audio_path=temp_audio_path,
                    language_hint=str(language_hint) if language_hint else None,
                    cross_lingual=str(cross_lingual).lower() in ("true", "1") if cross_lingual is not None else False,
                )
            elif text and str(text).strip():
                req = QueryRequest(
                    text=str(text).strip(),
                    language_hint=str(language_hint) if language_hint else None,
                    cross_lingual=str(cross_lingual).lower() in ("true", "1") if cross_lingual is not None else False,
                )

        if not req or (not req.text and not req.audio_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file' audio upload, 'audio_base64', or 'text' query must be provided.",
            )

        # Enforce request deadline with asyncio.wait_for
        try:
            return await asyncio.wait_for(
                orchestrator.execute(req),
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Query execution exceeded request deadline timeout ({config.REQUEST_TIMEOUT_SECONDS}s)"
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Request execution exceeded {config.REQUEST_TIMEOUT_SECONDS}s deadline.",
            )
            
    finally:
        # Clean up temporary audio file if created
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


@app.post("/tts")
@app.post("/api/tts")
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


@app.get("/", response_class=HTMLResponse)
async def serve_demo_ui():
    """Serves the interactive voice-enabled web demo."""
    demo_file = config.BASE_DIR / "demo" / "index.html"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Voice-Enabled Indic RAG Service</h1><p>API is running. Visit <a href='/docs'>/docs</a> for API specification.</p>"
