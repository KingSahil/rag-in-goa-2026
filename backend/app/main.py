import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.models import TextAsk
from app.retrieval import Retriever
from app.harness import RAGHarness
from app.stt import transcribe

app = FastAPI(title="RAGInGoa 2026 Voice RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = SentenceTransformer(settings.embed_model)
retriever = Retriever(settings.index_dir, embedder, settings.top_k_dense, settings.top_k_bm25)
harness = RAGHarness(retriever)

@app.get("/health")
def health():
    return {"ok": True, "languages": sorted(retriever.shards.keys())}

@app.post("/ask/text")
async def ask_text(body: TextAsk):
    result = await harness.run(body.query, body.language)
    return result

@app.post("/ask/voice")
async def ask_voice(file: UploadFile = File(...)):
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio")

    t0 = time.perf_counter()
    stt = await transcribe(audio, file.filename or "audio.wav")
    stt_ms = (time.perf_counter() - t0) * 1000
    result = await harness.run(stt["transcript"], stt.get("language_code"))

    result["transcript"] = stt["transcript"]
    result["detected_language"] = stt.get("language_code")
    result["timings_ms"]["stt_ms"] = stt_ms
    result["timings_ms"]["voice_total_ms"] = stt_ms + result["timings_ms"]["total"]
    return result
