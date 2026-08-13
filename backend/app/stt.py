import httpx
from app.config import settings

async def transcribe(audio_bytes: bytes, filename: str = "audio.webm"):
    if not settings.sarvam_key:
        raise RuntimeError("SARVAM_API_KEY is missing")

    lower = (filename or "").lower()
    mime = "audio/webm" if lower.endswith(".webm") else "audio/wav"
    files = {"file": (filename or "audio.webm", audio_bytes, mime)}
    data = {
        "model": "saaras:v3",
        "mode": "transcribe",
        "language_code": "unknown",
    }
    headers = {"api-subscription-key": settings.sarvam_key}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            "https://api.sarvam.ai/speech-to-text",
            files=files,
            data=data,
            headers=headers,
        )
        r.raise_for_status()
        return r.json()
