import json
import httpx

from app.config import settings

SYSTEM = """You are a grounded RAG answerer.
Rules:
- Answer ONLY from the supplied context.
- If the context does not contain enough evidence, say you cannot answer from the provided dataset.
- Never invent facts, numbers, names, dates, citations, or steps.
- Keep answers concise.
- Return strict JSON with keys: answer, grounded.
"""

async def generate(query: str, contexts: list[dict]) -> dict:
    if not settings.sarvam_key:
        raise RuntimeError("SARVAM_API_KEY is missing")

    context_text = "\n\n".join(
        f"[SOURCE {i+1}] {c['text']}" for i, c in enumerate(contexts)
    )
    payload = {
        "model": settings.sarvam_chat_model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_text}"},
        ],
        "temperature": 0,
        "reasoning_effort": None,
        "max_tokens": settings.gen_max_tokens,
        "stream": False,
    }
    headers = {
        "api-subscription-key": settings.sarvam_key,
        "Authorization": f"Bearer {settings.sarvam_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=settings.gen_timeout_s) as client:
        r = await client.post("https://api.sarvam.ai/v1/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()

    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        # Robustness fallback if the remote model returns a JSON code fence.
        cleaned = content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
