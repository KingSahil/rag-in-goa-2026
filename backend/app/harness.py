import asyncio
import time
from statistics import median

from app.config import settings
from app.guardrails import check_input, should_refuse, grounding_score
from app.llm import generate

class RAGHarness:
    def __init__(self, retriever):
        self.retriever = retriever

    async def run(self, query: str, language: str | None = None):
        t0 = time.perf_counter()
        ok, reason = check_input(query)
        if not ok:
            return {
                "answer": "I can’t help with that request.",
                "language": language or "unknown",
                "grounded": False,
                "refused": True,
                "refusal_reason": reason,
                "sources": [],
                "timings_ms": {"total": (time.perf_counter() - t0) * 1000},
            }

        t1 = time.perf_counter()
        results = await asyncio.wait_for(
            asyncio.to_thread(self.retriever.retrieve, query, language, settings.top_k_final),
            timeout=settings.rag_timeout_s,
        )
        t2 = time.perf_counter()

        refuse, reason = should_refuse(results)
        if refuse:
            return {
                "answer": "I couldn’t find enough relevant evidence in the dataset to answer that.",
                "language": language or "unknown",
                "grounded": False,
                "refused": True,
                "refusal_reason": reason,
                "sources": results,
                "timings_ms": {
                    "retrieval_ms": (t2 - t1) * 1000,
                    "total": (t2 - t0) * 1000,
                },
            }

        t3 = time.perf_counter()
        generated = await generate(query, results)
        t4 = time.perf_counter()

        answer = str(generated.get("answer", "")).strip()
        gscore = grounding_score(answer, [r["text"] for r in results])
        grounded = bool(generated.get("grounded", False)) and gscore >= 0.10

        if not grounded:
            answer = "I couldn’t verify that answer against the retrieved dataset."
            refused = True
            reason = "grounding_check_failed"
        else:
            refused = False
            reason = None

        return {
            "answer": answer,
            "language": language or (results[0]["language"] if results else "unknown"),
            "grounded": grounded,
            "refused": refused,
            "refusal_reason": reason,
            "sources": results,
            "timings_ms": {
                "retrieval_ms": (t2 - t1) * 1000,
                "generation_ms": (t4 - t3) * 1000,
                "total": (t4 - t0) * 1000,
                "grounding_ms": 0.0,
            },
        }
