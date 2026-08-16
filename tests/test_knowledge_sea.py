import asyncio
import sys
from pathlib import Path

# Force UTF-8 on Windows terminal stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.orchestrator import RAGPipelineOrchestrator
from pipeline.schemas import QueryRequest

async def main():
    print("Initializing Orchestrator...", flush=True)
    orch = RAGPipelineOrchestrator()
    queries = [
        "What is quantum computing?",
        "What is programming?",
        "What is backend development?"
    ]
    for q in queries:
        print("\n" + "=" * 60, flush=True)
        print(f"QUERY: {q}", flush=True)
        res = await orch.execute(QueryRequest(text=q))
        print(f"ANSWER: {res.answer[:150]}...", flush=True)
        print(f"SOURCE: {res.answer_source}", flush=True)
        print(f"OFF_TOPIC: {res.guardrail_flags.get('off_topic_detected')}", flush=True)
        print(f"RETRIEVED CHUNKS COUNT: {len(res.retrieved_chunks)}", flush=True)
        
        seen_texts = set()
        has_duplicates = False
        for i, c in enumerate(res.retrieved_chunks):
            snippet = c.text[:60].strip()
            print(f"  [{i+1}] chunk_id={c.chunk_id} final_score={c.final_score:.4f} match={round(c.final_score*100)}% text={snippet!r}", flush=True)
            if snippet in seen_texts:
                has_duplicates = True
            seen_texts.add(snippet)
            
        print(f"DUPLICATES DETECTED: {has_duplicates}", flush=True)
        assert not has_duplicates, "Found duplicates in retrieved chunks!"

if __name__ == "__main__":
    asyncio.run(main())
