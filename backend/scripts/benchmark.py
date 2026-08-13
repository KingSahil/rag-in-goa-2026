import argparse
import asyncio
import json
import statistics
import time

from app.config import settings
from app.harness import RAGHarness
from app.retrieval import Retriever
from sentence_transformers import SentenceTransformer

def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    idx = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[idx]

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    model = SentenceTransformer(settings.embed_model)
    retriever = Retriever(settings.index_dir, model, settings.top_k_dense, settings.top_k_bm25)
    harness = RAGHarness(retriever)

    qs = json.load(open(args.queries, encoding="utf-8"))
    latencies = []
    failures = 0

    for i in range(args.runs):
        q = qs[i % len(qs)]
        t0 = time.perf_counter()
        try:
            out = await harness.run(q["query"], q.get("language"))
            latencies.append(out["timings_ms"]["total"])
            if out.get("refused"):
                failures += 1
        except Exception as e:
            failures += 1
            print("ERROR:", repr(e))

    print({
        "runs": len(latencies),
        "failures": failures,
        "p50_ms": percentile(latencies, 50),
        "p70_ms": percentile(latencies, 70),
        "p100_ms": max(latencies) if latencies else None,
        "mean_ms": statistics.mean(latencies) if latencies else None,
        "under_200ms": sum(x < 200 for x in latencies) / max(1, len(latencies)),
    })

if __name__ == "__main__":
    asyncio.run(main())
