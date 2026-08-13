# RAGInGoa 2026 — Voice-Enabled Multilingual RAG

A reference implementation for the HH Goa 2026 Task 2:
Voice → Sarvam STT → multilingual hybrid retrieval → grounded answer → guardrails.

## Architecture

Browser microphone
→ `POST /ask/voice`
→ Sarvam Saaras v3 STT
→ query normalization
→ language shard selection
→ parallel retrieval:
  1. sentence-aware chunks
  2. adaptive sliding-window chunks
  3. semantic-boundary chunks
  4. BM25 lexical retrieval
→ reciprocal-rank fusion
→ lightweight reranking
→ grounded generation
→ deterministic grounding/off-topic guardrail
→ structured JSON response

Chunking and indexing are OFFLINE. They are never performed per request.

## Stack

Backend: FastAPI, FAISS, sentence-transformers, rank_bm25, Sarvam API.
Frontend: React + Vite.
Indexes: language-sharded FAISS + local metadata.

## Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env
```

Set:
- `SARVAM_API_KEY`
- `SARVAM_CHAT_MODEL=sarvam-30b`
- `EMBED_MODEL=intfloat/multilingual-e5-small`

Build an index sample first:

```bash
python scripts/build_index.py --languages hi,pa,en --rows-per-language 100000
# `en` reads English_passages from the Hindi configuration; there is no separate `en` config.
```

For the final submission, run the same pipeline over the largest feasible corpus and document the exact corpus/index size.

Start API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE=http://localhost:8000` if needed.

## Benchmark

Text-only benchmark:

```bash
cd backend
python scripts/benchmark.py --queries scripts/benchmark_queries.json --runs 100
```

It reports:
- retrieval
- rerank
- generation
- guardrail
- RAG-core end-to-end
- p50 / p70 / p100

Do not invent latency values in the submission. Paste the actual benchmark output.

## Latency design

The challenge's 200 ms requirement is handled as an engineering contract around the RAG core:
- offline chunking/indexing
- warmed embedding model
- warmed FAISS indexes
- language sharding
- parallel dense/BM25 retrieval
- small candidate set
- short generation
- no second LLM verification call in the hot path

Voice/STT timing is separately measured because external STT network/model latency is a different component. Sarvam also provides a streaming STT WebSocket, which can be used for the live voice UX.

## Guardrails

1. Off-topic: low retrieval confidence → refuse.
2. Unsafe: simple input safety filter → refuse.
3. Grounding: generated answer must contain evidence-related token overlap with retrieved context.
4. Structured output: Pydantic schema validation.
5. Failure recovery: timeouts + bounded retries for remote model calls.
6. Sources: every successful answer returns passage IDs and scores.

## Important

MSMARCO-XI contains 14 language configurations and around 10.1M train rows on Hugging Face. Each row contains translated query/answer data plus selected and unselected passages. Use streaming/sharding and deduplication during indexing rather than loading everything into RAM.
