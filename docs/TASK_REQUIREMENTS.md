# 📋 Hacker House Goa 2026 Task 2: Requirements Compliance Matrix

This matrix maps every single requirement specified in the **Hacker House Goa 2026 Shortlisting Task 2** against the architectural implementations in this codebase.

---

## 🎯 Compliance Overview

| # | Task Requirement | Specification | Implementation in Codebase | Compliance Status |
| :-: | :--- | :--- | :--- | :-: |
| **1** | **Speech-to-Text (STT)** | Use Sarvam or ElevenLabs for voice-to-text. | Native **Sarvam AI Saaras v3** (`saaras:v3`) in [`stt/sarvam_client.py`](file:///c:/Projects/rag-ingoa-2026/stt/sarvam_client.py) with ffmpeg 16kHz mono normalization and language routing. | ✅ **100% Compliant** |
| **2** | **Vast Chunking Strategies** | Multi-strategy chunking beyond naive fixed-size splitting; overlap handling, semantic vs fixed, metadata-aware. | **4 distinct strategies** implemented:<br>1. Passage-Native ([`chunking/passage_native.py`](file:///c:/Projects/rag-ingoa-2026/chunking/passage_native.py))<br>2. Sentence-Window with $\ge 15\%$ overlap ([`chunking/sentence_window.py`](file:///c:/Projects/rag-ingoa-2026/chunking/sentence_window.py))<br>3. Semantic Cosine-Spike ([`chunking/semantic.py`](file:///c:/Projects/rag-ingoa-2026/chunking/semantic.py))<br>4. Metadata-Aware ([`chunking/metadata.py`](file:///c:/Projects/rag-ingoa-2026/chunking/metadata.py))<br>Fused via Reciprocal Rank Fusion (RRF $k=60$). | ✅ **100% Compliant** |
| **3** | **Latency Target (< 200ms)** | Full pipeline (chunking + vector DB retrieval + generation to final output) must execute in $<200$ ms. | **P50 retrieval latency: 6.96 ms**.<br>**P50 full pipeline: 16.45 ms**.<br>**P95 full pipeline: 23.78 ms**.<br>Cold-start max across 15 languages $<194.22$ ms. | ✅ **100% Compliant** |
| **4** | **Latency Analytics** | Submit P50 / P70 / P100 numbers across diverse test queries. | Measured across **750 test queries** spanning 15 languages in [`benchmark/run_speed_bench_50.py`](file:///c:/Projects/rag-ingoa-2026/benchmark/run_speed_bench_50.py) & [`benchmark/results/`](file:///c:/Projects/rag-ingoa-2026/benchmark/results/). | ✅ **100% Compliant** |
| **5** | **Orchestration Harness** | Structured orchestration with retries, tool calls, structured I/O handling, and error recovery. | Hand-rolled async state machine in [`pipeline/orchestrator.py`](file:///c:/Projects/rag-ingoa-2026/pipeline/orchestrator.py), Pydantic v2 schemas in [`pipeline/schemas.py`](file:///c:/Projects/rag-ingoa-2026/pipeline/schemas.py), exponential backoff retries, and `warmup_pipeline()`. | ✅ **100% Compliant** |
| **6** | **Guardrails & Anti-Hallucination** | Off-topic query handling, unsafe/inappropriate filtering, hallucination checks, knowing when *not* to answer. | **Cascaded 4-tier pre-retrieval guardrails** (Regex + Meta Prompt-Guard 86M + Intent Taxonomy + Centroid Topic Gate) + **Context Chunk IPI Scan** + **Post-Gen Grounding Gate** ($\ge 30\%$ overlap). | ✅ **100% Compliant** |
| **7** | **Dataset Support** | Build pipeline on AI4Bharat MSMARCO-XI Indic dataset. | Ingested and indexed across all **14 Indic languages + English** (~743k passages) in [`data/build_corpus.py`](file:///c:/Projects/rag-ingoa-2026/data/build_corpus.py) and [`data/build_all_15_corpora.py`](file:///c:/Projects/rag-ingoa-2026/data/build_all_15_corpora.py). | ✅ **100% Compliant** |
