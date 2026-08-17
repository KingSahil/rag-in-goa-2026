# ⚡ Latency Optimization & Non-LLM Synthesis Strategies

## Overview
Executing Indic RAG pipelines within strict sub-200ms Service Level Agreements (SLAs) across 148,000+ vector corpora requires eliminating autoregressive LLM decoding bottlenecks.

This document details the low-level optimizations, ONNX runtimes, vector quantization, and mathematical context synthesis techniques utilized across the system.

---

## 🏎️ Core Latency Acceleration Techniques

### 1. In-Memory FAISS HNSW Graph Indexing
- **Algorithm**: Hierarchical Navigable Small World (`IndexHNSWFlat`)
- **Hyperparameters**:
  - $M = 32$ (Number of bi-directional links per node)
  - $efConstruction = 200$ (Construction search depth)
  - $efSearch = 64$ (Query search depth)
- **Performance**: Traversal across 148,545 vectors in **0.73 ms** on CPU with zero network serialization overhead.

### 2. INT8 Quantized ONNX Runtime
- **Embedding Model**: `intfloat/multilingual-e5-small`
- **Optimization**:
  - Dynamic INT8 quantization reducing weight memory from ~280MB to ~70MB.
  - VNNI CPU instruction set acceleration.
  - Fixed pre-allocated tensor buffers avoiding Python garbage collection pauses.
  - Intra-op parallelism configured via `config.ONNX_NUM_THREADS`.
- **Latency**: Reduces median query embedding from ~45ms down to **6.21 ms**.

### 3. Dynamic Concept Matrix & Vector LRU Cache
- **Component**: [`generation/answer_cache.py`](file:///c:/Projects/rag-ingoa-2026/generation/answer_cache.py)
- **Mechanics**:
  - Fast-path lookup for recurring queries using 384-d semantic cosine similarity ($\ge 0.92$).
  - Thread-safe locked in-memory ring buffer storing up to 2048 concept records.
- **Latency**: **0.23 ms** cache hit retrieval time.

### 4. Continuous TextRank & Economy SVD Matrix Decomposition
- **Component**: [`generation/extractive.py`](file:///c:/Projects/rag-ingoa-2026/generation/extractive.py)
- **Deterministic Synthesis without Generative LLMs**:
  1. *Graph Centrality*: Builds sentence cosine similarity matrix $W_{ij} = \max(0, \vec{s}_i \cdot \vec{s}_j)$ and performs power iteration ($d=0.85$, 12 iterations).
  2. *Economy SVD*: Decomposes sentence vectors ($M = U \Sigma V^T$) and retains singular components reaching $\ge 95\%$ cumulative variance.
  3. *Grammatical Sequencing*: Preserves document sentence order.
- **Latency**: **8.50 ms** on CPU (compared to 800ms+ for 70B LLM autoregressive token decoding).

---

## 📊 Measured Benchmark Summary (15 Languages)

```
====================================================================================================
HIGH-THROUGHPUT SPEED BENCHMARK (750 Queries Total across 15 Languages)
====================================================================================================
Throughput: 51.7 Queries / second
Mean Pipeline Latency: 19.22 ms
P50 Pipeline Latency: 16.45 ms
P95 Pipeline Latency: 23.78 ms
P99 Pipeline Latency: 57.71 ms
====================================================================================================
```
