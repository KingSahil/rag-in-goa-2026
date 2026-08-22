# 🌴 Indic RAG — Data Pipeline & GPU Indexing Visuals

```
========================================================================================
                      🔥 MULTILINGUAL VECTOR INGESTION ENGINE 🔥
========================================================================================
```

---

## 📊 1. MSMARCO-XI Ingestion & Streaming Deduplication

```
[STREAM] Ingesting Parquet Source: hf://datasets/microsoft/msmarco-xi (3.7 GB)
[STREAM] Deduplicating & Normalizing Indic Scripts...

┌──────────┬──────────────┬──────────────────┬───────────────┬───────────────────────┐
│ Language │ Script       │ Raw Ingested     │ Deduplicated  │ Normalized Passages   │
├──────────┼──────────────┼──────────────────┼───────────────┼───────────────────────┤
│ English  │ Latin        │ 100,000          │ 98,820        │ 98,820 (100.0%)       │
│ Hindi    │ Devanagari   │ 100,000          │ 98,820        │ 98,820 (100.0%)       │
│ Marathi  │ Devanagari   │ 100,000          │ 98,822        │ 98,822 (100.0%)       │
├──────────┼──────────────┼──────────────────┼───────────────┼───────────────────────┤
│ TOTAL    │ 3 Languages  │ 300,000          │ 296,462       │ 296,462 Clean Chunks  │
└──────────┴──────────────┴──────────────────┴───────────────┴───────────────────────┘
```

---

## ⚡ 2. GPU Ingestion & CUDA Acceleration Monitor (`nvidia-smi`)

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070 Ti      On |   00000000:01:00.0  On |                  N/A |
| 100%   76C    P0            242W / 285W |    6962MiB /  12282MiB |     100%      Default|
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     42190      C   python build_indexes_gpu.py                  6814MiB |
+-----------------------------------------------------------------------------------------+
```

---

## 🚀 3. GPU HNSW Vector Indexing Benchmark (`build_indexes_gpu.py`)

```
========================================================================================
>> Initializing Multilingual-E5 Embedder on CUDA (FP16)...
>> Embedding 296,462 passages across EN, HI, MR... [DONE in 11m 42s]
>> Building FAISS HNSW Index (M=32, efConstruction=64, metric=INNER_PRODUCT)...

[INDEX COMPLETE] passage_native.faiss
  ├── Total Vectors:     296,462
  ├── Embedding Dim:     384-d (Dense FP16)
  ├── Build Time:        8.52 seconds
  ├── Memory Footprint:  256.16 MB
  └── Query Latency:     0.84 ms / search

[INDEX COMPLETE] semantic_longdoc.faiss
  ├── Source Docs:       25 Wikipedia Articles (Local Ollama LLM Translated)
  ├── Semantic Chunks:   864 Chunks (English + Hindi + Marathi)
  ├── Build Time:        0.18 seconds
  └── Query Latency:     0.32 ms / search
========================================================================================
```

---

## 🧠 4. Wikipedia Long-Doc Local Translation Pipeline (100% Offline)

```mermaid
graph LR
    A["25 Wikipedia Long-Docs (EN)"] --> B["Local Ollama Translation LLM\n(Zero Cloud API Cost)"]
    B --> C["Devanagari Hindi Docs"]
    B --> D["Devanagari Marathi Docs"]
    C --> E["Recursive Semantic Chunking\n(500 tokens / 50 overlap)"]
    D --> E
    A --> E
    E --> F["864 Long-Doc Vectors (384-d)"]
    F --> G["FAISS HNSW Index Shard"]
```

---

## 🏷️ 5. Key Architecture Metric Badges

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  ⚡ 296k Vectors  │  🚀 HNSW FP16 Index  │  ⏱️ 8.5s Build  │  🔒 100% Local / Zero Cloud │
└───────────────────────────────────────────────────────────────────────────────────────┘
```
