# 📊 Comprehensive Benchmark Report & Latency Analytics

## Overview
Latency analytics and quality evaluations were performed across **15 languages** (English, Hindi, Tamil, Telugu, Bengali, Urdu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Assamese, Odia, Nepali, Sanskrit), testing retrieval accuracy, guardrail filtering, and high-throughput speed.

---

## 🏎️ 1. End-to-End Retrieval Latency Benchmark (`app/benchmark.py`)
Measures combined query embedding vectorization (`intfloat/multilingual-e5-small` INT8 ONNX) + FAISS HNSW graph traversal on CPU against the strict **50.0 ms budget** in `app/config.py`:

- **Environment**: `8 vCPUs | 15.78 GB RAM | Windows 11 (AMD64) | 100% CPU Execution`
- **Evaluation Command**: `python -m app.benchmark 50`
- **Result**: ✅ **PASS (P95: 7.97 ms vs 50.0 ms SLA — 84% faster than budget)**

| Stage | Mean | P50 | P95 | P99 | SLA Budget | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Query Embedding Vectorization** | 6.31 ms | 6.21 ms | 7.28 ms | 7.94 ms | — | ⚡ Sub-8ms ONNX |
| **FAISS HNSW Vector Search (148k vecs)** | 0.73 ms | 0.71 ms | 0.93 ms | 1.16 ms | — | ⚡ Sub-1ms Traversal |
| **Total Retrieval Latency (Embed + Search)** | **7.04 ms** | **6.96 ms** | **7.97 ms** | **8.95 ms** | **50.00 ms** | ✅ **PASS** |

---

## ❄️ 2. 15-Language Cold-Start SLA Benchmark (`benchmark/run_cold_start_bench.py`)
Evaluates cold-path retrieval, reranking, context safety scanning, and grounded generation across all 15 languages with **`bypass_cache=True`**:

- **Target SLA**: `< 200 ms` on cold uncached requests
- **SLA Pass Rate**: **15/15 (100.0%)** ✅

| Language | Code | Context Guard | Cross-Encoder Rerank | Generation | Total Cold Latency | SLA Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **English** | `en` | 0.99 ms | 53.40 ms | 0.92 ms | **120.48 ms** | ✅ **PASS** |
| **Hindi** | `hi` | 1.57 ms | 88.12 ms | 0.25 ms | **175.05 ms** | ✅ **PASS** |
| **Tamil** | `ta` | 1.35 ms | 79.50 ms | 0.19 ms | **173.49 ms** | ✅ **PASS** |
| **Telugu** | `te` | 1.21 ms | 90.72 ms | 0.18 ms | **164.77 ms** | ✅ **PASS** |
| **Bengali** | `bn` | 1.96 ms | 93.57 ms | 0.17 ms | **162.78 ms** | ✅ **PASS** |
| **Urdu** | `ur` | 1.58 ms | 103.12 ms | 0.21 ms | **169.99 ms** | ✅ **PASS** |
| **Marathi** | `mr` | 1.59 ms | 103.31 ms | 0.23 ms | **177.35 ms** | ✅ **PASS** |
| **Gujarati** | `gu` | 1.82 ms | 89.76 ms | 0.25 ms | **161.61 ms** | ✅ **PASS** |
| **Kannada** | `kn` | 1.82 ms | 76.86 ms | 0.27 ms | **167.62 ms** | ✅ **PASS** |
| **Malayalam** | `ml` | 1.82 ms | 83.41 ms | 0.16 ms | **170.34 ms** | ✅ **PASS** |
| **Punjabi** | `pa` | 2.21 ms | 100.67 ms | 0.19 ms | **181.81 ms** | ✅ **PASS** |
| **Assamese** | `as` | 1.83 ms | 110.64 ms | 0.23 ms | **194.22 ms** | ✅ **PASS** |
| **Odia** | `or` | 1.99 ms | 86.99 ms | 0.23 ms | **178.73 ms** | ✅ **PASS** |
| **Nepali** | `ne` | 1.31 ms | 109.84 ms | 0.23 ms | **184.45 ms** | ✅ **PASS** |
| **Sanskrit** | `sa` | 0.00 ms | 66.14 ms | 0.00 ms | **145.40 ms** | ✅ **PASS** |
| **Out-of-Domain Control** | `en` | 0.00 ms | 97.39 ms | 0.00 ms | **168.86 ms** | ✅ **PASS (Declined)** |
| **Safety Control** | `en` | 0.00 ms | 0.00 ms | 0.00 ms | **0.24 ms** | ✅ **PASS (Blocked)** |

---

## 🚀 3. High-Throughput Speed Benchmark (750 Queries Total)
- **Throughput**: **51.7 Queries / second**
- **P50 Latency**: **16.45 ms**
- **P95 Latency**: **23.78 ms**
- **P99 Latency**: **57.71 ms**
- **Mean Latency**: **19.22 ms**
