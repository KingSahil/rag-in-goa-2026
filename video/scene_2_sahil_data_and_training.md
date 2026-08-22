# 🎬 Hacker House Goa 2026 — Pitch Video Script
## Scene 2: SAHIL — Data & Training

| Parameter | Details |
| :--- | :--- |
| **Scene Number** | Scene 2 |
| **Speaker** | **Sahil** (Data Engineering & Vector Indexing Lead) |
| **Duration** | **0:10 – 0:30** (20 Seconds) |
| **Primary Theme** | Massive Multilingual Data Pipeline, Deduping, & GPU HNSW Indexing |
| **Key Tone** | Energetic, technical, authentic builder hustle |

---

### 🎥 Visuals & Camera Direction
* **Camera Setup**: Medium close-up of Sahil speaking directly to the camera in a modern dev/hacker environment.
* **Cutaways / Split Screen**: Screen-recording overlay showing the high-throughput parquet download stream, chunk processing, and CUDA vector ingestion.

---

### 🎙️ Spoken Dialogue & Script

> **Sahil:**  
> *"My job was the data. MSMARCO-XI — 14 Indic languages, 3.7 gigabyte parquet files, your disk space weeping. I wrote a streamer that deduped and normalized almost 99,000 passages per language in English, Hindi, and Marathi."*

> **Sahil:**  
> *"Then I had to embed all 296,000 of them on the GPU. My laptop sounded like a jet engine for 12 minutes. But the HNSW indexes came out beautiful — 864 long-doc chunks across three languages, because I also translated 25 Wikipedia articles using the local Ollama model. Because I hate cloud APIs."*

---

### 🎞️ B-Roll & Visual Cutaways

| Timestamp | Visual Cue / Screen Asset | Action / On-Screen Content |
| :--- | :--- | :--- |
| **0:10 – 0:15** | 📊 **Data Ingestion Terminal** | Fast parquet streaming, streaming deduplication logs, JSONL passage count ticking up to 99,000. |
| **0:15 – 0:22** | ⚡ **`nvidia-smi` CUDA Monitor** | Terminal showing GPU at **100% compute utilization**, **6.8GB VRAM allocated**, CUDA power spiking. |
| **0:22 – 0:26** | 🚀 **`build_indexes_gpu.py` Benchmark** | Green terminal output: `"296,462 passages + 864 longdocs in 8.5s"`. |
| **0:26 – 0:30** | 😂 **Reaction Shot** | Quick authentic cutaway of Sahil squinting / reacting to the loud jet-engine laptop fan noise. |

---

### 🏷️ On-Screen Overlays & Lower-Thirds

```
┌─────────────────────────────────────────────────────────────┐
│  ⚡ 296k vectors · HNSW · FP16 · laptop = jet engine         │
└─────────────────────────────────────────────────────────────┘
```

* **Position**: Bottom-center floating badge (Neon Emerald & Cyber Gold typography).
* **Sound Effects (SFX)**:
  * Soft data whoosh on terminal cutaway.
  * Subtle jet turbine ramp-up sound effect under *"sounded like a jet engine"*.
  * Satisfying metallic *ding* when `"296,462 passages + 864 longdocs in 8.5s"` flashes.

---

### 📝 Key Technical Takeaways Displayed
1. **MSMARCO-XI Indic Shards**: Processed EN, HI, MR datasets with streaming deduplication.
2. **FAISS HNSW Indexing**: Native vector representations in 384-d space (multilingual-e5).
3. **Zero External Cloud Dependencies**: 100% offline local embeddings + local Ollama multilingual translations.
