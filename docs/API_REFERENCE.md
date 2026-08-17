# 📡 API Reference & Schema Specification

## Base URL
- **Local**: `http://localhost:7860`
- **Hugging Face Space**: `https://ansh123456789-ragingoa.hf.space`

---

## 🚀 Endpoints

### 1. `POST /query`
Execute end-to-end Voice RAG query. Supports multipart audio file upload or JSON/form text bypass.

#### Request Form / JSON Parameters:
- `file` (*optional*, `UploadFile`): Audio file (`.wav`, `.mp3`, `.ogg`, `.webm`, `.m4a`).
- `text` (*optional*, `string`): Text query string for fast-path processing.
- `language_hint` (*optional*, `string`): ISO language code (e.g. `en`, `hi`, `ta`, `mr`) or `auto`.
- `cross_lingual` (*optional*, `boolean`, default: `false`): Enable cross-lingual retrieval across all indexed languages.
- `bypass_cache` (*optional*, `boolean`, default: `false`): Force cold retrieval by skipping semantic cache.

#### Sample Request (`curl`):
```bash
curl -X POST http://localhost:7860/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the capital of France?", "language_hint": "en"}'
```

#### Sample Response (`JSON`):
```json
{
  "query": "What is the capital of France?",
  "transcript": "What is the capital of France?",
  "language_detected": "en",
  "answer": "Paris is the capital and most populous city of France.",
  "answer_source": "extractive",
  "retrieved_chunks": [
    {
      "chunk_id": "en_passage_4821",
      "text": "Paris is the capital and most populous city of France...",
      "source_lang": "en",
      "chunk_strategy": "passage_native",
      "dense_score": 0.8841,
      "bm25_score": 12.45,
      "final_score": 0.8912,
      "contributing_strategies": ["passage_native", "sentence_window"],
      "metadata": {}
    }
  ],
  "guardrail_flags": {
    "unsafe_detected": false,
    "unsafe_reason": null,
    "intent_detected": false,
    "intent_type": null,
    "intent_reason": null,
    "off_topic_detected": false,
    "off_topic_distance": 0.1245,
    "off_topic_reason": null,
    "safety_model_failed": false,
    "model_failed": false,
    "grounding_passed": true,
    "grounding_score": 0.9421,
    "grounding_reason": "Grounded with high lexical and semantic overlap.",
    "decline_reason_code": null
  },
  "stage_timings": [
    {"stage": "stt_transcription", "ms": 0.0, "success": true, "fallback_used": false, "details": "Text bypass"},
    {"stage": "language_routing", "ms": 0.12, "success": true, "fallback_used": false, "details": "Routed to 'en'"},
    {"stage": "pre_retrieval_safety_guardrail", "ms": 0.84, "success": true, "fallback_used": false, "details": "Passed regex and prompt-guard"},
    {"stage": "query_intent_guardrail", "ms": 0.21, "success": true, "fallback_used": false, "details": "Factual intent"},
    {"stage": "query_embedding", "ms": 6.18, "success": true, "fallback_used": false, "details": "ONNX INT8 E5"},
    {"stage": "pre_retrieval_topic_guardrail", "ms": 0.41, "success": true, "fallback_used": false, "details": "On-topic (dist: 0.1245)"},
    {"stage": "vector_retrieval_and_merge", "ms": 0.89, "success": true, "fallback_used": false, "details": "FAISS HNSW + RRF"},
    {"stage": "generation", "ms": 8.42, "success": true, "fallback_used": false, "details": "TextRank + SVD Energy Synthesis"},
    {"stage": "post_generation_grounding_guardrail", "ms": 0.31, "success": true, "fallback_used": false, "details": "Grounded"}
  ],
  "retrieval_ms": 7.89,
  "total_ms": 17.38
}
```

---

### 2. `POST /tts`
Synthesize spoken audio from text using Sarvam AI Bulbul TTS.

#### Request Body (`JSON`):
```json
{
  "text": "नमस्ते, मैं आपकी सहायता कैसे कर सकता हूँ?",
  "target_language": "hi",
  "speaker": "anushka",
  "pace": 1.0
}
```

#### Response (`JSON`):
```json
{
  "audio_base64": "<base64_encoded_wav_audio>",
  "format": "audio/wav",
  "sarvam": true,
  "language": "hi-IN",
  "speaker": "anushka"
}
```

---

### 3. `GET /health`
System health check reporting loaded indexes, centroids, active languages, and guardrail statuses.

#### Response (`JSON`):
```json
{
  "status": "healthy",
  "configured_languages": ["en", "hi", "ta", "mr"],
  "embedding_model": "intfloat/multilingual-e5-small",
  "indexes_loaded": {
    "passage_native": 148545,
    "semantic_longdoc": 309
  },
  "centroids_available": ["en", "hi", "mr", "global"],
  "allow_network_calls": false,
  "sarvam_stt_configured": true,
  "sarvam_tts_configured": true,
  "semantic_answer_cache_configured": true,
  "request_timeout_seconds": 15.0,
  "query_intent_filter_enabled": true
}
```

---

### 4. `GET /languages`
Returns metadata, scripts, and MS MARCO source mappings for active languages.
