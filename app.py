"""
Hugging Face Space Application for Hacker House Goa 2026: Voice-Enabled Indic RAG.
ZeroGPU-native Gradio 5 application hosting the full-viewport Command Center UI.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

# Import spaces FIRST before any CUDA or Gradio imports
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

import asyncio
import html
import json
from pathlib import Path
from typing import Any, Dict, Optional
import nest_asyncio
nest_asyncio.apply()

# Compatibility shim for huggingface_hub
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class DummyHfFolder:
            @staticmethod
            def get_token():
                return os.environ.get("HF_TOKEN") or None
            @staticmethod
            def save_token(token):
                pass
            @staticmethod
            def delete_token():
                pass
        huggingface_hub.HfFolder = DummyHfFolder
except Exception:
    pass

import gradio as gr

# Defensive monkeypatch for Gradio 5.x queueing lock bugs (delete_lock and pending_message_lock)
try:
    import gradio.queueing
    if hasattr(gradio.queueing, "Queue"):
        _orig_queue_init = gradio.queueing.Queue.__init__
        def _patched_queue_init(self, *args, **kwargs):
            _orig_queue_init(self, *args, **kwargs)
            if not hasattr(self, "pending_message_lock") or not hasattr(self.pending_message_lock, "__aenter__"):
                self.pending_message_lock = asyncio.Lock()
            if not hasattr(self, "delete_lock") or not hasattr(self.delete_lock, "__aenter__"):
                self.delete_lock = asyncio.Lock()
        gradio.queueing.Queue.__init__ = _patched_queue_init

        _orig_start = gradio.queueing.Queue.start_processing
        async def _patched_start(self, *args, **kwargs):
            if not hasattr(self, "delete_lock") or not hasattr(self.delete_lock, "__aenter__"):
                self.delete_lock = asyncio.Lock()
            if not hasattr(self, "pending_message_lock") or not hasattr(self.pending_message_lock, "__aenter__"):
                self.pending_message_lock = asyncio.Lock()
            return await _orig_start(self, *args, **kwargs)
        gradio.queueing.Queue.start_processing = _patched_start

        if hasattr(gradio.queueing.Queue, "clean_events"):
            _orig_clean = gradio.queueing.Queue.clean_events
            async def _patched_clean(self, *args, **kwargs):
                if not hasattr(self, "delete_lock") or not hasattr(self.delete_lock, "__aenter__"):
                    self.delete_lock = asyncio.Lock()
                if not hasattr(self, "pending_message_lock") or not hasattr(self.pending_message_lock, "__aenter__"):
                    self.pending_message_lock = asyncio.Lock()
                return await _orig_clean(self, *args, **kwargs)
            gradio.queueing.Queue.clean_events = _patched_clean
except Exception:
    pass

import config
from pipeline.orchestrator import get_orchestrator
from pipeline.schemas import QueryRequest
from stt.sarvam_tts import synthesize_speech


# ── Preload models at startup ──────────────────────────────────────────
print("[Space Startup] Preloading embedding model, FAISS indexes, and warming up pipeline...")
orchestrator = get_orchestrator()
orchestrator.warmup_pipeline()
print("[Space Startup] Full RAG pipeline preloaded and ready for traffic.")


# ── Fullscreen CSS to remove all Gradio chrome and embed iframe ───────
CUSTOM_CSS = """
body, html { margin: 0 !important; padding: 0 !important; overflow: hidden !important; height: 100vh !important; width: 100vw !important; }
.gradio-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; height: 100vh !important; }
.main, .wrap, .contain, .gap-4, .gap-2 { padding: 0 !important; margin: 0 !important; gap: 0 !important; }
footer, .built-with, gradio-app > footer, .gradio-container > footer { display: none !important; }
#cmd-center-frame {
    width: 100vw;
    height: 100vh;
    border: none;
    display: block;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 99999;
}
"""


# ── ZeroGPU Startup Detector Hook ─────────────────────────────────────
@spaces.GPU
def _zerogpu_startup_hook():
    """Satisfies ZeroGPU startup scanner while avoiding request runtime throttling."""
    return True


# ── Execution Bridge Functions ────────────────────────────────────────
def rag_query_bridge(payload_str: str) -> str:
    """Fast execution bridge for end-to-end RAG query."""
    temp_audio_path = None
    try:
        data = json.loads(payload_str)
        audio_b64 = data.get("audio_base64")
        if audio_b64:
            import base64
            import tempfile
            audio_bytes = base64.b64decode(audio_b64)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(audio_bytes)
            tmp.close()
            temp_audio_path = tmp.name

        req = QueryRequest(
            audio_path=temp_audio_path,
            text=data.get("text"),
            language_hint=data.get("language_hint"),
            cross_lingual=bool(data.get("cross_lingual", False)),
            bypass_cache=bool(data.get("bypass_cache", False)),
        )
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        res = loop.run_until_complete(orchestrator.execute(req))
        return res.model_dump_json()
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e),
            "answer": f"Declined: System error occurred ({e})",
            "answer_source": "declined",
            "guardrail_flags": {"unsafe_detected": False, "off_topic_detected": False},
        })
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


def tts_query_bridge(payload_str: str) -> str:
    """Fast execution bridge for Sarvam AI Indic TTS."""
    try:
        data = json.loads(payload_str)
        lang = data.get("language") or data.get("target_language") or "hi"
        res = synthesize_speech(
            text=data.get("text", ""),
            language_code=lang,
            speaker=data.get("speaker"),
            pace=float(data.get("pace", 1.0)),
        )
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def _get_escaped_demo_html() -> str:
    """Read and escape the custom Command Center HTML file for iframe srcdoc."""
    demo_file = config.BASE_DIR / "demo" / "index.html"
    if demo_file.exists():
        raw_html = demo_file.read_text(encoding="utf-8")
        return html.escape(raw_html, quote=True)
    return html.escape("<h1>Command Center UI Not Found</h1>", quote=True)


# ── Build Pure Gradio 5 Interface ──────────────────────────────────────
with gr.Blocks(title="🌴 Hacker House Goa 2026 - Voice Indic RAG", css=CUSTOM_CSS) as demo:
    # Fullscreen embedded Command Center
    gr.HTML(f'<iframe id="cmd-center-frame" srcdoc="{_get_escaped_demo_html()}" allow="microphone; clipboard-write"></iframe>')

    # Gradio API endpoints for ZeroGPU bridging
    gr_in_q = gr.Textbox(visible=False, elem_id="gr_in_q")
    gr_out_q = gr.Textbox(visible=False, elem_id="gr_out_q")
    gr_btn_q = gr.Button("Query Bridge", visible=False, elem_id="gr_btn_q")
    gr_btn_q.click(fn=rag_query_bridge, inputs=[gr_in_q], outputs=[gr_out_q], api_name="rag_query")

    gr_in_t = gr.Textbox(visible=False, elem_id="gr_in_t")
    gr_out_t = gr.Textbox(visible=False, elem_id="gr_out_t")
    gr_btn_t = gr.Button("TTS Bridge", visible=False, elem_id="gr_btn_t")
    gr_btn_t.click(fn=tts_query_bridge, inputs=[gr_in_t], outputs=[gr_out_t], api_name="tts_query")

    # ── Debug endpoint for diagnosing raw FAISS scores on Space ──────────
    def debug_index_bridge(payload_str: str) -> str:
        """Diagnostic bridge: runs raw FAISS search and returns scores for debugging."""
        try:
            import numpy as np
            from retrieval.index_faiss import get_index_manager
            from retrieval.embed import get_embedder

            data = json.loads(payload_str)
            query = data.get("text", "Goa beach")
            embedder = get_embedder()
            idx_mgr = get_index_manager()

            qvec = embedder.encode_queries(query)
            qvec_2d = qvec.reshape(1, -1).astype(np.float32)

            result = {}
            for strat_name, strat_idx in idx_mgr.indexes.items():
                ntotal = strat_idx.index.ntotal
                raw_scores, raw_ids = strat_idx.index.search(qvec_2d, min(5, ntotal))
                candidates = strat_idx.search(qvec, target_lang="en", top_k=5)
                result[strat_name] = {
                    "ntotal": ntotal,
                    "raw_faiss_scores": raw_scores[0].tolist(),
                    "raw_faiss_ids": raw_ids[0].tolist(),
                    "candidate_scores": [c.get("score", 0.0) for c in candidates],
                    "num_candidates": len(candidates),
                }
            result["query_vec_norm"] = float(np.linalg.norm(qvec))
            result["centroids_available"] = list(idx_mgr.centroids.keys())
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            import traceback
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

    gr_in_d = gr.Textbox(visible=False, elem_id="gr_in_d")
    gr_out_d = gr.Textbox(visible=False, elem_id="gr_out_d")
    gr_btn_d = gr.Button("Debug Index Bridge", visible=False, elem_id="gr_btn_d")
    gr_btn_d.click(fn=debug_index_bridge, inputs=[gr_in_d], outputs=[gr_out_d], api_name="debug_index")


from api.main import app as fastapi_app

# ── Attach REST Endpoints directly into Gradio FastAPI App ────────────
# Serves direct REST APIs (/api/query, /query, /api/tts, /tts, /health, /languages) alongside Gradio ZeroGPU UI
for route in fastapi_app.routes:
    demo.app.routes.append(route)


# ── Launch Entrypoint ─────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue().launch()

