"""
Hugging Face Space Application for Hacker House Goa 2026: Voice-Enabled Indic RAG.
ZeroGPU-native Gradio 5 application hosting the full-viewport Command Center UI.
"""

import asyncio
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

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

# ZeroGPU decorator shim
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

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


# ── ZeroGPU Bridge Functions ──────────────────────────────────────────
@spaces.GPU
def rag_query_bridge(payload_str: str) -> str:
    """ZeroGPU execution bridge for end-to-end RAG query."""
    try:
        data = json.loads(payload_str)
        req = QueryRequest(
            text=data.get("text"),
            language_hint=data.get("language_hint"),
            cross_lingual=bool(data.get("cross_lingual", False)),
            bypass_cache=bool(data.get("bypass_cache", False)),
        )
        res = asyncio.run(orchestrator.execute(req))
        return res.model_dump_json()
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e),
            "answer": f"Declined: System error occurred ({e})",
            "answer_source": "declined",
            "guardrail_flags": {"unsafe_detected": False, "off_topic_detected": False},
        })


@spaces.GPU
def tts_query_bridge(payload_str: str) -> str:
    """ZeroGPU execution bridge for Sarvam AI Indic TTS."""
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


# ── Launch Entrypoint ─────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()

