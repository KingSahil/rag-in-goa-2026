import html
import json
from pathlib import Path
import gradio as gr
import requests
import threading
import time

def escape_srcdoc(text: str) -> str:
    return html.escape(text, quote=True)

def query_fn(payload_str):
    data = json.loads(payload_str)
    return json.dumps({
        "answer": f"Grounded response for {data.get('text')}",
        "answer_source": "extractive",
        "total_ms": 38.2
    })

demo_html = """<!DOCTYPE html>
<html>
<head><title>Test UI</title></head>
<body style="background:#020D08;color:white;">
  <h1>Retro Tropical Command Center</h1>
  <button id="testBtn">Test Query</button>
  <div id="res"></div>
  <script>
    document.getElementById('testBtn').onclick = async () => {
      const res = await fetch('/gradio_api/call/rag_query', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({data: [JSON.stringify({text: 'hello'})]})
      });
      const data = await res.json();
      const eventRes = await fetch('/gradio_api/call/rag_query/' + data.event_id);
      const text = await eventRes.text();
      document.getElementById('res').innerText = text;
    };
  </script>
</body>
</html>"""

with gr.Blocks(title="Test Space") as demo:
    gr.HTML(f'<iframe id="cmd-center-frame" srcdoc="{escape_srcdoc(demo_html)}" style="width:100vw;height:100vh;border:none;"></iframe>')
    inp = gr.Textbox(visible=False, elem_id="inp")
    out = gr.Textbox(visible=False, elem_id="out")
    btn = gr.Button("Submit", visible=False, elem_id="btn")
    btn.click(fn=query_fn, inputs=[inp], outputs=[out], api_name="rag_query")

def run():
    demo.queue().launch(server_port=7879, prevent_thread_lock=True)

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(2)
try:
    r = requests.get("http://127.0.0.1:7879/")
    print("Root Gradio status:", r.status_code)
    print("Contains srcdoc:", "srcdoc=" in r.text)
finally:
    demo.close()
