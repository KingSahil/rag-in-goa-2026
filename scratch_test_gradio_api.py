import gradio as gr
import json
import requests
import threading
import time

def query_fn(payload_str):
    data = json.loads(payload_str)
    return json.dumps({
        "answer": f"Grounded answer for {data.get('text')}",
        "answer_source": "extractive",
        "total_ms": 42.5
    })

with gr.Blocks(title="Test Space") as demo:
    inp = gr.Textbox(elem_id="inp")
    out = gr.Textbox(elem_id="out")
    btn = gr.Button("Submit", elem_id="btn")
    btn.click(fn=query_fn, inputs=[inp], outputs=[out], api_name="rag_query")

def run():
    demo.queue().launch(server_port=7878, prevent_thread_lock=True)

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(2)
try:
    res = requests.post(
        "http://127.0.0.1:7878/gradio_api/call/rag_query",
        json={"data": [json.dumps({"text": "what is cnn"})]}
    )
    print("Call status:", res.status_code, res.json())
    event_id = res.json().get("event_id")
    res2 = requests.get(f"http://127.0.0.1:7878/gradio_api/call/rag_query/{event_id}")
    print("Stream result:", res2.text)
finally:
    demo.close()
