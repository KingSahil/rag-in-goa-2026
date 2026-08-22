import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import requests
import uvicorn
import threading
import time

custom_app = FastAPI()

@custom_app.get("/demo-ui")
def demo_ui():
    return HTMLResponse("<h1>Demo UI Loaded</h1>")

@custom_app.post("/api/query")
def api_query():
    return JSONResponse({"answer": "Capital of India is New Delhi", "answer_source": "extractive"})

with gr.Blocks(title="Test Space") as demo:
    gr.HTML('<iframe id="cmd-frame" src="/demo-ui"></iframe>')

app = gr.mount_gradio_app(custom_app, demo, path="/gradio")

def run():
    uvicorn.run(app, host="127.0.0.1", port=7875, log_level="warning")

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(2)
try:
    r1 = requests.get("http://127.0.0.1:7875/demo-ui")
    print("/demo-ui status:", r1.status_code, r1.text)
    r2 = requests.post("http://127.0.0.1:7875/api/query")
    print("/api/query status:", r2.status_code, r2.json())
    r3 = requests.get("http://127.0.0.1:7875/gradio")
    print("/gradio status:", r3.status_code)
finally:
    pass
