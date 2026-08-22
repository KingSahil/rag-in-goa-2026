import gradio as gr
from fastapi.routing import APIRoute
from fastapi.responses import HTMLResponse, JSONResponse
import requests
import threading
import time

routes = [
    APIRoute('/demo-ui', lambda: HTMLResponse('<h1>OK</h1>'), methods=['GET']),
    APIRoute('/api/query', lambda: JSONResponse({'answer': 'Test'}), methods=['POST']),
]

demo = gr.Blocks()

def run():
    demo.launch(server_port=7874, app_kwargs={'routes': routes}, prevent_thread_lock=True)

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(2)
try:
    r1 = requests.get('http://127.0.0.1:7874/demo-ui')
    print('/demo-ui ->', r1.status_code, r1.text)
    r2 = requests.post('http://127.0.0.1:7874/api/query')
    print('/api/query ->', r2.status_code, r2.json())
finally:
    demo.close()
