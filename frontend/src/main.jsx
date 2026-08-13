import React, {useRef, useState} from "react";
import {createRoot} from "react-dom/client";
import "./style.css";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function App() {
  const [recording, setRecording] = useState(false);
  const [result, setResult] = useState(null);
  const [text, setText] = useState("");
  const chunks = useRef([]);
  const recorder = useRef(null);

  const start = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const r = new MediaRecorder(stream);
    recorder.current = r;
    chunks.current = [];
    r.ondataavailable = e => chunks.current.push(e.data);
    r.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks.current, {type: "audio/webm"});
      const fd = new FormData();
      fd.append("file", blob, "voice.webm");
      const res = await fetch(`${API}/ask/voice`, {method: "POST", body: fd});
      setResult(await res.json());
    };
    r.start();
    setRecording(true);
  };

  const stop = () => {
    recorder.current?.stop();
    setRecording(false);
  };

  const askText = async () => {
    if (!text.trim()) return;
    const res = await fetch(`${API}/ask/text`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query: text})
    });
    setResult(await res.json());
  };

  return <main>
    <div className="card">
      <p className="eyebrow">HH Goa 2026 · #RAGInGoa</p>
      <h1>Voice RAG</h1>
      <p className="sub">Speak in English or an Indic language. The answer is generated only from retrieved MSMARCO-XI context.</p>

      <button onClick={recording ? stop : start}>{recording ? "Stop recording" : "🎙 Speak"}</button>

      <div className="row">
        <input value={text} onChange={e=>setText(e.target.value)} placeholder="Or type a question…" />
        <button onClick={askText}>Ask</button>
      </div>

      {result && <section className="result">
        {result.transcript && <p><b>Transcript:</b> {result.transcript}</p>}
        <h2>{result.refused ? "Guardrail response" : result.answer}</h2>
        <p className="meta">
          {result.timings_ms?.voice_total_ms?.toFixed(1) ?? result.timings_ms?.total?.toFixed(1)} ms ·
          {result.grounded ? " grounded" : " refused / not grounded"}
        </p>
        {result.sources?.length > 0 && <div>
          <h3>Sources</h3>
          {result.sources.map(s => <article key={s.id}>
            <small>{s.id} · {s.chunk_strategy} · {s.score.toFixed(4)}</small>
            <p>{s.text}</p>
          </article>)}
        </div>}
      </section>}
    </div>
  </main>
}

createRoot(document.getElementById("root")).render(<App />);
