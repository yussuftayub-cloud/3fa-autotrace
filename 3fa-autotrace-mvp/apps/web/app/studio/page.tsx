"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Studio() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function runTrace() {
    if (!file) return;
    setBusy(true); setError(""); setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/api/trace`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Trace gagal");
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally { setBusy(false); }
  }

  function downloadEPS() {
    if (!result?.eps_base64) return;
    const bytes = Uint8Array.from(atob(result.eps_base64), c => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: "application/postscript" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "3fa-autotrace.eps"; a.click();
    URL.revokeObjectURL(url);
  }

  function downloadSVG() {
    if (!result?.svg) return;
    const blob = new Blob([result.svg], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "3fa-autotrace.svg"; a.click();
    URL.revokeObjectURL(url);
  }

  return <main className="studio">
    <header className="studio-nav"><a href="/" className="brand"><span>3FA</span> AUTO TRACE</a><a href="/subscription">Pro RM49.90/mo</a></header>
    <section className="studio-wrap">
      <div className="eyebrow">AI STUDIO</div>
      <h1>Auto Trace</h1>
      <p className="muted">Upload artwork and generate a first-pass vector trace.</p>
      <label className="upload-box">
        <input type="file" accept="image/png,image/jpeg,image/webp" onChange={e => setFile(e.target.files?.[0] || null)} />
        <div className="upload-icon">↑</div>
        <strong>{file ? file.name : "Choose artwork"}</strong>
        <small>PNG · JPG · WebP · maximum 15MB</small>
      </label>
      <button className="primary wide" disabled={!file || busy} onClick={runTrace}>{busy ? "Tracing..." : "Run AI Auto Trace →"}</button>
      {error && <div className="error">{error}</div>}
      {result && <section className="result">
        <div className="result-head"><div><div className="eyebrow">RESULT</div><h2>Vector ready</h2></div><div className="score">{result.quality}%<small>quality</small></div></div>
        <div className="preview" dangerouslySetInnerHTML={{__html: result.svg}} />
        <div className="stats"><span>{result.width} × {result.height}px</span><span>{result.contours} paths</span></div>
        <div className="actions"><button className="primary" onClick={downloadEPS}>Download EPS</button><button className="secondary" onClick={downloadSVG}>Download SVG</button></div>
        <p className="notice">MVP tracer: first-pass vectorization. Advanced colour tracing, node optimisation and production-grade smoothing will be added in the next engine.</p>
      </section>}
    </section>
  </main>;
}
