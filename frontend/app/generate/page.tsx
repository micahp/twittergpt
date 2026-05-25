"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function GeneratePage() {
  const [adapterDir, setAdapterDir] = useState<string | null>(null);
  const [topic, setTopic] = useState("");
  const [temp, setTemp] = useState(0.9);
  const [n, setN] = useState(5);
  const [tweets, setTweets] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { setAdapterDir(localStorage.getItem("tgpt_adapter_dir")); }, []);

  async function gen() {
    if (!adapterDir) return;
    setBusy(true); setErr(null);
    try {
      const r = await api.generate({ adapter_dir: adapterDir, topic: topic || null, n, temperature: temp });
      setTweets(r.tweets);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }

  if (!adapterDir) return <p>No trained model yet. <a href="/train">Train one first.</a></p>;

  return (
    <>
      <h1>Generate tweets in your voice</h1>
      <div className="card">
        <label>Topic (optional): <input value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="e.g. ai and texas" /></label>
        <div className="row">
          <label className="inline">Count: <input type="number" style={{ width: 70 }} value={n} onChange={(e) => setN(+e.target.value)} /></label>
          <label className="inline">Temp: <input type="number" step="0.1" style={{ width: 70 }} value={temp} onChange={(e) => setTemp(+e.target.value)} /></label>
        </div>
        <button className="btn" onClick={gen} disabled={busy}>{busy ? "Generating…" : "Generate"}</button>
        {err && <p style={{ color: "#f4212e" }}>{err}</p>}
      </div>

      {tweets.map((t, i) => (
        <div className="tweet" key={i}>
          {t}
          <div><button className="btn secondary" style={{ marginTop: 8 }} onClick={() => navigator.clipboard.writeText(t)}>Copy</button></div>
        </div>
      ))}
    </>
  );
}
