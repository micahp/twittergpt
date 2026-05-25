"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

export default function ReviewPage() {
  const router = useRouter();
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [opts, setOpts] = useState({ include_replies: true, strip_mentions: true, strip_urls: true, min_chars: 10 });
  const [meta, setMeta] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { setUploadId(localStorage.getItem("tgpt_upload_id")); }, []);

  async function prepare() {
    if (!uploadId) return;
    setBusy(true); setErr(null);
    try {
      const m = await api.prepare({ upload_id: uploadId, ...opts });
      setMeta(m);
      localStorage.setItem("tgpt_dataset_id", m.dataset_id);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }

  if (!uploadId) return <p>No upload found. <a href="/">Upload an archive first.</a></p>;

  const s = meta?.stats;
  return (
    <>
      <h1>Review your training set</h1>
      <p className="muted">Choose what counts as your voice, then build the dataset.</p>
      <div className="card">
        <label><input type="checkbox" checked={opts.include_replies}
          onChange={(e) => setOpts({ ...opts, include_replies: e.target.checked })} /> Include replies (your conversational voice)</label>
        <label><input type="checkbox" checked={opts.strip_mentions}
          onChange={(e) => setOpts({ ...opts, strip_mentions: e.target.checked })} /> Strip leading @mentions</label>
        <label><input type="checkbox" checked={opts.strip_urls}
          onChange={(e) => setOpts({ ...opts, strip_urls: e.target.checked })} /> Strip URLs</label>
        <label>Min characters: <input type="number" style={{ width: 80 }} value={opts.min_chars}
          onChange={(e) => setOpts({ ...opts, min_chars: +e.target.value })} /></label>
        <button className="btn" onClick={prepare} disabled={busy}>{busy ? "Building…" : "Build dataset"}</button>
        {err && <p style={{ color: "#f4212e" }}>{err}</p>}
      </div>

      {meta && (
        <div className="card">
          <h2>@{meta.account?.username} — {s.kept.toLocaleString()} examples kept</h2>
          <div className="grid muted">
            <div>Total tweets</div><div>{s.total.toLocaleString()}</div>
            <div>Retweets excluded</div><div>{s.retweets.toLocaleString()}</div>
            <div>Originals</div><div>{s.originals_in.toLocaleString()}</div>
            <div>Replies</div><div>{s.replies_in.toLocaleString()}</div>
            <div>Dropped (short/empty/dupe)</div><div>{(s.dropped_short + s.dropped_empty + s.dropped_dupe).toLocaleString()}</div>
            <div>Approx tokens</div><div>~{meta.approx_tokens_kept.toLocaleString()}</div>
          </div>
          <button className="btn" style={{ marginTop: 16 }} onClick={() => router.push("/train")}>Continue to training →</button>
        </div>
      )}
    </>
  );
}
