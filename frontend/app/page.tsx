"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

export default function UploadPage() {
  const router = useRouter();
  const [pct, setPct] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true); setErr(null); setPct(0);
    try {
      const { upload_id } = await api.upload(file, setPct);
      localStorage.setItem("tgpt_upload_id", upload_id);
      router.push("/review");
    } catch (e: any) {
      setErr(e.message || "upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <h1>Upload your Twitter archive</h1>
      <p className="muted">
        Request your archive from X (Settings → Your account → Download an archive),
        then drop the <code>.zip</code> here. We only read your tweets — media and DMs are ignored.
      </p>
      <div className="card">
        <input type="file" accept=".zip" onChange={onFile} disabled={busy} />
        {busy && (
          <div style={{ marginTop: 16 }}>
            <div className="bar"><span style={{ width: `${pct}%` }} /></div>
            <p className="muted">{pct}% uploaded…</p>
          </div>
        )}
        {err && <p style={{ color: "#f4212e" }}>{err}</p>}
      </div>
      <p className="muted">Large archives (multi-GB) may take a while. MVP uses a single upload;
        resumable chunked upload is planned (PRD §8).</p>
    </>
  );
}
