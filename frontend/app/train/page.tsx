"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

export default function TrainPage() {
  const router = useRouter();
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [profile, setProfile] = useState("local-3b");
  const [job, setJob] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const poll = useRef<any>(null);

  useEffect(() => { setDatasetId(localStorage.getItem("tgpt_dataset_id")); }, []);
  useEffect(() => () => clearInterval(poll.current), []);

  async function start() {
    if (!datasetId) return;
    setErr(null);
    try {
      const { job_id } = await api.train(datasetId, profile);
      poll.current = setInterval(async () => {
        const j = await api.job(job_id);
        setJob(j);
        if (j.status === "done") {
          localStorage.setItem("tgpt_adapter_dir", j.adapter_dir);
          clearInterval(poll.current);
        }
        if (j.status === "failed") clearInterval(poll.current);
      }, 3000);
    } catch (e: any) { setErr(e.message); }
  }

  if (!datasetId) return <p>No dataset. <a href="/review">Build one first.</a></p>;

  return (
    <>
      <h1>Train your model</h1>
      <div className="card">
        <label>Compute profile:
          <select value={profile} onChange={(e) => setProfile(e.target.value)}>
            <option value="local-3b">local-3b — Qwen2.5-3B on your desktop GPU</option>
            <option value="daytona-7b">daytona-7b — Qwen2.5-7B on Daytona (M2, stub)</option>
          </select>
        </label>
        <button className="btn" onClick={start} disabled={job && ["queued", "running"].includes(job.status)}>
          {job && ["queued", "running"].includes(job.status) ? "Training…" : "Start training"}
        </button>
        {err && <p style={{ color: "#f4212e" }}>{err}</p>}
      </div>

      {job && (
        <div className="card">
          <div className="row">
            <strong>Status:</strong> <span>{job.status}</span>
            <span className="muted">({job.backend} / {job.profile})</span>
          </div>
          {job.error && <p style={{ color: "#f4212e" }}>{job.error}</p>}
          {job.log_tail && <pre>{job.log_tail}</pre>}
          {job.status === "done" && (
            <button className="btn" onClick={() => router.push("/generate")}>Generate tweets →</button>
          )}
        </div>
      )}
    </>
  );
}
