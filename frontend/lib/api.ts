// Thin API client for the twitterGPT backend.
const BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

export const api = {
  base: BASE,

  upload(file: File, onProgress?: (pct: number) => void): Promise<{ upload_id: string; size_bytes: number }> {
    // XHR for upload progress (fetch has no upload progress event).
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const fd = new FormData();
      fd.append("file", file);
      xhr.open("POST", `${BASE}/upload`);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(Math.round((e.loaded / e.total) * 100));
      xhr.onload = () => (xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(xhr.responseText)));
      xhr.onerror = () => reject(new Error("upload failed"));
      xhr.send(fd);
    });
  },

  prepare(body: any) {
    return fetch(`${BASE}/dataset/prepare`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }).then(json<any>);
  },

  train(dataset_id: string, profile: string) {
    return fetch(`${BASE}/train`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id, profile }),
    }).then(json<{ job_id: string; backend: string; profile: string }>);
  },

  job(id: string) {
    return fetch(`${BASE}/jobs/${id}`).then(json<any>);
  },

  generate(body: any) {
    return fetch(`${BASE}/generate`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }).then(json<{ tweets: string[]; raw: string }>);
  },
};
