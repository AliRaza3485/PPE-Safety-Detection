"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Detection = {
  label: string;
  confidence: number;
  bbox: number[];
  violation: boolean;
};

type Summary = {
  total: number;
  counts: Record<string, number>;
  violation_count: number;
  compliant: boolean;
};

type PredictResponse = {
  filename: string;
  detections: Detection[];
  summary: Summary;
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function onSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(f ? URL.createObjectURL(f) : null);
  }

  async function onSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      setResult((await res.json()) as PredictResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const summary = result?.summary;

  return (
    <main>
      <h1>🦺 PPE Safety Detection</h1>
      <p className="subtitle">
        Upload a worksite image to detect hard hats, masks and safety vests —
        and flag any missing PPE.
      </p>

      <div className="card">
        <input type="file" accept="image/*" onChange={onSelect} />
        <div style={{ marginTop: 12 }}>
          <button onClick={onSubmit} disabled={!file || loading}>
            {loading ? "Detecting…" : "Detect PPE"}
          </button>
        </div>
        {preview && <img className="preview" src={preview} alt="preview" />}
        {error && <p className="violation">⚠️ {error}</p>}
      </div>

      {summary && (
        <div className="card">
          <span className={`badge ${summary.compliant ? "ok" : "bad"}`}>
            {summary.compliant ? "✅ Compliant" : `⚠️ ${summary.violation_count} violation(s)`}
          </span>

          <table>
            <thead>
              <tr>
                <th>Class</th>
                <th>Confidence</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {result!.detections.map((d, i) => (
                <tr key={i} className={d.violation ? "violation" : undefined}>
                  <td>{d.label}</td>
                  <td>{(d.confidence * 100).toFixed(1)}%</td>
                  <td>{d.violation ? "Violation" : "OK"}</td>
                </tr>
              ))}
              {result!.detections.length === 0 && (
                <tr>
                  <td colSpan={3}>No PPE-related objects detected.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
