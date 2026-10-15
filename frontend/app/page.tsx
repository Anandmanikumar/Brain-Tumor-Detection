"use client";

import { useEffect, useMemo, useState } from "react";

type Job = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  message?: string | null;
  prediction_available: boolean;
};

const modalities = [
  { key: "flair", label: "FLAIR", detail: "Fluid-sensitive sequence" },
  { key: "t1", label: "T1", detail: "Anatomical sequence" },
  { key: "t1ce", label: "T1ce", detail: "Post-contrast sequence" },
  { key: "t2", label: "T2", detail: "T2-weighted sequence" },
] as const;

export default function Home() {
  const [files, setFiles] = useState<Record<string, File | null>>({
    flair: null, t1: null, t1ce: null, t2: null,
  });
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const apiBase = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
  const apiKey = process.env.NEXT_PUBLIC_DEMO_API_KEY || "";

  const allSelected = useMemo(() => modalities.every(m => files[m.key]), [files]);
  const isActive = job?.status === "queued" || job?.status === "running";

  useEffect(() => {
    if (!job || !isActive) return;
    const start = Date.parse(job.created_at);
    const updateElapsed = () => setElapsed(Math.max(0, Math.floor((Date.now() - start) / 1000)));
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [job, isActive]);

  useEffect(() => {
    if (!job || !isActive) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch(`${apiBase}/api/jobs/${job.job_id}`, {
          headers: { "X-API-Key": apiKey },
          cache: "no-store",
        });
        if (!res.ok) throw new Error(`Status check failed (${res.status}).`);
        const next: Job = await res.json();
        if (!cancelled) {
          setJob(next);
          if (next.status === "completed" || next.status === "failed") return;
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Unable to check job status.");
      }
      if (!cancelled) window.setTimeout(poll, 5000);
    };
    const timer = window.setTimeout(poll, 2500);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [job?.job_id, job?.status, apiBase, apiKey, isActive]);

  function chooseFile(key: string, file: File | null) {
    setFiles(old => ({ ...old, [key]: file }));
    setError("");
  }

  async function submit() {
    setError("");
    setJob(null);
    if (!allSelected) {
      setError("Please select all four MRI modalities.");
      return;
    }
    if (!apiKey || apiKey.includes("REPLACE_WITH")) {
      setError("Configure NEXT_PUBLIC_DEMO_API_KEY in frontend/.env.local first.");
      return;
    }
    const form = new FormData();
    for (const modality of modalities) form.append(modality.key, files[modality.key] as File);
    setSubmitting(true);
    try {
      const response = await fetch(`${apiBase}/api/jobs`, {
        method: "POST",
        headers: { "X-API-Key": apiKey },
        body: form,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Upload failed (${response.status}).`);
      setJob(data as Job);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setSubmitting(false);
    }
  }

  async function downloadResult() {
    if (!job) return;
    setError("");
    try {
      const response = await fetch(`${apiBase}/api/jobs/${job.job_id}/result`, {
        headers: { "X-API-Key": apiKey },
      });
      if (!response.ok) throw new Error((await response.text()) || "Could not download result.");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `clusture_prediction_${job.job_id}.nii.gz`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Download failed.");
    }
  }

  const elapsedText = `${Math.floor(elapsed / 60).toString().padStart(2, "0")}:${(elapsed % 60).toString().padStart(2, "0")}`;

  return (
    <main className="shell">
      <nav className="nav">
        <div className="brandMark">C</div>
        <div className="brand">clusture<span> AI</span><small>RESEARCH SHOWCASE</small></div>
        <div className="navRight"><span className="statusDot" /> Experimental demo</div>
      </nav>

      <section className="hero">
        <div className="eyebrow"><span /> MEDICAL IMAGING · AI RESEARCH</div>
        <h1>Brain MRI<br /><em>segmentation.</em></h1>
        <p className="heroText">Explore an experimental nnU-Net workflow designed to assist research and radiology review by generating a tumor segmentation mask from multi-modal brain MRI.</p>
        <div className="heroMeta">
          <span><b>01</b> Upload study</span><span><b>02</b> Run inference</span><span><b>03</b> Review mask</span>
        </div>
      </section>

      <section className="workspace">
        <div className="sectionHead">
          <div><div className="eyebrow muted">WORKSPACE / 01</div><h2>New analysis</h2></div>
          <div className="formatPill">NIfTI · .nii / .nii.gz</div>
        </div>
        <p className="sectionIntro">Select all four modalities from the same patient and MRI study. Files must be correctly aligned and compatible with the model's expected input.</p>

        <div className="fileGrid">
          {modalities.map((m, i) => (
            <label className={`fileCard ${files[m.key] ? "selected" : ""}`} key={m.key}>
              <input type="file" accept=".nii,.gz,.nii.gz" onChange={e => chooseFile(m.key, e.target.files?.[0] || null)} />
              <div className="fileTop"><span className="fileIndex">0{i + 1}</span><span className={`fileIcon ${files[m.key] ? "ready" : ""}`}>{files[m.key] ? "✓" : "↥"}</span></div>
              <strong>{m.label}</strong><span className="fileDetail">{m.detail}</span>
              <span className="fileName">{files[m.key]?.name || "Choose NIfTI file"}</span>
            </label>
          ))}
        </div>

        <div className="processingNote">
          <div className="clockIcon">◷</div>
          <div><strong>Allow time for processing</strong><p>Current observed runtime is approximately 28–29 minutes per case. Actual duration may vary. You can leave this page open while the job runs.</p></div>
          <div className="runtime">~30<span>MIN</span></div>
        </div>

        <button className="primaryButton" disabled={!allSelected || submitting || !!isActive} onClick={submit}>
          {submitting ? "Uploading study…" : isActive ? "Inference in progress…" : "Start segmentation"} <span>↗</span>
        </button>
        {error && <div className="errorBox" role="alert">{error}</div>}

        {job && <div className="jobPanel">
          <div className="jobHeader"><div><div className="eyebrow muted">JOB STATUS</div><h3>{job.status === "completed" ? "Analysis complete" : job.status === "failed" ? "Analysis failed" : "Processing your study"}</h3></div><span className={`stateBadge ${job.status}`}>{job.status}</span></div>
          {(job.status === "queued" || job.status === "running") && <div className="progressTrack"><div className="progressIndeterminate" /></div>}
          <p>{job.message || "Waiting for the inference worker."}</p>
          {isActive && <p className="elapsed">Elapsed time: <strong>{elapsedText}</strong> · Keep this tab open for live updates.</p>}
          {job.status === "completed" && job.prediction_available && <button className="secondaryButton" onClick={downloadResult}>Download segmentation (.nii.gz) ↓</button>}
          <div className="jobId">Job ID: {job.job_id}</div>
        </div>}
      </section>

      <section className="resultsInfo">
        <div className="sectionHead"><div><div className="eyebrow muted">WORKSPACE / 02</div><h2>What comes next</h2></div></div>
        <div className="featureGrid">
          <article><span>01</span><h3>Segmentation mask</h3><p>Download the predicted NIfTI mask for review in a compatible medical-image viewer.</p></article>
          <article><span>02</span><h3>Visual review</h3><p>The next iteration can overlay the mask on the original MRI and provide slice navigation.</p></article>
          <article><span>03</span><h3>Evaluation</h3><p>Dice and HD95 require corresponding ground-truth masks; they cannot be inferred from a prediction alone.</p></article>
        </div>
      </section>

      <footer><div className="brand">clusture<span> AI</span></div><p>Research prototype · Not for clinical diagnosis or treatment decisions.</p></footer>
      <div className="disclaimer">This experimental output is not a diagnosis and has not been established as safe or effective for clinical use. All findings require qualified clinical review.</div>
    </main>
  );
}
