import React, { useRef, useState } from "react";
import { Upload, Wand2 } from "lucide-react";

const SAMPLES = ["/samples/sample1.jpg", "/samples/sample2.jpg", "/samples/sample3.jpg"];

export default function LiveDemo({ backend }) {
  const [orig, setOrig] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  const run = async (file) => {
    if (!file) return;
    setErr("");
    setResult(null);
    setOrig(URL.createObjectURL(file));
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${backend}/api/segment`, { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
      setResult(await r.json());
    } catch (e) {
      setErr(e.message || "Segmentation request failed");
    } finally {
      setBusy(false);
    }
  };

  const sample = async (url) => {
    const blob = await (await fetch(url)).blob();
    run(new File([blob], url.split("/").pop(), { type: blob.type || "image/jpeg" }));
  };

  return (
    <section className="block" id="demo">
      <div className="section-head">
        <div className="kicker">LIVE DEMO</div>
        <h2>Paint every person in a photo, right now</h2>
        <p>
          This runs the study's zero-shot reference model — DeepLabV3-ResNet50 with the COCO person class,
          the exact configuration scored at 0.7567 mean per-image IoU on Penn-Fudan — in the backend.
          Upload any street photo or pick a sample.
        </p>
      </div>
      <div className="glass demo-panel" data-testid="demo-panel">
        <div
          className={`dropzone ${drag ? "drag" : ""}`}
          data-testid="demo-dropzone"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); run(e.dataTransfer.files[0]); }}
        >
          <Upload size={26} style={{ marginBottom: 8 }} />
          <div style={{ fontSize: 15, fontWeight: 600 }}>Drop a photo here or click to choose one</div>
          <div style={{ fontSize: 12.5, marginTop: 6 }}>JPG or PNG, up to 8 MB. Images are not stored — only an anonymous count of runs.</div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            data-testid="demo-upload-input"
            onChange={(e) => run(e.target.files[0])}
          />
        </div>
        <div className="samples" data-testid="demo-samples">
          <span style={{ color: "var(--muted)", fontSize: 13, alignSelf: "center" }}>No photo handy? Try:</span>
          {SAMPLES.map((s, i) => (
            <img
              key={s}
              className="sample-thumb"
              data-testid={`demo-sample-${i + 1}`}
              src={s}
              alt={`sample ${i + 1}`}
              onClick={() => sample(s)}
            />
          ))}
        </div>
        {busy && (
          <div className="loading" data-testid="demo-loading">
            <div className="spin" /> Running DeepLabV3-ResNet50 on your image…
          </div>
        )}
        {err && <div className="err" data-testid="demo-error">{err}</div>}
        {result && orig && (
          <div data-testid="demo-result">
            <div className="demo-stage">
              <div className="imgbox">
                <img src={orig} alt="original" data-testid="demo-original-image" />
                <span className="tag">Input</span>
              </div>
              <div className="imgbox">
                <img
                  src={`data:image/png;base64,${result.overlay_png_base64}`}
                  alt="segmentation overlay"
                  data-testid="demo-overlay-image"
                />
                <span className="tag"><Wand2 size={11} style={{ verticalAlign: -2 }} /> Predicted person mask</span>
              </div>
            </div>
            <div className="stat-chips" data-testid="demo-stats">
              <span className="chip">person pixels <b>{result.person_pixel_pct}%</b></span>
              <span className="chip">inference <b>{result.inference_ms} ms</b></span>
              <span className="chip">resolved to <b>{result.image_size[0]}×{result.image_size[1]}</b></span>
              <span className="chip">model <b>DeepLabV3-ResNet50</b></span>
            </div>
            <div className="demo-note">{result.note}</div>
          </div>
        )}
      </div>
    </section>
  );
}
