import React from "react";

export default function Hero({ data }) {
  const h = data?.headline || {};
  const chips = [
    { num: h.best_mean_iou != null ? h.best_mean_iou.toFixed(4) : "—", lbl: "Best mean per-image IoU (fine-tuned DeepLabV3)", cls: "green" },
    { num: h.encoder_gain != null ? "+" + h.encoder_gain.toFixed(4) : "—", lbl: "Gain from ImageNet-pretrained encoder over from-scratch U-Net", cls: "" },
    { num: h.fp_before != null && h.fp_after != null ? `${h.fp_before.toFixed(4)} → ${h.fp_after.toFixed(4)}` : "—", lbl: "Distractor false-positive rate before → after distractor training", cls: "amber" },
    { num: <>{h.runs || 27}<small> runs / {h.seeds || 3} seeds</small></>, lbl: `${h.images || 170} images, frozen 102 / 34 / 34 split`, cls: "" },
  ];
  return (
    <header className="hero">
      <div className="eyebrow reveal">MACHINE LEARNING COURSE PROJECT · IIT PATNA · 2401CS09</div>
      <h1 className="reveal d1">
        Binary <span className="accent">person segmentation</span> on Penn-Fudan — and a stress test for
        reflections and shadows
      </h1>
      <p className="sub reveal d2">
        Nine model configurations were trained three times each under one frozen, leak-free protocol: a
        plain U-Net, U-Net variants with augmentation and loss ablations, a ResNet34-encoder U-Net and
        COCO-pretrained DeepLabV3. Six synthetic reflection- and shadow-like distractors measure whether
        models call mirror images "person". Everything below is generated from the public result files.
      </p>
      <div className="metric-row reveal d3" data-testid="hero-metrics">
        {chips.map((c, i) => (
          <div className="metric" key={i}>
            <div className={`num ${c.cls}`}>{c.num}</div>
            <div className="lbl">{c.lbl}</div>
          </div>
        ))}
      </div>
    </header>
  );
}
