import React from "react";

const STEPS = [
  { n: "01", t: "One frozen split, made once", p: "170 images split 102 / 34 / 34 with seed 20261103 and saved to splits.json. It is never regenerated, so every run in the study sees exactly the same exam." },
  { n: "02", t: "Train nine configurations × 3 seeds", p: "Plain U-Net (7.76 M params), augmentation and loss variants, ResNet34-encoder U-Net (24.52 M) and fine-tuned DeepLabV3 (42.00 M). Best epoch picked by validation IoU only." },
  { n: "03", t: "Test exactly once", p: "Threshold tuned on validation, then a single test pass: pooled IoU, mean per-image IoU with bootstrap CIs, precision, recall, F1, and a Fudan-versus-Penn breakdown." },
  { n: "04", t: "Stress-test with six distractors", p: "Floor reflections, framed mirror panels, cast and detached shadows, an animal-shaped shadow and a darkness patch — pasted only on background pixels so the ground truth never changes." },
];

export default function Method({ data }) {
  return (
    <section className="block" id="method">
      <div className="section-head">
        <div className="kicker">METHOD</div>
        <h2>How the study is built</h2>
        <p>
          Blue blocks below fit models on training images; peach blocks evaluate. Nothing that could leak
          the test set — checkpoint, epoch, threshold — is ever decided on it.
        </p>
      </div>
      <div className="grid-4">
        {STEPS.map((s) => (
          <div className="glass step-card" key={s.n} data-testid={`method-step-${s.n}`}>
            <span className="n">{s.n}</span>
            <h3>{s.t}</h3>
            <p>{s.p}</p>
          </div>
        ))}
      </div>
      <div className="glass figure-card" style={{ marginTop: 22 }}>
        <img src="/figures/architecture.png" alt="Study pipeline: blue training blocks, peach evaluation blocks" data-testid="method-architecture-figure" />
        <div className="cap">Figure: end-to-end pipeline. All fitting uses the 102 training images; the 34 test images are scored once.</div>
      </div>
      <div className="grid-2" style={{ marginTop: 22 }}>
        <div className="glass figure-card">
          <img src="/figures/distractors.png" alt="The six synthetic distractor types" data-testid="method-distractors-figure" />
          <div className="cap">The six synthetic distractors, pasted on background only — the yellow outline marks the region that must stay background.</div>
        </div>
        <div className="glass figure-card">
          <img src="/figures/eda.png" alt="Exploratory data analysis of Penn-Fudan" data-testid="method-eda-figure" />
          <div className="cap">
            Dataset audit: foreground is a small pixel fraction (so IoU, not accuracy), and validation is ~65% Penn versus ~55% in training — reported, not hidden.
          </div>
        </div>
      </div>
    </section>
  );
}
