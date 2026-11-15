import React from "react";

export default function Findings({ data }) {
  const eda = data?.eda || {};
  const paired = (data?.paired || []).filter((p) => p.excludes_zero);
  return (
    <section className="block" id="findings">
      <div className="section-head">
        <div className="kicker">DISCUSSION</div>
        <h2>Findings, stated carefully</h2>
      </div>
      <div className="grid-2">
        <div className="glass finding" data-testid="finding-encoder">
          <h3><span className="dot g" /> Pretraining beats recipe tweaks</h3>
          <p>
            The ImageNet-pretrained ResNet34 encoder added +0.0736 [0.0500, 0.0982] mean per-image IoU
            over an identically-trained from-scratch U-Net. Mild augmentation's +0.0271 [0.0086, 0.0446]
            is the only other interval that excludes zero. Everything else — stronger augmentation,
            BCE+Dice, fine-tuned DeepLabV3 over ResNet34 U-Net — is not separable at three seeds.
          </p>
        </div>
        <div className="glass finding" data-testid="finding-distractor">
          <h3><span className="dot a" /> Distractor training works, cheaply</h3>
          <p>
            FP_D fell from 0.1314 to 0.0273 with the largest cut exactly on the hardest distractor (mirror
            panel: 0.6209 → 0.0714), costing 0.003 clean IoU. The model learns context, not magic: a
            pixel-perfect mirror image of a person is a person at the pixel level, and no real-reflection
            photos were tested.
          </p>
        </div>
        <div className="glass finding" data-testid="finding-errors">
          <h3><span className="dot r" /> Some errors belong to the dataset</h3>
          <p>
            The worst test image ({eda.worst_image || "PennPed00042"}, IoU {eda.worst_iou || 0.298}) hides
            a partially visible pedestrian — and several large "false positives" are people missing from
            the instance masks. Inspecting mistakes before tuning saved wasted effort. Fudan images score
            ~0.06 higher than Penn across models ({eda.fudan_iou_augmented} vs {eda.penn_iou_augmented}
            for the augmented U-Net).
          </p>
        </div>
        <div className="glass finding" data-testid="finding-honesty">
          <h3><span className="dot" /> What this study does not claim</h3>
          <p>
            34 test images and 3 seeds keep every interval wide; one image is ~3% of the test set. Every
            photo contains a person, so nothing transfers to empty scenes. The distractor study is a
            synthetic stress test measured honestly — not a solution to real reflections.
          </p>
        </div>
      </div>
      {paired.length > 0 && (
        <div className="glass" style={{ padding: "18px 22px", marginTop: 22 }} data-testid="paired-note">
          <span style={{ color: "var(--muted)", fontSize: 13 }}>
            Paired hierarchical bootstrap intervals excluding zero:{" "}
            {paired.map((p) => `${p.comparison} (+${p.diff.toFixed(4)})`).join(" · ")}
          </span>
        </div>
      )}
    </section>
  );
}
