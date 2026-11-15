import React from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function Robustness({ data }) {
  const rb = data?.robustness || {};
  const chart = (rb.conditions || []).map((c, i) => ({
    cond: c,
    plain: rb.plain_resnet34?.[i],
    trained: rb.distractor_trained?.[i],
  }));

  return (
    <section className="block" id="robustness">
      <div className="section-head">
        <div className="kicker">ROBUSTNESS STUDY</div>
        <h2>Do models see people in mirrors?</h2>
        <p>
          Six distractor types were pasted into test backgrounds. The framed mirror panel fools every
          model trained without distractors — a doorway-shaped mirror looks like a walkable opening.
          Distractor training fixes exactly that case.
        </p>
      </div>
      <div className="glass chart-card" data-testid="robustness-chart">
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={chart} margin={{ left: 8, right: 16, top: 18, bottom: 4 }}>
            <CartesianGrid stroke="#2a3048" vertical={false} />
            <XAxis dataKey="cond" tick={{ fill: "#94a3b8", fontSize: 12 }} />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11.5 }} />
            <Tooltip contentStyle={{ background: "#131620", border: "1px solid #2a3048", borderRadius: 10, color: "#f8fafc", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12.5 }} />
            <Bar name="ResNet34 U-Net (plain)" dataKey="plain" fill="#f2b84b" radius={[5, 5, 0, 0]} barSize={26} />
            <Bar name="+ distractor training" dataKey="trained" fill="#c95b1b" radius={[5, 5, 0, 0]} barSize={26} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="grid-2" style={{ marginTop: 22 }}>
        <div className="glass figure-card">
          <img src="/figures/robustness.png" alt="Clean IoU and FP_D per model and condition" data-testid="robustness-full-figure" />
          <div className="cap">Clean IoU (top) and FP_D (bottom) for every model and distractor type, from the study's robustness_metrics.csv.</div>
        </div>
        <div className="glass figure-card">
          <img src="/figures/metric_comparison.png" alt="Test metrics for all runs" data-testid="robustness-metrics-figure" />
          <div className="cap">All test metrics side by side. FP_D drops 0.1314 → 0.0273 for the ResNet34 U-Net while clean IoU moves by only 0.003.</div>
        </div>
      </div>
    </section>
  );
}
