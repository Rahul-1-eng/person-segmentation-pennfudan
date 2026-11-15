import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

export default function Results({ data }) {
  const rows = data?.results || [];
  const chartData = rows
    .filter((r) => r.mean_iou > 0.3)
    .map((r) => ({ name: r.model.replace("ResNet34-encoder U-Net", "ResNet34 U-Net").replace("DeepLabV3", "DeepLab"), mean_iou: r.mean_iou }));
  const fpRows = rows.filter((r) => r.fp_d != null).map((r) => ({ name: r.model, fp_d: r.fp_d }));

  return (
    <section className="block" id="results">
      <div className="section-head">
        <div className="kicker">RESULTS</div>
        <h2>What the numbers say</h2>
        <p>
          Test results at t = 0.5 under the frozen protocol. Pretrained encoders separate clearly from
          from-scratch models; only two paired differences in the whole study exclude zero.
        </p>
      </div>
      <div className="table-wrap" data-testid="results-table">
        <table className="results">
          <thead>
            <tr>
              <th>Model</th><th>Family</th><th>Val IoU</th>
              <th className="num">Pooled IoU</th><th className="num">Mean image IoU</th>
              <th className="num">FP_D</th><th className="num">Params</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.model} className={r.model === "DeepLabV3 fine-tuned" ? "rowbest" : ""} data-testid={`results-row-${i}`}>
                <td>{r.model}</td>
                <td><span className={`pill ${r.kind === "pretrained" ? "pre" : r.kind === "scratch" ? "scr" : "tri"}`}>{r.kind}</span></td>
                <td className="mono" style={{ color: "var(--muted)", fontSize: 12.5 }}>{r.val_iou}</td>
                <td className="num">{r.pooled.toFixed(4)}</td>
                <td className="num">{r.mean_iou.toFixed(4)}{r.mean_iou_std ? ` ± ${r.mean_iou_std.toFixed(4)}` : ""}</td>
                <td className="num">{r.fp_d != null ? r.fp_d.toFixed(4) : "—"}</td>
                <td className="num" style={{ color: "var(--muted)" }}>{r.params}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid-2" style={{ marginTop: 22 }}>
        <div className="glass chart-card" data-testid="chart-mean-iou">
          <h3>Test mean per-image IoU</h3>
          <div className="sub">Every learned model vs the trivial pixel random forest</div>
          <ResponsiveContainer width="100%" height={330}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 30, right: 30, top: 12, bottom: 4 }}>
              <CartesianGrid stroke="#2a3048" horizontal={false} />
              <XAxis type="number" domain={[0, 0.9]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={128} tick={{ fill: "#94a3b8", fontSize: 10.5 }} />
              <Tooltip contentStyle={{ background: "#131620", border: "1px solid #2a3048", borderRadius: 10, color: "#f8fafc", fontSize: 12 }} />
              <Bar dataKey="mean_iou" radius={[0, 5, 5, 0]} barSize={16}
                fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="glass chart-card" data-testid="chart-fpd">
          <h3>Distractor false-positive rate FP_D</h3>
          <div className="sub">Fraction of pasted reflection/shadow regions called person (lower is better)</div>
          <ResponsiveContainer width="100%" height={330}>
            <BarChart data={fpRows} layout="vertical" margin={{ left: 30, right: 30, top: 12, bottom: 4 }}>
              <CartesianGrid stroke="#2a3048" horizontal={false} />
              <XAxis type="number" domain={[0, 0.2]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={128} tick={{ fill: "#94a3b8", fontSize: 10.5 }} />
              <Tooltip contentStyle={{ background: "#131620", border: "1px solid #2a3048", borderRadius: 10, color: "#f8fafc", fontSize: 12 }} />
              <ReferenceLine x={0.0273} stroke="#10b981" strokeDasharray="4 3"
                label={{ value: "0.0273 after distractor training", fill: "#10b981", fontSize: 10.5, position: "insideTopRight" }} />
              <Bar dataKey="fp_d" radius={[0, 5, 5, 0]} barSize={16} fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
