import React from "react";
import { Github, Download } from "lucide-react";

const CMD = [
  ["# clone and set up (Python 3.10+)", "cmt"],
  ["git clone https://github.com/Rahul-1-eng/person-segmentation-pennfudan.git", "hl"],
  ["cd person-segmentation-pennfudan", ""],
  ["pip install -r requirements.txt", ""],
  ["", ""],
  ["# get the dataset and make the frozen split (once, never regenerate)", "cmt"],
  ["python scripts/download_dataset.py", ""],
  ["python scripts/make_split.py --config config.yaml", ""],
  ["", ""],
  ["# train every configuration and regenerate all tables/figures", "cmt"],
  ["bash scripts/run_all.sh", ""],
  ["", ""],
  ["# or a single run", "cmt"],
  ["python -m person_segmentation.train --config config.yaml --experiment resnet34 --seed 1", ""],
];

export default function Reproduce() {
  return (
    <section className="block" id="reproduce">
      <div className="section-head">
        <div className="kicker">REPRODUCE</div>
        <h2>Every number traces back to a CSV</h2>
        <p>
          The repository holds the pipeline modules (data, model, losses, distractors, metrics, train,
          evaluate, robustness, baselines), the frozen split file and all result artifacts. Three commands
          reproduce the whole study.
        </p>
      </div>
      <div className="grid-2">
        <div className="codeblock" data-testid="reproduce-commands">
          {CMD.map(([line, cls], i) => (
            <div key={i} className={cls}>
              {line || "\u00a0"}
            </div>
          ))}
        </div>
        <div>
          <div className="glass finding" data-testid="reproduce-links">
            <h3><span className="dot g" /> Code and materials</h3>
            <p style={{ marginBottom: 14 }}>
              Repository: <a href="https://github.com/Rahul-1-eng/person-segmentation-pennfudan" target="_blank" rel="noreferrer" data-testid="reproduce-github-link">github.com/Rahul-1-eng/person-segmentation-pennfudan</a> (public)
            </p>
            <p>
              This page's downloads: the camera-ready report, the layman explainer and the poster in PNG
              and PDF. The editable poster (PPTX) and LaTeX source are in the repository under{" "}
              <span className="mono" style={{ fontSize: 12 }}>deliverables/poster/</span>.
            </p>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 16 }}>
              <a className="btn btn-blue" href="https://github.com/Rahul-1-eng/person-segmentation-pennfudan" target="_blank" rel="noreferrer" data-testid="reproduce-github-btn">
                <Github size={15} /> Open repository
              </a>
              <a className="btn btn-ghost" href="/downloads/Person_Segmentation_Report.pdf" download data-testid="reproduce-report-btn">
                <Download size={15} /> Report PDF
              </a>
              <a className="btn btn-ghost" href="/downloads/poster.pdf" download data-testid="reproduce-poster-btn">
                <Download size={15} /> Poster PDF
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
