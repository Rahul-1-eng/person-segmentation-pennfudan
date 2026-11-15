import React, { useEffect, useState } from "react";
import { Github, FileDown, Image as ImageIcon } from "lucide-react";
import Hero from "./components/Hero";
import LiveDemo from "./components/LiveDemo";
import Method from "./components/Method";
import Results from "./components/Results";
import Robustness from "./components/Robustness";
import Findings from "./components/Findings";
import Reproduce from "./components/Reproduce";

const BACKEND = process.env.REACT_APP_BACKEND_URL;

export default function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch(`${BACKEND}/api/summary`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(null));
  }, []);

  return (
    <div data-testid="app-root">
      <nav className="nav">
        <div className="brand">
          Penn-Fudan <span>/</span> Person Segmentation
        </div>
        <a className="navlink" href="#demo">Live demo</a>
        <a className="navlink" href="#method">Method</a>
        <a className="navlink" href="#results">Results</a>
        <a className="navlink" href="#robustness">Robustness</a>
        <div className="spacer" />
        <a
          data-testid="github-repo-link"
          className="btn btn-blue"
          href="https://github.com/Rahul-1-eng/person-segmentation-pennfudan"
          target="_blank"
          rel="noreferrer"
        >
          <Github size={16} /> Code
        </a>
      </nav>

      <Hero data={data} />
      <LiveDemo backend={BACKEND} />
      <Method data={data} />
      <Results data={data} />
      <Robustness data={data} />
      <Findings data={data} />
      <Reproduce />

      <footer className="footer" data-testid="site-footer">
        <div>
          <b style={{ color: "var(--text)" }}>Binary Person Segmentation on Penn-Fudan</b> — ML course
          project by Rahul Kumar Sahoo (2401CS09), Department of CSE, IIT Patna.
        </div>
        <div className="spacer" />
        <a href={`${process.env.PUBLIC_URL}/downloads/Person_Segmentation_Report.pdf`} target="_blank" rel="noreferrer" data-testid="download-report-link">
          <FileDown size={14} /> Report (PDF)
        </a>
        <a href={`${process.env.PUBLIC_URL}/downloads/Person_Segmentation_Explained.pdf`} target="_blank" rel="noreferrer" data-testid="download-explainer-link">
          <FileDown size={14} /> Explainer (PDF)
        </a>
        <a href={`${process.env.PUBLIC_URL}/downloads/poster.png`} target="_blank" rel="noreferrer" data-testid="download-poster-link">
          <ImageIcon size={14} /> Poster (PNG)
        </a>
      </footer>
    </div>
  );
}
