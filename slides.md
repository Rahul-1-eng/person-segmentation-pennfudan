# Slide outline (15 slides, built by scripts/create_presentation.py)

1. Title, name, roll number
2. Problem, motivation, hypotheses H1 to H3
3. Literature table and gap
4. Dataset overview and audit (eda.png)
5. Split and leakage controls
6. Workflow and architecture (architecture.png)
7. Baselines and experiment protocol
8. Main results table (results_table.csv; falls back to labelled v1 numbers if the v2 runs are missing)
9. Training curves and metric comparison
10. Qualitative results (qualitative.png)
11. Robustness to reflections and shadows (distractors.png, robustness.png)
12. Error analysis and ablations (overlay of worst images)
13. Limitations and implications
14. Conclusion and future work
15. References

Rebuild after any new run: `python scripts/make_figures.py` then `python scripts/create_presentation.py`.
