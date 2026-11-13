# Presenter notes

1. Say the task in one sentence: every pixel is person or not person; all people are merged.
2. State the three hypotheses and that success was fixed before the runs.
3. One gap: no person-free or reflection images in Penn-Fudan.
4. Foreground is small, so accuracy is useless; IoU is the metric.
5. Split frozen once; test used once; threshold from validation only.
6. Walk the architecture left to right: blue is training, orange is evaluation.
7. Same settings for every model; three seeds.
8. Read the table with intervals, not single numbers. If asked about v1: it was undertrained and inside noise.
9. Curves show whether training had converged.
10. Show failures first.
11. A mirror copy of a person looks like a person; only context separates them.
12. Worst cases are false positives; Penn is harder than Fudan.
13. Be direct about the 34-image test set.
14. Claim only what the final tables show.
