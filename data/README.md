# Dataset card: Penn-Fudan Pedestrian

- Source: https://www.cis.upenn.edu/~jshi/ped_html/ (Wang et al., ACCV 2007)
- 170 images, 345 labelled pedestrians, from Fudan University (Fudan*) and University of Pennsylvania (Penn*) street scenes
- Every image has at least one pedestrian; pedestrians are upright; no person-free images
- Mask: instance labels merged into one binary foreground mask for this project
- Split: 102 train / 34 validation / 34 test, frozen in artifacts/splits.json
- Licence: no explicit licence statement found on the source page; check before redistributing. Do not commit the images.
- Known bias: no reflections or lookalike shadows in the data, so robustness is tested with synthetic distractors plus a small self-annotated real set
