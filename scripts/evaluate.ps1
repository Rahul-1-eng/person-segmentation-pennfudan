$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PWD\src"
python -m person_segmentation.baselines --config config.yaml
python -m person_segmentation.evaluate --config config.yaml --zero-shot
python scripts/make_tables.py --config config.yaml
