param(
    [string[]]$Experiments = @("baseline", "augmented", "strong_aug", "resnet34", "distract", "resnet34_distract"),
    [int[]]$Seeds = @(20261103, 1, 2)
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PWD\src"
if (-not (Test-Path "artifacts/splits.json")) { python scripts/make_split.py --config config.yaml }
foreach ($experiment in $Experiments) {
    foreach ($seed in $Seeds) {
        python -m person_segmentation.train --config config.yaml --experiment $experiment --seed $seed
    }
}
python -m person_segmentation.baselines --config config.yaml
python -m person_segmentation.evaluate --config config.yaml --zero-shot
python -m person_segmentation.robustness --config config.yaml --zero-shot
python -m pip freeze > artifacts/environment.txt
python scripts/make_tables.py --config config.yaml --pair baseline augmented --pair augmented strong_aug --pair strong_aug resnet34 --pair resnet34 resnet34_distract
python scripts/overlay.py --config config.yaml --experiment augmented
python scripts/make_figures.py --config config.yaml
