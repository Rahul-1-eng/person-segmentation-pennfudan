from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_AUG: dict[str, float] = {
    "flip": 0.5,
    "affine": 0.7,
    "scale_min": 0.6,
    "scale_max": 1.4,
    "rotate": 10.0,
    "translate": 0.1,
    "color": 0.5,
    "color_strength": 0.2,
    "blur": 0.2,
    "noise": 0.2,
}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data_root: Path
    split_file: Path
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    device: str
    threshold: float
    checkpoint_dir: Path
    resize: str = "squash"
    scheduler: str = "cosine"
    tta: bool = False
    pretrained: bool = True
    distractor_probability: float = 0.3
    aug: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_AUG))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path)
        values: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
        root = source.parent
        data_root = Path(values["data_root"])
        split_file = Path(values["split_file"])
        checkpoint_dir = Path(values["checkpoint_dir"])
        if not data_root.is_absolute():
            data_root = root / data_root
        if not split_file.is_absolute():
            split_file = root / split_file
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = root / checkpoint_dir
        fractions = (
            float(values["train_fraction"])
            + float(values["validation_fraction"])
            + float(values["test_fraction"])
        )
        if abs(fractions - 1.0) > 1e-6:
            raise ValueError("train, validation, and test fractions must sum to 1")
        resize = str(values.get("resize", "squash"))
        if resize not in {"squash", "letterbox"}:
            raise ValueError("resize must be squash or letterbox")
        scheduler = str(values.get("scheduler", "cosine"))
        if scheduler not in {"cosine", "none"}:
            raise ValueError("scheduler must be cosine or none")
        aug = dict(DEFAULT_AUG)
        aug.update({key: float(value) for key, value in (values.get("aug") or {}).items()})
        return cls(
            seed=int(values["seed"]),
            data_root=data_root,
            split_file=split_file,
            image_size=int(values["image_size"]),
            batch_size=int(values["batch_size"]),
            epochs=int(values["epochs"]),
            learning_rate=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
            num_workers=int(values["num_workers"]),
            train_fraction=float(values["train_fraction"]),
            validation_fraction=float(values["validation_fraction"]),
            test_fraction=float(values["test_fraction"]),
            device=str(values["device"]),
            threshold=float(values["threshold"]),
            checkpoint_dir=checkpoint_dir,
            resize=resize,
            scheduler=scheduler,
            tta=bool(values.get("tta", False)),
            pretrained=bool(values.get("pretrained", True)),
            distractor_probability=float(values.get("distractor_probability", 0.3)),
            aug=aug,
        )
