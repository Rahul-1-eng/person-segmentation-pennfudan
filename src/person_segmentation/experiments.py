from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    model: str = "unet"
    aug: str = "none"
    dice_weight: float | None = None
    distractors: bool = False
    lr: float | None = None
    zero_shot: bool = False


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "baseline": ExperimentSpec(),
    "augmented": ExperimentSpec(aug="mild"),
    "advanced": ExperimentSpec(aug="mild", dice_weight=0.5),
    "advanced_ablation": ExperimentSpec(aug="mild", dice_weight=0.0),
    "strong_aug": ExperimentSpec(aug="strong"),
    "distract": ExperimentSpec(aug="strong", distractors=True),
    "resnet34": ExperimentSpec(model="resnet34", aug="strong"),
    "resnet34_distract": ExperimentSpec(model="resnet34", aug="strong", distractors=True),
    "deeplab_ft": ExperimentSpec(model="deeplab", aug="strong", lr=0.0001),
    "deeplab_zeroshot": ExperimentSpec(model="deeplab", zero_shot=True),
}


def trainable_experiments() -> list[str]:
    return [name for name, spec in EXPERIMENTS.items() if not spec.zero_shot]
