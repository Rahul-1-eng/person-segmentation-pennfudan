import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from .distract import random_distractor

FILL_COLOR = (124, 116, 104)


def source_of(name: str) -> str:
    return "fudan" if name.lower().startswith("fudan") else "penn"


def _affine_coefficients(size, scale, angle, shift_x, shift_y):
    cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
    center = size / 2.0
    a, b, d, e = cos, sin, -sin, cos
    c = center - (a * (center + shift_x) + b * (center + shift_y))
    f = center - (d * (center + shift_x) + e * (center + shift_y))
    return (a, b, c, d, e, f)


def image_mask_pairs(data_root: str | Path) -> list[tuple[str, str]]:
    root = Path(data_root)
    image_dir = root / "PNGImages"
    mask_dir = root / "PedMasks"
    if not mask_dir.exists():
        mask_dir = root / "PNGMasks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected PNGImages and PedMasks or PNGMasks under {root}"
        )
    pairs = []
    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = _find_mask_path(mask_dir, image_path.name)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        pairs.append((str(image_path), str(mask_path)))
    if not pairs:
        raise ValueError(f"No PNG image pairs found under {root}")
    return pairs


def _find_mask_path(mask_dir: Path, image_name: str) -> Path:
    image_path = Path(image_name)
    candidates = (
        mask_dir / image_name,
        mask_dir / f"{image_path.stem}_mask{image_path.suffix}",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def create_split(
    data_root: str | Path,
    output_path: str | Path,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[str]]:
    pairs = image_mask_pairs(data_root)
    names = [Path(image).name for image, _ in pairs]
    generator = random.Random(seed)
    generator.shuffle(names)
    train_end = round(len(names) * train_fraction)
    validation_end = train_end + round(len(names) * validation_fraction)
    split = {
        "train": sorted(names[:train_end]),
        "validation": sorted(names[train_end:validation_end]),
        "test": sorted(names[validation_end:]),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


class PennFudanDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        data_root: str | Path,
        names: Iterable[str],
        image_size: int,
        seed: int,
        aug: str = "none",
        resize: str = "squash",
        distractor_probability: float = 0.0,
        aug_config: dict[str, float] | None = None,
    ) -> None:
        root = Path(data_root)
        self.image_dir = root / "PNGImages"
        self.mask_dir = root / "PedMasks"
        if not self.mask_dir.exists():
            self.mask_dir = root / "PNGMasks"
        self.names = list(names)
        self.image_size = image_size
        self.seed = seed
        self.aug = aug
        self.resize = resize
        self.distractor_probability = distractor_probability
        self.aug_config = aug_config or {}
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.names)

    def load_pair(self, name: str) -> tuple[Image.Image, Image.Image]:
        image = Image.open(self.image_dir / name).convert("RGB")
        mask = Image.open(_find_mask_path(self.mask_dir, name)).convert("L")
        return self._resize(image, mask)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        name = self.names[index]
        image, mask = self.load_pair(name)
        generator = random.Random(self.seed + self.epoch * 100003 + index)
        if self.aug == "mild":
            image, mask = self._augment_mild(image, mask, generator)
        elif self.aug == "strong":
            image, mask = self._augment_strong(image, mask, generator)
        if self.distractor_probability > 0 and generator.random() < self.distractor_probability:
            image, _ = random_distractor(image, mask, generator)
        image_tensor, mask_tensor = to_tensors(image, mask)
        return image_tensor, mask_tensor, name

    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        size = self.image_size
        if self.resize == "squash":
            return (
                image.resize((size, size), Image.Resampling.BILINEAR),
                mask.resize((size, size), Image.Resampling.NEAREST),
            )
        scale = size / max(image.size)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        mask = mask.resize(new_size, Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (size, size), FILL_COLOR)
        mask_canvas = Image.new("L", (size, size), 0)
        offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
        canvas.paste(image, offset)
        mask_canvas.paste(mask, offset)
        return canvas, mask_canvas

    def _augment_mild(self, image, mask, generator):
        if generator.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if generator.random() < 0.5:
            scale = generator.uniform(0.9, 1.0)
            crop_size = max(1, round(self.image_size * scale))
            left = generator.randint(0, self.image_size - crop_size)
            top = generator.randint(0, self.image_size - crop_size)
            box = (left, top, left + crop_size, top + crop_size)
            image = image.crop(box).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            mask = mask.crop(box).resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )
        if generator.random() < 0.2:
            image = ImageEnhance.Brightness(image).enhance(generator.uniform(0.95, 1.05))
        return image, mask

    def _augment_strong(self, image, mask, generator):
        cfg = self.aug_config
        size = self.image_size
        if generator.random() < cfg["flip"]:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if generator.random() < cfg["affine"]:
            scale = generator.uniform(cfg["scale_min"], cfg["scale_max"])
            angle = math.radians(generator.uniform(-cfg["rotate"], cfg["rotate"]))
            shift_x = generator.uniform(-cfg["translate"], cfg["translate"]) * size
            shift_y = generator.uniform(-cfg["translate"], cfg["translate"]) * size
            coefficients = _affine_coefficients(size, scale, angle, shift_x, shift_y)
            image = image.transform(
                (size, size), Image.Transform.AFFINE, coefficients,
                Image.Resampling.BILINEAR, fillcolor=FILL_COLOR,
            )
            mask = mask.transform(
                (size, size), Image.Transform.AFFINE, coefficients,
                Image.Resampling.NEAREST, fillcolor=0,
            )
        if generator.random() < cfg["color"]:
            strength = cfg["color_strength"]
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = enhancer(image).enhance(generator.uniform(1 - strength, 1 + strength))
        if generator.random() < cfg["blur"]:
            image = image.filter(ImageFilter.GaussianBlur(generator.uniform(0.3, 1.5)))
        if generator.random() < cfg["noise"]:
            sigma = generator.uniform(2.0, 8.0)
            noise = np.random.default_rng(generator.randint(0, 2**31 - 1)).normal(0, sigma, (size, size, 3))
            array = np.asarray(image, dtype=np.float32) + noise
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        return image, mask


def to_tensors(image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
    return (
        torch.from_numpy(image_array).permute(2, 0, 1),
        torch.from_numpy(mask_array).unsqueeze(0),
    )


def loaders(
    data_root: str | Path,
    split: dict[str, list[str]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    aug: str = "none",
    resize: str = "squash",
    distractor_probability: float = 0.0,
    aug_config: dict[str, float] | None = None,
) -> dict[str, DataLoader]:
    datasets = {
        "train": PennFudanDataset(
            data_root, split["train"], image_size, seed, aug, resize,
            distractor_probability, aug_config,
        ),
        "validation": PennFudanDataset(data_root, split["validation"], image_size, seed, resize=resize),
        "test": PennFudanDataset(data_root, split["test"], image_size, seed, resize=resize),
    }
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }
