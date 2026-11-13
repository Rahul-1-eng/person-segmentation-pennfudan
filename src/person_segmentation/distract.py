import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

KINDS = (
    "floor_reflection",
    "mirror_panel",
    "cast_shadow",
    "detached_shadow",
    "animal_shadow",
    "darken",
)
SHADOW_COLOR = np.array([30.0, 30.0, 35.0], dtype=np.float32)


def _ranges(hard: bool) -> dict[str, tuple[float, float]]:
    if hard:
        return {"opacity": (0.55, 0.85), "blur": (3.0, 6.0), "tint": (0.35, 0.6)}
    return {"opacity": (0.3, 0.6), "blur": (0.5, 2.5), "tint": (0.15, 0.35)}


def _bbox(person: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(person)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _blur_alpha(alpha: np.ndarray, radius: float) -> np.ndarray:
    layer = Image.fromarray((np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8))
    layer = layer.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(layer, dtype=np.float32) / 255.0


def _compose(
    image: Image.Image,
    person: np.ndarray,
    content: np.ndarray,
    alpha: np.ndarray,
    blur: float,
) -> tuple[Image.Image, np.ndarray]:
    alpha = _blur_alpha(alpha, blur) if blur > 0 else alpha
    alpha = alpha * (~person)
    base = np.asarray(image, dtype=np.float32)
    out = base * (1.0 - alpha[..., None]) + content * alpha[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), alpha > 0.1


def _shadow_content(shape: tuple[int, int]) -> np.ndarray:
    return np.broadcast_to(SHADOW_COLOR, (*shape, 3))


def _warp_mask(person: np.ndarray, coefficients: tuple[float, ...]) -> np.ndarray:
    layer = Image.fromarray(person.astype(np.uint8) * 255)
    warped = layer.transform(
        layer.size, Image.Transform.AFFINE, coefficients, Image.Resampling.BILINEAR
    )
    return np.asarray(warped, dtype=np.float32) / 255.0


def floor_reflection(image, person, rng, ranges):
    box = _bbox(person)
    width, height = image.size
    x0, y0, x1, y1 = box
    person_height = y1 - y0
    gap = rng.randint(0, max(1, person_height // 30))
    top = y1 + gap
    room = height - top
    if room < max(4, int(0.15 * person_height)):
        return image, np.zeros(person.shape, dtype=bool)
    extent = min(person_height, room)
    pixels = np.asarray(image, dtype=np.float32)
    crop = pixels[y0:y1, x0:x1][::-1]
    shape = person[y0:y1, x0:x1][::-1].astype(np.float32)
    fade = np.linspace(1.0, 0.0, person_height, dtype=np.float32)[:, None]
    opacity = rng.uniform(*ranges["opacity"])
    alpha = np.zeros(person.shape, dtype=np.float32)
    content = np.zeros((height, width, 3), dtype=np.float32)
    alpha[top : top + extent, x0:x1] = (shape * fade * opacity)[:extent]
    content[top : top + extent, x0:x1] = crop[:extent]
    return _compose(image, person, content, alpha, rng.uniform(*ranges["blur"]))


def mirror_panel(image, person, rng, ranges):
    width, height = image.size
    x0, y0, x1, y1 = _bbox(person)
    panel_width = int(rng.uniform(0.22, 0.38) * width)
    panel_height = int(rng.uniform(0.5, 0.8) * height)
    best, best_overlap = None, 2.0
    for _ in range(12):
        left = rng.randint(0, max(0, width - panel_width))
        top = rng.randint(0, max(0, height - panel_height))
        overlap = float(person[top : top + panel_height, left : left + panel_width].mean())
        if overlap < best_overlap:
            best, best_overlap = (left, top), overlap
    left, top = best
    pixels = np.asarray(image, dtype=np.float32).copy()
    free = ~person
    rect = (slice(top, top + panel_height), slice(left, left + panel_width))
    tint = np.array([200.0, 210.0, 220.0], dtype=np.float32)
    strength = rng.uniform(*ranges["tint"])
    tinted = pixels[rect] * (1.0 - strength) + tint * strength
    pixels[rect] = np.where(free[rect][..., None], tinted, pixels[rect])
    frame = np.zeros(person.shape, dtype=bool)
    border = 3
    frame[top : top + panel_height, left : left + panel_width] = True
    frame[top + border : top + panel_height - border, left + border : left + panel_width - border] = False
    pixels = np.where((frame & free)[..., None], np.array([60.0, 60.0, 60.0], dtype=np.float32), pixels)
    crop = np.asarray(image, dtype=np.float32)[y0:y1, x0:x1][:, ::-1]
    shape = person[y0:y1, x0:x1][:, ::-1].astype(np.float32)
    box_height, box_width = y1 - y0, x1 - x0
    scale = min(0.8 * panel_width / box_width, 0.9 * panel_height / box_height)
    new_width = max(2, int(box_width * scale))
    new_height = max(2, int(box_height * scale))
    crop_image = Image.fromarray(np.ascontiguousarray(crop).astype(np.uint8)).resize(
        (new_width, new_height), Image.Resampling.BILINEAR
    )
    shape_image = Image.fromarray((shape * 255).astype(np.uint8)).resize(
        (new_width, new_height), Image.Resampling.BILINEAR
    )
    paste_left = left + (panel_width - new_width) // 2
    paste_top = top + panel_height - new_height - border - 2
    alpha = np.zeros(person.shape, dtype=np.float32)
    content = np.zeros((height, width, 3), dtype=np.float32)
    opacity = min(rng.uniform(*ranges["opacity"]) + 0.2, 0.95)
    y_end = min(height, paste_top + new_height)
    x_end = min(width, paste_left + new_width)
    ys, xs = max(0, paste_top), max(0, paste_left)
    if y_end > ys and x_end > xs:
        alpha[ys:y_end, xs:x_end] = (
            np.asarray(shape_image, dtype=np.float32)[ys - paste_top : y_end - paste_top, xs - paste_left : x_end - paste_left]
            / 255.0
            * opacity
        )
        content[ys:y_end, xs:x_end] = np.asarray(crop_image, dtype=np.float32)[
            ys - paste_top : y_end - paste_top, xs - paste_left : x_end - paste_left
        ]
    base = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return _compose(base, person, content, alpha, 0.5)


def cast_shadow(image, person, rng, ranges):
    x0, y0, x1, y1 = _bbox(person)
    flatten = rng.uniform(0.15, 0.35)
    shear = rng.uniform(0.2, 0.7) * rng.choice((-1.0, 1.0))
    coefficients = (
        1.0,
        -shear / flatten,
        shear * y1 / flatten,
        0.0,
        -1.0 / flatten,
        y1 * (1.0 + 1.0 / flatten),
    )
    alpha = _warp_mask(person, coefficients) * rng.uniform(*ranges["opacity"])
    return _compose(
        image, person, _shadow_content(person.shape), alpha, rng.uniform(*ranges["blur"])
    )


def detached_shadow(image, person, rng, ranges):
    width, _ = image.size
    x0, y0, x1, y1 = _bbox(person)
    person_width = x1 - x0
    room_left, room_right = x0, width - x1
    direction = 1.0 if room_right >= room_left else -1.0
    offset = direction * person_width * (1.05 + rng.uniform(0.0, 0.8))
    offset = float(np.clip(offset, -x0, width - x1))
    coefficients = (1.0, 0.0, -offset, 0.0, 1.0, 0.0)
    alpha = _warp_mask(person, coefficients) * rng.uniform(*ranges["opacity"])
    return _compose(
        image, person, _shadow_content(person.shape), alpha, rng.uniform(*ranges["blur"])
    )


def _animal(kind: str) -> Image.Image:
    canvas = Image.new("L", (100, 100), 0)
    draw = ImageDraw.Draw(canvas)
    if kind == "rabbit":
        draw.ellipse((30, 45, 75, 85), fill=255)
        draw.ellipse((38, 2, 50, 50), fill=255)
        draw.ellipse((55, 6, 67, 52), fill=255)
        draw.ellipse((68, 55, 92, 72), fill=255)
    elif kind == "bird":
        draw.polygon([(50, 55), (3, 22), (25, 62)], fill=255)
        draw.polygon([(50, 55), (97, 22), (75, 62)], fill=255)
        draw.ellipse((40, 45, 60, 78), fill=255)
        draw.polygon([(50, 76), (40, 97), (60, 97)], fill=255)
    else:
        draw.ellipse((15, 40, 70, 82), fill=255)
        draw.polygon([(60, 50), (97, 56), (97, 72), (60, 78)], fill=255)
        draw.polygon([(28, 44), (20, 8), (46, 40)], fill=255)
    return canvas


def animal_shadow(image, person, rng, ranges):
    width, height = image.size
    size = int(rng.uniform(0.15, 0.35) * height)
    shape = _animal(rng.choice(("rabbit", "bird", "dog")))
    if rng.random() < 0.5:
        shape = shape.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    shape = shape.rotate(rng.uniform(-20, 20), resample=Image.Resampling.BILINEAR)
    shape = shape.resize((size, size), Image.Resampling.BILINEAR)
    best, best_overlap = None, 2.0
    for _ in range(12):
        left = rng.randint(0, max(0, width - size))
        top = rng.randint(0, max(0, height - size))
        overlap = float(person[top : top + size, left : left + size].mean())
        if overlap < best_overlap:
            best, best_overlap = (left, top), overlap
    left, top = best
    alpha = np.zeros(person.shape, dtype=np.float32)
    patch = np.asarray(shape, dtype=np.float32) / 255.0
    y_end, x_end = min(height, top + size), min(width, left + size)
    alpha[top:y_end, left:x_end] = patch[: y_end - top, : x_end - left] * rng.uniform(*ranges["opacity"])
    return _compose(
        image, person, _shadow_content(person.shape), alpha, rng.uniform(*ranges["blur"])
    )


def darken(image, person, rng, ranges):
    width, height = image.size
    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    start_x = rng.randint(-width // 2, width // 2)
    end_x = rng.randint(width // 2, width + width // 2)
    top = rng.randint(-height // 3, height // 3)
    bottom = rng.randint(height // 2, height + height // 3)
    points = [
        (start_x, top),
        (end_x, top + rng.randint(-height // 6, height // 6)),
        (end_x, bottom),
        (start_x, bottom + rng.randint(-height // 6, height // 6)),
    ]
    draw.polygon(points, fill=255)
    alpha = np.asarray(layer, dtype=np.float32) / 255.0 * rng.uniform(*ranges["opacity"])
    return _compose(image, person, _shadow_content(person.shape), alpha, max(width // 32, 2))


GENERATORS = {
    "floor_reflection": floor_reflection,
    "mirror_panel": mirror_panel,
    "cast_shadow": cast_shadow,
    "detached_shadow": detached_shadow,
    "animal_shadow": animal_shadow,
    "darken": darken,
}


def add_distractor(
    image: Image.Image,
    mask: Image.Image,
    kind: str,
    rng: random.Random,
    hard: bool = False,
) -> tuple[Image.Image, np.ndarray]:
    person = np.asarray(mask) > 0
    empty = np.zeros(person.shape, dtype=bool)
    if not person.any():
        return image, empty
    return GENERATORS[kind](image, person, rng, _ranges(hard))


def random_distractor(
    image: Image.Image, mask: Image.Image, rng: random.Random
) -> tuple[Image.Image, np.ndarray]:
    return add_distractor(image, mask, rng.choice(KINDS), rng, hard=False)
