import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from person_segmentation.data import image_mask_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/PennFudanPed")
    parser.add_argument("--output", default="artifacts/dataset_audit.csv")
    args = parser.parse_args()
    rows = []
    for image_name, mask_name in image_mask_pairs(args.data_root):
        image = Image.open(image_name).convert("RGB")
        mask = np.asarray(Image.open(mask_name).convert("L"))
        rows.append(
            {
                "image": Path(image_name).name,
                "width": image.width,
                "height": image.height,
                "mask_foreground_pixels": int((mask > 0).sum()),
                "mask_foreground_fraction": float((mask > 0).mean()),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(rows)} image-mask pairs")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
