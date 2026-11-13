import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from person_segmentation.config import ExperimentConfig
from person_segmentation.data import create_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--split-file")
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    data_root = args.data_root or str(config.data_root)
    split_file = args.split_file or str(config.split_file)
    split = create_split(
        data_root,
        split_file,
        config.seed,
        config.train_fraction,
        config.validation_fraction,
    )
    print({key: len(value) for key, value in split.items()})


if __name__ == "__main__":
    main()
