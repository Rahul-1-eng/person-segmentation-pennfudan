import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data")
    parser.add_argument("--archive")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive) if args.archive else output / "PennFudanPed.zip"
    if not archive.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a readable zip archive: {archive}")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(output)
    extracted = output / "PennFudanPed"
    if not extracted.exists():
        raise FileNotFoundError(f"Expected extracted dataset at {extracted}")
    print(f"Dataset ready at {extracted}")
    if args.archive is None:
        archive.unlink()


if __name__ == "__main__":
    main()
