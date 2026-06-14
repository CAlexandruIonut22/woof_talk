#!/usr/bin/env python3
"""Download public canine vocalization datasets for model training."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

DATASETS = {
    "emotional-canines": "ArlingtonCL2/BarkopediaDogEmotionClassification_Data",
    "dogspeak": "ArlingtonCL2/DogSpeak_Dataset",
    "breed": "ArlingtonCL2/Barkopedia_DOG_BREED_CLASSIFICATION_DATASET",
    "sex": "ArlingtonCL2/Barkopedia_Dog_Sex_Classification_Dataset",
    "individual": "ArlingtonCL2/Barkopedia_Individual_Dog_Recognition_Dataset",
}


def download(name: str, target: Path) -> None:
    from datasets import load_dataset

    dataset_id = DATASETS[name]
    print(f"Downloading {dataset_id} ...")
    target.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_id)
    for split_name, split in dataset.items():
        split.save_to_disk(str(target / split_name))
    print(f"Saved to {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download canine vocalization datasets from Hugging Face.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="emotional-canines",
        help="Which dataset to download",
    )
    args = parser.parse_args()

    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in names:
        download(name, DATA_ROOT / name)


if __name__ == "__main__":
    main()
