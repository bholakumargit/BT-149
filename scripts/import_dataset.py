"""Download CIC-MalMem-2022 from Hugging Face into the local raw-data folder."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "bvk/CIC-MalMem-2022"
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_DIR / "data" / "raw" / "CIC-MalMem-2022.csv"
REQUIRED_COLUMNS = {"Class", "Category", "Filename"}


def import_dataset() -> Path:
    """Download the dataset split and save its feature table for local experiments."""
    dataset = load_dataset(DATASET_ID, split="train")
    missing_columns = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{DATASET_ID} is missing required columns: {missing}")
    if len(dataset) == 0:
        raise ValueError(f"{DATASET_ID} returned an empty training split.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(str(OUTPUT_PATH), index=False)

    class_counts = Counter(dataset["Class"])
    category_counts = Counter(dataset["Category"])
    print(f"Imported {len(dataset):,} records to {OUTPUT_PATH}")
    print(f"Class counts: {dict(sorted(class_counts.items()))}")
    print(f"Category counts: {dict(sorted(category_counts.items()))}")
    print("Filename is retained for grouping samples, not as a model feature.")
    return OUTPUT_PATH


if __name__ == "__main__":
    import_dataset()
