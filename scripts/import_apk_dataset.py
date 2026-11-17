"""Import and clean the Drebin-215 APK static-feature dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from malware_classifier.apk_data import (  # noqa: E402
    DEFAULT_CATEGORIES,
    DEFAULT_METADATA,
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    import_apk_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the Drebin-215 APK feature CSV."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, metadata = import_apk_dataset(
        args.source, args.output, args.metadata, args.categories
    )
    counts = metadata["counts"]
    print(f"Source: {metadata['source']['path']}")
    print(f"Source rows: {counts['source_rows']:,}")
    print(
        "Question-mark cells replaced: "
        f"{counts['question_mark_cells_replaced']:,} "
        f"({counts['question_mark_cells_by_column']})"
    )
    print(
        "Exact duplicate rows removed: "
        f"{counts['rows_removed_as_exact_duplicates']:,}"
    )
    print(
        "Conflicting feature patterns removed: "
        f"{counts['conflicting_feature_patterns_removed']:,} "
        f"({counts['rows_removed_for_conflicting_labels']:,} rows)"
    )
    print(f"Rows after cleaning: {counts['rows_after_cleaning']:,}")
    print(f"Class counts: {counts['class_counts']}")
    print(f"Feature count: {counts['feature_count']:,}")
    print(f"Wrote: {output}")
    print(f"Metadata: {args.metadata}")
    print("License: license not verified")


if __name__ == "__main__":
    main()
