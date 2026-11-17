"""Train the PDF malware detector from CIC-Evasive-PDFMal2022 static features.

The training table stores pdfid-style counts that are not always clean numbers
(for example ``1(1)``, ``unclear`` or ``-1``). This script normalises them with
the same meaning the runtime extractor uses, trains a LightGBM classifier and
writes the deployment metadata consumed by ``pdf_inference``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from malware_classifier.config import MODEL_DIR, RAW_DATA_DIR  # noqa: E402
from malware_classifier.pdf_features import (  # noqa: E402
    PDF_FEATURE_NAMES,
    PDF_MODEL_FEATURE_NAMES,
)

DEFAULT_DATASET = RAW_DATA_DIR / "PDFMalware2022.csv"
DEFAULT_METRICS = MODEL_DIR / "pdf_training_metrics.json"
DEFAULT_MODEL_METADATA = MODEL_DIR / "pdf_model_metadata.json"
MODEL_FILE = "pdf_lightgbm.txt"
CLASS_NAMES = ["Benign", "Malware"]
POSITIVE_LABEL = "Malware"
SEED = 42
THRESHOLD = 0.5
COUNT_PATTERN = re.compile(r"^(-?\d+)\s*\(\d+\)$")
HEADER_PATTERN = re.compile(r"%PDF-(\d)\.(\d)")


def clean_column(column: str, series: pd.Series) -> pd.Series:
    """Normalise one raw pdfid feature column to floats, NaN for unknown."""
    stripped = series.astype(str).str.strip()

    if column == "text":
        return stripped.map(
            lambda value: {"Yes": 1.0, "No": 0.0}.get(value, np.nan)
            if isinstance(value, str)
            else np.nan
        ).astype(float)

    if column == "header":
        def header(value: object) -> float:
            if not isinstance(value, str):
                return np.nan
            match = HEADER_PATTERN.search(value)
            return float(f"{match.group(1)}.{match.group(2)}") if match else np.nan

        return stripped.map(header).astype(float)

    def numeric(value: object) -> float:
        if not isinstance(value, str):
            return np.nan
        match = COUNT_PATTERN.match(value)
        if match:
            value = match.group(1)
        try:
            number = float(value)
        except ValueError:
            return np.nan
        return np.nan if number < 0 else number

    return stripped.map(numeric).astype(float)


def load_pdf_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"PDF dataset not found: {path}. Download PDFMalware2022.csv first."
        )
    frame = pd.read_csv(path)
    missing = {"Class", *PDF_FEATURE_NAMES}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["Class"]).copy()
    labels = (frame.pop("Class").astype(str).str.strip() == "Malicious").astype(int)

    features = pd.DataFrame(
        {
            column: clean_column(column, frame[column])
            for column in PDF_FEATURE_NAMES
        }
    )
    # Drop rows where every model feature is unknown.
    model_features = features.loc[:, PDF_MODEL_FEATURE_NAMES]
    usable = model_features.notna().any(axis=1)
    return model_features.loc[usable].reset_index(drop=True), labels.loc[usable].to_numpy()


def binary_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(
            precision_score(targets, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "threshold": threshold,
    }


def top_gain_features(
    feature_names: list[str], importances: np.ndarray, limit: int = 15
) -> list[dict[str, Any]]:
    order = np.argsort(importances)[::-1][:limit]
    return [
        {"feature": feature_names[index], "importance": float(importances[index])}
        for index in order
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model-metadata", type=Path, default=DEFAULT_MODEL_METADATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import lightgbm as lgb

    features, targets = load_pdf_dataset(args.dataset)
    feature_names = list(PDF_MODEL_FEATURE_NAMES)
    print(
        f"Loaded {len(features):,} rows, {len(feature_names)} features; "
        f"benign={int((targets == 0).sum())}, malicious={int((targets == 1).sum())}"
    )

    train_x, rest_x, train_y, rest_y = train_test_split(
        features, targets, test_size=0.2, random_state=SEED, stratify=targets
    )
    validation_x, test_x, validation_y, test_y = train_test_split(
        rest_x, rest_y, test_size=0.5, random_state=SEED, stratify=rest_y
    )
    split_sizes = {
        "train": int(len(train_x)),
        "validation": int(len(validation_x)),
        "test": int(len(test_x)),
    }
    print(f"Splits: {split_sizes}")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=min(4, os.cpu_count() or 1),
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        eval_set=[(validation_x, validation_y)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )

    probabilities = model.predict_proba(test_x)[:, 1]
    metrics = binary_metrics(test_y, probabilities, THRESHOLD)
    model_path = args.output_dir / MODEL_FILE
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))

    importance = model.booster_.feature_importance(importance_type="gain")
    generated_at = datetime.now(timezone.utc).isoformat()
    metrics_payload: dict[str, Any] = {
        "generated_at": generated_at,
        "dataset": args.dataset.name,
        "dataset_family": "CIC-Evasive-PDFMal2022",
        "rows": int(len(features)),
        "feature_count": len(feature_names),
        "classes": CLASS_NAMES,
        "class_counts": {
            "Benign": int((targets == 0).sum()),
            "Malware": int((targets == 1).sum()),
        },
        "split_sizes": split_sizes,
        "seed": SEED,
        "deployment": {
            "model": "lightgbm",
            "model_file": MODEL_FILE,
            "threshold": THRESHOLD,
        },
        "models": {
            "lightgbm": {
                "model_file": MODEL_FILE,
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
                "metrics": metrics,
                "top_gain_features": top_gain_features(feature_names, importance),
            }
        },
        "license": "license not verified",
        "split_strategy": "Stratified 80/10/10 train/validation/test split.",
        "feature_source": (
            "pdfid-style static features re-extracted from the PDF bytes at "
            "inference time; columns that the byte-level extractor cannot "
            "reproduce (metadata size, xref Length) are excluded from the model."
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics = args.metrics.with_suffix(".json.tmp")
    temporary_metrics.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_metrics.replace(args.metrics)

    model_metadata = {
        "dataset": "CIC-Evasive-PDFMal2022",
        "source_dataset": args.dataset.name,
        "deployment_model": "lightgbm",
        "model_file": MODEL_FILE,
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
        "positive_class": POSITIVE_LABEL,
        "threshold": THRESHOLD,
        "trained_at": generated_at,
        "license": "license not verified",
    }
    args.model_metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata = args.model_metadata.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(model_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_metadata.replace(args.model_metadata)

    print(
        f"Held-out: accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['f1_macro']:.4f}, roc_auc={metrics['roc_auc']:.4f}"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {args.metrics}")
    print(f"Saved model metadata: {args.model_metadata}")


if __name__ == "__main__":
    main()
