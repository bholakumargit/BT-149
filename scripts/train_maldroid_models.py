"""Train separate MalDroid-2020 MLPs for syscall and syscall/Binder features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from malware_classifier.config import (  # noqa: E402
    MODEL_DIR,
)
from malware_classifier.data import grouped_train_validation_test_indices  # noqa: E402
from malware_classifier.model import DROPOUT, HIDDEN_SIZES, MalwareMLP  # noqa: E402

SEED = 42
MAX_EPOCHS = 40
PATIENCE = 7
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

MODEL_CONFIGS = {
    "syscalls": "maldroid_syscalls_model.pt",
    "syscalls_binders": "maldroid_syscalls_binders_model.pt",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonicalize_targets(values: pd.Series) -> np.ndarray:
    targets = values.astype("string").str.strip()
    if targets.isna().any() or targets.eq("").any():
        raise ValueError("Class labels must not be missing or empty.")

    numeric = pd.to_numeric(targets, errors="coerce")
    if numeric.notna().all():
        if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            raise ValueError("Class labels must be finite values.")
        labels = [
            str(int(value)) if float(value).is_integer() else format(float(value), ".15g")
            for value in numeric
        ]
    else:
        labels = targets.astype(str).tolist()
    if len(set(labels)) < 2:
        raise ValueError("At least two distinct Class labels are required.")
    return np.asarray(labels, dtype=str)


def load_feature_table(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read a labeled feature CSV and reject malformed/non-numeric inputs."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training data not found: {dataset_path}")

    frame = pd.read_csv(dataset_path, low_memory=False)
    if "Class" not in frame.columns:
        raise ValueError(f"{dataset_path.name} must contain a 'Class' target column.")
    if frame.columns.duplicated().any():
        raise ValueError(f"{dataset_path.name} contains duplicate column names.")

    feature_names = [str(column) for column in frame.columns if column != "Class"]
    if not feature_names:
        raise ValueError(f"{dataset_path.name} contains no feature columns.")

    numeric = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric)
    if invalid.to_numpy().any():
        bad_columns = list(numeric.columns[invalid.any(axis=0)])
        raise ValueError(
            "Feature columns contain missing, non-numeric, or infinite values: "
            + ", ".join(bad_columns[:10])
        )

    features = numeric.to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature values exceed the supported float32 range.")
    targets = canonicalize_targets(frame["Class"])
    if len(targets) != len(features):
        raise ValueError("Feature rows and Class labels have different lengths.")
    return feature_names, features, targets


def feature_group_ids(features: np.ndarray) -> np.ndarray:
    """Give identical model inputs one split group to prevent duplicate leakage."""
    return np.asarray(
        [hashlib.sha256(row.tobytes()).hexdigest() for row in features],
        dtype=str,
    )


def remove_conflicting_duplicate_rows(
    features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Exclude identical feature vectors assigned different labels."""
    groups = feature_group_ids(features)
    labels_by_group: dict[str, set[str]] = {}
    for group, target in zip(groups, targets, strict=True):
        labels_by_group.setdefault(group, set()).add(str(target))

    conflicting_groups = {
        group for group, labels in labels_by_group.items() if len(labels) > 1
    }
    keep = np.asarray([group not in conflicting_groups for group in groups])
    removed_rows = int((~keep).sum())
    if removed_rows:
        features = features[keep]
        targets = targets[keep]
        groups = groups[keep]
    if len(np.unique(targets)) < 2:
        raise ValueError("Removing conflicting duplicate rows left fewer than two classes.")
    return features, targets, groups, len(conflicting_groups), removed_rows


def train_model(
    dataset_path: Path,
    model_key: str,
    output_dir: Path,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
) -> dict[str, Any]:
    seed_everything(SEED)
    feature_names, features, targets = load_feature_table(dataset_path)
    rows_before_filter = len(targets)
    features, targets, groups, conflict_groups, excluded_rows = (
        remove_conflicting_duplicate_rows(features, targets)
    )

    train_indices, validation_indices, test_indices = (
        grouped_train_validation_test_indices(targets, groups, random_state=SEED)
    )
    encoder = LabelEncoder()
    encoded_targets = encoder.fit_transform(targets)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_indices]).astype(np.float32)
    validation_features = scaler.transform(features[validation_indices]).astype(
        np.float32
    )
    test_features = scaler.transform(features[test_indices]).astype(np.float32)

    train_labels = encoded_targets[train_indices]
    validation_labels = encoded_targets[validation_indices]
    test_labels = encoded_targets[test_indices]
    class_counts = np.bincount(train_labels, minlength=len(encoder.classes_))
    class_weights = len(train_labels) / (
        len(encoder.classes_) * np.maximum(class_counts, 1)
    )

    train_data = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_labels.astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    validation_x = torch.from_numpy(validation_features).to(device)
    validation_y = torch.from_numpy(validation_labels.astype(np.int64)).to(device)
    test_x = torch.from_numpy(test_features).to(device)

    model = MalwareMLP(len(feature_names), len(encoder.classes_)).to(device)
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(batch_features)

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(validation_x), validation_y).item()

        epochs_trained = epoch
        mean_train_loss = total_train_loss / len(train_data)
        print(
            f"{model_key} epoch {epoch:02d}: "
            f"train_loss={mean_train_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"{model_key}: early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError(f"{model_key} training did not produce a model.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(test_x), dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    class_names = [str(label) for label in encoder.classes_]
    class_counts_summary = dict(
        sorted(Counter(targets.tolist()).items())
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision_macro": float(
            precision_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(test_labels, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=np.arange(len(class_names))
        ).tolist(),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    checkpoint = {
        "workflow": f"maldroid_{model_key}",
        "dataset": dataset_path.name,
        "target_column": "Class",
        "feature_names": feature_names,
        "class_names": class_names,
        "model_state_dict": best_state,
        "normalization_mean": scaler.mean_.tolist(),
        "normalization_scale": scaler.scale_.tolist(),
        "architecture": {
            "type": "MLP",
            "hidden_sizes": list(HIDDEN_SIZES),
            "activation": "ReLU",
            "dropout": DROPOUT,
        },
        "training_config": {
            "seed": SEED,
            "max_epochs": max_epochs,
            "epochs_trained": epochs_trained,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "best_validation_loss": best_validation_loss,
            "device": str(device),
            "duplicate_feature_groups_used_for_splitting": True,
        },
        "data_summary": {
            "rows_before_conflict_filter": rows_before_filter,
            "rows_excluded_for_conflicting_duplicate_labels": excluded_rows,
            "conflicting_duplicate_feature_groups": conflict_groups,
            "rows_used": len(targets),
            "feature_count": len(feature_names),
            "class_counts": class_counts_summary,
            "split_sizes": {
                "train": int(len(train_indices)),
                "validation": int(len(validation_indices)),
                "test": int(len(test_indices)),
            },
        },
        "test_metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / MODEL_CONFIGS[model_key]
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)

    print(
        f"{model_key} held-out test: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['f1_macro']:.4f}"
    )
    return {
        "dataset": dataset_path.name,
        "feature_count": len(feature_names),
        "class_names": class_names,
        "class_counts": class_counts_summary,
        "rows_excluded_for_conflicting_duplicate_labels": excluded_rows,
        "split_sizes": checkpoint["data_summary"]["split_sizes"],
        "metrics": metrics,
        "epochs_trained": epochs_trained,
        "model_file": checkpoint_path.name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train separate MalDroid-2020 syscall feature classifiers."
    )
    parser.add_argument("--syscalls", type=Path, required=True)
    parser.add_argument("--syscalls-binders", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        "syscalls": args.syscalls,
        "syscalls_binders": args.syscalls_binders,
    }
    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_family": "MalDroid-2020",
        "seed": SEED,
        "split_strategy": (
            "Stratified 80/10/10 splits; identical feature vectors are grouped "
            "to prevent cross-split leakage. Exact feature vectors assigned "
            "conflicting labels are excluded."
        ),
        "device": str(device),
        "models": {},
    }
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    for model_key, dataset_path in datasets.items():
        print(f"\nTraining {model_key} from {dataset_path}")
        results["models"][model_key] = train_model(
            dataset_path, model_key, args.output_dir, device, args.epochs
        )

    metrics_path = args.output_dir / "maldroid_training_metrics.json"
    temporary_metrics_path = metrics_path.with_suffix(".json.tmp")
    temporary_metrics_path.write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary_metrics_path, metrics_path)
    print(f"\nSaved MalDroid checkpoints and metrics under {args.output_dir}")


if __name__ == "__main__":
    main()
