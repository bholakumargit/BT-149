"""Train grouped-split PyTorch MLPs for detection and malware-family prediction."""

from __future__ import annotations

import argparse
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
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from malware_classifier.config import MODEL_DIR, RAW_DATA_DIR  # noqa: E402
from malware_classifier.data import (  # noqa: E402
    build_sample_groups,
    get_feature_columns,
    grouped_train_validation_test_indices,
)
from malware_classifier.model import DROPOUT, HIDDEN_SIZES, MalwareMLP  # noqa: E402

DATASET_PATH = RAW_DATA_DIR / "CIC-MalMem-2022.csv"
SEED = 42
MAX_EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_table(path: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {path}. Run scripts/import_dataset.py first."
        )

    frame = pd.read_csv(path)
    feature_names = get_feature_columns(list(frame.columns))
    numeric = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric)
    if invalid.to_numpy().any():
        bad_columns = list(numeric.columns[invalid.any(axis=0)])
        raise ValueError(
            "Feature columns contain missing, non-numeric, or infinite values: "
            + ", ".join(bad_columns)
        )
    return frame, feature_names, numeric.to_numpy(dtype=np.float32)


def create_mlp(input_size: int, output_size: int) -> MalwareMLP:
    return MalwareMLP(input_size, output_size)


def train_workflow(
    frame: pd.DataFrame,
    all_features: np.ndarray,
    feature_names: list[str],
    target_column: str,
    workflow_name: str,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(SEED)
    if target_column == "Class":
        selected = frame.reset_index(drop=True)
        selected_features = all_features
    else:
        malware_mask = frame["Class"].eq("Malware").to_numpy()
        selected = frame.loc[malware_mask].reset_index(drop=True)
        selected_features = all_features[malware_mask]

    targets = selected[target_column].astype(str).to_numpy()
    if target_column == "Category" and np.any(targets == "Benign"):
        raise ValueError("Family classification must use malware rows only.")

    groups = build_sample_groups(selected)
    train_indices, validation_indices, test_indices = (
        grouped_train_validation_test_indices(targets, groups, random_state=SEED)
    )

    encoder = LabelEncoder()
    encoded_targets = encoder.fit_transform(targets)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(selected_features[train_indices]).astype(
        np.float32
    )
    validation_features = scaler.transform(
        selected_features[validation_indices]
    ).astype(np.float32)
    test_features = scaler.transform(selected_features[test_indices]).astype(np.float32)

    train_labels = encoded_targets[train_indices]
    validation_labels = encoded_targets[validation_indices]
    test_labels = encoded_targets[test_indices]
    class_counts = np.bincount(train_labels, minlength=len(encoder.classes_))
    class_weights = len(train_labels) / (
        len(encoder.classes_) * np.maximum(class_counts, 1)
    )

    train_data = TensorDataset(
        torch.from_numpy(train_features), torch.from_numpy(train_labels.astype(np.int64))
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    validation_x = torch.from_numpy(validation_features).to(device)
    validation_y = torch.from_numpy(validation_labels.astype(np.int64)).to(device)
    test_x = torch.from_numpy(test_features).to(device)

    model = create_mlp(len(feature_names), len(encoder.classes_)).to(device)
    loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
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

        mean_train_loss = total_train_loss / len(train_data)
        epochs_trained = epoch
        print(
            f"{workflow_name} epoch {epoch:02d}: "
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
                print(f"{workflow_name}: early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError(f"{workflow_name} training did not produce a model.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(test_x), dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    class_names = [str(label) for label in encoder.classes_]
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
    if target_column == "Class" and "Malware" in class_names:
        malware_index = class_names.index("Malware")
        metrics["roc_auc"] = float(
            roc_auc_score(test_labels, probabilities[:, malware_index])
        )

    checkpoint = {
        "workflow": workflow_name,
        "target_column": target_column,
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
            "max_epochs": MAX_EPOCHS,
            "epochs_trained": epochs_trained,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "best_validation_loss": best_validation_loss,
            "device": str(device),
        },
        "split_sizes": {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "test_metrics": metrics,
    }
    checkpoint_path = MODEL_DIR / f"{workflow_name}_model.pt"
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)

    print(
        f"{workflow_name} held-out test: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['f1_macro']:.4f}"
    )
    return {
        "target_column": target_column,
        "classes": class_names,
        "class_counts": dict(sorted(Counter(targets).items())),
        "split_sizes": checkpoint["split_sizes"],
        "metrics": metrics,
        "epochs_trained": epochs_trained,
        "model_file": checkpoint_path.name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CIC-MalMem detection and malware-family MLPs."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=(
            "Feature CSV with Class, Category, and Filename columns. "
            "Defaults to data/raw/CIC-MalMem-2022.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset
    seed_everything(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    frame, feature_names, features = load_feature_table(dataset_path)
    print(f"Loaded {len(frame):,} rows and {len(feature_names)} numeric features.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path.name,
        "dataset_rows": len(frame),
        "feature_count": len(feature_names),
        "seed": SEED,
        "split_strategy": (
            "StratifiedGroupKFold; malware dump-number suffixes are removed so "
            "all rows sharing a sample hash stay together; "
            "benign rows split independently because their source filename is "
            "the shared placeholder 'Benign'."
        ),
        "models": {
            "detection": train_workflow(
                frame, features, feature_names, "Class", "detection", device
            ),
            "classification": train_workflow(
                frame, features, feature_names, "Category", "classification", device
            ),
        },
    }
    metrics_path = MODEL_DIR / "training_metrics.json"
    temporary_metrics_path = metrics_path.with_suffix(".json.tmp")
    temporary_metrics_path.write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary_metrics_path, metrics_path)
    print(f"Saved checkpoints and metrics under {MODEL_DIR}")


if __name__ == "__main__":
    main()
