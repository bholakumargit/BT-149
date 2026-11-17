"""Train and compare LightGBM and PyTorch MLP models for Drebin-215 APK features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from malware_classifier.config import (  # noqa: E402
    MODEL_DIR,
    PROCESSED_DATA_DIR,
)
from malware_classifier.data import grouped_train_validation_test_indices  # noqa: E402
from malware_classifier.model import DROPOUT, HIDDEN_SIZES, MalwareMLP  # noqa: E402

DEFAULT_DATASET = PROCESSED_DATA_DIR / "apk_drebin_clean.csv"
DEFAULT_METRICS = MODEL_DIR / "apk_training_metrics.json"
DEFAULT_MODEL_METADATA = MODEL_DIR / "apk_model_metadata.json"
LIGHTGBM_MODEL_FILE = "apk_lightgbm.txt"
MLP_MODEL_FILE = "apk_mlp.pt"
CLASS_NAMES = ["Benign", "Malware"]
POSITIVE_LABEL = "Malware"
SEED = 42
MAX_EPOCHS = 60
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


def load_apk_dataset(
    path: Path,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cleaned APK dataset not found: {path}. Run scripts/import_apk_dataset.py first."
        )
    frame = pd.read_csv(path)
    if "Class" not in frame.columns:
        raise ValueError(f"{path.name} must contain a 'Class' column.")
    feature_names = [column for column in frame.columns if column != "Class"]
    features = frame.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    targets = frame["Class"].astype(str).to_numpy()
    if set(targets) != set(CLASS_NAMES):
        raise ValueError(
            f"{path.name} must contain exactly the classes {CLASS_NAMES}."
        )
    return (
        frame,
        feature_names,
        features.to_numpy(dtype=np.float32),
        targets,
    )


def feature_group_ids(features: np.ndarray) -> np.ndarray:
    """Keep identical feature vectors in one split."""
    return np.asarray(
        [hashlib.sha256(row.tobytes()).hexdigest() for row in features],
        dtype=str,
    )


def binary_labels(targets: np.ndarray) -> np.ndarray:
    return (targets == POSITIVE_LABEL).astype(np.int64)


def evaluate_binary(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Return the project's standard held-out binary metrics."""
    true_labels = binary_labels(targets)
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision_macro": float(
            precision_score(true_labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(true_labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(true_labels, predictions, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(true_labels, predictions, average="weighted", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(true_labels, probabilities)),
        "confusion_matrix": confusion_matrix(
            true_labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            true_labels,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "threshold": threshold,
    }


def top_feature_importances(
    feature_names: list[str], importances: np.ndarray, limit: int = 15
) -> list[dict[str, Any]]:
    order = np.argsort(importances)[::-1][:limit]
    return [
        {"feature": feature_names[index], "importance": float(importances[index])}
        for index in order
    ]


def train_lightgbm(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    feature_names: list[str],
    model_path: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=min(4, os.cpu_count() or 1),
        verbosity=-1,
    )
    model.fit(
        train_features,
        binary_labels(train_targets),
        eval_set=[
            (validation_features, binary_labels(validation_targets)),
        ],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    probabilities = model.predict_proba(test_features)[:, 1]
    metrics = evaluate_binary(test_targets, probabilities)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))

    importance = model.booster_.feature_importance(importance_type="gain")
    result: dict[str, Any] = {
        "model_file": model_path.name,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "metrics": metrics,
        "top_gain_features": top_feature_importances(feature_names, importance),
    }

    try:
        import shap

        sample_size = min(200, len(test_features))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_features[:sample_size])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_absolute = np.abs(np.asarray(shap_values)).mean(axis=0)
        result["shap"] = {
            "status": "ok",
            "sample_rows": sample_size,
            "top_mean_absolute_features": top_feature_importances(
                feature_names, mean_absolute
            ),
        }
    except Exception as error:  # pragma: no cover - optional dependency/runtime
        result["shap"] = {"status": "unavailable", "error": str(error)}

    return result


def train_mlp(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    feature_names: list[str],
    model_path: Path,
) -> dict[str, Any]:
    seed_everything(SEED)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features).astype(np.float32)
    scaled_validation = scaler.transform(validation_features).astype(np.float32)
    scaled_test = scaler.transform(test_features).astype(np.float32)

    encoded_train = binary_labels(train_targets)
    encoded_validation = binary_labels(validation_targets)
    test_labels = binary_labels(test_targets)

    train_data = TensorDataset(
        torch.from_numpy(scaled_train),
        torch.from_numpy(encoded_train),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_x = torch.from_numpy(scaled_validation)
    validation_y = torch.from_numpy(encoded_validation)
    test_x = torch.from_numpy(scaled_test)

    model = MalwareMLP(len(feature_names), len(CLASS_NAMES))
    class_counts = np.bincount(encoded_train, minlength=len(CLASS_NAMES))
    class_weights = len(encoded_train) / (
        len(CLASS_NAMES) * np.maximum(class_counts, 1)
    )
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_features)

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(validation_x), validation_y
            ).item()

        epochs_trained = epoch
        print(
            f"mlp epoch {epoch:02d}: "
            f"train_loss={total_loss / len(train_data):.4f}, "
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
                print(f"mlp: early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a model.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(test_x), dim=1).cpu().numpy()[:, 1]

    metrics = evaluate_binary(test_targets, probabilities)
    checkpoint = {
        "workflow": "apk_drebin",
        "dataset": "Drebin-215",
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
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
        },
        "test_metrics": metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, model_path)
    return {
        "model_file": model_path.name,
        "epochs_trained": epochs_trained,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM and MLP models on Drebin-215 APK features."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--model-metadata", type=Path, default=DEFAULT_MODEL_METADATA
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    seed_everything(SEED)

    frame, feature_names, features, targets = load_apk_dataset(args.dataset)
    groups = feature_group_ids(features)
    train_indices, validation_indices, test_indices = (
        grouped_train_validation_test_indices(targets, groups, random_state=SEED)
    )
    split_sizes = {
        "train": int(len(train_indices)),
        "validation": int(len(validation_indices)),
        "test": int(len(test_indices)),
    }
    print(
        f"Loaded {len(frame):,} rows, {len(feature_names)} features; "
        f"splits: {split_sizes}"
    )

    lightgbm_result = train_lightgbm(
        features[train_indices],
        targets[train_indices],
        features[validation_indices],
        targets[validation_indices],
        features[test_indices],
        targets[test_indices],
        feature_names,
        args.output_dir / LIGHTGBM_MODEL_FILE,
    )
    mlp_result = train_mlp(
        features[train_indices],
        targets[train_indices],
        features[validation_indices],
        targets[validation_indices],
        features[test_indices],
        targets[test_indices],
        feature_names,
        args.output_dir / MLP_MODEL_FILE,
    )

    lightgbm_f1 = lightgbm_result["metrics"]["f1_macro"]
    mlp_f1 = mlp_result["metrics"]["f1_macro"]
    deployment_model = "lightgbm" if lightgbm_f1 >= mlp_f1 else "mlp"
    deployment_file = (
        LIGHTGBM_MODEL_FILE if deployment_model == "lightgbm" else MLP_MODEL_FILE
    )
    generated_at = datetime.now(timezone.utc).isoformat()

    metrics_payload: dict[str, Any] = {
        "generated_at": generated_at,
        "dataset": args.dataset.name,
        "dataset_family": "Drebin-215",
        "rows": len(frame),
        "feature_count": len(feature_names),
        "classes": CLASS_NAMES,
        "class_counts": dict(
            sorted(
                {
                    str(label): int(count)
                    for label, count in zip(
                        *np.unique(targets, return_counts=True), strict=True
                    )
                }.items()
            )
        ),
        "split_sizes": split_sizes,
        "seed": SEED,
        "selection_metric": "f1_macro",
        "deployment": {
            "model": deployment_model,
            "model_file": deployment_file,
            "threshold": 0.5,
        },
        "models": {
            "lightgbm": lightgbm_result,
            "mlp": mlp_result,
        },
        "license": "license not verified",
        "split_strategy": (
            "Stratified 80/10/10 split with identical feature vectors grouped "
            "so they cannot cross splits."
        ),
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics = args.metrics.with_suffix(".json.tmp")
    temporary_metrics.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_metrics.replace(args.metrics)

    model_metadata = {
        "dataset": "Drebin-215",
        "source_dataset": args.dataset.name,
        "deployment_model": deployment_model,
        "model_file": deployment_file,
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
        "positive_class": POSITIVE_LABEL,
        "threshold": 0.5,
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
        f"LightGBM held-out: accuracy={lightgbm_result['metrics']['accuracy']:.4f}, "
        f"macro_f1={lightgbm_f1:.4f}, roc_auc={lightgbm_result['metrics']['roc_auc']:.4f}"
    )
    print(
        f"MLP held-out: accuracy={mlp_result['metrics']['accuracy']:.4f}, "
        f"macro_f1={mlp_f1:.4f}, roc_auc={mlp_result['metrics']['roc_auc']:.4f}"
    )
    print(f"Deployment model: {deployment_model} ({deployment_file})")
    print(f"Saved metrics: {args.metrics}")
    print(f"Saved model metadata: {args.model_metadata}")


if __name__ == "__main__":
    main()
