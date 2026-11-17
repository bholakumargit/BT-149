"""Tests for the Drebin-215 import, cleaning, and inference paths."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from malware_classifier import apk_inference
from malware_classifier import app as app_module
from malware_classifier.apk_data import clean_drebin_frame, import_apk_dataset
from malware_classifier.app import app


def test_clean_drebin_replaces_question_marks_before_deduplication():
    raw = pd.DataFrame(
        {
            "f1": ["1", "1", "1", "0", "?", "9"],
            "f2": ["0", "0", "0", "1", "1", "9"],
            "class": ["B", "B", "S", "S", "S", "B"],
        }
    )

    cleaned, counts = clean_drebin_frame(raw)

    assert counts["question_mark_cells_replaced"] == 1
    assert counts["question_mark_cells_by_column"] == {"f1": 1}
    # The exact duplicate is removed first; the '?' row is not treated as a
    # duplicate of the explicit "0" row until after deduplication.
    assert counts["rows_removed_as_exact_duplicates"] == 1
    # The [1, 0] pattern appears under both labels and is removed entirely.
    assert counts["conflicting_feature_patterns_removed"] == 1
    assert counts["rows_removed_for_conflicting_labels"] == 2
    assert counts["rows_after_cleaning"] == 3
    assert counts["class_counts"] == {"Benign": 1, "Malware": 2}
    assert cleaned["Class"].tolist() == ["Malware", "Malware", "Benign"]
    assert cleaned[["f1", "f2"]].to_numpy().tolist() == [[0, 1], [0, 1], [9, 9]]


def test_clean_drebin_measures_counts_instead_of_hard_coding_them():
    raw = pd.DataFrame(
        {
            "f1": ["1", "2", "3"],
            "f2": ["0", "1", "2"],
            "class": ["B", "B", "S"],
        }
    )

    _, counts = clean_drebin_frame(raw)

    assert counts["source_rows"] == 3
    assert counts["source_label_counts"] == {"B": 2, "S": 1}
    assert counts["rows_after_cleaning"] == 3
    assert counts["class_counts"] == {"Benign": 2, "Malware": 1}


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"f1": ["1"]}), "target column"),
        (
            pd.DataFrame({"f1": ["1"], "class": ["X"]}),
            "unsupported labels",
        ),
        (
            pd.DataFrame({"f1": ["1"], "class": [None]}),
            "missing target labels",
        ),
    ],
)
def test_clean_drebin_rejects_invalid_sources(frame, message):
    with pytest.raises(ValueError, match=message):
        clean_drebin_frame(frame)


def test_import_apk_dataset_writes_cleaned_csv_and_metadata(tmp_path):
    source = tmp_path / "drebin.csv"
    source.write_text(
        "f1,f2,class\n1,0,B\n1,0,B\n0,1,S\n?,1,S\n",
        encoding="utf-8",
    )
    categories = tmp_path / "categories.csv"
    categories.write_text("f1,Manifest\nf2,API call\nclass,Label\n", encoding="utf-8")

    output, metadata = import_apk_dataset(
        source,
        tmp_path / "clean.csv",
        tmp_path / "metadata.json",
        categories,
    )

    assert output.is_file()
    assert metadata["counts"]["question_mark_cells_replaced"] == 1
    assert metadata["counts"]["rows_after_cleaning"] == 3
    assert metadata["license"] == "license not verified"
    assert metadata["features"]["category_counts"] == {"API call": 1, "Manifest": 1}
    written = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert written["counts"] == metadata["counts"]
    assert written["source"]["sha256"]


def test_parse_feature_csv_ignores_lowercase_class_column():
    from malware_classifier.inference import parse_feature_csv

    values = parse_feature_csv(
        b"f1,f2,class\n1,2,S\n",
        ["f1", "f2"],
    )

    assert values == [1.0, 2.0]


def test_apk_model_availability_reports_missing_metadata(tmp_path, monkeypatch):
    apk_inference.load_apk_metadata.cache_clear()
    monkeypatch.setattr(
        apk_inference, "APK_MODEL_METADATA_PATH", tmp_path / "missing.json"
    )

    assert apk_inference.apk_model_availability() == {
        "ready": False,
        "model": None,
        "model_file": None,
    }
    apk_inference.load_apk_metadata.cache_clear()


def test_apk_endpoint_rejects_unknown_files_without_models():
    response = TestClient(app).post(
        "/api/analyze/apk",
        files={"file": ("features.txt", b"f1,f2\n1,2\n", "text/plain")},
    )

    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]


def test_apk_status_in_model_status(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "apk_model_availability",
        lambda: {"ready": False, "model": None, "model_file": None},
    )

    payload = TestClient(app).get("/api/model/status").json()

    assert payload["apk_drebin"]["ready"] is False


def test_apk_inference_uses_deployment_metadata(tmp_path, monkeypatch):
    import lightgbm as lgb
    import numpy as np

    features = ["f1", "f2"]
    rows = np.array(
        [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0]], dtype=float
    )
    labels = np.array([0, 0, 1, 1, 0, 1])
    model = lgb.LGBMClassifier(
        n_estimators=8, num_leaves=2, min_child_samples=1, verbose=-1
    )
    model.fit(rows, labels)
    model.booster_.save_model(str(tmp_path / "apk_lightgbm.txt"))

    metadata_path = tmp_path / "apk_model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "feature_names": features,
                "class_names": ["Benign", "Malware"],
                "deployment_model": "lightgbm",
                "model_file": "apk_lightgbm.txt",
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    apk_inference.load_apk_metadata.cache_clear()
    apk_inference.load_lightgbm_model.cache_clear()
    monkeypatch.setattr(apk_inference, "APK_MODEL_METADATA_PATH", metadata_path)
    monkeypatch.setattr(apk_inference, "MODEL_DIR", tmp_path)

    result = apk_inference.analyze_apk_feature_csv(b"f1,f2,class\n1,1,S\n")

    apk_inference.load_apk_metadata.cache_clear()
    apk_inference.load_lightgbm_model.cache_clear()
    assert result["dataset"] == "Drebin-215"
    assert result["model"] == "lightgbm"
    assert set(result["prediction"]["probabilities"]) == {"Benign", "Malware"}
    assert result["prediction"]["label"] in {"Benign", "Malware"}


def test_apk_endpoint_returns_prediction_when_ready(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "apk_model_availability",
        lambda: {"ready": True, "model": "lightgbm", "model_file": "apk_lightgbm.txt"},
    )
    monkeypatch.setattr(
        app_module,
        "analyze_apk_feature_csv",
        lambda contents: {
            "dataset": "Drebin-215",
            "model": "lightgbm",
            "prediction": {
                "label": "Malware",
                "confidence": 0.9,
                "probabilities": {"Benign": 0.1, "Malware": 0.9},
            },
            "message": "Predicted Malware (90.0% confidence).",
        },
    )

    response = TestClient(app).post(
        "/api/analyze/apk",
        files={"file": ("row.csv", b"f1,f2\n1,2\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["prediction"]["label"] == "Malware"
