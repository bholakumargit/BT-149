import json

from fastapi.testclient import TestClient

from malware_classifier import app as app_module
from malware_classifier.app import app


def test_dataset_status_only_lists_csvs_with_required_labels(tmp_path, monkeypatch):
    (tmp_path / "valid.csv").write_text(
        "feature_a,Class,Category\n1,Benign,Benign\n", encoding="utf-8"
    )
    (tmp_path / "no_labels.csv").write_text("feature_a\n1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a dataset", encoding="utf-8")
    monkeypatch.setattr(app_module, "RAW_DATA_DIR", tmp_path)

    payload = TestClient(app).get("/api/dataset/status").json()

    assert payload["ready"] is True
    assert payload["datasets"] == ["valid.csv"]


def test_dataset_status_handles_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "RAW_DATA_DIR", tmp_path / "missing")

    payload = TestClient(app).get("/api/dataset/status").json()

    assert payload["ready"] is False
    assert payload["datasets"] == []


def test_metrics_endpoint_summarizes_training_files(tmp_path, monkeypatch):
    training_path = tmp_path / "training_metrics.json"
    training_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-10-19T11:07:51+00:00",
                "dataset": "CIC-MalMem-2022.csv",
                "dataset_rows": 58596,
                "feature_count": 55,
                "models": {
                    "detection": {
                        "classes": ["Benign", "Malware"],
                        "split_sizes": {"train": 10, "validation": 2, "test": 2},
                        "metrics": {
                            "accuracy": 0.99,
                            "precision_macro": 0.98,
                            "recall_macro": 0.97,
                            "f1_macro": 0.96,
                            "f1_weighted": 0.95,
                            "roc_auc": 0.995,
                        },
                    },
                    "classification": {
                        "classes": ["Trojan-Zeus"],
                        "split_sizes": {},
                        "metrics": {"accuracy": 0.5},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    maldroid_path = tmp_path / "maldroid_training_metrics.json"
    maldroid_path.write_text(
        json.dumps(
            {
                "models": {
                    "syscalls": {
                        "class_names": ["1", "2"],
                        "metrics": {"accuracy": 0.86, "f1_macro": 0.83},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "TRAINING_METRICS_PATH", training_path)
    monkeypatch.setattr(app_module, "MALDROID_METRICS_PATH", maldroid_path)

    payload = TestClient(app).get("/api/metrics").json()

    assert payload["available"] is True
    assert payload["cic_malmem"]["dataset_rows"] == 58596
    assert payload["cic_malmem"]["detection"]["accuracy"] == 0.99
    assert payload["cic_malmem"]["detection"]["precision_macro"] == 0.98
    assert payload["cic_malmem"]["classification"]["accuracy"] == 0.5
    assert payload["maldroid"]["syscalls"]["f1_macro"] == 0.83
    assert payload["maldroid"]["syscalls_binders"] is None


def test_metrics_endpoint_reports_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "TRAINING_METRICS_PATH", tmp_path / "missing_training.json"
    )
    monkeypatch.setattr(
        app_module, "MALDROID_METRICS_PATH", tmp_path / "missing_maldroid.json"
    )  # deliberately absent
    monkeypatch.setattr(
        app_module, "APK_METRICS_PATH", tmp_path / "missing_apk.json"
    )  # deliberately absent

    payload = TestClient(app).get("/api/metrics").json()

    assert payload["available"] is False
    assert payload["cic_malmem"] is None
    assert payload["apk_drebin"] is None
    assert "No training metrics" in payload["message"]
