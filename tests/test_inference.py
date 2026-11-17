import pytest

torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient

from malware_classifier import app as app_module
from malware_classifier import inference
from malware_classifier.app import app
from malware_classifier.model import DROPOUT, HIDDEN_SIZES, MalwareMLP


def write_checkpoint(path, feature_names, class_names, predicted_index):
    model = MalwareMLP(len(feature_names), len(class_names))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[-1].bias[predicted_index] = 5

    torch.save(
        {
            "feature_names": feature_names,
            "class_names": class_names,
            "normalization_mean": [0.0] * len(feature_names),
            "normalization_scale": [1.0] * len(feature_names),
            "architecture": {
                "type": "MLP",
                "hidden_sizes": list(HIDDEN_SIZES),
                "dropout": DROPOUT,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )


@pytest.fixture
def trained_test_models(tmp_path, monkeypatch):
    features = ["feature_a", "feature_b"]
    detection_path = tmp_path / "detection.pt"
    classification_path = tmp_path / "classification.pt"
    write_checkpoint(
        detection_path, features, ["Benign", "Malware"], predicted_index=1
    )
    write_checkpoint(
        classification_path,
        features,
        ["Trojan-Emotet", "Trojan-Zeus"],
        predicted_index=1,
    )

    inference.load_checkpoint.cache_clear()
    monkeypatch.setattr(inference, "DETECTION_MODEL_PATH", detection_path)
    monkeypatch.setattr(inference, "CLASSIFICATION_MODEL_PATH", classification_path)
    monkeypatch.setattr(app_module, "DETECTION_MODEL_PATH", detection_path)
    monkeypatch.setattr(app_module, "CLASSIFICATION_MODEL_PATH", classification_path)
    yield features
    inference.load_checkpoint.cache_clear()


def test_parse_feature_csv_reorders_columns_to_match_model():
    values = inference.parse_feature_csv(
        b"feature_b,feature_a\n2.5,1.5\n",
        ["feature_a", "feature_b"],
    )

    assert values == [1.5, 2.5]


def test_parse_feature_csv_ignores_label_and_index_columns():
    values = inference.parse_feature_csv(
        b"Unnamed: 0,Class,Category,Filename,feature_b,feature_a\n"
        b"0,Malware,Trojan-Zeus,sample-1.raw,2.5,1.5\n",
        ["feature_a", "feature_b"],
    )

    assert values == [1.5, 2.5]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"feature_a\n1\n", "missing features"),
        (b"feature_a,feature_b,notes\n1,2,extra\n", "unexpected columns"),
        (b"feature_a,feature_b\n1,nope\n", "must contain a numeric value"),
        (b"feature_a,feature_b\n1,2\n3,4\n", "exactly one feature record"),
    ],
)
def test_parse_feature_csv_rejects_invalid_inputs(contents, message):
    with pytest.raises(ValueError, match=message):
        inference.parse_feature_csv(contents, ["feature_a", "feature_b"])


def test_analyze_endpoint_returns_detection_and_family(trained_test_models):
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        files={"file": ("features.csv", b"feature_b,feature_a\n2,1\n", "text/csv")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["detection"]["label"] == "Malware"
    assert result["classification"]["label"] == "Trojan-Zeus"
    assert result["message"] == "Detection: Malware. Predicted family: Trojan-Zeus."


def test_analyze_endpoint_accepts_row_with_dataset_labels(trained_test_models):
    response = TestClient(app).post(
        "/api/analyze",
        files={
            "file": (
                "row.csv",
                b"Class,Category,Filename,feature_b,feature_a\n"
                b"Malware,Trojan-Zeus,sample-1.raw,2,1\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["detection"]["label"] == "Malware"
    assert response.json()["classification"]["label"] == "Trojan-Zeus"


def test_analyze_endpoint_rejects_non_csv_file(trained_test_models):
    response = TestClient(app).post(
        "/api/analyze",
        files={"file": ("sample.exe", b"not a feature table", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Raw malware files are not accepted" in response.json()["detail"]


def test_maldroid_endpoint_predicts_from_selected_feature_schema(tmp_path, monkeypatch):
    features = ["syscall_a", "syscall_b"]
    model_path = tmp_path / "maldroid.pt"
    write_checkpoint(model_path, features, ["1", "2"], predicted_index=1)
    inference.load_checkpoint.cache_clear()
    monkeypatch.setitem(inference.MALDROID_MODEL_PATHS, "syscalls", model_path)
    monkeypatch.setattr(app_module, "MALDROID_SYSCALL_MODEL_PATH", model_path)

    response = TestClient(app).post(
        "/api/analyze/maldroid/syscalls",
        files={
            "file": (
                "row.csv",
                b"syscall_a,syscall_b\n2,3\n",
                "text/csv",
            )
        },
    )

    inference.load_checkpoint.cache_clear()
    assert response.status_code == 200
    assert response.json()["dataset"] == "MalDroid-2020"
    assert response.json()["prediction"]["label"] == "2"
