"""Tests for raw APK/PDF upload analysis."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from malware_classifier import pdf_features, pdf_inference
from malware_classifier.app import app
from malware_classifier.pdf_features import (
    PDF_MODEL_FEATURE_NAMES,
    extract_pdf_features,
    pdf_indicator_flags,
)

TINY_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R /OpenAction 4 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj
4 0 obj << /S /JavaScript /JS (app.alert('hello')) >> endobj
trailer << /Root 1 0 R >>
startxref 0
%%EOF
"""


def test_pdf_feature_extraction_reports_structure_and_active_content():
    features = extract_pdf_features(TINY_PDF)

    assert features["header"] == 1.4
    assert features["pageno"] == 1
    assert features["Javascript"] == 1
    assert features["JS"] == 1
    assert features["OpenAction"] == 1
    assert features["obj"] == 4
    indicators = pdf_indicator_flags(TINY_PDF, features)
    assert "Embedded JavaScript" in indicators
    assert "Automatic action on open (/OpenAction)" in indicators


def test_pdf_feature_vector_contains_only_reproducible_model_features():
    features = extract_pdf_features(TINY_PDF)
    vector = pdf_features.pdf_feature_vector(features, PDF_MODEL_FEATURE_NAMES)

    assert len(vector) == len(PDF_MODEL_FEATURE_NAMES)
    assert "metadata size" not in PDF_MODEL_FEATURE_NAMES
    assert "Colors" not in PDF_MODEL_FEATURE_NAMES
    assert "JS" in PDF_MODEL_FEATURE_NAMES


@pytest.mark.parametrize("payload", [b"", b"not a pdf"])
def test_pdf_feature_extraction_rejects_non_pdf_bytes(payload):
    with pytest.raises(ValueError):
        extract_pdf_features(payload)


def test_pdf_analysis_returns_binary_verdict_without_invented_family():
    if not pdf_inference.pdf_model_availability()["ready"]:
        pytest.skip("PDF model is not trained in this checkout.")

    result = pdf_inference.analyze_pdf_bytes(TINY_PDF)

    assert result["verdict"] in {"Malware", "Benign"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["family"] is None
    assert result["family_available"] is False
    assert result["indicators"]


def test_upload_endpoint_rejects_non_apk_pdf_file():
    response = TestClient(app).post(
        "/api/analyze/file",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only Android packages" in response.json()["detail"]


def test_upload_endpoint_rejects_invalid_pdf_payload():
    response = TestClient(app).post(
        "/api/analyze/file",
        files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 400


def test_upload_endpoint_returns_pdf_verdict():
    if not pdf_inference.pdf_model_availability()["ready"]:
        pytest.skip("PDF model is not trained in this checkout.")

    response = TestClient(app).post(
        "/api/analyze/file",
        files={"file": ("sample.pdf", TINY_PDF, "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_type"] == "pdf"
    assert payload["filename"] == "sample.pdf"
    assert payload["verdict"] in {"Malware", "Benign"}
    assert isinstance(payload["confidence"], float)
    assert payload["model"]["name"] == "LightGBM"


def test_model_status_lists_pdf_and_apk_file_models():
    payload = TestClient(app).get("/api/model/status").json()

    assert "pdf" in payload
    assert "apk_file" in payload
    assert isinstance(payload["pdf"]["ready"], bool)
    assert isinstance(payload["apk_file"]["ready"], bool)
    assert isinstance(payload["upload_analysis_ready"], bool)


def test_pdf_metadata_rejects_mismatched_feature_order(tmp_path, monkeypatch):
    metadata_path = tmp_path / "pdf_model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "deployment_model": "lightgbm",
                "model_file": "missing.txt",
                "feature_names": ["JS", "Javascript"],
                "class_names": ["Benign", "Malware"],
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    pdf_inference.load_pdf_metadata.cache_clear()
    monkeypatch.setattr(pdf_inference, "PDF_MODEL_METADATA_PATH", metadata_path)

    with pytest.raises(pdf_inference.InvalidPdfModelError):
        pdf_inference.load_pdf_metadata(str(metadata_path))

    pdf_inference.load_pdf_metadata.cache_clear()
