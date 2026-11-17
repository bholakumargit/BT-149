import io
import struct
import zipfile

import pytest
from fastapi.testclient import TestClient

from malware_classifier.app import app
from malware_classifier.static_analysis import inspect_static_file


def build_minimal_pe() -> bytes:
    contents = bytearray(0x1A0)
    contents[:2] = b"MZ"
    struct.pack_into("<I", contents, 0x3C, 0x80)
    contents[0x80 : 0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", contents, 0x84, 0x14C, 1, 123, 0, 0, 0xE0, 0x102)
    struct.pack_into("<H", contents, 0x98, 0x10B)
    struct.pack_into("<I", contents, 0xA8, 0x1000)
    struct.pack_into("<HH", contents, 0xDC, 3, 0x140)
    section_offset = 0x178
    contents[section_offset : section_offset + 8] = b".text\0\0\0"
    struct.pack_into("<IIII", contents, section_offset + 8, 0x100, 0x1000, 0, 0)
    struct.pack_into("<I", contents, section_offset + 36, 0x60000020)
    return bytes(contents)


def build_minimal_apk() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("AndroidManifest.xml", b"<manifest android.permission.INTERNET/>")
        archive.writestr("classes.dex", b"dex")
        archive.writestr("lib/arm64-v8a/libsample.so", b"library")
    return stream.getvalue()


@pytest.mark.parametrize(
    ("filename", "contents", "container"),
    [
        (
            "sample.exe",
            build_minimal_pe(),
            "Portable Executable (PE)",
        ),
        (
            "sample.pdf",
            b"%PDF-1.7\n/Type /Page /JavaScript /OpenAction %%EOF",
            "PDF document (header signature detected)",
        ),
        (
            "sample.apk",
            build_minimal_apk(),
            "Android package (ZIP-based APK)",
        ),
        ("sample.doc", bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 64, "OLE Compound File"),
        ("sample.msi", bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 64, "OLE Compound File"),
    ],
)
def test_inspect_supported_file_types_without_execution(filename, contents, container):
    report = inspect_static_file(filename, contents)

    assert report["inspection_type"] == "static_metadata_only"
    assert report["verdict"] is None
    assert report["features"]["container"] == container
    assert report["features"]["sha256"]


def test_pe_extractor_reports_section_metadata():
    report = inspect_static_file("sample.exe", build_minimal_pe())

    assert report["features"]["section_count"] == 1
    assert report["features"]["sections"][0]["name"] == ".text"
    assert report["features"]["sections"][0]["executable"] is True


def test_apk_extractor_reports_archive_metadata():
    report = inspect_static_file("sample.apk", build_minimal_apk())

    assert report["features"]["dex_file_count"] == 1
    assert report["features"]["native_library_count"] == 1
    assert report["features"]["manifest_permission_strings"] == [
        "android.permission.INTERNET"
    ]


def test_inspect_rejects_wrong_signature_and_unsupported_extension():
    with pytest.raises(ValueError, match="valid DOS executable signature"):
        inspect_static_file("not-really.exe", b"ordinary text")
    with pytest.raises(ValueError, match="Unsupported file type"):
        inspect_static_file("sample.dll", b"MZ" + b"\0" * 64)


def test_inspect_endpoint_returns_static_metadata_for_pdf():
    response = TestClient(app).post(
        "/api/inspect",
        files={"file": ("sample.pdf", b"%PDF-1.7\n%%EOF", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["verdict"] is None
    assert response.json()["features"]["container"].startswith("PDF")


def test_inspect_endpoint_rejects_wrong_file_signature():
    response = TestClient(app).post(
        "/api/inspect",
        files={"file": ("sample.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert "PDF header" in response.json()["detail"]
