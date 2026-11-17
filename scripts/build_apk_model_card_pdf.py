"""Build the Drebin-215 APK model card as Markdown and PDF."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from malware_classifier.config import (  # noqa: E402
    APK_CLEAN_DATASET_PATH,
    APK_METADATA_PATH,
    APK_METRICS_PATH,
)

MARKDOWN_PATH = PROJECT_DIR / "docs" / "apk-model-card.md"
PDF_PATH = PROJECT_DIR / "docs" / "apk-model-card.pdf"
DISPLAY_NAMES = {"lightgbm": "LightGBM", "mlp": "PyTorch MLP"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run the APK training pipeline first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def format_metric(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "not available"


def confusion_rows(metrics: dict[str, Any]) -> list[list[str]]:
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 2:
        return [["not available", ""], ["", ""]]
    return [[str(cell) for cell in row] for row in matrix]


def metric_table(models: dict[str, Any]) -> Table:
    header = ["Model", "Accuracy", "Macro F1", "Weighted F1", "ROC-AUC"]
    rows = [header]
    for key, display in DISPLAY_NAMES.items():
        metrics = models.get(key, {}).get("metrics", {})
        rows.append(
            [
                display,
                format_metric(metrics.get("accuracy")),
                format_metric(metrics.get("f1_macro")),
                format_metric(metrics.get("f1_weighted")),
                format_metric(metrics.get("roc_auc")),
            ]
        )
    table = Table(rows, colWidths=[1.2 * inch, 1.0 * inch, 1.0 * inch, 1.1 * inch, 1.0 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fb3c8")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def build_markdown(metrics: dict[str, Any], cleaning: dict[str, Any]) -> str:
    models = metrics.get("models", {})
    deployment = metrics.get("deployment", {})
    counts = cleaning.get("counts", {})
    source = cleaning.get("source", {})
    source_name = Path(str(source.get("path", "unknown"))).name
    source_labels = counts.get("source_label_counts", {})
    if isinstance(source_labels, dict):
        labels_text = ", ".join(
            f"{label}: {count:,}" for label, count in source_labels.items()
        )
    else:
        labels_text = "unknown"
    lines = [
        "# Drebin-215 APK static-feature model card",
        "",
        "## Model details",
        "",
        f"- Dataset: {metrics.get('dataset_family', 'Drebin-215')}",
        f"- Raw source file: `{source_name}`",
        f"- Cleaned training file: `{metrics.get('dataset', 'unknown')}`",
        f"- Task: binary benign/malware classification from 215 pre-extracted APK features",
        f"- Deployment model: {DISPLAY_NAMES.get(deployment.get('model'), deployment.get('model'))}",
        f"- Decision threshold: {deployment.get('threshold', 0.5)}",
        f"- Generated: {metrics.get('generated_at', 'unknown')}",
        "- License: license not verified",
        "",
        "## Intended use",
        "",
        "Local academic experimentation and dashboard demonstrations only. The model",
        "classifies one row of pre-extracted numeric APK features. It does not inspect",
        "or execute APK binaries and must not be used as a production security control.",
        "",
        "## Data preparation",
        "",
        f"- Source rows: {counts.get('source_rows', 'unknown'):,}",
        f"- Source labels: {labels_text}",
        f"- `?` cells replaced with 0 before deduplication: {counts.get('question_mark_cells_replaced', 'unknown')}",
        f"- Exact duplicate rows removed: {counts.get('rows_removed_as_exact_duplicates', 'unknown'):,}",
        f"- Conflicting feature patterns removed: {counts.get('conflicting_feature_patterns_removed', 'unknown')} "
        f"({counts.get('rows_removed_for_conflicting_labels', 'unknown')} rows)",
        f"- Rows after cleaning: {counts.get('rows_after_cleaning', 'unknown'):,}",
        f"- Cleaned class counts: {counts.get('class_counts', 'unknown')}",
        "",
        "The `?` replacement happens before deduplication so the measured row counts",
        "stay reproducible. Identical numeric feature vectors are kept in one split.",
        "",
        "## Evaluation",
        "",
        f"- Split sizes: {metrics.get('split_sizes', 'unknown')}",
        f"- Seed: {metrics.get('seed', 'unknown')}",
        f"- Model selection metric: {metrics.get('selection_metric', 'unknown')}",
        "",
        "| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, display in DISPLAY_NAMES.items():
        model_metrics = models.get(key, {}).get("metrics", {})
        lines.append(
            f"| {display} | {format_metric(model_metrics.get('accuracy'))} | "
            f"{format_metric(model_metrics.get('f1_macro'))} | "
            f"{format_metric(model_metrics.get('f1_weighted'))} | "
            f"{format_metric(model_metrics.get('roc_auc'))} |"
        )
    lines.extend(["", "### Confusion matrices", ""])
    for key, display in DISPLAY_NAMES.items():
        matrix = confusion_rows(models.get(key, {}).get("metrics", {}))
        lines.append(f"{display} (rows = actual Benign/Malware, columns = predicted):")
        lines.append("")
        lines.append(f"- Benign: {matrix[0]}")
        lines.append(f"- Malware: {matrix[1]}")
        lines.append("")
    lines.extend(
        [
            "## Top LightGBM gain features",
            "",
        ]
    )
    for item in models.get("lightgbm", {}).get("top_gain_features", [])[:10]:
        lines.append(f"- `{item.get('feature')}`: {item.get('importance'):.2f}")
    shap_block = models.get("lightgbm", {}).get("shap", {})
    if shap_block.get("status") == "ok":
        lines.extend(
            [
                "",
                "## SHAP summary",
                "",
                f"- Sample rows explained: {shap_block.get('sample_rows', 'unknown')}",
                "",
            ]
        )
        for item in shap_block.get("top_mean_absolute_features", [])[:5]:
            lines.append(
                f"- `{item.get('feature')}`: {item.get('importance'):.4f} mean |SHAP|"
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Only 215 pre-extracted static features; no raw APK inspection.",
            "- The source archive contained no license document, so the dataset is",
            "  marked \"license not verified\".",
            "- Static features can be evaded by packing, obfuscation, and dynamic code loading.",
            "- Evaluation is a single grouped hold-out split, not a temporal benchmark.",
            "- Class imbalance remains (benign rows outnumber malware rows).",
            "",
            "## Ethical and safety notes",
            "",
            "The application never executes uploaded binaries. Raw malware samples are",
            "not redistributed with the repository, and dataset files remain local.",
            "",
        ]
    )
    return "\n".join(lines)


def build_pdf(markdown_text: str, output_path: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "CardTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=25,
            textColor=colors.HexColor("#17365d"),
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            "CardHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#17365d"),
            spaceBefore=12,
            spaceAfter=5,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "CardBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            spaceAfter=4,
        )
    )

    story: list[Any] = []
    table_lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["CardTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["CardHeading"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], styles["CardHeading"]))
        elif line.startswith("|"):
            table_lines.append(line)
            continue
        else:
            if table_lines:
                story.append(_markdown_table(table_lines))
                table_lines = []
            if not line:
                story.append(Spacer(1, 4))
            elif line.startswith("- "):
                story.append(
                    Paragraph(
                        "&#8226; " + _inline(line[2:]),
                        styles["CardBody"],
                        bulletText=None,
                    )
                )
            else:
                story.append(Paragraph(_inline(line), styles["CardBody"]))
    if table_lines:
        story.append(_markdown_table(table_lines))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Drebin-215 APK model card",
    )
    document.build(story)


def _inline(text: str) -> str:
    import html
    import re

    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', escaped)


def _markdown_table(lines: list[str]) -> Table:
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in lines
        if not set(line.replace("|", "").strip()) <= {"-", ":", " "}
    ]
    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fb3c8")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def main() -> None:
    metrics = load_json(APK_METRICS_PATH)
    cleaning = load_json(APK_METADATA_PATH)
    if not APK_CLEAN_DATASET_PATH.is_file():
        print(f"Warning: cleaned dataset not found at {APK_CLEAN_DATASET_PATH}")
    markdown_text = build_markdown(metrics, cleaning)
    MARKDOWN_PATH.write_text(markdown_text, encoding="utf-8")
    build_pdf(markdown_text, PDF_PATH)
    print(f"Wrote: {MARKDOWN_PATH}")
    print(f"Wrote: {PDF_PATH}")


if __name__ == "__main__":
    main()
