from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import pytest

from tescogpt.evaluation.workbook_import import (
    import_annotation_workbook,
    import_retrieval_review_workbook,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _annotation_workbook_with_values(
    tmp_path: Path,
    values: dict[str, str],
) -> Path:
    source = ROOT / "outputs" / "annotation_workbook" / "round1_annotation.xlsx"
    destination = tmp_path / "edited_annotation.xlsx"
    with zipfile.ZipFile(source) as input_archive, zipfile.ZipFile(
        destination, "w"
    ) as output_archive:
        for info in input_archive.infolist():
            data = input_archive.read(info.filename)
            if info.filename == "xl/worksheets/sheet1.xml":
                worksheet = ET.fromstring(data)
                for reference, value in values.items():
                    cell = worksheet.find(f".//{{{MAIN_NS}}}c[@r='{reference}']")
                    assert cell is not None
                    for child in list(cell):
                        cell.remove(child)
                    cell.set("t", "inlineStr")
                    inline = ET.SubElement(cell, f"{{{MAIN_NS}}}is")
                    text = ET.SubElement(inline, f"{{{MAIN_NS}}}t")
                    text.text = value
                data = ET.tostring(worksheet, encoding="utf-8", xml_declaration=True)
            output_archive.writestr(info, data)
    return destination


@pytest.mark.parametrize(
    ("workbook_name", "reference_name", "expected_rows"),
    [
        ("round1_annotation.xlsx", "round1_annotations.csv", 200),
        ("round2_annotation.xlsx", "round2_annotations.csv", 60),
    ],
)
def test_tracked_annotation_workbooks_import_without_context_drift(
    tmp_path: Path,
    workbook_name: str,
    reference_name: str,
    expected_rows: int,
) -> None:
    reference = ROOT / "data" / "golden" / reference_name
    output = tmp_path / reference_name
    shutil.copyfile(reference, output)

    progress = import_annotation_workbook(
        ROOT / "outputs" / "annotation_workbook" / workbook_name,
        output,
    )

    assert progress["row_count"] == expected_rows
    assert progress["completed_count"] == 0
    assert progress["file"] == reference_name
    imported = pd.read_csv(output, dtype="string", keep_default_na=False)
    expected = pd.read_csv(reference, dtype="string", keep_default_na=False)
    pd.testing.assert_frame_equal(imported, expected)


def test_annotation_import_preserves_human_values_and_is_atomic(tmp_path: Path) -> None:
    workbook = _annotation_workbook_with_values(
        tmp_path,
        {
            "D2": "other_or_unclear",
            "E2": "ESCALATE",
            "F2": "IMAGE_OR_MISSING_CONTEXT",
            "H2": "Acknowledge that the image is unavailable.",
            "I2": "Do not guess what the image contains.",
            "K2": "human_primary",
        },
    )
    reference = ROOT / "data" / "golden" / "round1_annotations.csv"
    output = tmp_path / "round1.csv"
    shutil.copyfile(reference, output)

    progress = import_annotation_workbook(workbook, output)

    assert progress["completed_count"] == 1
    imported = pd.read_csv(output, dtype="string", keep_default_na=False)
    first = imported.loc[imported["case_id"].eq("tesco-2950175")].iloc[0]
    assert first["intent_label"] == "other_or_unclear"
    assert first["annotator_id"] == "human_primary"

    atomic_output = tmp_path / "must_remain_blank.csv"
    shutil.copyfile(reference, atomic_output)
    original = atomic_output.read_bytes()
    with pytest.raises(ValueError, match="rows are not completely labelled"):
        import_annotation_workbook(workbook, atomic_output, require_complete=True)
    assert atomic_output.read_bytes() == original


def test_annotation_import_rejects_changed_case_identity(tmp_path: Path) -> None:
    workbook = _annotation_workbook_with_values(tmp_path, {"B2": "changed-case-id"})
    reference = ROOT / "data" / "golden" / "round1_annotations.csv"
    output = tmp_path / "round1.csv"
    shutil.copyfile(reference, output)
    original = output.read_bytes()

    with pytest.raises(ValueError, match="IDs differ from the frozen reference"):
        import_annotation_workbook(workbook, output)

    assert output.read_bytes() == original


def test_tracked_retrieval_workbook_imports_without_blinding_drift(
    tmp_path: Path,
) -> None:
    reference = ROOT / "data" / "review" / "retrieval_relevance.csv"
    output = tmp_path / "retrieval.csv"
    shutil.copyfile(reference, output)

    progress = import_retrieval_review_workbook(
        ROOT / "outputs" / "review_workbook" / "retrieval_relevance_review.xlsx",
        output,
    )

    assert progress["row_count"] == 117
    assert progress["completed_count"] == 0
    imported = pd.read_csv(output, dtype="string", keep_default_na=False)
    expected = pd.read_csv(reference, dtype="string", keep_default_na=False)
    pd.testing.assert_frame_equal(imported, expected)
