from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).parents[1]
SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"x": SHEET_NS}


def _sheet_xml(archive: ZipFile, sheet_name: str) -> ET.Element:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = next(
        node
        for node in workbook.findall("x:sheets/x:sheet", NS)
        if node.attrib["name"] == sheet_name
    )
    relationship_id = sheet.attrib[f"{{{REL_NS}}}id"]
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationship = next(
        node
        for node in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        if node.attrib["Id"] == relationship_id
    )
    target = relationship.attrib["Target"].lstrip("/")
    member = target if target.startswith("xl/") else f"xl/{target}"
    return ET.fromstring(archive.read(member))


def _cell(sheet: ET.Element, address: str) -> ET.Element:
    cell = sheet.find(f".//x:c[@r='{address}']", NS)
    if cell is None:
        raise AssertionError(f"Workbook cell is missing: {address}")
    return cell


def _value(cell: ET.Element) -> str:
    value = cell.find("x:v", NS)
    return "" if value is None or value.text is None else value.text


@pytest.mark.parametrize(
    ("workbook_name", "csv_name", "last_row"),
    [
        ("round1_annotation.xlsx", "round1_annotations.csv", 201),
        ("round2_annotation.xlsx", "round2_annotations.csv", 61),
    ],
)
def test_annotation_workbook_matches_the_frozen_human_contract(
    workbook_name: str,
    csv_name: str,
    last_row: int,
) -> None:
    workbook_path = ROOT / "outputs" / "annotation_workbook" / workbook_name
    with ZipFile(workbook_path) as archive:
        annotations = _sheet_xml(archive, "Annotations")
        instructions = _sheet_xml(archive, "Instructions")
        codebook = _sheet_xml(archive, "Codebook")

    formula = _cell(instructions, "B5").find("x:f", NS)
    assert formula is not None and formula.text is not None
    assert formula.text.startswith("COUNTIFS(")
    for column in ("D", "E", "F", "H", "I", "K"):
        assert f"${column}$2:${column}${last_row}" in formula.text
    assert _value(_cell(instructions, "B5")) == "0"
    assert _value(_cell(codebook, "B16")) == "1.1"

    row_numbers = [
        int(re.search(r"\d+$", cell.attrib["r"])[0])
        for cell in annotations.findall(".//x:c", NS)
    ]
    assert max(row_numbers) == last_row
    validations = annotations.findall(".//x:dataValidation", NS)
    assert {validation.attrib["sqref"] for validation in validations} == {
        f"D2:D{last_row}",
        f"E2:E{last_row}",
        f"F2:F{last_row}",
        f"J2:J{last_row}",
    }

    with (ROOT / "data" / "golden" / csv_name).open(
        encoding="utf-8", newline=""
    ) as source:
        expected_case_ids = [row["case_id"] for row in csv.DictReader(source)]
    workbook_case_ids = [
        _value(_cell(annotations, f"B{row_number}"))
        for row_number in range(2, last_row + 1)
    ]
    assert workbook_case_ids == expected_case_ids

    populated_completion_cells = []
    for row_number in range(2, last_row + 1):
        for column in ("D", "E", "F", "H", "I", "K"):
            value = _value(_cell(annotations, f"{column}{row_number}"))
            if value.strip():
                populated_completion_cells.append(f"{column}{row_number}")
    assert not populated_completion_cells

    assert not re.search(r"#(?:REF!|DIV/0!|VALUE!|NAME\?|N/A)", formula.text)
