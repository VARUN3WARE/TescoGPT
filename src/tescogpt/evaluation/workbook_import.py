"""Safe, dependency-free imports from the human-review XLSX workbooks."""

from __future__ import annotations

import os
import posixpath
import re
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

from tescogpt.evaluation.labels import (
    ANNOTATION_CONTEXT_COLUMNS,
    LABEL_COLUMNS,
    validate_annotations,
)
from tescogpt.evaluation.retrieval_review import (
    IMMUTABLE_REVIEW_COLUMNS,
    REVIEW_COLUMNS,
    validate_retrieval_review,
)

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REF = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")


def _column_index(reference: str) -> int:
    match = _CELL_REF.fullmatch(reference)
    if match is None:
        raise ValueError(f"Invalid workbook cell reference: {reference}")
    index = 0
    for character in match.group(1):
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(path))
    return [
        "".join(node.text or "" for node in item.findall(f".//{{{_MAIN_NS}}}t"))
        for item in root.findall(f"{{{_MAIN_NS}}}si")
    ]


def _worksheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationship_id = None
    for sheet in workbook.findall(f".//{{{_MAIN_NS}}}sheet"):
        if sheet.get("name") == sheet_name:
            relationship_id = sheet.get(f"{{{_DOC_REL_NS}}}id")
            break
    if not relationship_id:
        raise ValueError(f"Workbook does not contain worksheet: {sheet_name}")

    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = None
    for relationship in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship"):
        if relationship.get("Id") == relationship_id:
            target = relationship.get("Target")
            break
    if not target:
        raise ValueError(f"Workbook relationship is missing for worksheet: {sheet_name}")
    normalized = posixpath.normpath(
        target.lstrip("/") if target.startswith("/xl/") else posixpath.join("xl", target)
    )
    if normalized not in archive.namelist():
        raise ValueError(f"Workbook worksheet part is missing: {normalized}")
    return normalized


def _cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    if cell.find(f"{{{_MAIN_NS}}}f") is not None:
        raise ValueError(f"Formula cells are not allowed in review data: {cell.get('r', '')}")
    cell_type = cell.get("t", "n")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.findall(f".//{{{_MAIN_NS}}}t")
        )
    value = cell.find(f"{{{_MAIN_NS}}}v")
    text = value.text if value is not None and value.text is not None else ""
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except (IndexError, ValueError) as error:
            raise ValueError(f"Invalid shared-string reference in {cell.get('r', '')}") from error
    if cell_type == "b":
        return "TRUE" if text == "1" else "FALSE"
    if cell_type == "e":
        raise ValueError(f"Error cell is not allowed in review data: {cell.get('r', '')}")
    return text


def read_xlsx_sheet(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Read one rectangular review sheet without evaluating formulas or macros."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Workbook does not exist: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            strings = _shared_strings(archive)
            worksheet = ET.fromstring(archive.read(_worksheet_path(archive, sheet_name)))
    except (KeyError, zipfile.BadZipFile, ET.ParseError) as error:
        raise ValueError(f"Invalid XLSX workbook: {source}") from error

    cells: dict[tuple[int, int], str] = {}
    maximum_row = 0
    maximum_column = 0
    for cell in worksheet.findall(f".//{{{_MAIN_NS}}}c"):
        reference = cell.get("r", "")
        match = _CELL_REF.fullmatch(reference)
        if match is None:
            raise ValueError(f"Invalid workbook cell reference: {reference}")
        row_index = int(match.group(2)) - 1
        column_index = _column_index(reference)
        cells[(row_index, column_index)] = _cell_text(cell, strings)
        maximum_row = max(maximum_row, row_index + 1)
        maximum_column = max(maximum_column, column_index + 1)
    if maximum_row < 2 or maximum_column == 0:
        raise ValueError(f"Worksheet {sheet_name} contains no review rows")

    headers = [cells.get((0, column), "").strip() for column in range(maximum_column)]
    while headers and not headers[-1]:
        headers.pop()
    if not headers or any(not header for header in headers):
        raise ValueError(f"Worksheet {sheet_name} has blank header cells")
    if len(headers) != len(set(headers)):
        raise ValueError(f"Worksheet {sheet_name} has duplicate headers")
    rows = [
        [cells.get((row, column), "") for column in range(len(headers))]
        for row in range(1, maximum_row)
    ]
    return pd.DataFrame(rows, columns=headers, dtype="string")


def _assert_exact_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    *,
    label: str,
) -> pd.DataFrame:
    missing = sorted(set(expected) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(f"{label} columns do not match: {'; '.join(details)}")
    return frame.loc[:, list(expected)].fillna("").astype("string")


def _assert_immutable_content(
    imported: pd.DataFrame,
    reference_path: Path,
    *,
    id_column: str,
    immutable_columns: tuple[str, ...],
    label: str,
) -> None:
    if not reference_path.is_file():
        raise FileNotFoundError(f"Frozen {label} reference does not exist: {reference_path}")
    reference = pd.read_csv(reference_path, dtype="string", keep_default_na=False)
    missing = sorted(set(immutable_columns) - set(reference.columns))
    if missing:
        raise ValueError(f"Frozen {label} reference is missing: {', '.join(missing)}")
    if imported[id_column].duplicated().any() or reference[id_column].duplicated().any():
        raise ValueError(f"{label} IDs must be unique")
    if set(imported[id_column]) != set(reference[id_column]):
        raise ValueError(f"Workbook {label} IDs differ from the frozen reference")
    imported_context = (
        imported.loc[:, list(immutable_columns)].sort_values(id_column).reset_index(drop=True)
    )
    reference_context = (
        reference.loc[:, list(immutable_columns)]
        .astype("string")
        .sort_values(id_column)
        .reset_index(drop=True)
    )
    if imported_context.ne(reference_context).any(axis=None):
        changed = imported_context.loc[
            imported_context.ne(reference_context).any(axis=1), id_column
        ].head(5)
        raise ValueError(
            f"Workbook changed frozen {label} content for: {', '.join(changed)}"
        )


def _write_validated(
    frame: pd.DataFrame,
    destination: Path,
    validator: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        progress = validator(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return progress


def import_annotation_workbook(
    workbook_path: str | Path,
    output_path: str | Path,
    *,
    reference_path: str | Path | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Import the Annotations tab only after checking all frozen context."""
    destination = Path(output_path)
    reference = Path(reference_path) if reference_path is not None else destination
    columns = (*ANNOTATION_CONTEXT_COLUMNS, *LABEL_COLUMNS)
    frame = _assert_exact_columns(
        read_xlsx_sheet(workbook_path, "Annotations"),
        columns,
        label="Annotation workbook",
    )
    _assert_immutable_content(
        frame,
        reference,
        id_column="case_id",
        immutable_columns=ANNOTATION_CONTEXT_COLUMNS,
        label="annotation",
    )
    progress = _write_validated(
        frame,
        destination,
        lambda path: validate_annotations(
            path, require_complete=require_complete, require_blind=True
        ),
    )
    return {
        **progress,
        "file": destination.name,
        "workbook": Path(workbook_path).name,
        "worksheet": "Annotations",
        "output": destination.name,
    }


def import_retrieval_review_workbook(
    workbook_path: str | Path,
    output_path: str | Path,
    *,
    reference_path: str | Path | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Import the blinded relevance tab only after checking immutable evidence."""
    destination = Path(output_path)
    reference = Path(reference_path) if reference_path is not None else destination
    frame = _assert_exact_columns(
        read_xlsx_sheet(workbook_path, "Relevance Review"),
        REVIEW_COLUMNS,
        label="Retrieval review workbook",
    )
    _assert_immutable_content(
        frame,
        reference,
        id_column="retrieval_review_id",
        immutable_columns=IMMUTABLE_REVIEW_COLUMNS,
        label="retrieval review",
    )
    progress = _write_validated(
        frame,
        destination,
        lambda path: validate_retrieval_review(path, require_complete=require_complete),
    )
    return {
        **progress,
        "file": destination.name,
        "workbook": Path(workbook_path).name,
        "worksheet": "Relevance Review",
        "output": destination.name,
    }
