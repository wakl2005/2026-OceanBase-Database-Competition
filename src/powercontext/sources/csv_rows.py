# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Row-level CSV capture so one retrieval hit maps back to one original record."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"
_BOM = "\ufeff"


class CSVDocumentInput(BaseModel):
    """One offline CSV document selected by a primary key column."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One validated CSV row with its original column order and values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured CSV row that keeps the identity parts and the original fields."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if len(value) > MAX_SOURCE_ID_LENGTH:
            raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
        return value


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the delimiter-safe, bounded identity of one CSV row."""

    document = _require_document_id(document_id)
    key = _require_non_blank("primary_key_value", primary_key_value)
    identity = f"csv:{quote(document, safe='')}:{quote(key, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse one CSV document into validated rows, or fail without partial rows."""

    document_id = _require_document_id(value.document_id)
    primary_key = _require_non_blank("primary_key", value.primary_key)
    records = _read_records(value.content)

    if not records:
        raise ValueError("CSV document must contain a header row")  # noqa: TRY003
    header = records[0]
    if not header:
        raise ValueError("CSV header row must not be empty")  # noqa: TRY003
    for column in header:
        if not column.strip():
            raise ValueError("CSV header columns must not be blank")  # noqa: TRY003
    if len(set(header)) != len(header):
        raise ValueError("CSV header columns must be unique")  # noqa: TRY003
    if primary_key not in header:
        raise ValueError("primary_key must match one CSV header column exactly")  # noqa: TRY003

    key_index = header.index(primary_key)
    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(header):
            raise ValueError("CSV row width must match the header width")  # noqa: TRY003
        key_value = record[key_index]
        if not key_value.strip():
            raise ValueError("CSV primary key value must not be blank")  # noqa: TRY003
        if key_value in seen_keys:
            raise ValueError("CSV primary key values must be unique within a document")  # noqa: TRY003
        seen_keys.add(key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=key_value,
                columns=tuple(header),
                values=tuple(record),
            )
        )
    return tuple(rows)


class CSVRowSourceAdapter:
    """Resolve and read the runtime's row-level CSV Source."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one CSV row as text evidence without losing its column order."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


def _require_document_id(document_id: str) -> str:
    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty trimmed string")  # noqa: TRY003
    return document_id


def _require_non_blank(field: str, value: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")  # noqa: TRY003
    return value


def _read_records(content: bytes) -> list[list[str]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error
    if text.startswith(_BOM):
        text = text[1:]
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=",", quotechar='"', doublequote=True, strict=True)
    try:
        return [list(record) for record in reader]
    except csv.Error as error:
        raise ValueError(f"CSV content is malformed: {error}") from error


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSVRowSourceAdapter(),
    version=CSV_ROW_SOURCE_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
