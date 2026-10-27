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

"""Capture every physical page of a PDF report as page-level evidence."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from typing import Any, Protocol
from urllib.parse import quote

from pydantic import BaseModel, field_validator
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

PDF_PAGE_SOURCE_NAME = "pdf-page"
_TERM_PATTERN = re.compile(r"[a-z]+|\d+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "give",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "of",
        "on",
        "or",
        "our",
        "please",
        "respond",
        "show",
        "so",
        "some",
        "state",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "use",
        "using",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "capex": ("capital", "expenditures", "spending", "property"),
    "ebitda": ("operating", "income", "depreciation", "amortization"),
    "geographies": ("geographic", "regions", "segments", "countries"),
    "geography": ("geographic", "regions", "segments", "countries"),
    "legal": ("litigation", "proceedings", "lawsuits", "litigation"),
    "restructuring": ("restructuring", "severance", "charges"),
    "revenue": ("revenues", "sales", "net"),
    "employees": ("employees", "workforce", "personnel"),
    "dividends": ("dividends", "dividend"),
    "margin": ("margin", "percentages"),
    "profitability": ("income", "profit", "margin"),
}


class PDFDocumentInput(BaseModel):
    """One complete PDF document that is parsed into physical pages."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page of a PDF document."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured PDF page that keeps its document digest and physical page number."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if len(value) > MAX_SOURCE_ID_LENGTH:
            raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
        return value


class _ScopedMemory(Protocol):
    """The scoped Memory surface used to retrieve captured pages."""

    async def search(self, request: Any, /) -> Any: ...

    async def list(self, **kwargs: Any) -> Any: ...


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable identity of one physical page of one document revision."""

    identity = f"{PDF_PAGE_SOURCE_NAME}:{quote(document_id, safe='')}:{page_number}:{document_sha256}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:32]
        identity = f"{PDF_PAGE_SOURCE_NAME}:{digest}:{page_number}:{document_sha256[:32]}"
    return identity


def pdf_page_number_of(source_id: str) -> int | None:
    """Recover the physical page number from a page identity."""

    parts = source_id.split(":")
    if len(parts) != 4 or parts[0] != PDF_PAGE_SOURCE_NAME:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one complete PDF document."""

    document_id = value.document_id
    if not document_id or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    digest = hashlib.sha256(value.content).hexdigest()
    texts = _extract_pages(value.content)
    return tuple(
        PDFPageInput(
            document_id=document_id,
            document_sha256=digest,
            page_number=number,
            text=text,
        )
        for number, text in enumerate(texts, start=1)
    )


class PDFPageSourceAdapter:
    """Resolve and read the runtime's PDF page Source."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one PDF page as text evidence with its document identity."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


async def search_pdf_pages(
    memory: _ScopedMemory,
    question: str,
    limit: int = 10,
) -> tuple[Any, ...]:
    """Return the exact Memory citations of the pages that best answer a question."""

    if limit < 1:
        return ()
    candidates = await _pdf_page_candidates(memory, question)
    if not candidates:
        return ()
    ranked = _rank_pages(candidates, question)
    selected: list[Any] = []
    seen: set[tuple[str, int]] = set()
    for citation, page_key in ranked:
        if page_key in seen:
            continue
        seen.add(page_key)
        selected.append(citation)
        if len(selected) >= limit:
            break
    return tuple(selected)


async def _pdf_page_candidates(memory: _ScopedMemory, question: str) -> list[tuple[Any, tuple[str, int], str]]:
    """Collect the PDF pages reachable through the public Memory API."""

    candidates: list[tuple[Any, tuple[str, int], str]] = []
    listed = getattr(memory, "list", None)
    if callable(listed):
        page = await listed()
        for record in getattr(page, "entries", ()):
            entry = getattr(record, "entry", record)
            sources = getattr(entry, "sources", ())
            page_ref = next((source for source in sources if _page_number(source) is not None), None)
            if page_ref is None:
                continue
            citation = getattr(record, "citation", None)
            if citation is None:
                citation = _citation_of(record)
            if citation is None:
                continue
            candidates.append((citation, _page_key(page_ref), getattr(entry, "text", "")))
    if candidates:
        return candidates

    from powercontext.builtin.runtime import SearchMemoryRequest

    result = await memory.search(
        SearchMemoryRequest(query=question, mode="fts", limit=max(limit * 10, 100))
    )
    for hit in getattr(result, "hits", ()):
        citation = _citation_of(hit)
        if citation is None:
            continue
        candidates.append((citation, (getattr(hit, "entry_id", ""), 0), getattr(hit, "text", "")))
    return candidates


def _rank_pages(
    candidates: list[tuple[Any, tuple[str, int], str]],
    question: str,
) -> list[tuple[Any, tuple[str, int]]]:
    """Rank candidate pages by offline query coverage of the question."""

    terms = _query_terms(question)
    if not terms:
        return [(citation, key) for citation, key, _ in candidates]
    documents = [_terms(text) for _, _, text in candidates]
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(set(document))
    total = len(documents)
    weights = {term: _idf(frequencies.get(term, 0), total) for term in terms}
    scored: list[tuple[float, Any, tuple[str, int]]] = []
    for (citation, key, _), document in zip(candidates, documents, strict=True):
        scored.append((_score(document, terms, weights), citation, key))
    scored.sort(key=lambda item: (-item[0], item[2]))
    return [(citation, key) for _, citation, key in scored]


def _query_terms(question: str) -> tuple[str, ...]:
    tokens = _TERM_PATTERN.findall(question.lower())
    terms = {token for token in tokens if token not in _STOPWORDS}
    for token in tuple(terms):
        terms.update(_SYNONYMS.get(token, ()))
    return tuple(sorted(terms))


def _terms(text: str) -> list[str]:
    return _TERM_PATTERN.findall(text.lower())


def _idf(document_frequency: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return math.log((total + 1) / (document_frequency + 1)) + 1.0


def _score(document: list[str], terms: tuple[str, ...], weights: dict[str, float]) -> float:
    if not document:
        return 0.0
    counts = Counter(document)
    available = sum(weight for term, weight in weights.items() if counts.get(term))
    total = sum(weights.values())
    if not total:
        return 0.0
    coverage = available / total
    density = sum(weights[term] for term in terms if counts.get(term))
    return coverage + 0.05 * density / total


def _citation_of(record: Any) -> Any:
    citation = getattr(record, "citation", None)
    if citation is not None:
        return citation
    memory_ref = getattr(record, "memory_ref", None)
    entry_id = getattr(record, "entry_id", None)
    entry_version_id = getattr(record, "entry_version_id", None)
    if memory_ref is None or entry_id is None or entry_version_id is None:
        return None
    from powercontext.builtin.artifacts.memory.models import MemoryCitation

    return MemoryCitation(
        memory_ref=memory_ref,
        entry_id=entry_id,
        entry_version_id=entry_version_id,
    )


def _page_number(source: Any) -> int | None:
    source_id = getattr(source, "source_id", None)
    if not isinstance(source_id, str):
        return None
    if getattr(source, "source_type", None) != PDF_PAGE_SOURCE_NAME:
        return None
    return pdf_page_number_of(source_id)


def _page_key(source: Any) -> tuple[str, int]:
    return (getattr(source, "source_id", ""), _page_number(source) or 0)


def _extract_pages(content: bytes) -> list[str]:
    if not content:
        raise ValueError("PDF document must not be empty")  # noqa: TRY003
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise ValueError("PDF document requires a password")  # noqa: TRY003
        return [(page.extract_text() or "") for page in reader.pages]
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"PDF document could not be parsed: {error}") from error


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    version="1",
    projections=(PDFPageTextEvidenceProjection(),),
)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
