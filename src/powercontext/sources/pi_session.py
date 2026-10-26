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

"""Capture the visible events of one Pi session branch as immutable evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
_SESSION_VERSION = 3
_CHECKPOINT_VERSION = 1
_BOM = "\ufeff"
_CONTROL_TYPES = frozenset({"compaction", "branch_summary"})


class PiSessionInput(BaseModel):
    """One Pi session archive plus the branch leaf that selects the history path."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event of a Pi session branch."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured Pi event that keeps the complete original record."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if len(value) > MAX_SOURCE_ID_LENGTH:
            raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
        return value


@dataclass(frozen=True, slots=True)
class _ParsedSession:
    """One parsed session archive with its node graph and selected history path."""

    session_id: str
    nodes: dict[str, dict[str, JsonValue]]
    chain: tuple[str, ...]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable identity of one Pi event from its session and event ids."""

    if not session_id or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")  # noqa: TRY003
    if not event_id or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")  # noqa: TRY003
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        digest = hashlib.sha256(f"{session_id}\x1f{event_id}".encode()).hexdigest()
        identity = f"pi:{digest}"
    return identity


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Return the visible events of the selected branch, ordered from root to leaf."""

    parsed = _parse_session(value.content, value.leaf_id)
    events = []
    for event_id in parsed.chain:
        event = _visible_event(parsed.session_id, event_id, parsed.nodes[event_id])
        if event is not None:
            events.append(event)
    return tuple(events)


class PiEventSourceAdapter:
    """Resolve and read the runtime's Pi event Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one Pi event as text evidence without losing its identity."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class PiSessionConnector:
    """Import the visible events of one Pi session branch exactly once."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        """Submit the events that are not imported yet and propose the next checkpoint."""

        content = self._path.read_bytes()
        parsed = _parse_session(content, self._leaf_id)
        imported = _imported_digests(session.checkpoint, parsed.session_id)
        _reject_rewritten_events(imported, parsed)

        events = [event for event in parse_pi_session(PiSessionInput(content=content, leaf_id=self._leaf_id))]
        digests = dict(imported)
        for event in events:
            if event.event_id in digests:
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            digests[event.event_id] = _record_digest(event.record)
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=_build_checkpoint(parsed.session_id, digests),
        )


def _parse_session(content: bytes, leaf_id: str | None) -> _ParsedSession:
    records = _read_records(content)
    if not records:
        raise ValueError("Pi session archive must contain a session header")  # noqa: TRY003
    header = records[0]
    if header.get("type") != "session" or header.get("version") != _SESSION_VERSION:
        raise ValueError("Pi session header must declare type=session and version=3")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Pi session header must declare a non-empty id")  # noqa: TRY003

    nodes: dict[str, dict[str, JsonValue]] = {}
    for record in records[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("Pi session nodes must declare a non-empty id")  # noqa: TRY003
        if event_id in nodes:
            raise ValueError(f"Pi session node {event_id!r} is duplicated")  # noqa: TRY003
        parent_id = record.get("parentId")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError(f"Pi session node {event_id!r} declares an invalid parent")  # noqa: TRY003
        if parent_id is not None and parent_id not in nodes:
            raise ValueError(f"Pi session node {event_id!r} references an unknown parent")  # noqa: TRY003
        nodes[event_id] = record

    if not nodes:
        raise ValueError("Pi session archive must contain at least one node")  # noqa: TRY003
    leaf = leaf_id if leaf_id is not None else next(reversed(nodes))
    if leaf not in nodes:
        raise ValueError(f"Pi session leaf {leaf!r} does not exist")  # noqa: TRY003

    chain: list[str] = []
    seen: set[str] = set()
    current: str | None = leaf
    while current is not None:
        if current in seen:
            raise ValueError("Pi session history contains a cycle")  # noqa: TRY003
        seen.add(current)
        chain.append(current)
        parent = nodes[current].get("parentId")
        current = parent if isinstance(parent, str) else None
    chain.reverse()
    return _ParsedSession(session_id=session_id, nodes=nodes, chain=tuple(chain))


def _read_records(content: bytes) -> list[dict[str, JsonValue]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session archive must be valid UTF-8") from error
    if text.startswith(_BOM):
        text = text[1:]
    records: list[dict[str, JsonValue]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pi session line {number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"Pi session line {number} must be a JSON object")  # noqa: TRY003
        records.append(record)
    return records


def _visible_event(
    session_id: str,
    event_id: str,
    record: dict[str, JsonValue],
) -> PiEventInput | None:
    event_type = record.get("type")
    if not isinstance(event_type, str) or not event_type:
        return None
    visible = _visible_body(event_type, record)
    if visible is None:
        return None
    role, body = visible
    if not body.strip():
        return None
    parent_id = record.get("parentId")
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id if isinstance(parent_id, str) else None,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _visible_body(event_type: str, record: dict[str, JsonValue]) -> tuple[str, str] | None:
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        return role, _message_body(role, message)
    if event_type in _CONTROL_TYPES:
        summary = record.get("summary")
        return event_type, summary if isinstance(summary, str) else ""
    if event_type == "custom_message":
        return "custom", _content_text(record.get("content"))
    return None


def _message_body(role: str, message: dict[str, JsonValue]) -> str:
    if role == "toolResult":
        return f"[tool_result {message.get('toolName')} {message.get('toolCallId')}] {_content_text(message.get('content'))}"
    if role == "bashExecution":
        command = message.get("command")
        command_text = command if isinstance(command, str) else ""
        output = message.get("output")
        if not isinstance(output, str):
            return f"$ {command_text}"
        return f"$ {command_text}\n{output}"
    return _content_text(message.get("content"))


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            parts.append(text if isinstance(text, str) else "")
        elif block_type == "toolCall":
            parts.append(f"[tool_call {block.get('name')} {block.get('id')}] {_compact_json(block.get('arguments'))}")
    return "\n".join(parts)


def _compact_json(arguments: Any) -> str:
    value = {} if arguments is None else arguments
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _imported_digests(checkpoint: JsonValue | None, session_id: str) -> dict[str, str]:
    if not isinstance(checkpoint, dict):
        return {}
    sessions = checkpoint.get("sessions")
    if not isinstance(sessions, dict):
        return {}
    events = sessions.get(session_id)
    if not isinstance(events, dict):
        return {}
    return {key: value for key, value in events.items() if isinstance(key, str) and isinstance(value, str)}


def _reject_rewritten_events(imported: dict[str, str], parsed: _ParsedSession) -> None:
    for event_id, digest in imported.items():
        record = parsed.nodes.get(event_id)
        if record is None:
            continue
        if _record_digest(record) != digest:
            raise ValueError(f"Pi session event {event_id!r} was rewritten after it was imported")  # noqa: TRY003


def _build_checkpoint(session_id: str, digests: dict[str, str]) -> JsonValue:
    return {"version": _CHECKPOINT_VERSION, "sessions": {session_id: dict(sorted(digests.items()))}}


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    version="1",
    projections=(PiEventTextEvidenceProjection(),),
)


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
