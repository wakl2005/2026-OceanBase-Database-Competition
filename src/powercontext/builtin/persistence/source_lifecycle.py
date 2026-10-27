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

"""Persistence for the registered Source evidence graph and its event history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENT_TABLE,
    SOURCE_LIFECYCLE_NODE_TABLE,
)
from powercontext.limits import MAX_SCOPE_ID_LENGTH


@dataclass(frozen=True, slots=True)
class SourceLifecycleEventRecord:
    """One recorded lifecycle event."""

    sequence: int
    revision: int
    available: bool
    reason: str


class SourceLifecycleRepository:
    """Read and write the evidence graph of one Scope."""

    async def nodes(
        self,
        connection: AsyncConnection,
        scope_id: str,
        /,
    ) -> dict[str, Any]:
        """Return every registered node of one Scope, keyed by its node id."""

        _require_scope_id(scope_id)
        statement = (
            select(SOURCE_LIFECYCLE_NODE_TABLE)
            .where(SOURCE_LIFECYCLE_NODE_TABLE.c.scope_id == scope_id)
            .order_by(SOURCE_LIFECYCLE_NODE_TABLE.c.node_id)
        )
        rows = (await connection.execute(statement)).mappings().all()
        return {str(row["node_id"]): _decode_node(row) for row in rows}

    async def insert_node(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node: Any,
        /,
    ) -> None:
        """Store one registered node."""

        _require_scope_id(scope_id)
        await connection.execute(
            SOURCE_LIFECYCLE_NODE_TABLE.insert().values(
                scope_id=scope_id,
                node_id=node.node_id,
                kind=node.kind,
                parents=json.dumps(list(node.parents)),
                policy=node.policy,
                reference=json.dumps(_encode_reference(node.reference)),
                revision=node.revision,
                available=node.available,
                created_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )

    async def set_available(
        self,
        connection: AsyncConnection,
        scope_id: str,
        availability: dict[str, bool],
        /,
    ) -> None:
        """Store the computed availability of every registered node."""

        _require_scope_id(scope_id)
        for node_id, available in availability.items():
            await connection.execute(
                update(SOURCE_LIFECYCLE_NODE_TABLE)
                .where(
                    SOURCE_LIFECYCLE_NODE_TABLE.c.scope_id == scope_id,
                    SOURCE_LIFECYCLE_NODE_TABLE.c.node_id == node_id,
                )
                .values(available=available)
            )

    async def event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        sequence: int,
        /,
    ) -> SourceLifecycleEventRecord | None:
        """Return one recorded event, if it exists."""

        _require_scope_id(scope_id)
        statement = select(SOURCE_LIFECYCLE_EVENT_TABLE).where(
            SOURCE_LIFECYCLE_EVENT_TABLE.c.scope_id == scope_id,
            SOURCE_LIFECYCLE_EVENT_TABLE.c.node_id == node_id,
            SOURCE_LIFECYCLE_EVENT_TABLE.c.sequence == sequence,
        )
        row = (await connection.execute(statement)).mappings().one_or_none()
        if row is None:
            return None
        return SourceLifecycleEventRecord(
            sequence=int(row["sequence"]),
            revision=int(row["revision"]),
            available=bool(row["available"]),
            reason=str(row["reason"]),
        )

    async def latest_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
    ) -> SourceLifecycleEventRecord | None:
        """Return the highest recorded sequence of one node."""

        _require_scope_id(scope_id)
        statement = (
            select(SOURCE_LIFECYCLE_EVENT_TABLE)
            .where(
                SOURCE_LIFECYCLE_EVENT_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_EVENT_TABLE.c.node_id == node_id,
            )
            .order_by(SOURCE_LIFECYCLE_EVENT_TABLE.c.sequence.desc())
            .limit(1)
        )
        row = (await connection.execute(statement)).mappings().one_or_none()
        if row is None:
            return None
        return SourceLifecycleEventRecord(
            sequence=int(row["sequence"]),
            revision=int(row["revision"]),
            available=bool(row["available"]),
            reason=str(row["reason"]),
        )

    async def record_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        event: Any,
        /,
    ) -> None:
        """Store one applied lifecycle event."""

        _require_scope_id(scope_id)
        await connection.execute(
            SOURCE_LIFECYCLE_EVENT_TABLE.insert().values(
                scope_id=scope_id,
                node_id=event.node_id,
                sequence=event.sequence,
                revision=event.revision,
                available=event.available,
                reason=event.reason,
                applied_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )


def _decode_node(row: Any) -> Any:
    from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleNode

    reference = _decode_reference(row["reference"])
    return SourceLifecycleNode(
        node_id=str(row["node_id"]),
        kind=str(row["kind"]),
        parents=tuple(json.loads(row["parents"])),
        policy=str(row["policy"]),
        reference=reference,
        revision=int(row["revision"]),
        available=bool(row["available"]),
    )


def _encode_reference(reference: Any) -> dict[str, Any]:
    from powercontext.artifacts import ArtifactRef
    from powercontext.builtin.artifacts.memory.models import MemoryCitation

    if isinstance(reference, MemoryCitation):
        return {
            "kind": "memory",
            "memory_ref": reference.memory_ref.model_dump(mode="json"),
            "entry_id": reference.entry_id,
            "entry_version_id": reference.entry_version_id,
        }
    if isinstance(reference, ArtifactRef):
        return {"kind": "topic", "artifact_ref": reference.model_dump(mode="json")}
    return {
        "kind": "source",
        "source_type": reference.source_type,
        "source_id": reference.source_id,
        "revision": getattr(reference, "revision", None),
    }


def _decode_reference(payload: str) -> Any:
    from powercontext.artifacts import ArtifactRef
    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.sources.models import SourceRef

    data = json.loads(payload)
    kind = data.get("kind")
    if kind == "memory":
        return MemoryCitation(
            memory_ref=ArtifactRef.model_validate(data["memory_ref"]),
            entry_id=data["entry_id"],
            entry_version_id=data["entry_version_id"],
        )
    if kind == "topic":
        return ArtifactRef.model_validate(data["artifact_ref"])
    return SourceRef(source_type=data["source_type"], source_id=data["source_id"])


def _require_scope_id(scope_id: str) -> None:
    if not isinstance(scope_id, str) or not scope_id.strip() or len(scope_id) > MAX_SCOPE_ID_LENGTH:
        raise ValueError("scope-id")


__all__ = ["SourceLifecycleEventRecord", "SourceLifecycleRepository"]
