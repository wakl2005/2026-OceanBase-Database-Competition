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

"""Source withdrawal and restore propagated over the evidence dependency graph.

A Scope registers the Sources it captured, the Memory entries and Topic Memory
revisions that depend on them, and the policy that connects them. A lifecycle
event withdraws or restores one Source; the change is then propagated to every
descendant, so the native Memory and Topic Memory surfaces stop serving the
content that lost its evidence and start serving it again once the evidence
returns. Deactivating never rewrites a body, so an exact historical reference
keeps resolving to the version it named.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.persistence.source_lifecycle import SourceLifecycleRepository
from powercontext.builtin.runtime.errors import InvalidRuntimeRequestError
from powercontext.limits import MAX_SCOPE_ID_LENGTH, MAX_SOURCE_ID_LENGTH

MAX_LIFECYCLE_REASON_LENGTH = 2_000

SourceLifecycleKind: Literal["source", "memory", "topic"] = "source"
SourceLifecyclePolicy: Literal["all", "any"] = "all"
SourceLifecyclePurpose: Literal["generation", "audit"] = "generation"


class SourceLifecycleError(ValueError):
    """Raised when one lifecycle request contradicts the registered graph."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Source lifecycle error: {code}")


class SourceLifecycleEvent(BaseModel):
    """One withdrawal or restore of a registered Source."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    node_id: Annotated[str, Field(min_length=1, max_length=MAX_SOURCE_ID_LENGTH)]
    reason: Annotated[str, Field(min_length=1, max_length=MAX_LIFECYCLE_REASON_LENGTH)]
    revision: Annotated[StrictInt, Field(gt=0)]
    sequence: Annotated[StrictInt, Field(gt=0)]
    available: StrictBool

    @field_validator("node_id")
    @classmethod
    def reject_blank_node_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("node_id must not be blank")  # noqa: TRY003
        return value

    @field_validator("reason")
    @classmethod
    def reject_blank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")  # noqa: TRY003
        return value


@dataclass(frozen=True, slots=True)
class SourceLifecycleNode:
    """One registered node of the Scope evidence graph."""

    node_id: str
    kind: str
    parents: tuple[str, ...]
    policy: str
    reference: Any
    revision: int
    available: bool


class SourceLifecycleService:
    """Register the evidence graph of one Scope and propagate its events.

    The service owns no storage of its own: the registered graph and the event
    history live in the Scope database, so a restarted Runtime observes the same
    graph and the same availability.
    """

    def __init__(self, runtime: Any, scope_id: str, repository: SourceLifecycleRepository) -> None:
        self._runtime = runtime
        self.scope_id = scope_id
        self._repository = repository

    # -- registration ---------------------------------------------------

    async def register_source(
        self,
        node_id: str,
        source_ref: Any,
        /,
        *,
        revision: int = 1,
    ) -> None:
        """Register one captured Source as the root of the evidence graph."""

        node = await self._source_node(node_id, source_ref, revision=revision)
        await self._register(node)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: tuple[str, ...],
        policy: str = "all",
    ) -> None:
        """Register one Memory entry and the parents that support its body."""

        node = await self._memory_node(node_id, citation, parents=parents, policy=policy)
        await self._register(node)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: tuple[str, ...],
        policy: str = "all",
    ) -> None:
        """Register one Topic Memory revision and the parents that support it."""

        node = await self._topic_node(node_id, reference, parents=parents, policy=policy)
        await self._register(node)

    # -- events ---------------------------------------------------------

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one lifecycle event and return every currently available node."""

        async with self._runtime._lifecycle_transaction(self.scope_id) as connection:
            nodes = await self._repository.nodes(connection, self.scope_id)
            source = nodes.get(event.node_id)
            if source is None or source.kind != "source":
                raise SourceLifecycleError("unknown-source")
            if source.revision != event.revision:
                raise SourceLifecycleError("revision-mismatch")
            recorded = await self._repository.event(connection, self.scope_id, event.node_id, event.sequence)
            if recorded is not None:
                # The same sequence was already applied. Replaying the very same
                # event is idempotent; a different event under the same sequence
                # contradicts the recorded one.
                if (
                    recorded.revision != event.revision
                    or recorded.available != event.available
                    or recorded.reason != event.reason
                ):
                    raise SourceLifecycleError("sequence-conflict")
            else:
                latest = await self._repository.latest_event(connection, self.scope_id, event.node_id)
                if latest is not None:
                    if event.sequence < latest.sequence:
                        raise SourceLifecycleError("sequence-regression")
                    if event.sequence == latest.sequence:
                        raise SourceLifecycleError("sequence-conflict")
                await self._repository.record_event(connection, self.scope_id, event)
            active = _propagate(nodes, event.node_id, event.available)
            await self._repository.set_available(
                connection,
                self.scope_id,
                {node_id: node_id in active for node_id in nodes},
            )
        await self._propagate_to_runtime(active)
        return sorted(active)

    # -- resolution -----------------------------------------------------

    async def resolve(self, node_id: str, /, *, purpose: str = "generation") -> Any:
        """Return the original reference of one node for the requested purpose."""

        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("unknown-purpose")
        async with self._runtime._lifecycle_transaction(self.scope_id) as connection:
            nodes = await self._repository.nodes(connection, self.scope_id)
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("unavailable-node")
        return node.reference

    async def active_nodes(self) -> list[str]:
        """Return every currently available node of this Scope."""

        async with self._runtime._lifecycle_transaction(self.scope_id) as connection:
            nodes = await self._repository.nodes(connection, self.scope_id)
        return sorted(node_id for node_id, node in nodes.items() if node.available)

    async def inactive_topic_references(self) -> frozenset[str]:
        """Return the artifact ids of the Topic Memory nodes without support."""

        async with self._runtime._lifecycle_transaction(self.scope_id) as connection:
            nodes = await self._repository.nodes(connection, self.scope_id)
        return frozenset(
            str(node.reference.artifact_id)
            for node in nodes.values()
            if node.kind == "topic" and not node.available
        )

    # -- internals ------------------------------------------------------

    async def _register(self, node: SourceLifecycleNode) -> None:
        async with self._runtime._lifecycle_transaction(self.scope_id) as connection:
            nodes = await self._repository.nodes(connection, self.scope_id)
            existing = nodes.get(node.node_id)
            if existing is not None:
                if _same_binding(existing, node):
                    return
                raise SourceLifecycleError("conflicting-registration")
            if node.kind != "source":
                if not node.parents:
                    raise SourceLifecycleError("missing-parents")
                if len(set(node.parents)) != len(node.parents):
                    raise SourceLifecycleError("duplicate-parent")
                for parent in node.parents:
                    if parent == node.node_id:
                        raise SourceLifecycleError("self-dependency")
                    if parent not in nodes:
                        raise SourceLifecycleError("unknown-parent")
                if _reaches(nodes, node.parents, node.node_id):
                    raise SourceLifecycleError("dependency-cycle")
            active = _propagate({**nodes, node.node_id: node}, node.node_id, node.available)
            await self._repository.insert_node(connection, self.scope_id, node)
            await self._repository.set_available(
                connection,
                self.scope_id,
                {node_id: node_id in active for node_id in {**nodes, node.node_id: node}},
            )
        await self._propagate_to_runtime(active)

    async def _source_node(self, node_id: str, source_ref: Any, *, revision: int) -> SourceLifecycleNode:
        _require_node_id(node_id)
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("invalid-revision")
        await self._runtime._require_source_reference(self.scope_id, source_ref)
        return SourceLifecycleNode(
            node_id=node_id,
            kind="source",
            parents=(),
            policy="all",
            reference=source_ref,
            revision=revision,
            available=True,
        )

    async def _memory_node(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> SourceLifecycleNode:
        _require_node_id(node_id)
        _require_policy(policy)
        _require_parents(parents)
        await self._runtime._require_memory_citation(self.scope_id, citation)
        return SourceLifecycleNode(
            node_id=node_id,
            kind="memory",
            parents=tuple(parents),
            policy=policy,
            reference=citation,
            revision=1,
            available=False,
        )

    async def _topic_node(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> SourceLifecycleNode:
        _require_node_id(node_id)
        _require_policy(policy)
        _require_parents(parents)
        await self._runtime._require_topic_reference(self.scope_id, reference)
        return SourceLifecycleNode(
            node_id=node_id,
            kind="topic",
            parents=tuple(parents),
            policy=policy,
            reference=reference,
            revision=reference.revision,
            available=False,
        )

    async def _propagate_to_runtime(self, active: set[str]) -> None:
        await self._runtime._apply_lifecycle_availability(self.scope_id, active)


class SourceLifecycleApplication:
    """Expose the Source lifecycle service of one Scope."""

    def __init__(self, runtime: Any, repository: SourceLifecycleRepository | None = None) -> None:
        self._runtime = runtime
        self._repository = SourceLifecycleRepository() if repository is None else repository

    def for_scope(self, scope_id: str) -> SourceLifecycleService:
        """Return the lifecycle service of one Scope."""

        if not scope_id or len(scope_id) > MAX_SCOPE_ID_LENGTH or scope_id != scope_id.strip():
            raise InvalidRuntimeRequestError("scope-id")
        return SourceLifecycleService(self._runtime, scope_id, self._repository)


def _require_node_id(node_id: str) -> None:
    if not isinstance(node_id, str) or not node_id.strip() or len(node_id) > MAX_SOURCE_ID_LENGTH:
        raise SourceLifecycleError("invalid-node-id")


def _require_policy(policy: str) -> None:
    if policy not in {"all", "any"}:
        raise SourceLifecycleError("invalid-policy")


def _require_parents(parents: tuple[str, ...]) -> None:
    if not parents:
        raise SourceLifecycleError("missing-parents")
    for parent in parents:
        if not isinstance(parent, str) or not parent.strip():
            raise SourceLifecycleError("invalid-parent")


def _same_binding(left: SourceLifecycleNode, right: SourceLifecycleNode) -> bool:
    if left.kind != right.kind or left.policy != right.policy or left.parents != right.parents:
        return False
    if left.kind == "memory":
        return _citation_key(left.reference) == _citation_key(right.reference)
    if left.kind == "topic":
        return left.reference == right.reference
    return _source_key(left.reference) == _source_key(right.reference)


def _citation_key(citation: Any) -> tuple[str, str, str]:
    return (citation.memory_ref.artifact_id, citation.entry_id, citation.entry_version_id)


def _source_key(source_ref: Any) -> tuple[str, str]:
    return (source_ref.source_type, source_ref.source_id)


def _reaches(nodes: dict[str, SourceLifecycleNode], parents: tuple[str, ...], target: str) -> bool:
    """Report whether any parent already depends on the candidate node."""

    pending = list(parents)
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        node = nodes.get(current)
        if node is not None:
            pending.extend(node.parents)
    return False


def _propagate(nodes: dict[str, SourceLifecycleNode], source_id: str, available: bool) -> set[str]:
    """Compute the available nodes after one Source changed state.

    A derived node stays available while its policy holds over its parents. The
    graph is acyclic by construction, so the nodes are evaluated in dependency
    order; the stored order of the graph never decides the outcome.
    """

    state: dict[str, bool] = {}
    for node_id, node in nodes.items():
        if node.kind == "source":
            state[node_id] = available if node_id == source_id else node.available
    for node_id, node in _dependency_order(nodes).items():
        if node.kind == "source":
            continue
        support = [state.get(parent, False) for parent in node.parents]
        if not support:
            state[node_id] = False
            continue
        state[node_id] = all(support) if node.policy == "all" else any(support)
    return {node_id for node_id, is_available in state.items() if is_available}


def _dependency_order(nodes: dict[str, SourceLifecycleNode]) -> dict[str, SourceLifecycleNode]:
    """Order the derived nodes so every parent is evaluated before its child."""

    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    pending: dict[str, int] = {}
    for node_id, node in nodes.items():
        parents = [parent for parent in node.parents if parent in nodes]
        pending[node_id] = len(parents)
        for parent in parents:
            children[parent].append(node_id)
    ready = [node_id for node_id, count in pending.items() if count == 0]
    ordered: dict[str, SourceLifecycleNode] = {}
    while ready:
        node_id = ready.pop()
        ordered[node_id] = nodes[node_id]
        for child in children[node_id]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    for node_id, node in nodes.items():
        # A cycle cannot be registered; the remaining nodes are still evaluated
        # so a corrupted graph never leaves a node without a state.
        ordered.setdefault(node_id, node)
    return ordered


__all__ = [
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleNode",
    "SourceLifecycleService",
]
