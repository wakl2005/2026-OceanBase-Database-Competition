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

"""Coalesce concurrent embedding requests and cache their successful vectors."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from powercontext.builtin.inference.models import EmbeddingResult
from powercontext.builtin.inference.usage import InferenceUsage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
    from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True)
class _Slot:
    """One shared text whose vector is produced by exactly one provider call."""

    text: str
    future: asyncio.Future
    batch: _Batch | None = None


@dataclass(slots=True)
class _Batch:
    """One provider call that several callers may wait for."""

    texts: tuple[str, ...]
    slots: tuple[_Slot, ...]
    usage: InferenceUsage = field(default_factory=lambda: InferenceUsage(requests=0))


class CoalescingEmbeddingModel:
    """Share one embedding provider across concurrent callers.

    Callers that ask for the same text inside the coalescing window wait for one
    provider call, and successful vectors are reused from a bounded LRU cache.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must not be negative")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._pending: dict[str, _Slot] = {}
        self._inflight: dict[str, _Slot] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def profile(self) -> EmbeddingProfile:
        """Expose the wrapped deployment profile unchanged."""

        return self._model.profile

    async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
        """Embed one ordered text batch, sharing work with concurrent callers."""

        values = tuple(texts)
        if not values:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        vectors: dict[str, tuple[float, ...]] = {}
        slots: list[_Slot] = []
        for text in dict.fromkeys(values):
            cached = self._cached(text)
            if cached is not None:
                vectors[text] = cached
            else:
                slots.append(self._slot(text))
        if not slots:
            return EmbeddingResult(
                vectors=tuple(vectors[text] for text in values),
                usage=InferenceUsage(requests=0),
            )
        if self._coalesce_seconds:
            await asyncio.sleep(self._coalesce_seconds)
        else:
            await asyncio.sleep(0)
        self._drain()
        results = await asyncio.gather(*(asyncio.shield(slot.future) for slot in slots))
        for slot, vector in zip(slots, results, strict=True):
            vectors[slot.text] = vector
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in values),
            usage=self._usage(slots),
        )

    def _slot(self, text: str) -> _Slot:
        """Return the shared slot of one text, creating it when needed."""

        inflight = self._inflight.get(text)
        if inflight is not None:
            return inflight
        slot = self._pending.get(text)
        if slot is None:
            slot = _Slot(text=text, future=asyncio.get_running_loop().create_future())
            slot.future.add_done_callback(_consume_exception)
            self._pending[text] = slot
        return slot

    def _drain(self) -> None:
        """Send every pending text once, in bounded provider batches."""

        if not self._pending:
            return
        items = tuple(self._pending.items())
        self._pending.clear()
        for start in range(0, len(items), self._max_batch_size):
            chunk = items[start : start + self._max_batch_size]
            batch = _Batch(texts=tuple(text for text, _ in chunk), slots=tuple(slot for _, slot in chunk))
            for text, slot in chunk:
                slot.batch = batch
                self._inflight[text] = slot
            task = asyncio.create_task(self._run_batch(batch))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_batch(self, batch: _Batch) -> None:
        """Run one provider call and publish its vectors to every waiter."""

        try:
            result = await self._model.embed(batch.texts)
            vectors = tuple(result.vectors)
            if len(vectors) != len(batch.texts):
                raise ValueError("embedding model returned a different number of vectors than texts")  # noqa: TRY003
            batch.usage = result.usage
            for text, vector in zip(batch.texts, vectors, strict=True):
                self._store(text, vector)
            for slot, vector in zip(batch.slots, vectors, strict=True):
                if not slot.future.done():
                    slot.future.set_result(vector)
        except BaseException as error:
            for slot in batch.slots:
                if not slot.future.done():
                    slot.future.set_exception(error)
        finally:
            for text in batch.texts:
                self._inflight.pop(text, None)

    def _cached(self, text: str) -> tuple[float, ...] | None:
        if self._cache_size <= 0:
            return None
        vector = self._cache.get(text)
        if vector is None:
            return None
        self._cache.move_to_end(text)
        return vector

    def _store(self, text: str, vector: tuple[float, ...]) -> None:
        if self._cache_size <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _usage(self, slots: list[_Slot]) -> InferenceUsage:
        """Aggregate the provider usage of every distinct batch this call waited on."""

        batches: list[_Batch] = []
        seen: set[int] = set()
        for slot in slots:
            batch = slot.batch
            if batch is None or id(batch) in seen:
                continue
            seen.add(id(batch))
            batches.append(batch)
        return InferenceUsage(
            requests=sum(batch.usage.requests for batch in batches),
            input_tokens=_sum_optional(batch.usage.input_tokens for batch in batches),
            output_tokens=_sum_optional(batch.usage.output_tokens for batch in batches),
        )


def _sum_optional(values: object) -> int | None:
    numbers = [value for value in values if isinstance(value, int)]  # type: ignore[union-attr]
    return sum(numbers) if numbers else None


def _consume_exception(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


__all__ = ["CoalescingEmbeddingModel"]
