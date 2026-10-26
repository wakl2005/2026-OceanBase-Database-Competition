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

"""Score coarse Memory candidates with one external HTTP scoring service."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANKER_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Rerank Memory candidates with a batched HTTP scoring service."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANKER_POLICY_ID,
    ) -> None:
        if not endpoint or not endpoint.strip():
            raise ValueError("reranker endpoint must be a non-empty string")  # noqa: TRY003
        if batch_size < 1:
            raise ValueError("reranker batch size must be positive")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate in batches and select the globally best ranks."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = tuple(range(start + 1, start + len(batch) + 1))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text}
                    for rank, candidate in zip(ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            if not 200 <= response.status_code < 300:
                raise httpx.HTTPStatusError(
                    f"reranker service returned HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            requests += 1
            scores.update(_parse_scores(response, set(ranks)))
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        selected = tuple(rank for rank, _ in ranked)
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=requests),
            discarded_rank_count=len(candidates) - len(selected),
        )


def _parse_scores(response: httpx.Response, expected_ranks: set[int]) -> dict[int, float]:
    """Validate one batch response and map its scores back to the global ranks."""

    try:
        payload: Any = response.json()
    except ValueError as error:
        raise ValueError("reranker service returned invalid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("reranker response must contain exactly a scores array")  # noqa: TRY003
    raw_scores = payload["scores"]
    if not isinstance(raw_scores, list):
        raise ValueError("reranker scores must be an array")  # noqa: TRY003
    scores: dict[int, float] = {}
    for item in raw_scores:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("each reranker score must contain exactly rank and score")  # noqa: TRY003
        rank = item["rank"]
        score = item["score"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank not in expected_ranks:
            raise ValueError("reranker score rank must identify one requested candidate")  # noqa: TRY003
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("reranker score must be a number")  # noqa: TRY003
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("reranker score must be finite")  # noqa: TRY003
        if rank in scores:
            raise ValueError("reranker score rank must appear once per batch")  # noqa: TRY003
        scores[rank] = value
    if set(scores) != expected_ranks:
        raise ValueError("reranker response must cover every requested rank exactly once")  # noqa: TRY003
    return scores


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


__all__ = ["DEFAULT_HTTP_RERANKER_POLICY_ID", "HTTPMemoryReranker"]
