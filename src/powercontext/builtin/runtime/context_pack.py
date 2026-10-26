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

"""Context Pack wire protocol for incremental context transfer.

A pack carries the prepared context as deterministic chunks. The first transfer
sends the whole pack; later transfers send a delta that keeps a prefix of the
previous pack and appends the rest, so a receiver can rebuild the current
context from the pack it already holds.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CONTEXT_CHUNK_BYTES = 512


class ContextPackError(ValueError):
    """Protocol or chunk validation error."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete chunked context payload."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int = 0

    def __post_init__(self) -> None:
        items = tuple(self.items)
        for item in items:
            if not isinstance(item, bytes):
                raise ContextPackError("context pack items must be bytes")  # noqa: TRY003
        _validate_digest(self.digest)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "byte_count", _payload_byte_count({**_pack_payload(items), "digest": self.digest}))

    @classmethod
    def build(cls, items: Iterable[bytes]) -> ContextPack:
        """Build one pack from already chunked content."""

        values = tuple(items)
        for item in values:
            if not isinstance(item, bytes):
                raise ContextPackError("context pack items must be bytes")  # noqa: TRY003
        return cls(items=values, digest=_pack_digest(values))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CONTEXT_CHUNK_BYTES) -> ContextPack:
        """Chunk one prepared context deterministically."""

        if not isinstance(content, str):
            raise ContextPackError("context pack content must be text")  # noqa: TRY003
        return cls.build(_chunk_content(content, chunk_bytes))

    def to_bytes(self) -> bytes:
        """Serialize the pack into its canonical wire form."""

        payload = {**_pack_payload(self.items), "digest": self.digest, "byte_count": self.byte_count}
        return _canonical(payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and validate one serialized pack."""

        payload = _decode_object(data)
        if set(payload) != {"schema", "items", "digest", "byte_count"}:
            raise ContextPackError(  # noqa: TRY003
                "context pack payload must contain exactly schema, items, digest and byte_count"
            )
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("unsupported context pack schema")  # noqa: TRY003
        items = _decode_items(payload["items"])
        digest = payload["digest"]
        if not isinstance(digest, str) or digest != _pack_digest(items):
            raise ContextPackError("context pack digest does not match its items")  # noqa: TRY003
        byte_count = payload["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int):
            raise ContextPackError("context pack byte_count must be an integer")  # noqa: TRY003
        encoded = _canonical({**_pack_payload(items), "digest": digest, "byte_count": byte_count})
        if len(encoded) != byte_count:
            raise ContextPackError("context pack byte_count does not match its encoding")  # noqa: TRY003
        return cls(items=items, digest=digest)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """One incremental update that reuses a prefix of the previous pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        appended = tuple(self.appended)
        for item in appended:
            if not isinstance(item, bytes):
                raise ContextPackError("context pack delta items must be bytes")  # noqa: TRY003
        _validate_digest(self.base_digest)
        _validate_digest(self.target_digest)
        if isinstance(self.prefix_items, bool) or not isinstance(self.prefix_items, int):
            raise ContextPackError("context pack delta prefix_items must be an integer")  # noqa: TRY003
        if self.prefix_items < 0:
            raise ContextPackError("context pack delta prefix_items must not be negative")  # noqa: TRY003
        object.__setattr__(self, "appended", appended)

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Describe how the target pack extends the base pack."""

        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    def to_bytes(self) -> bytes:
        """Serialize the delta into its canonical wire form."""

        payload = {
            "schema": CONTEXT_PACK_DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [_encode_item(item) for item in self.appended],
        }
        return _canonical(payload)

    @property
    def byte_count(self) -> int:
        """The exact wire size of this delta."""

        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and validate one serialized delta."""

        payload = _decode_object(data)
        expected = {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
        if set(payload) != expected:
            raise ContextPackError("context pack delta payload does not match the schema")  # noqa: TRY003
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("unsupported context pack delta schema")  # noqa: TRY003
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("context pack delta digests must be text")  # noqa: TRY003
        prefix_items = payload["prefix_items"]
        if isinstance(prefix_items, bool) or not isinstance(prefix_items, int):
            raise ContextPackError("context pack delta prefix_items must be an integer")  # noqa: TRY003
        if prefix_items < 0:
            raise ContextPackError("context pack delta prefix_items must not be negative")  # noqa: TRY003
        appended = _decode_items(payload["appended"])
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from the base pack and one delta."""

    if base.digest != delta.base_digest:
        raise ContextPackError("context pack delta does not apply to this base pack")  # noqa: TRY003
    if delta.prefix_items > len(base.items):
        raise ContextPackError("context pack delta prefix exceeds the base pack")  # noqa: TRY003
    target = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("context pack delta does not rebuild the target pack")  # noqa: TRY003
    return target


def _pack_payload(items: tuple[bytes, ...]) -> dict[str, object]:
    return {"schema": CONTEXT_PACK_SCHEMA, "items": [_encode_item(item) for item in items]}


def _pack_digest(items: tuple[bytes, ...]) -> str:
    return f"sha256:{hashlib.sha256(_canonical(_pack_payload(items))).hexdigest()}"


def _payload_byte_count(payload: dict[str, object]) -> int:
    count = len(_canonical({**payload, "byte_count": 0}))
    while True:
        candidate = len(_canonical({**payload, "byte_count": count}))
        if candidate == count:
            return count
        count = candidate


def _encode_item(item: bytes) -> str:
    return base64.b64encode(item).decode("ascii")


def _decode_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("context pack items must be an array")  # noqa: TRY003
    items: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError("context pack items must be Base64 text")  # noqa: TRY003
        try:
            items.append(base64.b64decode(entry, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError("context pack items must be Base64 text") from error
    return tuple(items)


def _decode_object(data: bytes) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise ContextPackError("context pack payload must be bytes")  # noqa: TRY003
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("context pack payload must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError("context pack payload must be a JSON object")  # noqa: TRY003
    return payload


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_digest(digest: object) -> None:
    if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
        raise ContextPackError("context pack digest must use sha256:<hex>")  # noqa: TRY003


def _chunk_content(content: str, chunk_bytes: int) -> tuple[bytes, ...]:
    """Split one text into line-aligned chunks that stay below the byte limit."""

    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
        raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
    data = content.encode("utf-8")
    if not data:
        return ()
    chunks: list[bytes] = []
    for line in _split_lines(data):
        while line:
            if len(line) <= chunk_bytes:
                chunks.append(line)
                break
            cut = _utf8_boundary(line, chunk_bytes)
            if cut == 0:
                raise ContextPackError("chunk_bytes cannot hold one valid UTF-8 character")  # noqa: TRY003
            chunks.append(line[:cut])
            line = line[cut:]
    return tuple(chunks)


def _split_lines(data: bytes) -> list[bytes]:
    lines: list[bytes] = []
    start = 0
    for index, byte in enumerate(data):
        if byte == 0x0A:
            lines.append(data[start : index + 1])
            start = index + 1
    if start < len(data):
        lines.append(data[start:])
    return lines


def _utf8_boundary(data: bytes, limit: int) -> int:
    cut = min(limit, len(data))
    while cut > 0:
        try:
            data[:cut].decode("utf-8")
        except UnicodeDecodeError:
            cut -= 1
        else:
            return cut
    return 0


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CONTEXT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
