"""Per-search pubsub for live pipeline events.

Phase 3 demo gate: the frontend renders a live timeline (extract → load_corpus
→ cosine → rank) while the search runs. To make late WebSocket subscribers
work without races, every emitted event is appended to a bounded *history*
buffer. New subscribers replay the history then start receiving live events;
once the run sends `rank.done` the channel is closed and the emitter sits in a
short retention window so reconnects can still pick up the trailing summary.

The registry is process-local (single uvicorn worker assumed for Phase 3 —
the demo deployment is single-node). For multi-worker scale we'd need Redis
pubsub, but that's explicitly out of scope per PLAN.md §11.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Final

# How long an emitter sticks around after `rank.done` so that late subscribers
# (the frontend reconnects after a tab refresh, the test fixture queries history
# after the search finished) can still replay events.
RETENTION_SECONDS: Final[float] = 60.0

# Cap subscriber queues so a stuck consumer can't OOM the server.
MAX_QUEUE_SIZE: Final[int] = 512

# Sentinel kinds the emitter recognises. Other strings pass through untouched.
KIND_STAGE_START: Final[str] = "stage.start"
KIND_STAGE_DONE: Final[str] = "stage.done"
KIND_FEATURE_START: Final[str] = "feature.start"
KIND_FEATURE_DONE: Final[str] = "feature.done"
KIND_RANK_TICK: Final[str] = "rank.tick"
KIND_RANK_DONE: Final[str] = "rank.done"

_TERMINAL_KINDS: Final[frozenset[str]] = frozenset({KIND_RANK_DONE})


@dataclass(frozen=True)
class PipelineEvent:
    """A single timeline tick, replayable to late subscribers."""

    seq: int
    kind: str
    payload: dict[str, Any]
    at_ms: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "payload": self.payload,
            "at_ms": self.at_ms,
        }


class PipelineEmitter:
    """Single-producer, many-consumer event channel for one search run."""

    def __init__(self, stream_id: str) -> None:
        self.stream_id = stream_id
        self._history: list[PipelineEvent] = []
        self._subscribers: list[asyncio.Queue[PipelineEvent | None]] = []
        self._lock = asyncio.Lock()
        self._seq = 0
        self._done = False
        self._done_at: float | None = None
        self._started_at_perf = time.perf_counter()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def history(self) -> tuple[PipelineEvent, ...]:
        """Snapshot of all events seen so far — primarily for tests."""
        return tuple(self._history)

    async def emit(
        self, kind: str, payload: dict[str, Any] | None = None
    ) -> PipelineEvent:
        """Publish an event to history + every active subscriber."""
        async with self._lock:
            if self._done:
                # After `rank.done` the channel is sealed; ignore late emits
                # rather than rewriting history mid-replay.
                last = self._history[-1] if self._history else None
                if last is not None:
                    return last
            self._seq += 1
            elapsed_ms = int((time.perf_counter() - self._started_at_perf) * 1000)
            event = PipelineEvent(
                seq=self._seq,
                kind=kind,
                payload=dict(payload or {}),
                at_ms=elapsed_ms,
            )
            self._history.append(event)
            for q in self._subscribers:
                _put_nowait_drop(q, event)
            if kind in _TERMINAL_KINDS:
                self._done = True
                self._done_at = time.monotonic()
                for q in self._subscribers:
                    _put_nowait_drop(q, None)
            return event

    async def close_with_error(self, message: str) -> None:
        """Mark the stream as failed so subscribers don't hang on partial runs."""
        await self.emit("error", {"message": message})
        async with self._lock:
            self._done = True
            self._done_at = time.monotonic()
            for q in self._subscribers:
                _put_nowait_drop(q, None)

    async def subscribe(self) -> AsyncIterator[PipelineEvent]:
        """Yield history then live events; ends after the terminal sentinel."""
        async with self._lock:
            queue: asyncio.Queue[PipelineEvent | None] = asyncio.Queue(
                maxsize=MAX_QUEUE_SIZE
            )
            for ev in self._history:
                _put_nowait_drop(queue, ev)
            if self._done:
                _put_nowait_drop(queue, None)
            self._subscribers.append(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)


def _put_nowait_drop(
    queue: asyncio.Queue[PipelineEvent | None], item: PipelineEvent | None
) -> None:
    """Best-effort enqueue; drop on full so a stalled WS client can't block search."""
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        # We choose to drop rather than block — for a stalled WS client the
        # next event still arrives, and the history buffer remains intact for
        # any fresh subscriber that joins after the stall clears.
        pass


class _Registry:
    """Holds emitters keyed by stream_id with TTL-based cleanup."""

    def __init__(self) -> None:
        self._emitters: dict[str, PipelineEmitter] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> PipelineEmitter:
        async with self._lock:
            self._gc_locked()
            stream_id = uuid.uuid4().hex
            emitter = PipelineEmitter(stream_id=stream_id)
            self._emitters[stream_id] = emitter
            return emitter

    async def get(self, stream_id: str) -> PipelineEmitter | None:
        async with self._lock:
            self._gc_locked()
            return self._emitters.get(stream_id)

    def reset(self) -> None:
        """Synchronous test hook — safe because tests own the event loop."""
        self._emitters.clear()

    def size(self) -> int:
        return len(self._emitters)

    def _gc_locked(self) -> None:
        now = time.monotonic()
        expired = [
            sid
            for sid, em in self._emitters.items()
            if em._done_at is not None and (now - em._done_at) > RETENTION_SECONDS
        ]
        for sid in expired:
            del self._emitters[sid]


_REGISTRY = _Registry()


async def create_stream() -> PipelineEmitter:
    """Allocate a new emitter and return it — caller publishes the stream_id."""
    return await _REGISTRY.create()


async def get_stream(stream_id: str) -> PipelineEmitter | None:
    """Look up an existing emitter or return `None` if expired/unknown."""
    return await _REGISTRY.get(stream_id)


def reset() -> None:
    """Drop all emitters — tests call this between cases for isolation."""
    _REGISTRY.reset()


def registry_size() -> int:
    """Expose the live-emitter count for diagnostics and tests."""
    return _REGISTRY.size()
