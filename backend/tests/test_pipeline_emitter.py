"""Unit tests for the per-search pipeline pubsub emitter and registry."""

from __future__ import annotations

from typing import Any

import pytest

from app.services import pipeline_emitter
from app.services.pipeline_emitter import PipelineEmitter


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Each test gets a clean global registry."""
    pipeline_emitter.reset()
    yield
    pipeline_emitter.reset()


@pytest.mark.asyncio
async def test_create_stream_returns_unique_ids() -> None:
    a = await pipeline_emitter.create_stream()
    b = await pipeline_emitter.create_stream()
    assert a.stream_id != b.stream_id
    assert pipeline_emitter.registry_size() == 2


@pytest.mark.asyncio
async def test_get_stream_returns_none_for_unknown_id() -> None:
    assert await pipeline_emitter.get_stream("does-not-exist") is None


@pytest.mark.asyncio
async def test_emit_assigns_monotonic_seq_and_records_history() -> None:
    em = PipelineEmitter("s1")
    e1 = await em.emit("stage.start", {"name": "extract"})
    e2 = await em.emit("stage.done", {"name": "extract", "elapsed_ms": 5})
    assert (e1.seq, e2.seq) == (1, 2)
    history = em.history
    assert [ev.kind for ev in history] == ["stage.start", "stage.done"]
    assert history[1].payload["elapsed_ms"] == 5


@pytest.mark.asyncio
async def test_subscribe_replays_history_then_yields_live_events() -> None:
    em = PipelineEmitter("s2")
    await em.emit("stage.start", {"name": "extract"})
    await em.emit("feature.done", {"name": "hsv"})

    iterator = em.subscribe().__aiter__()
    replay1 = await iterator.__anext__()
    replay2 = await iterator.__anext__()
    assert (replay1.kind, replay2.kind) == ("stage.start", "feature.done")

    # Now publish a live event after subscribing.
    await em.emit("stage.done", {"name": "extract"})
    live = await iterator.__anext__()
    assert live.kind == "stage.done"

    # Terminal event closes the iterator.
    await em.emit("rank.done", {"top_k": 3})
    final = await iterator.__anext__()
    assert final.kind == "rank.done"
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()


@pytest.mark.asyncio
async def test_terminal_event_closes_late_subscribers_immediately() -> None:
    """A subscriber attaching after `rank.done` still gets full replay + sentinel."""
    em = PipelineEmitter("s3")
    await em.emit("stage.start", {"name": "extract"})
    await em.emit("rank.done", {"top_k": 1})

    received: list[str] = []
    async for ev in em.subscribe():
        received.append(ev.kind)
    assert received == ["stage.start", "rank.done"]


@pytest.mark.asyncio
async def test_emit_after_terminal_is_ignored() -> None:
    em = PipelineEmitter("s4")
    await em.emit("rank.done", {"top_k": 0})
    # Further emits should not extend history.
    await em.emit("stage.start", {"name": "should not appear"})
    kinds = [ev.kind for ev in em.history]
    assert kinds == ["rank.done"]


@pytest.mark.asyncio
async def test_reset_clears_registry() -> None:
    await pipeline_emitter.create_stream()
    assert pipeline_emitter.registry_size() == 1
    pipeline_emitter.reset()
    assert pipeline_emitter.registry_size() == 0


@pytest.mark.asyncio
async def test_multiple_subscribers_each_see_full_event_stream() -> None:
    em = PipelineEmitter("s5")

    async def consume() -> list[str]:
        return [ev.kind async for ev in em.subscribe()]

    import asyncio

    a = asyncio.create_task(consume())
    b = asyncio.create_task(consume())
    # Yield once so both subscribers register before we emit.
    await asyncio.sleep(0)
    await em.emit("stage.start", {"name": "extract"})
    await em.emit("rank.done", {"top_k": 0})

    out_a = await a
    out_b = await b
    assert out_a == out_b == ["stage.start", "rank.done"]


@pytest.mark.asyncio
async def test_close_with_error_emits_error_and_seals_channel() -> None:
    em = PipelineEmitter("s6")
    await em.close_with_error("boom")
    received: list[dict[str, Any]] = []
    async for ev in em.subscribe():
        received.append({"kind": ev.kind, "payload": ev.payload})
    assert received == [{"kind": "error", "payload": {"message": "boom"}}]


@pytest.mark.asyncio
async def test_event_to_jsonable_round_trips() -> None:
    em = PipelineEmitter("s7")
    ev = await em.emit("rank.tick", {"rank": 1, "image_id": 42, "score": 0.91})
    payload = ev.to_jsonable()
    assert payload["seq"] == 1
    assert payload["kind"] == "rank.tick"
    assert payload["payload"] == {"rank": 1, "image_id": 42, "score": 0.91}
    assert payload["at_ms"] >= 0
