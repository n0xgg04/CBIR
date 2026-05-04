"""WebSocket pipeline streaming + stream allocation endpoint.

Two endpoints back the Phase 3 demo gate:

    POST   /api/v1/search/streams         → allocate a stream_id
    WS     /api/v1/ws/search/{stream_id}  → stream pipeline events live

The frontend opens the WS *before* triggering POST /api/v1/search so it
catches the very first `stage.start`. The emitter also buffers history, so a
slightly late subscriber still replays everything from `seq=1`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from fastapi import Path as PathParam

from app.services import pipeline_emitter

router = APIRouter(tags=["pipeline"])


@router.post(
    "/api/v1/search/streams",
    summary="Allocate a stream_id for live pipeline streaming.",
    status_code=status.HTTP_201_CREATED,
)
async def create_stream() -> dict[str, str]:
    emitter = await pipeline_emitter.create_stream()
    return {"stream_id": emitter.stream_id}


@router.websocket("/api/v1/ws/search/{stream_id}")
async def ws_search_stream(
    websocket: WebSocket,
    stream_id: Annotated[str, PathParam(min_length=1, max_length=64)],
) -> None:
    """Subscribe to a search's pipeline timeline as JSON events."""
    emitter = await pipeline_emitter.get_stream(stream_id)
    if emitter is None:
        # 1008 = "Policy Violation" — closest match for "stream not found".
        await websocket.close(code=1008, reason="unknown stream_id")
        return
    await websocket.accept()
    try:
        async for event in emitter.subscribe():
            await websocket.send_json(event.to_jsonable())
    except WebSocketDisconnect:
        # Client hung up mid-stream — the emitter cleans the queue itself.
        return
    finally:
        # If we exited the async-for normally (terminal sentinel) we still
        # need to close the socket so the client sees a clean shutdown.
        try:
            await websocket.close()
        except RuntimeError:
            # Already closed by the disconnect handler — nothing to do.
            pass
