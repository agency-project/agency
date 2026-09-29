"""Response streaming shared by every adapter that proxies the host LLM stream."""

from __future__ import annotations


async def stream_response(router, token, context, model, formatter):
    """A TUI interrupt must close the upstream request, not just its HTTP socket."""
    import anyio

    stream = router.dispatch_stream_async(token, context)
    try:
        async for item in stream:
            if item.get("type") == "done":
                for frame in formatter([item], model):
                    yield frame
    finally:
        with anyio.CancelScope(shield=True):
            await stream.aclose()


async def start_streaming_response(request, frames):
    """Return a StreamingResponse only once `frames` has produced its first frame.

    Starlette commits HTTP 200 as soon as it starts a StreamingResponse, so an
    upstream failure after that point reaches the client as a truncated SSE
    body ("incomplete chunked read") instead of the provider's error. Pulling
    the first frame first lets a pre-stream failure become a real HTTP error.
    """
    import asyncio

    import anyio
    from fastapi.responses import JSONResponse, StreamingResponse

    from ..clients.host_services_client import HostDispatchError

    async def wait_for_disconnect():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    first_task = asyncio.create_task(anext(frames))
    disconnect_task = asyncio.create_task(wait_for_disconnect())
    first_received = False
    try:
        done, _ = await asyncio.wait(
            {first_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if disconnect_task in done:
            return JSONResponse({"error": "client disconnected"}, status_code=499)
        first = first_task.result()
        first_received = True
    except HostDispatchError as exc:
        return JSONResponse(
            {
                "error": {
                    "message": str(exc),
                    "type": "upstream_error",
                    "transient": exc.transient,
                }
            },
            status_code=exc.status_code,
        )
    except Exception as exc:
        # StopAsyncIteration (no events at all) lands here too.
        message = str(exc) or f"upstream stream failed: {type(exc).__name__}"
        return JSONResponse(
            {"error": {"message": message, "type": "upstream_error", "transient": True}},
            status_code=502,
        )
    finally:
        # A disconnect before HTTP headers must cancel the blocked model read
        # just as a later StreamingResponse disconnect does.
        with anyio.CancelScope(shield=True):
            first_task.cancel()
            disconnect_task.cancel()
            await asyncio.gather(first_task, disconnect_task, return_exceptions=True)
            if not first_received:
                await frames.aclose()

    async def response_frames():
        try:
            yield first
            async for frame in frames:
                yield frame
        finally:
            await frames.aclose()

    return StreamingResponse(response_frames(), media_type="text/event-stream")
