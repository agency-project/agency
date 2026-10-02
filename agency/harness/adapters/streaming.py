"""Response streaming shared by every adapter that proxies the host LLM stream."""

from __future__ import annotations


_STREAM_END = object()


async def with_keepalive(stream, interval_s: float):
    """Yield *stream*'s items, and None after every *interval_s* of silence (0 = never).

    One producer task reads the whole stream so it is never iterated from two tasks."""
    import asyncio

    if not interval_s:
        async for item in stream:
            yield item
        return
    queue: asyncio.Queue = asyncio.Queue()

    async def produce():
        try:
            async for item in stream:
                await queue.put((item, None))
        except Exception as exc:
            await queue.put((None, exc))
        else:
            await queue.put((_STREAM_END, None))

    producer = asyncio.create_task(produce())
    try:
        while True:
            try:
                item, exc = await asyncio.wait_for(queue.get(), interval_s)
            except asyncio.TimeoutError:
                yield None
                continue
            if exc is not None:
                raise exc
            if item is _STREAM_END:
                return
            yield item
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


async def stream_response(
    router,
    token,
    context,
    model,
    formatter,
    *,
    keepalive_frame=None,
    keepalive_s: float = 0,
    error_frame=None,
    **dispatch_kwargs,
):
    """A TUI interrupt must close the upstream request, not just its HTTP socket.

    With keepalive_s set, *keepalive_frame* is sent after that long without upstream output.
    An upstream error after a frame was sent becomes *error_frame(exc)* when one is given."""
    import anyio

    stream = router.dispatch_stream_async(token, context, **dispatch_kwargs)
    items = with_keepalive(stream, keepalive_s if keepalive_frame else 0)
    sent_frame = False
    try:
        async for item in items:
            if item is None:
                sent_frame = True
                yield keepalive_frame
            elif item.get("type") == "done":
                for frame in formatter([item], model):
                    sent_frame = True
                    yield frame
    except Exception as exc:
        # Before any frame, start_streaming_response turns the error into an HTTP status.
        if not sent_frame or error_frame is None:
            raise
        yield error_frame(exc)
    finally:
        with anyio.CancelScope(shield=True):
            await items.aclose()
            await stream.aclose()


def upstream_error(exc: BaseException) -> "tuple[str, int, bool]":
    """(message, status_code, transient) for an upstream stream failure."""
    from ..clients.host_services_client import HostDispatchError

    if isinstance(exc, HostDispatchError):
        return str(exc), exc.status_code, exc.transient
    return str(exc) or f"upstream stream failed: {type(exc).__name__}", 502, True


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
