"""agMockBackend — replays a previously-recorded agent execution instead of
calling a real model.

Loads the `llm_block`-finalized exchanges from another agent's own
agDataLogger sqlite db (`agdatalogger.py`'s `events` table) and replays them
back, one exchange per `dispatch()`/`dispatch_stream()` call, in the order
they were originally recorded. Matching is purely positional -- the Nth call
this backend receives replays the Nth successful exchange in the source db
-- since `call_label` (a fresh uuid per HTTP attempt, see
`llm_handler_server.py`) has no stable meaning across two different runs and
can't be used to correlate "the same logical call".

Only `llm_block`-type exchanges (successful completions) are replayed;
`llm_stream_error`/`llm_stream_cancelled` groups in the source db are
skipped. Reproducing the original failure/retry sequence is out of scope.

Does not subclass `agllm` -- `LlmHandlerServer` only ever calls `.model`,
`.dispatch()`, `.dispatch_stream()`, and `.fetch_context_limit()` on a
backend (see the plain `_RecordingBackend` test double in
`test_llm_handler_server.py`), so there's no need for the real backends'
`_call_backend*`/`_format_*` machinery here.
"""

from __future__ import annotations

import copy
import json
import math
import random
import sqlite3
import time
from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    from ..configs.agconfig import agconfig as agconfig_cls

_METADATA_BLOCK_INDEX = 2**31 - 1
_CHUNKS_PER_BLOCK = 4


def _load_replay_exchanges(db_path: str) -> "list[list[dict]]":
    """Read every finalized `llm_block` exchange from *db_path*, in the
    order they were recorded, grouped by `call_label`."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT call_label, payload FROM events WHERE type = 'llm_block' ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    exchanges: "list[list[dict]]" = []
    current_label = object()
    for call_label, payload_json in rows:
        if call_label != current_label:
            exchanges.append([])
            current_label = call_label
        exchanges[-1].append(json.loads(payload_json))
    return exchanges


def _split_text(text: str, chunk_count: int) -> "list[str]":
    """Split *text* into up to *chunk_count* pieces, re-concatenable
    verbatim (each piece after the first carries its own leading space)."""
    words = text.split(" ")
    if len(words) <= 1 or chunk_count <= 1:
        return [text]
    chunk_count = min(chunk_count, len(words))
    size = math.ceil(len(words) / chunk_count)
    pieces = []
    for i in range(0, len(words), size):
        piece = " ".join(words[i : i + size])
        pieces.append(piece if not pieces else " " + piece)
    return pieces


def _split_block_into_chunks(block: dict) -> "list[dict]":
    """Re-fragment one already-merged block back into synthetic
    `block_delta`-shaped pieces, so the real merge loop in
    `llm_handler_server.py`'s `_run_stream_producer` reconstructs it exactly
    the way it would a real backend's stream -- this mock never needs to
    duplicate that merge logic."""
    block_type = block.get("type")
    if block_type in ("text", "thinking"):
        text = block.get("text") or ""
        chunks = [{"text": piece} for piece in _split_text(text, _CHUNKS_PER_BLOCK)] or [
            {"text": ""}
        ]
        if block_type == "thinking" and block.get("signature"):
            chunks.append({"signature": block["signature"]})
        return chunks
    if block_type == "tool_use":
        chunks = [{"id": block.get("id", ""), "name": block.get("name", "")}]
        arguments = block.get("arguments") or ""
        if arguments:
            chunks.extend(
                {"arguments": piece} for piece in _split_text(arguments, _CHUNKS_PER_BLOCK)
            )
        return chunks
    # citations-only deltas, provider-native/unknown block types: replay
    # whole, rather than guess at a splitting scheme for a shape we don't
    # specifically understand.
    return [{k: v for k, v in block.items() if k not in ("type", "index", "ts_start", "ts_end")}]


def exact_replay_timing(exchange: dict, chunk_counts: "list[int]") -> "Iterator[float]":
    """First delay = the exchange's recorded TTFT; each block's own recorded
    duration (`ts_end - ts_start`, both persisted by `_tag_metadata_block`'s
    caller in `llm_handler_server.py`) is split evenly across that block's
    synthetic chunks. Exact at the aggregate (TTFT, per-block duration)
    level; intra-block per-chunk cadence is an even-split approximation,
    since true per-fragment arrival timestamps aren't persisted."""
    metadata = exchange["metadata"]
    ttft_s = (metadata.get("ttft_ms") or 0) / 1000
    yield ttft_s
    content_blocks = exchange["blocks"]
    for block, chunk_count in zip(content_blocks, chunk_counts):
        duration = max(0.0, (block.get("ts_end", 0) or 0) - (block.get("ts_start", 0) or 0))
        per_chunk = duration / chunk_count if chunk_count else 0.0
        for _ in range(chunk_count):
            yield per_chunk


def instant_timing(exchange: dict, chunk_counts: "list[int]") -> "Iterator[float]":
    del exchange
    for _ in range(sum(chunk_counts) + 1):
        yield 0.0


def constant_timing(ttft_s: float, tpot_s: float) -> "Callable[[dict, list[int]], Iterator[float]]":
    """Factory: a fixed TTFT before the first chunk, then a fixed delay
    before every chunk after ("time per output token" -- a chunk stands in
    for a token at this replay's re-fragmentation granularity, same
    approximation `poisson_timing`'s `rate_hz` already makes)."""

    def _timing_fn(exchange: dict, chunk_counts: "list[int]") -> "Iterator[float]":
        del exchange
        yield ttft_s
        for _ in range(sum(chunk_counts)):
            yield tpot_s

    return _timing_fn


def poisson_timing(
    rate_hz: float, ttft_mean_s: float = 0.3, seed: "int | None" = None
) -> "Callable[[dict, list[int]], Iterator[float]]":
    """Factory: returns a timing_fn sampling inter-chunk delays from an
    exponential distribution (Poisson arrival process) at *rate_hz*, with a
    separately-parameterized mean TTFT."""
    rng = random.Random(seed)

    def _timing_fn(exchange: dict, chunk_counts: "list[int]") -> "Iterator[float]":
        del exchange
        yield rng.expovariate(1 / ttft_mean_s) if ttft_mean_s > 0 else 0.0
        for _ in range(sum(chunk_counts)):
            yield rng.expovariate(rate_hz) if rate_hz > 0 else 0.0

    return _timing_fn


_BUILTIN_TIMING_MODES: "dict[str, Callable]" = {
    "exact": exact_replay_timing,
    "instant": instant_timing,
}


class _MockBackend:
    """Backend selected by `provider="mock"`. Holds the given agconfig
    directly (not a clone -- so `agconfig.timing_fn = ...`/
    `agconfig.timing_mode = ...` mutations made after construction take
    effect on the next call).

    Replay state (the loaded exchange list and the position within it) is
    per-instance: reconstructing the backend (e.g. a later `set_config()`
    call on the same `LlmHandlerServer`) starts replay over from the
    beginning. Not addressed here -- out of scope for the deterministic
    single-run-replay use case this backend targets."""

    def __init__(self, agconfig: "agconfig_cls") -> None:
        self.agconfig = agconfig
        self._exchanges: "list[list[dict]] | None" = None
        self._next_index = 0

    @property
    def model(self) -> str:
        return self.agconfig.model

    def _ensure_loaded(self) -> "list[list[dict]]":
        if self._exchanges is None:
            if not self.agconfig.replay_db_path:
                raise ValueError(
                    "mock LLM backend: replay_db_path is not set -- point it at the "
                    "source agent's own <agname>_data.sqlite3 (agconfig(provider='mock',"
                    " replay_db_path=...))"
                )
            self._exchanges = _load_replay_exchanges(self.agconfig.replay_db_path)
        return self._exchanges

    def _next_exchange(self) -> "list[dict]":
        exchanges = self._ensure_loaded()
        if self._next_index >= len(exchanges):
            raise RuntimeError(
                f"mock LLM backend: replay exhausted after {self._next_index} exchange(s) "
                f"from {self.agconfig.replay_db_path!r}"
            )
        blocks = exchanges[self._next_index]
        self._next_index += 1
        return blocks

    def _resolve_timing_fn(self) -> "Callable[[dict, list[int]], Iterator[float]]":
        if self.agconfig.timing_fn is not None:
            return self.agconfig.timing_fn
        if self.agconfig.timing_mode == "poisson":
            return poisson_timing(
                self.agconfig.poisson_rate_hz,
                self.agconfig.poisson_ttft_mean_s,
                self.agconfig.poisson_seed,
            )
        if self.agconfig.timing_mode == "constant":
            return constant_timing(self.agconfig.constant_ttft_s, self.agconfig.constant_tpot_s)
        return _BUILTIN_TIMING_MODES.get(self.agconfig.timing_mode, exact_replay_timing)

    def fetch_context_limit(self) -> int:
        return (
            int(self.agconfig.context_limit) if self.agconfig.context_limit is not None else 200_000
        )

    def dispatch(self, request: dict) -> dict:
        del request
        blocks = copy.deepcopy(self._next_exchange())
        metadata = next((b for b in blocks if b.get("type") == "metadata"), {})
        return {
            "message": {"role": "assistant", "blocks": blocks},
            "usage": metadata.get("usage"),
            "stop_reason": metadata.get("stop_reason"),
        }

    def dispatch_stream(self, request: dict, on_client=None) -> "Iterator[dict]":
        del request, on_client
        blocks = copy.deepcopy(self._next_exchange())
        metadata = next((b for b in blocks if b.get("type") == "metadata"), {})
        content_blocks = [b for b in blocks if b.get("type") != "metadata"]

        per_block_chunks = [_split_block_into_chunks(block) for block in content_blocks]
        chunk_counts = [len(chunks) for chunks in per_block_chunks]
        delays = self._resolve_timing_fn()(
            {"blocks": content_blocks, "metadata": metadata}, chunk_counts
        )

        for block, chunks in zip(content_blocks, per_block_chunks):
            for chunk in chunks:
                delay = next(delays, 0.0)
                if delay:
                    time.sleep(delay)
                yield {
                    "type": "block_delta",
                    "index": block.get("index", _METADATA_BLOCK_INDEX),
                    "block_type": block.get("type"),
                    **chunk,
                }
        yield {
            "type": "usage",
            "usage": metadata.get("usage"),
            "stop_reason": metadata.get("stop_reason"),
        }
