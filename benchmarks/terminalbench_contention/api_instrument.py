"""Route Agency's OpenAI-compatible model calls through the shared budget.

Codex talks to Agency's in-sandbox proxy; Agency's host ``LlmHandlerServer``
dispatches through ``_OpenAICompatibleBackend`` to LiteLLM. That path is left
intact. This module only observes it, in the worker process:

* an httpx request hook admits every HTTP attempt (including the OpenAI SDK's
  own retries) through ``SharedBudget`` before it is sent;
* an httpx response hook records status, latency and 429s;
* the streamed chunks are observed for first-token time, completion time and
  model-reported usage, which settles the reservation.

Request and response bodies, URLs, headers and credentials are never written.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from .api_budget import AdmissionCancelled, DeadlineReached, SharedBudget

CHAT_PATH_SUFFIX = "/chat/completions"
SAFE_RESPONSE_HEADERS = ("x-request-id", "openai-processing-ms", "x-litellm-call-id")


def _usage_fields(usage) -> "dict | None":
    """Extract token counts from an SDK usage object or dict."""
    if usage is None:
        return None
    get = usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d)
    details = get("prompt_tokens_details")
    cached = 0
    if details is not None:
        cached = (
            details.get("cached_tokens")
            if isinstance(details, dict)
            else getattr(details, "cached_tokens", 0)
        ) or 0
    completion_details = get("completion_tokens_details")
    reasoning = 0
    if completion_details is not None:
        reasoning = (
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else getattr(completion_details, "reasoning_tokens", 0)
        ) or 0
    prompt = get("prompt_tokens") or 0
    completion = get("completion_tokens") or 0
    return dict(
        prompt_tokens=int(prompt),
        completion_tokens=int(completion),
        total_tokens=int(get("total_tokens") or (prompt + completion)),
        cached_tokens=int(cached),
        reasoning_tokens=int(reasoning),
    )


class ApiTelemetry:
    """Per-worker recorder. One instance serves every model call of one agent."""

    def __init__(self, budget: "SharedBudget | None", path, *, agent_id: str, route="litellm"):
        self.budget = budget
        self.path = Path(path)
        self.agent_id = agent_id
        self.route = route
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._open: dict = {}
        self._open_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ output

    def emit(self, kind: str, **fields) -> None:
        record = dict(kind=kind, agent_id=self.agent_id, mono_ns=time.monotonic_ns(), **fields)
        line = json.dumps(record, default=str) + "\n"
        with self._write_lock, self.path.open("a") as stream:
            stream.write(line)

    # ------------------------------------------------------- logical requests

    def begin(self, *, streaming: bool) -> dict:
        record = dict(
            request_id=uuid.uuid4().hex,
            streaming=streaming,
            created_ns=time.monotonic_ns(),
            attempts=[],
            settled=False,
            first_chunk_ns=None,
            usage=None,
        )
        self._local.current = record
        with self._open_lock:
            self._open[record["request_id"]] = record
        return record

    def current(self) -> "dict | None":
        return getattr(self._local, "current", None)

    def clear_current(self) -> None:
        self._local.current = None

    def _active_attempt(self, record) -> "dict | None":
        attempts = record["attempts"]
        return attempts[-1] if attempts else None

    # ------------------------------------------------------------- httpx hooks

    def on_request(self, request) -> None:
        if request.method != "POST" or not request.url.path.endswith(CHAT_PATH_SUFFIX):
            return  # model listing and other metadata calls are not model requests
        record = self.current()
        if record is None:  # a call made outside the patched dispatch paths
            record = self.begin(streaming=False)
            record["unscoped"] = True
        previous = self._active_attempt(record)
        if previous is not None and not previous.get("closed"):
            # The SDK is retrying without having delivered a response
            # (connection error or timeout on the previous attempt).
            self._close_attempt(record, previous, reason="transport_error", status_code=None)
        index = len(record["attempts"])
        attempt_id = f"{record['request_id']}.{index}"
        ready_ns = time.monotonic_ns()
        attempt = dict(attempt=index, attempt_id=attempt_id, ready_ns=ready_ns, closed=False)
        record["attempts"].append(attempt)
        body = request.content
        admission = None
        if self.budget is not None:
            try:
                admission = self._admit(body, attempt_id, index)
            except (DeadlineReached, AdmissionCancelled) as exc:
                attempt.update(closed=True, outcome="wait_censored")
                self.emit(
                    "wait_censored",
                    request_id=record["request_id"],
                    attempt=index,
                    ready_ns=ready_ns,
                    wait_seconds=(time.monotonic_ns() - ready_ns) / 1e9,
                    reason=type(exc).__name__,
                )
                raise
        admitted_ns = time.monotonic_ns()
        attempt.update(admitted_ns=admitted_ns, admission=admission)
        request.extensions["agency_tb_attempt"] = (record["request_id"], index)
        self.emit(
            "attempt_admitted",
            request_id=record["request_id"],
            attempt=index,
            route=self.route,
            ready_ns=ready_ns,
            admitted_ns=admitted_ns,
            api_wait_seconds=(admitted_ns - ready_ns) / 1e9,
            request_bytes=len(body),
            **(
                {
                    k: v
                    for k, v in admission.items()
                    if k not in ("request_id", "agent_id", "attempt")
                }
                if admission
                else {}
            ),
        )

    def _admit(self, body: bytes, attempt_id: str, index: int) -> dict:
        from agency.observability.profiler import agprof

        with agprof.span("api:admission_wait") as span:
            admission = self.budget.admit(
                body, request_id=attempt_id, agent_id=self.agent_id, attempt=index
            ).to_dict()
            annotate = getattr(span, "annotate", None)
            if annotate is not None:
                annotate(
                    api_wait_seconds=admission["wait_seconds"],
                    admission_charge=admission["charge"],
                    estimated_prompt_tokens=admission["estimated_prompt_tokens"],
                    reserved_completion_tokens=admission["reserved_completion_tokens"],
                    available_before=admission["available_before"],
                    blocked_by=",".join(admission["blocked_by"]),
                )
        return admission

    def on_response(self, response) -> None:
        marker = response.request.extensions.get("agency_tb_attempt")
        if marker is None:
            return
        request_id, index = marker
        with self._open_lock:
            record = self._open.get(request_id)
        if record is None or index >= len(record["attempts"]):
            return
        attempt = record["attempts"][index]
        now = time.monotonic_ns()
        status = response.status_code
        attempt.update(status_code=status, headers_ns=now)
        headers = response.headers
        self.emit(
            "attempt_response",
            request_id=request_id,
            attempt=index,
            status_code=status,
            headers_ns=now,
            response_headers_wait_seconds=(now - attempt["admitted_ns"]) / 1e9,
            retry_after=headers.get("retry-after"),
            quota_headers={
                k: v
                for k, v in headers.items()
                if k.startswith("x-ratelimit-") or k in SAFE_RESPONSE_HEADERS
            },
        )
        if status == 429 and self.budget is not None:
            self.budget.observe_rate_limit(
                retry_after=headers.get("retry-after"),
                reset_tokens=headers.get("x-ratelimit-reset-tokens"),
                reset_requests=headers.get("x-ratelimit-reset-requests"),
                request_id=attempt["attempt_id"],
                agent_id=self.agent_id,
            )
        if status >= 400:
            self._close_attempt(record, attempt, reason=f"http_{status}", status_code=status)

    def _close_attempt(self, record, attempt, *, reason, status_code, usage=None) -> None:
        """Settle one HTTP attempt's reservation exactly once."""
        if attempt.get("closed"):
            return
        attempt["closed"] = True
        attempt["outcome"] = "completed" if usage is not None else reason
        settlement = None
        if self.budget is not None and attempt.get("admission") is not None:
            if usage is not None:
                settlement = self.budget.reconcile(
                    attempt["attempt_id"],
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    cached_tokens=usage["cached_tokens"],
                    total_tokens=usage["total_tokens"],
                    agent_id=self.agent_id,
                )
            else:
                settlement = self.budget.fail(
                    attempt["attempt_id"],
                    reason=reason,
                    status_code=status_code,
                    agent_id=self.agent_id,
                )
        attempt["settlement"] = settlement

    # ------------------------------------------------------- stream observation

    def observe_chunk(self, record, chunk) -> None:
        if record["first_chunk_ns"] is None:
            record["first_chunk_ns"] = time.monotonic_ns()
        usage = _usage_fields(getattr(chunk, "usage", None))
        if usage is not None:
            record["usage"] = usage

    def finish(self, record, *, outcome: str, error: "BaseException | None" = None) -> None:
        """Close a logical request: settle the open attempt and write its summary."""
        if record["settled"]:
            return
        record["settled"] = True
        end_ns = time.monotonic_ns()
        attempts = record["attempts"]
        last = self._active_attempt(record)
        usage = record["usage"] if outcome == "completed" else None
        if last is not None and not last.get("closed"):
            self._close_attempt(
                record,
                last,
                reason=outcome if outcome != "completed" else "no_usage_reported",
                status_code=last.get("status_code"),
                usage=usage,
            )
        with self._open_lock:
            self._open.pop(record["request_id"], None)
        first = attempts[0] if attempts else None
        admitted = [a for a in attempts if a.get("admission")]
        settlement = last.get("settlement") if last else None
        self.emit(
            "request",
            request_id=record["request_id"],
            route=self.route,
            streaming=record["streaming"],
            outcome=outcome,
            error_type=type(error).__name__ if error is not None else None,
            # Timeline of the logical request (monotonic ns, host-wide clock).
            ready_ns=first["ready_ns"] if first else record["created_ns"],
            admitted_ns=last.get("admitted_ns") if last else None,
            headers_ns=last.get("headers_ns") if last else None,
            first_chunk_ns=record["first_chunk_ns"],
            completed_ns=end_ns,
            # Durations (seconds).
            api_wait_seconds=sum(
                (a["admitted_ns"] - a["ready_ns"]) / 1e9 for a in attempts if "admitted_ns" in a
            ),
            response_latency_seconds=(
                (last["headers_ns"] - last["admitted_ns"]) / 1e9
                if last and "headers_ns" in last and "admitted_ns" in last
                else None
            ),
            time_to_first_chunk_seconds=(
                (record["first_chunk_ns"] - last["admitted_ns"]) / 1e9
                if last and record["first_chunk_ns"] and "admitted_ns" in last
                else None
            ),
            stream_seconds=(
                (end_ns - record["first_chunk_ns"]) / 1e9 if record["first_chunk_ns"] else None
            ),
            total_seconds=(end_ns - (first["ready_ns"] if first else record["created_ns"])) / 1e9,
            # Retries.
            attempts=len(attempts),
            retries=max(0, len(attempts) - 1),
            http_429=sum(a.get("status_code") == 429 for a in attempts),
            status_codes=[a.get("status_code") for a in attempts],
            attempt_outcomes=[a.get("outcome") for a in attempts],
            # Estimates (charged at admission) vs. model-reported usage.
            estimated_prompt_tokens=(
                admitted[-1]["admission"]["estimated_prompt_tokens"] if admitted else None
            ),
            reserved_completion_tokens=(
                admitted[-1]["admission"]["reserved_completion_tokens"] if admitted else None
            ),
            admission_charge=admitted[-1]["admission"]["charge"] if admitted else None,
            admission_charge_all_attempts=sum(a["admission"]["charge"] for a in admitted),
            available_budget_at_admission=(
                admitted[-1]["admission"]["available_before"] if admitted else None
            ),
            waiters_at_admission=(
                admitted[-1]["admission"]["waiters_at_admission"] if admitted else None
            ),
            reported_prompt_tokens=usage["prompt_tokens"] if usage else None,
            reported_completion_tokens=usage["completion_tokens"] if usage else None,
            reported_cached_tokens=usage["cached_tokens"] if usage else None,
            reported_reasoning_tokens=usage["reasoning_tokens"] if usage else None,
            reported_total_tokens=usage["total_tokens"] if usage else None,
            reconciliation=settlement,
        )


class _ObservedStream:
    """Iterate an SDK stream while timing chunks and capturing reported usage."""

    def __init__(self, raw_stream, telemetry: ApiTelemetry, record: dict) -> None:
        self._raw = raw_stream
        self._telemetry = telemetry
        self._record = record

    def __iter__(self):
        telemetry, record = self._telemetry, self._record
        try:
            for chunk in self._raw:
                telemetry.observe_chunk(record, chunk)
                yield chunk
        except GeneratorExit:
            telemetry.finish(record, outcome="cancelled")
            raise
        except BaseException as error:
            telemetry.finish(record, outcome="stream_error", error=error)
            raise
        else:
            telemetry.finish(
                record, outcome="completed" if record["usage"] else "no_usage_reported"
            )

    def __getattr__(self, name):
        return getattr(self._raw, name)


def install(backend_class, telemetry: ApiTelemetry) -> None:
    """Instrument ``_OpenAICompatibleBackend`` in this process only."""
    uninstall(backend_class)  # never stack two layers of hooks
    original_make_client = backend_class.make_client
    original_stream = backend_class._call_backend_stream
    original_call = backend_class._call_backend
    original_format = backend_class._format_context_backend_to_agency

    def make_client(backend, timeout):
        client = original_make_client(backend, timeout)
        transport = client._client
        transport.event_hooks["request"].append(telemetry.on_request)
        transport.event_hooks["response"].append(telemetry.on_response)
        return client

    def call_backend_stream(backend, backend_request, on_client=None):
        record = telemetry.begin(streaming=True)
        try:
            raw_stream, client = original_stream(backend, backend_request, on_client)
        except BaseException as error:
            telemetry.finish(record, outcome="request_error", error=error)
            raise
        finally:
            telemetry.clear_current()
        return _ObservedStream(raw_stream, telemetry, record), client

    def call_backend(backend, backend_request):
        record = telemetry.begin(streaming=False)
        try:
            result = original_call(backend, backend_request)
        except BaseException as error:
            telemetry.finish(record, outcome="request_error", error=error)
            raise
        finally:
            telemetry.clear_current()
        record["first_chunk_ns"] = time.monotonic_ns()
        record["usage"] = _usage_fields(getattr(result, "usage", None))
        telemetry.finish(record, outcome="completed" if record["usage"] else "no_usage_reported")
        return result

    backend_class.make_client = make_client
    backend_class._call_backend_stream = call_backend_stream
    backend_class._call_backend = call_backend
    backend_class._tb_contention_originals = (
        original_make_client,
        original_stream,
        original_call,
        original_format,
    )


def uninstall(backend_class) -> None:
    originals = getattr(backend_class, "_tb_contention_originals", None)
    if originals is None:
        return
    backend_class.make_client, backend_class._call_backend_stream = originals[0], originals[1]
    backend_class._call_backend = originals[2]
    del backend_class._tb_contention_originals
