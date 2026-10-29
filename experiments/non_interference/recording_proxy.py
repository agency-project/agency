"""Identical non-retrying LiteLLM recorder and shared admission control for both arms."""

from pathlib import Path
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
import http.server
import json
import sqlite3
import threading
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parent
MODEL = "claude-haiku-5-5"
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@contextmanager
def limiter_database():
    # Serialize the short transactions, including commit, across both arms.
    # Closing the connection here also prevents lingering SQLite readers.
    with (ROOT / "admission.guard").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        connection = sqlite3.connect(ROOT / "admission.sqlite3", timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def admit(reservation):
    """Sliding minute quotas; UTF-8 request bytes plus max output is conservative for text."""
    config = json.loads((ROOT / "configuration.json").read_text())["admission"]
    if reservation > config["tokens_per_minute"]:
        raise ValueError("Request exceeds experiment token admission budget")
    started = time.time()
    lock_retries = 0
    while True:
        try:
            with limiter_database() as db:
                db.execute("CREATE TABLE IF NOT EXISTS admissions (started REAL, tokens INTEGER)")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS inflight (id TEXT PRIMARY KEY, expires REAL)"
                )
                db.execute("BEGIN IMMEDIATE")
                now = time.time()
                db.execute("DELETE FROM admissions WHERE started < ?", (now - 60,))
                db.execute("DELETE FROM inflight WHERE expires < ?", (now,))
                count, total = db.execute(
                    "SELECT count(*),coalesce(sum(tokens),0) FROM admissions"
                ).fetchone()
                active = db.execute("SELECT count(*) FROM inflight").fetchone()[0]
                if (
                    count < config["requests_per_minute"]
                    and total + reservation <= config["tokens_per_minute"]
                    and active < config["max_inflight"]
                ):
                    db.execute("INSERT INTO admissions VALUES (?,?)", (now, reservation))
                    identifier = uuid.uuid4().hex
                    db.execute("INSERT INTO inflight VALUES (?,?)", (identifier, now + 600))
                    return {
                        "admitted_unix_s": now,
                        "wait_s": now - started,
                        "reserved_tokens": reservation,
                        "reservation_id": identifier,
                        "active_requests_at_admission": active + 1,
                        "database_lock_retries": lock_retries,
                    }
        except sqlite3.OperationalError as error:
            if not any(word in str(error).lower() for word in ("locked", "busy")):
                raise
            lock_retries += 1
        time.sleep(0.2)


def release_admission(identifier):
    retries = 0
    while True:
        try:
            with limiter_database() as db:
                db.execute("DELETE FROM inflight WHERE id=?", (identifier,))
            return retries
        except sqlite3.OperationalError as error:
            if not any(word in str(error).lower() for word in ("locked", "busy")):
                raise
            retries += 1
            time.sleep(0.2)


def events(body):
    if body.lstrip().startswith("{"):
        try:
            yield json.loads(body)
        except ValueError:
            return
    else:
        for line in body.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    yield json.loads(line[6:])
                except ValueError:
                    continue


def summarize(directory):
    calls = []
    for path in sorted(Path(directory).glob("*.metadata.json")):
        prefix = str(path).removesuffix(".metadata.json")
        meta = json.loads(path.read_text())
        request_path = Path(prefix + ".request.json")
        request = json.loads(request_path.read_text()) if request_path.exists() else {}
        if not meta["path"].split("?")[0].endswith("/messages"):
            continue
        body_path = Path(prefix + ".response.body")
        body = body_path.read_text(errors="replace") if body_path.exists() else ""
        usage, tools, models, errors = {}, 0, [], []
        complete = False
        for event in events(body):
            if event.get("type") == "message_start":
                message = event["message"]
                usage.update(message.get("usage") or {})
                models.append(message.get("model"))
            elif event.get("type") == "message_delta":
                usage.update(event.get("usage") or {})
            elif event.get("type") == "message_stop":
                usage.update(event.get("usage") or {})
                complete = True
            elif event.get("type") == "content_block_start":
                tools += event.get("content_block", {}).get("type") == "tool_use"
            elif event.get("type") == "message":
                usage.update(event.get("usage") or {})
                models.append(event.get("model"))
                tools += sum(block.get("type") == "tool_use" for block in event.get("content", []))
                complete = True
            if event.get("error"):
                errors.append(event["error"])
        # A successful stream without final usage is unknown, never an invented zero.
        usage_valid = complete and all(isinstance(usage.get(k), int) for k in USAGE_FIELDS[:2])
        calls.append(
            {
                **meta,
                "usage": usage,
                "usage_valid": usage_valid,
                "model": request.get("model"),
                "response_models": models,
                "tool_names": [t.get("name") for t in request.get("tools", [])],
                "tool_calls": tools,
                "stream_errors": errors,
            }
        )

    def unknown_error(call):
        local = call.get("forwarded_upstream") is False or (
            call.get("status") is None
            and not call.get("admission")
            and "database is locked" in call.get("error", "").lower()
        )
        return (
            not local
            and (call.get("status") is None or call.get("status", 0) >= 500)
            and not call["usage_valid"]
        )

    return {
        "model_requests": len(calls),
        "calls": calls,
        "usage_complete": any(c["usage_valid"] for c in calls)
        and not any(unknown_error(c) for c in calls)
        and all(
            c["usage_valid"] for c in calls if c.get("status") is not None and c["status"] < 400
        ),
        "usage": {k: sum(c["usage"].get(k, 0) for c in calls) for k in USAGE_FIELDS},
    }


class RecordingProxy:
    def __init__(self, directory, key, gateway, proxy_token):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.router = json.loads((ROOT / "configuration.json").read_text())["router"].rstrip("/")
        self.counter = 0
        self.lock = threading.Lock()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_):
                return

            def do_GET(self):
                self.forward()

            def do_POST(self):
                self.forward()

            def forward(self):
                supplied = self.headers.get("x-api-key") or self.headers.get(
                    "Authorization", ""
                ).removeprefix("Bearer ")
                if not hmac.compare_digest(supplied, proxy_token):
                    self.send_error(401)
                    return
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                with owner.lock:
                    number = owner.counter
                    owner.counter += 1
                prefix = owner.directory / f"{number:04d}"
                if body:
                    prefix.with_suffix(".request.json").write_bytes(body)
                safe = {
                    k.lower(): v
                    for k, v in self.headers.items()
                    if k.lower() in {"anthropic-version", "anthropic-beta"}
                }
                prefix.with_suffix(".request-headers.json").write_text(json.dumps(safe))
                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower()
                    not in {
                        "host",
                        "content-length",
                        "authorization",
                        "x-api-key",
                        "connection",
                        "accept-encoding",
                    }
                }
                headers.update(
                    {
                        "x-api-key": owner.key,
                        "Authorization": "Bearer " + owner.key,
                        "User-Agent": "agency-non-interference/1.0",
                        "Cache-Control": "no-cache",
                    }
                )
                meta = {
                    "path": self.path,
                    "method": self.command,
                    "start_unix_s": time.time(),
                    "forwarded_upstream": False,
                    "recorder_source_sha256": SOURCE_SHA256,
                }
                try:
                    if body and self.path.split("?")[0].endswith(
                        ("/messages", "/messages/count_tokens")
                    ):
                        request = json.loads(body)
                        if request.get("model") != MODEL:
                            raise ValueError(
                                "Refusing unpinned model: " + str(request.get("model"))
                            )
                        meta["admission"] = admit(len(body) + request.get("max_tokens", 0))
                    with httpx.Client(timeout=180, trust_env=False) as client:
                        meta["forwarded_upstream"] = True
                        with client.stream(
                            self.command, owner.router + self.path, headers=headers, content=body
                        ) as response:
                            meta["status"] = response.status_code
                            meta["headers"] = {
                                k: v
                                for k, v in response.headers.items()
                                if k.lower()
                                in {
                                    "request-id",
                                    "x-request-id",
                                    "x-litellm-call-id",
                                    "x-litellm-response-cost",
                                    "x-litellm-cache-key",
                                    "x-litellm-cache-hit",
                                    "retry-after",
                                    "retry-after-ms",
                                    "x-should-retry",
                                }
                            }
                            self.send_response(response.status_code)
                            self.send_header(
                                "Content-Type",
                                response.headers.get("Content-Type", "application/json"),
                            )
                            for name, value in meta["headers"].items():
                                self.send_header(name, value)
                            self.send_header("Connection", "close")
                            self.end_headers()
                            with prefix.with_suffix(".response.body").open("wb") as saved:
                                for chunk in response.iter_bytes():
                                    saved.write(chunk)
                                    saved.flush()
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                except Exception as exc:
                    meta["error"] = str(exc).replace(owner.key, "[REDACTED]")
                finally:
                    meta["end_unix_s"] = time.time()
                    if meta.get("admission"):
                        try:
                            meta["release_database_lock_retries"] = release_admission(
                                meta["admission"]["reservation_id"]
                            )
                        except Exception as error:
                            meta["release_error"] = str(error).replace(owner.key, "[REDACTED]")
                    prefix.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))

        self.server = http.server.ThreadingHTTPServer((gateway, 0), Handler)
        # Join request threads before summarizing, including interrupted auxiliary streams.
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://{gateway}:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
