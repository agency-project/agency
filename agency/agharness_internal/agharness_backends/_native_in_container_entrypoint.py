"""Entrypoint executed INSIDE a container-backed sandbox (via
`native.launch_in_container_entrypoint`'s `python3 <this file's path>
...`) -- never imported directly by host-side code.

**Deliberately run as a raw script, never as `python3 -m package.module`:**
`-m` forces Python to import every parent package first (`agency`, then
`agency.agharness_internal`, ...), which transitively imports
`agency/agllm.py`'s `import openai` (and other host-venv-only
dependencies) at module level -- confirmed to fail with
`ModuleNotFoundError: No module named 'openai'` against the actual
`agency-sandbox:latest` base image before `agutil.ensure_python_packages_
in_container` existed to install it. Running this file directly by path
sidesteps needing `agency/__init__.py` at all for THIS file's own code
(no relative imports here) -- it deliberately does not reuse
`agtool`/`agskill`/`agschema` (whose import chains pull in `agent`/
`agllm_backends`, hence `openai`/`anthropic`/`boto3` at module level) and
reimplements the react loop directly, stdlib-plus-httpx only (`httpx` is
confirmed present in the base image; `fastapi`/`uvicorn` are not needed
here since this process serves its own protocol via a raw `socketserver`,
not an ASGI app). The one exception: `agtool_pure.py` (edit's fuzzy-match
replace pipeline, read's pagination, glob/grep's command+parsing logic) is
loaded directly by file path too (`_load_agtool_pure()` below) -- it has
zero relative/agency imports by design, so this works without the same
`ModuleNotFoundError` risk, and means those built-in tools share their
actual algorithm with the host-side sandboxed tools rather than a
hand-copied duplicate.

Wire protocol: length-prefixed JSON over a Unix domain socket -- an 8-byte
big-endian length header, then that many UTF-8 JSON bytes, in both
directions. (A bare single `recv()` -- this file's very first version --
only works for tiny fixed replies like `ping`; a multi-turn react loop's
message history easily exceeds one `recv()` buffer's worth of bytes, and
naive framing would silently truncate.)

Two request shapes:
- `{"op": "ping"}` -- health check proving the launch+bridge mechanism
  itself works (see agharness_backends/native.py's Phase 3a tests):
  confirms the bind-mounted `agency` package is visible via a plain
  filesystem check, not a live import, so it doesn't conflate "is the
  launch+bridge mechanism correct" with "is the image fully provisioned."
- `{"op": "run", "token", "terminus_sock", "mcp_server_sock", "model",
  "messages", "custom_tools", "suppress_builtins", "max_steps"}` -- runs a
  minimal react loop: dispatch each LLM turn to `agllm_terminus`'s
  `/internal/dispatch` (the same host-side, credential-holding service
  `agproxy_llm.py` uses -- see agharness_internal/agllm_terminus.py),
  execute any resulting tool calls, and loop until the model responds with
  no further tool calls or `max_steps` is exhausted. `_BUILTIN_TOOL_SCHEMAS`/
  `_TOOL_DISPATCH` (bash/read/write/edit/glob/grep/webfetch/todowrite) run
  locally -- plain `subprocess`/file I/O, no bridging needed, this process
  already runs inside the container -- and are always available unless
  `suppress_builtins` is set (native.py sends this for `skill.replace_tools`,
  which replaces the whole tool set rather than extending it). `mcp_server_
  sock`, if given, is the bind-mounted UDS path to agmcp_server.py's shared
  MCP server (Phase 4): its CURRENT tool set (reserve_cpu/cpu_release/
  daemon_release/submit_output/ask_human) is discovered dynamically (not
  hardcoded here) and dispatched via the real `mcp` client library, the same
  protocol every harness's own native MCP client speaks -- always merged in
  regardless of `suppress_builtins` (these are host control-plane tools, not
  part of a skill's own tool-set choice). `custom_tools` (`skill.add_tools`/
  `replace_tools`) are host-authored `fn`s cloudpickled by native.py and
  shipped here as base64 bytes, loaded and called against a minimal
  `agdata_pure.py` stand-in for the real `agdata`/`agerror` -- see
  `_make_custom_tool_handler`'s own section docstring for the full mechanism
  and its one inherent, documented limitation (a closure that captures a
  live host-only object may unpickle without error yet not work correctly).
  Every tool result (built-in, MCP, or custom) over `_TOOL_OUTPUT_OFFLOAD_
  CHARS` is saved whole to a local file and replaced with a short note
  (`_offload_if_oversized`), mirroring `agtool.py`'s `dispatch_tools()`
  offload behavior for the host-side path. Structured-output validation
  itself (deciding when `submit_output` has covered every required field,
  and reprompting if not) lives one level up, in native.py's `_NativeBackend.
  execute()` -- this entrypoint just runs however many `run` requests it's
  given. Compaction (`_maybe_compact`, `_fetch_context_limit`) runs at the
  top of every loop iteration -- same threshold/tail-selection/pruning
  algorithm `agllm.py`'s own `maybe_compact()` uses (shared via `agllm_pure.
  py`, loaded the same way as `agtool_pure.py`), with the summarization call
  itself going through this loop's own `_dispatch_via_terminus` instead of
  `agllm.call()`'s retry/streaming machinery.
"""

from __future__ import annotations

import json
import os
import socketserver
import struct
import subprocess
import sys
import uuid

# Deliberately a bare string literal, NOT imported from
# agutil.AGENCY_PACKAGE_CONTAINER_MOUNT -- importing anything under
# `agency.*` from this file re-triggers the exact `agency/__init__.py`
# import chain (and its `import openai`, etc.) this file exists to avoid
# for its OWN code. Keep in sync with agutil.AGENCY_PACKAGE_CONTAINER_MOUNT
# by hand. The env var override exists solely so a fast, no-Docker test
# seam can point this at a real repo checkout's own `agency/` directory
# (which already has `agtool_pure.py`/`agllm_pure.py` at the same relative
# path) and import/run this module's loop logic directly, in-process --
# unset in every real launch (native.py never sets it), so production
# behavior is unchanged.
_AGENCY_PACKAGE_CONTAINER_MOUNT = os.environ.get(
    "AGENCY_PACKAGE_CONTAINER_MOUNT", "/opt/agency_pkg"
)

_DEFAULT_MAX_STEPS = 20
_BASH_TIMEOUT_S = 120
_MAX_PROFILER_ATTRIBUTE_CHARS = 16 * 1024


# ---------------------------------------------------------------------------
# Length-prefixed framing -- shared by both directions (this process reading
# a request / writing a response, and native.py's host-side client doing the
# same in reverse). See module docstring for why a bare recv() isn't enough.
# ---------------------------------------------------------------------------


def _recv_exactly(sock, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"connection closed with {remaining} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_framed(sock) -> dict:
    header = _recv_exactly(sock, 8)
    (length,) = struct.unpack(">Q", header)
    body = _recv_exactly(sock, length)
    return json.loads(body.decode("utf-8"))


def _send_framed(sock, obj: dict) -> None:
    body = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack(">Q", len(body)) + body)


# ---------------------------------------------------------------------------
# Tool dispatch -- plain subprocess/file I/O, since this process already
# runs inside the container's own filesystem/process namespace. No
# agsandbox facade involved: that abstraction exists to bridge a HOST
# process into a container, which is exactly the hop this process doesn't
# need to make for its own tool calls.
# ---------------------------------------------------------------------------


def _run_bash_tool(arguments_json: str) -> str:
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    command = args.get("command", "") if isinstance(args, dict) else ""
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            timeout=_BASH_TIMEOUT_S,
            text=True,
        )
        # No tail-truncation here (an earlier version silently kept only the
        # last 40k chars) -- the full output now flows into
        # _offload_if_oversized() below, which saves it whole to a file
        # instead of throwing away everything before the tail.
        output = proc.stdout + proc.stderr
        return json.dumps({"output": output, "returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"command timed out after {_BASH_TIMEOUT_S}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Read/write/edit/glob/grep -- plain local file I/O and subprocess calls,
# same reasoning as bash above (this process already sits inside the
# container's own filesystem, so there is no `sandbox.exec()`/agtool
# bridge to reuse and no reason to invent one). The fuzzy-match replace
# pipeline (edit) and pagination logic (read) are non-trivial enough that
# hand-duplicating them here would be a real drift risk, so they're loaded
# from `agtool_pure.py` -- a module with zero relative/agency imports,
# built exactly so it can be loaded this way, by raw file path, without
# triggering `agency/__init__.py`'s own heavier import chain (see that
# module's docstring). The glob/grep command strings and output parsing
# come from the same module, so the search semantics are byte-for-byte
# identical to the host-side sandboxed tools, not a second implementation
# that could quietly diverge.
# ---------------------------------------------------------------------------


def _load_agtool_pure():
    import importlib.util

    path = f"{_AGENCY_PACKAGE_CONTAINER_MOUNT}/agency/agtool_pure.py"
    spec = importlib.util.spec_from_file_location("agtool_pure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_agllm_pure():
    """Same technique as `_load_agtool_pure()` above, for the compaction
    algorithms this loop's own compaction (see `_maybe_compact` below)
    shares with `agllm.py`'s -- see agllm_pure.py's own docstring."""
    import importlib.util

    path = f"{_AGENCY_PACKAGE_CONTAINER_MOUNT}/agency/agllm_pure.py"
    spec = importlib.util.spec_from_file_location("agllm_pure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_agprof_emit():
    """Load the stdlib-only remote emitter without importing ``agency``."""
    import importlib.util

    path = f"{_AGENCY_PACKAGE_CONTAINER_MOUNT}/agency/profiler/agprof_emit.py"
    spec = importlib.util.spec_from_file_location("agprof_emit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_agtool_pure = _load_agtool_pure()
_agllm_pure = _load_agllm_pure()
_agprof_emit = _load_agprof_emit()

_READ_DEFAULT_LIMIT = _agtool_pure.READ_DEFAULT_LIMIT


def _parse_tool_args(arguments_json: str) -> dict:
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    return args if isinstance(args, dict) else {}


def _run_read_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    file_path = str(args.get("file_path", ""))
    offset = int(args.get("offset") or 1)
    limit = int(args.get("limit") or _READ_DEFAULT_LIMIT)

    if os.path.isdir(file_path):
        try:
            entries = sorted(os.listdir(file_path))
        except OSError as e:
            return json.dumps({"error": str(e)})
        start = offset - 1
        page = entries[start : start + limit]
        return json.dumps(
            {
                "path": file_path,
                "type": "directory",
                "entries": page,
                "total": len(entries),
                "truncated": start + len(page) < len(entries),
            }
        )
    if not os.path.isfile(file_path):
        return json.dumps({"error": f"Not found: {file_path}"})
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return json.dumps({"error": str(e)})
    result = _agtool_pure.paginate_text(content, offset, limit)
    result["path"] = file_path
    return json.dumps(result)


def _run_write_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    file_path = str(args.get("file_path", ""))
    content = str(args.get("content", ""))
    try:
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps(
            {"path": file_path, "created": True, "bytes_written": len(content.encode())}
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_edit_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    file_path = str(args.get("file_path", ""))
    old_string = str(args.get("old_string", ""))
    new_string = str(args.get("new_string", ""))
    replace_all = bool(args.get("replace_all", False))

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {file_path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

    try:
        updated = _agtool_pure.replace(content, old_string, new_string, replace_all)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated)
        return json.dumps({"path": file_path, "success": True})
    except (ValueError, OSError) as e:
        return json.dumps({"error": str(e)})


def _run_glob_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    pattern = str(args.get("pattern", ""))
    path = str(args.get("path") or "/workspace")
    try:
        proc = subprocess.run(
            ["bash", "-c", _agtool_pure.glob_command(pattern, path)],
            capture_output=True,
            timeout=30,
            text=True,
        )
        return json.dumps(_agtool_pure.parse_glob_output(proc.stdout))
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_grep_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    pattern = str(args.get("pattern", ""))
    path = str(args.get("path") or "/workspace")
    include = args.get("include")
    try:
        proc = subprocess.run(
            ["bash", "-c", _agtool_pure.grep_command(pattern, path, include)],
            capture_output=True,
            timeout=30,
            text=True,
        )
        return json.dumps(_agtool_pure.parse_grep_json_output(proc.stdout))
    except Exception as e:
        return json.dumps({"error": str(e)})


_WEBFETCH_MAX_BYTES = 5 * 1024 * 1024
_WEBFETCH_DEFAULT_TIMEOUT = 30
_WEBFETCH_MAX_TIMEOUT = 120


def _run_webfetch_tool(arguments_json: str) -> str:
    import httpx
    import html2text

    args = _parse_tool_args(arguments_json)
    url = str(args.get("url", ""))
    fmt = str(args.get("format") or "markdown")
    timeout = min(int(args.get("timeout") or _WEBFETCH_DEFAULT_TIMEOUT), _WEBFETCH_MAX_TIMEOUT)

    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": "URL must start with http:// or https://"})

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; agency-bot/1.0)",
        "Accept": "text/html,text/plain,*/*;q=0.8",
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {url}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

    if len(resp.content) > _WEBFETCH_MAX_BYTES:
        return json.dumps({"error": "Response too large (>5 MB)"})

    content_type = resp.headers.get("content-type", "")
    body = resp.text
    if fmt == "markdown" and "text/html" in content_type:
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        output = h.handle(body)
    elif fmt == "text" and "text/html" in content_type:
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        output = h.handle(body)
    else:
        output = body
    return json.dumps({"url": url, "content_type": content_type, "output": output})


# Module-level store, same shape as the host-side todowrite tool -- but
# unlike that shared-process global (a real, pre-existing bug: every native
# agent sharing the host process shares the same list), this process
# belongs to exactly one agent, so a module-level list here is correct, not
# a bug waiting to bite a second agent.
_todo_store: list = []


def _run_todowrite_tool(arguments_json: str) -> str:
    global _todo_store
    args = _parse_tool_args(arguments_json)
    todos = args.get("todos")
    if todos is None:
        return json.dumps({"error": "todos field is required"})
    if not isinstance(todos, list):
        return json.dumps({"error": "todos must be a list"})
    _todo_store = [dict(t) for t in todos]
    pending = sum(1 for t in _todo_store if t.get("status") not in ("completed", "cancelled"))
    return json.dumps(
        {
            "todos": _todo_store,
            "count": len(_todo_store),
            "pending": pending,
            "output": json.dumps(_todo_store, indent=2),
        }
    )


# Same default as agtool.py's `_AgToolFields.output_offload_chars` (the
# host-side dispatch_tools()'s own threshold) -- kept as a plain constant
# since this entrypoint can't import agconfig (see module docstring on why
# it avoids the `agency.*` import chain entirely).
_TOOL_OUTPUT_OFFLOAD_CHARS = 40_000
# A plain module-level constant (not inlined) so a test can monkeypatch it
# to a writable tmp dir -- `/workspace` is only guaranteed to exist and be
# writable inside a real container.
_OFFLOAD_DIR = "/workspace/long_tool_call_outputs"


def _offload_if_oversized(fn_name: str, tc_id: str, result_content: str) -> str:
    """Mirrors agtool.py's `dispatch_tools()` offload behavior, generic
    across every tool (built-in or MCP) rather than special-cased per tool
    -- a result over the threshold is saved whole to a file and replaced
    with a short note pointing at it, so the model can `read` it instead of
    the full content bloating every subsequent turn's context. No
    `sandbox.write_file()` bridge needed the way the host-side version
    needs one: this process already runs inside the container, so it's a
    plain local file write. `read` is always available here (one of this
    entrypoint's built-ins, see `_BUILTIN_TOOL_SCHEMAS`), so -- unlike the
    host-side version -- there's no lazy tool-injection step needed either."""
    if len(result_content) <= _TOOL_OUTPUT_OFFLOAD_CHARS:
        return result_content
    try:
        parsed = json.loads(result_content)
        file_body = (
            parsed.get("content", result_content) if isinstance(parsed, dict) else result_content
        )
    except json.JSONDecodeError:
        file_body = result_content
    if not isinstance(file_body, str):
        file_body = json.dumps(file_body)
    safe_id = tc_id.replace("-", "")[:12]
    offload_path = f"{_OFFLOAD_DIR}/{fn_name}_{safe_id or uuid.uuid4().hex[:12]}.txt"
    try:
        os.makedirs(os.path.dirname(offload_path), exist_ok=True)
        with open(offload_path, "w", encoding="utf-8") as f:
            f.write(file_body)
        return json.dumps(
            {
                "note": f"Output was too large and has been saved to {offload_path}. "
                "Use the read tool to access it."
            }
        )
    except Exception as e:
        print(
            f"[native-entrypoint] WARNING: failed to offload large tool output to {offload_path}: {e}"
        )
        return result_content


_TOOL_DISPATCH = {
    "bash": _run_bash_tool,
    "read": _run_read_tool,
    "write": _run_write_tool,
    "edit": _run_edit_tool,
    "glob": _run_glob_tool,
    "grep": _run_grep_tool,
    "webfetch": _run_webfetch_tool,
    "todowrite": _run_todowrite_tool,
}


def _tool_schema(name: str, description: str, params: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": params},
    }


# Same name/params/description as the host-side sandboxed tools
# (agency/tools/*.py) -- sourced from the same `agtool_pure` module they
# themselves import their PARAMS dicts from, so there is exactly one place
# that defines what these tools look like to a model, not two copies that
# could drift.
_BUILTIN_TOOL_SCHEMAS = {
    "bash": _tool_schema(
        "bash",
        "Execute a bash command inside the sandbox and return its output.",
        _agtool_pure.BASH_PARAMS,
    ),
    "read": _tool_schema(
        "read",
        "Read a file (with optional offset/limit) or list a directory inside the sandbox.",
        _agtool_pure.READ_PARAMS,
    ),
    "write": _tool_schema(
        "write",
        "Write content to a file inside the sandbox, creating parent directories if needed.",
        _agtool_pure.WRITE_PARAMS,
    ),
    "edit": _tool_schema(
        "edit",
        "Replace a string in a file inside the sandbox. Uses fuzzy matching as fallback.",
        _agtool_pure.EDIT_PARAMS,
    ),
    "glob": _tool_schema(
        "glob",
        "Find files matching a glob pattern inside the sandbox. Defaults to searching /workspace.",
        _agtool_pure.GLOB_PARAMS,
    ),
    "grep": _tool_schema(
        "grep",
        "Search for a regex pattern in file contents inside the sandbox. Defaults to searching /workspace.",
        _agtool_pure.GREP_PARAMS,
    ),
    "webfetch": _tool_schema(
        "webfetch",
        "Fetch a URL and return its content as text, markdown, or raw HTML.",
        _agtool_pure.WEBFETCH_PARAMS,
    ),
    "todowrite": _tool_schema(
        "todowrite",
        "Update the todo list with a new set of items.",
        _agtool_pure.TODOWRITE_PARAMS,
    ),
}


# ---------------------------------------------------------------------------
# MCP-backed tools (Phase 4) -- reserve_cpu/cpu_release/daemon_release/
# submit_output, served by the shared host-side agmcp_server over the same
# kind of bind-mounted UDS bridge as agllm_terminus. Uses the real `mcp`
# client library (ensured present by native.py's launcher before this
# process starts -- see that module) rather than a hand-rolled JSON-RPC
# client, so this speaks the exact same protocol every harness's own native
# MCP client does -- no second, potentially-divergent implementation.
# ---------------------------------------------------------------------------


def _mcp_tool_schemas(mcp_sock: str, token: str) -> list:
    """Discover the MCP server's current tool set and convert each to the
    OpenAI function-calling schema shape the LLM dispatch `tools` kwarg
    needs -- fetched dynamically rather than hardcoded here, so this file
    never needs updating just because agmcp_server.py's own tool set
    changes."""
    import asyncio

    async def go():
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        transport = httpx2.AsyncHTTPTransport(uds=mcp_sock)
        http_client = httpx2.AsyncClient(
            transport=transport, headers={"Authorization": f"Bearer {token}"}
        )
        async with streamable_http_client("http://agmcp-server/mcp", http_client=http_client) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": t.input_schema,
                        },
                    }
                    for t in result.tools
                ]

    return asyncio.run(go())


def _call_mcp_tool(mcp_sock: str, token: str, tool_name: str, arguments: dict) -> dict:
    import asyncio

    async def go():
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        transport = httpx2.AsyncHTTPTransport(uds=mcp_sock)
        http_client = httpx2.AsyncClient(
            transport=transport, headers={"Authorization": f"Bearer {token}"}
        )
        async with streamable_http_client("http://agmcp-server/mcp", http_client=http_client) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                if result.structured_content is not None:
                    return result.structured_content
                if result.content:
                    return {"result": result.content[0].text}
                return {}

    return asyncio.run(go())


def _make_mcp_tool_handler(mcp_sock: str, token: str, tool_name: str):
    def handler(arguments_json: str) -> str:
        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        try:
            result = _call_mcp_tool(mcp_sock, token, tool_name, arguments)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        return json.dumps(result)

    return handler


# ---------------------------------------------------------------------------
# Custom tools (`skill.add_tools`/`replace_tools`) -- a host-authored
# `fn: Callable[[agdata], agdata]`, cloudpickled by native.py (host-side,
# where the real closure and its captured state live) and shipped here as
# base64 bytes in `req["custom_tools"]`. Loaded lazily (only if any request
# actually carries one) via `agdata_pure.py` -- a minimal, standalone-
# loadable stand-in for the real `agency.agdata.agdata`/`agerror` (see that
# module's own docstring for why a full drop-in isn't needed and can't be
# loaded here the way `agtool_pure.py`/`agllm_pure.py` are).
#
# `cloudpickle.loads` resolves any reference to the real `agdata`/`agerror`
# classes inside the shipped closure by `(module, qualname)` --
# `("agency.agdata", "agdata")` / `("agency.agdata", "agerror")` always,
# regardless of how the user's code imported them. `_load_custom_tool_fn`
# temporarily registers fake `agency`/`agency.agdata` modules in
# `sys.modules`, for the duration of the `cloudpickle.loads` call only,
# exposing THIS file's shim classes under those names so every such
# reference resolves transparently without ever importing the real
# `agency` package (which would re-trigger `agency/__init__.py`'s heavy
# eager-import chain -- exactly what this entrypoint exists to avoid).
#
# cloudpickle.loads succeeding here is not a guarantee the closure will
# actually behave correctly: a closure that captures a live host-only
# object (a real `agSandbox`, a `threading.Lock`, an open socket) rather
# than being self-contained may still unpickle without error, yet reference
# a non-functional duplicate once it's running in this process. Nothing
# can mechanically detect that in general -- see native.py's own module
# docstring for this same caveat, documented rather than silently
# mishandled, matching this codebase's existing convention for known gaps.
# ---------------------------------------------------------------------------

_agdata_pure = None
_agdata_stub_lock = None  # lazily created -- see _load_custom_tool_fn


def _load_agdata_pure():
    """Same technique as `_load_agtool_pure()`/`_load_agllm_pure()` above,
    for the minimal `agdata`/`agerror` stand-in a shipped custom tool's
    `fn` is called with. Cached at module scope (imported at most once per
    process) since, unlike the built-in tools' pure modules, this one is
    only ever needed when a request actually carries `custom_tools`."""
    global _agdata_pure
    if _agdata_pure is None:
        import importlib.util

        path = f"{_AGENCY_PACKAGE_CONTAINER_MOUNT}/agency/agdata_pure.py"
        spec = importlib.util.spec_from_file_location("agdata_pure", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _agdata_pure = module
    return _agdata_pure


def _load_custom_tool_fn(fn_b64: str):
    """Loads a shipped closure via `cloudpickle.loads`, with the
    `agency`/`agency.agdata` `sys.modules` stub installed for the DURATION
    OF THIS CALL ONLY -- not left registered afterward. This matters far
    beyond tidiness: in a real container process, nothing else ever imports
    the real `agency` package, so a permanent stub would be harmless there;
    but the fast, no-Docker test seam (`NativeLoopHarness`) runs this exact
    function IN-PROCESS, in the same Python process as the rest of the test
    suite, where the real `agency` package genuinely is already imported.
    A stub left registered permanently would silently replace `sys.modules
    ["agency"]`/`["agency.agdata"]` with these fakes for every test that
    runs afterward in the same pytest session -- a real, confirmed
    regression caught by this phase's own full-suite verification pass
    (agteam/agsync/agent's structured-output tests all depend on the REAL
    `agdata`'s Future-wrapping, which this minimal shim deliberately omits).
    Once `cloudpickle.loads` returns, the reconstructed function's own
    `__globals__` already holds direct references to the resolved shim
    classes -- pickle's class-reference resolution happens exactly once,
    during this call, not on every later invocation of the loaded `fn` --
    so restoring `sys.modules` immediately afterward is safe. A fresh
    throwaway module is always used for the `agency` stub (never the real
    `sys.modules["agency"]`, even if already present) so this never mutates
    the real package object's own attributes either, only the `sys.modules`
    dict entries -- those are what's saved and restored. Lock-protected
    since the entrypoint's socket server is threaded (`_Server`) and two
    custom-tool loads could otherwise race on the same process-global
    `sys.modules` dict."""
    import base64
    import sys
    import threading
    import types

    import cloudpickle

    global _agdata_stub_lock
    if _agdata_stub_lock is None:
        _agdata_stub_lock = threading.Lock()

    agdata_pure_mod = _load_agdata_pure()
    agency_stub = types.ModuleType("agency")
    agdata_stub = types.ModuleType("agency.agdata")
    agdata_stub.agdata = agdata_pure_mod.agdata
    agdata_stub.agerror = agdata_pure_mod.agerror
    agdata_stub.AgError = agdata_pure_mod.AgError
    agency_stub.agdata = agdata_stub

    with _agdata_stub_lock:
        saved_agency = sys.modules.get("agency")
        saved_agdata = sys.modules.get("agency.agdata")
        sys.modules["agency"] = agency_stub
        sys.modules["agency.agdata"] = agdata_stub
        try:
            return cloudpickle.loads(base64.b64decode(fn_b64))
        finally:
            if saved_agency is not None:
                sys.modules["agency"] = saved_agency
            else:
                sys.modules.pop("agency", None)
            if saved_agdata is not None:
                sys.modules["agency.agdata"] = saved_agdata
            else:
                sys.modules.pop("agency.agdata", None)


def _make_custom_tool_handler(tool_name: str, fn_b64: str):
    """Returns a handler(arguments_json) -> str, same contract as every
    other entry in `_TOOL_DISPATCH`/`_make_mcp_tool_handler`. The shipped
    `fn` is loaded once, lazily, on the FIRST call to this tool (not at
    tool-list-construction time) -- so a `cloudpickle.loads` failure (e.g.
    a version mismatch, or a third-party import the closure needs that
    isn't installed in this minimal container) surfaces as a normal tool-
    call error the model can see and react to, rather than aborting the
    whole run before the model ever gets a turn."""
    loaded: "list" = []  # 0 or 1 element -- lazy-init cache, closed over below

    def handler(arguments_json: str) -> str:
        if not loaded:
            try:
                loaded.append(_load_custom_tool_fn(fn_b64))
            except Exception as e:
                return json.dumps(
                    {
                        "error": f"tool '{tool_name}' failed to load inside the container: "
                        f"{type(e).__name__}: {e}"
                    }
                )
        fn = loaded[0]
        agdata_pure_mod = _load_agdata_pure()
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            arg = agdata_pure_mod.agdata(**(args if isinstance(args, dict) else {}))
            result = fn(arg)
            return json.dumps(result.to_dict())
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    return handler


# ---------------------------------------------------------------------------
# LLM dispatch, via agllm_terminus -- never a direct call to a real
# provider from inside the container (no real credentials live here).
# ---------------------------------------------------------------------------


_DISPATCH_MAX_RETRIES = 5
_DISPATCH_BASE_BACKOFF_S = 1.0
_DISPATCH_MAX_BACKOFF_S = 20.0


def _dispatch_retry_backoff_s(attempt: int) -> float:
    import random

    return random.uniform(0, min(_DISPATCH_MAX_BACKOFF_S, _DISPATCH_BASE_BACKOFF_S * (2**attempt)))


def _dispatch_via_terminus(
    terminus_sock: str,
    token: str,
    kwargs: dict,
    timeout_s: float = 300,
    max_retries: int = _DISPATCH_MAX_RETRIES,
    profiler=None,
) -> dict:
    """POST to the terminus with `stream=True` and reassemble the streamed
    chunks into `{"message": {"role": "assistant", "content", "tool_calls"},
    "usage": {"prompt_tokens", "completion_tokens", "total_tokens"} | None}`
    -- mirrors `agllm.py`'s own reassembly (accumulate content deltas by
    concatenation, tool-call argument deltas by index; see `agllm.py`'s
    `call()`). **Always streaming, never `stream=False`**: some
    backends' non-streaming shortcut is a narrow, single-purpose helper that
    doesn't support tool calls or even basic serialization at all --
    confirmed by a real `500` (`AttributeError: '_AnthropicNonStreamResponse'
    object has no attribute 'model_dump'`) against the real Bedrock/Anthropic
    backend during development. Streaming is the path every backend actually
    supports fully, since it's what `agllm.py`'s own native call site always
    uses.

    **This is where native's own retry-on-transient-error lives, not
    agllm_terminus.py.** The terminus deliberately makes exactly one
    attempt per request and classifies failures (503 = safe to retry, 400
    = don't bother) rather than retrying itself -- see that module's own
    comment for why: it always streams, so once it commits to a response
    it can no longer safely resend, and retrying underneath whatever an
    external harness's own CLI already does on its end would stack two
    uncoordinated retry/timeout layers. Native's loop has no such CLI
    underneath it, so it's the one place that actually needs this
    protection, and it's scoped narrowly: only a 503 (the terminus's own
    honest "nothing was sent yet" signal) or a failure to even reach the
    terminus (`httpx.ConnectError`/`TimeoutException`, opening the
    connection itself) triggers a retry. A failure partway through
    `resp.iter_lines()` -- after real content may already have been
    reassembled -- is NOT caught here and propagates to the caller
    (`_Handler.handle()`'s own try/except), since retrying after partial
    output would silently corrupt the conversation, same reasoning as the
    terminus's own no-retry-after-first-chunk rule."""
    import httpx
    import time

    if profiler is None:
        profiler = _agprof_emit.RemoteProfilerEmitter(None, token)
    kwargs = dict(kwargs)
    kwargs["stream"] = True
    transport = httpx.HTTPTransport(uds=terminus_sock)

    last_error = "dispatch failed with no attempts made"
    for attempt in range(max_retries):
        content_parts: "list[str]" = []
        tool_calls_raw: "dict[int, dict]" = {}
        usage: "dict | None" = None
        try:
            with httpx.Client(
                transport=transport, base_url="http://agllm-terminus", timeout=timeout_s
            ) as client:
                with client.stream(
                    "POST", "/internal/dispatch", json={"token": token, "kwargs": kwargs}
                ) as resp:
                    if resp.status_code == 503:
                        resp.read()
                        last_error = f"terminus dispatch failed: {resp.status_code} {resp.text}"
                        if attempt < max_retries - 1:
                            delay_s = _dispatch_retry_backoff_s(attempt)
                            with profiler.span(
                                "llm:retry_backoff",
                                metadata={"attempt": attempt, "delay_ms": delay_s * 1000},
                            ):
                                time.sleep(delay_s)
                            continue
                        return {"error": last_error}
                    if resp.status_code != 200:
                        resp.read()
                        return {
                            "error": f"terminus dispatch failed: {resp.status_code} {resp.text}"
                        }

                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[len("data: ") :]
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        # The usage-bearing chunk (stream_options.include_usage,
                        # already turned on terminus-side) typically carries an
                        # empty choices list -- check it independent of the
                        # choices/delta parsing below, same as agllm.py's own
                        # `call()` does for the host-side loop.
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        for tc_delta in delta.get("tool_calls") or []:
                            idx = tc_delta.get("index", 0)
                            slot = tool_calls_raw.setdefault(
                                idx,
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tc_delta.get("id"):
                                slot["id"] = tc_delta["id"]
                            fn_delta = tc_delta.get("function") or {}
                            if fn_delta.get("name"):
                                slot["function"]["name"] += fn_delta["name"]
                            if fn_delta.get("arguments"):
                                slot["function"]["arguments"] += fn_delta["arguments"]

            message = {"role": "assistant", "content": "".join(content_parts) or None}
            if tool_calls_raw:
                message["tool_calls"] = [tool_calls_raw[i] for i in sorted(tool_calls_raw)]
            return {"message": message, "usage": usage}

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_error = f"terminus unreachable: {e}"
            if attempt < max_retries - 1:
                delay_s = _dispatch_retry_backoff_s(attempt)
                with profiler.span(
                    "llm:retry_backoff",
                    metadata={"attempt": attempt, "delay_ms": delay_s * 1000},
                ):
                    time.sleep(delay_s)
                continue
            return {"error": last_error}

    return {"error": last_error}


# ---------------------------------------------------------------------------
# Compaction -- same algorithm agllm.py's own maybe_compact()/compact() use
# (agllm_pure, shared -- see that module's docstring), just dispatched
# through THIS loop's own terminus connection (_dispatch_via_terminus) for
# the summarization call, since there's no agllm.py/agllm.call() available
# here. Belongs in this loop, not agllm_terminus.py: the terminus never
# holds a conversation's `messages` list at all, it only ever sees
# whatever one request hands it -- compaction is inherently the job of
# whoever owns the growing history, which for native is this loop (for an
# external harness, it's that harness's own already-built-in auto-compact,
# nothing to do here).
# ---------------------------------------------------------------------------


def _fetch_context_limit(terminus_sock: str, token: str, timeout_s: float = 30) -> "int | None":
    """Ask the terminus for this agent's model's context window --
    reuses agllm.fetch_context_limit's real lookup (model listing/known
    static limits), since this loop has no agllm.py of its own to compute
    it locally. Returns None on any failure -- compaction just never
    triggers in that case, same as agllm.py's own graceful-when-unknown
    behavior (agllm.maybe_compact returns early when context_limit is
    None)."""
    import httpx

    try:
        transport = httpx.HTTPTransport(uds=terminus_sock)
        with httpx.Client(
            transport=transport, base_url="http://agllm-terminus", timeout=timeout_s
        ) as client:
            resp = client.post("/internal/context_limit", json={"token": token})
            if resp.status_code != 200:
                return None
            return resp.json().get("context_limit")
    except Exception:
        return None


def _maybe_compact(
    messages: list,
    context_limit: "int | None",
    terminus_sock: str,
    token: str,
    model: str,
    previous_summary: "str | None",
    profiler,
) -> "tuple[list, str | None]":
    """Compact `messages` if they're near `context_limit`. Returns
    (messages, previous_summary) -- both unchanged if compaction doesn't
    trigger or the summarization call itself fails (best-effort: a failed
    summary attempt must not crash the whole run; the next turn's own
    dispatch will surface a real error if the context truly is too large,
    just one step later than agllm.py's own compact()-raises behavior)."""
    if context_limit is None:
        return messages, previous_summary
    token_count = _agllm_pure.estimate_messages_tokens(messages)
    if not _agllm_pure.should_compact(token_count, context_limit):
        return messages, previous_summary
    sys_msg, task_input, head, tail = _agllm_pure.split_for_compaction(messages, context_limit)
    if not head:
        return messages, previous_summary
    head = _agllm_pure.prune_tool_outputs(head)
    summary_messages = _agllm_pure.build_summary_prompt_messages(task_input, head, previous_summary)
    with profiler.span(
        "llm:compact",
        metadata={"messages_before": len(messages), "compacted_head_messages": len(head)},
    ) as compact_span:
        resp = _dispatch_via_terminus(
            terminus_sock,
            token,
            {"model": model, "messages": summary_messages},
            profiler=profiler,
        )
        if "error" in resp:
            compact_span.annotate(outcome="failure", error=str(resp["error"]))
    if "error" in resp:
        return messages, previous_summary
    summary = (resp["message"].get("content") or "").strip()
    return _agllm_pure.assemble_compacted_messages(sys_msg, task_input, summary, tail), summary


# ---------------------------------------------------------------------------
# Pause/inbox check-in -- bridges the same two calls execute_react() makes
# in-process at the top of every ReAct iteration (ag._check_pause(),
# ag._drain_inbox()) via the shared agharness_messenger.py. See that
# module's own docstring for why this only meaningfully helps native (an
# external harness's internal loop is opaque, no mid-turn injection hook
# exists for it).
# ---------------------------------------------------------------------------


def _check_in(messenger_sock: str, token: str) -> list:
    """Blocks (host-side, inside the messenger's own request handler)
    until this agent's pause clears, then returns any pending inbox
    messages as ordinary user-role message dicts, ready to append
    directly to `messages`. Best-effort: any failure here (messenger
    unreachable, timeout) must not crash the run -- treated as "nothing to
    check in," same fail-open philosophy as agpolicy_hook.py's own
    network-failure handling. No timeout is set on the request itself
    (`httpx.Timeout(None)`): blocking indefinitely while paused is the
    entire point, not a fault to guard against."""
    import httpx

    try:
        transport = httpx.HTTPTransport(uds=messenger_sock)
        with httpx.Client(
            transport=transport, base_url="http://agharness-messenger", timeout=httpx.Timeout(None)
        ) as client:
            resp = client.post("/internal/check_in", json={"token": token})
            if resp.status_code != 200:
                return []
            return resp.json().get("messages") or []
    except Exception:
        return []


def _run_react_loop(req: dict) -> dict:
    profiler = _agprof_emit.RemoteProfilerEmitter(req.get("profiler_sock"), req["token"])
    try:
        response = _run_react_loop_inner(req, profiler)
    finally:
        # The host unregisters this token immediately after receiving our
        # response. Flush all span-end events before that response is sent.
        profiler.close()
    response["profiler_dropped_events"] = profiler.dropped_events
    return response


def _run_react_loop_inner(req: dict, profiler) -> dict:
    token = req["token"]
    terminus_sock = req["terminus_sock"]
    model = req.get("model", "")
    messages = list(req["messages"])
    max_steps = req.get("max_steps") or _DEFAULT_MAX_STEPS
    context_limit = _fetch_context_limit(terminus_sock, token)
    # Accumulated across every dispatch this call makes (one per ReAct
    # step) -- returned to native.py so it can update prev_ctx.total_
    # input_tokens/total_output_tokens for real, same contract execute_
    # react()'s llm_result gives it. Compaction's own summarization
    # dispatch is deliberately not counted here, matching agllm.py's own
    # compact() which doesn't fold its summary call's usage back into the
    # skill's running totals either.
    total_input_tokens = 0
    total_output_tokens = 0
    previous_summary: "str | None" = None
    messenger_sock = req.get("messenger_sock")

    # Per-request dispatch table: _TOOL_DISPATCH's local built-ins
    # (bash/read/write/edit/glob/grep/webfetch/todowrite) are always
    # available -- no caller needs to enumerate them, this entrypoint owns
    # its own tool set -- unless `suppress_builtins` is set (native.py sends
    # this when `skill.replace_tools is not None`, matching the deleted
    # `_build_toolkit()`'s own precedence: replace_tools replaces the whole
    # set, add_tools only ever extends it). Plus, if this launch was given
    # an MCP server bridge, that server's CURRENT tool set (resource
    # control, output submission, ask_human -- Phase 4) discovered
    # dynamically -- always merged in regardless of suppress_builtins,
    # since those are host control-plane tools, not part of the skill's own
    # tool-set choice. `req["custom_tools"]` (add_tools/replace_tools,
    # cloudpickled host-side by native.py -- see this file's module
    # docstring) are added last, skipped on name collision. Built fresh per
    # request rather than at module scope, since each MCP/custom-tool
    # entry's handler closes over THIS request's own mcp_sock/token/fn.
    suppress_builtins = bool(req.get("suppress_builtins"))
    dispatch = {} if suppress_builtins else dict(_TOOL_DISPATCH)
    tools = [] if suppress_builtins else list(_BUILTIN_TOOL_SCHEMAS.values())
    _have_tool = set() if suppress_builtins else set(_BUILTIN_TOOL_SCHEMAS.keys())
    mcp_sock = req.get("mcp_server_sock")
    if mcp_sock:
        for schema in _mcp_tool_schemas(mcp_sock, token):
            tool_name = schema["function"]["name"]
            tools.append(schema)
            _have_tool.add(tool_name)
            dispatch[tool_name] = _make_mcp_tool_handler(mcp_sock, token, tool_name)
    for ct in req.get("custom_tools") or []:
        tool_name = ct["name"]
        if tool_name in _have_tool:
            continue
        tools.append(_tool_schema(tool_name, ct.get("description", ""), ct.get("params") or {}))
        _have_tool.add(tool_name)
        dispatch[tool_name] = _make_custom_tool_handler(tool_name, ct["fn_b64"])

    turn_offset = int(req.get("profiler_turn_offset") or 0)
    for step in range(max_steps):
        turn_index = turn_offset + step
        with profiler.span(
            f"turn{turn_index}",
            span_id=f"turn:{turn_index}",
            metadata={"turn_index": turn_index},
        ) as turn_span:
            # Same order execute_react() uses in-process: check pause/drain
            # inbox, THEN compact, THEN dispatch.
            if messenger_sock:
                messages.extend(_check_in(messenger_sock, token))
            messages, previous_summary = _maybe_compact(
                messages,
                context_limit,
                terminus_sock,
                token,
                model,
                previous_summary,
                profiler,
            )
            kwargs = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            resp = _dispatch_via_terminus(terminus_sock, token, kwargs, profiler=profiler)
            if "error" in resp:
                turn_span.annotate(outcome="failure", error=str(resp["error"]))
                return {
                    "status": "error",
                    "message": str(resp["error"]),
                    "turn_count": step + 1,
                }

            usage = resp.get("usage") or {}
            total_input_tokens += usage.get("prompt_tokens", 0) or 0
            total_output_tokens += usage.get("completion_tokens", 0) or 0

            message = resp["message"]
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            turn_span.annotate(tool_calls=len(tool_calls))
            if not tool_calls:
                return {
                    "status": "done",
                    "messages": messages,
                    "final_text": message.get("content") or "",
                    "usage": {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                    },
                    "turn_count": step + 1,
                }

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                handler = dispatch.get(fn_name)
                with profiler.span(
                    f"tool:{fn_name}",
                    span_id=f"tool:{tc['id']}",
                    metadata={
                        "tool_call_id": tc["id"],
                        "arguments": fn_args[:_MAX_PROFILER_ATTRIBUTE_CHARS],
                    },
                ) as tool_span:
                    result_content = (
                        handler(fn_args)
                        if handler is not None
                        else json.dumps({"error": f"unknown tool: {fn_name}"})
                    )
                    result_content = _offload_if_oversized(fn_name, tc["id"], result_content)
                    tool_ok = handler is not None
                    try:
                        parsed_result = json.loads(result_content)
                        if isinstance(parsed_result, dict) and "error" in parsed_result:
                            tool_ok = False
                    except (json.JSONDecodeError, TypeError):
                        pass
                    tool_span.annotate(
                        outcome="success" if tool_ok else "failure",
                        result=result_content[:_MAX_PROFILER_ATTRIBUTE_CHARS],
                    )
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result_content}
                )

    return {
        "status": "error",
        "message": f"exceeded max_steps={max_steps} without a final answer",
        "turn_count": max_steps,
    }


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            req = _recv_framed(self.request)
        except (json.JSONDecodeError, UnicodeDecodeError, ConnectionError, struct.error):
            return  # malformed/disconnected request -- nothing sane to reply with

        op = req.get("op")
        if op == "ping":
            # A plain filesystem check, not `import agency` -- see module
            # docstring for why a live import doesn't belong in this
            # foundation's health check.
            marker = f"{_AGENCY_PACKAGE_CONTAINER_MOUNT}/agency/agskill.py"
            resp = {
                "status": "ok",
                "agency_package_visible": os.path.isfile(marker),
                "agency_package_marker_path": marker,
                "pid": os.getpid(),
            }
        elif op == "run":
            try:
                resp = _run_react_loop(req)
            except Exception as e:
                resp = {"status": "error", "message": f"{type(e).__name__}: {e}"}
        else:
            resp = {"status": "error", "message": f"unknown op {op!r}"}

        _send_framed(self.request, resp)


class _Server(socketserver.ThreadingUnixStreamServer):
    # Threaded: a `run` request can take a while (multiple LLM turns), and
    # a `ping` health check should still get an immediate answer rather
    # than queueing behind it.
    allow_reuse_address = True
    daemon_threads = True


def main(argv: "list[str] | None" = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit("usage: python3 _native_in_container_entrypoint.py <uds-path>")
    sock_path = argv[0]
    if os.path.exists(sock_path):
        os.remove(sock_path)

    server = _Server(sock_path, _Handler)
    # The entrypoint runs as the container's root user, while the host-side
    # caller commonly runs as an unprivileged user. The socket lives in a
    # bind mount, so the default root-owned 0755 socket rejects that caller
    # with EACCES even though it can see the path. Restrict this permission
    # change to the per-sandbox, randomly named UDS itself; the directory is
    # still the host-managed gateway mount.
    os.chmod(sock_path, 0o666)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if os.path.exists(sock_path):
            os.remove(sock_path)


if __name__ == "__main__":
    main()
