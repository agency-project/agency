"""Capture both native API boundaries inside the isolated study daemon."""

from pathlib import Path

if Path("/var/run/agency_logs").is_dir():
    import json
    import threading
    import agency.harness.adapters.claude_code as claude_module
    from agency.harness.adapters.claude_code import ClaudeCodeAdapter
    from agency.harness.adapters.codex import CodexAdapter

    # Session-title suppression saves a model call in normal Agency runs.
    # This experiment explicitly disables every optimization, including auxiliary calls.
    claude_module._is_session_title_request = lambda body: False
    original_register = ClaudeCodeAdapter.register

    def transparent_register(self, app, router):
        original_register(self, app, router)
        # The normal adapter estimates count_tokens locally. The baseline must
        # instead use the same provider counter as direct Claude Code.
        from fastapi.responses import Response
        from starlette.routing import request_response
        import httpx

        async def count_tokens(request):
            token = claude_module.extract_bearer_token(request)
            if not token or not router.validate_token(token):
                return Response(status_code=401)
            settings = json.loads(Path("/study-control/recorder.json").read_text())
            headers = {
                k: v
                for k, v in request.headers.items()
                if k.lower() in {"anthropic-beta", "anthropic-version", "content-type"}
            }
            headers["x-api-key"] = settings["proxy_token"]
            with_body = await request.body()
            async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
                response = await client.post(
                    settings["base_url"] + "/v1/messages/count_tokens",
                    headers=headers,
                    content=with_body,
                )
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
            )

        for route in app.routes:
            if getattr(route, "path", None) == "/v1/messages/count_tokens":
                route.app = request_response(count_tokens)

    ClaudeCodeAdapter.register = transparent_register
    lock = threading.Lock()
    counts = {}

    def allocate(name, suffix):
        p = Path("/var/run/agency_logs") / name
        p.mkdir(exist_ok=True)
        with lock:
            n = counts.get(name, 0)
            counts[name] = n + 1
        return p / f"{n:03d}{suffix}"

    def patch(cls, header_names):
        original = cls._format_context_harness_to_agency
        stream = cls._format_agency_stream_to_harness
        response = cls._format_context_agency_to_harness

        def incoming(self, body, headers=None, query_params=None):
            p = allocate("native-incoming", ".request.json")
            p.write_text(json.dumps(body, indent=2) + "\n")
            safe = {
                k.lower(): v
                for k, v in (headers or {}).items()
                if k.lower() in header_names
                or (cls is CodexAdapter and k.lower().startswith("x-codex-"))
            }
            p.with_name(p.name.replace("request.json", "headers.json")).write_text(
                json.dumps(safe, indent=2) + "\n"
            )
            return original(self, body, headers, query_params)

        def outgoing(self, items, model, **kwargs):
            p = allocate("native-outgoing", ".response.sse")
            with p.open("w") as out:
                for chunk in stream(self, items, model, **kwargs):
                    out.write(chunk)
                    out.flush()
                    yield chunk

        def nonstream(self, item, model, **kwargs):
            value = response(self, item, model, **kwargs)
            allocate("native-outgoing", ".response.json").write_text(
                json.dumps(value, indent=2) + "\n"
            )
            return value

        cls._format_context_harness_to_agency = incoming
        cls._format_agency_stream_to_harness = outgoing
        cls._format_context_agency_to_harness = nonstream

    patch(ClaudeCodeAdapter, {"anthropic-beta", "anthropic-version"})
    patch(CodexAdapter, {"openai-beta", "x-openai-api-version", "originator", "user-agent"})
