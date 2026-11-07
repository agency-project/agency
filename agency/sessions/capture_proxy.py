"""Opt-in Claude Messages capture through a process-scoped HTTPS proxy.

Only api.anthropic.com:443 is decrypted, using an ephemeral local CA. Other
CONNECT destinations remain opaque tunnels. Native credentials and hostname
are retained; only Messages bodies enter the existing capture/SQLite pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import os
import ssl
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import proxy_bypass_environment

import anyio
import certifi
import h11
import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from .capture import CaptureGateway, _headers, _private_file, _validate_native_upstream

_API_HOST = "api.anthropic.com"


def _network_environment(env, cwd):
    names = (
        "NODE_EXTRA_CA_CERTS",
        "SSL_CERT_FILE",
        "CLAUDE_CODE_CERT_STORE",
        "CLAUDE_CODE_CLIENT_CERT",
        "CLAUDE_CODE_CLIENT_KEY",
        "CLAUDE_CODE_CLIENT_KEY_PASSPHRASE",
        "https_proxy",
        "HTTPS_PROXY",
        "http_proxy",
        "HTTP_PROXY",
        "no_proxy",
        "NO_PROXY",
    )
    result = {name: str(env[name]) for name in names if env.get(name) is not None}
    for name in (
        "NODE_EXTRA_CA_CERTS",
        "SSL_CERT_FILE",
        "CLAUDE_CODE_CLIENT_CERT",
        "CLAUDE_CODE_CLIENT_KEY",
    ):
        if result.get(name) and not Path(result[name]).is_absolute():
            result[name] = str((Path(cwd) / result[name]).resolve())
    return result


@dataclass(frozen=True)
class ClaudeNativeCaptureConfig:
    api_base_url: str = "https://api.anthropic.com"
    native = True
    harness = "claude_code"
    model = None
    key_env = None

    def __post_init__(self):
        _validate_native_upstream(self.api_base_url)

    def upstream(self, headers, path, query):
        return self.api_base_url.rstrip("/") + "/" + path + ("?" + query if query else "")


def _client_tls(env):
    """Trust the user's original CA material, never our interception CA."""
    try:
        stores = {s.strip() for s in env.get("CLAUDE_CODE_CERT_STORE", "bundled,system").split(",")}
        if not stores or not stores <= {"bundled", "system"}:
            raise ValueError("unsupported Claude certificate store selection")
        context = (
            ssl.create_default_context(cafile=env.get("SSL_CERT_FILE"))
            if "system" in stores
            else ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        )
        if "bundled" in stores:
            context.load_verify_locations(certifi.where())
        if env.get("NODE_EXTRA_CA_CERTS"):
            context.load_verify_locations(env["NODE_EXTRA_CA_CERTS"])
        if env.get("CLAUDE_CODE_CLIENT_CERT"):
            context.load_cert_chain(
                env["CLAUDE_CODE_CLIENT_CERT"],
                env.get("CLAUDE_CODE_CLIENT_KEY"),
                env.get("CLAUDE_CODE_CLIENT_KEY_PASSPHRASE"),
            )
        return context
    except OSError as exc:
        raise ValueError("cannot load Claude capture CA or client certificate") from exc


def _certificates(directory: Path, extra_ca: str | None):
    """Private, non-exported CA and API leaf; no system trust changes."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    serial = x509.random_serial_number()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"Agency capture {serial:x}")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(serial)
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=7))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _API_HOST)]))
        .issuer_name(ca.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=7))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(_API_HOST)]), critical=False)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )
    ca_bytes = ca.public_bytes(serialization.Encoding.PEM)
    bundle = (Path(extra_ca).read_bytes() + b"\n" if extra_ca else b"") + ca_bytes
    for filename, data in {
        "ca.pem": ca_bytes,
        "trust.pem": bundle,
        "leaf.pem": leaf.public_bytes(serialization.Encoding.PEM) + ca_bytes,
        "leaf.key": leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    }.items():
        with _private_file(directory / filename) as file:
            file.write(data)
    # The CA signing key is never saved, and cannot sign another hostname later.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(directory / "leaf.pem", directory / "leaf.key")
    context.set_alpn_protocols(["http/1.1"])
    return context


class ClaudeCaptureGateway(CaptureGateway):
    def __init__(self, session, config):
        self.original_env = getattr(session, "_capture_network_environment", None)
        if self.original_env is None:
            self.original_env = _network_environment(os.environ, session.cwd)
        super().__init__(session, config, tls_context=_client_tls(self.original_env))
        self._thread = None
        self._loop = None
        self._server = None
        self._temporary = None
        self._connections = set()
        self.port = None

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="agency-claude-capture-")
        self.directory = Path(self._temporary.name)
        try:
            self._server_tls = _certificates(
                self.directory, self.original_env.get("NODE_EXTRA_CA_CERTS")
            )
            ready = threading.Event()
            errors = []

            def run():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                try:
                    self._server = self._loop.run_until_complete(
                        asyncio.start_server(self._connection, "127.0.0.1", 0)
                    )
                    self.port = self._server.sockets[0].getsockname()[1]
                except Exception as exc:
                    errors.append(exc)
                finally:
                    ready.set()
                try:
                    if not errors:
                        self._loop.run_forever()
                finally:
                    self._loop.close()

            self._thread = threading.Thread(target=run, name="claude-capture-proxy", daemon=True)
            self._thread.start()
            if not ready.wait(10) or errors:
                raise RuntimeError("Claude capture proxy could not start")
        except Exception:
            self.close()
            raise

    def configure_environment(self, env):
        proxy = f"http://agency:{self.session.host_token}@127.0.0.1:{self.port}"
        env.update(
            HTTPS_PROXY=proxy,
            https_proxy=proxy,
            NO_PROXY="localhost,127.0.0.1,::1",
            no_proxy="localhost,127.0.0.1,::1",
            NODE_EXTRA_CA_CERTS=str(self.directory / "trust.pem"),
        )

    def _proxy_for(self, host):
        env = self.original_env
        bypass = (env.get("no_proxy") or env.get("NO_PROXY") or "").replace(" ", ",")
        if proxy_bypass_environment(host, {"no": bypass}):
            return None
        return (
            env.get("https_proxy")
            or env.get("HTTPS_PROXY")
            or env.get("http_proxy")
            or env.get("HTTP_PROXY")
        )

    def _client_options(self, url):
        proxy = self._proxy_for(urlsplit(url).hostname)
        if proxy and urlsplit(proxy).scheme == "https":
            proxy = httpx.Proxy(proxy, ssl_context=self._tls)
        return {"trust_env": False, "verify": self._tls, "proxy": proxy}

    def close(self):
        if self._loop is not None and self._loop.is_running():

            async def shutdown():
                self._server.close()
                tasks = list(self._connections)
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                await self._server.wait_closed()

            asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(timeout=10)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(5)
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    async def _connection(self, reader, writer):
        task = asyncio.current_task()
        self._connections.add(task)
        try:
            connection = h11.Connection(h11.SERVER, max_incomplete_event_size=65536)
            request, _ = await self._read_request(connection, reader)
            if request is None:
                return
            headers = httpx.Headers(request.headers)
            try:
                authorization = base64.b64decode(
                    headers.get("proxy-authorization", "").removeprefix("Basic "), validate=True
                )
            except ValueError:
                authorization = b""
            if not hmac.compare_digest(
                authorization, ("agency:" + self.session.host_token).encode()
            ):
                writer.write(
                    connection.send(
                        h11.Response(
                            status_code=407,
                            headers=[
                                ("proxy-authenticate", 'Basic realm="Agency"'),
                                ("content-length", "0"),
                            ],
                        )
                    )
                    + connection.send(h11.EndOfMessage())
                )
                await writer.drain()
                return
            if request.method != b"CONNECT":
                writer.write(
                    connection.send(
                        h11.Response(status_code=405, headers=[("content-length", "0")])
                    )
                    + connection.send(h11.EndOfMessage())
                )
                await writer.drain()
                return
            target = urlsplit("//" + request.target.decode("ascii"))
            if (
                not target.hostname
                or target.username
                or target.password
                or target.path
                or target.query
                or target.fragment
                or not target.port
            ):
                return
            if target.hostname == _API_HOST and target.port == 443:
                writer.write(connection.send(h11.Response(status_code=200, headers=[])))
                await writer.drain()
                await writer.start_tls(self._server_tls, ssl_handshake_timeout=10)
                await self._http(reader, writer)
            else:
                upstream_reader, upstream_writer = await self._tunnel(target.hostname, target.port)
                try:
                    writer.write(connection.send(h11.Response(status_code=200, headers=[])))
                    await writer.drain()
                    trailing, _ = connection.trailing_data
                    if trailing:
                        upstream_writer.write(trailing)
                        await upstream_writer.drain()
                    async with anyio.create_task_group() as group:

                        async def relay(source, destination):
                            try:
                                while data := await source.read(65536):
                                    destination.write(data)
                                    await destination.drain()
                            except OSError:
                                pass
                            finally:
                                group.cancel_scope.cancel()

                        group.start_soon(relay, reader, upstream_writer)
                        group.start_soon(relay, upstream_reader, writer)
                finally:
                    await self._close_writer(upstream_writer)
        except (OSError, ValueError, h11.ProtocolError, TimeoutError):
            # Never log proxy auth, provider credentials or arbitrary TLS errors.
            pass
        finally:
            try:
                await self._close_writer(writer)
            finally:
                self._connections.discard(task)

    @staticmethod
    async def _close_writer(writer):
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), 1)
        except (OSError, TimeoutError):
            writer.transport.abort()
        except asyncio.CancelledError:
            writer.transport.abort()
            raise

    async def _tunnel(self, host, port):
        """Honor the original proxy/bypass routing without decrypting TLS."""
        proxy = self._proxy_for(host)
        if not proxy:
            return await asyncio.wait_for(asyncio.open_connection(host, port), 10)
        target = urlsplit(proxy)
        if target.scheme not in {"http", "https"} or not target.hostname:
            raise ValueError("unsupported upstream proxy")
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                target.hostname,
                target.port or (443 if target.scheme == "https" else 80),
                ssl=self._tls if target.scheme == "https" else None,
            ),
            10,
        )
        try:
            authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
            headers = [("host", authority)]
            if target.username is not None:
                credential = unquote(target.username) + ":" + unquote(target.password or "")
                headers.append(
                    (
                        "proxy-authorization",
                        "Basic " + base64.b64encode(credential.encode()).decode(),
                    )
                )
            connection = h11.Connection(h11.CLIENT)
            writer.write(
                connection.send(h11.Request(method="CONNECT", target=authority, headers=headers))
                + connection.send(h11.EndOfMessage())
            )
            await writer.drain()
            while True:
                event = connection.next_event()
                if event is h11.NEED_DATA:
                    connection.receive_data(await asyncio.wait_for(reader.read(65536), 10))
                elif isinstance(event, h11.Response):
                    if event.status_code != 200:
                        raise OSError("upstream proxy rejected CONNECT")
                    return reader, writer
                else:
                    raise OSError("invalid upstream proxy response")
        except BaseException:
            await self._close_writer(writer)
            raise

    @staticmethod
    async def _read_request(connection, reader):
        request, body = None, bytearray()
        while True:
            event = connection.next_event()
            if event is h11.NEED_DATA:
                connection.receive_data(await reader.read(65536))
            elif isinstance(event, h11.Request):
                request = event
            elif isinstance(event, h11.Data):
                body.extend(event.data)
            elif isinstance(event, h11.EndOfMessage):
                return request, bytes(body)
            elif isinstance(event, h11.ConnectionClosed):
                return None, b""
            else:
                raise h11.RemoteProtocolError("unexpected proxy event")

    async def _http(self, reader, writer):
        connection = h11.Connection(h11.SERVER, max_incomplete_event_size=65536)
        while True:
            incoming, raw = await self._read_request(connection, reader)
            if incoming is None:
                return
            headers = httpx.Headers(incoming.headers)
            target = urlsplit(incoming.target.decode("ascii"))
            if (
                headers.get("host", "").lower() not in {_API_HOST, _API_HOST + ":443"}
                or target.netloc
                and target.hostname != _API_HOST
            ):
                raise h11.RemoteProtocolError("unexpected model authority")
            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": incoming.method.decode(),
                "scheme": "https",
                "path": target.path,
                "query_string": target.query.encode(),
                "headers": [
                    (k, v)
                    for k, v in incoming.headers
                    if k.lower() not in {b"proxy-authorization", b"x-agency-token"}
                ]
                + [(b"x-agency-token", self.session.host_token.encode())],
                "server": (_API_HOST, 443),
            }
            body_sent, pending = False, bytearray()

            async def receive():
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {"type": "http.request", "body": raw, "more_body": False}
                try:
                    while data := await reader.read(65536):
                        pending.extend(data)
                except OSError:
                    # TLS aborts/RSTs are an ASGI disconnect, just like EOF.
                    pass
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.start":
                    data = connection.send(
                        h11.Response(status_code=message["status"], headers=message["headers"])
                    )
                else:
                    data = connection.send(h11.Data(data=message.get("body", b"")))
                    if not message.get("more_body", False):
                        data += connection.send(h11.EndOfMessage())
                writer.write(data)
                await writer.drain()

            request = Request(scope, receive)
            if target.path in {"/v1/messages", "/v1/messages/count_tokens"}:
                response = await self.forward(request, target.path.lstrip("/"))
            else:
                response = await self._unrecorded(request, target.path, raw)
            try:
                await response(scope, receive, send)
            except* OSError:
                # StreamingResponse task groups can wrap a failed socket send.
                # Its recording finalizer has already retained partial bytes.
                pass
            if connection.our_state is not h11.DONE or connection.their_state is not h11.DONE:
                return
            connection.start_next_cycle()
            if pending:
                connection.receive_data(bytes(pending))

    async def _unrecorded(self, request, path, raw):
        """API auxiliary traffic may include authentication; never archive it."""
        if request.headers.get("upgrade"):
            return Response(status_code=501)
        url = self._upstream(request.headers, path.lstrip("/"), request.url.query)
        client = httpx.AsyncClient(
            **self._client_options(url), follow_redirects=False, timeout=1200
        )
        try:
            upstream = await client.send(
                client.build_request(
                    request.method,
                    url,
                    headers=self._forward_headers(request.headers),
                    content=raw,
                ),
                stream=True,
            )
        except httpx.HTTPError:
            await client.aclose()
            return Response(status_code=502)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await client.aclose()
            raise

        async def chunks():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                with anyio.CancelScope(shield=True):
                    await upstream.aclose()
                    await client.aclose()

        response = StreamingResponse(chunks(), status_code=upstream.status_code)
        response.raw_headers = [
            (k.encode("latin-1"), v.encode("latin-1")) for k, v in _headers(upstream.headers)
        ]
        return response
