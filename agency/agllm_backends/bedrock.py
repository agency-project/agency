"""Amazon Bedrock backend(s), plus the separate "Claude Platform on AWS"
backend (provider='anthropicAWS'/'anthropic_aws') -- grouped here as the two
AWS-hosted flavors of Anthropic access, distinct from Bedrock's actual other
models.

Claude models on Amazon Bedrock are NOT served through the OpenAI-compatible
Mantle gateway used by every other Bedrock model — Mantle's `/v1/models`
never lists an `anthropic.*` model, and every Claude model ID 404s there.
Claude models on Bedrock are only reachable through Bedrock's native
invoke_model API, in the Anthropic Messages API shape (the `anthropic` SDK's
`AnthropicBedrock` client), and only via an inference-profile ID (e.g.
`us.anthropic.claude-sonnet-5`) rather than the bare
`anthropic.claude-sonnet-5` foundation-model ID — the bare ID 400s with "on-
demand throughput isn't supported."

`for_config()` (`.base`) picks between `_AnthropicBedrockBackend` and
`_OpenAICompatibleBedrockBackend` based on the model ID
(`_is_anthropic_bedrock_model()`), and routes provider='anthropicAWS' to
`_AnthropicAWSBackend` directly (no model-based branching -- there's no
non-Anthropic equivalent on that product). All three Anthropic-family
backends here reuse `.anthropic._AnthropicBedrockChatClient`, which adapts
the Messages API to the OpenAI chat.completions interface (streaming and
non-streaming) the rest of agllm.py is built around, so its streaming /
tool-call / retry / compaction logic needs no changes to support any of them.
"""

from __future__ import annotations
import os
import httpx
import openai

from .base import (
    _AgProviderBackendConfig,
    _OPENAI_GEN_FIELDS,
    _VLLM_EXTRA_GEN_FIELDS,
    agllm_backend,
)
from .openai import _OpenAICompatibleBackend
from .anthropic import (
    _AnthropicBedrockChatClient,
    _ANTHROPIC_BEDROCK_MODEL_RE,
    _known_anthropic_context_window,
)

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None


class agBedrockBackendConfig(_AgProviderBackendConfig):
    """agLLMBackendConfig restricted to the fields Amazon Bedrock backends
    read. `for_config()` picks between two backends under the hood based on
    `model`: non-Anthropic models go through `_OpenAICompatibleBedrockBackend`
    (the OpenAI-compatible Mantle gateway -- full generation surface, same as
    `agVLLMBackendConfig`); Anthropic models on Bedrock go through
    `_AnthropicBedrockBackend`, which -- like `agAnthropicBackendConfig` --
    only applies temperature, top_p, max_tokens, and extra_body["top_k"],
    silently ignoring the rest. `provider` is fixed to "bedrock"."""

    _PROVIDER = "bedrock"
    _ALLOWED_FIELDS = (
        frozenset({"model", "api_key", "region", "context_limit"})
        | _OPENAI_GEN_FIELDS
        | _VLLM_EXTRA_GEN_FIELDS
    )


def _is_anthropic_bedrock_model(model: str) -> bool:
    return bool(_ANTHROPIC_BEDROCK_MODEL_RE.match(model or ""))


# ---------------------------------------------------------------------------
# AWS Bedrock SigV4 auth
# ---------------------------------------------------------------------------


class _BedrockSigV4Auth(httpx.Auth):
    """httpx auth handler that signs requests with AWS SigV4 for Amazon Bedrock."""

    def __init__(self, region: str, api_key: str | None = None) -> None:
        import boto3
        from botocore.credentials import Credentials

        self._region = region
        if api_key:
            parts = api_key.split(":", 2)
            if len(parts) < 2:
                raise ValueError(
                    "Bedrock api_key must be 'ACCESS_KEY_ID:SECRET_ACCESS_KEY' "
                    "or 'ACCESS_KEY_ID:SECRET_ACCESS_KEY:SESSION_TOKEN'."
                )
            self._creds = Credentials(
                access_key=parts[0],
                secret_key=parts[1],
                token=parts[2] if len(parts) == 3 else None,
            )
        else:
            creds = boto3.Session(region_name=region).get_credentials()
            if creds is None:
                raise RuntimeError(
                    "No AWS credentials found for Amazon Bedrock. "
                    "Set api_key='ACCESS_KEY_ID:SECRET_ACCESS_KEY' in the llm_config, "
                    "or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars, "
                    "or run: aws configure"
                )
            self._creds = creds

    def auth_flow(self, request: httpx.Request):
        import botocore.auth
        import botocore.awsrequest

        aws_req = botocore.awsrequest.AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content or b"",
            headers={
                k: v
                for k, v in request.headers.items()
                if k.lower() not in ("host", "content-length")
            },
        )
        botocore.auth.SigV4Auth(
            self._creds.get_frozen_credentials(), "bedrock", self._region
        ).add_auth(aws_req)
        for k, v in aws_req.headers.items():
            request.headers[k] = v
        yield request


class _OpenAICompatibleBedrockBackend(_OpenAICompatibleBackend):
    """Bedrock models reachable through the OpenAI-compatible Mantle gateway —
    every Bedrock model except Anthropic's own (see module docstring above)."""

    def make_client(self, timeout: httpx.Timeout) -> openai.OpenAI:
        region = self.region or "us-east-1"
        api_key = self.api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or None
        mantle_url = f"https://bedrock-mantle.{region}.api.aws/v1"
        runtime_url = f"https://bedrock-runtime.{region}.amazonaws.com"
        # A Bedrock API key (e.g. "ABSK...") is a single opaque bearer token
        # for the Mantle gateway. AWS access/secret key pairs for SigV4 signing
        # are always "ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]" — the
        # colon is what distinguishes the two.
        if api_key and ":" not in api_key:
            return openai.OpenAI(api_key=api_key, base_url=mantle_url, timeout=timeout)
        if not api_key:
            try:
                from aws_bedrock_token_generator import provide_token as _provide_token

                os.environ.setdefault("AWS_DEFAULT_REGION", region)
                token = _provide_token(region=region)
                return openai.OpenAI(api_key=token, base_url=mantle_url, timeout=timeout)
            except ImportError:
                pass
        return openai.OpenAI(
            api_key="bedrock",
            base_url=runtime_url,
            http_client=httpx.Client(
                auth=_BedrockSigV4Auth(region, api_key=api_key), timeout=timeout
            ),
        )

    def tokenize_url(self) -> "str | None":
        return None  # Bedrock has no vLLM-style /tokenize endpoint


class _AnthropicBedrockBackend(agllm_backend):
    """Claude models on Amazon Bedrock — native invoke_model API via the
    anthropic SDK's AnthropicBedrock client (Messages API shape)."""

    def make_client(self, timeout: httpx.Timeout) -> _AnthropicBedrockChatClient:
        if _anthropic_sdk is None:
            raise RuntimeError(
                "Anthropic models on Bedrock require the 'anthropic' package: pip install anthropic"
            )
        region = self.region or "us-east-1"
        anthropic_client = _anthropic_sdk.AnthropicBedrock(aws_region=region, timeout=timeout)
        return _AnthropicBedrockChatClient(anthropic_client)

    def list_models(self) -> list:
        return []  # Bedrock's native invoke_model API has no OpenAI-style /v1/models

    def tokenize_url(self) -> "str | None":
        return None

    def known_context_limit(self, model: str) -> "int | None":
        return _known_anthropic_context_window(model)


class _AnthropicAWSBackend(agllm_backend):
    """Claude Platform on AWS via the anthropic SDK's AnthropicAWS client.

    Auth (resolved by the SDK): SigV4 via the default AWS credential chain,
    explicit aws_access_key/aws_secret_key, or an API key (config `api_key` /
    ANTHROPIC_AWS_API_KEY). Requires workspace_id (config /
    ANTHROPIC_AWS_WORKSPACE_ID) and aws_region (config `region` or
    `aws_region` / AWS_REGION) unless base_url is set.
    """

    def _client_kwargs(self, timeout: httpx.Timeout) -> dict:
        kwargs: dict = dict(timeout=timeout)
        api_key = self.api_key or os.environ.get("ANTHROPIC_AWS_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        for key in ("aws_access_key", "aws_secret_key", "aws_session_token", "aws_profile"):
            value = getattr(self, key)
            if value:
                kwargs[key] = value
        region = self.aws_region or self.region
        if region:
            kwargs["aws_region"] = region
        workspace_id = (
            self.workspace_id
            or os.environ.get("ANTHROPIC_AWS_WORKSPACE_ID")
            or os.environ.get("ANTHROPIC_WORKSPACE_ID")
        )
        if workspace_id:
            kwargs["workspace_id"] = workspace_id
        base_url = (
            self.base_url
            or os.environ.get("ANTHROPIC_AWS_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
        )
        if base_url:
            kwargs["base_url"] = base_url
        return kwargs

    def make_client(self, timeout: httpx.Timeout) -> _AnthropicBedrockChatClient:
        if _anthropic_sdk is None:
            raise RuntimeError(
                "provider='anthropicAWS' requires the 'anthropic' package: pip install anthropic"
            )
        if not hasattr(_anthropic_sdk, "AnthropicAWS"):
            raise RuntimeError(
                "provider='anthropicAWS' requires a recent 'anthropic' package with AnthropicAWS support"
            )
        anthropic_client = _anthropic_sdk.AnthropicAWS(**self._client_kwargs(timeout))
        return _AnthropicBedrockChatClient(anthropic_client)

    def list_models(self) -> list:
        if _anthropic_sdk is None or not hasattr(_anthropic_sdk, "AnthropicAWS"):
            return []
        client = _anthropic_sdk.AnthropicAWS(
            **self._client_kwargs(httpx.Timeout(self.model_listing_timeout_seconds))
        )
        return list(client.models.list())

    def tokenize_url(self) -> "str | None":
        return None

    def known_context_limit(self, model: str) -> "int | None":
        return _known_anthropic_context_window(model)
