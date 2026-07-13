# Amazon Bedrock backends, and Claude Platform on AWS (`agllm_backends/bedrock.py`)

> Three backend classes live here: `_OpenAICompatibleBedrockBackend`, `_AnthropicBedrockBackend`, and `_AnthropicAWSBackend`. The latter two reuse the Messages-API adapter documented in [anthropic.md](anthropic.md) — see there for how `_AnthropicBedrockChatClient` and its supporting translation code actually work. See [base.md](base.md) for backend selection.

Claude models on Amazon Bedrock are **not** served through the OpenAI-compatible Mantle gateway used by every other Bedrock model — Mantle's `/v1/models` never lists an `anthropic.*` model, and every Claude model ID 404s there. Claude models on Bedrock are only reachable through Bedrock's native `invoke_model` API, in the Anthropic Messages API shape, and only via an inference-profile ID (e.g. `us.anthropic.claude-sonnet-5`) rather than the bare `anthropic.claude-sonnet-5` foundation-model ID — the bare ID 400s with "on-demand throughput isn't supported." This is why `for_config()` routes `provider="bedrock"` to one of *two* different backend classes depending on the model ID, rather than one.

## Which Bedrock backend gets picked

```python
if provider == "bedrock":
    if _is_anthropic_bedrock_model(model):       # matches (region.)?anthropic\.
        return _AnthropicBedrockBackend(agconfig)
    return _OpenAICompatibleBedrockBackend(agconfig)
```

## `_OpenAICompatibleBedrockBackend`

Every non-Anthropic Bedrock model, via the OpenAI-compatible Mantle gateway. Subclasses [openai.md](openai.md)'s `_OpenAICompatibleBackend`, overriding only `make_client()` to resolve the right base URL and auth:

Authentication is resolved in this order:

1. `api_key` starting with `"ABSK"` / any colon-free value — a direct Mantle bearer token, sent straight to `https://bedrock-mantle.{region}.api.aws/v1`.
2. No `api_key`, `aws_bedrock_token_generator` package installed — automatic token generation via that package, same Mantle URL.
3. `api_key` as `"ACCESS_KEY_ID:SECRET_ACCESS_KEY"` or `"...:SESSION_TOKEN"` (colon-delimited) — SigV4 signing (`_BedrockSigV4Auth`) against `https://bedrock-runtime.{region}.amazonaws.com` instead of Mantle.
4. No `api_key` at all — SigV4 using boto3's ambient credential chain (env vars, `~/.aws/credentials`, instance role).

`region` defaults to `us-east-1` if unset. `tokenize_url()` returns `None` — Bedrock has no vLLM-style `/tokenize` endpoint.

### `_BedrockSigV4Auth`

An `httpx.Auth` handler that signs each request with AWS SigV4 (via `botocore.auth.SigV4Auth`), for the credential-pair/ambient-boto3 auth paths above. Parses a colon-delimited `api_key` into `Credentials(access_key, secret_key, token=...)`, or falls back to `boto3.Session(region_name=region).get_credentials()` — raising `RuntimeError` immediately if that also finds nothing, rather than deferring to an opaque signing failure later.

## `_AnthropicBedrockBackend`

Claude models on Bedrock, via the anthropic SDK's `AnthropicBedrock` client (native `invoke_model`, Messages API shape):

```python
def make_client(self, timeout):
    anthropic_client = _anthropic_sdk.AnthropicBedrock(aws_region=self.region or "us-east-1", timeout=timeout)
    return _AnthropicBedrockChatClient(anthropic_client)  # see anthropic.md
```

`list_models()` always returns `[]` — Bedrock's native `invoke_model` API has no OpenAI-style `/v1/models`; `known_context_limit()` falls back to the static lookup table in [anthropic.md](anthropic.md).

## `_AnthropicAWSBackend`

A separate product from Bedrock: **Claude Platform on AWS**, via the anthropic SDK's `AnthropicAWS` client (`provider="anthropicAWS"` or the snake_case alias `"anthropic_aws"` — no dedicated `agXXXBackendConfig` view exists for this provider; set `provider` directly via a plain `agConfig` dict). Auth, resolved by the SDK itself: SigV4 via the default AWS credential chain, explicit `aws_access_key`/`aws_secret_key` config fields, or an API key (config `api_key` / `ANTHROPIC_AWS_API_KEY` env var). Requires `workspace_id` (config field, or `ANTHROPIC_AWS_WORKSPACE_ID`/`ANTHROPIC_WORKSPACE_ID` env vars) and `aws_region` (config `region`/`aws_region`, or `AWS_REGION`) unless `base_url` is set.

Reuses the same [anthropic.md](anthropic.md) adapter machinery as `_AnthropicBedrockBackend` and `_AnthropicBackend` — all three only ever need `.messages.create()`, which `AnthropicAWS`, `AnthropicBedrock`, and `Anthropic` all expose alike.

## Usage

```python
llm = agllm(agConfig({
    "agllm_backend": {
        "provider": "bedrock",
        "region":   "us-east-1",
        "model":    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    }
}))
```

or via the typed config view (see [base.md](base.md)):

```python
from agency.agllm_backends import agBedrockBackendConfig
cfg = agConfig(agBedrockBackendConfig(region="us-east-1", model="..."))
```
