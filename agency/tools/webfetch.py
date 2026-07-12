import httpx
import html2text
from ..agdata import agdata, agerror
from ..agutil import format_exception
from ..agtool import agtool

_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120


def _run(arg: agdata) -> agdata:
    url: str = str(arg.url)  # type: ignore[arg-type]
    fmt: str = str(getattr(arg, "format", "markdown") or "markdown")
    timeout: int = min(
        int(getattr(arg, "timeout", _DEFAULT_TIMEOUT) or _DEFAULT_TIMEOUT), _MAX_TIMEOUT
    )

    if not url.startswith(("http://", "https://")):
        return agerror("URL must start with http:// or https://")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; agency-bot/1.0)",
        "Accept": "text/html,text/plain,*/*;q=0.8",
    }

    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return agerror(f"HTTP {e.response.status_code}: {url}\n{format_exception(e)}")
    except Exception as e:
        return agerror(format_exception(e))

    if len(resp.content) > _MAX_BYTES:
        return agerror("Response too large (>5 MB)")

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

    return agdata(url=url, content_type=content_type, output=output)


webfetch = agtool(
    name="webfetch",
    fn=_run,
    run_in_subprocess=False,
    description="Fetch a URL and return its content as text, markdown, or raw HTML.",
    params={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch (must be http:// or https://)"},
            "format": {
                "type": "string",
                "enum": ["markdown", "text", "html"],
                "description": "Output format (default: markdown)",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (max 120, default 30)",
            },
        },
        "required": ["url"],
    },
)
