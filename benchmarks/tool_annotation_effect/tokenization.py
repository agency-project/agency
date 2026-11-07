"""Optional local-file tokenization, with no download or provider calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def annotation_tokens(events, encode):
    total = 0
    for event in events:
        if event["kind"] == "tool_annotation" and event["annotation"].get("raw") is not None:
            text = json.dumps(event["annotation"]["raw"], ensure_ascii=False)
            total += len(encode(text))
    return total


def local_tokenizer(path):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise RuntimeError(
            "Install optional tokenizers and supply a local tokenizer JSON; no tokenizer is downloaded"
        ) from None
    path = Path(path)
    tokenizer = Tokenizer.from_file(str(path))
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids, {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
