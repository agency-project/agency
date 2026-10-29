"""Native wire fidelity audits adapted from the prior validation study."""

from collections import Counter
import copy
import hashlib
import json
import re


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def requests(directory):
    values = []
    for path in sorted((directory / "provider").glob("*.request.json")):
        meta = path.with_name(path.name.replace("request.json", "metadata.json"))
        if meta.exists() and read(meta)["path"].split("?")[0].endswith("/count_tokens"):
            continue
        value = read(path)
        if value.get("model"):
            values.append((path, value))
    return values


def preservation(directory):
    incoming = sorted((directory / "logs/native-incoming").glob("*.request.json"))
    outgoing = requests(directory)
    by_hash = {}
    for path, value in outgoing:
        by_hash.setdefault(digest(value), []).append(path)
    failures = []
    incoming_pairs = Counter()
    incoming_hashes = set()
    for path in incoming:
        value_hash = digest(read(path))
        incoming_hashes.add(value_hash)
        matches = by_hash.get(value_hash, [])
        if not matches:
            failures.append(path.name)
        expected = read(path.with_name(path.name.replace("request.json", "headers.json")))
        incoming_pairs[(value_hash, digest(expected))] += 1
    # Identical bodies can be sent with different beta headers during native
    # compatibility recovery. Match complete (body, headers) occurrences, not
    # every body occurrence against every header variant.
    remaining = incoming_pairs.copy()
    header_failures = []
    for path, value in outgoing:
        actual = read(path.with_name(path.name.replace("request.json", "request-headers.json")))
        key = (digest(value), digest(actual))
        if remaining[key]:
            remaining[key] -= 1
        elif key[0] in incoming_hashes:
            header_failures.append(path.name)
    unmatched_outgoing = [p.name for p, v in outgoing if digest(v) not in incoming_hashes]
    incoming_counts = Counter(digest(read(path)) for path in incoming)
    outgoing_counts = Counter(digest(value) for _, value in outgoing)
    count_mismatches = {
        key: {"incoming": incoming_counts[key], "outgoing": outgoing_counts[key]}
        for key in incoming_counts.keys() | outgoing_counts.keys()
        if incoming_counts[key] != outgoing_counts[key]
    }
    return dict(
        incoming_requests=len(incoming),
        outgoing_requests=len(outgoing),
        json_mismatches=failures,
        header_mismatches=header_failures,
        unmatched_outgoing=unmatched_outgoing,
        count_mismatches=count_mismatches,
        all_preserved=bool(incoming)
        and not (failures or header_failures or unmatched_outgoing or count_mismatches),
    )


def first_task_request(directory):
    return next((v for _, v in requests(directory) if v.get("tools")), None)


def response_preservation(directory):
    """Audit successful task response control envelopes after SDK/host relay."""

    def control(path):
        events = []
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                except ValueError:
                    continue
                if event.get("type") in ("message_start", "message_delta"):
                    events.append(event)
        return events

    emitted = {}
    for path in (directory / "logs/native-outgoing").glob("*.response.sse"):
        events = control(path)
        first = next((e for e in events if e["type"] == "message_start"), None)
        if first:
            emitted.setdefault(first["message"]["id"], []).append((path, events))
    checked = 0
    mismatches = []
    missing = []
    for path, request in requests(directory):
        response = path.with_name(path.name.replace("request.json", "response.body"))
        if not response.exists():
            continue
        events = control(response)
        first = next((e for e in events if e["type"] == "message_start"), None)
        if not first:
            continue  # HTTP errors are covered by retry regression tests.
        checked += 1
        matches = emitted.get(first["message"]["id"], [])
        if not matches:
            missing.append(path.name)
        elif not any(events == actual for _, actual in matches):
            mismatches.append(path.name)
    return dict(
        successful_native_responses=checked,
        missing_native_response=missing,
        control_envelope_mismatches=mismatches,
        all_preserved=bool(checked) and not (missing or mismatches),
        scope="message_start and message_delta envelopes, including refusal/context metadata; content fragmentation and HTTP error bodies excluded",
    )


def normalized_initial_request(request):
    """Ignore only native fresh-session IDs and the isolated session cwd/slug."""
    value = copy.deepcopy(request)
    if not value:
        return value
    encoded = json.dumps(value)
    match = re.search(r"Primary working directory: ([^\\]+)\\n", encoded)
    cwd = match.group(1) if match else None

    def normalize(item):
        if isinstance(item, str) and cwd:
            return item.replace(cwd, "<SESSION_CWD>").replace(
                re.sub(r"[^a-zA-Z0-9]", "-", cwd), "<SESSION_SLUG>"
            )
        if isinstance(item, list):
            return [normalize(x) for x in item]
        if isinstance(item, dict):
            return {k: normalize(v) for k, v in item.items()}
        return item

    value = normalize(value)
    metadata = value.get("metadata", {})
    if isinstance(metadata.get("user_id"), str):
        native = json.loads(metadata["user_id"])
        for key in ("device_id", "session_id"):
            if key in native:
                native[key] = "<FRESH_SESSION_ID>"
        metadata["user_id"] = json.dumps(native, sort_keys=True)
    return value
