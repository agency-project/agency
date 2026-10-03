"""Static documentation inventory and MkDocs hooks; never import Agency."""

from __future__ import annotations

import ast
from html import escape
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO = "https://github.com/agency-project/agency"


def inventory():
    return json.loads((DOCS / "api/public-api.json").read_text())


def config_tables():
    """Extract dataclass fields/default factories without executing source."""
    source = ROOT / "agency/configs/agconfig.py"
    tree = ast.parse(source.read_text())
    result = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name in {"agconfig", "confignamespace"}:
            continue
        result.extend([f"### {node.name}", "", "| Field | Type | Default |", "| --- | --- | --- |"])
        for field in node.body:
            if (
                not isinstance(field, ast.AnnAssign)
                or not isinstance(field.target, ast.Name)
                or field.target.id.startswith("_")
            ):
                continue
            annotation = ast.unparse(field.annotation).strip("'\"")
            default = ast.unparse(field.value) if field.value else "(required)"
            if (
                isinstance(field.value, ast.Call)
                and isinstance(field.value.func, ast.Name)
                and field.value.func.id == "field"
            ):
                factory = next(k.value for k in field.value.keywords if k.arg == "default_factory")
                default = f"fresh {ast.unparse(factory)}()"
            result.append(
                f"| `{field.target.id}` | `{annotation.replace('|', '&#124;')}` | `{default.replace('|', '&#124;')}` |"
            )
        result.append("")
    return "\n".join(result)


def inventory_table():
    manifest = inventory()
    pages = {e["object"]: e["page"] for e in manifest["objects"]}
    rows = ["| Root import | Reference |", "| --- | --- |"]
    for name, obj in manifest["root_exports"].items():
        rows.append(f"| `from agency import {name}` | [{obj}]({pages[obj]}) |")
    rows += ["", "## Additional imports", "", "| Import path | Reference |", "| --- | --- |"]
    roots = set(manifest["root_exports"].values())
    for entry in manifest["objects"]:
        if entry["object"] not in roots:
            rows.append(f"| `{entry['object']}` | [{entry['page'][:-3]}]({entry['page']}) |")
    rows += [
        "",
        "## Subpackage exports and aliases",
        "",
        "Public-facing subpackages below are inventoried separately. Backend/engine extension protocols have explicit exclusions.",
        "",
        "| Import path | Destination or rationale |",
        "| --- | --- |",
    ]
    for module, entries in manifest["subpackage_exports"].items():
        for name, entry in entries.items():
            destination = (
                f"[{entry['page'][:-3]}]({entry['page']})"
                if entry.get("page")
                else entry["rationale"]
            )
            rows.append(f"| `{module}.{name}` | {destination} |")
    rows += [
        "",
        "## Exclusions",
        "",
        "These diagnostic integrations are used by live examples but are not application APIs.",
        "",
        "| Example import | Rationale |",
        "| --- | --- |",
    ]
    for name, reason in manifest["excluded_example_imports"].items():
        rows.append(f"| `{name}` | {reason} |")
    rows += [
        "",
        "The following public-looking members belong to implementation protocols. Their exclusions are explicit so changes still receive review.",
        "",
        "| Object/member | Rationale |",
        "| --- | --- |",
    ]
    for entry in manifest["objects"]:
        for name, reason in entry.get("excluded_members", {}).items():
            rows.append(f"| `{entry['object']}.{name}` | {reason} |")
    return "\n".join(rows)


def on_page_markdown(markdown, page, config, files):
    markdown = markdown.replace("<!-- CONFIG_TABLES -->", config_tables())
    markdown = markdown.replace("<!-- API_INVENTORY -->", inventory_table())

    # GitHub-relative repository/archival links remain useful in source Markdown;
    # the site sends them to the matching repository file instead of broken URLs.
    def link(match):
        raw = match[1]
        parsed = urlsplit(raw)
        if parsed.scheme or raw.startswith(("#", "/")):
            return match[0]
        target = (Path(page.file.abs_src_path).parent / parsed.path).resolve()
        try:
            repo_path = target.relative_to(ROOT).as_posix()
        except ValueError:
            return match[0]
        file = (
            files.get_file_from_path(target.relative_to(DOCS).as_posix())
            if target.is_relative_to(DOCS)
            else None
        )
        in_site = file is not None and file.inclusion.is_included()
        if in_site:
            return match[0]
        if not target.exists():
            raise ValueError(f"{page.file.src_uri}: missing local target {raw}")
        url = f"{REPO}/blob/master/{repo_path}"
        if parsed.fragment:
            url += "#" + parsed.fragment
        return "](" + url + ")"

    return re.sub(r"\]\(([^)\s]+)\)", link, markdown)


def on_post_build(config):
    if any(name == "agency" or name.startswith("agency.") for name in sys.modules):
        raise RuntimeError("Documentation build imported Agency; static extraction is required")

    site = Path(config["site_dir"]).resolve()
    redirects = json.loads((ROOT / "tools/docs/redirects.json").read_text())
    for previous, destination in redirects.items():
        output = (site / previous).resolve()
        if not output.is_relative_to(site) or output.exists():
            raise ValueError(f"Redirect would overwrite a page or leave the site: {previous}")
        output.parent.mkdir(parents=True, exist_ok=True)
        url = escape(destination, quote=True)
        output.write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<title>Page moved</title><meta name="robots" content="noindex">'
            f'<meta http-equiv="refresh" content="0; url={url}"></head>'
            f'<body><p>This page moved. <a href="{url}">Continue to the documentation.</a></p>'
            "</body></html>\n"
        )
