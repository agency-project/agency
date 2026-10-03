"""Check source-backed public API coverage, examples and local/rendered links."""

from __future__ import annotations

import argparse
import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import textwrap
from urllib.parse import unquote, urlsplit

from griffe import GriffeLoader
import yaml
from docs_support import DOCS, ROOT, inventory


def check_inventory():
    manifest = inventory()
    tree = ast.parse((ROOT / "agency/__init__.py").read_text())
    exports = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
    )
    assert set(exports) == set(manifest["root_exports"]), (
        "Root exports changed: update reference inventory"
    )
    assert len(exports) == len(set(exports)), "Duplicate root exports"
    loader = GriffeLoader(search_paths=[ROOT], allow_inspection=False)
    loader.load("agency")
    loader.resolve_aliases(implicit=True, external=False)
    destinations = {}
    for entry in manifest["objects"]:
        name = entry["object"]
        obj = loader.modules_collection[name]
        page = DOCS / "api" / entry["page"]
        assert page.is_file(), f"Missing API destination: {name}"
        assert f"::: {name}" in page.read_text(), f"Missing generated reference: {name}"
        directive = re.search(
            rf"^::: {re.escape(name)}\n((?:[ \t]+[^\n]*\n|\n)*)", page.read_text(), re.M
        )
        options = yaml.safe_load(textwrap.dedent(directive[1])) or {}
        assert options.get("options", {}).get("members", []) == entry.get("members", []), (
            f"Rendered member selection differs from inventory: {name}"
        )
        destinations[name] = entry["page"]
        for member in entry.get("members", []):
            assert member in obj.members, f"Missing source member: {name}.{member}"
        selected = set(entry.get("members", []))
        excluded = entry.get("excluded_members", {})
        declared_public = {
            key
            for key, value in obj.members.items()
            if not key.startswith("_")
            and not value.is_alias
            and (value.is_function or "property" in value.labels)
        }
        assert declared_public <= selected | set(excluded), (
            f"Unreviewed public methods: {name}: {declared_public - selected - set(excluded)}"
        )
        assert not selected.intersection(excluded), f"Selected and excluded: {name}"
        assert set(excluded) <= obj.members.keys(), f"Stale member exclusions: {name}"
    assert set(manifest["root_exports"].values()) <= destinations.keys()
    config_source = ast.parse((ROOT / "agency/configs/agconfig.py").read_text())
    namespaces = {
        "agency.configs.agconfig." + node.name
        for node in config_source.body
        if isinstance(node, ast.ClassDef)
    }
    assert namespaces <= destinations.keys(), "Uninventoried configuration namespace"
    for module, entries in manifest["subpackage_exports"].items():
        source = ast.parse((ROOT / module.replace(".", "/") / "__init__.py").read_text())
        exported = next(
            ast.literal_eval(node.value)
            for node in source.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            )
        )
        assert set(exported) == set(entries), f"Subpackage exports changed: {module}"
        imports = {}
        for node in source.body:
            if isinstance(node, ast.ImportFrom):
                base = module + "." + node.module if node.module else module
                for alias in node.names:
                    imports[alias.asname or alias.name] = base + "." + alias.name
        for name, entry in entries.items():
            obj = loader.modules_collection[imports[name]]
            actual = obj.final_target.path if obj.is_alias else obj.path
            assert actual == entry["object"], f"Subpackage import target changed: {module}.{name}"
            assert entry.get("page") or entry.get("rationale"), (
                f"Missing destination: {module}.{name}"
            )
            if entry.get("page"):
                assert (DOCS / "api" / entry["page"]).is_file()
    # Validate aliases/imports from source, with the root Agent assignment handled explicitly.
    assert any(
        isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Name)
        and n.value.id == "agent"
        and any(isinstance(t, ast.Name) and t.id == "Agent" for t in n.targets)
        for n in tree.body
    ), "Agent alias changed"
    imported = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = "agency." + node.module if node.level == 1 else node.module
            for alias in node.names:
                imported[alias.asname or alias.name] = module + "." + alias.name
    for name, canonical in manifest["root_exports"].items():
        if name == "Agent":
            continue
        obj = loader.modules_collection[imported[name]]
        actual = obj.final_target.path if obj.is_alias else obj.path
        assert actual == canonical, f"Root import target changed: {name}: {actual} != {canonical}"
    known = set(destinations) | {"agency." + name for name in exports}
    known |= {"agency.orchestrator.get_orchestrator", "agency.orchestrator.peek_orchestrator"}
    excluded = manifest["excluded_example_imports"]
    seen_excluded = set()
    for path in (ROOT / "examples").glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("agency")
            ):
                for alias in node.names:
                    name = node.module + "." + alias.name
                    assert name in known or excluded.get(name), (
                        f"Uninventoried example import: {path.name}: {name}"
                    )
                    if name in excluded:
                        loader.modules_collection[name]  # It must actually exist, even if excluded.
                        seen_excluded.add(name)
    assert seen_excluded == set(excluded), "Stale example import exclusions"
    return len(exports), len(destinations)


def check_markdown():
    count = 0
    paths = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "examples/README.md",
        *(
            path
            for path in DOCS.rglob("*.md")
            if "node_modules" not in path.relative_to(DOCS).parts
        ),
    ]
    for path in paths:
        source = path.read_text()
        for match in re.finditer(r"^```(?:python|py)\s*\n(.*?)^```", source, re.M | re.S):
            ast.parse(match[1], filename=str(path))
            count += 1
        # Ignore source code fences when checking Markdown link targets.
        prose = re.sub(r"^```.*?^```", "", source, flags=re.M | re.S)
        for match in re.finditer(r"\]\(([^)\s]+)\)", prose):
            raw = match[1]
            parsed = urlsplit(raw)
            if parsed.scheme or raw.startswith(("#", "/")):
                continue
            assert (path.parent / unquote(parsed.path)).exists(), f"Missing link in {path}: {raw}"
    return count


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag in {"a", "link", "script", "img"}:
            target = attrs.get("href") if tag in {"a", "link"} else attrs.get("src")
            if target:
                self.links.append(target)


def check_site(site):
    site = site.resolve()
    pages = {}
    for path in site.rglob("*.html"):
        parser = Links()
        parser.feed(path.read_text())
        pages[path] = parser
    count = 0
    for path, parser in pages.items():
        # MkDocs' generic 404 page uses absolute deployment-relative URLs.
        if path.name == "404.html":
            continue
        for raw in parser.links:
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or raw.startswith("/"):
                continue
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if target.is_dir():
                target /= "index.html"
            assert target.is_file(), f"Missing rendered target {path.relative_to(site)}: {raw}"
            if parsed.fragment and target in pages:
                assert unquote(parsed.fragment) in pages[target].ids, (
                    f"Missing rendered anchor {path.relative_to(site)}: {raw}"
                )
            count += 1
    assert not (site / "archive").exists(), "Archives leaked into primary site"
    for entry in inventory()["objects"]:
        page = pages[site / "api" / entry["page"][:-3] / "index.html"]
        for name in [
            entry["object"],
            *[entry["object"] + "." + m for m in entry.get("members", [])],
        ]:
            assert name in page.ids, f"API reference did not render: {name}"
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path)
    args = parser.parse_args()
    roots, objects = check_inventory()
    snippets = check_markdown()
    print(
        f"Covered {roots} root exports and {objects} source objects; parsed {snippets} Python snippets; local targets exist."
    )
    if args.site:
        print(
            f"Checked {check_site(args.site)} rendered local links and anchors; archive excluded."
        )
    assert not any(name == "agency" or name.startswith("agency.") for name in sys.modules), (
        "Static check imported Agency"
    )


if __name__ == "__main__":
    main()
