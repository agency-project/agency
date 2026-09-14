"""Build Agency's embedded Perfetto assets (requires network on first build).

Run: python -m agency.observability.agwebui.build_perfetto
The pinned source/dependencies remain under ignored runs/, and the built
assets under ignored static/perfetto/. No separate viewer server is started.
"""

from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
REVISION = "6f78923bd6e6f9bfd9078155226f76df8e0a007c"
SOURCE = ROOT / "runs" / "perfetto-viewer"
DEST = Path(__file__).parent / "static/perfetto"
PRESENTATION = Path(__file__).with_name("perfetto_presentation.ts")
ORIGINAL = """    const flowStartTs =
      flow.flowToDescendant || flow.begin.sliceStartTs >= flow.end.sliceStartTs
        ? flow.begin.sliceStartTs
        : flow.begin.sliceEndTs;"""
PATCHED = """    // Agency: parent-child relationships connect the starts of both spans.
    const flowStartTs = flow.begin.sliceStartTs;"""


def _fingerprint() -> str:
    # Include the builder itself so changes to packaging or patches invalidate
    # the cache, as well as the explicitly pinned upstream revision.
    return hashlib.sha256(
        REVISION.encode() + Path(__file__).read_bytes() + PRESENTATION.read_bytes()
    ).hexdigest()


def viewer_is_current() -> bool:
    try:
        manifest = json.loads((DEST / "agency-build.json").read_text())
        files = manifest["files"]
        return (
            manifest["fingerprint"] == _fingerprint()
            and "index.html" in files
            and all((DEST / name).is_file() for name in files)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _build_viewer(*, skip_deps: bool) -> None:
    source = SOURCE
    if (source / ".git").exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True
        )
        if revision.returncode == 0 and revision.stdout.strip() != REVISION:
            # Keep previous cached revisions intact when upgrading Perfetto.
            source = SOURCE.with_name(f"perfetto-viewer-{REVISION}")
    source.mkdir(parents=True, exist_ok=True)

    def run(*args):
        subprocess.run(args, cwd=source, check=True)

    if not (source / ".git").exists():
        run("git", "init")
        run("git", "remote", "add", "origin", "https://github.com/google/perfetto.git")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True
    )
    if revision.returncode != 0 or revision.stdout.strip() != REVISION:
        run("git", "fetch", "--depth", "1", "origin", REVISION)
        run("git", "checkout", "--detach", "FETCH_HEAD")
    renderer = source / "ui/src/core_plugins/dev.perfetto.FlowEvents/flow_events_renderer.ts"
    text = renderer.read_text()
    if ORIGINAL in text:
        renderer.write_text(text.replace(ORIGINAL, PATCHED, 1))
    elif PATCHED not in text:
        raise RuntimeError("Unrecognized Perfetto renderer; cannot apply start-to-start arrows.")
    plugin = source / "ui/src/plugins/dev.agency.Presentation"
    plugin.mkdir(exist_ok=True)
    shutil.copy2(PRESENTATION, plugin / "index.ts")
    defaults = source / "ui/src/core/embedder/default_plugins.ts"
    text = defaults.read_text()
    if "'dev.agency.Presentation'" not in text:
        anchor = "  'dev.perfetto.TraceProcessorTrack',"
        if anchor not in text:
            raise RuntimeError("Unrecognized Perfetto default plugin list")
        defaults.write_text(text.replace(anchor, anchor + "\n  'dev.agency.Presentation',"))
    if not skip_deps:
        run("tools/install-build-deps", "--ui")
    run("ui/build")
    # An interrupted copy must not leave a build marked current.
    (DEST / "agency-build.json").unlink(missing_ok=True)
    shutil.copytree(source / "ui/out/dist", DEST, dirs_exist_ok=True)
    # Normalize Vite's duplicated font prefixes for serving under /perfetto/.
    for css in DEST.glob("*/frontend.css"):
        css.write_text(re.sub(r"url\((?:\.\./|/)*assets/assets/", "url(assets/", css.read_text()))
    shutil.copy2(source / "LICENSE", DEST / "LICENSE")
    (DEST / "agency-build.txt").unlink(missing_ok=True)
    files = sorted(str(path.relative_to(DEST)) for path in DEST.rglob("*") if path.is_file())
    (DEST / "agency-build.json").write_text(
        json.dumps(
            {
                "revision": REVISION,
                "fingerprint": _fingerprint(),
                "files": files,
            }
        )
    )


def ensure_viewer(*, force: bool = False, skip_deps: bool = False) -> None:
    """Build missing/stale assets once, before the web UI starts listening."""
    if not force and viewer_is_current():
        return
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    print("[agwebui] Preparing Perfetto viewer (first build may take several minutes)…", flush=True)
    with (SOURCE.parent / ".perfetto-build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Another web UI process may have completed the same build while we waited.
        if not force and viewer_is_current():
            return
        try:
            _build_viewer(skip_deps=skip_deps)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "Perfetto build failed; the web UI was not started. See the build output above. "
                "Retry with: python -m agency.observability.agwebui.build_perfetto"
            ) from error
    print(f"[agwebui] Embedded Perfetto ready: {DEST}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-deps", action="store_true", help="Use installed build dependencies")
    parser.add_argument("--force", action="store_true", help="Rebuild even if assets are current")
    args = parser.parse_args()
    ensure_viewer(force=args.force, skip_deps=args.skip_deps)


if __name__ == "__main__":
    main()
