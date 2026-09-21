# Data needed for the integration test

Frozen recordings of `django__django-11066` from
[bug_localization](https://github.com/agency-project/agency-benchmarks/tree/main/targeted/bug_localization),
one per harness. Used by `tests/integration/test_replay_real_recording.py`.

## When to refresh

- Agency's own transcript format changes, or
- you want a different recorded conversation.

## Steps

- Set where your `agency` checkout is (used in step 2):
  ```bash
  AGENCY_DIR=~/agency
  ```

- **1. Record a run for each harness**, in your `agency-benchmarks` checkout:
  ```bash
  cd targeted/bug_localization
  for h in native claude_code codex; do
    python benchmark.py --harness "$h" --instance django__django-11066   # add --base-url/--model if needed
  done
  ```

- **2. Freeze each run's db and copy `result.json`** — run this from the same directory:
  ```bash
  python3 - "$AGENCY_DIR" <<'EOF'
  import sqlite3, shutil, sys
  from pathlib import Path

  DEST = Path(sys.argv[1]) / "tests" / "fixtures" / "bug_localization"

  for harness in ("native", "claude_code", "codex"):
      src_dir = next(Path("output").glob(f"{harness}/*/django__django-11066"))
      src_db = next((src_dir / "agent_logs").glob("agent_*_data.sqlite3"))
      dest_dir = DEST / harness
      dest_dir.mkdir(parents=True, exist_ok=True)
      dest_db = dest_dir / "recording.sqlite3"
      dest_db.unlink(missing_ok=True)

      conn = sqlite3.connect(str(src_db))
      conn.execute(f"VACUUM INTO {str(dest_db)!r}")  # not a plain copy -- see note below
      conn.close()

      shutil.copy(src_dir / "result.json", dest_dir / "result.json")
      print(harness, "->", dest_db)
  EOF
  ```

- **3. Confirm it still passes:**
  ```bash
  cd "$AGENCY_DIR" && pytest tests/integration/test_replay_real_recording.py -v
  ```

## Why `VACUUM INTO`

A plain file copy can miss most of the conversation. For example, fresh writes sit in an uncheckpointed
`-wal` file next to the main `.sqlite3` file, not in the file itself. `VACUUM INTO` opens
the db normally first, which merges that in, then writes one clean, complete file.

## Note

The test's expected answer comes straight from the `result.json` you drop in each
harness's folder.
