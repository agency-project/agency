"""Lets `python3 -m tandem_harness ...` work as a shorthand for
`python3 -m tandem_harness.cli ...` -- see `__init__.py`'s docstring for
the required invocation (`agency/` on `PYTHONPATH`, never
`agency.tandem_harness`)."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
