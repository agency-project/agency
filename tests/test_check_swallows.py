import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_swallows", ROOT / "scripts" / "check_swallows.py"
)
check_swallows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_swallows)


def test_agency_package_has_no_silent_swallows():
    assert check_swallows.main([str(ROOT / "agency")]) == 0


@pytest.mark.parametrize(
    "body, flagged",
    [
        ("except Exception:\n    return None", True),
        ("except Exception:\n    pass", True),
        ("except:\n    x = 1", True),
        ("except (ValueError, Exception):\n    return None", True),
        ("except Exception:  # swallow-ok:\n    return None", True),
        ("except Exception:  # swallow-ok: probe only\n    return None", False),
        ("except Exception as exc:\n    return str(exc)", False),
        ("except Exception:\n    raise", False),
        ("except Exception:\n    print('failed', file=sys.stderr)", False),
        ("except Exception:\n    logger.warning('failed')", False),
        ("except Exception:\n    traceback.print_exc()", False),
        ("except ValueError:\n    return None", False),
    ],
)
def test_swallow_detection(tmp_path, body, flagged):
    src = tmp_path / "sample.py"
    src.write_text(
        "def f():\n    try:\n        g()\n"
        + "\n".join("    " + line for line in body.split("\n"))
        + "\n"
    )
    assert bool(check_swallows.find_swallows(src)) == flagged
