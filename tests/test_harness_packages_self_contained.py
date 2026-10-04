import ast
from pathlib import Path

import pytest

AGENCY = Path(__file__).resolve().parent.parent / "agency"


# These run as top-level packages inside the sandbox (PYTHONPATH=/opt/agency_pkg/agency),
# so an import reaching outside them fails at startup there, though it works on the host.
@pytest.mark.parametrize("package", ["native_harness", "tandem_harness"])
def test_harness_package_imports_nothing_outside_itself(package):
    bad = []
    for path in sorted((AGENCY / package).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (
                node.level > 1
                or (node.level == 0 and (node.module or "").split(".")[0] == "agency")
            ):
                bad.append(f"{path.relative_to(AGENCY)}:{node.lineno}")
            elif isinstance(node, ast.Import) and any(
                a.name.split(".")[0] == "agency" for a in node.names
            ):
                bad.append(f"{path.relative_to(AGENCY)}:{node.lineno}")
    assert not bad, f"imports outside {package}: {bad}"
