"""`api/` is the only package a client may import (CLAUDE.md §4, #35).

Checked on the source rather than at runtime, because a forbidden import inside a function body or
under `TYPE_CHECKING` is still a client depending on internals, and the runtime only notices the
ones that happen to execute. `app.py` is included even though CI cannot import it (it needs
pygame): reading it is not importing it, and it is the module most likely to reach for a shortcut.
"""

import ast
from pathlib import Path

import pytest

from tools.sources import iter_source_files

_CLIENTS = Path(__file__).resolve().parents[2] / "clients"

# Everything behind the boundary. `worlds` is on the list because a world definition is core
# config. A client asks for a world by name and never builds one.
_FORBIDDEN = ("core", "persistence", "metrics", "worlds")


def imported_modules(source: str) -> list[str]:
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules


def forbidden(module: str) -> bool:
    return module.split(".")[0] in _FORBIDDEN


# The project's own sources only: not the game client's npm install, which may hold Python (#306).
_SOURCES = list(iter_source_files(_CLIENTS))


def test_there_are_client_modules_to_check():
    """Otherwise a moved directory would make the rule below pass by checking nothing."""
    assert any(path.name == "app.py" for path in _SOURCES)


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: str(p.relative_to(_CLIENTS)))
def test_a_client_module_imports_nothing_behind_the_boundary(path):
    reached = [m for m in imported_modules(path.read_text(encoding="utf-8")) if forbidden(m)]

    assert reached == [], f"{path.name} imports {reached}; go through `api` instead"


@pytest.mark.parametrize(
    "source",
    [
        "from core.world.fence import Fence",
        "import persistence",
        "def f():\n    from metrics import Sample",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from worlds.demo import WORLDS",
    ],
)
def test_the_check_catches_every_way_of_reaching_past(source):
    assert any(forbidden(m) for m in imported_modules(source))


def test_the_check_allows_the_boundary_itself():
    assert not any(forbidden(m) for m in imported_modules("import api\nfrom api import Frame"))
