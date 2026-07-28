#!/usr/bin/env python3
"""Fail if any file under core/ imports a wall-clock module.

CLAUDE.md §2.1: the tick counter is the only clock; real time exists only to compute how many
ticks are owed, and that computation belongs outside core/ (in scheduler/). A wall-clock read
inside simulation logic would make offline catch-up a second code path instead of the same
simulation with rendering off. Run as `python tools/check_wall_clock_imports.py`; exits
non-zero and prints every offending import if new code under core/ starts reading wall-clock
time.
"""
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_ROOT = REPO_ROOT / "core"
WALL_CLOCK_MODULES = {"time", "datetime"}


def iter_source_files(root):
    yield from root.rglob("*.py")


def wall_clock_imports(node):
    if isinstance(node, ast.Import):
        return [
            alias.name
            for alias in node.names
            if alias.name.split(".")[0] in WALL_CLOCK_MODULES
        ]
    if isinstance(node, ast.ImportFrom):
        if node.module and node.module.split(".")[0] in WALL_CLOCK_MODULES:
            return [node.module]
    return []


def find_violations(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        violations.extend(wall_clock_imports(node))
    return violations


def main():
    offenses = []
    for path in iter_source_files(CORE_ROOT):
        for module in find_violations(path):
            offenses.append(f"{path.relative_to(REPO_ROOT)}: imports {module!r}")

    if offenses:
        print("Found wall-clock imports under core/ (forbidden — see CLAUDE.md §2.1):")
        for offense in offenses:
            print(f"  {offense}")
        return 1

    print("No wall-clock imports found under core/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
