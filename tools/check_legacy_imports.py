#!/usr/bin/env python3
"""Fail if any file outside legacy/ imports from it.

legacy/ holds the 2017-2023 prototype (see legacy/README.md and CLAUDE.md §1): it does not
run and must not be extended. Run as `python tools/check_legacy_imports.py`; exits non-zero
and prints every offending import if new code starts depending on it.
"""
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_DIRS = {'.git', 'legacy'}


def iter_source_files(root):
    for path in root.rglob('*.py'):
        if not set(path.relative_to(root).parts) & EXCLUDED_DIRS:
            yield path


def imports_legacy(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names if alias.name == 'legacy' or alias.name.startswith('legacy.')]
    if isinstance(node, ast.ImportFrom):
        if node.module and (node.module == 'legacy' or node.module.startswith('legacy.')):
            return [node.module]
    return []


def find_violations(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        violations.extend(imports_legacy(node))
    return violations


def main():
    offenses = []
    for path in iter_source_files(REPO_ROOT):
        for module in find_violations(path):
            offenses.append(f'{path.relative_to(REPO_ROOT)}: imports {module!r}')

    if offenses:
        print('Found imports from legacy/ outside legacy/ (forbidden — see legacy/README.md):')
        for offense in offenses:
            print(f'  {offense}')
        return 1

    print('No imports from legacy/ found outside legacy/.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
