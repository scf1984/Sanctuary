"""Fail if any file outside legacy/ imports from it.

legacy/ holds the 2017-2023 prototype (see legacy/README.md and CLAUDE.md §1): it does not
run and must not be extended. Run as `python -m tools.check_legacy_imports` from the repository
root; exits non-zero and prints every offending import if new code starts depending on it.

Run as a *module*, not as a path, because it imports `tools.sources` — shared with its sibling
check so the walk and the decode exist once (#88). Executing the file directly puts `tools/` on
`sys.path` instead of the repository root, and the import fails outright rather than quietly
finding something else.
"""
import ast
import sys
from pathlib import Path

from tools.sources import iter_source_files, parse_source

REPO_ROOT = Path(__file__).resolve().parent.parent
# The prototype quarantine itself: legacy/ is full of imports from legacy/, and they are the one
# place those are allowed. Everything else this walk skips — virtualenvs, worktrees, caches — is
# environment rather than rule, and belongs to iter_source_files rather than here.
SKIPPED_DIRS = frozenset({'legacy'})


def imports_legacy(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names if alias.name == 'legacy' or alias.name.startswith('legacy.')]
    if isinstance(node, ast.ImportFrom):
        if node.module and (node.module == 'legacy' or node.module.startswith('legacy.')):
            return [node.module]
    return []


def find_violations(path):
    violations = []
    for node in ast.walk(parse_source(path)):
        violations.extend(imports_legacy(node))
    return violations


def main():
    offenses = []
    for path in iter_source_files(REPO_ROOT, SKIPPED_DIRS):
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
