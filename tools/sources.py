"""One walk over this repository's Python sources, and one way to read them.

Both checks in this directory ask "does any file under X do Y?", and both got the same two things
wrong in the same way, because the second was written by copying the first (#88): each decoded
source with the *locale's* encoding, which is cp1252 on Windows and raises on the first byte that
codepage cannot map, and the legacy check walked the entire repository including virtualenvs. CI
never saw either, since Linux's locale encoding is already UTF-8 and the runner has no virtualenv
in the tree.

The reason to share the walk rather than fix it twice is that fixing it twice leaves the next
check to be written by copying one of them, and inheriting the third copy of the bug.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path


def iter_source_files(root: Path, skip_names: frozenset[str] = frozenset()) -> Iterator[Path]:
    """Every `.py` file under `root` that is part of this project, in sorted path order.

    Sorted so that a failing run names its offences in the same sequence twice, which is what
    makes the output of a check diffable between runs.

    Two rules prune directories, and both are deliberately *structural* rather than a list of
    names to skip:

    - **A name starting with a dot.** `.git`, `.tox`, `.mypy_cache`, `.ruff_cache`,
      `.pytest_cache` — and `.claude/worktrees`, which holds entire second checkouts of this
      repository (CLAUDE.md §8.9 puts every issue in one), so a check run from the main tree
      would otherwise scan another branch's files and report offences that are not in the tree
      it was asked about.
    - **A directory containing `pyvenv.cfg`**, which is what makes a directory a virtualenv
      (PEP 405) whatever it has been named. A name list is the wrong instrument here: §8.8
      records this repository committing a `.venv312/` precisely because the `.venv/` pattern did
      not match it, and `venv/` without the leading dot is as common a convention as `.venv/`.

    skip_names: directories the caller excludes as a matter of *project rule* rather than
        environment hygiene — `legacy/` for the legacy-import check, which must not scan the
        prototype it exists to quarantine.
    """
    for directory, subdirectories, filenames in root.walk():
        # Assigning into the slice is what prunes the walk: `Path.walk` reads this list to decide
        # where to descend, so an excluded directory is never entered rather than entered and
        # filtered. That is the difference between skipping site-packages and parsing it.
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if not name.startswith(".")
            and name not in skip_names
            and not (directory / name / "pyvenv.cfg").exists()
        )
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield directory / filename


def parse_source(path: Path) -> ast.Module:
    """Parse `path`, decoding it the way Python itself decodes source.

    Read as bytes and handed to `ast.parse` unmodified, so the *compiler* applies Python's source
    encoding rules (PEP 3120: UTF-8 unless the file says otherwise) instead of this module
    guessing. The point is not that UTF-8 is the better guess — it is that there is no encoding
    argument here to forget, which is how #88 happened twice.
    """
    return ast.parse(path.read_bytes(), filename=str(path))
