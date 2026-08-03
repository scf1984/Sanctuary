"""The shared source walk both checks in tools/ are built on (#88).

Two defects are pinned here, and both were invisible on the CI runner: source was decoded with
the *locale's* encoding, which is UTF-8 on Linux and cp1252 on Windows, and the walk descended
into virtualenvs, which the runner does not have. Neither can be caught by a test that merely
runs the checks on Linux and looks at the exit code, so both tests below reproduce the condition
rather than the platform.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.sources import iter_source_files, parse_source

REPO_ROOT = Path(__file__).resolve().parents[2]

# U+201D RIGHT DOUBLE QUOTATION MARK is 0xE2 0x80 0x9D in UTF-8, and cp1252 has no mapping for
# 0x9D at all — so a file containing it decodes cleanly as UTF-8 and raises as cp1252. That is
# byte-for-byte the failure in #88's traceback ("can't decode byte 0x9d"), which came from a
# package fixture inside a local .venv.
UNDECODABLE_IN_CP1252 = '"""A curly quote: ”"""\nimport os\n'

CHECKS = ("tools.check_wall_clock_imports",)


def test_parses_a_file_the_locale_encoding_cannot_decode(tmp_path):
    source = tmp_path / "curly.py"
    source.write_text(UNDECODABLE_IN_CP1252, encoding="utf-8")

    # Reaching the import at all means the docstring above it decoded: the defect raised while
    # reading, before any node was visited.
    tree = parse_source(source)

    assert [node.names[0].name for node in tree.body if hasattr(node, "names")] == ["os"]


@pytest.mark.parametrize("check", CHECKS)
def test_check_never_reads_source_with_the_locale_default_encoding(check):
    """Run the check with Python configured to reject an unspecified text encoding.

    `PYTHONWARNDEFAULTENCODING` makes every `read_text()`/`open()` that omits `encoding` emit an
    `EncodingWarning`, and `-W error` turns that into a crash. This is what makes the Windows
    failure reproducible from Linux CI: the *warning* fires wherever the omission is, rather than
    only where the locale happens to disagree with the bytes on disk.
    """
    completed = subprocess.run(
        [sys.executable, "-W", "error::EncodingWarning", "-m", check],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONWARNDEFAULTENCODING": "1"},
        capture_output=True,
    )

    output = completed.stdout.decode("utf-8", errors="replace")
    errors = completed.stderr.decode("utf-8", errors="replace")
    assert completed.returncode == 0, f"{check} failed:\n{errors}"
    # Asserting on the output too, so that a check which somehow exits 0 without walking anything
    # cannot pass this vacuously.
    assert output.startswith("No "), output


def test_skips_virtualenvs_whatever_they_are_named(tmp_path):
    """A virtualenv is a directory containing `pyvenv.cfg` (PEP 405), not a directory called
    `.venv`. CLAUDE.md §8.8 records this repository committing a `.venv312/` that the `.venv/`
    pattern did not match, so the name is exactly the thing not to test on."""
    for name in ("venv", ".venv312", "my-env"):
        environment = tmp_path / name / "Lib" / "site-packages"
        environment.mkdir(parents=True)
        (tmp_path / name / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
        (environment / "vendored.py").write_text("import time\n", encoding="utf-8")

    (tmp_path / "mine.py").write_text("import os\n", encoding="utf-8")

    assert list(iter_source_files(tmp_path)) == [tmp_path / "mine.py"]


def test_skips_dot_directories(tmp_path):
    # .claude/worktrees is the one that matters beyond hygiene: CLAUDE.md §8.9 puts every issue in
    # a worktree there, so each is a whole second checkout of this repository. Without this rule a
    # check run from the main tree reports offences that live on somebody else's branch.
    for directory in (".git", ".mypy_cache", ".claude/worktrees/fix-1/core"):
        (tmp_path / directory).mkdir(parents=True)
        (tmp_path / directory / "elsewhere.py").write_text("import time\n", encoding="utf-8")

    (tmp_path / "mine.py").write_text("import os\n", encoding="utf-8")

    assert list(iter_source_files(tmp_path)) == [tmp_path / "mine.py"]


def test_yields_in_sorted_path_order(tmp_path):
    for name in ("zebra.py", "aardvark.py", "moose.py"):
        (tmp_path / name).write_text("import os\n", encoding="utf-8")
    nested = tmp_path / "beta"
    nested.mkdir()
    (nested / "inner.py").write_text("import os\n", encoding="utf-8")

    found = [path.relative_to(tmp_path).as_posix() for path in iter_source_files(tmp_path)]

    assert found == ["aardvark.py", "moose.py", "zebra.py", "beta/inner.py"]
