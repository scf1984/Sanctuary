import sys

# Deliberately imports no prototype module: those don't run yet (CLAUDE.md §1).


def test_python_floor():
    assert sys.version_info >= (3, 12)
