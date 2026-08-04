from tools.check_wall_clock_imports import CORE_ROOT, find_violations
from tools.sources import iter_source_files


def test_core_has_no_wall_clock_imports():
    # CLAUDE.md §2.1: no wall-clock reads anywhere in simulation logic. This mirrors
    # what CI does for this rule — running it under pytest, rather than
    # only as a standalone script, is what makes it an enforced check today rather than a tool
    # nothing consults (see CLAUDE.md §4 on decorative abstractions).
    offenses = [
        f"{path}: imports {module!r}"
        for path in iter_source_files(CORE_ROOT)
        for module in find_violations(path)
    ]
    assert offenses == []
