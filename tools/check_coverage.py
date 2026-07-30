"""Fail if any module under core/ falls below the per-module coverage floor (#47).

Run as `python -m tools.check_coverage` from the repository root, after a run that wrote
`coverage.json` (see the `coverage` job in .github/workflows/ci.yml). Prints every module's figure
in ascending order — the report *is* the output, so a passing run still shows what is weakest —
and exits non-zero naming any module under the floor.

**Per module, not a global percentage, because the global number cannot detect the thing worth
detecting.** CLAUDE.md §6 is explicit that coverage is a weak proxy for correctness here — safety
comes from invariants, statistical and property tests — so the only job this check has is catching
a module that arrived with no tests at all. A repository-wide threshold is the wrong instrument for
that: `core/` currently holds 1,603 statements at 98%, so a brand-new untested 200-statement module
would pull the total to about 87% and sail past any threshold set with headroom, while one module
rotting to nothing can be masked indefinitely by the others improving. A floor applied to each
module independently cannot be averaged away.

The floor is deliberately far below what a tested module scores, so that ordinary refactoring never
trips it. Chasing the number is the failure mode §8.1 warns about — "a test asserting behaviour
nothing depends on is a maintenance cost wearing the costume of safety" — and a gate set just under
the current figure manufactures exactly that pressure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_JSON = REPO_ROOT / "coverage.json"

# Measured, not chosen (CLAUDE.md §8.5). Two populations were measured on the commit that added
# this check, over the same 794-test suite:
#
#   - modules with real tests:  93% - 100%   (lowest: core/selection.py at 93%)
#   - modules merely imported:  14% -  59%   (highest of meaningful size: core/behaviour/
#                                             exertion.py at 59%, 29 statements)
#
# The import-only figures come from running coverage over a single test that imports every module
# under core/ and exercises nothing; `def` and `class` statements execute at import, so an untested
# module scores well above zero. 70% sits in the empty band between the two: 23 points of headroom
# below today's weakest tested module, 11 above the highest untested one.
MINIMUM_MODULE_COVERAGE = 70.0

# Files with almost no executable content are the known blind spot, and excluding them is honest
# rather than lenient. core/ecology/aging.py scored **89% while completely untested**, because 8 of
# its 9 statements are imports and a `def`. No percentage floor can separate that from a tested
# module, so a percentage is not the instrument for these files — and inventing a second rule to
# catch them would be a gate that fires on files too small to hide a bug in. The threshold below is
# the smallest module that carries real branching in core/ today (core/ecology/aging.py has 9
# statements; the next smallest, core/genetics/distance.py, has 16).
MINIMUM_STATEMENTS_TO_JUDGE = 12


def module_rows(coverage_data: dict) -> list[tuple[str, int, float]]:
    """`(module, statements, percent)` for every measured file, weakest first.

    Sorted by coverage so the output leads with whatever is closest to failing, and by name within
    a tie so two runs over the same tree print the same thing (the reason `tools.sources` sorts its
    walk).
    """
    rows = [
        (
            # coverage.json keys are OS-native paths; posix form so a failure reads the same on
            # Windows and on the Linux runner.
            Path(name).as_posix(),
            summary["num_statements"],
            summary["percent_covered"],
        )
        for name, file_data in coverage_data["files"].items()
        for summary in [file_data["summary"]]
    ]
    return sorted(rows, key=lambda row: (row[2], row[0]))


def main() -> int:
    if not COVERAGE_JSON.exists():
        print(f"No coverage data at {COVERAGE_JSON}; run pytest --cov-report=json first.")
        return 1

    rows = module_rows(json.loads(COVERAGE_JSON.read_text(encoding="utf-8")))
    if not rows:
        # An empty report means the run measured nothing, which passes every threshold vacuously.
        # That is the one way this check could report success while telling us nothing at all.
        print("Coverage data contains no files; nothing was measured.")
        return 1

    print(f"Per-module coverage (floor {MINIMUM_MODULE_COVERAGE:.0f}% over "
          f"{MINIMUM_STATEMENTS_TO_JUDGE}+ statements), weakest first:")
    offenses = []
    for module, statements, percent in rows:
        judged = statements >= MINIMUM_STATEMENTS_TO_JUDGE
        failed = judged and percent < MINIMUM_MODULE_COVERAGE
        note = "" if judged else "  (too small to judge)"
        print(f"  {'FAIL' if failed else 'ok  '} {percent:6.2f}%  {statements:5d} stmts  "
              f"{module}{note}")
        if failed:
            offenses.append(f"{module}: {percent:.2f}% over {statements} statements")

    if offenses:
        print(
            f"\n{len(offenses)} module(s) below the {MINIMUM_MODULE_COVERAGE:.0f}% floor. This "
            "gate exists to catch a module that arrived untested, not to be topped up to a "
            "number — if these lines are genuinely not worth testing, delete them (CLAUDE.md "
            "§8.2) rather than writing a test nobody would miss (§8.1):"
        )
        for offense in offenses:
            print(f"  {offense}")
        return 1

    print(f"\nAll {len(rows)} measured modules clear the floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
