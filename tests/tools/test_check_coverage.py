"""The per-module coverage gate (#47).

The gate's whole purpose is catching a module that arrived with no tests, and the reason it is
per-module rather than a global percentage is that the global number *cannot* — 1,600 statements at
98% absorb a new untested module without dropping below any threshold set with headroom. That
property is what `test_a_global_threshold_would_have_passed_this` pins, using the real measured
figures, because it is the argument for the whole design and a future reader will otherwise wonder
why `--cov-fail-under` was not enough.

`main()` is exercised through a temporary `coverage.json` rather than the repository's own, so these
tests assert the gate's behaviour instead of restating today's coverage — which would make them fail
on every unrelated change and be deleted within a week (CLAUDE.md §8.1).
"""

import json
from pathlib import Path

import pytest

from tools import check_coverage
from tools.check_coverage import (
    MINIMUM_MODULE_COVERAGE,
    MINIMUM_STATEMENTS_TO_JUDGE,
    main,
    module_rows,
)


def write_report(tmp_path: Path, monkeypatch, files: dict[str, tuple[int, float]]) -> Path:
    """Point the checker at a synthetic report of `{path: (statements, percent)}`."""
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps(
            {
                "files": {
                    name: {"summary": {"num_statements": statements, "percent_covered": percent}}
                    for name, (statements, percent) in files.items()
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(check_coverage, "COVERAGE_JSON", report)
    return report


class TestModuleRows:
    def test_orders_weakest_first(self):
        rows = module_rows(
            {
                "files": {
                    "core/b.py": {"summary": {"num_statements": 10, "percent_covered": 95.0}},
                    "core/a.py": {"summary": {"num_statements": 20, "percent_covered": 40.0}},
                }
            }
        )
        assert [name for name, _statements, _percent in rows] == ["core/a.py", "core/b.py"]

    def test_ties_break_by_name_so_two_runs_print_the_same_thing(self):
        rows = module_rows(
            {
                "files": {
                    "core/z.py": {"summary": {"num_statements": 5, "percent_covered": 80.0}},
                    "core/a.py": {"summary": {"num_statements": 5, "percent_covered": 80.0}},
                }
            }
        )
        assert [name for name, _statements, _percent in rows] == ["core/a.py", "core/z.py"]

    def test_native_path_separators_are_reported_as_posix(self):
        """coverage.json keys are OS-native, so a Windows failure would otherwise read differently
        from the same failure on the Linux runner."""
        rows = module_rows(
            {
                "files": {
                    "core\\world\\tick.py": {
                        "summary": {"num_statements": 30, "percent_covered": 100.0}
                    }
                }
            }
        )
        assert rows[0][0] == "core/world/tick.py"


class TestGate:
    def test_passes_when_every_module_clears_the_floor(self, tmp_path, monkeypatch):
        write_report(tmp_path, monkeypatch, {"core/a.py": (100, MINIMUM_MODULE_COVERAGE + 1.0)})
        assert main() == 0

    def test_fails_a_module_below_the_floor(self, tmp_path, monkeypatch):
        write_report(tmp_path, monkeypatch, {"core/a.py": (100, MINIMUM_MODULE_COVERAGE - 1.0)})
        assert main() == 1

    def test_names_the_offending_module(self, tmp_path, monkeypatch, capsys):
        write_report(
            tmp_path,
            monkeypatch,
            {
                "core/tested.py": (100, 99.0),
                "core/untested.py": (100, 20.0),
            },
        )
        assert main() == 1
        out = capsys.readouterr().out
        assert "core/untested.py" in out.split("below the")[1]
        assert "core/tested.py" not in out.split("below the")[1]

    def test_a_tiny_module_is_not_judged(self, tmp_path, monkeypatch):
        """core/ecology/aging.py scored 89% completely untested: 8 of its 9 statements are imports
        and a `def`, which execute at import time. No percentage can separate that from a tested
        module, so files this small are reported and not gated."""
        write_report(
            tmp_path, monkeypatch, {"core/tiny.py": (MINIMUM_STATEMENTS_TO_JUDGE - 1, 10.0)}
        )
        assert main() == 0

    def test_a_module_at_the_size_threshold_is_judged(self, tmp_path, monkeypatch):
        write_report(tmp_path, monkeypatch, {"core/big.py": (MINIMUM_STATEMENTS_TO_JUDGE, 10.0)})
        assert main() == 1

    def test_missing_report_fails_rather_than_passing_vacuously(self, tmp_path, monkeypatch):
        monkeypatch.setattr(check_coverage, "COVERAGE_JSON", tmp_path / "absent.json")
        assert main() == 1

    def test_empty_report_fails_rather_than_passing_vacuously(self, tmp_path, monkeypatch):
        """A run that measured nothing clears every threshold, which is the one way this gate could
        report success while telling us nothing (CLAUDE.md §8.7: fail loudly)."""
        write_report(tmp_path, monkeypatch, {})
        assert main() == 1

    def test_the_report_is_printed_even_when_the_gate_passes(self, tmp_path, monkeypatch, capsys):
        """"Coverage is reported per module" is half of what #47 asks for, and a gate that prints
        only on failure never reports anything."""
        write_report(tmp_path, monkeypatch, {"core/a.py": (100, 95.0), "core/b.py": (50, 88.0)})
        assert main() == 0
        out = capsys.readouterr().out
        assert "core/a.py" in out
        assert "core/b.py" in out


class TestWhyPerModule:
    def test_a_global_threshold_would_have_passed_this(self):
        """The measurement that rules out `--cov-fail-under`, using core/'s real figures.

        At the commit that added this gate: 1,603 statements, 27 missed, 98% covered. Adding a
        200-statement module with zero real testing — say 30% from its import-time `def` lines —
        leaves the repository-wide figure comfortably above any threshold that also has room for
        ordinary refactoring, while the per-module floor catches it outright.
        """
        covered, total = 1603 - 27, 1603
        untested_statements, untested_fraction = 200, 0.30

        combined = (covered + untested_statements * untested_fraction) / (
            total + untested_statements
        )

        assert combined * 100 > 85.0, "a global gate with headroom would not fire"
        assert untested_fraction * 100 < MINIMUM_MODULE_COVERAGE, "the per-module floor does"


@pytest.mark.parametrize("name", ["MINIMUM_MODULE_COVERAGE", "MINIMUM_STATEMENTS_TO_JUDGE"])
def test_thresholds_are_documented_where_they_are_defined(name):
    """#47's "the threshold is documented with its reasoning" — enforced, since a bare number here
    is exactly the "chase a number" failure the issue warns against."""
    source = Path(check_coverage.__file__).read_text(encoding="utf-8")
    declaration = source.index(f"{name} = ")
    preceding_comment = source[:declaration].rstrip().rsplit("\n\n", 1)[-1]
    assert preceding_comment.lstrip().startswith("#"), f"{name} has no rationale comment"
    assert len(preceding_comment) > 200, f"{name}'s rationale is too thin to be a reason"
