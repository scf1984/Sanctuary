"""Wall-clock assertions run only on uninstrumented code (#232).

A test that asserts a duration or a ratio of durations measures the code only when nothing else is
spending time inside it. Under coverage or a debugger, every line also pays the tracer, and the
coverage job is the longest, most contended run in CI. A timing assertion there measures the runner
rather than the code (§8.5), so it flakes on changes that touch nothing it times. One did, on a
comment-only PR (#229).

The `test` matrix jobs run uninstrumented on every PR, so each timing claim is still checked there.
Skipping, rather than loosening the budgets, keeps those checks as tight as the spikes justified.

The condition is evaluated when the test module is imported, during collection. pytest-cov has
already started coverage by then, so `Coverage.current()` sees it.
"""

from __future__ import annotations

import sys

import pytest


def _instrumented() -> bool:
    # `sys.gettrace()` catches debuggers and coverage's trace-function cores, but coverage can
    # also run on `sys.monitoring` (Python 3.12+), which leaves no trace function behind. So ask
    # coverage directly as well.
    if sys.gettrace() is not None:
        return True
    try:
        import coverage
    except ImportError:
        return False
    return coverage.Coverage.current() is not None


wall_clock = pytest.mark.skipif(
    _instrumented(),
    reason="wall-clock assertion skipped under coverage/tracing; the uninstrumented jobs run it (#232)",
)
