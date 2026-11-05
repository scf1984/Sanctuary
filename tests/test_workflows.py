"""The workflow files parse, and no CI job has silently vanished (#134).

GitHub Actions rejects a workflow with a duplicate mapping key, and it fails with no job-level
signal at all: the run fails in 0s with no logs, no pull_request run is created, and every gate
in the file stops being enforced. `lint` was lost exactly that way in PR #133. Inserting the
`coverage` job consumed the `lint:` header, so lint's steps became a second `steps:` key under
`coverage`.

`yaml.safe_load` cannot catch it. It accepts duplicate keys and keeps the last one, so the broken
file parsed cleanly and reported three jobs. Hence the strict loader below, and hence the job-name
assertion: a parse alone does not notice that a job is gone.

Known limit, recorded on #134: this runs inside the suite, so it protects the author before a
push, not a branch whose workflow is so broken that the suite never runs.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"

# Every gate in ci.yml. Adding or renaming a job means editing this set. That is the point: a job
# that disappears without anyone editing this line is the failure #134 is about.
CI_JOBS = {"test", "version-floor", "coverage", "lint"}


class DuplicateKeyError(ValueError):
    pass


class StrictLoader(yaml.SafeLoader):
    """`SafeLoader` that raises on a repeated key instead of silently keeping the last one."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise DuplicateKeyError(
                    f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
                )
            seen.add(key)
        return super().construct_mapping(node, deep)


def load(text):
    return yaml.load(text, Loader=StrictLoader)


WORKFLOW_FILES = sorted(WORKFLOWS.glob("*.yml"))


def test_there_are_workflows_to_check():
    # Guards the glob: a moved directory would otherwise make every parametrized case vanish and
    # the suite pass having checked nothing.
    assert CI in WORKFLOW_FILES


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_every_workflow_parses_without_duplicate_keys(path):
    assert isinstance(load(path.read_text(encoding="utf-8")), dict)


def test_every_ci_gate_is_still_a_job():
    jobs = load(CI.read_text(encoding="utf-8"))["jobs"]
    assert CI_JOBS <= set(jobs)


def test_a_swallowed_job_header_is_caught():
    """The #133 incident, reproduced against today's ci.yml: delete the `lint:` header, so its body
    merges into the job above it."""
    text = CI.read_text(encoding="utf-8")
    assert "\n  lint:\n" in text
    broken = text.replace("\n  lint:\n", "\n", 1)

    # What made it invisible: the lenient loader accepts it and quietly loses a job.
    assert "lint" not in yaml.safe_load(broken)["jobs"]

    with pytest.raises(DuplicateKeyError, match="duplicate key"):
        load(broken)
