# Workflows

## `claude.yml` — delegated implementation

Mention `@claude` in a comment on an issue or pull request, or open an issue containing `@claude`,
and the agent starts a run.

### Setup

Authentication is already configured: `/install-github-app` installed the Claude GitHub App and
stored a `CLAUDE_CODE_OAUTH_TOKEN` repository secret. Nothing further is required for the workflow
to run.

One thing is still worth doing:

**Protect `master`.** Settings → Branches → require a pull request before merging.
The workflow itself holds only `contents: read`, so it cannot push to `master` — the action mints
its own installation token via OIDC to create branches. Branch protection is therefore defence in
depth rather than the sole control, but it is what guarantees agent work arrives as a reviewable
pull request rather than a commit.

### Access control

Two independent gates, deliberately:

- The job's `if` condition rejects any `author_association` outside `OWNER`, `MEMBER`, or
  `COLLABORATOR`, before a runner starts.
- `claude-code-action` independently requires write access of the triggering user.

The first exists because this repository is **public**: without it, anyone could open an issue
containing `@claude` and start a run. The second exists because the first is a workflow expression,
and workflow expressions are easier to get subtly wrong than the action's own check. Neither should
be the only thing standing between a public repository and someone else's token spend.

### What the agent is told

`CLAUDE.md` loads automatically and is not duplicated here. The workflow appends only the rules most
likely to be skipped under delegation — blockers are binding (§7.1), abstractions are owned by their
issue (§7.2), tests come first where the contract is checkable (§8.1), every line must be
justifiable (§8.2), arrays document shape/dtype/unit (§8.4), and performance claims cite benchmarks
(§8.5).

These go in `--append-system-prompt`, not the `prompt` input. In tag mode `prompt` *supplements the
triggering comment*, so rules placed there would be mixed into the user's request rather than acting
as standing guidance.

---

## `claude-code-review.yml` — automatic pull request review

Added by `/install-github-app`. Reviews every pull request on open, update, reopen, and
ready-for-review.

> **Note:** automatic PR review was considered and **declined** during planning (see #44), on the
> grounds that it had not yet earned its cost. It arrived as part of the app installer rather than
> by that decision, and is currently live on every PR.
>
> Two options, both defensible: keep it, in which case it usefully enforces the `CLAUDE.md` §8
> practice rules that are otherwise unenforced; or delete it, restoring the original decision.
> It should not stay by accident.

If kept, it is worth pointing at `CLAUDE.md` §8 explicitly, so review checks this project's actual
rules — justification, test-first, array units, no speculative abstraction — rather than generic
style.

### Automated blocker enforcement

Still not adopted. §7.1 depends on each agent reading `CLAUDE.md` and honouring it. If violations
appear in practice, revisit: a workflow that keeps the `blocked` label accurate as blockers close,
and refuses agent work on an issue with open blockers, would make the rule mechanical.
