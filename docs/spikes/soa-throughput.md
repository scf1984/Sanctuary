# Spike: SoA throughput and the catch-up budget

Tracks issue #1.

## Status: blocked on execution, not yet measured

This spike could not be completed end-to-end. The agent session that authored this report had no
permission to execute Python (`python3 -c ...`, `python3 <script>`, and `python3 -m pip ...` were
all rejected by the harness with "This command requires approval", confirmed both directly and via
a fresh subagent). The `.github/workflows/claude.yml` workflow that triggers this agent does not
set `claude_args: --allowed-tools`, so no Bash execution permission was granted for this run.

What follows is the benchmark design and harness, ready to run, plus the shape of the report this
issue calls for. The measured numbers, the recommendation, and any resulting edit to the ratio
table in `CLAUDE.md` §2.1 are **not yet filled in**. Per CLAUDE.md §8.5 ("measure, do not guess")
and the entire premise of this issue, those numbers must not be estimated or fabricated.

### To unblock

Either:
- Run `python3 docs/spikes/soa_throughput_bench.py` locally (only dependency is NumPy, already a
  hard dependency of the core per CLAUDE.md §3) and paste the output back into this issue so the
  report and `CLAUDE.md` §2.1 can be finalized, or
- Re-run this agent with Bash execution permitted (e.g. `claude_args: '--allowed-tools
  "Bash(python3:*)"'` in `.github/workflows/claude.yml`).

## Why

Every offline-advancement decision in `CLAUDE.md` §2.1 and §2.4 rests on an estimate of
~10⁷ entity-updates/sec from a NumPy structure-of-arrays core. That number has never been
measured. This spike measures it, and checks what it implies for closing a 7-day absence.

## Method

`docs/spikes/soa_throughput_bench.py` is throwaway spike code (CLAUDE.md §8.3) — not part of
`core/`, not to be imported by anything else. It builds one representative tick that touches every
entity the way the real core will (CLAUDE.md §2.3):

1. position integration (`x += vx * dt`, clamped to world bounds)
2. a spatial-hash neighbour lookup (grid-cell hashing, standing in for `InteractionGrid`)
3. an energy upkeep decrement, scaled by local crowding
4. a threshold comparison (`energy <= 0`, i.e. starving)
5. a masked selection applying the consequence (refeed, standing in for the free-list
   replacement a real tick performs)

The same five steps are implemented twice: once over global NumPy arrays (structure-of-arrays),
once over a plain Python list of `__slots__` objects with a per-tick `dict`-based neighbour grid.
Both are benchmarked at **1,000 / 5,000 / 20,000 / 100,000** rows, with a warmup period before
timing to avoid first-touch page-fault skew. A doubling-copy (`np.concatenate`) is timed
separately at each size, representing the array-growth mitigation in CLAUDE.md §2.3 item 1.

7-day catch-up wall-clock is derived from the measured SoA per-tick time: at 1 tick = 1 sim-minute
(CLAUDE.md §2.1), a 7-day absence owes `7 × 24 × 60 = 10,080` ticks, so
`wall_clock_seconds = 10,080 × seconds_per_soa_tick`.

## Results

*(pending — see Status above)*

| n | SoA updates/s | Python updates/s | SoA/Python ratio | growth copy (ms) | 7-day catch-up (s) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | TBD | TBD | TBD | TBD | TBD |
| 5,000 | TBD | TBD | TBD | TBD | TBD |
| 20,000 | TBD | TBD | TBD | TBD | TBD |
| 100,000 | TBD | TBD | TBD | TBD | TBD |

## Recommendation

*(pending real numbers)* — once measured: either confirm the CLAUDE.md §2.1 ratio table as-is, or
state precisely what it must change to (tick size, live rate, or feeding-event ratio) and update
that table in the same PR that fills in this report.
