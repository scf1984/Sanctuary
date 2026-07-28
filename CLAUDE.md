# Sanctuary

An evolving-ecosystem simulation game. The player is a steward of a living world: they watch it,
intervene in it, and live with the consequences. Animals eat, breed, and die; traits are inherited
with mutation; isolated populations diverge and eventually split into new species. The world keeps
running while the player is away.

The design rule that governs everything below: **we author the physics, not the outcomes.** No
behaviour, population level, or evolutionary result should be hardcoded if it can instead fall out
of energy, terrain, and selection.

---

## 1. Status

The repository currently holds a 2017–2023 prototype that **does not run**. It is kept for
reference and for the ideas worth carrying forward, not as a foundation.

Carry forward:

- **Trait genetics** (`traits.py`) — Gaussian inheritance around the parental mean, clamped to a
  bounded drift range. Conceptually sound; this is the core of the game.
- **Entity indirection by id** — entities referenced by id and resolved through a central store,
  never by direct object pointer, so stale references to dead entities are detectable.
- **Spatial hashing** (`InteractionGrid`) — cell size derived from the maximum sensing range.

Do not carry forward:

- Singleton metaclass on `World` / `Blackboard` / `EventHeap`. It silently ignores constructor
  arguments after the first call and makes test isolation impossible.
- `Event` as a dataclass with a `Callable` field that subclasses override with a method — the two
  mechanisms fight and positional construction misassigns fields.
- Base classes living in package `__init__.py` with subclasses in a sibling module.
- The `stats` package: unreachable, imports symbols that do not exist, and `Stat.update` fuses four
  responsibilities.
- Decorative abstractions: `StateTransitions` and `FoodChain` are declared and then never consulted.

Known bugs in the prototype are catalogued in the revival issues; they are useful as a list of
things the new core must not reproduce.

---

## 2. Settled decisions

These were decided deliberately. Reopen them explicitly, not by accident.

### 2.1 Time

- **The tick counter is the only clock.** No wall-clock reads anywhere in simulation logic. Real
  time exists solely to compute how many ticks are owed.
- **The world advances at the same sim-rate whether or not anyone is watching.** Offline is not a
  different simulation; it is the same simulation with rendering off.
- **The renderer interpolates between ticks.** Visual smoothness must not constrain tick size.

Baseline ratios — tunable, but they must be tuned *as a table*, not as constants that drift apart:

| Quantity | Baseline | Notes |
|---|---|---|
| Tick | 1 sim-minute | |
| Live rate | 1 tick / real second | sim-day ≈ 24 real minutes |
| Herbivore lifespan | ~1 sim-year | ≈ 6 real days |
| Generation time | ~2 sim-months | ≈ 1 real day |
| Feeding events per lifetime | **~10²** | reality is 10³–10⁴; see below |

The feeding ratio is the main suspension-of-disbelief lever. Compressing the clock uniformly gives
an animal thousands of meals per lifetime, so nothing legible happens between births. Digestion must
run *much* slower relative to aging than reality. Distort ratios deliberately and record them here.

### 2.2 Randomness and reproducibility

- **The simulation is not deterministic.** Two identical states may evolve differently. This is a
  design choice, not an accident.
- Seeds are still logged per run so a crash can be replayed for debugging. That is an engineering
  tool, never a player-facing promise.
- **Consequence — competitions are statistical.** "State A beat state B" is meaningless from a
  single run, because variance between two runs of the *same* state can exceed the difference
  between A and B. Competitions run N replicates per starting state and report distributions
  ("A held higher biodiversity in 73 of 100 runs"), not winners.
- **Consequence — no golden-output tests.** Correctness is enforced by invariants (§6).

### 2.3 Simulation core

- **Structure-of-arrays over NumPy.** Entities are rows in typed arrays; updates are vectorized
  batch operations, not per-object Python calls. This is what makes offline catch-up affordable.
- **Domain services own column blocks.** `Ecology`, `Genetics`, `Behaviour` and friends each own the
  arrays they govern and expose ecological verbs — `feed(predators, prey)`, `inherit(parents)`,
  `sense(observers)`. Vectorized inside, domain language outside. **Raw row indices must never
  escape a service boundary**; callers pass and receive selections. Per-entity view objects are
  permitted for debugging and UI only, never in a tick loop — a view is a Python-level scalar
  access and forfeits the entire performance case for SoA.

- **One global array set. No per-species storage anywhere.**
  - *Physical state* — position, energy, age, health, species id, drive scores — lives in a single
    global array set covering all creatures.
  - *Genes* live in a single global `(entities × genes)` matrix over one shared gene vocabulary.
    Every creature has every gene slot.
  - *Species differ by expression mask*, not by layout. A species registry marks which genes each
    species expresses; unexpressed genes are inert but **still inherited**, so dormant traits can
    resurface generations later when conditions or masks change. This is a feature, not a
    concession — it is how atavism and latent variation become possible.
  - **Speciation is a species-id write plus a new mask row.** Nothing is reallocated, copied, or
    restructured. That is the point.

  Rationale: emergent speciation multiplies species at runtime *by design*. Per-species arrays would
  make every system loop over species, so a world that speciated into 50 populations of 40 animals
  would issue 50 tiny NumPy calls per system per tick, and Python call overhead would dominate —
  success at speciation would degrade performance, which is exactly backwards. With one global
  layout, every system is a single vectorized pass regardless of species count, and speciation is
  nearly free. The cost is memory on unexpressed gene columns, which is negligible (100k creatures ×
  60 genes × float32 ≈ 24 MB).

  Consequence to accept deliberately: all species draw from **one fixed gene vocabulary**. Adding a
  genuinely new gene means widening the matrix, a schema migration for existing worlds. Gene
  vocabulary is therefore versioned and additive-only; see the genetics issues.
- **Capacity grows dynamically**, with these mitigations against reallocation pauses:
  1. Free-list row reuse, so churn does not grow capacity — only a new high-water mark does.
  2. Growth checked at tick boundaries only, never mid-tick, so no vectorized op holds a view into
     an array being resized. This is the real bug risk, more than the stall.
  3. Preemptive growth during offline catch-up, sized from the high-water mark plus headroom.
  4. Chunked paging held in reserve, only if measurement shows real hitches.
- **Population is emergent.** Carrying capacity = area × primary productivity ÷ per-animal upkeep.
  Never set population directly.
- **The engine ceiling is invisible.** As capacity is approached, density-dependent mortality
  intensifies — crowding, disease, starvation — so the plateau reads as ecology. The hard ceiling
  exists behind that only to keep arrays safe.

### 2.4 Offline advancement

- Wake schedule is **configurable per world** and may decay (hourly for a day, then daily, etc.).
- **The schedule controls when compute happens, not how fast the world moves.** Waking hourly and
  then daily batches the same ticks differently; it must not make the world age more slowly.
- **Standing policies** let the player pre-authorize action while away: auto-cull above a threshold,
  hold fences closed, emergency feed. Without these, a crash on day 3 of an absence is an obituary
  rather than a decision.
- Population warnings are generated during catch-up and surfaced on return.

### 2.5 Energy and genetics

- **Hard energy budget.** Every trait charges continuous upkeep against one metabolic pool — speed,
  size, sight range, gestation. There is no free lunch, so the environment picks the optimum and
  different biomes select different builds without designer intervention. This is the mechanism
  that makes climate zones matter genetically.
- **Closed nutrient loop.** Energy enters only as sunlight driving plant growth. Carcasses decompose
  and return nutrients to soil. Without closure, populations either explode or flatline.
- **Heritable drive weights.** Behaviour is a fixed set of authored drives (hunger, thirst, fear,
  lust, fatigue) competing each tick by utility score — but *the weights and thresholds are genes*.
  Boldness, sociality, and parental investment therefore evolve rather than being designed.
  Behaviour stays explainable ("it fled because fear outscored hunger"), which the intervention
  gameplay depends on.
- **Emergent speciation.** Genetic distance accumulates between isolated populations; past a
  threshold they can no longer interbreed and are tracked as a new species the player may name.
  This makes isolation — by fence or by terrain — the most rewarding intervention in the game.

### 2.6 World and space

- **Heightmap terrain from the start.** Elevation drives movement cost, line-of-sight occlusion,
  downhill water flow and pooling, and temperature by altitude. Climate zones are *consequences of
  terrain*, not painted regions, and mountain ranges become natural isolation barriers.
- **Animals may leave the surface.** Flight and swimming depth are real mechanics, so positions
  carry a z coordinate and the spatial index is volumetric.
  > ⚠️ This is the most expensive decision in this document. It makes spatial indexing, sensing, and
  > rendering 3D problems and complicates the vectorized core. It should be delivered in stages:
  > z-capable data model first, surface-locked movement next, true volumetric flight/swimming last.
  > Everything in §2.6 above works without it.
- World size is set by climate-zone variety and animal home range — large enough to feel vast and
  hold distinct zones, small enough that animals do not wander into irrelevance. Not all zones
  appear in every world.

### 2.7 Player and stakes

- **Steward with constraints.** Interventions cost a limited resource. Extinctions are permanent.
  Total collapse ends the run.
- Interventions include fencing to isolate populations, culling (never total eradication of a
  species by a single action), terraforming, introducing species, and standing policies.
- Metrics — biodiversity, population trends, inter-species interaction — are a secondary surface,
  suitable for a dashboard client or a phone widget.

---

## 3. Target platform

- **Python 3.12+.** The prototype accidentally required exactly 3.10 (it used `X | Y` at runtime,
  needing 3.10+, and `@classmethod` stacked on `@property`, removed in 3.13). Pin the floor in
  `pyproject.toml` and test it in CI so this cannot recur.
- NumPy is a hard dependency of the core.
- The simulation core must remain importable and runnable headless, with no UI dependency.

### 3.1 Deployment target

The destination is a **hosted multi-world service**: worlds tick server-side independently of any
client, with accounts, persistence, and shareable metrics feeding dashboards and phone widgets.

It is delivered in stages, so that ecology work is never blocked on infrastructure:

1. **Headless core + explicit `api/` boundary.** No process separation. Clients embed it.
2. **Thin single-user service.** One process, one world, no accounts. Establishes a real protocol
   boundary and lets the world tick independently of any UI.
3. **Multi-world and persistence at scale.**
4. **Accounts, auth, sharing, competition hosting.**

Stage 1 is a hard prerequisite for everything: if `core/` is clean and UI-free, stages 2–4 are
wrapping work. If it is not, no amount of service code will save it.

### 3.2 Infrastructure

- **Managed cloud, defined in Terraform.** Compute, managed Postgres, object storage, networking and
  IAM are all code. No hand-assembled consoles.
- **Snapshots in object storage, metrics in Postgres.** Snapshots are stored as **opaque versioned
  blobs** — their internals are never modelled in SQL, or every core change becomes a migration, and
  the snapshot format will churn for a long time.
- **Backup and restore is a correctness requirement, not ops hygiene.** Because the simulation is
  non-deterministic (§2.2), a world cannot be regenerated from its seed. The snapshot is the only
  copy in existence. Losing one permanently destroys a world that may represent weeks of play, and
  no amount of compute recovers it. Restores are therefore *tested on a schedule*, never assumed.
- **Deploys must drain, not kill.** Stopping a process without snapshotting every live world
  discards sim-time that cannot be recomputed identically.
- **The repository is public.** No secret ever enters the repo, the container image, or committed
  Terraform state. Secret scanning blocks merge.
- CI enforces lint, types, coverage, container builds, **performance regression gates** (throughput
  is what the offline design rests on), and scheduled long-run soak tests.

Delegated implementation via the GitHub Claude agent is enabled (#44). Automated blocker enforcement
and automated PR review were considered and **deliberately not adopted**, so §7.1 and §8 rely on
each agent reading this file. If violations appear in practice, revisit that decision.

### 3.3 Visualization

The viewer at this stage is a **diagnostic instrument for simulation development**, not a game view.
It exists to answer "why did this population crash", and its priorities are, in order: terrain and
water rendering with elevation shading; entities colored by species; overlays for energy, hunger,
and population density; pause / step / scrub; and click-to-inspect dumping an individual's genes,
drives, and current utility scores.

The prototype's renderer is not a starting point — it draws zero-size ovals (`bbox(0.00)`), has no
terrain, no species distinction, and no way to pause or inspect.

---

## 4. Architecture boundaries

```
core/        simulation. no rendering, no wall clock, no I/O. importable standalone.
  world      terrain, climate, water, tick loop
  entities   global SoA physical state, free lists, capacity growth
  genetics   global gene matrix, species expression masks, inheritance, distance, speciation
  behaviour  drives, utility scoring, action resolution
  ecology    energy flow, feeding, reproduction, decomposition
persistence/ snapshots, world config, intervention history
scheduler/   offline wake schedules, catch-up execution, standing policies
metrics/     biodiversity and population statistics, warnings
api/         boundary between core and any client. the only thing clients may import.
service/     process wrapper around api/. protocol, sessions, world lifecycle.
clients/     diagnostic viewer, dashboard. never import core internals, only api.
```

Rules:

- Nothing in `core/` may read a wall clock, perform I/O, or import from `clients/` or `service/`.
- Every domain service owns its arrays. Row indices do not cross service boundaries.
- Iterate snapshots, never live views. The prototype's `World.update` iterated a live `dict` view
  and would have raised as soon as reproduction worked.
- No singletons. Pass the world/context explicitly.
- If a rule is declared as data (a transition graph, a food web), it must be *consulted* by the code
  that it governs. Do not ship decorative abstractions.

---

## 5. Open questions

Not yet decided. Do not assume answers — ask.

- Seasons and weather as drivers — migration, hibernation, breeding seasonality.
- The concrete intervention catalogue and what each costs.
- Precise metric definitions (species count vs. Shannon index vs. within-species genetic diversity).
- Competition format: replicate count, duration, termination condition, what is measured.
- Whether the player names species on speciation, and how lineage is displayed.
- Client rendering technology for the diagnostic viewer.

---

## 6. Testing

This section covers *which kinds* of tests to write. §8.1 covers *when* to write them — test-first
where a contract is checkable, explore-then-lock-in where it is not.

Non-determinism rules out golden-output tests. Use instead:

- **Invariants**, asserted every tick in debug builds: energy is never created, populations are
  never negative, no entity leaves the world bounds, no entity occupies a free-list row, total
  nutrients are conserved across the loop.
- **Statistical tests** over many seeds: a population under predation should trend toward higher
  speed; an isolated population should accumulate genetic distance; a world with no sunlight should
  collapse. Assert distributions and directions, never exact values.
- **Property-based tests** for genetics: inheritance stays within the clamp range, genetic distance
  is symmetric and satisfies the triangle inequality.
- **Performance tests** as first-class: catch-up throughput must stay within the budget in §2.1, and
  regressions there are bugs, not nice-to-haves.

---

## 7. Working agreement for issues

Work in this repository is tracked as GitHub issues. Agents and humans must follow these rules.

### 7.1 Blockers are binding

Every issue that depends on unfinished work carries a section like:

```
## ⛔ Blocked by
- #4 SoA entity store
- #11 Metabolic energy budget
```

**Do not begin implementation while any blocker is open.** If you pick up an issue and find an open
blocker, comment saying so and stop. Do not stub, mock, or "temporarily" reimplement a blocker's
scope in order to proceed — that is how two incompatible versions of the same abstraction get built.

**The list in an issue body is a snapshot from when it was written, not live state.** Blockers close
over time, and an issue whose body still lists them may be perfectly ready to start. Always check:

```
gh issue view <n> --json state,title
```

Read-only `gh` queries are pre-approved in `.claude/settings.json`, and `GH_TOKEN` is set in the
workflow, so this works without asking for permission. **If a query appears to require approval,
that is a configuration bug worth reporting — not a reason to skip the check, and not a reason to
assume the issue is blocked.** Concluding "I could not query the API" and then proceeding anyway is
the failure this rule exists to prevent.

The `blocked` label tracks the same information, but it is maintained by hand and can lag. Live
issue state is authoritative; the label is a convenience for filtering.

An issue with no `⛔ Blocked by` section is ready to start.

### 7.2 Abstractions are owned by their issue

The issue that introduces an abstraction defines it. Downstream issues consume that abstraction as
built; they do not redesign it. If a downstream issue cannot be implemented against the existing
abstraction, that is a signal to reopen the upstream issue and discuss — not to work around it
locally.

### 7.3 Definition of done

An issue is done when: the code is merged; tests were written as §8.1 requires; anything in §5 it
answers has been moved out of Open Questions and into a settled section; any performance claim it
makes is backed by a benchmark rather than an estimate; and every line in the diff can be justified
under §8.2.

---

## 8. Engineering practice

These are rules, not aspirations. Work that violates them is incomplete regardless of whether it
runs correctly.

### 8.1 Tests come first — where they can

Write the test before the implementation wherever the contract is checkable in advance:

- **Data structures and pure functions** — the entity store, `Selection` algebra, spatial queries,
  snapshot round-trips.
- **Genetics** — property tests are writable before any implementation exists: inheritance stays
  within the clamp range, distance is symmetric and satisfies the triangle inequality.
- **Invariants** — add the invariant to the harness *before* the system that must satisfy it.
- **Bug fixes, always.** A fix without a test that failed before it is not a fix.

Be honest about where test-first does not apply. **Ecological tuning cannot be test-driven** — you
do not know in advance whether predator-prey cycles feel right or whether a ratio produces legible
pacing, so there is no failing test to write. For that work: explore freely, then **lock the result
in with a statistical test before merging**, so the tuning cannot silently regress. Exploration is
not an exemption from tests; it is a different order.

Do not write a test you would not miss. A test asserting behaviour nothing depends on is a
maintenance cost wearing the costume of safety.

### 8.2 Every line must be justified

For any line in a diff you should be able to say why it exists and what breaks without it. If you
cannot, delete it.

- **No code for later.** Speculative generality is the most expensive habit in this repository's
  history: the prototype shipped a state-transition graph and a food web that were declared, wired
  into nothing, and consulted by no one.
- **No parameter, flag, or hook without a caller that needs it now.**
- **No defensive checks against conditions that cannot occur.** If it genuinely cannot occur, it
  belongs in the invariant harness, not as a branch in a hot loop.
- **Deleted code is deleted, not commented out.** Git remembers.

### 8.3 Abstractions are earned, not anticipated

This repository has an unusual exception, so state it plainly: a small number of issues *own* an
abstraction and must design it up front, because many downstream issues depend on its shape. Those
are named — #4 storage, #5 service boundary and `Selection`, #13 gene matrix, #22 drive scoring,
#35 the `api/` surface. For those, design before use is correct.

**Everywhere else the default is the opposite: write the concrete thing, and abstract on the third
repetition, not the first.** An abstraction with a single caller is a guess.

When an abstraction is introduced, its issue states what it is *for* and what it *forbids*. An
abstraction that forbids nothing is not an abstraction — it is indirection.

### 8.4 Vectorized code carries a readability tax — pay it explicitly

SoA was chosen for speed (§2.3) and it costs clarity. That trade is only worth it if the cost is
actively repaid:

- Every array-holding attribute documents **shape, dtype, and unit** — e.g. `(n_entities,) float32,
  joules`.
- Selections and masks are named for what they select — `starving`, `in_range` — never `mask1`.
- A vectorized expression beyond a few operations carries a comment stating its **ecological
  meaning**, not a restatement of the NumPy.
- **Units are stated everywhere.** The prototype compared a degree-valued sight angle against a
  radian difference; the check silently passed always, and nothing caught it.

### 8.5 Measure, do not guess

Any performance claim — in code, a comment, an issue, or a PR description — must cite a benchmark.
"This should be faster" does not justify complexity. The corollary binds equally: do not optimise
without a measurement showing the cost is real. Chunked paging (§2.3) is held in reserve for exactly
this reason.

### 8.6 Comments explain why

The code states what. Comments state why this way rather than the obvious alternative, why a
constant holds that value, and what breaks if it changes. A comment restating the line beneath it is
deleted on sight.

Decisions of consequence belong in this file, not in a comment where one reader will find them.

### 8.7 Fail loudly

No silent fallbacks, no quiet defaults. The prototype's singleton metaclass silently ignored
constructor arguments after the first call, so a wrong world size went unnoticed for years. Prefer a
raised error to a defaulted value, and a tripped invariant to a clamped number.

### 8.8 Commits and pull requests

- One issue per pull request, referencing it.
- **Green CI is a precondition for review, not a stage of it.** The prototype's final commit broke
  the test suite by deleting a symbol the tests imported, and it went unnoticed for two years.
- A PR description states what changed, why, and what was measured.
