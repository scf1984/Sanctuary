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

This table's tick size and live rate rest on an assumed ~10⁷ entity-updates/sec from a NumPy SoA
core (§2.3). That assumption is now measured and confirmed:
[`docs/spikes/soa-throughput.md`](docs/spikes/soa-throughput.md) found 6.9M–11.6M updates/sec
across 1,000–100,000 rows, with a 7-day offline catch-up resolving in ~108 seconds at 100,000
entities — no change to the table above is required.

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
  Upkeep is charged from the **expressed** phenotype, so a gene's cost and its benefit are
  inseparable — an unexpressed gene is carried and inherited but neither paid for nor gained from.
  Cost coefficients are a per-world table (`core.ecology.metabolism.MetabolismConfig`), never
  constants in `core/`, and **every gene in the vocabulary must declare a cost, zero included**: a
  gene added without one would silently become a free trait, which defeats the whole budget. Any
  gene that *reduces* upkeep — insulation damping thermoregulation cost is the first — must itself
  charge a positive cost, or it is unbounded free benefit and runs away in every climate.
- **Closed nutrient loop.** Energy enters only as sunlight driving plant growth. Carcasses decompose
  and return nutrients to soil. Without closure, populations either explode or flatline.
  **Plants are a field, not entities** — decided in #18, which owned the choice. Growth is a
  per-cell quantity over the terrain grid (`core.ecology.plants`), advanced as whole-array
  operations; there is no plant row in the entity store. Plant *identity* buys the ecology nothing
  — a herbivore grazes a patch, not a numbered shrub — while costing a row and a genome each on a
  population that would outnumber the animals by orders of magnitude. Local competition, the one
  thing the field model had to keep, survives: grazing depletes the cell it happens in, and
  contending grazers split it. Reopen this only for a mechanic that genuinely needs a specific
  plant (a fruiting tree an animal returns to), and as a new issue rather than a retrofit.
  Nutrients are conserved exactly, across soil, standing biomass, and a ledger of what grazing has
  carried out of the field and decomposition has yet to return.
- **Foraging perception is a field query; foraging *choice* is a drive** — decided in #93, which
  owned it. `Plants.perceive(x, y, radius)` reports every patch a forager can find and what stands
  on it. It ranks nothing: which patch is worth walking to weighs payoff against the cost of the
  walk, and that belongs to the hunger drive (#22), not to the field. The settled scoring rule the
  drive implements is `biomass / (1 + distance / forage_reluctance)`, argmax — distance-discounted
  rather than raw, because a grazer that crosses its whole sight range for a marginally richer cell
  neither feeds efficiently nor produces the local grazing pressure the field model exists to
  express. `forage_reluctance` is per-world config owned by the service that scores, never by
  `Plants`, which would otherwise carry a coefficient it never reads (§8.2). Small values keep
  grazers local and strip ground bare before they move; large ones spread pressure out.
  **Sight range gates perception**, supplied as a caller-computed radius exactly as `SpatialIndex`
  takes its cell size: the field knows nothing of genes, and unlimited perception would leave sight
  range charged by the metabolic budget while buying nothing but predator avoidance.
- **Heritable drive weights.** Behaviour is a fixed set of authored drives (hunger, thirst, fear,
  lust, fatigue) competing each tick by utility score — but *the weights and thresholds are genes*.
  Boldness, sociality, and parental investment therefore evolve rather than being designed.
  Behaviour stays explainable ("it fled because fear outscored hunger"), which the intervention
  gameplay depends on.
- **Fear is a noisy-OR over perception channels** — decided in #22, which owns it. A *channel* is
  one sense, with its own physics, reporting a detection probability in `[0, 1]` per entity:

  ```
  perceived_k(i) = Σ_modalities acuity_m(i) × Σ_j emission_m(j) × signature_k(j) × transmission_m(i, j)
  p_c(i)  = saturate( Σ_k aversion_k(i) × perceived_k(i) )
  fear(i) = weight × (1 − Π_c (1 − p_c(i)))
  ```

  **Nothing fears a species. Everything fears a signature.** Each creature carries a position in a
  fixed *d*-dimensional **cue space** — what it smells and looks like — and, separately, an
  **aversion** vector over that same space saying what frightens it. Both are ordinary continuous
  genes, so both are inherited with mutation and both are under selection.

  This is forced, not chosen. A gene "fear of species 47" cannot exist: species are created at
  runtime by speciation, gene columns are fixed at a vocabulary version, and a per-species gene
  would mean a schema migration on every split (§2.3). Making fear heritable therefore *requires*
  factoring danger through a fixed-width cue space. An earlier draft of this section used an
  authored species×species threat matrix; it was wrong for exactly this reason and was removed
  before it shipped.

  **What this buys, unauthored.** A co-evolutionary arms race — prey evolve aversion pointed at
  whatever signature predators emit, predators evolve signatures that drift away from it. Batesian
  mimicry — a harmless lineage whose signature drifts toward a feared one is avoided for free.
  Cannibalism — you smell like yourself, so a lineage whose aversion points at its own signature
  fears its own kind, and can evolve into and out of that. None of these is a mechanic anyone
  implements; they are consequences of the encoding.

  **Modalities differ only in `transmission`.** Scent diffuses (a field); sight is occluded and
  range-limited (pairwise). `acuity_m` is what the animal paid to detect on that modality and
  `emission_m` is how loudly it broadcasts on it, so both ends of every perception are genetic.

  **d = 8.** Enough headroom that two unrelated lineages drifting into the same signature reads as
  evolved mimicry rather than as an accident of a cramped space. Widening is additive and versioned
  (§2.3), so this is a floor; narrowing is not possible.

  **An animal does not perceive itself.** Its own deposit is subtracted from what it samples, which
  is exact rather than approximate because a separable normalized blur's diagonal factorizes per
  axis. Without it, any lineage whose aversion overlapped its own signature — every cannibal —
  would be permanently terrified while standing alone in an empty world.

  Reserved gene block, so that one vocabulary migration covers the mechanics that need it rather
  than three:

  | genes | meaning | cost |
  |---|---|---|
  | `signature_0..7` | position in cue space — what I smell and look like | 0 |
  | `aversion_0..7` | direction in cue space — what frightens me | 0 |
  | `scent_emission` | broadcast strength on the scent modality | see below |
  | `scent_acuity`, `sight_acuity` | detection sensitivity per modality | positive |
  | `camouflage` | damps visual conspicuousness | **positive**, per the insulation rule above |
  | `sex_allocation`, `selfing_rate` | reproduction, continuously (#20) | #20's to set |

  **Camouflage is environment-dependent**: visual conspicuousness is `size × (1 − camouflage ×
  match(local terrain))`, so the same allele is excellent on scree and useless on grass, and
  climate zones select for different camouflage without a designer — the same argument metabolism
  makes for insulation.

  **Scent emission has no cost line above because charging it would be a trap.** Low emission is
  already a survival benefit, so a positive cost would make silence both cheaper *and* safer and
  drive emission to zero in every lineage. Its counterweight has to be a benefit that scales with
  emission — being findable by mates — which is #20's to supply. Until then emission is authored
  per world and not under selection, and this is a known gap rather than a settled answer.

  Noisy-OR rather than a sum or a max: it is the correct composition for independent evidence
  (seeing *and* smelling a predator is worse than either alone), it stays bounded in `[0, 1]` like
  every other drive score, and — the reason it is settled here rather than left open — **adding a
  channel later cannot inflate existing scores past saturation**, so a new sense does not force a
  retune of every other drive's weight.

  **Channels are added, never restructured.** Nothing outside a channel knows how its probability
  was computed, which is what keeps the sequencing of the sensing work free:

  | channel | transmission | acuity gate | owned by |
  |---|---|---|---|
  | scent | advected, diffused per-cue-channel field | `scent_acuity` | #22 |
  | sight | pairwise, line-of-sight and FOV filtered | `sight_acuity` | #24 |

  **Scent is a field and sight is pairwise for physical reasons, not performance ones.** Scent
  diffuses and advects, so wind is a drift term inside the field update and the plume is
  asymmetric for free — a predator approaching from downwind is genuinely stealthy, and nothing
  pairwise expresses that without re-deriving plume geometry per pair. Sight is occluded and
  directional, and a blurred field smears threat straight through a ridge, which is exactly what
  #24 exists to prevent. That the cheap channel is also the one every animal uses every tick is a
  consequence, not the motivation — though it is a load-bearing one: a per-observer nearest-threat
  query over the whole population measured **6.3 s/tick at 100,000 entities** against a 1 s tick
  (§2.1), which is what ruled the pairwise-only design out (#96).

  **A keener nose detects from further away, it is not more frightened.** Concentration decays
  monotonically from the source and detection is a threshold on concentration, so sensitivity and
  range are the *same* parameter for a plume — one blur, no multi-scale bands. This is why the
  scent gene may multiply a sampled field value where a sight gene may not: for sight it would
  make a far-seeing animal merely more afraid of the same thing, which is the wrong selection
  pressure. The detection threshold is what gives the gene teeth; without it every animal detects
  everything faintly and the gene only scales panic.

  **Speciation costs fear nothing at all.** A daughter species inherits its parents' signature and
  aversion genes like any other trait, and the cue field has no per-species structure to extend, so
  there is no table to grow and no id to look up. This is the encoding paying for itself: §2.3's
  "speciation is a species-id write plus a new mask row" stays literally true.

  The cue field itself is **not** fear's property: it is a general facility (`core.ecology.cues`)
  over the terrain grid, mirroring the plant field above. Predators locating prey (#19) and animals
  locating mates (#20) are the *same* query with a different vector — attraction toward a signature
  instead of away from one — so fear is its first reader, not its owner.
- **Emergent speciation.** Genetic distance accumulates between isolated populations; past a
  threshold they can no longer interbreed and are tracked as a new species the player may name.
  This makes isolation — by fence or by terrain — the most rewarding intervention in the game.
  **Reproductive isolation degrades, it does not switch**: interbreeding probability falls
  continuously to zero at the same threshold a split fires on, so by the time two populations
  separate they were already hybridising at a negligible rate. Drift rates and the threshold that
  separates isolated from mixed populations are measured in
  [`docs/spikes/speciation-drift.md`](docs/spikes/speciation-drift.md); the threshold is a
  per-world parameter, never hardcoded in `core/`.

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

**Rendering technology: pygame**, decided when building the minimal live view (issue #12). Pause,
single-step, and adjustable speed need a real per-frame event loop and immediate-mode blitting; a
plotting library's redraw-the-whole-figure model (e.g. matplotlib) fights that instead of
supporting it, and a heavier GUI toolkit (Qt, etc.) buys nothing this diagnostic instrument needs.
Terrain and water render once, as a static NumPy RGB array blitted per generation; only entity
positions and the status readout update every frame.

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

- Seasons and weather as drivers — migration, hibernation, breeding seasonality. Wind is a
  consequence of this rather than a separate question, and scent-on-wind (#97) waits on it.
- **What seeds a founder population's genes**, specifically its aversion vectors (§2.5). If fear is
  purely genetic and founders start random, nothing knows what to fear until selection teaches it,
  and selection teaches it by killing enough prey to shift the distribution. For a slow-breeding
  species the population may crash before the lesson lands, and the result looks like a bug rather
  than like evolution. Options span seeding founders with aversion already pointed at the
  signatures present, seeding at random and accepting a violent first few generations as the
  premise, and a per-world switch. Nothing is decided; the mechanic in §2.5 is unaffected either
  way, since this is an initial condition and not a rule.
- The concrete intervention catalogue and what each costs.
- Precise metric definitions (species count vs. Shannon index vs. within-species genetic diversity).
- Competition format: replicate count, duration, termination condition, what is measured.
- Whether the player names species on speciation, and how lineage is displayed. The mechanic
  itself is settled (§2.5): `core.genetics.speciation` splits a diverged population and records
  the parent link in a `Lineage`. What is still open is purely the player-facing half — naming and
  presentation — so `split()` returns an opaque id and stores no name.

---

## 6. Testing

This section covers *which kinds* of tests to write. §8.1 covers *when* to write them — test-first
where a contract is checkable, explore-then-lock-in where it is not.

Non-determinism rules out golden-output tests. Use instead:

- **Invariants**, asserted every tick in debug builds: energy is never created, populations are
  never negative, no entity leaves the world bounds, no entity occupies a free-list row, total
  nutrients are conserved across the loop.
  **An invariant is not confined to entities** — decided in #91. A check returns `None` when it
  holds and a `Violation` when it does not, describing the breach in its own terms, because the
  original contract returned offending *row indices* and the nutrient pool lives on the plant
  field's grid cells, which have no rows. Anything a check needs beyond the entity store — a
  field, a config table — is **bound by closure when the invariant is built**, following the
  precedent `no_entity_leaves_world_bounds` set with its rectangle. That keeps one predicate
  signature for every invariant and avoids a world-context argument enumerating domains that do
  not exist yet (§8.2); the cross-domain case is covered by the same mechanism — close over the
  field, receive the store.
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

**A closed blocker is not automatically a satisfied one.** If a blocker is closed but its deliverable
plainly does not exist — a benchmark issue whose numbers are still `TBD`, a CI issue with no
pipeline — say so and stop, exactly as if it were open. Closing is an administrative act; the
deliverable is the thing downstream work actually depends on. Do not build on a foundation that was
marked done but is not.

### 7.2 Abstractions are owned by their issue

The issue that introduces an abstraction defines it. Downstream issues consume that abstraction as
built; they do not redesign it. If a downstream issue cannot be implemented against the existing
abstraction, that is a signal to reopen the upstream issue and discuss — not to work around it
locally.

### 7.3 Definition of done

**Do not close an issue whose deliverable does not exist.** If the work could not be completed —
tooling missing, permission denied, blocked on something unforeseen — leave it open and say why. An
issue closed with a placeholder deliverable is worse than one left open, because everything
downstream then treats a gap as a foundation. #1 was closed carrying a benchmark report of `TBD`,
and #4 inherited it as a satisfied blocker; that is the failure mode.

An issue is done when: the code is merged; tests were written as §8.1 requires; anything in §5 it
answers has been moved out of Open Questions and into a settled section; any performance claim it
makes is backed by a benchmark rather than an estimate; and every line in the diff can be justified
under §8.2.

### 7.4 Problems found in passing are filed, not mentioned

**Anything you notice that is wrong and is not in your issue's scope becomes a GitHub issue before
you finish.** Not a line in a PR description, not a comment in the code, not a remark in chat —
those are read once and then gone, and the problem is rediscovered months later by someone who
assumes it is new.

This applies to: a test that fails for reasons unrelated to your change, a deliverable that is
missing from a closed issue (§7.3), an abstraction that could not be used as designed (§7.2), a
constant that contradicts the ratio table in §2.1, dead or decorative code (§8.2), and a
performance claim you find is not backed by a measurement (§8.5). It applies whether or not you
believe anyone will act on it. Filing costs a minute; the alternative is that the next person
pays for the discovery again.

Rules for filing:

- **Do not fix it in the same pull request.** One issue per PR (§8.8). Fixing an unrelated problem
  inline makes the diff unreviewable and hides the fix from anyone searching for it later.
- **Say where you found it.** "Found while implementing #16 (PR #78)" is what lets a reader
  reconstruct the context.
- **Include the evidence, not the impression.** The failing assertion and its numbers, the command
  you ran, the commit you verified against. A report that cannot be reproduced will be closed as
  stale.
- **If the problem is that a closed issue's deliverable does not exist, reopen that issue** rather
  than filing a new one — §7.3 makes it that issue's unfinished work, and a duplicate splits the
  history. File separately only when the original genuinely delivered its scope and something new
  has since broken.
- **Report tooling and permission failures too.** A `gh` query that unexpectedly needs approval is
  a configuration bug worth an issue (§7.1), not a reason to work around it silently.

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
- **Stage files by name. Never `git add -A`, `git add .`, or `git commit -a`.** You are responsible
  for every path in a commit, and a wildcard stages whatever happens to be sitting in the tree —
  a local virtualenv, a scratch script, a downloaded fixture, another task's half-finished edit.
  `.gitignore` is a safety net with holes in it, not the control: it only ever lists the mess
  someone already made, so the first time a new kind of artifact appears it is unignored by
  definition. This has already cost us once — a `.venv312/` directory (not matched by the
  `.venv/` pattern) was swept into a commit and pushed before anyone noticed.
  Run `git status` first, stage the paths you meant, and confirm with `git status --short` that
  nothing else came along.

### 8.9 Every issue is worked in a worktree, off a freshly aligned master

This is the workflow for *all* issues, not a convenience for large ones. In order:

1. **Align local `master` with the remote before branching.** `git fetch origin`, then
   fast-forward. Branching off a stale `master` produces a diff full of changes someone else
   already merged, and the conflict is discovered at review time rather than at minute one. This
   repository has several long-lived worktrees, so a local `master` that has not been touched in
   days is the normal case, not the exception.
2. **Do the work in a git worktree**, one per issue, branch named for the issue
   (`fix/91-field-invariants`). Never on `master`, and never in a worktree that already holds
   another issue's work — §8.8's staging rule assumes the tree contains only what you put there,
   and a shared tree breaks that assumption before you type a command.
3. **Open the pull request from that branch**, referencing the issue.
4. **Remove the worktree once the PR is open.** Leaving it behind is how the next task ends up
   started in the wrong tree, and how a stale branch's virtualenv gets swept into a commit. The
   branch survives on the remote; the working copy has done its job.

The stash stack is shared across every worktree of a repository. A bare `git stash` / `git stash
pop` can therefore pop work belonging to a different worktree entirely. Prefer a temporary WIP
commit; if you must stash, tag it (`git stash push -u -m "<tag>"`) and restore by SHA with
`git stash apply`, never `pop`.
