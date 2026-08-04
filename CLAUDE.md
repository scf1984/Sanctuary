# Sanctuary

An evolving-ecosystem simulation game. The player is a steward of a living world: they watch it,
intervene in it, and live with the consequences. Animals eat, breed, and die; traits are inherited
with mutation; isolated populations diverge and eventually split into new species. The world keeps
running while the player is away.

The design rule that governs everything below: **we author the physics, not the outcomes.** No
behaviour, population level, or evolutionary result should be hardcoded if it can instead fall out
of energy, terrain, and selection.

---

## 1. What the prototype taught

A 2017–2023 prototype preceded this core. **It is gone** — deleted in #220 once every idea worth
taking from it had been taken, because a quarantined tree nobody may import is not reference
material, it is five tool exclusions and a CI guard defending against a mistake that can no longer
be made (§8.2). `git show 19a5718` has it if a reading is ever wanted.

What it gave, and where each went:

- **Trait genetics** — a draw around the parental mean, clamped to a bounded drift range. The
  *shape* was the thing worth carrying and it is the core of the game; the arithmetic was not sound
  and was re-derived in #104 (§2.5). Its spread coefficient made a closed pool's variance grow by
  half every generation, and its clamp was tight enough to crush that back — two errors cancelling
  into something that looked like convergence, which is why §1 once calling it "conceptually sound"
  was too generous.
- **Entity indirection by id** — entities referenced by id and resolved through a central store,
  never by direct object pointer, so stale references to dead entities are detectable
  (`core.entities.store`).
- **Spatial hashing** — cell size derived from the interaction range. What survives is the idea,
  in `core.ecology.contact`; the pairwise structure it came in did not, because #96 measured a
  per-observer query at **6.3 s/tick at 100,000 entities** against a 1 s tick.

**What it taught by counterexample is the part that still governs new code**, which is why this
list stays here rather than going with the source:

- Singleton metaclasses on `World` / `Blackboard` / `EventHeap`. They silently ignore constructor
  arguments after the first call, so a wrong world size went unnoticed for years — §8.7's founding
  example, and why §4 forbids singletons outright.
- `Event` as a dataclass with a `Callable` field that subclasses override with a method. The two
  mechanisms fight, and positional construction misassigns fields.
- Base classes living in a package `__init__.py` with subclasses in a sibling module.
- A `stats` package that was unreachable, imported symbols that did not exist, and fused four
  responsibilities into one `update`.
- **Decorative abstractions.** `StateTransitions` and `FoodChain` were declared, wired into
  nothing, and consulted by no one. This is the repository's most expensive habit and it is why §4
  requires a rule declared as data to be *consulted* by the code it governs.
- **A degree-valued sight angle compared against a radian difference.** Both are floats, so the
  check silently passed always and nothing could catch it — §8.4's rule that units are stated
  everywhere, and the same shape as #112's elevation-in-metres against positions in world units.
- **Live-view iteration**: `World.update` iterated a live `dict` view and would have raised the
  first time anything was born. §4's "iterate snapshots, never live views".
- **A reversed `atan2` argument order**, so every heading computed from it was wrong.
- **A renderer drawing zero-size ovals** (`bbox(0.00)`) — §3.3's note that it was not a starting
  point.

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

**The order systems run within a tick is a rule, not an implementation detail.** Each system reads
what the previous ones wrote, so the order moves outcomes and is therefore part of the MAJOR
version (§2.8), frozen for the life of a world. It is declared explicitly and never inherited from
import order or from whatever sequence a test happened to use. Settled order, owned by #115:

| # | system | why here |
|---|---|---|
| 1 | plant growth | consumes nutrients returned last tick; a one-tick lag is invisible on a field |
| 2 | cue field rebuild | must precede any sensing, or animals smell **last tick's** world |
| 3 | drive scoring / option sampling | |
| 4 | movement | acts on this tick's decision, not a stale one |
| 5 | exertion recovery | immediately **after** movement, the only thing that adds to the column: the tick's effort is spent and then the tick's rest is taken. Whatever reads `exertion` therefore always reads a recovered value, never a raw one (#107) |
| 6 | feeding, then drinking | you eat and drink where you arrived, not where you left |
| 7 | dehydration, then metabolic upkeep | **after feeding**: an animal standing on a full meadow must not be killed by upkeep it could have paid. "Died on top of food" reads as a bug even when the arithmetic is right, and "died of thirst in a lake" reads the same way. Water loss precedes upkeep so the tick's bill reflects the tick's dryness (#156) |
| 8 | death and decomposition | starvation is only meaningful once the tick's upkeep is charged |
| 9 | age increment | closes a whole tick of living, and runs **before** births — see below |
| 10 | reproduction | **after death**, so rows freed this tick are immediately reusable and a world at capacity can still breed |
| 11 | speciation | periodicity undecided; see #115 |

Two rules the order exists to enforce, which outlive any particular sequence:

- **A newborn does not act in the tick it is born.** A row allocated mid-tick is invisible to the
  systems that already ran and visible to those that follow, so a newborn would be half-simulated —
  sensing a world it never moved in, or moving on a decision nothing scored for it. Reproduction
  therefore runs late, and an entity begins its first *whole* tick before anything asks what it
  wants.
- **`age` counts whole ticks lived**, which is why the increment precedes reproduction rather than
  following it. Incrementing after birth would make a newborn one tick old having lived none.

The order above is **declared as data and consulted**, in `core.world.assembly.TICK_ORDER`: the
assembly builds its systems into a mapping by name and sequences them *by* that tuple, so a system
added without being placed, or a name placed with nothing behind it, raises at assembly time rather
than running wherever import order put it (§4 forbids a rule declared as data that nothing reads).
`build_world` is the only place a store, its services and a loop are wired together — a second
assembly is what §7.2 exists to prevent, and the viewer's world is now config handed to this one.

`TICK_ORDER` holds the **implemented prefix** of the table, in the table's relative order. Feeding
(#19), death and decomposition (#21), reproduction (#20) and speciation (#16) have no system yet
and are absent rather than stubbed (§8.2), so the assembled world runs without eating, dying or
breeding. Movement, by contrast, now acts for **every** animal rather than for one drive's winners:
#114 replaced the argmax over drives with a sampled heading, so there is no longer a subset "driven
by" hunger and no animal stands still merely because the drive that won has no mechanic. That was
#126, and it is closed by the encoding rather than by a special case.

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
- **Population is emergent, and the only way to know what a world holds is to run it.** Carrying
  capacity is area × primary productivity ÷ per-animal upkeep in the same sense that a thrown
  stone's landing point is a function of its launch angle: true, and not a substitute for throwing
  it. Never set population directly, and **never predict it either** — settled in #216, which built
  a forward model and had it removed.

  The model divided field production by what one founding animal costs to keep alive. It measured
  within 4–5% on the world it was built against, and then #179 added predation — a second exit for
  energy, into carrion that rots having fed nobody — and the same model over-predicted by **10×**
  (5,361 against 521 living). Nothing about it was wrong; it was answering a question about a world
  that had stopped existing.

  That is the general failure and the reason this is a rule rather than a preference. **A predictor
  encodes the mechanics it was written against, so every mechanic added afterwards silently
  invalidates it** — and it invalidates it *quietly*, by continuing to return a plausible number.
  A simulation cannot drift from the world in that way, because it *is* the world. The repository
  would have to maintain two models of one ecology and keep them in step, which §2.1's warning about
  constants that drift apart says about a *pair of coefficients*; two whole models is that failure
  at the largest scale it can occur at.

  So: to learn what a set of tunings implies, **build the world and advance it.** That is slower and
  it is the only answer that stays true. It is also what #101's evolved starting states already do —
  generate headless, run long, keep what stabilises — so the honest version of "what does this
  config give me" is a search over simulated worlds, never a formula evaluated against one.

  **This does not weaken "author the physics, not the outcomes"**, and neither did the forecast:
  writing `population = 5000` authors an outcome, while choosing how bright the sun is does not.
  What was wrong was not the wish to size a world, but the belief that the size could be *computed*
  instead of observed.

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
  Cost coefficients are a per-world table, never constants in `core/`, and **every gene in the
  vocabulary must declare a cost, zero included**: a gene added without one would silently become a
  free trait, which defeats the whole budget. Any gene that *reduces* upkeep — insulation damping
  thermoregulation cost is the first — must itself charge a positive cost, or it is unbounded free
  benefit and runs away in every climate.

  **A gene declares its cost, expression mode and unit in one place** — `core.genetics.registry`,
  settled in #111, which owned it. A `GeneSpec` carries all three, and the registry it builds is
  what `Metabolism` and `ExpressionTable` both read, so the two can no longer disagree. Note what
  this does to the completeness rule above: it stops being a *check* and becomes structural, since
  a gene without a cost or a mode is not something the type can express. That is §8.7's preference
  for an unrepresentable failure over a validated one, and it deleted both checks rather than
  moving them.

  **The unit is consulted, not documented.** `GeneRegistry.index_of` takes the unit its caller
  expects, so a config naming a gene it will read as a length is handed a length or an error —
  the check that would have caught #112, where both sides of the mismatch were floats and nothing
  could notice. A gene map is generated from the registry by `describe()` rather than maintained
  as prose here, because a table in this document drifts from the code within two issues.

  **The clamp in `inherit_genes` is a numerical backstop, not the mechanism that bounds traits.**
  Energy and selection are the mechanism: a trait that drifts upward costs more upkeep, so its
  bearer starves sooner and leaves fewer offspring, and the equilibrium sits wherever marginal
  benefit meets marginal cost. The rule this implies is sharper than the cost table alone:
  **every gene needs either an energy cost or a selective consequence.** A gene with neither is a
  free random walk. Exactly one class deliberately is — cue signature (below) has no cost and
  nothing selecting on it, and that neutral drift is a molecular clock, which is precisely what
  makes two isolated populations recognisably different.

- **Inheritance: a logistic draw, floored by a gene, bounded additively** — settled in #104, which
  owned it. An offspring gene is drawn around the parental mean with a spread that is **half the
  parents' gap or the offspring's own `mutability`, whichever is larger**, and clamped to
  `drift_margin` spreads outside the parental min/max.

  Each of those three parts replaced something that was wrong, and the middle one was a defect
  rather than a preference. Spreading the draw by parental disagreement *alone* means identical
  parents produce an identical offspring, so a closed population converges and the more alike its
  members become the faster they become alike: an ending state where every birth is a copy and
  nothing new appears again. [`docs/spikes/speciation-drift.md`](docs/spikes/speciation-drift.md)
  measures both halves — an unfloored pool loses **99.5%** of its within-pool spread by generation
  100 and is still falling, while a floored one settles and holds.

  **The floor is a gene, so a lineage evolves its own evolvability.** Low in a stable world, higher
  in a volatile one, and it needs no energy cost to bound it because high mutability already pays
  for itself in unfit offspring — the one case where a gene's selective consequence is immediate
  enough that the metabolic budget is not required (see the rule above).

  **The draw is logistic**, which is the difference of two Gumbels and therefore the extreme-value
  form: if a parent produces a hundred eggs and two survive, the survivors are that brood's
  extremes, and §2.1 already compresses reproduction rather than simulating each egg. The *two-way*
  form is the point — a one-sided Gumbel's mean sits above its mode, so every trait would ratchet
  upward regardless of selection and the budget would be fighting a built-in bias. What it buys over
  a Gaussian is fatter tails, so a lineage can leave a local optimum in one jump instead of only
  crawling.

  **The clamp is additive, in units of the draw's own spread**, because genes are signed (below) and
  the multiplicative range it replaced *inverted* below zero: two parents at -3 and -2 produced
  `low = -2, high = -3`. An additive margin is translation-invariant, so a gene at -5 drifts exactly
  as one at +5.

  **The coefficient relating spread to the parental gap is derived, not chosen.** For parents drawn
  from a pool of variance `σ²`, their midpoint carries `σ²/2` and a draw of standard deviation
  `k·|a−b|` adds `2k²σ²`, so one generation multiplies variance by `(1/2 + 2k²)` — which is 1 only at
  `k = 1/2`. The legacy value was `1/√2`, inflating variance by half again every generation, and
  **the old rule converged only because its tight multiplicative clamp crushed the excess**: two
  errors cancelling, which is why §1 calling that formula "conceptually sound" was too generous. Any
  future change here re-derives `k` rather than tuning it, because `k` decides whether a closed pool
  freezes, holds, or explodes.

- **Genes are signed, and every gene declares how it is read** — also #104. Storage is ℝ; a gene's
  *expression mode* says what a stored value means, and `Genetics.expressed` applies it at the one
  place a phenotype is produced (`core.genetics.expression`):

  | mode | used by | why |
  |---|---|---|
  | magnitude (`abs`) | size, speed, acuity, camouflage, insulation, mutability, commitment (#100) | a quantity cannot be negative |
  | raw, signed | cue signature, aversion | sign carries information and doubles the discriminating power of cue space |
  | exponential (`exp`) | choice temperature (#114) | a **scale**, which must stay strictly positive and multiply rather than add |

  The exponential mode was added by #114 and is worth distinguishing from magnitude, since both
  produce non-negative phenotypes. A magnitude *folds* at zero: two genes either side of it express
  identically, and drift through zero is a reflection. That is right for a size and wrong for a
  scale, where zero is not the smallest useful value but a division by zero — a Boltzmann
  temperature of zero has no distribution. `exp` never reaches zero however far the gene drifts, so
  the degenerate case is unrepresentable rather than clamped (§8.7), and equal steps in storage are
  equal *ratios* in the phenotype, which is the arithmetic a scale actually wants.

  A fourth reading — squashed to [0, 1], for #99's `sex_allocation` and `selfing_rate` — is named
  here and deliberately **not implemented**, because those genes do not exist yet and a mode nothing
  declares is §8.2's speculative generality.

  Two consequences bind. **A mode is required, never defaulted**: an undeclared gene would be taken
  as signed, and a signed `size` is a body with negative mass. And **the mode is what guarantees
  upkeep is non-negative**, not inheritance — from which follows the rule settled in #136:
  **only a gene read as a magnitude may carry a cost**, checked when `Metabolism` is built. The
  exponential mode is non-negative too and is *permitted* a cost by the same rule, asked as
  `ExpressionMode.always_non_negative` rather than as a list of modes — the property is what #136
  needs, and a list would have to be revisited by every mode added after it.

  The reason is that upkeep is a *sum*, so the damage begins long before any total goes negative. A
  `SIGNED` gene is founded across zero by design, so a positive cost on one contributes a negative
  term: an animal is charged less for pointing its aversion one way round than the other, at equal
  magnitude and therefore equal usefulness. Measured on the assembly fixture as it stood at
  `468457d` — which costed every gene at 0.01, cue genes included — aversion at −1 paid **0.098**
  against **0.418** at +1: a 4.3× discount on total upkeep, silently accepted, with selection free
  to chase it. That is the hard budget running in reverse, and it is the whole
  of the defect; `Ecology.spend` already refuses an outright negative bill (#25), so the extreme
  case was never energy creation but a mid-tick crash naming a module that did nothing wrong.

  Both failures are one misconfiguration, and it is knowable from the two config tables alone —
  which is why it is rejected at construction rather than guarded against per tick (§8.2, §8.7).
  #111 folded the expression mode and the cost onto one `GeneSpec`, so this is now checked where
  both facts first meet — `core.genetics.registry` — rather than by `Metabolism` taking a modes
  mapping it consulted once and never read again.

  **Effort is charged, not just distance.** Fleeing and chasing both cost energy at a premium over
  walking, and hiding costs energy to suppress scent. This is what makes hunger close off options
  rather than merely reading high: a starving animal can neither run nor hide, a predator pays for
  every chase it loses, and prey pay for every escape. Settled in #25, which owned it:

  ```
  cost = size × (transport_cost × distance × (1 + exertion_premium × pace) + climb_cost × gain)
  ```

  **The premium is a per-world-unit multiplier on `pace`, not a flag naming the drive.** Pricing
  distance alone would make a chase merely long; it is the per-unit term that makes it expensive.
  `core.behaviour.movement` therefore knows nothing about what fleeing *is*. A `MovementConfig`
  declares `walking_pace` and `exertion_premium` as one pair: they are the two halves of the
  walk/sprint ratio, and §2.1's warning about constants drifting apart applies to them exactly.

  **Pace is bought with the advantage the chosen option held over standing still** — settled in
  #203, which owned it. `pace` had been a *scalar argument*, and the only caller passed
  `walking_pace`, so `exertion_premium` multiplied a constant: no animal had ever hurried, and the
  whole walk/sprint mechanism was machinery nothing could exercise. The rule that replaced it:

  ```
  urge  = Σ_drives ( appeal_d(chosen) − appeal_d(rest) ) × urgency_d      recorded as `choice_urge`
  pace  = 1 − (1 − walking_pace) × exp(−haste × max(urge, 0))
  ```

  **A drive makes an animal hurry by scoring, not by naming a speed.** That is what the earlier
  wording — "a drive that wants urgency passes a higher number" — turns out to mean once #114
  removed the single winning drive: there is no drive to ask, only a summed utility, so the number
  a drive passes is its own urgency and it arrives through the sum. #24's flight and #179's chase
  are then priced without this module changing, exactly as promised, and without either of them
  declaring a pace.

  **Measured against resting, because the choice is shift-invariant.** A constant added to every
  option changes nothing about which is picked, so an absolute utility carries a component no
  decision depends on; the null option is the one thing every animal always has, which makes it
  the natural zero. **The commitment bonus is excluded**: it decides *which* option wins, not how
  badly it is wanted (#100), and folding it in would make an animal hurry for already being in
  motion and dawdle toward real danger for having settled — the hysteresis band would become an
  accelerator.

  **Saturating rather than linear, and scaled by a gene rather than a constant.** The utility sum
  has no ceiling, so anything linear needs a cutoff, and a cutoff is a second constant to keep in
  step with `walking_pace`; exponential decay of the *remaining* headroom needs neither. The scale
  is `haste` because "how much advantage counts as a lot" is not a fact about the world that could
  be measured once — it depends on the drive weights an animal carries, which are themselves genes
  (#23). Selection therefore calibrates it per lineage. `haste` charges no upkeep for the reason
  §2.5 exempts `mutability`: hurrying already pays the premium on every world unit, so the
  selective consequence is immediate.

  This mechanism is **correct and very nearly inert today**, and that is worth stating rather than
  discovering later: at the founding `choice_temperature` an animal takes its best option 13–27% of
  the time against 11% by chance, so the option it did take is mostly noise and its advantage over
  resting is negative for the median animal. Only 0.5% of decisions in a settled world produce any
  hurry at all. The cause is #205's, not this rule's, and it suppresses #24 and #179 the same way.

  **Velocity is state, and it changes at a bounded rate** — settled in #204, which owned it.
  Nothing had momentum: an animal was repositioned toward a target each tick, so one at full speed
  could reverse for free.

  ```
  desired  = pace × top_speed, pointed at the target (zero for an animal asking to stop)
  velocity ← velocity + clamp(desired − velocity, agility / size)
  ```

  **This is what lets a chase have an outcome.** With free turning a pursuit is not a pursuit —
  the faster animal simply arrives, because there is nothing to out-manoeuvre. Given velocity that
  cannot change instantly, a fast heavy predator overshoots and a nimble prey jinks, so whether the
  two come into contact is an outcome of the physics. **#179 therefore needs no kill rule**: contact
  decides, exactly as contact already decides mating (#20). If that issue ships a resolution
  formula, momentum was missing and the formula is standing in for it.

  **Divided by size, so mass resists a change of direction.** That is free physics rather than an
  authored penalty, and it gives `size` its first downside beyond upkeep — a real speed-against-
  agility axis, which is the difference between a predator and its prey being *different kinds of
  animal* and being two speed values. `agility` must charge a positive cost, per the insulation
  rule above: there is no world in which turning faster is worse.

  **The change is capped in magnitude, never per axis**, or an animal would turn faster along the
  diagonals than along the axes — the grid leaking into the physics, which is the same defect
  §2.5 rejects 8-way movement for.

  **Velocity is written from the displacement that happened, never the one intended.** An animal
  that could only afford half its step carries half the speed into the next tick and one that ran
  into the world edge carries none, so running out of energy and running into a wall each need no
  rule of their own. It is also what bounds velocity by top speed **without a cap**: the new value
  is never longer than the intended one, which lies between the old velocity and `top_speed × pace`,
  so the bound holds by induction from a standing start. §2.5 rejected a speed cap outright and this
  is what replaces it — an argument, asserted by `no_entity_exceeds_its_top_speed` in the invariant
  harness (§6) rather than clamped in a hot loop (§8.2).

  **A target is a bearing reference, not a destination.** A step used to be `min(reach, distance to
  target)`, so an animal stopped dead on its mark; that is not expressible once velocity is state,
  and `Behaviour.chosen_target` never asked for it — the point it returns is `position + look_ahead
  × heading`, a sample of which way is good. A fast animal therefore overshoots it. **Stopping is
  the same rule and not a branch**: an animal handed its own position wants zero velocity and brakes
  at its agility, which is what stops "sprint, then brake free" being a way out of a chase that
  costs nothing.

  Both are **MAJOR** under §2.8, and both were free because nothing had shipped. Measurements —
  the urge and pace distributions, what momentum costs in effective travel, and the throughput of
  the step — are in [`docs/spikes/pace-and-momentum.md`](docs/spikes/pace-and-momentum.md).

  **Only elevation *gain* is charged.** Descent costs its horizontal distance and no more —
  raising a body against gravity is work in a way that lowering it is not. That asymmetry alone is
  what makes a ridge a barrier and a valley a corridor, so §2.6's heightmap becomes the isolation
  mechanism #16 needs with nobody placing a barrier.

  **`gain` is the total climbed along the path, never the difference between its ends** — settled
  in #113, which owned it. A step is priced by walking every cell it crosses and summing each
  crossing's rise; elevation is bilinear, so the profile bends only at grid lines, and those
  crossings are exactly where it can change direction.

  Sampling the two ends instead nets a descent against a climb, which is not a small error in a
  number — it is the barrier disappearing. Impassable ground is expressed *entirely* through what
  it costs, so terrain the cost function never looks at is not impassable at all: a rim-to-rim
  stride over a gorge came out level, and one long step crossed for free what the same ground
  charged for when walked in short ones. The property that pins it is that **path cost is additive
  under subdivision** — one stride costs what the same journey costs in pieces, because it is the
  same journey.

  Two consequences worth keeping. **The iteration count is per tick, not per animal**: one pass
  advances every mover to its own next crossing, so the walk stays vectorized (§2.3) and takes as
  many passes as the *longest* step needs rather than one per animal. And therefore **no speed cap
  is needed** — a lineage that evolves an absurd top speed makes ticks slower, which #46's gates
  say out loud, rather than making the answer quietly wrong (§8.7). Capping was proposed and
  rejected: top speed is a gene under selection, and an authored ceiling on an evolving trait is
  precisely what "author the physics, not the outcomes" forbids.

  The budget is spent *as* the path is walked, which is what makes "where did it run out" a
  well-defined question: an animal with barely enough energy stops at the foot of a wall rather
  than partway up it in proportion to a flat average.

  **The pool gates the step; it does not merely record it.** An animal that cannot pay for the
  whole step travels the fraction it can afford, and an empty one does not move at all. This is
  what turns "a starving animal can neither run nor hide" from a description into a mechanism, and
  it is why hunger closing off options needs no separate rule.

  **Every withdrawal from the pool goes through `Ecology.spend`.** `Ecology` owns the `energy`
  column, so a mover cannot subtract from it directly (§2.3) — it hands over a bill. That is what
  keeps the closed loop auditable as more systems come to spend: #19's chase and #20's gestation
  are the same call again, and the floor at zero lives in exactly one place.

  **Hiding is still open.** Suppressing scent has no cost today because nothing hides —
  `scent_emission` is authored per world and not under selection (see the reserved gene block
  below), so there is no emission for effort to suppress. It arrives with #20's mate-finding
  benefit, which is what gives emission a counterweight in the first place.
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
- **Predation: killing and eating are two acts, and conflating them makes both impossible** —
  settled in #179, which owned it, and it is what finally gave #185's carrion field a mass source.
  Before it, every animal in every world was a grazer: `diet_animal_derived` was inherited and
  expressed, and its animal half bought **nothing**, so an allocation away from plants was pure loss
  and selection could only drive it to zero. Half of #102's encoding was unreachable, and with it
  the whole reason cue space exists — an arms race needs something to run from.

  ```
  damage  = strike_power × (size / prey_size) × animal_share ** p     out of the victim's pool
  carrion = damage                                                    onto the ground it fell on
  meal    = intake_rate × size × animal_share                         grazed off that ground, later
  ```

  **The first build made a strike *be* a mouthful, and the measurement refuted it before the
  reasoning did.** Over 2,000 ticks the flesh allocation fell from 0.50 to **0.04** — carnivory
  selected out of the world entirely, in every world tried. A body is an order of magnitude larger
  than a mouthful (~50 energy units against ~4), so killing took a dozen consecutive ticks of
  contact that two moving animals never have, while the allocation charged its opportunity cost in
  grass from the first tick. The trade could not pay at any sun.

  Killing is bounded by **force** and eating by a **gut**, and nothing makes those one number. The
  gap between them is not destroyed — it is the body lying on the ground, which is exactly the mass
  source #185 was waiting for. So the two issues are one delivery rather than a dependency.

  **A strike feeds the attacker nothing directly.** It eats by *standing on* the carcass and
  grazing it, with the identical contended verb `Plants.graze` uses. Three things then exist that
  nobody implemented: **scavenging**, a **reason to defend a kill** (a second animal on the same
  cell takes a share), and the first real payoff for #100's `commitment` — a lineage too flighty to
  hold a bearing kills and walks away from the meal.

  **Damage follows the frontier, not the raw allocation**, and this is the sharper half of the
  rule. `Feeding` deliberately does not scale a grazing mouthful by diet, because the only victim
  of a wasted mouthful is the eater's own tick. A bite taken from another animal is different in
  kind: the cost falls on a **third party**, so selection cannot correct it. Read linearly, every
  grazer 7% allocated toward meat wounded whichever neighbour it stood beside, and the population
  bled into the carrion field faster than anything could eat it — **35,000 energy units of meat
  standing at 2,000 ticks** while the flesh allocation collapsed to 0.07. Pricing the weapon on
  `share ** p` — the same convex curve #102 already prices the gut on — is what makes a half-hearted
  predator harmless, and it is the general rule: **a trait whose cost lands on someone else must be
  priced convexly, because the budget cannot bound it.**

  **Size divides, so a big animal is hard to kill.** Free physics rather than an authored defence,
  and it gives `size` its first benefit beyond a larger mouthful — until now it only ever charged
  upkeep (#17), locomotion (#25) and turning inertia (#204). A predator/prey axis needs both
  directions to be worth having.

  **Nothing here decides a kill**, exactly as §2.5 settled when momentum shipped. There is no chase
  resolution, no success probability, no trait ratio deciding an outcome: contact decides, getting
  there is a pursuit fought out in `core.behaviour.movement`, and whether a wound is fatal is
  `Ecology.starving` reading an empty pool. The cap at what the victim holds is the kill rule and
  the multi-tick kill at once.

  **Hunger points at whatever the diet is allocated toward**, blending the forage field with the
  smell of meat — live animals through the cue field, carcasses through a diffusion of the carrion
  mass by the plant field's own operator. So one drive steers a grazer to grass and a carnivore to
  prey, and an omnivore genuinely splits its attention, which is the behavioural half of #102's
  frontier. Answering #179's "how is prey located" with the cue field rather than sight (#24) keeps
  this a one-system delivery; the consequence accepted is that **a predator cannot specialise on one
  prey's signature**, because smell is a blend. An *appetite direction* in cue space — the mirror of
  `aversion` — is the richer answer and stays open on #179.

  **Measured, and the result is a negative worth recording** (§8.5).
  [`docs/spikes/predation_viability.py`](docs/spikes/predation_viability.py): the world survives,
  carrion cycles rather than accumulating, and the flesh allocation settles near **0.11–0.29**
  instead of collapsing — but **no world produces a specialist carnivore.** Sweeping `strike_power`
  over 60/200/600 and `decay_rate` over 0.05/0.15 raises the mean allocation (0.16 → 0.33) and
  crushes the population (3,067 → 17); not one cell of that grid held a single animal above 0.7.

  The likely cause is not a coefficient: **one interbreeding population cannot hold two strategies.**
  `p > 1` makes the middle the worst place to be, so a herbivore and a carnivore lineage should
  separate — but every generation recombines them straight back into it. What predation is missing
  is **reproductive isolation**, which is #16. That is the design's own claim arriving as a
  measurement rather than an argument: a trophic pyramid needs speciation, and speciation is
  deferred until a world can be watched. Recorded on #179 rather than tuned around.

- **Thirst is a deficit that raises the cost of living** — settled in #156, which owned it, and it
  is the last of #126's paralysed drives to get a mechanic. `Thirst` had scored since #22 with
  nothing to act on: no hydration column, no way to drink, no consequence for never drinking. Both
  world configs held its weight an order of magnitude below every other drive, and said in a comment
  that this was a workaround rather than an ecology.

  ```
  dehydration ← min(dehydration + loss_rate × (1 + heat_scaling × heat above neutral), 1)
  dehydration ← max(dehydration − drink_rate, 0)          standing at water
  upkeep      ← upkeep × (1 + dehydration_penalty × dehydration)
  ```

  **The column is a deficit, not a level, and that is what makes a newborn safe.** `dehydration` is
  0 for a watered animal, so clearing a reused row to zero means "not thirsty" — the same reset
  every other behavioural column already gets. Stored as a *reserve*, zero would mean born
  completely dry, and every young would die in its first tick unless something remembered to seed
  it: a rule nobody would notice was missing until births existed. It is `exertion`'s shape (#107)
  for `exertion`'s reason, and the two are deliberately identical in kind.

  **It is a fraction, so body size cancels.** A large animal holds more water and loses more, both
  scaling together, so one loss rate means the same thing to a mouse and to an elephant — #107's
  argument again. A capacity in water units would need a second constant relating size to reserve
  and would buy nothing the fraction does not already say.

  **Death by dehydration falls out; there is no dehydration check anywhere.** A dry animal costs
  more to run, so it empties its pool and dies through `Ecology.starving` and `Death`, both
  unchanged. That is the shape §2.5 already settles for senescence, and the reason the repository
  has one answer to "how does a failing body kill its owner" — a `dehydration >= 1 → dead` branch
  would be a second mortality path no invariant covers and every future depletion mechanic would
  copy. It also gets the ecology right: a parched animal is not on a timer, it is **expensive**, so
  it must eat more while it should be drinking, and thirst and hunger compete for the same animal's
  next heading rather than taking turns.

  **Heat moved from wanting to losing.** The old thirst score read ambient temperature directly;
  that term is not deleted but relocated, because heat does not make an animal want water, it makes
  an animal lose water, and wanting follows from having lost. Once it is a rate rather than a score,
  one number — the deficit — is both the urgency and the thing drinking removes.

  **Drinking is a rate, and that is what makes a waterhole dangerous.** A drink spans ticks like a
  meal. With water static (#165 — a lake cannot shrink) there is no depletion to contend over, so
  **time at the water's edge is the only cost a waterhole can charge**, and it is the real one: an
  animal standing still by a lake is an animal a predator can reach (#179). It costs no energy, for
  the reason resting does not: the cost was paid walking there.

  **Finding water is a diffused field, exactly as finding food is**, spread by the same cost-aware
  operator (#93) and gated by the same acuity threshold. Sampling `Water.is_drinkable_at` at the
  candidate headings was rejected in design: candidates sit one `look_ahead` away and lakes are
  sparse, so a binary reading lets thirst steer only when a candidate lands *on* water — #126 again
  in a new place. The field is **built once and never rebuilt**, because water is derived from
  terrain and never advanced; when #165 makes it dynamic this becomes a per-tick system beside
  `rebuild_forage`.

  **The measured consequence is that the workaround is gone.** With the thirst weight founded like
  every other drive — `(0.6, 1.4)` where it sat at `(0.1, 0.3)` — the demo world runs 900 ticks and
  *grows*, 200 founders to 999 and 322 on two seeds, with mean dehydration settling at 0.12–0.22
  and the driest animals reaching 1.0 and dying. `test_a_flat_drives_weight_cannot_change_the_
  decision` records the crossing: thirst was in that test's list of drives whose weight provably
  cannot move a decision, and it is now in the control group beside hunger.

  **MAJOR** under §2.8: it moves what animals do.

- **Foraging perception is a diffused field; foraging *choice* is the gradient of it** — decided in
  #93, which owned it. `core.world.diffusion.CostAwareDiffusion` spreads standing crop over the
  terrain, so every cell holds how much grazing is *reachable* from it, and a forager reads a
  heading straight off the gradient. The field ranks nothing and gates nothing: it answers "which
  way is better from here", and whether that reading is worth acting on belongs to the hunger drive
  (#22).

  **Directions, not targets.** An earlier version of this rule had the field report candidate
  patches and the drive take the argmax of `biomass / (1 + distance / forage_reluctance)`. That is a
  distance discount and nothing else, so a meadow across a gorge scored exactly as well as one on
  open ground the same number of world units away — and terrain is supposed to be what shapes an
  ecology. Diffusing the crop makes the distance discount fall out of the spreading, and making the
  spreading **cost-aware** makes the climb discount the same mechanism rather than a second
  coefficient beside the first. §2.1's warning about constants that must be tuned as a table is the
  argument: two coefficients describing one preference will drift apart.

  So `forage_reluctance` is gone, and the field's `range` is what it became — how far food
  advertises itself; small values keep grazers local and strip ground bare before they move, large
  ones spread pressure out. It costs passes *quadratically*, because diffusion widens with the
  square root of time, so it is a knob with a real price rather than a free preference.

  **A signal routes around a barrier rather than being attenuated through it**, because spreading is
  a walk over the neighbour graph: what arrives behind a wall is whatever came round the end of it.
  That is what will make a fence (#27) an intervention rather than a multiplier.

  **Sight gates by threshold, not by radius.** One field serves the whole population, so a
  per-animal radius is not expressible; acuity instead scales what is sampled, and a reading below
  the drive's `detection_threshold` is nothing found. This is the identical rule §2.5 already
  settles for scent, for the identical reason — sensitivity and range are the same parameter for a
  diffused field — and without it every animal would detect every meadow faintly, leaving sight
  range charged by the metabolic budget while buying nothing but predator avoidance.

  The operator is deliberately **not** plant-specific: `core.ecology.cues` wants exactly it, since
  scent currently diffuses through a mountain unattenuated. Converting that is filed separately,
  because `CueField.sample_excluding_self` subtracts the *exact* diagonal of its blur and that
  factorisation exists only because a separable blur is separable — a per-edge conductance is not.
- **Heritable drive weights.** Behaviour is a fixed set of authored drives (hunger, thirst, fear,
  lust, fatigue) competing each tick by utility score — but *the weights and thresholds are genes*.
  Boldness, sociality, and parental investment therefore evolve rather than being designed.
  Behaviour stays explainable, which the intervention gameplay depends on.
- **Drives score *options*, not animals** — decided in #114, which owned it and which reopened #22's
  abstraction to do so. Each tick an animal considers `N` jittered candidate headings plus a **null
  option**, every drive scores every candidate, and one is sampled:

  ```
  utility(option) = Σ_drives urgency_d × appeal_d(option)
  ```

  A drive therefore answers two questions that #22 fused into one. `urgency` is *how much* — #22's
  score under its proper name, a property of the animal. `appeal` is *which way* — a reading of the
  world at each candidate, on the [0, 1] scale every drive shares because the utilities are summed.

  **The reason is that an argmax over drives makes a drive without a mechanic dangerous.** Under
  #22 the winning drive decided the action, so a winner that had no way to act left the animal
  standing still with nothing saying so: the first assembled world had all forty founders wanting
  water in a world with no way to drink, and not one moved for the entire run (#126). Scoring
  options inverts that. **A drive that perceives nothing returns a flat appeal, which shifts no
  ranking**, so missing mechanics degrade to *indifference* rather than to paralysis — and it
  becomes safe to add a drive before its senses exist.

  Three things fall out rather than being built, which is the argument for the shape:

  - **Rest needs no mode, flag or state column.** It is the null option winning, and an animal that
    picks it proposes no displacement, pays no transport cost, and therefore recovers (#107).
  - **Fear needs no flee-target machinery.** It is appeal with the sign flipped.
  - **Explainability improves rather than surviving.** `breakdown()` reports each drive's
    contribution *to the option actually taken* — "62% of that heading was hunger" — which says
    strictly more than a winner's name, and is only definable once an option is chosen. Two animals
    with identical hunger, one facing a meadow and one facing bare rock, are not equally explained
    by "hunger 0.6". `winning_drive` and `driven_by` were **deleted, not redefined**.

  **The choice is sampled, not maximised**, from a Boltzmann distribution at a per-animal
  temperature (`choice_temperature`, read exponentially — see the mode table above). A cold animal
  takes its best option nearly always and a warm one explores, so exploration is a *trait under
  selection* rather than a tuning constant. It is drawn by the Gumbel-max trick, which is exactly a
  categorical draw from the softmax in one vectorized pass (§2.3) — the same extreme-value machinery
  inheritance already uses, for the same reason: the sampling is the point, not the mean.

  **Headings are continuous and never snapped to grid neighbours**, because 8-way movement gives
  staircase paths and a diagonal bias. Fields are gridded; positions and headings are not. `N` is an
  **engine resolution knob** and deliberately not a gene — it decides how finely the world is
  sampled, not what any animal wants — and the candidates are **jittered per entity**, without which
  every animal evaluates the identical absolute directions and the population moves in lockstep.
  The jitter is what makes angular resolution effectively continuous across a population, and it is
  why a two-stage refinement pass was rejected: refining around the coarse best is a hill-climb, so
  it biases toward the argmax *before* the softmax can blur it, paying twice to partly undo the
  exploration the temperature exists to create.

  **The chosen heading is stored, not returned.** Behavioural momentum needs a choice that persists
  across ticks, so `choice_heading` and `choice_moving` are columns `Behaviour` owns. Two columns
  rather than one because a resting animal *keeps* its heading — the commitment bonus below needs
  something to continue — so the heading alone cannot also say whether it moved. Neither is
  inherited and both are reset on row reuse: a newborn has made no choices. Storing a **heading**
  rather than an option index is what lets the bonus reward how *well* a direction is continued
  (`cos` of the turn) rather than testing an index for equality, and it survives a change to `N`,
  which an index would not — it would silently mean a different direction.

- **What is held across ticks is a bearing, and how hard it is held is a gene** — decided in #100,
  which owned it, over the term #114 had already built as a per-world constant. An option earns a
  bonus in proportion to how well it continues what the animal was already doing, and that bonus is
  the expressed `commitment` gene:

  ```
  utility(option) += commitment × continuation(option, last tick's choice)
  ```

  where `continuation` is `cos` of the turn for a travelling option and 1 for staying put when the
  animal was already stopped — see "Standing still is an incumbent too" below, which is not a
  special case but the same rule applied to the one option that has no heading.

  **"Target" was the wrong noun and is gone.** #100 was written when a drive picked a coordinate and
  held it; the design has since moved to gradients and headings (#93, #114), so what persists is a
  *bearing*. That makes commitment more important rather than less — pure gradient-following dithers
  and oscillates in a way target-seeking does not, and momentum is the standard fix.

  **The bonus is hysteresis, not a weight, and the difference is which thing it attaches to.**
  Because it favours the *incumbent* bearing rather than any particular drive, holding a heading
  requires a challenger to stay under `+commitment` while turning requires one to exceed
  `−commitment` — two thresholds separated by `2 × commitment`, which is exactly hysteresis with the
  band set by the gene. A bonus attached to a fixed drive would not be: that is only a weight
  change. So no second mechanism, no override path, and no explicit "give up" rule is needed; the
  earlier proposal to add one was withdrawn on exactly this ground.

  **A gene rather than a constant, because both failure modes are lethal and neither is the
  designer's to place.** A lineage that dithers re-decides every tick and never gets anywhere; a
  lineage that cannot be interrupted walks into whatever it stopped noticing. Selection is what
  finds the band between them, and it finds a *different* one per environment — which is the whole
  argument of "author the physics, not the outcomes". It costs no upkeep for the same reason
  `mutability` does not: being wrong at either extreme is its own price.

  **Standing still is an incumbent too, and it needs its own term.** `cos` is the dot product of two
  unit headings and rest is the zero vector, so the travelling term hands the null option nothing —
  *by construction, not by decision*. Left there, an animal could never hold a rest across ticks:
  it would settle, shed one tick of exertion, and be back on its feet before recovery amounted to
  anything, so resting would be a duty cycle no lineage could evolve away from. The travelling block
  is therefore shifted down by `commitment` exactly when rest was the last choice:

  ```
  travelling(option) += commitment × (cos(option − heading) − rested)
  staying            += commitment × rested
  ```

  so **getting up costs precisely what stopping costs**, and the two states sit in one symmetric
  band of the same width as the one between opposite bearings. A resting animal keeps its heading,
  so the travelling term still ranks the ways it *could* resume — it prefers the way it was facing,
  one band below staying put.

  **What stops an animal dozing through a predator is selection, not the encoding.** This is where
  #100's second comment was going and it is worth stating plainly, because the opposite was tried
  first: leaving rest un-holdable does close the failure mode, and it closes it by *authoring the
  outcome* — deleting the trait rather than pricing it. A lineage whose commitment is high enough to
  sleep through an approaching threat is eaten; one whose commitment is too low never recovers and
  is run down. The band is therefore expected to evolve **small but non-zero**, and where it settles
  is a reading of how dangerous a given world is — which is a result worth having rather than a
  constant worth choosing. Fear needing to clear the band is the mechanism, not a bug in it.

  **It is read as a magnitude**, so a stored value drifting below zero is a *strongly* committed
  animal rather than one paid to reverse. A signed bonus would reward the option pointing backwards,
  which is a spin rather than a preference, and the world is refused at construction if the gene's
  mode could ever express one (§8.7) — asked as `ExpressionMode.always_non_negative` rather than as
  a list of modes, which is #136's rule reaching its second consumer. Magnitude and not the
  exponential its neighbour `choice_temperature` uses, because zero is a *reachable and meaningful*
  value here — an animal that decides afresh every tick — where a zero temperature is a division by
  zero. That is the distinction the mode table draws, applied.

  The cost is `N × n_drives × n_entities` field reads per tick, and §8.5 required it measured rather
  than asserted: [`docs/spikes/option-sampling-cost.md`](docs/spikes/option-sampling-cost.md) finds
  `choose()` at 362 ms/tick for 100,000 entities at `N=8` against §2.1's 1,000 ms budget, with **one
  additional candidate costing 0.19 ms per 1,000 entities**. So `N` is genuinely affordable to raise,
  which is what the rejection of two-stage refinement depended on. Candidate positions are sampled
  **once per entity and shared by every drive** — the largest option-scaled term at 48.6 ms, which
  per-drive sampling would pay four times for an identical result. The measurement also found the
  worry misaimed: the dominant cost is the forage field's diffusion (110–145 ms, independent of
  population), not the option multiplier.
- **Exertion is a column, and it is work per unit of body size** — decided in #107, which owned it.
  Fatigue scored health deficit alone, so an animal that had sprinted across a ridge and one that
  had stood still all tick were indistinguishable at equal health, and resting was selected for
  only as recovery from injury. `core.behaviour.exertion.Exertion` owns the column; movement hands
  over what a step took exactly as it hands `Ecology` the bill for it.

  **Not the energy bill, and not the energy pool.** The movement bill is `size × (haul + climb)` and
  what accumulates is the parenthesised half, so one saturation constant means the same tiredness to
  a mouse and to an elephant — accumulating the size-scaled bill would leave a large animal
  permanently exhausted by an ordinary walk. Reading the pool instead was rejected outright: hunger
  already scores on exactly that quantity, and two drives reading one number is how a drive contest
  becomes a coin flip.
  Distance between position snapshots was rejected too, because `TickLoop` samples those once per
  `advance()` call rather than once per tick, and §2.4 forbids batching from changing outcomes.

  **Fatigue composes its two terms by noisy-OR**, the same rule §2.5 settles for fear's perception
  channels, and for the same reasons: independent reasons to do one thing should compound, the
  score stays in `[0, 1]` like every other drive, and a third reason to rest can be added later
  without inflating this one past saturation and forcing every other drive's weight to be retuned.
  The repository deliberately has *one* answer to "how do independent urgencies combine".

  **Recovery is geometric and free.** A fixed subtraction would let a hard enough tick outrun
  recovery without bound while an easy one floored at zero, so the same rate would mean "recovers
  quickly" for a walker and "never recovers" for a sprinter; a fraction gives every animal the same
  half-life and can never drive the column negative without a clamp. Resting costs nothing beyond
  the upkeep of existing — charging for it would make rest a third way to starve rather than the
  escape from exertion it exists to be.

  This is what #23 needs in place first: once the fatigue weight is a gene, selection tunes it
  against whatever fatigue reads, and adding the exertion term afterwards would change what that
  gene means for every world already carrying it — a rules fork under §2.8 rather than a fix.

  **A drive's influence over *direction* is how far its appeal varies across the options, not how
  much it wants** — settled in #207, which owned it, and it is the general rule the fatigue fix is
  the first instance of. `utility(option) = Σ urgency × appeal(option)`, so a constant added to
  every option cancels: urgency decides how much a drive *contributes*, and only its **spread**
  can move a ranking. Nothing in the design controlled that quantity or made it visible, and the
  two drives that mattered differed in it by 4.4×.

  Fatigue scored 1 on the null option and 0 on every travelling one, so its spread *equalled its
  entire urgency* by construction — the loudest voice in the contest, spent on one bit of
  information. Hunger reads a field diffused over `range` at candidates `look_ahead` away, ranks
  them **0.995–0.998 correctly by available forage**, and wins by a hair. The measured consequence
  is that hunger never once decided a direction at any choice temperature: the heading actually
  taken tracked fatigue when animals were decisive and noise when they were not.

  ```
  appeal(travelling) = (1 − travel_effort) × exp(−ascent / climb_tolerance)
  appeal(staying)    = 1
  ```

  **Naming the rest/travel gap is the fix**; grading it by terrain is what turns a bit into a
  direction. `travel_effort` is a per-world statement about how much walking takes out of a tired
  animal, where the implicit value was "everything". Ascent is the only thing that *can*
  discriminate — every candidate sits one `look_ahead` away, so distance is identical across the
  options — and it discriminates in the direction that matters: a tired animal prefers the valley
  to the ridge, so where a herd settles reads the terrain rather than an authored bedding ground.
  **Only gain counts**, matching what a step actually costs (#113); fatigue calling a descent
  restful while `Movement` charges it as level ground would be two readings of one physical fact
  drifting apart, which is the shape of defect #112 was.

  The null option is identified by **zero displacement rather than by its column index**, so a
  candidate clipped onto the animal's own position at a world edge reads as the rest it is.

  **This and `choice_temperature` must be tuned as a table** (§2.1), which is the sharper lesson.
  Either measured alone says nothing: the old veto is harmless at the shipped temperature and
  **pathological at a decisive one — 85% of the population permanently at rest** — while grading it
  at the shipped temperature is ecologically neutral. Only the corner where both move shows the
  gain, and #205's negative answer on the temperature was therefore *conditional on this defect*
  and should be revisited now that it is gone. Measurements in
  [`docs/spikes/who-steers.md`](docs/spikes/who-steers.md).

  **What is deliberately not done here** is the general fix: normalising every drive's spread so
  that urgency alone decides influence. It is the right shape and it reshapes #22's contract for
  every drive present and future, so it wants its own decision rather than arriving behind a
  tuning change — and it forecloses a drive that genuinely should be all-or-nothing from saying so.
  Recorded on #207 with the alternatives.

  **MAJOR** under §2.8: it moves what animals do.
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

  **d = 8.** Cue space works like colour: three numbers describe every colour there is, including
  ones never mixed before, and a new colour never needs a fourth channel. Likewise a slot never
  means "wolf" — slot 3 means nothing on its own. Species are *points*, so their number is
  unbounded; the dimension count only decides how many kinds of thing can coexist **without being
  confused**. Eight leaves enough headroom that two unrelated lineages drifting onto the same point
  reads as evolved mimicry rather than as an accident of a cramped space. Widening is additive and
  versioned (§2.3), so this is a floor; narrowing is not possible.

  **Slots are how many smells exist; directions are how many opinions a creature has about one.**
  Each aversion direction produces exactly one number — a dot product against the air — and one
  number answers one question. A creature therefore carries **two aversion directions**, because a
  single one can only point at one region of cue space: aimed between two unrelated threats it also
  fires at everything *between* them, harmless creatures included. Each `(direction × sense)` pair
  is one channel of the noisy-OR above, so this needs no new machinery and #24's sight widens the
  same product.

  **Nothing anywhere lists which species interacts with which.** There is no threat table, no
  predator/prey mask, no compatibility matrix. Every interaction is a number:

  | question | answered by |
  |---|---|
  | Do I fear you? | aversion direction · air |
  | Do I want to mate with you? | *my own signature* · air — free, no genes, and it tracks speciation automatically |
  | **Can** we produce offspring? | genetic distance (#16) |
  | Do I want to eat you? | #19's to decide — it may not use smell at all |
  | Can I digest you once caught? | diet genes, which are not cue genes |

  Note the split on the last two: finding food and being able to use it are different questions, and
  a creature can be drawn to something it cannot digest. The species *expression mask* of §2.3 is
  unrelated to all of this — it governs which genes a species switches on, never who interacts with
  whom.

  **How a lineage comes to fear the right thing.** It does not learn, and nothing registers
  anything. Creatures whose aversion happens to point where a predator's signature sits notice it
  and survive; those pointing elsewhere are eaten. After enough generations the population's
  aversion tracks the predator — and when the predator's signature drifts, the tracking either
  follows or that lineage pays. A newly split species is feared correctly from the instant of the
  split, because both halves inherit the parent's signature.

  **Smell is blunt, and must be.** The air holds a *blend* — a wolf and a rabbit nearby arrive as
  one mixed reading, not two. Only a linear readout composes correctly on a blend, because the
  response to a sum is the sum of the responses; template-matching against a mixture would report a
  third thing that is not there. So a creature fears a *region* of cue space rather than a species,
  which is exactly what makes mimicry free. Sight (#24) perceives individuals rather than a blend
  and is therefore not under this constraint.

  **Founders are evolved, never authored** — decided alongside the above, and it is what makes the
  whole encoding honest. A new world does not begin from hand-written creatures: worlds are
  generated headless from naive founders, run long, and most collapse; the ones that stabilise are
  kept and shipped as starting states (#101). Hand-seeding aversion vectors would mean writing down
  that rabbits fear wolves, which is authoring the outcome rather than the physics. Evolving them
  offline means selection decides before the player ever arrives, by exactly the mechanism that
  keeps deciding during play.

  Two consequences follow. **Snapshots become content, not merely saves** — §3.2 already treats a
  snapshot as the only copy of a world in existence, and now some are also shipped starting
  material. And **the gene vocabulary becomes far more expensive to widen**, because a starting
  state is bound to a vocabulary version, so a migration moves shipped content and not just player
  saves. Settle the vocabulary before generating starting states.

  **An animal does not perceive itself.** Its own deposit is subtracted from what it samples, which
  is exact rather than approximate because a separable normalized blur's diagonal factorizes per
  axis. Without it, any lineage whose aversion overlapped its own signature — every cannibal —
  would be permanently terrified while standing alone in an empty world.

  Reserved gene block, so that one vocabulary migration covers the mechanics that need it rather
  than three:

  | genes | meaning | cost |
  |---|---|---|
  | `signature_0..7` | position in cue space — what I smell and look like | 0 |
  | `aversion0_0..7`, `aversion1_0..7` | two directions in cue space — what frightens me | 0 |
  | `scent_emission` | broadcast strength on the scent modality | see below |
  | `scent_acuity`, `sight_acuity` | detection sensitivity per modality | positive |
  | `camouflage` | damps visual conspicuousness | **positive**, per the insulation rule above |
  | `sex_allocation`, `selfing_rate` | reproduction, continuously (#20) | #20's to set |
  | `maturity_age` | ticks before an animal seeks a mate at all | 0 — late maturity is already paid for in generations forgone |
  | `senescence_resistance` | damps how fast performance traits degrade with age | **positive**, per the insulation rule |
  | `commitment` | how doggedly last tick's *bearing* is held — half the width of the band a rival drive must clear (#100) | 0 — dithering and tunnel vision are each their own price |
  | `choice_temperature` | how much an animal explores rather than taking its best option (#114) | 0 — exploring badly is its own price, exactly as `mutability` above |
  | `mutability` | the floor under an offspring's inherited drift (#104) | 0 — an unfit brood is its own price; see the inheritance rule above |

  **Senescence is degradation, not a timer.** There is no `life_expectancy` gene and no death clock.
  Instead, *performance* traits decay with age — speed, sight and scent acuity — while identity
  traits (cue signature) and capacity traits (size) do not. An old animal is slower, so it catches
  less and escapes less, and it eventually cannot cover its own upkeep. **Death then falls out of
  starvation and predation, mechanisms that already exist**, rather than from an age check; #21
  needs no separate mortality rule for old age, and `Ecology.starving` is already the path.

  A lifespan gene was considered and rejected: living longer is pure benefit, so it runs away and
  every lineage evolves toward immortality. `senescence_resistance` inverts that into something the
  budget already knows how to bound — it *reduces* degradation and therefore must charge positive
  upkeep, exactly as insulation must. Biologically this is right too: bodily maintenance and repair
  are metabolically expensive. The equilibrium is then set by the environment rather than by a
  designer — high-predation worlds favour breed-fast-die-young, safe ones favour the long-lived —
  which is real life-history theory falling out of the energy budget.

  Degradation applies **at expression time**, alongside the species mask in `Genetics.expressed`, so
  it costs no column and no extra pass. Note the consistent consequence: since upkeep is charged
  from the expressed phenotype, an atrophied trait also costs less to maintain.

  **Decay is strictly positive and never reaches zero, and the formula guarantees that rather than
  a clamp doing so:**

  ```
  rate  = base_decay / (1 + senescence_resistance)     base_decay > 0, validated in config
  decay = exp(−rate × age)                             always in (0, 1]
  ```

  The `1 +` denominator is the shape insulation already uses against thermoregulation, for the same
  reason: a gene that only ever *reduces* something must have diminishing returns, or it buys its
  way out of the mechanism entirely — here, into immortality. `senescence_resistance` is read as a
  magnitude so the denominator is never below 1, and the exponential approaches zero without
  reaching it, so an old animal becomes negligibly slow rather than literally motionless and can
  never rejuvenate. Enforcing this in the *calculation* rather than by clamping the gene is
  deliberate: genes drift freely and the clamp is only a numerical backstop (above), so the formula
  has to be what holds the property. It is then asserted in the invariant harness (§6), which is
  exactly what §8.2 asks for when something genuinely cannot occur.

  This makes §2.1's "herbivore lifespan ≈ 1 sim-year" an **outcome to tune the degradation rate
  toward**, not a constant to set. `maturity_age` was a config constant when the drive system was
  built (`LustConfig.maturity_age`) and is properly a gene, so age at first reproduction — one of
  the most strongly selected life-history traits there is — evolves rather than being chosen.

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
- **A fence is a refusal, where terrain is a price** — settled in #27, which owned it, and it is
  the second verb the player has. §2.5 calls isolation the most rewarding intervention in the game,
  and everything it needed was built for it by other issues: the forage and water fields spread over
  a **neighbour graph** (#93), so what arrives behind a barrier is only what came round the end of
  it; `Movement` already visits every cell edge a step crosses (#113); and #26 already records,
  prices and applies an intervention at a tick boundary.

  **A barrier is an edge, not a cell.** A fence stands *between* two cells and does not occupy one.
  As blocked cells it would take ground out of the world — ground that grows nothing, that nothing
  stands on, and that would have to be excluded from the nutrient ledger and from every field's
  normalisation. Two boolean grids hold every edge exactly once (`blocked_north`, `blocked_west`),
  so a cell's south edge *is* its neighbour's north and the two can never disagree.

  **Terrain is expressed entirely through what it costs; a fence is the one thing that is not**, and
  that distinction is deliberate rather than an exception to forget. A ridge is expensive, so an
  animal desperate enough can pay and cross — which is exactly the pressure terrain exists to apply.
  Priced instead of refused, a fence would be a *very steep hill*: a desperate enough animal crosses
  it, enough animals cross eventually, and the isolation the player paid for leaks. Worse, the
  animal that tried would empty its pool against the wire and die there, so a fence would read as a
  killing field rather than a boundary.

  **Two lattices, offset by half a cell, and missing that is what made it leak.** Elevation is
  bilinear between nodes on the integer lattice, so a step's profile bends there and that is where
  the cell walk samples (#113). A *cell* — which is what holds biomass, carrion and a fence — is the
  square centred on a node, so its edges lie on the **half-integer** lattice. Walking only the node
  crossings, a segment steps clean over a fence in its middle. And a step that lands exactly *on* a
  cell boundary indexes to the cell beyond it, so stopping on the line reads as already through.
  Measured before both were fixed: a 39×39 pen lost 3 of 20 penned animals per 100 ticks and gained
  strangers at the same rate, while every direct test of the barrier passed. The walk now visits
  both lattices and stops a thousandth of a cell short of the wire.

  **Measured** (§8.5): over 400 ticks with the population growing from 133 to about 1,200 outside,
  **not one animal crossed in either direction**.

  **What a fence does not stop is scent.** `core.ecology.cues` blurs with a separable box kernel
  rather than the neighbour-graph walk, because `sample_excluding_self` subtracts the exact diagonal
  of that blur and the factorisation only holds for a separable kernel. So a fenced world's animals
  smell each other through the wire, and converting the cue field is #139's. Worth knowing before
  reading a fenced world's fear scores.

  **A fence does not evict.** Whoever is inside when it goes up is inside. That is what makes it an
  isolation rather than a round-up, and it is why *did I fence enough of them, with enough food and
  water?* is the player's question to get wrong.

  **MAJOR** under §2.8: it moves what animals do.
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
- **One length unit, and it is not a physical one** — decided in #112, which owned it. x, y, z,
  elevation, water depth, cell size, speed, sight range and diffusion range are all *world units*.
  Elevation was documented in metres while x and y were world units, which left
  `climb_cost / transport_cost` — the ratio that decides whether a mountain range is a barrier at
  all — resting on a conversion factor nothing declared and nothing checked. It read sensibly,
  never raised, and was always wrong, exactly like the prototype's degree-valued sight angle
  compared against a radian difference (§8.4).

  The general rule, of which this is the first instance: **prefer floating units over grounding.**
  The simulation does not need its lengths to be metres, its masses kilograms, or its energies
  real joules. Grounding a unit invites Earth-calibrated constants that carry no meaning here and
  cannot be tuned freely — `Climate.lapse_rate` defaulting to Earth's tropospheric value was the
  first, and read against world units it cooled a peak by hundredths of a degree, quietly removing
  altitude from climate.

  Lengths were converted first (#112) and **energy followed (#123)**: the metabolic pool, every
  upkeep and locomotion coefficient, and plant biomass are all in *energy units*, the pool's own
  unit with no physical claim. Mass never became a unit at all — `size` is an expressed gene value,
  never kilograms — and it is checked anyway, because grounding energy is only tempting when there
  is a body mass to calibrate it against.

  Two consequences worth stating, because they are what keep it fixed:

  - **An absolute length gets no default; a ratio may.** `TerrainConfig.min_elevation` and
    `max_elevation` are required, because relief only means something against a world extent the
    caller chose — the range they replaced defaulted to a thousand cells' worth of climb and
    nothing noticed. A default that is a *normalisation* (`cell_size = 1.0`, one cell is one world
    unit) or a *reference point* (`sea_level_elevation = 0.0`) is fine; an arbitrary magnitude is
    not.
  - **A units mismatch cannot fail a test on its own**, since both sides are floats, so the check
    is on prose: `tests/test_physical_units.py` fails if anything under `core/` or `clients/` names
    a physical length, energy or mass unit. The length check caught 17 declarations when it was
    written and the energy check 32. It is **one module for the whole rule, not one per dimension**
    — the next dimension to be converted extends the table there rather than adding a third file.
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
- **An intervention is recorded data applied at a tick boundary, never a mutation from outside** —
  settled in #26, which owns the framework. Every part of that sentence is load-bearing:

  - *Recorded*, so a world's history is a list of what the player did rather than a diff nobody can
    read. §3.2 stores it beside the snapshot, and #33's standing policies write into the same list.
  - *Applied at a tick boundary*, so nothing edits a column halfway through a vectorized pass — the
    hazard §2.3 makes capacity growth wait for, and the same answer.
  - *Never a mutation from outside*, so `clients/` and `service/` cannot reach into a store at all.
    Without it the live path and the offline path would be two mechanisms, and §2.4 forbids
    batching from changing outcomes — which it silently would, if a click went one way and a
    caught-up policy went another.

  **Refusals are recorded too**, and not for symmetry: a player who was away for three days needs
  to know their emergency cull did not happen and why. §2.4 makes absence the normal case, so an
  intervention that quietly did nothing is exactly the obituary this section exists to avoid. A
  refusal is never charged, and it is not retried at a later boundary — retrying would fire it at
  some unpredictable moment against a world the player made no decision about.

  **The queue drains at the *start* of a tick**, before the systems. That puts what the player just
  did in front of the same tick's invariant pass (§6), so an intervention that breaks a
  conservation law is caught by the world it broke rather than surfacing later somewhere blameless;
  and it means the tick *reacts* to the intervention rather than leaving it inert until the next
  one.

  **A cull returns the bodies to the ground they fell on**, and no invariant made it. That is worth
  recording because the obvious justification is wrong and was checked: releasing animals holding
  1,950 energy units leaves `total_nutrients()` unchanged either way, because the ledger counts
  their energy as *exported* whether or not it ever comes back. Conservation holds over a world
  where the nutrient is stranded — a gap in the check itself, filed as #214. So the argument is
  ecological: a body decomposes where it falls, a cull is that compressed into one tick, and a cull
  that skipped it would sterilise the ground the player used it on.

  **The catalogue and the currency are still open** (§5). #26 built the framework and one
  intervention; #27's fence and #28's siblings own their own effects, and what *generates* the
  player's budget is a design decision rather than a number — so the ledger spends a balance it is
  given and nothing creates one.

### 2.8 Rule versions

- **Every world stores the runner version it was created under, and always runs under *that
  version's rules*.** Not "old worlds under new logic." A rule change forks the runner; it never
  retroactively rewrites what an existing world is.

  This is not conservatism about file formats. A world's populations *are* an equilibrium reached
  under one particular metabolic cost table, one inheritance rule, one set of drive weights. Swap
  any of those underneath a running world and the equilibrium is no longer an equilibrium —
  populations crash, and the player watches an ecosystem they were stewarding collapse for reasons
  invisible to them and attributable to nothing they did. Since the simulation is non-deterministic
  (§2.2) the world cannot be regenerated, and §3.2 already establishes that the snapshot is the only
  copy in existence. A rules upgrade applied in place is therefore an unrecoverable, unattributable
  loss of weeks of play.

  It also protects content: starting states are evolved rather than authored (§2.5, #101), so a
  shipped starting state is bound to the rule set it was grown under as much as to a gene
  vocabulary version.

- **Version the whole runner, never individual rules.** One version is one coherent rule set. Rules
  interact — inheritance against the metabolic table against drive weights — so per-rule versioning
  would multiply into combinations nobody has ever run and no test covers.

- **A version's behaviour is frozen and pinned by tests** for as long as that version is supported.
  This is the one place in this repository where old code is kept deliberately rather than deleted
  (§8.2), and where a test asserts behaviour that nothing new depends on (§8.1). Both exceptions are
  intentional and neither generalises: they apply to *retired rule sets*, not to speculative
  generality.

- **Most changes do not fork the version.** Only changes to what the world *does*. The runner
  version is `MAJOR.MINOR`:

  - **MAJOR is the rule set**, and a world pins its major forever. Cost tables, inheritance,
    drive scoring, thresholds, anything that moves an outcome.
  - **MINOR is behaviour-neutral improvement** — optimisation, diagnostics, I/O, rendering. A world
    always runs the *newest* minor of its major, so a throughput win or a viewer fix reaches every
    existing world for free. Without this split, versioning would freeze performance work along with
    the rules, which is the opposite of the intent.

- **"Behaviour-neutral" must be demonstrated, not asserted** (§8.5). Non-determinism (§2.2) makes
  "it looks the same" untestable by eye, so there are two tiers:

  1. **Same seed, identical output.** Cheap and sufficient — most refactors pass it outright.
  2. **If that fails**, which vectorisation often causes by changing RNG draw order without changing
     the distribution, the change must show *statistical equivalence* over replicates, using the
     same distribution machinery competitions use (#41, #42). Failing that, it is a major.

  "I am confident this optimisation preserves behaviour" is not evidence.

- **A bug fix that changes outcomes is a major.** Uncomfortable, and correct: worlds grew under the
  buggy behaviour and their equilibria depend on it, so a silent correction collapses them exactly
  as any other rule change would. **The one exception is a fix restoring a violated invariant**,
  which ships as a minor — invariants are not versioned (below), so restoring one is a repair rather
  than a rule change, and a world violating an invariant is already broken.

- **Versions have a lifecycle**, and its stages are not yet settled — at minimum a version is
  current (new worlds get it), then supported (existing worlds keep running), and eventually
  something happens at the end. What that end is — frozen forever, migrated with the player's
  consent, or read-only — is an open question (§5).

- **Invariants are not versioned.** Energy is never created, nutrients are conserved, no entity
  leaves the world bounds (§6). These hold in *every* version, and a version that violates one is a
  bug rather than a variant. The line is: a version may change what the world *does*, never what is
  physically possible in it.

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
- **A snapshot carries state; the config carries the rules** — settled in #31, which owns it.
  `persistence.load` restores into a world that has *already* been assembled from a config, and
  refuses if that config is not the one the snapshot was taken under. §2.8 requires a world to keep
  running under its own rules forever, and this is what makes the alternative impossible rather
  than merely discouraged: a snapshot cannot be opened into a differently-tuned world at all. The
  check is a **fingerprint** — a hash over the config tree — and not a serialisation, which is why
  it can be exhaustive about *rules* while storing none of them.

  Serialising the config is the other half and is not done: until it is, a snapshot is portable
  between processes running the same code and not between versions of it, and the fingerprint is
  what turns that from a silent hazard into a refused load (§7.3 — #31 stays open).

  **Only what cannot be recomputed is stored.** Terrain, water, the cue field and the forage field
  are pure functions of the config or of the stored state, and a stored derivation is one that can
  disagree with what it came from. Terrain becomes state the day terraforming (#152) can edit it,
  and that is a schema bump rather than a silent widening.

  Two pieces are stored that look like implementation detail and are not. **The generator's state**,
  so a reloaded world draws onward instead of replaying the sequence it already used — §2.2 promises
  no determinism, but a repetition compounding with every reload is not what it licenses. And **the
  free list in its own order**, which `row_ids` cannot express: the row a newborn lands in decides
  which element of every vectorized draw it receives, so a re-derived list makes a reloaded world
  diverge from the saved one at its first birth. With both, a save and reload continues *bit for
  bit* — which is a stronger promise than §2.2 requires and a much easier one to test.
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

**A position snapshot is never read without the occupancy that qualifies it** — decided in #119,
which owned it, and it governs every future overlay (#39) rather than the entity dots alone.
`EntityStore.release` clears `alive` and the id mapping but deliberately leaves `x`, `y` and `z`
untouched, since `allocate` overwrites whatever its caller seeds. A snapshot is therefore full
*capacity*, not population, and rendering it whole draws a corpse frozen at its death site, in its
species colour, forever. `TickLoop` consequently snapshots `row_ids` at the same instant as
positions, and the renderer's entry point takes both — the mistake is unrepresentable rather than
merely documented.

**The viewer plots time as well as space** — #39. A position snapshot cannot show a crash, because
a crash is a shape in *time*: `clients/viewer/charts.py` draws the series `metrics.MetricHistory`
records (#30), stacked bottom-left, with population and condition always present and one gene's
mean and spread beside them on a key. Which trait is under selection is the question being asked
and it changes run to run, so it is a keypress rather than a constant.

Three things about it are rules rather than preferences:

- **A chart is handed numbers that already crossed the metric boundary**, and reduces nothing. That
  is what keeps it drawable against a series that arrived over a socket rather than out of the
  process (§3.1, §4). If a client ever has to compute what it draws, the reduction is on the wrong
  side of the boundary.
- **Geometry lives in a collectable module, never in `app.py`**, which is #110's rule applied
  again: `charts.py` decides where a point goes and is tested without a display, and `app.py`
  blits the surface and binds the keys. An inverted axis is otherwise invisible — a chart drawn
  upside down looks entirely plausible.
- **A flat series is drawn centred**, not pinned to an edge. A population holding steady and one
  that has flatlined at zero are both flat, and putting them at different heights would read
  meaning into a constant that does not carry any. The label carries the value; the line carries
  the shape.

**Colour encodes condition, not identity — and the ramp is validated, not chosen** (#39). Every
mechanic this view exists to show is something happening *to* an animal, and until the colour key
existed a herd was a field of identical dots: species colouring says nothing while there is one
species, and there is one species until #16. So an animal's fill is a **deficit** — hunger, thirst,
or age — and every mode darkens toward death. Read the other way up, by the reserve, the animal in
trouble would be the faintest mark on screen, which is backwards for an instrument whose whole job
is spotting trouble.

Three things about how it is drawn are rules:

- **One hue, light to dark, quantised to five steps.** Magnitude takes a sequential ramp and never a
  rainbow. Quantising means every colour that reaches the screen is a step the checks actually saw,
  and that a thousand dots read as bands rather than as mush at four pixels across. Orange rather
  than the conventional blue because the *surface* has already spent blue on standing water.
- **Every dot is cased**, and that is forced rather than decorative. Measured against this viewer's
  own terrain, **no single fill clears 3:1 across it** — the dark ramp steps vanish on dark ground
  (1.02:1) and the light steps on bright ground (1.80:1). A light ring around a darker fill gives
  every mark two tones, so one of them always separates from whatever is behind it. This is why a
  map mark is cased and a chart mark is not: a chart has one surface, a map has every surface.
- **A reference is a fact about the world, never a maximum off the field.** `apply_field_overlay`
  already said so; the carrion layer is what proved it costs something to get wrong. Scaled to a
  whole body — the obvious "1.0 means a corpse" — the brightest cell in a settled world rendered at
  **13% alpha**, so predation ran and the layer looked empty. Scaled to `strike_power`, the most one
  strike can take, the same cell reads 39%. **Rendering the view and looking at it is what caught
  that**, and no test would have: every assertion about that layer passed at either scale.

`FIELD_LAYERS` is a table of `(service, attribute)` rather than a naming convention, because the
fields genuinely live on different services — `getattr(plants, layer)` stopped being true the moment
a layer came from the carrion field.

**Export is a client's job and only a client's.** `core/` may perform no I/O at all (§4) and
`metrics/` deliberately performs none either — it produces serialisable values and stops, because
on the shared machine §3.1 is heading for, the process that records a series is not the process
that writes a file. `clients/viewer/export.py` writes both CSV and JSON: the flat one is what a
spreadsheet and a notebook open without being told anything, and the nested one is the sample as
recorded, which is the same shape that would come off a socket. Per-gene columns are **named**
rather than positional, because a positional column silently means a different gene the first time
the vocabulary is widened, which §2.3 makes an additive and expected event.

**Occupancy is identity, not a flag.** Ids are never reused, so comparing two id snapshots
distinguishes the three cases an `alive` bit cannot: a row still holding the same entity (blend
it), a row now empty (drop it), and **a row freed and handed to a newborn inside the interval**
(draw it where it is now). The third is why this is an id comparison: `alive` reads True at both
ends of that interval and hides the swap entirely, so the newborn would streak in from wherever
its predecessor fell. §2.1 puts death before reproduction *within* a tick precisely so freed rows
are reusable immediately, which makes that the ordinary path once #20 and #21 land, not an edge
case.

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
- **`metrics/` reads `core/`, and never the reverse.** A world that cannot be observed is a smaller
  loss than one that cannot run without an observer, and `core/` staying importable standalone is
  what §3.1's stage 1 rests on. `TickLoop` therefore names its recorder by a structural type
  (`MetricRecorder`) rather than importing one, and `build_world` does not construct it — the
  composition root attaches it afterwards, because a recorder reads a *finished* world.
- **A metric crossing this boundary is a plain value, never an array view.** §3.1 puts the
  simulation on a shared machine with clients asking for view information, so anything returned is
  something a client may hold across a tick and send over a socket: floats, ints, and lists of them.
  Equally, **statistics are computed on the core side, never in a client** — a histogram reduced in
  `clients/` cannot cross a socket, and it puts a pass over the whole gene matrix on the wrong side
  of the boundary. The client draws what it is handed.
- **A series is recorded as the world runs, never recomputed on request.** The simulation is
  non-deterministic (§2.2), so a replay answers a different question than the one asked; and a
  metric that only exists while somebody is watching has a hole in it for every absence, which
  §2.4's offline advancement makes the normal case rather than the exception.
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
- **The end of a rule version's lifecycle** (§2.8). A version is current, then supported; what
  happens after is undecided — frozen and runnable forever, migrated only with the player's
  explicit consent, or read-only. This decides how long old rule sets and their pinning tests must
  be carried, so it is a cost question as much as a design one.
- The concrete intervention catalogue and what each costs, and **what generates the player's
  intervention currency** — settled only in the negative so far: not plain time-ticks. Whatever
  generates it is what the game rewards, so it is a design decision rather than a number. Candidates
  and their consequences are recorded on #26. Any income defined on ecosystem state waits on the
  metric definitions below.
- **The remaining half of the metric definitions.** Species count and Shannon entropy over species
  abundance are still open, and they are deferred *with* #16 — with one species they read 1 and 0
  forever, so there is nothing yet to argue about and nothing to test an argument against. #30 owns
  them and reserves them rather than leaving them for whoever reaches for them first.

  The third — within-species genetic diversity — is settled as far as the numbers to compute it
  *from*: `metrics.Sample.expressed_spread`, the per-gene standard deviation of the expressed
  phenotype. The composition into one scalar is **deliberately not shipped**, and the reason is
  measured: the obvious mean-of-variances reads 17.5 on the demo world, of which almost all is
  `maturity_age` alone, because it is founded over 40–120 ticks while `size` is founded over
  0.8–1.2. That is #193's defect on the genetic *distance* metric, and it wants one scale-free
  composition rather than two — so the scalar waits on #193, and a misleading number was not
  shipped in the meantime (§8.7).
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
  nutrients are conserved across the loop, and senescence decay stays within `(0, 1]` (§2.5).
  **Invariants are never versioned** (§2.8): they hold under every rule set, so they are also the
  line between a fix that ships in place and one that forks the rules.
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

**Coverage is a gate against untested modules, never a target** — decided in #47, which owned it.
Safety here comes from the four kinds above; a line-coverage percentage measures none of them, so
the gate's only job is catching a module that arrived with no tests at all.

That job forces the shape. `tools/check_coverage.py` applies a floor **per module**, because a
repository-wide percentage cannot detect what matters: `core/` holds ~1,600 statements at 98%, so a
new untested 200-statement module pulls the total to about 87% and clears any threshold set with
enough headroom for ordinary refactoring, while one module rotting to nothing is masked indefinitely
by the others improving. A per-module floor cannot be averaged away.

The floor is **70%, measured rather than chosen** (§8.5). Over the same suite, modules with real
tests scored 93–100% and modules that were only ever imported scored 14–59% — `def` and `class`
statements execute at import, so untested code never scores zero once something imports it. 70% sits
in the empty band between the two populations, deliberately far below the weakest tested module so
that refactoring never trips it. **A gate set just under the current figure is the failure mode**,
because it manufactures exactly the pressure §8.1 warns about: tests written to move a number are
tests nobody would miss.

The known blind spot is recorded rather than patched over: `core/ecology/aging.py` scored **89% while
completely untested**, because 8 of its 9 statements are imports and a `def`. No percentage can
separate that from a tested module, so files under 12 statements are reported and not gated. This is
the clearest possible illustration of why coverage carries no correctness weight here.

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

### 7.5 Notions and decisions live in issues, not in conversation

§7.4 is about *problems*. This is about **notions** — design questions, half-formed mechanics,
"should we do X", options weighed and not chosen. They are the larger category and they are the one
that evaporates.

**Any idea raised in discussion becomes an issue before the discussion ends.** A design conversation
produces a dozen notions, two get built, and the rest exist only in a transcript nobody reads again.
They are then rediscovered months later by someone who assumes they are new, or re-litigated because
nothing recorded why they were dropped. Chat is not a record; the issue tracker is.

This applies to ideas that will never be built. An issue saying "we considered this and chose not
to, here is why" costs a minute and saves the same argument being had twice.

**An issue that poses a choice carries a decision gate**, and a gate is four things, not one:

| part | why |
|---|---|
| the question, stated so it can be answered yes or no | "how should water work" is not a gate; "is water derived or integrated state" is |
| the options | including the one nobody likes |
| the consequence of each | what the world does differently |
| **what each forecloses** | the part everyone omits, and the only part that is hard to reconstruct later |

**A recommendation is not a gate.** Recording only the preferred answer hides the alternatives, so a
later reader cannot tell whether the others were rejected or merely unimagined. Give the
recommendation *and* the table.

**Decisions in the negative are recorded with their reason.** "A toroidal world breaks the
boundary-seeded priority flood in `Water`" is what stops the question being asked a third time. A
rejected option with no recorded reason is not a decision, it is a gap that will be filled by
whoever asks next.

**A cluster of related decisions gets a register on its umbrella issue**, ordered by *when each must
be decided* rather than by importance. Some decisions have real deadlines: anything binding a gene
vocabulary or a rule set is free until #101 generates starting states and expensive afterwards,
because from then on it migrates shipped content rather than code (§2.5, §2.8). A decision with a
deadline and no date is a decision that will be made by accident.

**Where a notion lands matters.** Put it on the issue that owns the abstraction (§7.2), not on a new
one — a second issue about the same abstraction is how two designs for one thing get built. File
separately only when the notion is genuinely its own scope.

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
  energy units`. The unit is the world's own, never a physical one (§2.6).
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
