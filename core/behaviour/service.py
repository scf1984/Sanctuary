"""Behaviour domain service: drives scoring *options*, not entities (issues #22, #114, #100).

#22 shipped drives that scored each animal — "how hungry is this one" — and resolved by argmax into
a single winning drive. That had a structural flaw: only a drive with a mechanic behind it could do
anything, so a winner without one left the animal standing still and nothing said so. The first
assembled world had all forty founders wanting water in a world with no way to drink, and not one
of them moved for the entire run (#126).

**Directions, not winners.** Each tick every animal considers a handful of candidate headings plus
the option of staying put, every drive scores every candidate, and the animal samples one. So:

- **A drive with no perception cannot freeze anything.** It contributes urgency and a flat opinion
  about direction, which shifts nothing. Missing mechanics degrade to indifference rather than to
  paralysis, which is what #126 needed and what makes it safe to add a drive before its senses.
- **Fear needs no flee-target machinery** — it is appeal with the sign flipped.
- **Rest needs no mode or state column** — it is the null option winning, and an animal that picks
  it pays no transport cost and therefore recovers (#107).

**Explainability survives and improves.** #22's load-bearing property was that "it fled because fear
outscored hunger" is recoverable from the store. The replacement is a per-drive decomposition of the
*chosen* heading, which says more than a winner's name ever did.

`winning_drive` and `driven_by` are deleted rather than redefined (#114). There is no single winning
drive to report, and their two consumers never needed one: #19's feeding is "an animal standing on
biomass eats" and #20's mating is "two compatible animals in one place may mate". Drives choose
where to go; interactions fall out of where you are.

**What is held across ticks is a bearing, and how hard it is held is a gene** (#100). A candidate
earns a bonus in proportion to how well it continues last tick's heading, and the bonus is the
expressed `commitment` gene rather than a per-world constant. Because it favours the *incumbent*
bearing and not any particular drive, it is hysteresis rather than a weight — a challenger must
clear a band `2 × commitment` wide to turn the animal — and a gene is what puts the band's width
under selection: a lineage that dithers wastes every step re-deciding, and one that cannot be
interrupted walks into whatever it stopped noticing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from core.entities.store import EntityStore
from core.genetics.registry import GeneRegistry, Unit
from core.genetics.service import Genetics
from core.selection import Selection
from core.services import ColumnRegistry, DomainService
from core.world.terrain import Terrain


class DriveRegistrationError(Exception):
    """A drive cannot be registered: its name is taken, or the score block is already full."""


@dataclass(frozen=True)
class BehaviourConfig:
    """Per-world option-sampling parameters — never constants in `core/` (§2.1).

    n_candidates: how many headings each animal evaluates, besides staying put. An **engine
        resolution knob**, the same category as cell size, and deliberately not a gene: it decides
        how finely the world is sampled, not what any animal wants.
    look_ahead: world units along a heading at which that heading is judged — what "toward" means,
        since a drive reads its field there. Per-world rather than per-animal because it is the
        same class of knob as `n_candidates`; making it the animal's own reach would put the speed
        gene, which `Movement` owns, inside this service (§2.3).
    commitment_gene: the gene whose expressed value is the bonus a candidate earns in proportion to
        how well it continues last tick's bearing (#100). Zero makes every tick an independent
        decision, which reads as dithering; large values make an animal dogged. Its expression mode
        must be one that cannot produce a negative phenotype — see `Behaviour.__init__`.
    choice_temperature_gene: the gene whose expressed value is the Boltzmann temperature. Read
        `EXPONENTIAL` so it is strictly positive however far the gene drifts (#111) — a zero
        temperature divides by zero and a negative one inverts every preference the animal has.
    """

    n_candidates: int
    look_ahead: float
    commitment_gene: str
    choice_temperature_gene: str

    def __post_init__(self) -> None:
        if self.n_candidates < 1:
            raise ValueError(f"n_candidates must be at least 1, got {self.n_candidates}")
        if self.look_ahead <= 0.0:
            raise ValueError(f"look_ahead must be positive, got {self.look_ahead}")


class Drive(Protocol):
    """A named appetite: how much each entity wants something, and which way that lies.

    Collaborators — the plant field, the climate, the genetics service — are bound when the drive is
    constructed, not passed per call, following the precedent §6 sets for invariants: one uniform
    signature, with whatever a drive needs closed over rather than a context argument enumerating
    domains that do not all exist yet (§8.2).
    """

    name: str

    def urgency(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32, unit-free: how much this entity wants it, in row order.

        Zero means no pull at all. This is #22's `score` under its proper name: it says *how much*,
        never *where*.
        """
        ...

    def appeal(self, selection: Selection, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """(len(selection), n_options) float32: how good each candidate looks, per entity.

        x, y: (len(selection), n_options) float64 world positions — the point each option would take
            the animal toward, already clipped into the world. Sampled **once per entity** by the
            caller and shared by every drive, because independent sampling per drive multiplies
            field reads by the drive count for nothing (#114).

        Scale is shared with every other drive, since utilities are summed. A drive that perceives
        nothing returns a constant — indifference, which shifts no choice — and must say in its own
        docstring which issue supplies the perception it lacks.
        """
        ...


class Behaviour(DomainService):
    """Owns `drive_scores` and `choice_heading`: what every entity wants, and which way it went.

    Registration order is the column order within the score block. Unlike #22 it is no longer a
    tie-break, because nothing resolves by argmax over drives any more — utilities are summed, and
    a sum does not care in what order it was accumulated.
    """

    owns = ("drive_scores", "choice_heading", "choice_moving")

    # Narrows DomainService.store (typed `object`, the base being store-shape-agnostic) to the
    # concrete EntityStore whose blocks this service fills.
    store: EntityStore

    def __init__(
        self,
        store: EntityStore,
        registry: ColumnRegistry,
        genetics: Genetics,
        genes: GeneRegistry,
        terrain: Terrain,
        config: BehaviourConfig,
    ) -> None:
        super().__init__(store, registry)
        self.genetics = genetics
        self.terrain = terrain
        self.config = config
        self._drives: list[Drive] = []
        # A temperature is a bare scale on a utility, so it carries no dimension of its own; so is
        # a bonus added to one.
        self._temperature_index = genes.index_of(
            config.choice_temperature_gene, unit=Unit.DIMENSIONLESS
        )
        self._commitment_index = genes.index_of(config.commitment_gene, unit=Unit.DIMENSIONLESS)
        # A negative bonus rewards the option that *reverses* last tick's bearing, which is a spin
        # rather than a preference. Genes drift freely across zero (§2.5), so what forbids it is the
        # expression mode and nothing else — refused here, at the one place both facts are in hand,
        # rather than clamped every tick (§8.7). Asked as a property, not as a list of modes, for
        # the reason #136 gives: the next mode added answers on `ExpressionMode` once.
        mode = genes.spec(config.commitment_gene).expression_mode
        if not mode.always_non_negative:
            raise ValueError(
                f"commitment gene '{config.commitment_gene}' is read as {mode.value}, which can "
                "express a negative phenotype; a negative commitment rewards the option that "
                "reverses last tick's bearing"
            )

    @property
    def drive_names(self) -> tuple[str, ...]:
        """Registered drive names, in the column order they occupy in `drive_scores`."""
        return tuple(drive.name for drive in self._drives)

    @property
    def n_options(self) -> int:
        """Candidate headings plus the null option. The null option is always the last column."""
        return self.config.n_candidates + 1

    @property
    def _width(self) -> int:
        return int(self.store.drive_scores.shape[1])

    def register(self, drive: Drive) -> None:
        """Add `drive` to the contest, in the next free column of the score block.

        Raises DriveRegistrationError on a duplicate name — names address drives from the viewer and
        from `breakdown` — or if the block has no column left. The block's width is fixed at store
        construction, so overflowing it is a world-assembly error and belongs at assembly time
        rather than at the first tick (§8.7).
        """
        if drive.name in self.drive_names:
            raise DriveRegistrationError(f"a drive named '{drive.name}' is already registered")
        if len(self._drives) == self._width:
            raise DriveRegistrationError(
                f"the store's drive_scores block holds {self._width} drives and all are taken; "
                f"construct the EntityStore with a wider n_drives to register '{drive.name}'"
            )
        self._drives.append(drive)

    def candidate_headings(self, selection: Selection, rng: np.random.Generator) -> np.ndarray:
        """(len(selection), n_candidates) float64 radians: the directions each entity considers.

        Evenly spaced, then **jittered per entity** by up to one spacing. Without the jitter every
        animal evaluates the identical absolute directions, so a population converges on the same
        few headings and moves in lockstep along them. The jitter costs one uniform draw and makes
        angular resolution effectively continuous across a population — which is also why #114
        rejected a two-stage refinement pass: raising `n_candidates` and relying on this is cheaper
        than a hill-climb that biases toward the argmax before sampling can blur it.
        """
        n = len(selection)
        spacing = 2.0 * np.pi / self.config.n_candidates
        base = np.arange(self.config.n_candidates, dtype=np.float64) * spacing
        jitter = rng.uniform(0.0, spacing, size=n)[:, None]
        return base[None, :] + jitter

    def candidate_positions(
        self, selection: Selection, headings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(n, n_options) world positions: where each option points, null option last.

        Clipped into the world, because a heading near the edge points out of it and the fields a
        drive samples raise outside their bounds. The null option is the animal's own position, so
        "stay put" needs no special case downstream — it is an option proposing no displacement.
        """
        mask = selection.to_mask()
        x = self.store.x[mask].astype(np.float64)[:, None]
        y = self.store.y[mask].astype(np.float64)[:, None]
        reach = self.config.look_ahead
        return (
            np.clip(
                np.concatenate([x + reach * np.cos(headings), x], axis=1),
                0.0,
                self.terrain.world_width,
            ),
            np.clip(
                np.concatenate([y + reach * np.sin(headings), y], axis=1),
                0.0,
                self.terrain.world_height,
            ),
        )

    def utilities(
        self,
        selection: Selection,
        headings: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        commitment: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """The summed utility of every option, and each drive's contribution to it.

        `utility(option) = SUM over drives of urgency * appeal(option)`, which is what makes a
        mildly hungry animal's food preference weigh less than a starving one's without either
        drive knowing the other exists.

        commitment: (len(selection),) how doggedly each animal holds last tick's bearing — the
            expressed `commitment` gene, handed in rather than read here so one phenotype read
            serves both genes this service consults, exactly as `x` and `y` are sampled once and
            shared by every drive (#114).

        Returns the total *and* the per-drive contributions, because the decomposition is the
        explanation the viewer shows (§3.3) and recomputing it later would let an explanation drift
        from the choice it explains.
        """
        n = len(selection)
        total = np.zeros((n, self.n_options), dtype=np.float64)
        contributions: dict[str, np.ndarray] = {}
        for drive in self._drives:
            urgency = np.asarray(drive.urgency(selection), dtype=np.float64)
            if urgency.shape != (n,):
                raise ValueError(
                    f"drive '{drive.name}' reported urgency of shape {urgency.shape} for {n} "
                    f"entities; expected ({n},)"
                )
            appeal = np.asarray(drive.appeal(selection, x, y), dtype=np.float64)
            if appeal.shape != (n, self.n_options):
                # Checked rather than left to NumPy: an (n, 1) return broadcasts cleanly across
                # every option and would silently make the drive indifferent.
                raise ValueError(
                    f"drive '{drive.name}' scored appeal of shape {appeal.shape}; expected "
                    f"({n}, {self.n_options})"
                )
            contribution = urgency[:, None] * appeal
            contributions[drive.name] = contribution
            total += contribution

        # Continuing last tick's bearing is rewarded by *how well* it is continued, not by an
        # equality test: `cos` of the turn angle falls off smoothly, so a slight correction keeps
        # almost all of the bonus and a reversal loses it. An option-index comparison could not
        # express that, which is why the column stores a heading (#114). The null option gets no
        # bonus, since staying put continues no direction.
        #
        # The bonus goes to the *incumbent* bearing rather than to any fixed drive, which is what
        # makes it hysteresis: holding needs a challenger under `+c` and turning needs one over
        # `-c`, so the band is `2c` wide and `commitment` sets it (#100).
        previous = self.store.choice_heading[selection.to_mask()].astype(np.float64)[:, None]
        total[:, : self.config.n_candidates] += commitment[:, None] * np.cos(headings - previous)
        return total, contributions

    def choose(self, selection: Selection, rng: np.random.Generator) -> None:
        """Pick one option per entity and record the decision.

        Returns nothing: the choice is a *fact in the store*, which is what lets movement read it
        one system later without this call handing anything across the gap (§2.1).

        Sampled from a Boltzmann distribution over utilities at each animal's own temperature, so a
        cold animal takes the best option nearly always and a warm one explores. The draw uses the
        Gumbel-max trick — add a Gumbel variate to each scaled utility, take the argmax — which is
        exactly a categorical draw from the softmax and is one vectorized pass rather than a
        per-entity `rng.choice` (§2.3). It is the same extreme-value machinery §2.5 already uses for
        inheritance, for the same reason: the sampling is the point, not the mean.

        Records the chosen heading for the next tick's commitment bonus. An animal that stays put
        keeps its previous heading rather than losing it, so a rested animal resumes the way it was
        going instead of choosing afresh from nothing.
        """
        headings = self.candidate_headings(selection, rng)
        x, y = self.candidate_positions(selection, headings)
        # One phenotype read for both genes this service consults: `expressed` rebuilds the whole
        # (n, n_genes) block per call, so asking it twice in one tick is a block nobody needed.
        expressed = self.genetics.expressed(selection)
        total, contributions = self.utilities(
            selection, headings, x, y, expressed[:, self._commitment_index].astype(np.float64)
        )

        temperature = expressed[:, self._temperature_index]
        scaled = total / temperature.astype(np.float64)[:, None]
        chosen = np.argmax(scaled + rng.gumbel(size=scaled.shape), axis=1)

        rows = np.arange(len(selection))
        stayed = chosen == self.config.n_candidates
        previous = self.store.choice_heading[selection.to_mask()].astype(np.float64)
        # Clipping the index reads a real heading for movers and a discarded one for those who
        # stayed; padding the null column into `headings` would need a heading it does not have.
        walked = headings[rows, np.minimum(chosen, self.config.n_candidates - 1)]
        self.write(
            "choice_heading", selection, np.where(stayed, previous, walked).astype(np.float32)
        )
        self.write("choice_moving", selection, ~stayed)
        self._record(selection, contributions, chosen)

    def _record(
        self, selection: Selection, contributions: dict[str, np.ndarray], chosen: np.ndarray
    ) -> None:
        """Store each drive's contribution *to the option actually taken*.

        Not the bare urgency: two animals with identical hunger, one facing a meadow and one facing
        bare rock, are not equally explained by "hunger 0.6". What the viewer needs is how much of
        *this* decision each drive accounts for, which is only defined once an option is chosen.
        """
        n = len(selection)
        rows = np.arange(n)
        scores = np.zeros((n, self._width), dtype=np.float32)
        for column, drive in enumerate(self._drives):
            scores[:, column] = contributions[drive.name][rows, chosen]
        self.write("drive_scores", selection, scores)

    def scores(self, selection: Selection) -> np.ndarray:
        """(len(selection), n_drive_columns) float32: the raw score block, in ascending row order."""
        return self.store.drive_scores[selection.to_mask()]

    def breakdown(self, selection: Selection) -> dict[str, np.ndarray]:
        """Each drive's contribution to the chosen heading, by name — the "why" behind the move.

        "62% of that heading was hunger, 30% was fear" is strictly more informative than a winner's
        name, and it is what replaces #22's `winning_drive` for the viewer (§3.3). Returned over the
        whole selection rather than one entity, so overlays and click-to-inspect read one call.
        """
        scores = self.scores(selection)
        return {drive.name: scores[:, column] for column, drive in enumerate(self._drives)}

    def headings(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32 radians: the direction each entity last chose to travel."""
        return self.store.choice_heading[selection.to_mask()]

    def chosen_target(self, selection: Selection) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) each (len(selection),) float64: where the stored decision points.

        Recomputed from the stored heading rather than carried out of `choose` in a variable, so
        the decision survives as a *fact in the store* between the two systems that §2.1 keeps
        separate — scoring runs at position 3 in the tick and movement at 4, and fusing them into
        one call would make the order a detail of this module rather than a declared rule.

        An animal that chose to stay is handed its own position, so `Movement.step` prices a step
        of zero and it pays nothing — which is exactly what makes rest recover exertion (#107)
        without anything branching on a resting state.
        """
        mask = selection.to_mask()
        x = self.store.x[mask].astype(np.float64)
        y = self.store.y[mask].astype(np.float64)
        heading = self.store.choice_heading[mask].astype(np.float64)
        moving = self.store.choice_moving[mask]
        reach = np.where(moving, self.config.look_ahead, 0.0)
        return (
            np.clip(x + reach * np.cos(heading), 0.0, self.terrain.world_width),
            np.clip(y + reach * np.sin(heading), 0.0, self.terrain.world_height),
        )
