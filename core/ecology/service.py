"""Ecology domain service: the metabolic pool and the upkeep drawn against it (CLAUDE.md §2.5,
issue #17).

Owns the `energy` column on the shared entity store — the single pool every trait charges against.
This module only ever *removes* energy. Income is somebody else's: sunlight into plants (#18) and
transfers through feeding (#19). Keeping every withdrawal here and the income there is what makes
the closed loop of §2.5 auditable at all, and it is why `spend()` is the only method that writes:
metabolic upkeep goes through it, and so does the locomotion bill `core.behaviour.movement` (#25)
hands over, since a service that does not own `energy` cannot subtract from it itself.

Death is not here either. An entity whose pool reaches zero is *starving*, exposed as a Selection
for #21 to turn into carcasses and decomposition. Splitting it this way keeps the two decisions —
"has run out of energy" and "is therefore dead" — owned by the issues that can actually justify
them, rather than having a metabolism module quietly decide mortality.
"""

from __future__ import annotations

import numpy as np

from core.ecology.metabolism import Metabolism
from core.ecology.plants import Plants
from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.selection import Selection
from core.services import ColumnRegistry, DomainService
from core.world.climate import Climate


class Ecology(DomainService):
    """Charges every entity's expressed phenotype against its energy pool, once per tick.

    genetics: consulted for expressed phenotype only — this service never writes a gene. Read
        across the boundary through `Genetics.expressed`, so the species expression mask is
        applied by the service that owns it (CLAUDE.md §2.3).
    climate: the temperature field thermoregulation cost is sampled from, at each entity's own
        position. Queried as a continuous field, never as a zone label (`core.world.climate`).
    metabolism: the per-world cost table (`core.ecology.metabolism`).
    """

    owns = ("energy",)

    # Narrows DomainService.store (typed `object`, since the base is store-shape-agnostic) to the
    # concrete EntityStore whose `energy`, `x`, `y` and `alive` columns this service reads.
    store: EntityStore

    def __init__(
        self,
        store: EntityStore,
        registry: ColumnRegistry,
        genetics: Genetics,
        climate: Climate,
        metabolism: Metabolism,
        plants: Plants,
    ) -> None:
        super().__init__(store, registry)
        self.genetics = genetics
        self.climate = climate
        self.metabolism = metabolism
        self.plants = plants

    def energy(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32, energy units: the current pool, in ascending row order."""
        return self.store.energy[selection.to_mask()]

    def upkeep(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32, energy units per tick: what one tick will cost each entity.

        Exposed separately from `drain` because it is a pure read: the diagnostic viewer's energy
        overlay and any future intervention that asks "what does this animal cost to keep" need
        the number without spending it.
        """
        mask = selection.to_mask()
        temperature = self.climate.temperature_at(self.store.x[mask], self.store.y[mask])
        base = self.metabolism.upkeep(self.genetics.expressed(selection), temperature)
        # A dry animal costs more to run, and that is the whole of how dehydration kills (#156):
        # it empties the pool faster, `starving` reads the empty pool and `Death` frees the row,
        # both unchanged. A `dehydration >= 1 -> dead` branch would be a second mortality path no
        # invariant covers, and every future depletion mechanic would copy it.
        return base * (
            1.0 + self.metabolism.config.dehydration_penalty * self.store.dehydration[mask]
        )

    def drain(self, selection: Selection) -> None:
        """Charge one tick of upkeep to `selection`, flooring the pool at zero.

        `selection` is the caller's choice of who metabolises; pass the live entities. Nothing
        here filters, because a tick loop draining anything other than the living is a bug in the
        loop, not a condition to absorb silently (§8.7).
        """
        self.spend(selection, self.upkeep(selection))

    def spend(self, selection: Selection, cost: np.ndarray) -> None:
        """Charge `cost` energy units to `selection`'s pools, flooring each at zero.

        The floor is what makes the pool a hard budget rather than a debt (CLAUDE.md §2.5): an
        entity can be emptied, never overdrawn, which is the invariant #7 asserts every tick. The
        shortfall is not carried forward — an animal that could not pay is already starving, and
        how badly it failed to pay changes nothing downstream.

        Upkeep is not the only draw on the pool. §2.5's "effort is charged, not just distance"
        makes locomotion a second one (`core.behaviour.movement`, #25), and this service owns
        `energy`, so a mover cannot write the column itself and hands the bill here instead.
        Exposed as a general charge rather than as a `charge_movement` method because nothing
        about a floored subtraction is specific to what the energy went on, and #19's chase and
        #20's gestation are the same operation again.

        Raises ValueError for a negative charge: energy entering the world is #18's sunlight and
        #19's feeding, never a cost with its sign flipped, and §2.5's closed loop has no other
        income (§8.7).

        **Burning energy excretes nutrients, and that is what closes the loop** (#21). Metabolism
        and locomotion are respiration: the energy is gone, but the nutrients it was carried in are
        not, and they land in the cell the animal is standing in. Decomposition alone would not do
        it — over a life an animal eats `H`, assimilates `H×c` and burns `S` of that, so a carcass
        can only ever return `H×c − S` and every unit it ever metabolised would sit on the export
        ledger forever while the field starved.

        It lives here rather than in the upkeep system because *every* draw on the pool is
        respiration — #25's locomotion as much as basal cost, and #20's gestation when it arrives —
        and this is the one place they all pass through. What is returned is what was **actually
        burned**, which is not the bill: the pool floors at zero, so an animal charged more than it
        holds pays what it has, and excreting the bill instead would invent nutrients.
        """
        cost = np.asarray(cost, dtype=np.float32)
        if np.any(cost < 0.0):
            raise ValueError(
                "spend() charges energy and cannot be negative; income belongs to #18 and #19"
            )
        pool = self.energy(selection)
        burned = np.minimum(pool, cost)
        self.write("energy", selection, pool - burned)

        mask = selection.to_mask()
        self.plants.return_nutrients(self.store.x[mask], self.store.y[mask], burned)

    def gain(self, selection: Selection, energy: np.ndarray) -> None:
        """Credit `energy` units to `selection`'s pools — the only thing that adds to them.

        This is the income half §2.5's closed loop needs, and it is deliberately as narrow as
        `spend`: sunlight enters the world in `core.ecology.plants` and reaches an animal only
        here, through a transfer that some other module has already decided the size of. Feeding
        (#19) is the first caller, and it hands over what a mouthful was worth *after* conversion —
        this method does no conversion of its own, because the efficiency that bounds a transfer is
        a property of the eater's gut (`core.ecology.diet`) and not of the pool.

        Raises ValueError for a negative credit, mirroring `spend`. An income with its sign flipped
        is a withdrawal that no cost table accounts for, so the loop would leak with nothing to
        flag it (§8.7).

        There is no ceiling. A full animal is one that stops eating — which is hunger's business
        (#22) and feeding's, not the pool's — and capping the column here would silently destroy
        energy that the nutrient ledger has already recorded as leaving the field.
        """
        energy = np.asarray(energy, dtype=np.float32)
        if np.any(energy < 0.0):
            raise ValueError(
                "gain() credits energy and cannot be negative; a withdrawal belongs in spend()"
            )
        self.write("energy", selection, self.energy(selection) + energy)

    def transfer(
        self, donors: Selection, recipients: Selection, energy: np.ndarray
    ) -> None:
        """Move `energy` from each donor to the recipient in the same position, burning none of it.

        Paired positionally, row i of `donors` against row i of `recipients`.

        This is the third thing that can happen to the pool and it is neither of the other two.
        `spend` destroys energy and excretes the nutrients it was carried in (#21); `gain` is income
        from outside. A transfer does neither: the energy stays inside the population and only
        changes owner, which is what gestation is (#20). Routing a gestation cost through `spend`
        would return the nutrients to the soil *and* hand the energy to the offspring — inventing
        it — which is the kind of double entry §2.5's closed loop cannot survive.

        Raises if a donor cannot afford its share. Flooring at zero would credit a recipient more
        than the donor gave up, and energy created is a §6 invariant rather than a preference. The
        caller is the one that knows who can afford to breed, so it does the filtering (§8.7).
        """
        if len(donors) != len(recipients):
            raise ValueError(
                f"donor and recipient selections must have equal length: "
                f"{len(donors)} vs {len(recipients)}"
            )
        energy = np.asarray(energy, dtype=np.float32)
        if np.any(energy < 0.0):
            raise ValueError("transfer() moves energy and cannot be negative; see spend()")

        available = self.energy(donors)
        if np.any(energy > available):
            raise ValueError(
                "cannot transfer more energy than a donor holds; filter to those that can afford "
                "it before calling, because flooring here would create energy (§6)"
            )

        self.write("energy", donors, available - energy)
        self.write("energy", recipients, self.energy(recipients) + energy)

    def kill(self, rows: np.ndarray, damage: np.ndarray) -> np.ndarray:
        """Take `damage` out of each row's pool without crediting it anywhere; return what was taken.

        Addressed by **row index** rather than by `Selection`, because predation pairs an attacker
        with a victim by *position* and a mask carries no order — the defect #191 hit on
        `Genetics.inherit` and fixed the same way.

        This is the fourth thing that can happen to the pool, and it is none of the other three.
        `spend` destroys energy and excretes the nutrients it was carried in (#21); `gain` is income
        from outside; `transfer` moves energy between pools losslessly (#20). A kill *removes* it
        from the animal without destroying it and without handing it to anybody — the flesh is
        still there, it is simply no longer alive. `core.ecology.predation` hands the same quantity
        to the carrion field in the same tick, which is what makes that true rather than a leak.

        **Nothing is excreted here**, deliberately, and that is what distinguishes this from
        `spend`. What a body loses to a wound is not respired: it lies on the ground as meat, still
        outstanding on `Plants.exported_nutrients` exactly as it was while alive, and it is
        `Carrion.decompose` that finally pays it back. Excreting it here *and* depositing it would
        return the same nutrients twice, which is the double entry §2.5's closed loop cannot
        survive (§6).

        Returns `(n,) float32, energy units` — what each row actually lost, capped at what it held,
        because a body cannot yield more than it is. **That cap is the kill rule and the multi-tick
        kill at once**: a strike larger than the victim's whole pool simply empties it, and a
        smaller one leaves a wounded animal to be finished later. Nothing decides "does it die" —
        `Ecology.starving` reads an empty pool and `Death` frees the row, both already in the tick.
        """
        damage = np.asarray(damage, dtype=np.float32)
        if np.any(damage < 0.0):
            raise ValueError(
                "kill() removes energy and cannot be negative; a wound that heals is not this"
            )
        taken = np.minimum(self.store.energy[rows], damage).astype(np.float32)
        self.write_at("energy", rows, self.store.energy[rows] - taken)
        return taken

    def starving(self, selection: Selection) -> Selection:
        """The entities in `selection` whose pool has run out — energy at or below zero.

        Restricted to live rows: a released row's `energy` still reads whatever it held, and #21
        will turn this selection into deaths, so a free row appearing in it would "kill" an entity
        that does not exist.
        """
        return Selection.from_mask(
            selection.to_mask() & self.store.alive & (self.store.energy <= 0.0)
        )
