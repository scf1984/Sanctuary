"""Fencing: the intervention that isolates a population (CLAUDE.md §2.5, §2.7, issue #27).

§2.5 calls this the most rewarding intervention in the game, and the reason is that it is the one
the player can use to *cause* the thing the whole design is pointed at: pen a group off, and the two
halves stop mixing. Everything a fence needs was built for it and by other issues —
`CostAwareDiffusion` spreads over a neighbour graph so food and water route round a barrier rather
than through it (#93), `Movement` already visits every cell edge a step crosses (#113), and #26's
framework already records, prices and applies an intervention at a tick boundary.

**A fence encloses a rectangle**, and that is the whole geometry. Not an arbitrary polyline: the
verb this issue exists to provide is *isolate this group*, a rectangle is what a player drags, and a
perimeter is what isolates. An arbitrary line is a richer tool and a different one — it partitions a
world rather than penning part of it — and it belongs to whoever needs it, with the same `Barriers`
underneath (§8.3: the concrete thing first).

**It fences the perimeter, never the interior.** Blocking every edge inside the rectangle would not
pen a herd, it would freeze each animal in its own cell.

**A fence does not evict.** Whoever is inside when it goes up is inside; whoever is outside stays
out. That is what makes it an isolation rather than a round-up, and it is why the interesting
question — *did I fence enough of them, and enough food and water to keep them alive?* — is the
player's to get wrong.
"""

from __future__ import annotations

from core.world.barriers import Barriers


class Fence:
    """Enclose a world-unit rectangle, blocking movement and perception across its perimeter.

    barriers: the world's barrier grids, bound at construction rather than passed, per
        `Intervention`.
    The rectangle is given as two opposite corners in world units and normalised here, so a player
    dragging a box gets the same fence whichever direction they drag it.
    """

    name = "fence"

    def __init__(
        self,
        barriers: Barriers,
        first_x: float,
        first_y: float,
        second_x: float,
        second_y: float,
    ) -> None:
        self.barriers = barriers
        self.min_x = min(first_x, second_x)
        self.max_x = max(first_x, second_x)
        self.min_y = min(first_y, second_y)
        self.max_y = max(first_y, second_y)
        self._blocked = 0

    def cost(self) -> float:
        """One unit per world unit of perimeter, so a bigger pen costs more to build.

        A placeholder rate and openly so, exactly as `Cull.cost` is: what an intervention *should*
        cost is §5's open question and the currency that pays for it is #26's. What matters is that
        the cost scales with the thing being built rather than being flat — a flat price would make
        fencing the whole map the obviously correct opening move.
        """
        return 2.0 * ((self.max_x - self.min_x) + (self.max_y - self.min_y))

    def refusal(self) -> str | None:
        """Refuse a rectangle that cannot enclose anything.

        Checked against the world at the boundary rather than when the player clicked, as every
        refusal is (#26). Nothing here can go stale — terrain does not move — but the shape of the
        check is the one every future intervention copies, and a fence thinner than a cell is a
        real thing a player can draw by mis-clicking.
        """
        cell = self.barriers.terrain.cell_size
        if self.max_x - self.min_x < cell or self.max_y - self.min_y < cell:
            return (
                f"a fence must enclose at least one cell, and "
                f"{self.max_x - self.min_x:.2f} x {self.max_y - self.min_y:.2f} world units is "
                f"thinner than {cell:.2f}"
            )
        return None

    def apply(self, store) -> None:
        """Raise the fence.

        Takes the store to satisfy `Intervention`, and does not touch it — a fence changes the
        *world*, not the animals in it, which is exactly the distinction that makes "a fence does
        not evict" true rather than merely intended.
        """
        self._blocked = self.barriers.enclose(self.min_x, self.min_y, self.max_x, self.max_y)
