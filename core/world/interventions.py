"""Interventions: what the player does to a world, and what it costs (CLAUDE.md §2.7, issue #26).

**An intervention is recorded data applied at a tick boundary, never a mutation from outside.**
That is the whole of this issue's contract and every part of it is load-bearing:

- *Recorded*, so a world's history is a list of what the player did rather than a diff nobody can
  read. §3.2 stores that history beside the snapshot, and #33's standing policies write into the
  same list — a policy that fired while the player was away must be as visible as a click.
- *Applied at a tick boundary*, so nothing edits a column halfway through a vectorized pass. This
  is the same hazard §2.3 has capacity growth wait for, and the same answer.
- *Never a mutation from outside*, so `clients/` and `service/` cannot reach into a store at all.
  Without this the intervention path and the offline path would be two different mechanisms, and
  §2.4 forbids batching from changing outcomes — which it silently would, if a live click went one
  way and a caught-up policy went another.

**Refusals are recorded too, and that is not symmetry for its own sake.** A player who was away for
three days needs to know that their emergency cull did not happen and why; §2.4 already makes
absence the normal case rather than the exception. An intervention that quietly did nothing is the
obituary that §2.7's whole design exists to avoid.

**What is deliberately not here.** No catalogue and no income. §5 leaves both open: the concrete
interventions belong to #27's fence and #28's siblings, and *what generates the currency* is a
design decision recorded on this issue rather than a number to invent — settled only in the
negative so far (not plain time-ticks). A `grant()` would be a guess at the answer, so the ledger
spends a balance it is given and nothing here creates one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from core.entities.store import EntityStore


class Intervention(Protocol):
    """One thing the player can do, declaring what it costs, when it may happen, and what it does.

    Collaborators are bound when the intervention is *constructed*, not passed to `apply` — the
    precedent §6 sets for invariants and #22 sets for drives. One uniform signature, with whatever
    a particular intervention reaches for closed over, and no world-context argument enumerating
    domains that do not all exist yet (§8.2). A cull holds the ecology it refunds through; a fence
    (#27) will hold the terrain it writes into; neither needs the other's collaborators named in a
    shared parameter list.
    """

    name: str

    def cost(self) -> float:
        """What this costs the player's budget. Charged only if it is actually applied."""
        ...

    def refusal(self) -> Optional[str]:
        """Why this cannot happen right now, or `None` if it can.

        A sentence for the player rather than a code, because the only consumer is a status line
        or a history entry and an enum would need a second table to render it. Checked at the
        boundary rather than when the player asks, since the world moves in between — the animals
        a cull named may have starved before it was due.
        """
        ...

    def apply(self, store: EntityStore) -> None:
        """Do it. Called only after `refusal` returned `None` and the cost was affordable."""
        ...


@dataclass(frozen=True)
class Record:
    """One line of a world's history: what was asked, when, and what happened.

    Every field is a plain value, for the reason `metrics.Sample` is (§3.1): this is the surface a
    dashboard reads and a snapshot stores, and on the shared machine the process that records the
    history is not the one that displays it.

    refusal: `None` where the intervention was applied, and the reason where it was not. Both are
        history — see the module docstring on why a refusal that is not recorded is the failure
        §2.7 exists to avoid.
    cost: what was actually charged, so zero on a refusal. Recording the *quoted* cost on a refusal
        would make the history's costs unsummable, and a ledger that cannot be reconciled against
        its own history is not a ledger.
    """

    tick: int
    name: str
    cost: float
    refusal: Optional[str] = None

    @property
    def applied(self) -> bool:
        return self.refusal is None


class Interventions:
    """The queue, the budget, and the history — one object because they are one accounting.

    Splitting them would let a balance be spent without a matching line, which is precisely the
    reconciliation §3.2 needs to survive a restore.

    balance: what the player has left to spend. Given at construction and never granted here: what
        *generates* it is an open question (§5) and inventing an answer would settle it by accident.
    """

    def __init__(self, balance: float) -> None:
        if balance < 0:
            raise ValueError(f"a budget cannot start negative, got {balance}")
        self.balance = float(balance)
        self.history: list[Record] = []
        self._pending: list[Intervention] = []

    def request(self, intervention: Intervention) -> None:
        """Queue `intervention` for the next tick boundary.

        Deliberately returns nothing and refuses nothing. Whether it *can* happen is a question
        about the world at the moment it runs, and answering it here would be answering it about a
        world that has since moved on — so the only honest answer at request time is "recorded".
        The player learns the outcome from the history, which is the same place they learn what a
        standing policy did while they were away.
        """
        self._pending.append(intervention)

    @property
    def pending(self) -> int:
        """How many requests are waiting for the next boundary."""
        return len(self._pending)

    def apply_pending(self, store: EntityStore, tick: int) -> list[Record]:
        """Apply everything queued, in the order it was asked for, and record each outcome.

        Order is request order rather than cost or priority: the player asked for these in a
        sequence and the second may depend on the first having happened. Anything cleverer would be
        a scheduling policy nobody has asked for (§8.2).

        The queue is drained whether or not each item is applied. A refused intervention is not
        retried at the next boundary — it would then fire at some unpredictable later moment, which
        is worse than not firing: the player would be looking at a world changed by a decision they
        made about a different one.
        """
        queued, self._pending = self._pending, []
        records = [self._resolve(intervention, store, tick) for intervention in queued]
        self.history.extend(records)
        return records

    def _resolve(self, intervention: Intervention, store: EntityStore, tick: int) -> Record:
        """Charge and apply one intervention, or record why not. Never raises for a refusal.

        The budget is checked *after* the intervention's own precondition, so a player who cannot
        afford something is told the interesting reason first — "that would wipe out the species"
        is more useful than "you cannot afford it", and the second is true of everything once the
        budget is empty.
        """
        refusal = intervention.refusal()
        cost = intervention.cost()
        if refusal is None and cost > self.balance:
            refusal = f"costs {cost:g} against a balance of {self.balance:g}"
        if refusal is not None:
            return Record(tick=tick, name=intervention.name, cost=0.0, refusal=refusal)

        self.balance -= cost
        intervention.apply(store)
        return Record(tick=tick, name=intervention.name, cost=cost)
