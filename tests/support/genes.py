"""Gene registries for tests, built from the handful of gene names the suite uses.

Every test that touches genes needs a `GeneRegistry`, and building one inline would mean restating
each gene's expression mode and unit in a dozen files. That is precisely the drift #111 exists to
remove, reintroduced in the test suite — and it would bite the same way, since a test file that
disagreed with `demo_world` about whether `speed` is a length would pass while describing a
different world.

A gene's mode and unit are facts about that gene rather than about any one test, so they live here
once. Cost is not: it is the per-world economy (§2.5), so each test states the costs it needs.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from core.genetics.registry import ExpressionMode, GeneRegistry, GeneSpec, Unit

# Cue space is signed — a signature is a position in it, an aversion a direction through it — and
# everything else the suite names is a quantity that cannot go negative (#104).
_SIGNED_PREFIXES = ("signature_", "aversion")

# Top speed is world units per tick and the tick is unitless, so it is the one length among the
# gene names the suite uses. `Movement` asserts this, which is the point of the unit field (#111).
_LENGTH_GENES = frozenset({"speed"})

# A Boltzmann temperature is read through `exp` so it is strictly positive however far the gene
# drifts (#114); `abs` would fold it at zero, and a zero temperature is a division by zero.
_EXPONENTIAL_GENES = frozenset({"choice_temperature"})

# Diet is an allocation rather than a set of capacities (#102): the genes say how a fixed budget is
# *split*, so they are read on [0, 1] and the shares they imply sum to one by construction.
_UNIT_INTERVAL_PREFIXES = ("diet_",)


def _mode_for(name: str) -> ExpressionMode:
    if name in _EXPONENTIAL_GENES:
        return ExpressionMode.EXPONENTIAL
    if name.startswith(_UNIT_INTERVAL_PREFIXES):
        return ExpressionMode.UNIT_INTERVAL
    if name.startswith(_SIGNED_PREFIXES):
        return ExpressionMode.SIGNED
    return ExpressionMode.MAGNITUDE


def gene_registry(
    names: Iterable[str], costs: Mapping[str, float] | None = None
) -> GeneRegistry:
    """A registry over `names`, costing whatever `costs` says and zero for anything it omits."""
    charged = costs or {}
    return GeneRegistry(
        tuple(
            GeneSpec(
                name=name,
                cost=charged.get(name, 0.0),
                expression_mode=_mode_for(name),
                unit=Unit.LENGTH if name in _LENGTH_GENES else Unit.DIMENSIONLESS,
                description=f"test gene {name}",
            )
            for name in names
        )
    )
