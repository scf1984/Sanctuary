"""The world's units are its own, and none of them is grounded to a physical one (§2.6).

The rule is *prefer floating units over grounding*: the simulation does not need its lengths to be
metres, its masses kilograms, or its energies real joules. Naming a physical unit invites
Earth-calibrated constants that carry no meaning here and cannot be tuned freely.

Both casualties so far were real, and neither raised:

- **Length (#112).** Elevation was documented in metres while x, y and `cell_size` were in world
  units, so `MovementConfig.climb_cost` (per metre) and `transport_cost` (per world unit) were
  denominated differently and their *ratio* — the thing that decides whether a mountain range is a
  barrier at all (§2.6, #16) — rested on a conversion factor nothing declared and nothing checked.
  Same class of defect as the prototype comparing a degree-valued sight angle against a radian
  difference (§8.4). `Climate.lapse_rate` defaulting to Earth's 6.5 °C/km was the concrete cost:
  correct about Earth, meaningless here, and under world units it cooled a mountain peak by
  hundredths of a degree, quietly removing altitude from climate entirely.
- **Energy (#123).** A joule is a real physical quantity, so a coefficient written down in joules
  invites being checked against reality rather than against this world — a metabolic rate "about
  right for a 3 kg mammal" imports Earth's calibration into a world whose tick is one sim-minute
  and whose animals get ~10² meals per lifetime (§2.1), deliberately unlike reality.

A units mismatch cannot fail a test on its own — both sides are floats — so the only thing that
keeps this fixed is a check on what the code *says*. Hence a check on prose.

One file rather than one per dimension: this is a single rule with several instances, and the next
dimension to be converted should extend the table below rather than add a third module.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# core/ is the simulation and clients/ reads its fields; between them they hold every quantity this
# decision governs. Both are project directories with no virtualenv or nested checkout inside
# them, so a plain rglob is exact here and needs none of tools/'s pruning.
SCANNED_ROOTS = ("core", "clients")

# Word boundaries matter throughout: "parameter" and "diameter" contain "meter", "programs"
# contains "gram", and matching those would make the check unusable rather than strict.
GROUNDED_LENGTH_UNITS = re.compile(
    r"\b(meters?|metres?|kilometers?|kilometres?|km|centimeters?|centimetres?|cm|mm|miles?|feet|inches)\b",
    re.IGNORECASE,
)

# Energy and the mass it would be calibrated against. Deliberately omits the ambiguous
# abbreviations — bare "cal", "g", "N" and "W" are ordinary identifiers in vectorized code, and a
# check that fires on a variable named `n` is a check somebody deletes.
GROUNDED_ENERGY_UNITS = re.compile(
    r"\b(joules?|kilojoules?|kJ|calories?|kcal|watts?|kilowatts?|kW"
    r"|kilograms?|kg|grams?|milligrams?|mg|pounds?|lbs?|ounces?|tonnes?)\b",
    re.IGNORECASE,
)


def _offenses(pattern: re.Pattern[str]) -> list[str]:
    """Every line under the scanned roots that names one of `pattern`'s units, as `path:line: text`."""
    return [
        f"{path.relative_to(REPO_ROOT).as_posix()}:{number}: {match.group(0)!r} in {line.strip()}"
        for root in SCANNED_ROOTS
        for path in sorted((REPO_ROOT / root).rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        for match in [pattern.search(line)]
        if match
    ]


def test_no_length_is_grounded_to_a_physical_unit():
    """No file under core/ or clients/ names a real-world length unit (#112).

    Lengths are *world units*: x, y, z, elevation, water depth, cell size, speed, sight range and
    diffusion range are all denominated the same way, which is what makes `climb_cost /
    transport_cost` a statement about terrain rather than about a missing conversion factor.
    """
    offenses = _offenses(GROUNDED_LENGTH_UNITS)
    assert offenses == [], "\n".join(offenses)


def test_no_energy_is_grounded_to_a_physical_unit():
    """No file under core/ or clients/ names a real-world energy or mass unit (#123).

    Energy is in *energy units* — the metabolic pool's own unit, with no physical claim. Mass is
    covered by the same check because grounding energy is only tempting when there is a body mass to
    calibrate it against; `size` is an expressed gene value, not kilograms.
    """
    offenses = _offenses(GROUNDED_ENERGY_UNITS)
    assert offenses == [], "\n".join(offenses)
