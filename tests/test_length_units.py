"""The world has exactly one length unit, and nothing in it is grounded to a physical one (#112).

Elevation was documented in metres while x, y and `cell_size` were in world units, so
`MovementConfig.climb_cost` (per metre) and `transport_cost` (per world unit) were denominated
differently and their *ratio* — the thing that decides whether a mountain range is a barrier at
all (§2.6, #16) — rested on a conversion factor nothing declared and nothing checked. It read
sensibly, never raised, and was always wrong: the same shape of defect as the prototype comparing
a degree-valued sight angle against a radian difference (§8.4).

A units mismatch cannot fail a test on its own — both sides are floats — so the only thing that
keeps this fixed is a check on what the code *says*. Hence a check on prose.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# core/ is the simulation and clients/ reads its fields; between them they hold every length this
# decision governs. Both are project directories with no virtualenv or nested checkout inside
# them, so a plain rglob is exact here and needs none of tools/'s pruning.
SCANNED_ROOTS = ("core", "clients")

# Word boundaries matter: "parameter" and "diameter" contain "meter", and matching those would
# make the check unusable rather than strict.
GROUNDED_LENGTH_UNITS = re.compile(
    r"\b(meters?|metres?|kilometers?|kilometres?|km|centimeters?|centimetres?|cm|mm|miles?|feet|inches)\b",
    re.IGNORECASE,
)


def test_no_length_is_grounded_to_a_physical_unit():
    """No file under core/ or clients/ names a real-world length unit.

    The rule is *prefer floating units over grounding*: the simulation does not need its lengths to
    be metres, and naming one invites Earth-calibrated constants that carry no meaning here. The
    first was real — `Climate.lapse_rate` defaulted to Earth's 6.5 °C/km, which under world units
    cooled a mountain peak by hundredths of a degree and quietly removed altitude from climate
    entirely.
    """
    offenses = [
        f"{path.relative_to(REPO_ROOT).as_posix()}:{number}: {match.group(0)!r} in {line.strip()}"
        for root in SCANNED_ROOTS
        for path in sorted((REPO_ROOT / root).rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        for match in [GROUNDED_LENGTH_UNITS.search(line)]
        if match
    ]

    assert offenses == [], "\n".join(offenses)
