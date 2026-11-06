# Thermal tolerance: does hot and cold ground evolve different animals?

Issue #176, which replaced the one symmetric `insulation` gene with `heat_tolerance` and
`cold_tolerance` and made thermoregulation quadratic in the deviation from neutral. Measured on
`feat/176-thermal-tolerance`, Windows, Python 3.13.5, NumPy 2.5.4. The bench scripts are
`docs/spikes/thermal_divergence.py` and `docs/spikes/thermal_viability.py` (living count and mean
thermal genes every 500 ticks; run on the parent commit for the "before" column).

**The headline result: the direction is right in every seed, and the size is small.** The price
level is settled by construction and pinned by tests. The evolutionary response on the demo
geography is consistent in sign but too small to lock in as a statistical test.

## 1. What is settled by construction

These are unit-level facts, tested in `tests/core/ecology/test_metabolism.py` and
`tests/core/ecology/test_selection_pressure.py`:

- **The cheapest build differs by side.** Swept over a 61 × 61 grid of (heat, cold) tolerance at
  5 and 35 around a neutral of 20, the cheapest build at 35 carries heat tolerance and no
  cold tolerance, and the cheapest at 5 is its exact mirror. Under `insulation` the two curves
  were identical to the last digit (#176's evidence).
- **Survival crosses over.** A heat-tolerant cohort outlasts a cold-tolerant one 20 temperature units above
  neutral and loses to it 20 below, in all four seeds. Neither build wins everywhere.

## 2. The demo world stays viable

The demo map sits entirely on the hot side of its neutral temperature (26.2-29.4 against 20, in temperature units),
with a mean cell deviation of 8.0. The quadratic rate was re-derived from the linear one as
`0.01 / 8.0 = 0.00125`, so the typical animal's bill is unchanged and only the curve's shape moved.

`build_demo_world`, 200 founders, 4,000 ticks:

| seed | living at 4,000, before (`insulation`, linear) | living at 4,000, after |
|---:|---:|---:|
| 1 | 3,205 | 3,933 |
| 2 | 3,714 | 3,962 |
| 3 | 3,823 | 4,154 |
| 4 | 3,657 | 4,452 |
| 5 | 3,868 | 3,518 |
| **mean** | **3,653** | **4,004** |

Every seed grows from 200 into the thousands both before and after. The after column is about 10%
higher on average, which is within what four seeds of a non-deterministic world can tell apart
from noise (§2.2), so this says *still viable*, not *more viable*.

On this all-hot map, heat tolerance rose in 4 of 5 seeds (e.g. 0.250 → 0.322) and cold tolerance,
which does nothing here and costs 0.01 per unit, fell in 3 of 5 (e.g. 0.272 → 0.191 in seed 5).
That is the expected sign, slowly.

## 3. Divergence across a split map

The demo map has no cold ground, so nothing can diverge on it. `thermal_divergence.py` builds the
demo world with two numbers changed:

- `neutral_temperature` moves to the map's **median** cell temperature (about 28.0), so about half
  the land is on each side;
- `thermoregulation_rate` is set so the **mean bill over the land is the demo world's 0.08 per
  tick**. The pressure is the same; only its sign now varies in space.

Nothing else is tuned. Six seeds, 200 founders, 4,000 ticks; "hot" means an animal's own cell is
above neutral. `corr` is the correlation, over living animals, between local deviation from
neutral and `heat_tolerance − cold_tolerance`:

| seed | n | heat tol. hot / cold | cold tol. hot / cold | corr at 0 | corr at 4,000 |
|---:|---:|---|---|---:|---:|
| 1 | 3,747 | 0.234 / 0.220 | 0.243 / 0.249 | +0.126 | +0.120 |
| 2 | 4,211 | 0.261 / 0.255 | 0.225 / 0.232 | −0.049 | +0.059 |
| 3 | 3,812 | 0.271 / 0.259 | 0.301 / 0.310 | +0.065 | +0.083 |
| 4 | 4,608 | 0.244 / 0.238 | 0.304 / 0.309 | +0.036 | +0.043 |
| 5 | 4,083 | 0.290 / 0.285 | 0.278 / 0.329 | +0.034 | +0.156 |
| 6 | 3,172 | 0.359 / 0.349 | 0.364 / 0.356 | +0.027 | +0.049 |

- Heat tolerance is higher on hot ground in **6 of 6** seeds.
- Cold tolerance is higher on cold ground in **5 of 6**.
- The correlation at tick 4,000 is positive in **6 of 6**. Each is at least 2.7 standard errors from
  zero at these n, though offspring settling near their parents makes samples non-independent, so
  that understates the error.

But **the effect is about 0.01 in gene value against means near 0.25**, and the founders' chance
correlation at tick 0 was already +0.03 to +0.13 in five seeds. The change from tick 0 to tick
4,000 is large in two seeds (+0.11, +0.12) and near zero in the rest.

### Why so small, and why it was not tuned

The split map's whole temperature range is about 3 degrees, so the deviation either side is at
most about 1.6. Animals range across that in far fewer ticks than selection needs to act, so the
hot and cold sub-populations keep mixing. That is gene flow swamping a weak, fine-grained gradient,
which is real population genetics rather than a defect. Per this issue's instruction the
coefficients were **not** tuned until divergence appeared: the result above is what the demo
geography gives with the demo world's own thermal pressure.

## What this means for #16 and #159

- Climate now **can** contribute to ecological isolation, where before it structurally could not.
- Whether it contributes **enough** depends on how far apart in temperature isolated populations
  live, and on how long they are isolated. The demo map's 3-degree range is too mild to show it in
  4,000 ticks. A world with real climate zones, mountain barriers (§2.6) or seasons (#159) is where
  this should be re-measured, and #16's speciation work is where "enough" gets defined.
- No statistical test locks in the evolutionary result, because it is not robust at this scale
  (§8.1 explore-then-lock-in only locks in what is robust). The price-level contract in section 1
  is locked in.

## 4. Re-measured with seasons (#159)

The demo world now has seasons and weather: a 720-tick year (six founding generations),
`amplitude_fraction = 0.5`, and weather of 1.0 temperature units per spell, ~30 ticks long, about
five systems across the map. `thermal_divergence.py` gained a mode argument. `seasonal` runs the
demo world's own climate, and `static` runs the same world with `seasons=None`. Both classify
"hot ground" by the **static** field, so the two modes sort animals identically and differ only in
what the animals live through. Same split-map setup as section 3, same six seeds, 4,000 ticks,
measured on `feat/159-derived-weather`. Both columns were run on that branch, so they share
everything merged since #176 (senescence #239/#244, float64 energy #240, temperature units #238).
The static column therefore differs from section 3's table, which predates those changes.

Gap = (mean on hot ground) − (mean on cold ground), so a positive heat gap and a negative cold gap
are the expected signs. Δcorr = `corr` at 4,000 minus at tick 0.

| seed | static: heat gap | static: cold gap | static: Δcorr | seasonal: heat gap | seasonal: cold gap | seasonal: Δcorr |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.010 | −0.003 | +0.095 | +0.028 | −0.084 | +0.119 |
| 2 | +0.011 | −0.017 | +0.067 | +0.060 | −0.039 | +0.204 |
| 3 | +0.028 | −0.086 | +0.126 | +0.043 | −0.023 | +0.101 |
| 4 | +0.005 | −0.043 | +0.139 | +0.041 | −0.017 | +0.126 |
| 5 | −0.004 | −0.010 | +0.112 | +0.045 | −0.015 | +0.170 |
| 6 | +0.016 | −0.029 | +0.140 | +0.021 | +0.004 | +0.139 |
| **mean** | **+0.011** | **−0.031** | **+0.113** | **+0.040** | **−0.029** | **+0.143** |

What seasons changed, and what they did not:

- **Heat tolerance diverges about four times as much**: a mean gap of +0.040 against +0.011,
  in the expected direction in 6 of 6 seeds (5 of 6 static). The two ranges barely overlap (static
  −0.004 to +0.028, seasonal +0.021 to +0.060).
- **Cold tolerance diverges no more than before**: −0.029 against −0.031, expected sign in 5 of 6
  (6 of 6 static).
- **The overall correlation grows a little more** (+0.143 against +0.113), and is positive in every
  seed in both modes.
- **The larger effect is not divergence at all: both tolerances rise everywhere.** Population means
  at tick 4,000 sit around 0.3–0.5 under seasons against 0.15–0.4 static. A climate that varies in
  time selects for a wider comfortable band in every lineage, which is the generalist response
  #176's discussion predicted ("in a world where temperature varies in *time*, … insulation regains
  its value").

So seasons make climate a stronger axis on the heat side, and leave it a weak one overall. The
reason is the same as in section 3, and seasons cannot fix it: the demo map's latitude span is about
2 temperature units, the seasonal swing is a fraction of that by construction (#159's inversion
guard), and animals cross the whole map faster than selection acts. Still **not locked in** by a
statistical test: six seeds whose ranges nearly touch are a direction, not a robust result. The
place this becomes decisive is a world with a wider thermal span or barriers between zones, which
is #16's to define.

Viability with seasons, `thermal_viability.py` (seasonal, the demo world as shipped) against the
same world with `seasons=None`, 200 founders, 4,000 ticks:

| seed | living, static | living, seasonal |
|---:|---:|---:|
| 1 | 3,117 | 2,983 |
| 2 | 3,181 | 2,488 |
| 3 | 3,722 | 3,201 |
| 4 | 3,448 | 3,414 |
| 5 | 3,757 | 3,751 |
| **mean** | **3,445** | **3,167** |

Every seed grows from 200 into the thousands. Seasons cost about 8% of the standing population on
average, which is what a winter is for, and none of the five comes close to collapse.
