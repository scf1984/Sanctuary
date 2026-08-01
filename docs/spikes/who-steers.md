# Who steers: what decisiveness buys, and why hunger never decides

Issue #205, which asked whether founders start too **indecisive**. The answer is no, and the
measurements that say so found something else instead.

Everything below is `docs/spikes/who_steers.py` on `711b105`, over the demo world, 200 founders,
400 ticks, seeds 1–3.

---

## 1. The question, and the wrong answer

`Behaviour.choose` samples from `exp(utility / temperature)`. Founded at `exp(0) = 1`, animals took
their best option 13–27% of the time against 11% by chance, and #205 recommended tightening the
founding range so the drives would steer.

**That recommendation was wrong.** Founding every animal at a fixed temperature and running the
world:

| temperature | takes its best option | forage rank of the heading taken | living | median energy |
|---:|---:|---:|---:|---:|
| 0.08 | 0.94 | **0.394** | 1,426 | 48.7 |
| 0.14 | 0.91 | **0.395** | 1,523 | 51.8 |
| 0.22 | 0.86 | 0.432 | 1,876 | 59.3 |
| 0.37 | 0.68 | 0.505 | 2,424 | 61.1 |
| 0.61 | 0.42 | **0.523** | 2,553 | 60.8 |
| 1.00 *(today)* | 0.26 | 0.521 | 2,289 | 60.8 |

**Forage rank** is where the chosen heading sits, by the forage field, among the candidates that
animal considered. Chance is 0.5.

A decisive population forages *worse than random* and carries a third fewer animals in worse
condition. Every reading moves the same way, and today's setting is within noise of the best one.

> An earlier metric — standing crop under the population over the field mean — was tried and
> discarded. It is confounded: animals eat what they stand on, so a *better* forager depresses its
> own numerator. The rank measure is taken over one decision's own option set and is immune to that.

---

## 2. What is actually wrong

Scoring the same decision per drive — where would each drive, **on its own**, have sent the animal?

| temperature | chosen | hunger | thirst | fear | lust | fatigue |
|---:|---:|---:|---:|---:|---:|---:|
| 0.08 | 0.416 | **0.998** | 0.521 | 0.521 | 0.511 | 0.415 |
| 0.08 | 0.405 | **0.995** | 0.488 | 0.488 | 0.476 | 0.400 |
| 1.00 | 0.520 | **0.996** | 0.459 | 0.459 | 0.458 | 0.504 |
| 1.00 | 0.528 | **0.998** | 0.500 | 0.500 | 0.499 | 0.492 |

**Hunger scores 0.995–0.998.** It knows which way the food is, essentially perfectly, at every
temperature. #93's cost-aware diffused field and the acuity-gated read of it work exactly as
designed.

And it never decides anything. At temperature 0.08 the chosen heading's rank (0.416) tracks
**fatigue's** (0.400–0.415), not hunger's. At temperature 1.00 it sits at chance, because the draw
is noise. **Hunger has never once been the drive that picked the direction.**

Fatigue's preference is the null option — stay put — and an animal's own cell is systematically
poor forage because it has been eating there. So a decisive animal reliably rests on ground it has
already stripped, and that is the whole of the below-chance result.

*(The first hypothesis was lust pulling animals into crowds, where the grass is gone. It is wrong:
lust ranks 0.46–0.51, indistinguishable from chance. Recorded so it is not re-proposed.)*

---

## 3. The mechanism: influence is spread, not urgency

`utility(option) = Σ urgency_d × appeal_d(option)`. A drive's urgency decides how much it
*contributes*; only how far its appeal **varies across options** can move a ranking. Measured on the
shipped config:

| drive | mean urgency | spread over options |
|---|---:|---:|
| hunger | 0.665 | 0.210 |
| thirst | 0.037 | 0.000 |
| fear | 0.208 | 0.000 |
| lust | 0.006 | 0.002 |
| **fatigue** | **0.921** | **0.921** |
| commitment (the hysteresis band, `2 × gene`) | — | 0.292 |
| **total** | — | **1.189** |

Fatigue's spread is **4.4× hunger's**, and it is the largest single term in the sum.

The reason is structural, not a mis-set weight:

- **Fatigue is all-or-nothing about direction.** Its appeal is its full urgency on the null option
  and zero on every travelling one, so its spread *equals* its urgency by construction.
- **Hunger is nearly flat about direction.** The forage field is diffused over `range = 4.0` and
  candidates sit at `look_ahead = 4.0`, so neighbouring candidates read a smoothed field at similar
  points. Hunger ranks them correctly — 0.998 — and the margin it wins by is small.

So hunger is a perfectly reliable compass that whispers, next to a drive that shouts one bit.
Lowering the temperature amplifies both equally and therefore changes nothing about who wins; it
only removes the noise that was letting hunger through by accident.

Thirst and fear spread exactly 0.000, which is correct and documented — both are flat until #24 and
a drinking mechanic exist. They contribute urgency and no direction, exactly as §2.5 intends.

---

## 4. What this closes and what it opens

**#205's gate is answered in the negative.** The founding `choice_temperature` range stays as it
is. Lowering it degrades foraging, condition and population on every seed measured; the statistic
that prompted the issue is real, and it is a symptom.

**The real defect is filed separately** — a drive's influence on direction is its across-option
spread, and nothing in the design controls that or even makes it visible. It is the reason `haste`
(#203) fires on 0.5% of decisions, and it will silence #24's flight and #179's chase the same way.

**One property was locked in on the way past.** `tests/core/ecology/test_foraging_finds_food.py`
asserts hunger's preferred heading beats 0.9 and a flat drive's does not. Nothing tested that
end to end before, and nothing else in the suite would notice if it broke: a world of animals
foraging at random still eats, breeds and stabilises, so every population figure would stay
plausible.

---

## 5. The fix, and the interaction it exposed (#207)

Fatigue now grades travelling options instead of vetoing them:

```
appeal(travelling) = (1 − travel_effort) × exp(−ascent / climb_tolerance)
appeal(staying)    = 1
```

Shipped at `travel_effort = 0.25`, `climb_tolerance = 4.0`. Both measured rather than chosen. The
ascent between an animal and one of its candidates runs p50 0.33, p99 1.46, max 1.87 world units
over a settled world, so a tolerance of 4.0 is roughly twice the largest rise — the discount is
gentle and graded across the observed range instead of saturating on ordinary ground.

### Neither knob measured alone says anything

This is §2.1's "tune as a table" as a concrete result. Sweeping `travel_effort` at the shipped
temperature moves **nothing**: forage rank sits at 0.496–0.520 across the whole range from 0.1 to
1.0, indistinguishable from the 0.521 baseline, because at that temperature the Gumbel noise is
larger than any drive's spread. And #205 already measured that sharpening the temperature alone
makes everything worse.

The pathology only appears in the corner where both move:

| travel_effort | climb_tolerance | temperature | living | energy | forage rank | resting | fatigue spread | hunger spread |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 *(the veto)* | — | 1.00 | 2,187 / 2,372 | 60.4 / 61.1 | 0.523 / 0.517 | 0.26 | **0.93 / 0.90** | 0.20 / 0.21 |
| 1.00 *(the veto)* | — | 0.22 | 1,856 / 1,873 | 59.5 / 59.7 | **0.454 / 0.419** | **0.85** | 0.87 / 0.85 | 0.18 / 0.20 |
| 0.25 | 4.0 | 1.00 | 2,124 / 2,330 | 60.4 / 61.4 | 0.528 / 0.512 | 0.14 | **0.34** | 0.20 |
| 0.25 | 4.0 | 0.22 | **2,728 / 2,863** | 59.4 / 60.8 | 0.533 / 0.539 | 0.40 | 0.34 | 0.15 |
| 0.10 | 4.0 | 0.22 | 2,468 / 2,739 | 59.1 / 60.0 | **0.554 / 0.547** | 0.25 | **0.22** | 0.17 |
| 0.25 | 4.0 | 0.08 | 1,865 / 1,993 | 58.4 / 58.7 | 0.475 / 0.445 | **0.88** | 0.34 | 0.14 |

Read the corners:

- **The veto is harmless warm and catastrophic cold.** At temperature 0.22 it puts **85% of the
  population permanently at rest** with a median exertion of 0.02 — the herd sits down and stops.
  That is the state the shipped temperature was hiding, and any change that made animals more
  decisive would have walked straight into it.
- **The graded drive is neutral warm.** Population and condition match the baseline within seed
  noise, which is what makes this shippable on its own: a latent pathology removed without moving
  the equilibrium anyone would notice.
- **Together they are better than either.** 2,728–2,863 living at the same condition is a **25%
  larger population** than the shipped corner, with a healthy 40% resting share.
- **Very cold is still bad** even graded: at 0.08 the herd sits down again, because fatigue's 0.34
  still beats hunger's 0.14 reliably once the noise is gone.

### What this does not fix

Hunger's spread is 0.15–0.28 throughout. Fatigue is no longer four times louder, but it is still
louder, and commitment's band (0.29) is comparable to both. **Foraging never rises far above
chance** — 0.554 at the very best cell against 0.521 shipped. Making hunger actually decide needs
either the general normalisation or a wider spread of its own, both of which are on #207 and
neither of which is a tuning change.

### Consequence for #205

Its gate was answered "no" on evidence that is now known to be **conditional on the veto**. With
fatigue graded, the decisive setting is the best cell measured rather than the worst. That is a
follow-up for that issue with these numbers, not something this change should smuggle in.
