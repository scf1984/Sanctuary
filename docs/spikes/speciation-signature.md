# Spike: what the signature block does in a live world (#16)

The owner chose the 8 scent-signature genes as the only genes genetic distance reads (#16, 2026-11-19).
Before any threshold is chosen, this measures what those genes actually do in the demo world: how far
apart animals sit within one interbreeding population, how far apart two sides of a fence drift, and
how much that depends on how widely founders' signatures are spread.

Produced by `docs/spikes/speciation_signature.py` plus a longer follow-up run, demo world, 200
founders. The fenced runs enclose `(10, 10)–(70, 40)` at tick 0, about 28% of the 80×80 map. A
generation is about 120 ticks (mean maturity plus mean gestation), so 3,000 ticks is roughly 25
generations. Windows 10 / Python 3.13.

Distances are Euclidean over the 8 signature genes, read through `Genetics.heritable`. "pair" is the
distance between two random living animals; "2-means sep" is the distance between the centroids of
the two clusters a farthest-point-seeded 2-means finds in the whole population; "cross-fence
centroid" is the distance between the inside and outside centroids.

## Founders as shipped: signatures uniform over (0, 1)

| seed | tick | living | pair p50 | 2-means sep | inside | cross-fence centroid | cross pair p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 200 | 1.149 | 0.597 | 60 | 0.137 | 1.139 |
| 1 | 1000 | 259 | 0.894 | 0.788 | 17 | 0.598 | 0.933 |
| 1 | 3000 | 1915 | 0.875 | 0.732 | 332 | 0.777 | 1.014 |
| 2 | 0 | 200 | 1.139 | 0.540 | 66 | 0.100 | 1.138 |
| 2 | 1000 | 108 | 0.888 | 1.345 | 21 | 0.698 | 1.019 |
| 2 | 3000 | 2430 | 0.655 | 1.976 | 412 | 0.718 | 0.830 |
| 1, **unfenced** | 3000 | 2977 | 0.586 | 0.310 | | | |

**The fence produces a large divergence quickly, but it is a founder effect, not isolation.** Both
worlds crash from 200 to about 60 in the first 500 ticks and regrow. The pen's survivors are some
15 animals sampled from founders spread across the whole of cue space, so their mean lands far from
everyone else's at once (0.6–0.7 by tick 1,000). That is real divergence between the two sides, but
**a typical cross-fence pair is no further apart than a typical pair inside one side** (0.83–1.01
against 0.66–0.88). Two animals from opposite sides would still find each other and breed. Any
threshold on individual pairs that lets one side breed internally also lets the sides interbreed.

## Founders narrowed: signatures uniform over (0.45, 0.55)

| seed | tick | living | mean mutability | pair p50 | 2-means sep | inside | cross-fence centroid |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 200 | 0.020 | 0.115 | 0.060 | 60 | 0.014 |
| 1 | 1000 | 160 | 0.050 | 0.241 | 0.347 | 11 | 0.096 |
| 1 | 2000 | 1708 | 0.073 | 0.461 | 0.387 | 9 | 0.164 |
| 1 | 3000 | 1854 | **0.115** | 0.813 | 0.640 | 31 | 0.152 |
| 2 | 0 | 200 | 0.020 | 0.114 | 0.054 | 66 | 0.010 |
| 2 | 1000 | 98 | 0.014 | 0.121 | 0.091 | 49 | 0.076 |
| 2 | 3000 | 936 | 0.007 | 0.116 | 0.105 | 613 | 0.100 |
| 2 | 4000 | 1059 | 0.008 | | 0.101 | 610 | 0.099 |
| 2 | 6000 | 1545 | **0.011** | | 0.113 | 727 | 0.110 |

Two things, and both bind the design.

**1. Mutability evolves, by an order of magnitude, and the signature spread follows it.** Seed 1's
mean `mutability` rose sixfold in 3,000 ticks and its pairwise signature distance sevenfold; seed 2's
mutability fell to a third. The inheritance floor is a gene (§2.5) and selection moves it. So **an
absolute signature threshold means a different thing in every world, and in one world at different
times**: seed 1's whole-population 2-means separation reached 0.64 with no isolation behind it, which
would split any threshold set from seed 2's numbers. A gate has to be stated relative to the species'
own spread, or it measures mutability rather than isolation.

**2. Drift alone is slow at these population sizes.** In seed 2 the pen held 600–700 animals, and the
two sides' centroids sat about one typical pairwise distance apart (0.10–0.11) from tick 3,000 to
6,000 without growing much. A 2-means split of the whole population finds the same separation, so at
50 generations a clustering cannot yet tell the fenced world from an unfenced one. The drift spike
(`speciation-drift.md`) used pools of 40, and expected centroid drift shrinks with pool size, so its
"tens of generations" does not transfer to pens of hundreds.

## What this means for #16 and #292

- **The signature block works as a clock**: with narrow founders, the two sides of a fence drift
  apart slowly and steadily, and nothing else moves the centroids. But at live rate (§2.1, about
  2 minutes a generation on the demo world) the separation a gate could trust takes **hundreds of
  generations, hours of real time**. That fits "the world keeps running while you're away", and it
  does not fit #292's done-when of seeing a split within one session.
- **Wide founding makes the first split quick and wrong**: a founder effect separates the sides'
  means while their members still overlap as mates.
- **One absolute number cannot serve both individuals and groups.** Interbreeding acts on a *pair*,
  and two individuals of one species sit about √2 × the per-gene spread × √8 apart. Splitting acts on
  *centroids*, which sit much closer. A threshold on one is not the same quantity on the other, and
  §2.5's "the same threshold a split fires on" only holds once both are expressed in the species' own
  spread.
