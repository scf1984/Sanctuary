# What genetic distance is actually made of

Measured for #193, which asks how genes should be commensurated before `distance.between` takes a
Euclidean norm over them. That is not a question to settle from first principles — it depends on
which genes actually drift and how much each contributes in practice — so this is the data.

Produced by `docs/spikes/gene_observatory.py`, seed 0, 200 founders, 2,000 ticks, sampling every
100. Windows 11 / Python 3.12.10 / NumPy 2.5.1.

## The result

At tick 2,000 — 4,533 living animals, 2,000 sampled pairs. Distance is a Euclidean norm, so each
gene's share of the *squared* distance is the squared gap in that gene:

| gene | mean squared gap | share of distance |
|---|---:|---:|
| `maturity_age` | 459.25 | **64.1%** |
| `gestation_length` | 245.40 | **34.3%** |
| `sight` | 1.92 | 0.3% |
| `aversion0_6` | 0.82 | 0.1% |
| `aversion1_1` | 0.58 | 0.1% |
| `speed` | 0.58 | 0.1% |
| *remaining 30 genes* | 5.74 | 0.8% |

**Two genes are 98.4% of the metric.**

Grouped by what the genes are for:

| block | share of the vocabulary | share of distance |
|---|---:|---:|
| life history (`maturity_age`, `gestation_length`) | 5.6% | **98.4%** |
| cue space (24 signature and aversion genes) | 66.7% | **1.2%** |
| everything else (size, speed, sight, diet, …) | 27.8% | 0.4% |

## Why

Nothing is wrong with the arithmetic; the norm is unweighted and the founding ranges differ by two
orders of magnitude:

| gene | founding range |
|---|---|
| `maturity_age` | (40, 120) |
| `gestation_length` | (20, 60) |
| `sight` | (2, 6) |
| `signature_0..7` | (0, 1) |
| `size` | (0.8, 1.2) |

A gene contributes in proportion to its natural magnitude, so "genetic distance" currently means
**"difference in age at first reproduction"**.

## Why this matters more than it looks

§2.5 makes cue space the molecular clock: *"neutral drift is a molecular clock, which is precisely
what makes two isolated populations recognisably different."* Under this metric that block is 1.2%
of the signal. Two populations could diverge completely in cue space — becoming mutually invisible
as mates in every sense §2.5 describes, since mate-finding reads the searcher's own signature (#188)
— and still register as one species, because `maturity_age` happened not to drift.

## The live consequence, separately

`ConceptionConfig.speciation_threshold` ships at **8.0**, chosen without measurement in #191.
Median pairwise distance over this run held between **19.6 and 29.6**.

Instrumented over 300 ticks of the same world: **5,326 candidate couples formed, and 3.5% had a
non-zero interbreeding probability** (median 0.0000, max 0.775). The population grows through that
tail. The gate is not gating reproductive isolation — it is rejecting animals whose maturity differs
from their neighbours'.

The drift spike's threshold of **0.12** is also obsolete: it was measured against a small gene set
with unit-scale ranges, and nothing about it survives a vocabulary containing a gene ranged
(40, 120).

## What this does not decide

The commensuration itself. #193 carries the gate — a per-gene scale on `GeneSpec`, z-scoring against
population variance, restricting the metric to cue genes, or leaving it unweighted — and this file
is the evidence, not the answer. What it does establish is that "leave it unweighted" is not a
neutral option: it is a choice that speciation should track whichever gene happens to have the
widest founding range.

## The tool

`gene_observatory.py` writes two CSVs and depends on nothing beyond numpy and the standard library:

```
python -m docs.spikes.gene_observatory --ticks 3000 --out run
```

- `run-genes.csv` — per (tick, gene): p10 / median / p90 / mean / std of the expressed value across
  the living population, and drift from where the founders started.
- `run-distance.csv` — per (tick, gene): mean squared gap, share of distance, and the median total.

It is a spike, not a metric. #30 owns metric definitions and #157 owns showing trait distributions
to a player; this exists to answer one question, as `speciation_drift_spike.py` did for #104.
