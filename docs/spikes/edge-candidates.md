# Edge candidates: keeping options in the world without the wall doing anything else

Issue #209. `Behaviour.candidate_positions` clips each heading's look-ahead point into the world.
For an animal standing exactly on the boundary, some headings clip onto the animal's own position.
That option proposes no displacement, yet it sits under a travelling index, so winning it
recorded `choice_moving = True` for an animal that went nowhere.

**Decision (user, 2026-11-06):** `choice_moving` keeps its index meaning ("did not pick the null
column"). Animals must not leave the world. The edge effect is to be removed where the options are
made, not by redefining the flag.

So the question measured here is narrower than "how should a boundary work". It is: **which way of
removing the folded options changes nothing else?**

## Method

`edge_candidates.py`, in this directory. Every variant starts from the same evenly spaced,
jittered headings, on the demo world (`build_demo_world`, n = 2,000, `look_ahead = 4`), 4 seeds.

- **Folded options**: the population is placed exactly on each corner and each mid-wall point, and
  the script counts travelling options whose point equals the animal's position.
- **Density**: every drive is removed, so only the commitment bonus shapes a choice, and the
  population is stepped 400 ticks through the real `choose` -> `chosen_target` -> `Movement.step`.
  With no drives, any departure from uniform density is the boundary rule's doing. Density in the
  edge band (within one look-ahead of a wall) and in the four corner squares is measured over the
  second half, as a ratio to each region's area share, so 1.0 is no edge effect.
- **Skew**: mean resultant length of movers' headings. 0 is no preferred direction.

## Results

| variant | folded, corner | folded, mid-wall | edge density | corner density | skew |
|---|---:|---:|---:|---:|---:|
| clip (before) | **0.250** | 0.000 | 1.010 ± 0.024 | 0.955 ± 0.039 | 0.0020 |
| **shipped** | **0.000** | 0.000 | 1.010 ± 0.024 | 0.953 ± 0.037 | 0.0020 |
| mirror | 0.000 | 0.000 | 0.285 ± 0.012 | 0.069 ± 0.012 | 0.0021 |
| redraw | 0.000 | 0.000 | 0.382 ± 0.013 | 0.132 ± 0.026 | 0.0023 |
| turn every outward heading | 0.000 | 0.000 | 0.834 ± 0.031 | 0.485 ± 0.018 | 0.0021 |

Variants:
- **mirror** negates an outward heading's component.
- **redraw** resamples an outward heading from the headings that stay inside.
- **turn** rotates every outward heading to the nearest heading that stays inside.
- **shipped** applies the same rotation as turn, but *only* to headings that clipping would fold
  onto the animal.

## What it shows

- **Clipping was not the edge effect.** Under plain clipping an indifferent population is uniform
  up to the wall (1.01) and into the corners (0.95, within noise). Clipping projects an outward
  move along the wall, which is what lets a random walk stay uniform at a boundary.
- **Every full-reach rule creates one.** If every option must travel a full look-ahead *inside*
  the world, an animal at the wall can only be offered inward and along-wall moves. Its walk then
  drifts away from the wall. The edge band ends up at 0.29–0.83 of its share and corners at
  0.07–0.49: walls repel animals, which is a much larger defect than #209.
- **The real defect is exact folding**, and it is a corner phenomenon. A quarter of a cornered
  animal's travelling options pointed into the corner and clipped onto it. Mid-wall, a fold needs a
  heading exactly perpendicular to the wall, which the per-entity jitter makes measure-zero. In a
  live demo run (3 seeds, 300 ticks with full drives) about 1 animal in 10⁴ stands in a corner, and
  3 in 10⁵ travelling options are folded.
- **Shipped removes the folds and moves nothing else**: folded 0.25 -> 0, while density and skew are
  unchanged to three decimals. A folded heading is turned to the nearest heading whose full-reach
  point is inside, which at a corner means along one of the two walls. That is what clipping does
  to every heading that only grazes a wall.

## Consequences

- `choice_moving` and `Fatigue`'s zero-displacement test for rest now agree on every option
  `Behaviour` offers. The disagreement #209 recorded cannot arise.
- Movement and the commitment bonus read the stored heading, and the stored heading for a re-aimed
  option is the re-aimed one, so `chosen_target` points where the option was judged.
- `_unfold` raises if no heading from a boundary point keeps its look-ahead inside the world, which
  needs a world smaller than about one look-ahead across. That is a configuration error, so it fails loudly
  (§8.7) rather than being clamped.
