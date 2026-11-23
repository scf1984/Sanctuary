// Two frames blended by id (CLAUDE.md §2.1: the renderer interpolates between ticks; §3.3: by
// identity, never by row). An animal in both glides, one gone from the second has died and is
// dropped, and one new in the second is drawn where it is. Keying by id rather than by position
// in the list matters because a newborn often takes a row freed the same tick, and the world lists
// animals in row order.
//
// Matching is done once per tick (`pair`) and blending once per drawn frame (`blendInto`). Measured
// at 100,000 animals, matching inside every drawn frame cost 49 ms, three drawn frames' budget;
// split, the per-frame blend is a plain loop over arrays.

import type { Frame } from "../api/values";
import { decodeColumn } from "../api/wire";

/** A frame's columns, decoded once. */
export interface Positions {
  readonly tick: number;
  readonly ids: Float64Array;
  readonly x: Float32Array;
  readonly y: Float32Array;
  /** Each animal's species id (#314), in the same order. */
  readonly species: Int32Array;
}

export function positions(frame: Frame): Positions {
  const ids = decodeColumn(frame.ids).values;
  const x = decodeColumn(frame.x).values;
  const y = decodeColumn(frame.y).values;
  const species = decodeColumn(frame.species).values;
  if (
    !(ids instanceof Float64Array) ||
    !(x instanceof Float32Array) ||
    !(y instanceof Float32Array) ||
    !(species instanceof Int32Array)
  ) {
    throw new TypeError(
      `unexpected frame dtypes: ids ${frame.ids.dtype}, x ${frame.x.dtype}, y ${frame.y.dtype}, species ${frame.species.dtype}`,
    );
  }
  return { tick: frame.tick, ids, x, y, species };
}

/** Two consecutive frames, matched: `from[i]` is where `next`'s animal i was listed in
 * `previous`, or -1 for one born since. */
export interface Pairing {
  readonly previous: Positions | null;
  readonly next: Positions;
  readonly from: Int32Array;
}

export function pair(previous: Positions | null, next: Positions): Pairing {
  const from = new Int32Array(next.ids.length).fill(-1);
  if (previous !== null) {
    const at = new Map<number, number>();
    for (let j = 0; j < previous.ids.length; j++) at.set(previous.ids[j] as number, j);
    for (let i = 0; i < next.ids.length; i++) from[i] = at.get(next.ids[i] as number) ?? -1;
  }
  return { previous, next, from };
}

/** Where every animal of the pairing's `next` is drawn, `t` of the way from `previous` (0) to
 * `next` (1), written into `x` and `y` (each at least `next`'s length) in `next`'s order, so it
 * lines up with anything else read at `next`'s tick. */
export function blendInto(pairing: Pairing, t: number, x: Float32Array, y: Float32Array): void {
  const { previous, next, from } = pairing;
  const s = Math.min(1, Math.max(0, t));
  for (let i = 0; i < next.x.length; i++) {
    const j = from[i] as number;
    const nx = next.x[i] as number;
    const ny = next.y[i] as number;
    if (previous === null || j < 0) {
      x[i] = nx;
      y[i] = ny;
    } else {
      const px = previous.x[j] as number;
      const py = previous.y[j] as number;
      x[i] = px + (nx - px) * s;
      y[i] = py + (ny - py) * s;
    }
  }
}

/** `blendInto`, allocating: for tests and one-off reads. */
export function blend(previous: Positions | null, next: Positions, t: number): { x: Float32Array; y: Float32Array } {
  const x = new Float32Array(next.x.length);
  const y = new Float32Array(next.y.length);
  blendInto(pair(previous, next), t, x, y);
  return { x, y };
}
