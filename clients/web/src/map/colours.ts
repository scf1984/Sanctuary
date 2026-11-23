// What colour each dot is, decided once a tick and kept apart from the renderer so it is tested
// without a GPU. A dot is a fill and a ring: the fill says what the mode asks (a deficit or a
// species), and the ring is the light casing every mark needs on this terrain (CLAUDE.md §3.3),
// unless a selection dims the animal.

import { CASING, conditionColor, DIMMED_FILL, DIMMED_RING, speciesColor } from "./palette";

/** The map's colour mode that shows species rather than a condition (#314, decision 4). */
export const SPECIES_MODE = "species";

/** Neutral grey while a frame's condition has not arrived (never a ramp step: it would lie). */
export const UNKNOWN = 0x9a9a94;

export interface Tints {
  readonly fill: Uint32Array;
  readonly ring: Uint32Array;
}

/**
 * Tints for `count` animals listed in a frame's order.
 *
 * deficits: each animal's deficit, for a condition mode; null shows species, or grey in a condition
 *   mode whose reading has not arrived (`bySpecies` says which).
 * species: each animal's species id, from the same frame.
 * highlight: a species to pick out, or null. Everyone else is dimmed, in either mode.
 */
export function dotTints(
  count: number,
  bySpecies: boolean,
  deficits: Float32Array | null,
  species: ArrayLike<number>,
  highlight: number | null,
): Tints {
  const fill = new Uint32Array(count);
  const ring = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const kind = species[i] as number;
    if (highlight !== null && kind !== highlight) {
      fill[i] = DIMMED_FILL;
      ring[i] = DIMMED_RING;
      continue;
    }
    ring[i] = CASING;
    if (bySpecies) fill[i] = speciesColor(kind);
    else fill[i] = deficits === null ? UNKNOWN : conditionColor(deficits[i] as number);
  }
  return { fill, ring };
}
