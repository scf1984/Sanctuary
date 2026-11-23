// The diagnostic viewer's colours, carried over so the two clients read a world the same way
// (CLAUDE.md §3.3; clients/viewer/render.py holds the measurements behind each).

/** Condition, light to dark: one hue, five validated steps, darkening toward death. */
export const CONDITION_RAMP: readonly number[] = [0xfad4c4, 0xf4a582, 0xeb6834, 0xb8451d, 0x7d2c0f];

/** The ring around every dot. No single fill clears 3:1 against all terrain, so marks are cased. */
export const CASING = 0xfcfcfb;

/** Fences: the one hue nothing else uses, so a fence reads as the player's own mark. */
export const FENCE = 0x4a3aa7;

/** The ramp step for a deficit in [0, 1]: quantised, never blended. */
export function conditionStep(deficit: number): number {
  const step = Math.floor(Math.min(1, Math.max(0, deficit)) * CONDITION_RAMP.length);
  return Math.min(CONDITION_RAMP.length - 1, step);
}

export function conditionColor(deficit: number): number {
  return CONDITION_RAMP[conditionStep(deficit)] as number;
}

export function hex(color: number): string {
  return `#${color.toString(16).padStart(6, "0")}`;
}

/** Species, by id (#314): a categorical palette in a fixed order, stepped for the dark map and
 * validated by the dataviz checks (lightness band, chroma, colour-blind and normal-vision
 * separation between neighbours, 3:1 against the surface; all pass). A colour follows its
 * species for life, so a split never repaints the species that were already there. Past eight the
 * colours repeat, as decided on #314: selecting a species in the list is what tells two apart. */
export const SPECIES_PALETTE: readonly number[] = [
  0x3987e5, 0xd95926, 0x199e70, 0xc98500, 0xd55181, 0x0ca30c, 0x9085e9, 0xe66767,
];

/** Fill and ring for the animals a selection is not about: dark and nearly flat, so the selected
 * species stands out on every terrain without the others vanishing entirely. */
export const DIMMED_FILL = 0x3a3c37;
export const DIMMED_RING = 0x6a6c64;

export function speciesColor(species: number): number {
  const slots = SPECIES_PALETTE.length;
  return SPECIES_PALETTE[((species % slots) + slots) % slots] as number;
}
