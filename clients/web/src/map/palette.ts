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
