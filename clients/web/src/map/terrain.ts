// The ground as pixels: a hypsometric tint, hillshaded, with standing water over it. A port of
// clients/viewer/render.py's `elevation_shading` and `apply_water_overlay`, so the two clients
// shade one world alike. Shading is presentation, so it is computed here rather than by the
// world (CLAUDE.md §4 asks that of statistics, not of colour).

import type { TerrainView } from "../api/values";
import { decodeColumn } from "../api/wire";

const STOPS = [0, 0.5, 1] as const;
const COLORS = [
  [60, 110, 60],
  [140, 120, 80],
  [245, 245, 245],
] as const;
const LIGHT_ALTITUDE = (45 * Math.PI) / 180;
const LIGHT_AZIMUTH = (135 * Math.PI) / 180;
const WATER = [40, 90, 200] as const;
/** World units of depth at which water reaches its fullest tint. */
const WATER_REFERENCE_DEPTH = 3;

function tint(normalized: number, channel: 0 | 1 | 2): number {
  const [low, high, lo, hi] =
    normalized <= STOPS[1] ? [COLORS[0], COLORS[1], STOPS[0], STOPS[1]] : [COLORS[1], COLORS[2], STOPS[1], STOPS[2]];
  return low[channel] + (high[channel] - low[channel]) * ((normalized - lo) / (hi - lo));
}

/** RGBA, row-major, one pixel per cell: `rows × cols × 4`. */
export function terrainPixels(view: TerrainView): { rows: number; cols: number; rgba: Uint8ClampedArray } {
  const elevation = decodeColumn(view.elevation);
  const slope = decodeColumn(view.slope).values;
  const aspect = decodeColumn(view.aspect).values;
  const depth = decodeColumn(view.water_depth).values;
  const [rows, cols] = elevation.shape as [number, number];
  const heights = elevation.values;
  let lowest = Infinity;
  let highest = -Infinity;
  for (const h of heights) {
    lowest = Math.min(lowest, h);
    highest = Math.max(highest, h);
  }
  const span = highest - lowest;
  const zenith = Math.PI / 2 - LIGHT_ALTITUDE;
  const rgba = new Uint8ClampedArray(rows * cols * 4);
  for (let i = 0; i < rows * cols; i++) {
    const normalized = span > 0 ? ((heights[i] as number) - lowest) / span : 0;
    const s = slope[i] as number;
    const incidence =
      Math.cos(zenith) * Math.cos(s) + Math.sin(zenith) * Math.sin(s) * Math.cos((aspect[i] as number) - LIGHT_AZIMUTH);
    const brightness = 0.3 + 0.7 * Math.min(1, Math.max(0, incidence));
    const d = depth[i] as number;
    const alpha = d > 0 ? Math.max(0.5, Math.min(1, d / WATER_REFERENCE_DEPTH) * 0.85) : 0;
    for (const c of [0, 1, 2] as const) {
      // The viewer truncates the shaded value to uint8 before blending water, so this does too.
      const shaded = Math.trunc(Math.min(255, tint(normalized, c) * brightness));
      rgba[i * 4 + c] = Math.trunc(shaded * (1 - alpha) + WATER[c] * alpha);
    }
    rgba[i * 4 + 3] = 255;
  }
  return { rows, cols, rgba };
}
