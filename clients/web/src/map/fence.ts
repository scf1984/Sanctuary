// A fence as the player draws it and as the world holds it. The world holds a fence on cell
// *edges* (#27), so it is drawn as lines between cells: filling cells would put it half a cell
// off and make a pen look one cell smaller than the one the animals are held by (§3.3).

import type { BarrierView, FenceRequest } from "../api/values";
import { decodeColumn } from "../api/wire";

/** A drag between two world points, clamped to the map: the request a release would make. */
export function fenceRequest(a: readonly [number, number], b: readonly [number, number], width: number, height: number): FenceRequest {
  const clampX = (x: number) => Math.min(width, Math.max(0, x));
  const clampY = (y: number) => Math.min(height, Math.max(0, y));
  return {
    $type: "FenceRequest",
    first_x: clampX(a[0]),
    first_y: clampY(a[1]),
    second_x: clampX(b[0]),
    second_y: clampY(b[1]),
  };
}

export type Segment = readonly [number, number, number, number];

/** One segment per blocked edge, in world units: `[x0, y0, x1, y1]`. Cell (r, c) spans
 * `[c, c + 1) × [r, r + 1)` cells (#162), so its north edge sits at y = r and its west at x = c. */
export function fenceSegments(view: BarrierView, cellSize: number): Segment[] {
  const north = decodeColumn(view.blocked_north);
  const west = decodeColumn(view.blocked_west).values;
  const [rows, cols] = north.shape as [number, number];
  const segments: Segment[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const x = c * cellSize;
      const y = r * cellSize;
      if (north.values[i]) segments.push([x, y, x + cellSize, y]);
      if (west[i]) segments.push([x, y, x, y + cellSize]);
    }
  }
  return segments;
}
