// World units to screen pixels and back. The world is top-down with y growing downward, as the
// viewer draws it (clients/viewer/render.py), so row r is drawn above row r + 1.

export interface Camera {
  /** The world point at the screen's top-left corner, world units. */
  readonly left: number;
  readonly top: number;
  /** Pixels per world unit. */
  readonly scale: number;
}

export interface Extent {
  readonly width: number;
  readonly height: number;
}

/** Pixels kept clear at the screen's top for the status bar, so a fitted world starts below it. */
export const TOP_INSET = 56;

/** The whole world, centred and as large as fits below `top` pixels kept clear. */
export function fit(world: Extent, screen: Extent, top = 0): Camera {
  const height = Math.max(1, screen.height - top);
  const scale = Math.min(screen.width / world.width, height / world.height);
  return {
    scale,
    left: (world.width - screen.width / scale) / 2,
    top: (world.height - height / scale) / 2 - top / scale,
  };
}

export function toScreen(camera: Camera, x: number, y: number): [number, number] {
  return [(x - camera.left) * camera.scale, (y - camera.top) * camera.scale];
}

export function toWorld(camera: Camera, px: number, py: number): [number, number] {
  return [camera.left + px / camera.scale, camera.top + py / camera.scale];
}

/** Moved by a drag of (dx, dy) pixels: the ground follows the pointer. */
export function pan(camera: Camera, dx: number, dy: number): Camera {
  return { ...camera, left: camera.left - dx / camera.scale, top: camera.top - dy / camera.scale };
}

/** Zoomed by `factor` about a screen point, which stays over the same world point, with the scale
 * held inside `bounds` (pixels per world unit) so the map can neither vanish nor become one cell. */
export function zoomAt(
  camera: Camera,
  factor: number,
  px: number,
  py: number,
  bounds: { readonly min: number; readonly max: number },
): Camera {
  const scale = Math.min(bounds.max, Math.max(bounds.min, camera.scale * factor));
  const [wx, wy] = toWorld(camera, px, py);
  return { scale, left: wx - px / scale, top: wy - py / scale };
}
