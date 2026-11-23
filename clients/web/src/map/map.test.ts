import { describe, expect, it } from "vitest";

import type { BarrierView, Column, TerrainView } from "../api/values";
import { fit, pan, toScreen, toWorld, zoomAt } from "./camera";
import { fenceRequest, fenceSegments } from "./fence";
import { dotTints, UNKNOWN } from "./colours";
import { CASING, CONDITION_RAMP, conditionStep, DIMMED_FILL, SPECIES_PALETTE, speciesColor } from "./palette";
import { terrainPixels } from "./terrain";

function grid(dtype: "<f4" | "|b1", rows: number, cols: number, values: number[]): Column {
  const bytes = dtype === "<f4" ? new Uint8Array(new Float32Array(values).buffer) : new Uint8Array(values);
  let text = "";
  for (const byte of bytes) text += String.fromCharCode(byte);
  return { $type: "Column", dtype, shape: [rows, cols], data: btoa(text) };
}

describe("camera", () => {
  const world = { width: 100, height: 50 };

  it("fits the world, centred", () => {
    const camera = fit(world, { width: 400, height: 400 });
    expect(camera.scale).toBe(4);
    expect(toScreen(camera, 0, 25)).toEqual([0, 200]);
    expect(toScreen(camera, 100, 25)).toEqual([400, 200]);
  });

  it("round-trips a point", () => {
    const camera = { left: 3, top: -7, scale: 2.5 };
    const [x, y] = toWorld(camera, ...toScreen(camera, 12, 34));
    expect(x).toBeCloseTo(12);
    expect(y).toBeCloseTo(34);
  });

  it("moves the ground with the pointer when panned", () => {
    const camera = { left: 0, top: 0, scale: 2 };
    expect(toScreen(pan(camera, 10, -4), 0, 0)).toEqual([10, -4]);
  });

  it("keeps the point under the pointer still while zooming", () => {
    const camera = { left: 10, top: 10, scale: 2 };
    const before = toWorld(camera, 120, 80);
    const after = toWorld(zoomAt(camera, 1.7, 120, 80, { min: 0.1, max: 100 }), 120, 80);
    expect(after[0]).toBeCloseTo(before[0]);
    expect(after[1]).toBeCloseTo(before[1]);
  });

  it("holds the zoom inside its bounds", () => {
    expect(zoomAt({ left: 0, top: 0, scale: 2 }, 1000, 0, 0, { min: 1, max: 8 }).scale).toBe(8);
    expect(zoomAt({ left: 0, top: 0, scale: 2 }, 0.001, 0, 0, { min: 1, max: 8 }).scale).toBe(1);
  });
});

describe("fences", () => {
  it("clamps a drag to the map", () => {
    expect(fenceRequest([-5, 10], [120, 60], 100, 50)).toEqual({
      $type: "FenceRequest",
      first_x: 0,
      first_y: 10,
      second_x: 100,
      second_y: 50,
    });
  });

  it("draws a blocked edge on the cell boundary, not across the cell", () => {
    // Cell (1, 2) has its north edge blocked; cell (0, 1) its west edge. Cell size 2.
    const view: BarrierView = {
      $type: "BarrierView",
      blocked_north: grid("|b1", 2, 3, [0, 0, 0, 0, 0, 1]),
      blocked_west: grid("|b1", 2, 3, [0, 1, 0, 0, 0, 0]),
    };
    expect(fenceSegments(view, 2)).toEqual([
      [2, 0, 2, 2],
      [4, 2, 6, 2],
    ]);
  });
});

describe("condition colour", () => {
  it("is quantised to the ramp and darkens toward death", () => {
    expect(conditionStep(0)).toBe(0);
    expect(conditionStep(0.19)).toBe(0);
    expect(conditionStep(0.2)).toBe(1);
    expect(conditionStep(1)).toBe(CONDITION_RAMP.length - 1);
  });

  it("clips what arrives outside [0, 1] rather than indexing off the ramp", () => {
    expect(conditionStep(-3)).toBe(0);
    expect(conditionStep(7)).toBe(CONDITION_RAMP.length - 1);
  });
});

describe("species colour", () => {
  it("follows the species id, so a split never repaints the species already there", () => {
    expect(speciesColor(0)).toBe(SPECIES_PALETTE[0]);
    expect(speciesColor(3)).toBe(SPECIES_PALETTE[3]);
  });

  it("repeats past the palette rather than inventing a hue", () => {
    expect(speciesColor(SPECIES_PALETTE.length + 1)).toBe(SPECIES_PALETTE[1]);
  });
});

describe("dot tints", () => {
  const species = new Int32Array([0, 1, 1]);

  it("colours by species in species mode, every ring cased", () => {
    const { fill, ring } = dotTints(3, true, null, species, null);

    expect(Array.from(fill)).toEqual([speciesColor(0), speciesColor(1), speciesColor(1)]);
    expect(Array.from(ring)).toEqual([CASING, CASING, CASING]);
  });

  it("colours by deficit in a condition mode, and grey while it has not arrived", () => {
    expect(dotTints(1, false, new Float32Array([0.99]), species, null).fill[0]).toBe(CONDITION_RAMP[4]);
    expect(dotTints(1, false, null, species, null).fill[0]).toBe(UNKNOWN);
  });

  it("dims everyone outside a selected species, in either mode", () => {
    for (const bySpecies of [true, false]) {
      const { fill, ring } = dotTints(3, bySpecies, new Float32Array([0, 0, 0]), species, 1);

      expect(fill[0]).toBe(DIMMED_FILL);
      expect(ring[0]).not.toBe(CASING);
      expect(ring[1]).toBe(CASING);
      expect(fill[1]).not.toBe(DIMMED_FILL);
    }
  });
});

describe("terrain", () => {
  const flat = (elevation: number[], depth: number[]): TerrainView => ({
    $type: "TerrainView",
    elevation: grid("<f4", 1, elevation.length, elevation),
    slope: grid("<f4", 1, elevation.length, elevation.map(() => 0)),
    aspect: grid("<f4", 1, elevation.length, elevation.map(() => 0)),
    water_depth: grid("<f4", 1, depth.length, depth),
  });

  it("tints low ground green and high ground white, as the viewer does", () => {
    // Flat ground at 45° light: brightness 0.3 + 0.7 × cos(45°).
    const brightness = 0.3 + 0.7 * Math.cos(Math.PI / 4);
    const { rgba } = terrainPixels(flat([0, 1], [0, 0]));
    expect(Array.from(rgba.slice(0, 4))).toEqual([Math.trunc(60 * brightness), Math.trunc(110 * brightness), Math.trunc(60 * brightness), 255]);
    expect(Array.from(rgba.slice(4, 8))).toEqual([Math.trunc(245 * brightness), Math.trunc(245 * brightness), Math.trunc(245 * brightness), 255]);
  });

  it("puts any standing water at least half way to blue", () => {
    // A puddle blends at the viewer's floor of 0.5 toward water (40, 90, 200).
    const brightness = 0.3 + 0.7 * Math.cos(Math.PI / 4);
    const { rgba } = terrainPixels(flat([0, 0], [0, 0.01]));
    expect(rgba[4 + 2]).toBe(Math.trunc(Math.trunc(60 * brightness) * 0.5 + 200 * 0.5));
  });
});

describe("fitting below the status bar", () => {
  it("starts the world below the inset and still shows all of it", () => {
    const camera = fit({ width: 100, height: 100 }, { width: 400, height: 456 }, 56);
    expect(camera.scale).toBe(4);
    expect(toScreen(camera, 50, 0)[1]).toBeCloseTo(56);
    expect(toScreen(camera, 50, 100)[1]).toBeCloseTo(456);
  });
});
