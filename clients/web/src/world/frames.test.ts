import { describe, expect, it } from "vitest";

import { blend, type Positions } from "./frames";

const at = (tick: number, ids: number[], x: number[], y: number[]): Positions => ({
  tick,
  ids: new Float64Array(ids),
  x: new Float32Array(x),
  y: new Float32Array(y),
});

describe("blend", () => {
  it("glides an animal present in both frames", () => {
    const { x, y } = blend(at(1, [5], [0], [10]), at(2, [5], [4], [20]), 0.25);
    expect(Array.from(x)).toEqual([1]);
    expect(Array.from(y)).toEqual([12.5]);
  });

  it("matches by id, not by position in the list", () => {
    // Row order shifts as rows are reused: animal 8 was listed second and is now first.
    const { x } = blend(at(1, [5, 8], [0, 100], [0, 0]), at(2, [8, 5], [110, 10], [0, 0]), 0.5);
    expect(Array.from(x)).toEqual([105, 5]);
  });

  it("draws a newborn where it is rather than gliding from a dead animal's row", () => {
    const { x } = blend(at(1, [5], [50], [0]), at(2, [6], [3], [0]), 0.5);
    expect(Array.from(x)).toEqual([3]);
  });

  it("drops an animal that died, since only the second frame's animals are drawn", () => {
    const { x } = blend(at(1, [5, 6], [0, 0], [0, 0]), at(2, [6], [2], [0]), 0.5);
    expect(x.length).toBe(1);
  });

  it("holds at the newer frame once the blend is complete", () => {
    expect(Array.from(blend(at(1, [5], [0], [0]), at(2, [5], [4], [0]), 3).x)).toEqual([4]);
  });

  it("draws the first frame as it is", () => {
    expect(Array.from(blend(null, at(2, [5], [4], [0]), 0).x)).toEqual([4]);
  });
});
