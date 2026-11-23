import { describe, expect, it } from "vitest";

import type { Pace } from "../api/values";
import { blendPeriod, paceLabel, speeds } from "./pace";

const pace = (ticks: number, own = 1, fastest = 4): Pace => ({
  $type: "Pace",
  ticks_per_second: ticks,
  own_ticks_per_second: own,
  fastest_ticks_per_second: fastest,
});

describe("speeds", () => {
  it("doubles from the world's own rate up to the fastest it sustains", () => {
    expect(speeds(pace(1))).toEqual([1, 2, 4]);
  });

  it("ends on the fastest when doubling would step past it", () => {
    expect(speeds(pace(1, 1, 3))).toEqual([1, 2, 3]);
  });

  it("offers only the own rate when the world sustains nothing faster", () => {
    expect(speeds(pace(1, 1, 1))).toEqual([1]);
  });

  it("never offers a rate past the fastest, even when the own rate is", () => {
    expect(speeds(pace(1, 5, 4))).toEqual([4]);
  });
});

describe("paceLabel", () => {
  it("says paused at zero", () => {
    expect(paceLabel(pace(0))).toBe("paused");
  });

  it("reads a rate as a multiple of the world's own", () => {
    expect(paceLabel(pace(1))).toBe("1×");
    expect(paceLabel(pace(4))).toBe("4×");
    expect(paceLabel(pace(3, 2, 4))).toBe("1.5×");
  });
});

describe("blendPeriod", () => {
  it("is the gap between frames while they arrive at the rate", () => {
    expect(blendPeriod(1000, 1, 1)).toBe(1000);
    expect(blendPeriod(1000, 4, 4)).toBe(1000);
  });

  it("does not spend a pause gliding: the first frame after one blends over its own ticks' time", () => {
    // Paused for a minute, then resumed at 1 tick a second: one tick should take one second.
    expect(blendPeriod(60_000, 1, 1)).toBe(1000);
  });

  it("keeps up with a backlog arriving faster than the rate", () => {
    // Catch-up pays a chunk of 60 ticks every 200 ms: blend at the pace they come, not at 60 s.
    expect(blendPeriod(200, 60, 1)).toBe(200);
  });

  it("follows the gap while paused, when a backlog is still being paid", () => {
    expect(blendPeriod(300, 60, 0)).toBe(300);
  });

  it("stays within a tenth of a second and five seconds", () => {
    expect(blendPeriod(10, 1, 1)).toBe(100);
    expect(blendPeriod(20_000, 30, 1)).toBe(5000);
  });
});
