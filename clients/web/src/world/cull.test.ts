import { describe, expect, it } from "vitest";

import { clampCount, cullable, suggestedCount } from "./cull";

describe("cullable", () => {
  it("is everything above the guard", () => {
    expect(cullable(40, 6)).toBe(34);
  });

  it("is nothing at or below the guard", () => {
    expect(cullable(6, 6)).toBe(0);
    expect(cullable(3, 6)).toBe(0);
  });
});

describe("clampCount", () => {
  it("keeps a count inside what may be taken", () => {
    expect(clampCount(100, 40, 6)).toBe(34);
    expect(clampCount(0, 40, 6)).toBe(1);
    expect(clampCount(-5, 40, 6)).toBe(1);
    expect(clampCount(7.6, 40, 6)).toBe(8);
  });

  it("falls back to one for a field that is not a number", () => {
    expect(clampCount(Number.NaN, 40, 6)).toBe(1);
  });

  it("is zero when the species is already at its guard", () => {
    expect(clampCount(5, 6, 6)).toBe(0);
  });
});

describe("suggestedCount", () => {
  it("starts at a tenth of what may be taken, and at least one", () => {
    expect(suggestedCount(206, 6)).toBe(20);
    expect(suggestedCount(8, 6)).toBe(1);
  });
});
