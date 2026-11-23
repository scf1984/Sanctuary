import { describe, expect, it } from "vitest";

import type { Sample, SpeciesReport } from "../api/values";
import { lineageRows, newSplits, sparkline, speciesLabel, speciesSeries } from "./lineage";

const species = (id: number, parent: number | null, name: string | null = null): SpeciesReport => ({
  $type: "SpeciesReport",
  id,
  name,
  parent,
  split_tick: parent === null ? null : 100 * id,
  founders: parent === null ? null : 20,
  living: 10,
});

const sample = (living: number[]): Sample =>
  ({ $type: "Sample", species_living: living }) as unknown as Sample;

describe("lineageRows", () => {
  it("puts each daughter under its parent, oldest first", () => {
    const rows = lineageRows([species(0, null), species(1, 0), species(2, 1), species(3, 0)]);

    expect(rows.map((row) => [row.species.id, row.depth])).toEqual([
      [0, 0],
      [1, 1],
      [2, 2],
      [3, 1],
    ]);
  });

  it("lists every founding species at the top level", () => {
    const rows = lineageRows([species(0, null), species(1, null)]);

    expect(rows.map((row) => row.depth)).toEqual([0, 0]);
  });
});

describe("speciesLabel", () => {
  it("is the player's name once given", () => {
    expect(speciesLabel(species(2, 0, "Hill folk"))).toBe("Hill folk");
  });

  it("is a placeholder numbered from one until then", () => {
    expect(speciesLabel(species(0, null))).toBe("Species 1");
  });
});

describe("speciesSeries", () => {
  it("reads a species as none alive before it existed", () => {
    expect(speciesSeries([sample([40]), sample([30, 10]), sample([25, 0])], 1)).toEqual([0, 10, 0]);
  });
});

describe("newSplits", () => {
  it("finds daughters not seen before and never a founding species", () => {
    const now = [species(0, null), species(1, 0), species(2, 0)];

    expect(newSplits(now, new Set([0, 1])).map((s) => s.id)).toEqual([2]);
    expect(newSplits([species(0, null)], new Set())).toEqual([]);
  });
});

describe("sparkline", () => {
  it("scales to the box with zero on the baseline", () => {
    expect(sparkline([0, 5, 10], 20, 10)).toBe("0.0,10.0 10.0,5.0 20.0,0.0");
  });

  it("draws nothing for an empty series", () => {
    expect(sparkline([], 20, 10)).toBe("");
  });
});
