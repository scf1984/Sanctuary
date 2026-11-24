import { describe, expect, it } from "vitest";

import type { ScenarioReport, SpeciesReport } from "../api/values";
import { goalStatus, progressShare } from "./goal";
import { currentHint, markDone, readSeen, writeSeen, type Facts, type Seen } from "./onboarding";

const report = (fields: Partial<ScenarioReport>): ScenarioReport => ({
  $type: "ScenarioReport",
  name: "first-split",
  title: "A species of your own",
  goal_text: "Split one off.",
  briefing: [],
  outcome: "open",
  progress: null,
  parent: null,
  daughter: null,
  split_tick: null,
  due_tick: null,
  resolved_tick: null,
  reason: null,
  ...fields,
});

const species = (id: number, name: string | null = null): SpeciesReport => ({
  $type: "SpeciesReport",
  id,
  name,
  parent: id === 0 ? null : 0,
  split_tick: id === 0 ? null : 1440,
  founders: id === 0 ? null : 40,
  living: 10,
});

describe("progressShare", () => {
  it("is null until something has been measured", () => {
    expect(progressShare(report({ progress: null }))).toBeNull();
  });

  it("is clipped to the bar, since the split lands in the generation the reading reaches 1", () => {
    expect(progressShare(report({ progress: 1.3 }))).toBe(1);
    expect(progressShare(report({ progress: 0.42 }))).toBeCloseTo(0.42);
  });
});

describe("goalStatus", () => {
  it("reports progress as a percentage before the split", () => {
    expect(goalStatus(report({ progress: 0.355 }), [species(0)])).toContain("36% of the way");
  });

  it("says when nothing has been measured rather than showing zero", () => {
    expect(goalStatus(report({}), [species(0)])).toContain("Not measured yet");
  });

  it("names both species and the day they must live to once split", () => {
    const text = goalStatus(
      report({ progress: 1, parent: 0, daughter: 1, split_tick: 1440, due_tick: 1560 }),
      [species(0, "Grazers"), species(1, "Hill folk")],
    );

    expect(text).toBe("Hill folk split off from Grazers, Day 2, 00:00. Keep both alive until Day 2, 02:00.");
  });

  it("gives the reason for a loss and says the world goes on", () => {
    const text = goalStatus(
      report({ outcome: "lost", resolved_tick: 60, reason: "the daughter species died out within a generation of the split" }),
      [species(0)],
    );

    expect(text).toContain("the daughter species died out");
    expect(text).toContain("The world keeps running.");
  });
});

const facts = (fields: Partial<Facts> = {}): Facts => ({ fenced: false, split: false, named: false, ...fields });
const seen = (fields: Partial<Seen> = {}): Seen => ({ briefing: true, done: [], ...fields });

describe("currentHint", () => {
  it("shows nothing until the briefing is closed", () => {
    expect(currentHint(seen({ briefing: false }), facts())).toBeNull();
  });

  it("asks for a fence first, then for a faster pace", () => {
    expect(currentHint(seen(), facts())).toBe("fence");
    expect(currentHint(seen(), facts({ fenced: true }))).toBe("pace");
  });

  it("skips a hint whose deed the world already shows", () => {
    expect(currentHint(seen({ done: ["pace"] }), facts({ fenced: true }))).toBeNull();
  });

  it("asks for a name only once a species has split off", () => {
    const after = seen({ done: ["fence", "pace"] });

    expect(currentHint(after, facts({ fenced: true }))).toBeNull();
    expect(currentHint(after, facts({ fenced: true, split: true }))).toBe("species");
    expect(currentHint(after, facts({ fenced: true, split: true, named: true }))).toBeNull();
  });

  it("never shows a dismissed hint again", () => {
    expect(currentHint(markDone(seen(), "fence"), facts())).toBe("pace");
  });
});

class Memory implements Storage {
  private items = new Map<string, string>();
  get length() {
    return this.items.size;
  }
  clear() {
    this.items.clear();
  }
  getItem(key: string) {
    return this.items.get(key) ?? null;
  }
  key(index: number) {
    return [...this.items.keys()][index] ?? null;
  }
  removeItem(key: string) {
    this.items.delete(key);
  }
  setItem(key: string, value: string) {
    this.items.set(key, value);
  }
}

describe("what a browser has seen", () => {
  it("round-trips per scenario", () => {
    const storage = new Memory();
    writeSeen(storage, "first-split", { briefing: true, done: ["fence"] });

    expect(readSeen(storage, "first-split")).toEqual({ briefing: true, done: ["fence"] });
    expect(readSeen(storage, "another")).toEqual({ briefing: false, done: [] });
  });

  it("reads nothing from storage that is missing, refuses, or holds junk", () => {
    const refusing = new Memory();
    refusing.getItem = () => {
      throw new Error("blocked");
    };
    const junk = new Memory();
    junk.setItem("sanctuary.onboarding.first-split", "{not json");

    expect(readSeen(undefined, "first-split")).toEqual({ briefing: false, done: [] });
    expect(readSeen(refusing, "first-split")).toEqual({ briefing: false, done: [] });
    expect(readSeen(junk, "first-split")).toEqual({ briefing: false, done: [] });
  });

  it("drops hint names it does not know", () => {
    const storage = new Memory();
    storage.setItem("sanctuary.onboarding.first-split", JSON.stringify({ briefing: true, done: ["fence", "dance"] }));

    expect(readSeen(storage, "first-split").done).toEqual(["fence"]);
  });
});
