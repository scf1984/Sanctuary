// What the goal panel says (#317). The service decides where the goal stands (metrics.goals) and
// hands over the scenario's own words; this only turns that report into lines a person reads.

import type { ScenarioReport, SpeciesReport } from "../api/values";
import { simTime } from "./clock";
import { speciesLabel } from "./lineage";

/** Progress as a share of the bar, in [0, 1], or null while nothing has been measured. Clipped,
 * because the check splits a species in the generation its reading reaches 1, so a value past it
 * is only how far past the line the split landed. */
export function progressShare(report: ScenarioReport): number | null {
  if (report.progress === null) return null;
  return Math.min(Math.max(report.progress, 0), 1);
}

/** Where the goal stands, in one sentence. */
export function goalStatus(report: ScenarioReport, species: readonly SpeciesReport[]): string {
  const named = (id: number | null) => {
    const found = species.find((s) => s.id === id);
    return found === undefined ? "the species" : speciesLabel(found);
  };
  const when = (tick: number | null) => (tick === null ? "" : simTime(tick));
  if (report.outcome === "met") {
    return `Done, ${when(report.resolved_tick)}: ${named(report.daughter)} and ${named(report.parent)} both lived a generation past the split.`;
  }
  if (report.outcome === "lost") {
    return `Lost, ${when(report.resolved_tick)}: ${report.reason ?? "a species died out"}. The world keeps running.`;
  }
  if (report.daughter !== null) {
    return `${named(report.daughter)} split off from ${named(report.parent)}, ${when(report.split_tick)}. Keep both alive until ${when(report.due_tick)}.`;
  }
  const share = progressShare(report);
  if (share === null) return "Not measured yet: the groups are checked about once a generation.";
  return `The two most different groups are ${Math.round(share * 100)}% of the way to a split.`;
}
