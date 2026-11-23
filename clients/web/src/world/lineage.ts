// The species list as a tree (#314, decision 5): each species under the one it split from, so the
// list reads as the world's history. Pure, so the order is tested without a browser.

import type { Sample, SpeciesReport } from "../api/values";

export interface LineageRow {
  readonly species: SpeciesReport;
  /** How many splits separate it from a founding species: the indent. */
  readonly depth: number;
}

/** Depth first: a parent, then each of its daughters' subtrees, oldest daughter first. Ids are
 * handed out in the order species arise, so ordering by id within a parent is ordering by age. */
export function lineageRows(species: readonly SpeciesReport[]): LineageRow[] {
  const children = new Map<number | null, SpeciesReport[]>();
  for (const report of species) {
    const siblings = children.get(report.parent) ?? [];
    siblings.push(report);
    children.set(report.parent, siblings);
  }
  const rows: LineageRow[] = [];
  const visit = (parent: number | null, depth: number) => {
    const under = (children.get(parent) ?? []).slice().sort((a, b) => a.id - b.id);
    for (const report of under) {
      rows.push({ species: report, depth });
      visit(report.id, depth + 1);
    }
  };
  visit(null, 0);
  return rows;
}

/** What a species is called: the player's name, or a neutral placeholder until they give one.
 * Numbered from 1, because "Species 0" reads as a mistake. */
export function speciesLabel(species: Pick<SpeciesReport, "id" | "name">): string {
  return species.name ?? `Species ${species.id + 1}`;
}

/** One species' living count at every recorded sample, oldest first. A sample from before the
 * species existed did not list it, which reads as none alive rather than as a gap. */
export function speciesSeries(history: readonly Sample[], species: number): number[] {
  return history.map((sample) => sample.species_living[species] ?? 0);
}

/** Splits recorded since `known` were made: the species whose ids are past the last one seen. */
export function newSplits(species: readonly SpeciesReport[], knownIds: ReadonlySet<number>): SpeciesReport[] {
  return species.filter((report) => report.parent !== null && !knownIds.has(report.id));
}

/** An SVG polyline's points for `series`, scaled into `width` × `height`, the baseline at zero so
 * a species halving looks like halving. A flat series sits on its own value, not on an edge. */
export function sparkline(series: readonly number[], width: number, height: number): string {
  if (series.length === 0) return "";
  const peak = Math.max(1, ...series);
  const step = series.length > 1 ? width / (series.length - 1) : 0;
  return series
    .map((value, i) => `${(i * step).toFixed(1)},${(height - (value / peak) * height).toFixed(1)}`)
    .join(" ");
}
