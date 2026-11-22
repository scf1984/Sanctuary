// How fast the world runs while a player watches (#307, #311). The service holds the rate and the
// ceiling; this decides only what to offer a person and how long a blend between frames takes.

import type { Pace } from "../api/values";

/** The rates offered, in ticks a second: the world's own, doubling, up to the fastest it sustains
 * (CLAUDE.md §2.4: the ceiling is measured, and a rate past it only builds a debt). Doubling
 * because a player reads speed as a multiple, and the fastest closes the list so the measured
 * ceiling is always reachable. */
export function speeds(pace: Pace): number[] {
  const fastest = pace.fastest_ticks_per_second;
  const offered: number[] = [];
  for (let rate = pace.own_ticks_per_second; rate < fastest; rate *= 2) offered.push(rate);
  offered.push(fastest);
  return offered;
}

/** The rate as a person reads it: "paused", or a multiple of the world's own. */
export function paceLabel(pace: Pace): string {
  if (pace.ticks_per_second === 0) return "paused";
  const multiple = pace.ticks_per_second / pace.own_ticks_per_second;
  return `${Number(multiple.toFixed(2))}×`;
}

/** Milliseconds a blend toward a new frame takes, given the gap since the last frame arrived, the
 * ticks between the two frames, and the rate (CLAUDE.md §2.1: the renderer interpolates between
 * ticks).
 *
 * Normally the gap. The ticks' own time at the rate caps it, so a frame arriving after a pause
 * does not spend the whole pause gliding to where it is. A backlog paid faster than the rate
 * arrives with short gaps, and the gap wins there, so the dots keep up. At a pause the ticks'
 * time is unbounded, so only a backlog still being paid moves anything. */
export function blendPeriod(gapMs: number, ticks: number, ticksPerSecond: number): number {
  const ticksTime = ticksPerSecond > 0 ? (ticks / ticksPerSecond) * 1000 : Infinity;
  return Math.min(5000, Math.max(100, Math.min(gapMs, ticksTime)));
}
