// A tick as a time a person can read. One tick is one sim-minute (CLAUDE.md §2.1), so this is
// arithmetic on the only clock there is, never a reading of the wall clock.

const MINUTES_PER_DAY = 24 * 60;

export function simTime(tick: number): string {
  const day = Math.floor(tick / MINUTES_PER_DAY) + 1;
  const minute = tick % MINUTES_PER_DAY;
  const hh = String(Math.floor(minute / 60)).padStart(2, "0");
  const mm = String(minute % 60).padStart(2, "0");
  return `Day ${day}, ${hh}:${mm}`;
}
