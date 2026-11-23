// What a cull may ask for (#28). The world refuses anything else at the boundary; this only keeps
// the control from offering a cull that is bound to be refused. The guard is the world's own
// number (WorldInfo.cull_survivors), never one the client chooses.

/** The most a cull may take from a species with `living` born animals: everything above the guard. */
export function cullable(living: number, guard: number): number {
  return Math.max(living - guard, 0);
}

/** What the count field should hold after the player typed `typed`: a whole number from 1 to the
 * most that may be taken, or 0 when nothing may be. */
export function clampCount(typed: number, living: number, guard: number): number {
  const most = cullable(living, guard);
  if (most === 0) return 0;
  if (!Number.isFinite(typed)) return 1;
  return Math.min(Math.max(Math.round(typed), 1), most);
}

/** A first suggestion: a tenth of what may be taken, so the first click is a thinning, not a purge. */
export function suggestedCount(living: number, guard: number): number {
  return clampCount(Math.ceil(cullable(living, guard) / 10), living, guard);
}
