// Onboarding (#317, decision 6): a briefing on first open, then one hint at a time, each anchored
// to the control it describes and gone once the player has done what it says. What a viewer has
// seen is about the person, not the world, so it lives in this browser and never in the snapshot.

export type HintId = "fence" | "pace" | "species";

export interface Seen {
  /** Whether the briefing has been read and closed. */
  readonly briefing: boolean;
  /** Hints done or dismissed, never shown again in this browser. */
  readonly done: readonly HintId[];
}

/** What has happened in the world that a hint describes, read from what the service hands over. */
export interface Facts {
  /** A fence has gone up, whoever built it. */
  readonly fenced: boolean;
  /** A species has split off, so there is a new row in the species list. */
  readonly split: boolean;
  /** Some species has a name the player gave it. */
  readonly named: boolean;
}

const NOTHING_SEEN: Seen = { briefing: false, done: [] };

/** The hint to show now, if any. One at a time, in the order a first game needs them: build a
 * pen, then speed the wait up, then, once a split has happened, name what it made. A hint whose
 * deed the world already shows is skipped, so a player returning on a second device is not told
 * to build a fence that is standing. */
export function currentHint(seen: Seen, facts: Facts): HintId | null {
  const done = (id: HintId) => seen.done.includes(id);
  if (!seen.briefing) return null;
  if (!done("fence") && !facts.fenced) return "fence";
  if (!done("pace")) return "pace";
  if (facts.split && !done("species") && !facts.named) return "species";
  return null;
}

export function markDone(seen: Seen, id: HintId): Seen {
  return seen.done.includes(id) ? seen : { ...seen, done: [...seen.done, id] };
}

const key = (scenario: string) => `sanctuary.onboarding.${scenario}`;

/** What this browser has seen of `scenario`'s onboarding. Storage can be missing or refuse (a
 * private window, blocked site data), and then the briefing shows again, which is harmless. */
export function readSeen(storage: Storage | undefined, scenario: string): Seen {
  try {
    const raw = storage?.getItem(key(scenario));
    if (raw === null || raw === undefined) return NOTHING_SEEN;
    const parsed = JSON.parse(raw) as Partial<Seen>;
    return {
      briefing: parsed.briefing === true,
      done: Array.isArray(parsed.done) ? parsed.done.filter(isHint) : [],
    };
  } catch {
    return NOTHING_SEEN;
  }
}

export function writeSeen(storage: Storage | undefined, scenario: string, seen: Seen): void {
  try {
    storage?.setItem(key(scenario), JSON.stringify(seen));
  } catch {
    // Not kept: the page still works, and the hints come back next time.
  }
}

function isHint(value: unknown): value is HintId {
  return value === "fence" || value === "pace" || value === "species";
}
