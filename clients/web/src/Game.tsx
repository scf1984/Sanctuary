// One world on screen: the map, the clock, the fence tool and the species list. State that changes
// every tick (the animals) goes straight to the renderer; what a person reads (status, price,
// outcomes, species) is React's.

import { useCallback, useEffect, useRef, useState } from "react";

import * as client from "./api/client";
import type { FenceRequest, Ledger, Pace, Progress, Quote, ScenarioReport, SpeciesReport, Status, WorldInfo } from "./api/values";
import { decodeColumn } from "./api/wire";
import { Briefing, Hint, HINT_TEXT } from "./hud/Goal";
import { Hud, type Notice } from "./hud/Hud";
import { SpeciesPanel } from "./hud/Species";
import { pan, toWorld, zoomAt } from "./map/camera";
import { dotTints, SPECIES_MODE } from "./map/colours";
import { fenceRequest } from "./map/fence";
import { MapRenderer } from "./map/renderer";
import { positions, type Positions } from "./world/frames";
import { newSplits, speciesLabel, speciesSeries } from "./world/lineage";
import { currentHint, markDone, readSeen, writeSeen, type HintId, type Seen } from "./world/onboarding";
import { speeds } from "./world/pace";

export type Tool = "look" | "fence";

/** How far the map may zoom in, pixels per world unit. Out, it stops at a quarter of the fit. */
const MAX_SCALE = 64;
/** A press that moves less than this many pixels is a click, not a drag. */
const DRAG_THRESHOLD = 4;
/** Milliseconds a drag must rest before it is priced, so a quote is not asked for every pixel. */
const QUOTE_DELAY = 120;
/** Milliseconds between re-reading the history for the species sparklines. A sample lands every
 * twenty ticks in the demo world, and the history is the largest read there is. */
const HISTORY_INTERVAL = 5000;

type Gesture =
  | { kind: "pan"; x: number; y: number }
  | { kind: "fence"; from: [number, number]; x: number; y: number; moved: boolean };

export function Game({ world }: { world: WorldInfo }) {
  const host = useRef<HTMLDivElement>(null);
  const renderer = useRef<MapRenderer | null>(null);
  const gesture = useRef<Gesture | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [connected, setConnected] = useState(false);
  const [mode, setMode] = useState<string>(world.conditions[0] ?? "hunger");
  const [tool, setTool] = useState<Tool>("look");
  const [draft, setDraft] = useState<FenceRequest | null>(null);
  const [pending, setPending] = useState<FenceRequest | null>(null);
  const [quote, setQuote] = useState<Quote | null>(null);
  const [ledger, setLedger] = useState<Ledger | null>(null);
  const [notices, setNotices] = useState<Notice[]>([]);
  const [failure, setFailure] = useState<string | null>(null);
  const [pace, setPace] = useState<Pace | null>(null);
  /** The rate a resume returns to: the last one chosen, so pausing a fast-forward and resuming
   * keeps the fast-forward. */
  const resumeAt = useRef<number | null>(null);
  const [species, setSpecies] = useState<readonly SpeciesReport[]>([]);
  const [series, setSeries] = useState<ReadonlyMap<number, readonly number[]>>(new Map());
  const [selected, setSelected] = useState<number | null>(null);
  /** Species ids this page has already shown, so a split is announced once. Null until the first
   * read, which announces nothing: those splits happened before the player arrived, and the list
   * shows them, each unnamed daughter marked to be named. */
  const knownSpecies = useRef<Set<number> | null>(null);
  const historyRead = useRef(0);
  const [scenario, setScenario] = useState<ScenarioReport | null>(null);
  /** What this browser has seen of the scenario's onboarding (#317), once the scenario is known. */
  const [seen, setSeen] = useState<Seen | null>(null);
  const [briefing, setBriefing] = useState(false);

  // A request's outcome lands in the ledger at the next tick boundary (§2.7), so after asking, the
  // next progress event is when to look. `known` is how many records the player has been shown.
  const awaitingOutcome = useRef(false);
  const known = useRef<number | null>(null);
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  /** The frame on screen and its deficits, kept so a new selection or mode repaints at once. */
  const shown = useRef<{ frame: Positions; deficits: Float32Array | null } | null>(null);

  const paint = useCallback(() => {
    const current = shown.current;
    if (current === null) return;
    const { frame, deficits } = current;
    const bySpecies = modeRef.current === SPECIES_MODE;
    const tints = dotTints(frame.ids.length, bySpecies, deficits, frame.species, selectedRef.current);
    renderer.current?.setFrame(frame, tints, performance.now());
  }, []);

  const notify = useCallback((notice: Omit<Notice, "id">) => {
    const id = Date.now() + Math.random();
    setNotices((all) => [...all.slice(-3), { ...notice, id }]);
    setTimeout(() => setNotices((all) => all.filter((n) => n.id !== id)), 8000);
  }, []);

  // -- frames ------------------------------------------------------------------------------------

  const fetching = useRef(false);
  const again = useRef(false);
  const refreshFrame = useCallback(async () => {
    if (fetching.current) {
      again.current = true;
      return;
    }
    fetching.current = true;
    try {
      do {
        again.current = false;
        if (modeRef.current === SPECIES_MODE) {
          // Species ride the frame itself, so there is no second call to keep in step.
          shown.current = { frame: positions(await client.frame()), deficits: null };
        } else {
          // Frame and condition are two calls, so a tick can land between them. A condition is only
          // drawn against the frame of its own tick (api.values.ConditionView); otherwise ask again.
          let [frame, condition] = await Promise.all([client.frame(), client.condition(modeRef.current)]);
          if (condition.tick !== frame.tick) {
            [frame, condition] = await Promise.all([client.frame(), client.condition(modeRef.current)]);
          }
          const deficits = condition.tick === frame.tick ? decodeColumn(condition.values).values : null;
          shown.current = { frame: positions(frame), deficits: deficits instanceof Float32Array ? deficits : null };
        }
        paint();
      } while (again.current);
    } catch (error) {
      setFailure(String(error));
    } finally {
      fetching.current = false;
    }
  }, [paint]);

  // -- species -----------------------------------------------------------------------------------

  const refreshSpecies = useCallback(async () => {
    const next = await client.species();
    const known = knownSpecies.current;
    if (known === null) {
      // A world that has already split opens on species: that is what there is to see.
      if (next.length > 1) setMode(SPECIES_MODE);
    } else {
      const byId = new Map(next.map((report) => [report.id, report]));
      for (const split of newSplits(next, known)) {
        const parent = split.parent === null ? undefined : byId.get(split.parent);
        const from = parent === undefined ? "its parent" : speciesLabel(parent);
        notify({
          tone: "ok",
          text: `A new species has split off from ${from}, with ${split.founders ?? 0} founders. Name it in the species list.`,
        });
        setMode(SPECIES_MODE);
      }
    }
    knownSpecies.current = new Set(next.map((report) => report.id));
    setSpecies(next);
  }, [notify]);

  // -- the scenario (#317) ------------------------------------------------------------------------

  /** The outcome this page last showed, so the moment a goal is met or lost is announced once. Null
   * until the first read, which announces nothing: an outcome reached while the player was away is
   * on the goal panel when they arrive. */
  const shownOutcome = useRef<string | null>(null);
  const refreshScenario = useCallback(async () => {
    const next = await client.scenario();
    if (next !== null && shownOutcome.current === "open" && next.outcome !== "open") {
      notify(
        next.outcome === "met"
          ? { tone: "ok", text: `Goal done: ${next.title}. The world keeps running as a sandbox.` }
          : { tone: "refused", text: `Goal lost: ${next.reason ?? "a species died out"}.` },
      );
    }
    shownOutcome.current = next?.outcome ?? null;
    setScenario(next);
  }, [notify]);

  const remember = useCallback(
    (change: (current: Seen) => Seen) => {
      setSeen((current) => {
        if (current === null || scenario === null) return current;
        const next = change(current);
        writeSeen(storage(), scenario.name, next);
        return next;
      });
    },
    [scenario],
  );

  const closeBriefing = useCallback(() => {
    setBriefing(false);
    remember((current) => ({ ...current, briefing: true }));
  }, [remember]);

  const dismissHint = useCallback((hint: HintId) => remember((current) => markDone(current, hint)), [remember]);

  const refreshHistory = useCallback(async () => {
    historyRead.current = performance.now();
    const samples = await client.history();
    const count = samples.reduce((widest, sample) => Math.max(widest, sample.species_living.length), 0);
    setSeries(new Map(Array.from({ length: count }, (_, id) => [id, speciesSeries(samples, id)])));
  }, []);

  const rename = useCallback(
    async (id: number, name: string) => {
      try {
        const named = await client.name(id, name);
        setSpecies((all) => all.map((report) => (report.id === named.id ? named : report)));
      } catch (error) {
        notify({ tone: "refused", text: `Could not rename: ${String(error)}` });
      }
    },
    [notify],
  );

  const refreshLedger = useCallback(async () => {
    const next = await client.ledger();
    if (known.current !== null) {
      for (const record of next.records.slice(known.current)) {
        notify(
          record.refusal === null
            ? { tone: "ok", text: `${label(record.name)} ${DONE[record.name] ?? "done"} for ${record.cost.toFixed(0)}.` }
            : { tone: "refused", text: `${label(record.name)} refused: ${record.refusal}` },
        );
      }
    }
    known.current = next.records.length;
    setLedger(next);
  }, [notify]);

  // -- pace --------------------------------------------------------------------------------------

  const showPace = useCallback((next: Pace) => {
    setPace(next);
    if (renderer.current !== null) renderer.current.ticksPerSecond = next.ticks_per_second;
  }, []);

  const refreshPace = useCallback(async () => {
    try {
      showPace(await client.pace());
    } catch (error) {
      notify({ tone: "refused", text: `Could not read the pace: ${String(error)}` });
    }
  }, [showPace, notify]);

  const choosePace = useCallback(
    async (ticksPerSecond: number) => {
      try {
        const next = await client.setPace(ticksPerSecond);
        showPace(next);
        if (ticksPerSecond > 0) resumeAt.current = ticksPerSecond;
        if (ticksPerSecond > next.own_ticks_per_second) dismissHint("pace");
      } catch (error) {
        notify({ tone: "refused", text: error instanceof client.Refused ? error.message : String(error) });
      }
    },
    [showPace, notify, dismissHint],
  );

  const togglePause = useCallback(() => {
    if (pace === null) return;
    if (pace.ticks_per_second > 0) void choosePace(0);
    else void choosePace(resumeAt.current ?? pace.own_ticks_per_second);
  }, [pace, choosePace]);

  // -- the world ---------------------------------------------------------------------------------

  useEffect(() => {
    let live = true;
    let stop = () => {};
    (async () => {
      const element = host.current;
      if (element === null) return;
      const map = await MapRenderer.create(element, world);
      if (!live) {
        map.destroy();
        return;
      }
      renderer.current = map;
      const [terrain, barriers, first] = await Promise.all([client.terrain(), client.barriers(), client.status()]);
      map.setTerrain(terrain);
      map.setBarriers(barriers);
      setStatus(first);
      await Promise.all([refreshFrame(), refreshLedger(), refreshSpecies(), refreshHistory(), refreshScenario()]);
      stop = client.follow((event) => {
        setProgress(event);
        setStatus(event.status);
        void refreshFrame();
        // Every tick, not only after a request: income lands at each metrics sample (#308), so
        // the balance moves on its own.
        void refreshLedger();
        // Every tick too: the counts move, and a split rides `status.splits` (#314).
        void refreshSpecies();
        void refreshScenario();
        if (performance.now() - historyRead.current > HISTORY_INTERVAL) void refreshHistory();
        if (awaitingOutcome.current) {
          awaitingOutcome.current = false;
          void client.barriers().then((view) => renderer.current?.setBarriers(view));
        }
      }, (open) => {
        setConnected(open);
        // The service returns the world to its own rate when the last stream closes (§2.4), so a
        // dropped and reopened stream may find a different pace than the one this page chose.
        if (open) void refreshPace();
      });
    })().catch((error: unknown) => setFailure(String(error)));
    return () => {
      live = false;
      stop();
      renderer.current?.destroy();
      renderer.current = null;
    };
  }, [world, refreshFrame, refreshLedger, refreshPace, refreshSpecies, refreshHistory, refreshScenario]);

  // The briefing opens by itself once, the first time this browser meets the scenario.
  const scenarioName = scenario?.name ?? null;
  useEffect(() => {
    if (scenarioName === null) return;
    const read = readSeen(storage(), scenarioName);
    setSeen(read);
    if (!read.briefing) setBriefing(true);
  }, [scenarioName]);

  const fenced = ledger?.records.some((record) => record.name === "fence" && record.refusal === null) ?? false;
  const hint =
    scenario === null || seen === null || briefing
      ? null
      : currentHint(seen, {
          fenced,
          split: species.length > 1,
          named: species.some((report) => report.name !== null),
        });

  useEffect(() => {
    void refreshFrame();
  }, [mode, refreshFrame]);

  useEffect(() => {
    paint();
  }, [selected, paint]);

  // -- pricing -----------------------------------------------------------------------------------

  const priced = pending ?? draft;
  useEffect(() => {
    setQuote(null);
    if (priced === null) return;
    let current = true;
    const timer = setTimeout(() => {
      client.quote(priced).then(
        (answer) => current && setQuote(answer),
        (error: unknown) => current && setQuote({ $type: "Quote", cost: 0, refusal: String(error) }),
      );
    }, QUOTE_DELAY);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [priced]);

  useEffect(() => {
    const map = renderer.current;
    map?.setDraft(priced === null ? null : [priced.first_x, priced.first_y, priced.second_x, priced.second_y]);
  }, [priced]);

  const cancel = useCallback(() => {
    gesture.current = null;
    setDraft(null);
    setPending(null);
  }, []);

  const build = useCallback(async () => {
    if (pending === null) return;
    try {
      await client.request(pending);
      awaitingOutcome.current = true;
      notify({ tone: "info", text: "Fence queued: it goes up at the next tick." });
    } catch (error) {
      notify({ tone: "refused", text: error instanceof client.Refused ? error.message : String(error) });
    }
    setPending(null);
    setDraft(null);
  }, [pending, notify]);

  // -- culling (#28) -------------------------------------------------------------------------------

  const quoteCull = useCallback(
    (species: number, count: number) => client.quote({ $type: "CullRequest", species_id: species, count }),
    [],
  );

  const cull = useCallback(
    async (species: number, count: number) => {
      try {
        await client.request({ $type: "CullRequest", species_id: species, count });
        awaitingOutcome.current = true;
        notify({ tone: "info", text: `Cull of ${count} queued: it happens at the next tick.` });
      } catch (error) {
        notify({ tone: "refused", text: error instanceof client.Refused ? error.message : String(error) });
      }
    },
    [notify],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // Typing a species' name is not a command.
      if (event.target instanceof HTMLInputElement) return;
      if (event.key === "Escape") cancel();
      if (event.key === "f") setTool((t) => (t === "fence" ? "look" : "fence"));
      if (event.key === " " && !(event.target instanceof HTMLButtonElement)) {
        event.preventDefault();
        togglePause();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cancel, togglePause]);

  // -- pointer -----------------------------------------------------------------------------------

  const local = (event: React.PointerEvent | React.WheelEvent): [number, number] => {
    const box = (event.currentTarget as HTMLElement).getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top];
  };

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    const map = renderer.current;
    if (map === null || pending !== null) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const [x, y] = local(event);
    gesture.current =
      tool === "fence" && event.button === 0
        ? { kind: "fence", from: toWorld(map.camera, x, y), x, y, moved: false }
        : { kind: "pan", x, y };
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const map = renderer.current;
    const current = gesture.current;
    if (map === null || current === null) return;
    const [x, y] = local(event);
    if (current.kind === "pan") {
      map.setCamera(pan(map.camera, x - current.x, y - current.y));
      gesture.current = { kind: "pan", x, y };
      return;
    }
    if (!current.moved && Math.hypot(x - current.x, y - current.y) < DRAG_THRESHOLD) return;
    current.moved = true;
    setDraft(fenceRequest(current.from, toWorld(map.camera, x, y), world.width, world.height));
  };

  const onPointerUp = () => {
    const current = gesture.current;
    gesture.current = null;
    if (current?.kind === "fence" && current.moved && draft !== null) {
      // A fence cannot be taken back once built (§3.3), so a release asks rather than builds.
      setPending(draft);
    } else if (current?.kind === "fence") {
      setDraft(null);
    }
  };

  const onWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    const map = renderer.current;
    if (map === null) return;
    const [x, y] = local(event);
    map.setCamera(zoomAt(map.camera, Math.exp(-event.deltaY * 0.0015), x, y, { min: map.fitScale() / 4, max: MAX_SCALE }));
  };

  return (
    <div className="game">
      <div
        ref={host}
        className={`map tool-${tool}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={cancel}
        onWheel={onWheel}
        onContextMenu={(event) => event.preventDefault()}
      />
      <Hud
        world={world}
        status={status}
        progress={progress}
        connected={connected}
        pace={pace}
        speeds={pace === null ? [] : speeds(pace)}
        onPace={(rate) => void choosePace(rate)}
        onTogglePause={togglePause}
        mode={mode}
        onMode={setMode}
        tool={tool}
        onTool={setTool}
        onFit={() => renderer.current?.fitView()}
        draft={draft}
        pending={pending}
        quote={quote}
        balance={ledger?.balance ?? null}
        incomePerTick={ledger?.income_per_tick ?? null}
        onBuild={build}
        onCancel={cancel}
        notices={notices}
        failure={failure}
        scenario={scenario}
        species={species}
        onBriefing={() => setBriefing(true)}
        hint={hint}
        onDismissHint={dismissHint}
      />
      {species.length > 0 && (
        <SpeciesPanel
          species={species}
          series={series}
          selected={selected}
          onSelect={setSelected}
          onRename={(id, name) => void rename(id, name)}
          guard={world.cull_survivors}
          balance={ledger?.balance ?? null}
          onQuote={quoteCull}
          onCull={cull}
          hint={hint === "species" ? <Hint text={HINT_TEXT.species} onDismiss={() => dismissHint("species")} /> : undefined}
        />
      )}
      {briefing && scenario !== null && <Briefing scenario={scenario} first={seen?.briefing === false} onClose={closeBriefing} />}
    </div>
  );
}

/** What a landed intervention did, by its recorded name: "Fence built", "Cull done". */
const DONE: Readonly<Record<string, string>> = { fence: "built", cull: "done" };

/** This browser's storage, or nothing where reading it throws (a sandboxed frame). */
function storage(): Storage | undefined {
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}

/** An intervention's recorded name for a person: "fence" reads as "Fence". */
function label(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1);
}
