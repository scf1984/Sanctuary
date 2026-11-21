// One world on screen: the map, the clock, and the fence tool. State that changes every tick (the
// animals) goes straight to the renderer; what a person reads (status, price, outcomes) is React's.

import { useCallback, useEffect, useRef, useState } from "react";

import * as client from "./api/client";
import type { FenceRequest, Ledger, Progress, Quote, Status, WorldInfo } from "./api/values";
import { decodeColumn } from "./api/wire";
import { Hud, type Notice } from "./hud/Hud";
import { pan, toWorld, zoomAt } from "./map/camera";
import { fenceRequest } from "./map/fence";
import { MapRenderer } from "./map/renderer";
import { positions } from "./world/frames";

export type Tool = "look" | "fence";

/** How far the map may zoom in, pixels per world unit. Out, it stops at a quarter of the fit. */
const MAX_SCALE = 64;
/** A press that moves less than this many pixels is a click, not a drag. */
const DRAG_THRESHOLD = 4;
/** Milliseconds a drag must rest before it is priced, so a quote is not asked for every pixel. */
const QUOTE_DELAY = 120;

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

  // A request's outcome lands in the ledger at the next tick boundary (§2.7), so after asking, the
  // next progress event is when to look. `known` is how many records the player has been shown.
  const awaitingOutcome = useRef(false);
  const known = useRef<number | null>(null);
  const modeRef = useRef(mode);
  modeRef.current = mode;

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
        // Frame and condition are two calls, so a tick can land between them. A condition is only
        // drawn against the frame of its own tick (api.values.ConditionView); otherwise ask again.
        let [frame, condition] = await Promise.all([client.frame(), client.condition(modeRef.current)]);
        if (condition.tick !== frame.tick) {
          [frame, condition] = await Promise.all([client.frame(), client.condition(modeRef.current)]);
        }
        const deficits = condition.tick === frame.tick ? decodeColumn(condition.values).values : null;
        renderer.current?.setFrame(positions(frame), deficits instanceof Float32Array ? deficits : null, performance.now());
      } while (again.current);
    } catch (error) {
      setFailure(String(error));
    } finally {
      fetching.current = false;
    }
  }, []);

  const refreshLedger = useCallback(async () => {
    const next = await client.ledger();
    if (known.current !== null) {
      for (const record of next.records.slice(known.current)) {
        notify(
          record.refusal === null
            ? { tone: "ok", text: `${label(record.name)} built for ${record.cost.toFixed(0)}.` }
            : { tone: "refused", text: `${label(record.name)} refused: ${record.refusal}` },
        );
      }
    }
    known.current = next.records.length;
    setLedger(next);
  }, [notify]);

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
      await Promise.all([refreshFrame(), refreshLedger()]);
      stop = client.follow((event) => {
        setProgress(event);
        setStatus(event.status);
        void refreshFrame();
        // Every tick, not only after a request: income lands at each metrics sample (#308), so
        // the balance moves on its own.
        void refreshLedger();
        if (awaitingOutcome.current) {
          awaitingOutcome.current = false;
          void client.barriers().then((view) => renderer.current?.setBarriers(view));
        }
      }, setConnected);
    })().catch((error: unknown) => setFailure(String(error)));
    return () => {
      live = false;
      stop();
      renderer.current?.destroy();
      renderer.current = null;
    };
  }, [world, refreshFrame, refreshLedger]);

  useEffect(() => {
    void refreshFrame();
  }, [mode, refreshFrame]);

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

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") cancel();
      if (event.key === "f") setTool((t) => (t === "fence" ? "look" : "fence"));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cancel]);

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
      />
    </div>
  );
}

/** An intervention's recorded name for a person: "fence" reads as "Fence". */
function label(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1);
}
