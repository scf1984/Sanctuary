// Everything a person reads over the map: where the world stands, what the colours mean, and the
// fence tool's price. It draws what it is handed and computes nothing about the world (§4).

import type { FenceRequest, Progress, Quote, Status, WorldInfo } from "../api/values";
import type { Tool } from "../Game";
import { CONDITION_RAMP, FENCE, hex } from "../map/palette";
import { simTime } from "../world/clock";

export interface Notice {
  readonly id: number;
  readonly tone: "ok" | "refused" | "info";
  readonly text: string;
}

const CONDITION_HELP: { readonly [mode: string]: string } = {
  hunger: "how short of energy each animal is",
  thirst: "how much water each animal has lost",
  age: "how far through its life each animal is",
};

interface Props {
  world: WorldInfo;
  status: Status | null;
  progress: Progress | null;
  connected: boolean;
  mode: string;
  onMode: (mode: string) => void;
  tool: Tool;
  onTool: (tool: Tool) => void;
  onFit: () => void;
  draft: FenceRequest | null;
  pending: FenceRequest | null;
  quote: Quote | null;
  balance: number | null;
  /** Currency units a tick, at the last metrics sample (#308). */
  incomePerTick: number | null;
  onBuild: () => void;
  onCancel: () => void;
  notices: readonly Notice[];
  failure: string | null;
}

/** One tick is one sim-minute (CLAUDE.md §2.1). */
const TICKS_PER_DAY = 24 * 60;

const number = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 0 });

export function Hud(props: Props) {
  const { world, status, progress, connected } = props;
  const catchingUp = progress !== null && progress.done < progress.owed;
  return (
    <>
      <header className="bar">
        <span className="title">{world.name}</span>
        {status !== null && (
          <>
            <span title={`tick ${number(status.tick)}: one tick is one sim-minute`}>{simTime(status.tick)}</span>
            <span>
              <strong>{number(status.living)}</strong> alive
            </span>
          </>
        )}
        {props.balance !== null && (
          <span title="what you have to spend on interventions">
            <strong>{number(props.balance)}</strong> to spend
            {props.incomePerTick !== null && props.incomePerTick > 0 && (
              <span
                className="income"
                title="income: paid on the fewest animals alive over the last generation, so a boom pays once it has held and a crash cuts it at once"
              >
                {" "}
                +{number(props.incomePerTick * TICKS_PER_DAY)} a day
              </span>
            )}
          </span>
        )}
        {catchingUp && (
          <span className="catching-up">
            catching up: {number(progress.done)} of {number(progress.owed)} ticks
          </span>
        )}
        <span className={`link ${connected ? "on" : "off"}`} title={connected ? "watching: the world shows each tick as it happens" : "reconnecting to the world"}>
          {connected ? "live" : "reconnecting"}
        </span>
      </header>

      {status?.collapsed && <div className="banner">This world has ended: nothing in it is alive, and nothing can be born.</div>}
      {props.failure !== null && <div className="banner failure">Lost the world: {props.failure}</div>}

      <aside className="tools">
        <div className="group">
          <button className={props.tool === "look" ? "active" : ""} onClick={() => props.onTool("look")} title="drag to pan, scroll to zoom">
            Look
          </button>
          <button className={props.tool === "fence" ? "active" : ""} onClick={() => props.onTool("fence")} title="drag a box to fence it in (F)">
            Fence
          </button>
          <button onClick={props.onFit} title="show the whole world">
            Fit
          </button>
        </div>
        {props.tool === "fence" && props.pending === null && (
          <p className="hint">
            Drag a box to fence it in. Whoever is inside stays inside. Right-drag pans.
          </p>
        )}
        {(props.draft !== null || props.pending !== null) && <Price quote={props.quote} balance={props.balance} />}
        {props.pending !== null && (
          <div className="group confirm">
            <button className="primary" disabled={props.quote === null || props.quote.refusal !== null} onClick={props.onBuild}>
              Build fence{props.quote !== null && props.quote.refusal === null ? ` for ${number(props.quote.cost)}` : ""}
            </button>
            <button onClick={props.onCancel}>Cancel</button>
          </div>
        )}
      </aside>

      <footer className="legend">
        <div className="modes">
          {world.conditions.map((mode) => (
            <button key={mode} className={props.mode === mode ? "active" : ""} onClick={() => props.onMode(mode)}>
              {mode}
            </button>
          ))}
        </div>
        <div className="ramp" aria-label={`colour shows ${CONDITION_HELP[props.mode] ?? props.mode}`}>
          {CONDITION_RAMP.map((color) => (
            <span key={color} style={{ background: hex(color) }} />
          ))}
        </div>
        <div className="ramp-labels">
          <span>wanting nothing</span>
          <span>at the edge</span>
        </div>
        <p className="help">Colour shows {CONDITION_HELP[props.mode] ?? props.mode}.</p>
        <p className="help">
          <span className="swatch" style={{ background: hex(FENCE) }} /> fence
        </p>
      </footer>

      <ul className="notices">
        {props.notices.map((notice) => (
          <li key={notice.id} className={notice.tone}>
            {notice.text}
          </li>
        ))}
      </ul>
    </>
  );
}

function Price({ quote, balance }: { quote: Quote | null; balance: number | null }) {
  if (quote === null) return <p className="price">Pricing…</p>;
  if (quote.refusal !== null) return <p className="price refused">Not possible: {quote.refusal}</p>;
  return (
    <p className="price">
      Costs <strong>{number(quote.cost)}</strong>
      {balance !== null && <> of {number(balance)}</>}. It cannot be taken down.
    </p>
  );
}
