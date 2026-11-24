// The lineage list (#314, decision 5): every species the world has held, each under the one it split
// from, with its colour, its name, how many are alive and how that has gone. It draws what it is
// handed (§4): the counts come from the service, the tree and the scaling from pure modules.
//
// It is also where a cull is asked for (#28): a cull thins one species, chosen at random, so the
// species' own row is the control. Like a fence, it shows its price before it commits.

import { useEffect, useState, type ReactNode } from "react";

import type { Quote, SpeciesReport } from "../api/values";
import { hex, speciesColor } from "../map/palette";
import { simTime } from "../world/clock";
import { clampCount, cullable, suggestedCount } from "../world/cull";
import { lineageRows, sparkline, speciesLabel } from "../world/lineage";

/** The longest name the service accepts (api.session.NAME_LIMIT). */
const NAME_LIMIT = 40;
const SPARK_WIDTH = 64;
const SPARK_HEIGHT = 16;
/** Milliseconds a typed count must rest before it is priced, as a fence drag must (Game.tsx). */
const QUOTE_DELAY = 150;

interface Props {
  species: readonly SpeciesReport[];
  /** Each species' living count at every recorded sample, by species id. */
  series: ReadonlyMap<number, readonly number[]>;
  selected: number | null;
  onSelect: (species: number | null) => void;
  onRename: (species: number, name: string) => void;
  /** The fewest born animals a cull may leave of a species: the world's guard (#28). */
  guard: number;
  balance: number | null;
  onQuote: (species: number, count: number) => Promise<Quote>;
  onCull: (species: number, count: number) => Promise<void>;
  /** An onboarding hint anchored to this list (#317), or nothing. */
  hint?: ReactNode;
}

const number = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 0 });

export function SpeciesPanel({ species, series, selected, onSelect, onRename, guard, balance, onQuote, onCull, hint }: Props) {
  const rows = lineageRows(species);
  const byId = new Map(species.map((report) => [report.id, report]));
  return (
    <aside className="species">
      <h2>Species</h2>
      {hint}
      {species.length === 1 && (
        <p className="help">
          One species so far. Fence a group off and wait: isolated long enough, it drifts apart and splits into a new one.
        </p>
      )}
      <ul>
        {rows.map(({ species: report, depth }) => (
          <SpeciesRow
            key={report.id}
            report={report}
            depth={depth}
            parent={report.parent === null ? null : (byId.get(report.parent) ?? null)}
            series={series.get(report.id) ?? []}
            selected={selected === report.id}
            onSelect={() => onSelect(selected === report.id ? null : report.id)}
            onRename={(name) => onRename(report.id, name)}
            guard={guard}
            balance={balance}
            onQuote={(count) => onQuote(report.id, count)}
            onCull={(count) => onCull(report.id, count)}
          />
        ))}
      </ul>
      {selected !== null && (
        <button className="clear" onClick={() => onSelect(null)}>
          Show every species
        </button>
      )}
    </aside>
  );
}

function SpeciesRow(props: {
  report: SpeciesReport;
  depth: number;
  parent: SpeciesReport | null;
  series: readonly number[];
  selected: boolean;
  onSelect: () => void;
  onRename: (name: string) => void;
  guard: number;
  balance: number | null;
  onQuote: (count: number) => Promise<Quote>;
  onCull: (count: number) => Promise<void>;
}) {
  const { report, parent } = props;
  const [editing, setEditing] = useState<string | null>(null);
  const [culling, setCulling] = useState(false);
  const most = cullable(report.living, props.guard);
  const extinct = report.living === 0;
  const unnamed = report.name === null && parent !== null;

  const commit = () => {
    if (editing !== null && editing.trim() !== "" && editing.trim() !== report.name) props.onRename(editing);
    setEditing(null);
  };

  return (
    <li
      className={`${props.selected ? "selected" : ""} ${extinct ? "extinct" : ""}`}
      style={{ paddingLeft: `${8 + props.depth * 14}px` }}
    >
      <div className="line">
        <button
          className="pick"
          onClick={props.onSelect}
          title={props.selected ? "show every species" : "pick out this species on the map"}
          aria-pressed={props.selected}
        >
          <span className="dot" style={{ background: hex(speciesColor(report.id)) }} />
        </button>
        {editing === null ? (
          <button className="name" onClick={() => setEditing(report.name ?? "")} title="rename">
            {speciesLabel(report)}
            {unnamed && <span className="new">name it</span>}
          </button>
        ) : (
          <input
            autoFocus
            maxLength={NAME_LIMIT}
            value={editing}
            placeholder={speciesLabel(report)}
            onChange={(event) => setEditing(event.target.value)}
            onBlur={commit}
            onKeyDown={(event) => {
              if (event.key === "Enter") commit();
              if (event.key === "Escape") setEditing(null);
            }}
            aria-label={`name for ${speciesLabel(report)}`}
          />
        )}
        <span className="count" title={extinct ? "extinct: none alive, and none can be born" : "alive now"}>
          {extinct ? "extinct" : number(report.living)}
        </span>
        {props.series.length > 1 && (
          <svg className="spark" width={SPARK_WIDTH} height={SPARK_HEIGHT} aria-label="living over the recorded history">
            <polyline points={sparkline(props.series, SPARK_WIDTH, SPARK_HEIGHT)} stroke={hex(speciesColor(report.id))} />
          </svg>
        )}
        {!extinct && (
          <button
            className={`cull-open ${culling ? "active" : ""}`}
            disabled={most === 0 && !culling}
            onClick={() => setCulling((open) => !open)}
            title={
              most === 0
                ? `At the guard: a cull must leave at least ${props.guard}, so the species can regrow`
                : "remove some of this species"
            }
          >
            cull
          </button>
        )}
      </div>
      {parent !== null && report.split_tick !== null && (
        <p className="origin">
          Split from {speciesLabel(parent)} on {simTime(report.split_tick)}, {number(report.founders ?? 0)} founders
        </p>
      )}
      {culling && !extinct && (
        <CullForm
          report={report}
          guard={props.guard}
          balance={props.balance}
          onQuote={props.onQuote}
          onCull={async (count) => {
            await props.onCull(count);
            setCulling(false);
          }}
          onClose={() => setCulling(false)}
        />
      )}
    </li>
  );
}

/** How many to take, what it costs, and a confirm. Nothing is asked of the world until the confirm,
 * and the outcome is announced when it lands at the next tick, as a fence's is. */
function CullForm(props: {
  report: SpeciesReport;
  guard: number;
  balance: number | null;
  onQuote: (count: number) => Promise<Quote>;
  onCull: (count: number) => Promise<void>;
  onClose: () => void;
}) {
  const { report, guard, onQuote } = props;
  const [count, setCount] = useState(() => suggestedCount(report.living, guard));
  const [quote, setQuote] = useState<Quote | null>(null);
  const [sending, setSending] = useState(false);
  const most = cullable(report.living, guard);
  // The species may shrink while the form is open; the count follows it down.
  const asked = clampCount(count, report.living, guard);

  useEffect(() => {
    setQuote(null);
    if (asked === 0) return;
    let current = true;
    const timer = setTimeout(() => {
      onQuote(asked).then(
        (answer) => current && setQuote(answer),
        (error: unknown) => current && setQuote({ $type: "Quote", cost: 0, refusal: String(error) }),
      );
    }, QUOTE_DELAY);
    return () => {
      current = false;
      clearTimeout(timer);
    };
    // Keyed on the count alone: `onQuote` is a fresh closure on every render of the panel, and
    // re-pricing on each would ask the service again every tick for the same answer.
  }, [asked]);

  if (most === 0) {
    return (
      <div className="cull">
        <p className="help">
          Only {number(report.living)} left. A cull must leave at least {number(guard)}, so the species can always regrow.
        </p>
        <button onClick={props.onClose}>Close</button>
      </div>
    );
  }

  const ready = quote !== null && quote.refusal === null && !sending;
  return (
    <form
      className="cull"
      onSubmit={(event) => {
        event.preventDefault();
        if (!ready) return;
        setSending(true);
        void props.onCull(asked).finally(() => setSending(false));
      }}
    >
      <label>
        Remove
        <input
          type="number"
          min={1}
          max={most}
          step={1}
          value={count}
          onChange={(event) => setCount(event.target.valueAsNumber)}
          onBlur={() => setCount(asked)}
          aria-label={`how many of ${speciesLabel(report)} to remove`}
        />
        of {number(report.living)}, chosen at random
      </label>
      <p className="help">
        At most {number(most)}: a cull always leaves at least {number(guard)} so the species can regrow. The bodies are left
        where they fall, for scavengers, and rot back into the soil.
      </p>
      {quote === null ? (
        <p className="price">Pricing…</p>
      ) : quote.refusal !== null ? (
        <p className="price refused">Not possible: {quote.refusal}</p>
      ) : (
        <p className="price">
          Costs <strong>{number(quote.cost)}</strong>
          {props.balance !== null && <> of {number(props.balance)}</>}. The dead do not come back.
        </p>
      )}
      <div className="actions">
        <button type="submit" className="primary" disabled={!ready}>
          Cull {number(asked)}
          {quote !== null && quote.refusal === null ? ` for ${number(quote.cost)}` : ""}
        </button>
        <button type="button" onClick={props.onClose}>
          Cancel
        </button>
      </div>
    </form>
  );
}
