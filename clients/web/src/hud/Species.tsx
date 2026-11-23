// The lineage list (#314, decision 5): every species the world has held, each under the one it split
// from, with its colour, its name, how many are alive and how that has gone. It draws what it is
// handed (§4): the counts come from the service, the tree and the scaling from pure modules.

import { useState } from "react";

import type { SpeciesReport } from "../api/values";
import { hex, speciesColor } from "../map/palette";
import { simTime } from "../world/clock";
import { lineageRows, sparkline, speciesLabel } from "../world/lineage";

/** The longest name the service accepts (api.session.NAME_LIMIT). */
const NAME_LIMIT = 40;
const SPARK_WIDTH = 64;
const SPARK_HEIGHT = 16;

interface Props {
  species: readonly SpeciesReport[];
  /** Each species' living count at every recorded sample, by species id. */
  series: ReadonlyMap<number, readonly number[]>;
  selected: number | null;
  onSelect: (species: number | null) => void;
  onRename: (species: number, name: string) => void;
}

const number = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 0 });

export function SpeciesPanel({ species, series, selected, onSelect, onRename }: Props) {
  const rows = lineageRows(species);
  const byId = new Map(species.map((report) => [report.id, report]));
  return (
    <aside className="species">
      <h2>Species</h2>
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
}) {
  const { report, parent } = props;
  const [editing, setEditing] = useState<string | null>(null);
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
      </div>
      {parent !== null && report.split_tick !== null && (
        <p className="origin">
          Split from {speciesLabel(parent)} on {simTime(report.split_tick)}, {number(report.founders ?? 0)} founders
        </p>
      )}
    </li>
  );
}
