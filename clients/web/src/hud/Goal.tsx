// The scenario's goal and its briefing (#317). Both draw what the service hands over: the words are
// the scenario's own (worlds/scenarios.py) and where the goal stands is metrics.goals' reading.

import type { ScenarioReport, SpeciesReport } from "../api/values";
import { goalStatus, progressShare } from "../world/goal";

export function GoalPanel(props: {
  scenario: ScenarioReport;
  species: readonly SpeciesReport[];
  onBriefing: () => void;
}) {
  const { scenario } = props;
  const share = progressShare(scenario);
  const beforeSplit = scenario.daughter === null && scenario.outcome === "open";
  return (
    <section className={`goal ${scenario.outcome}`} aria-label="goal">
      <h2>
        {scenario.title}
        {scenario.outcome !== "open" && <span className="outcome">{scenario.outcome === "met" ? "done" : "lost"}</span>}
      </h2>
      <p className="goal-text">{scenario.goal_text}</p>
      {beforeSplit && (
        <div
          className="meter"
          role="meter"
          aria-label="how far apart the two most different groups have drifted"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={share === null ? undefined : Math.round(share * 100)}
          title="Checked about once a generation. Even an unfenced herd reads a quarter to a half, and a pen sits there for most of the wait; the reading jumps once the fenced group stands out, a few generations before the split."
        >
          <span style={{ width: `${(share ?? 0) * 100}%` }} />
        </div>
      )}
      <p className="status">{goalStatus(scenario, props.species)}</p>
      <button className="link-button" onClick={props.onBriefing}>
        Read the briefing again
      </button>
    </section>
  );
}

/** `first` is the briefing opening by itself, before the game; otherwise the player asked to read it again. */
export function Briefing(props: { scenario: ScenarioReport; first: boolean; onClose: () => void }) {
  const { scenario } = props;
  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-labelledby="briefing-title">
      <div className="briefing">
        <h1 id="briefing-title">{scenario.title}</h1>
        <p className="goal-text">{scenario.goal_text}</p>
        {scenario.briefing.map((paragraph) => (
          <p key={paragraph}>{paragraph}</p>
        ))}
        <div className="actions">
          <button className="primary" onClick={props.onClose} autoFocus>
            {props.first ? "Start" : "Back to the world"}
          </button>
        </div>
      </div>
    </div>
  );
}

/** A hint anchored to the control it describes, with a way to put it away for good. */
export function Hint(props: { text: string; onDismiss: () => void; className?: string }) {
  return (
    <div className={`hint-bubble ${props.className ?? ""}`} role="note">
      <span>{props.text}</span>
      <button className="dismiss" onClick={props.onDismiss} aria-label="dismiss this hint" title="dismiss">
        ×
      </button>
    </div>
  );
}

/** What each hint says. Written beside the controls they point at, not with the scenario, since
 * they describe this client's buttons rather than the world. */
export const HINT_TEXT = {
  fence: "Start here: press Fence, then drag a box around some animals. Give them grass and water.",
  pace: "A split takes hours. Fast-forward while you watch, or close the page: the world keeps running.",
  species: "A new species has split off. Press “name it” beside it in this list to give it a name.",
} as const;
