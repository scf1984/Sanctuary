import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { SpeciesReport } from "../api/values";
import { SpeciesPanel } from "./Species";

const species = (id: number, parent: number | null, name: string | null = null): SpeciesReport => ({
  $type: "SpeciesReport",
  id,
  name,
  parent,
  split_tick: parent === null ? null : 100 * id,
  founders: parent === null ? null : 20,
  living: 10,
});

const panel = (reports: SpeciesReport[]) => {
  const html = renderToStaticMarkup(
    <SpeciesPanel
      species={reports}
      series={new Map()}
      selected={null}
      onSelect={() => {}}
      onRename={() => {}}
      guard={6}
      balance={null}
      onQuote={() => Promise.reject(new Error("not asked"))}
      onCull={() => Promise.resolve()}
    />,
  );
  // No DOM in the test environment, so the markup is read as text. A button's contents hold no
  // nested button, so a lazy match to its closing tag is exactly one button.
  const buttons = [...html.matchAll(/<button([^>]*)>(.*?)<\/button>/g)].map(([, attributes = "", inner = ""]) => ({
    clipped: /class="name"/.test(attributes),
    title: /title="([^"]*)"/.exec(attributes)?.[1] ?? "",
    text: inner.replace(/<[^>]*>/g, ""),
  }));
  return { html, buttons };
};

describe("SpeciesPanel", () => {
  // #322: the name button clips its overflow, so anything inside it past the name is cut off. The
  // species hint tells the player to press "name it", so it must be a control of its own.
  it("offers 'name it' as its own control, outside the clipped name", () => {
    const { buttons } = panel([species(0, null), species(1, 0), species(2, 1)]);

    expect(buttons.filter((b) => !b.clipped && b.text === "name it")).toHaveLength(2);
    expect(buttons.filter((b) => b.clipped).map((b) => b.text)).toEqual(["Species 1", "Species 2", "Species 3"]);
  });

  // #325: a deep row leaves the name a few dozen pixels, so its tooltip is where a cut-off name
  // can still be read without opening the rename field.
  it("names the species in the clipped name's tooltip", () => {
    const { buttons } = panel([species(0, null), species(1, 0, "Ridge runners of the north")]);

    expect(buttons.filter((b) => b.clipped).map((b) => b.title)).toEqual([
      "Species 1, click to rename",
      "Ridge runners of the north, click to rename",
    ]);
  });

  it("does not offer a name for the founding species or one already named", () => {
    const { html } = panel([species(0, null), species(1, 0, "Ridge runners")]);

    expect(html).not.toContain("name it");
  });
});
