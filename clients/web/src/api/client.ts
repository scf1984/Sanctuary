// The service's routes as typed calls. Same origin as the page (#292, decision 5), so a path is
// all a call needs.

import type {
  BarrierView,
  ConditionView,
  EntityReport,
  FieldView,
  Frame,
  Ledger,
  Progress,
  Quote,
  Request,
  Sample,
  Status,
  TerrainView,
  WorldInfo,
} from "./values";
import { expectType } from "./wire";

/** The world declined, and said why (HTTP 409, `api.Refused`). */
export class Refused extends Error {}

async function call(path: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(path, init);
  const body: unknown = await response.json();
  if (response.ok) return body;
  const message = (body as { error?: string }).error ?? response.statusText;
  if (response.status === 409) throw new Refused(message);
  throw new Error(`${path}: ${message}`);
}

function post(path: string, request: Request): Promise<unknown> {
  return call(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
}

export const world = async (): Promise<WorldInfo> => expectType(await call("/world"), "WorldInfo");
export const status = async (): Promise<Status> => expectType(await call("/status"), "Status");
export const terrain = async (): Promise<TerrainView> => expectType(await call("/terrain"), "TerrainView");
export const frame = async (): Promise<Frame> => expectType(await call("/frame"), "Frame");
export const barriers = async (): Promise<BarrierView> => expectType(await call("/barriers"), "BarrierView");
export const ledger = async (): Promise<Ledger> => expectType(await call("/ledger"), "Ledger");
export const condition = async (mode: string): Promise<ConditionView> =>
  expectType(await call(`/condition/${encodeURIComponent(mode)}`), "ConditionView");
export const field = async (layer: string): Promise<FieldView> =>
  expectType(await call(`/field/${encodeURIComponent(layer)}`), "FieldView");

export async function history(): Promise<readonly Sample[]> {
  const samples = await call("/history");
  if (!Array.isArray(samples)) throw new TypeError("expected a list of Samples");
  return samples.map((sample) => expectType(sample, "Sample"));
}

/** One animal, or null if it has died (ids are never reused, so null is final). */
export async function inspect(id: number): Promise<EntityReport | null> {
  const report = await call(`/entity/${id}`);
  return report === null ? null : expectType(report, "EntityReport");
}

/** What `request` would cost, and whether it would be refused, as the world stands now. */
export const quote = async (request: Request): Promise<Quote> => expectType(await post("/quote", request), "Quote");

/** Queue `request` for the next tick boundary. Its outcome arrives in the ledger (§2.7). */
export async function request(request: Request): Promise<void> {
  await post("/request", request);
}

/** Follow the event stream. Opening it is what makes the world "watched" (#36). The browser
 * reconnects by itself after a drop; `onState` says which state the stream is in. */
export function follow(onProgress: (progress: Progress) => void, onState: (open: boolean) => void): () => void {
  const source = new EventSource("/events");
  source.addEventListener("progress", (event) => {
    onProgress(expectType(JSON.parse((event as MessageEvent<string>).data), "Progress"));
  });
  source.onopen = () => onState(true);
  source.onerror = () => onState(false);
  return () => source.close();
}
