// Every path the service answers (service/server.py's `_READS`, `_WRITES` and the event stream).
// Read by the client and by vite.config.ts, which forwards each to the service in development.
export const ROUTES = [
  "/world",
  "/status",
  "/terrain",
  "/frame",
  "/barriers",
  "/history",
  "/ledger",
  "/condition",
  "/field",
  "/entity",
  "/quote",
  "/request",
  "/events",
] as const;
