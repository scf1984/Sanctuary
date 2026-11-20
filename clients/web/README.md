# Sanctuary game client

The browser client for the playable slice (#292, #306). It talks to `python -m service` over HTTP
and never reads a file of the Python tree. Design decisions are in `CLAUDE.md` §3.3.

## Play

```sh
cd clients/web
npm ci
npm run build
cd ../..
python -m service --state saves/world1
```

Then open http://127.0.0.1:8765. The service serves this build at `/`.

## Develop

Run the service as above, then in another terminal:

```sh
cd clients/web
npm run dev
```

Vite serves the client with hot reload and forwards API routes to the service on port 8765
(`SANCTUARY_SERVICE` overrides that).

## Checks

```sh
npm run typecheck
npm test
```

`src/api/values.ts` is generated. After changing `api/values.py`, run
`python -m api.typescript --write` from the repository root and commit the result. CI fails
while it is stale.
