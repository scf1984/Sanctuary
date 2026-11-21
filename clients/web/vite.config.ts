import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import { ROUTES } from "./src/api/routes.ts";

// `npm run dev` serves the client with hot reload and forwards every API route to a running
// `python -m service`, so development needs no CORS either (#292, decision 5). The route list is
// the client's own (src/api/routes.ts), so a route added there is forwarded without editing this.
const SERVICE = process.env.SANCTUARY_SERVICE ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(ROUTES.map((route) => [route, { target: SERVICE, changeOrigin: false }])),
  },
  // PixiJS is most of the bundle (~500 kB), and the game is served locally, so one chunk is fine.
  build: { outDir: "dist", emptyOutDir: true, sourcemap: true, chunkSizeWarningLimit: 800 },
});
