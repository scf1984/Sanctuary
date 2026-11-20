import { useEffect, useState } from "react";

import * as client from "./api/client";
import type { WorldInfo } from "./api/values";
import { Game } from "./Game";

export function App() {
  const [world, setWorld] = useState<WorldInfo | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => {
    client.world().then(setWorld, (error: unknown) => setFailure(String(error)));
  }, []);

  if (failure !== null) {
    return (
      <div className="splash">
        <h1>Sanctuary</h1>
        <p>Could not reach the world: {failure}</p>
        <p>Is <code>python -m service</code> running?</p>
      </div>
    );
  }
  if (world === null) {
    return (
      <div className="splash">
        <h1>Sanctuary</h1>
        <p>Opening the world…</p>
      </div>
    );
  }
  return <Game world={world} />;
}
