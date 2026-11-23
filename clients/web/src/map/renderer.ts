// The map, drawn by PixiJS on WebGL (#292, decision 7). Imperative on purpose: a frame changes
// thousands of dots and React would re-render nothing useful for it, so React owns the panels and
// this owns the canvas. Everything it decides about geometry lives in pure modules beside it
// (camera, fence, terrain, palette, ../world/frames), which are tested without a GPU.

import { Application, Container, Graphics, Particle, ParticleContainer, Sprite, Texture } from "pixi.js";

import type { BarrierView, TerrainView, WorldInfo } from "../api/values";
import { blendInto, pair, type Pairing, type Positions } from "../world/frames";
import { blendPeriod } from "../world/pace";
import { type Camera, fit, TOP_INSET, toScreen } from "./camera";
import { type Tints, UNKNOWN } from "./colours";
import { fenceSegments, type Segment } from "./fence";
import { CASING, FENCE } from "./palette";
import { terrainPixels } from "./terrain";

/** Pixels: a dot's fill radius, and the casing ring's width around it. */
const DOT_RADIUS = 2.5;
const CASING_WIDTH = 1;

function disc(radius: number): Texture {
  const size = Math.ceil(radius * 2) + 2;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (context === null) throw new Error("no 2D context to draw a dot texture with");
  context.fillStyle = "#ffffff";
  context.beginPath();
  context.arc(size / 2, size / 2, radius, 0, Math.PI * 2);
  context.fill();
  return Texture.from(canvas);
}

export class MapRenderer {
  readonly app: Application;
  private readonly world: WorldInfo;
  private readonly ground = new Container();
  private readonly fences = new Graphics();
  private readonly draft = new Graphics();
  private readonly casing: ParticleContainer;
  private readonly fill: ParticleContainer;
  private readonly casingTexture = disc(DOT_RADIUS + CASING_WIDTH);
  private readonly fillTexture = disc(DOT_RADIUS);
  private segments: Segment[] = [];
  private draftRect: readonly [number, number, number, number] | null = null;
  private pairing: Pairing | null = null;
  /** Blended positions, reused across drawn frames rather than allocated per frame. */
  private x = new Float32Array(0);
  private y = new Float32Array(0);
  private arrived = 0;
  /** Milliseconds between the last two frames: how long a blend from one to the next takes. */
  private period = 1000;
  /** The world's rate, which bounds how long a blend may take (`blendPeriod`). */
  ticksPerSecond = 1;
  camera: Camera;

  private constructor(app: Application, world: WorldInfo) {
    this.app = app;
    this.world = world;
    this.camera = fit(world, { width: app.screen.width, height: app.screen.height }, TOP_INSET);
    // The ring's colour moves too: a selected species dims everyone else's ring with their fill.
    this.casing = new ParticleContainer({ dynamicProperties: { position: true, color: true } });
    this.fill = new ParticleContainer({ dynamicProperties: { position: true, color: true } });
    app.stage.addChild(this.ground, this.fences, this.casing, this.fill, this.draft);
    app.ticker.add(() => this.draw());
  }

  static async create(host: HTMLElement, world: WorldInfo): Promise<MapRenderer> {
    const app = new Application();
    await app.init({ resizeTo: host, background: 0x1d1f1c, antialias: true, preference: "webgl" });
    host.appendChild(app.canvas);
    return new MapRenderer(app, world);
  }

  destroy(): void {
    this.app.destroy(true, { children: true, texture: true });
  }

  /** The whole world, centred, at the current size. */
  fitView(): void {
    this.setCamera(fit(this.world, { width: this.app.screen.width, height: this.app.screen.height }, TOP_INSET));
  }

  /** The scale at which the world fits, pixels per world unit: the zoom bounds are read off it. */
  fitScale(): number {
    return fit(this.world, { width: this.app.screen.width, height: this.app.screen.height }, TOP_INSET).scale;
  }

  setCamera(camera: Camera): void {
    this.camera = camera;
    this.ground.position.set(-camera.left * camera.scale, -camera.top * camera.scale);
    this.ground.scale.set(camera.scale);
    this.drawFences();
    this.drawDraft();
  }

  setTerrain(view: TerrainView): void {
    const { rows, cols, rgba } = terrainPixels(view);
    const canvas = document.createElement("canvas");
    canvas.width = cols;
    canvas.height = rows;
    const context = canvas.getContext("2d");
    if (context === null) throw new Error("no 2D context to draw the terrain with");
    context.putImageData(new ImageData(new Uint8ClampedArray(rgba), cols, rows), 0, 0);
    const texture = Texture.from(canvas);
    // One texel is one cell, so neighbouring cells must not be smeared into each other.
    texture.source.scaleMode = "nearest";
    const sprite = new Sprite(texture);
    sprite.scale.set(this.world.cell_size);
    this.ground.removeChildren();
    this.ground.addChild(sprite);
    this.setCamera(this.camera);
  }

  setBarriers(view: BarrierView): void {
    this.segments = fenceSegments(view, this.world.cell_size);
    this.drawFences();
  }

  /** A new frame to blend toward, with each animal's colours (`dotTints`), or null while they
   * cannot be known. The same tick again only repaints, which is how a new selection is shown. */
  setFrame(frame: Positions, tints: Tints | null, now: number): void {
    const latest = this.pairing?.next ?? null;
    if (latest !== null && frame.tick === latest.tick) {
      if (tints !== null) this.colour(tints);
      return;
    }
    if (latest !== null) {
      this.period = blendPeriod(now - this.arrived, frame.tick - latest.tick, this.ticksPerSecond);
    }
    this.pairing = pair(latest, frame);
    this.arrived = now;
    this.resize(frame.ids.length);
    if (this.x.length < frame.ids.length) {
      this.x = new Float32Array(frame.ids.length);
      this.y = new Float32Array(frame.ids.length);
    }
    // A new frame lists different animals in each slot, so last tick's colours would be someone
    // else's: without this tick's colours, every dot goes grey.
    this.colour(tints);
  }

  /** Colours change once a tick, so they are set when a frame arrives, not on every drawn frame. */
  private colour(tints: Tints | null): void {
    const fill = this.fill.particleChildren;
    const ring = this.casing.particleChildren;
    for (let i = 0; i < fill.length; i++) {
      (fill[i] as Particle).tint = tints === null ? UNKNOWN : (tints.fill[i] as number);
      (ring[i] as Particle).tint = tints === null ? CASING : (tints.ring[i] as number);
    }
  }

  /** The fence being dragged, in world units, or null. */
  setDraft(rect: readonly [number, number, number, number] | null): void {
    this.draftRect = rect;
    this.drawDraft();
  }

  private resize(count: number): void {
    for (const [container, texture] of [
      [this.casing, this.casingTexture],
      [this.fill, this.fillTexture],
    ] as const) {
      const particles = container.particleChildren;
      while (particles.length < count) {
        particles.push(new Particle({ texture, anchorX: 0.5, anchorY: 0.5, tint: container === this.casing ? CASING : UNKNOWN }));
      }
      particles.length = count;
      container.update();
    }
  }

  private draw(): void {
    if (this.pairing === null) return;
    const { x, y } = this;
    blendInto(this.pairing, (performance.now() - this.arrived) / this.period, x, y);
    const casing = this.casing.particleChildren;
    const fill = this.fill.particleChildren;
    const { left, top, scale } = this.camera;
    for (let i = 0; i < fill.length; i++) {
      // `toScreen`, inlined: this loop runs for every animal on every drawn frame.
      const px = ((x[i] as number) - left) * scale;
      const py = ((y[i] as number) - top) * scale;
      const ring = casing[i] as Particle;
      const dot = fill[i] as Particle;
      ring.x = dot.x = px;
      ring.y = dot.y = py;
    }
  }

  private drawFences(): void {
    this.fences.clear();
    if (this.segments.length === 0) return;
    for (const [x0, y0, x1, y1] of this.segments) {
      const [a, b] = toScreen(this.camera, x0, y0);
      const [c, d] = toScreen(this.camera, x1, y1);
      this.fences.moveTo(a, b).lineTo(c, d);
    }
    this.fences.stroke({ width: 3, color: FENCE, cap: "square" });
  }

  private drawDraft(): void {
    this.draft.clear();
    if (this.draftRect === null) return;
    const [x0, y0, x1, y1] = this.draftRect;
    const [a, b] = toScreen(this.camera, Math.min(x0, x1), Math.min(y0, y1));
    const [c, d] = toScreen(this.camera, Math.max(x0, x1), Math.max(y0, y1));
    this.draft.rect(a, b, c - a, d - b).fill({ color: FENCE, alpha: 0.12 }).stroke({ width: 2, color: FENCE });
  }
}
