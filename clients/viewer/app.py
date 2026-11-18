"""Thin pygame wiring for the diagnostic world view (CLAUDE.md §3.3).

All the logic worth testing lives in render.py, charts.py, export.py and playback.py, every one of
them pygame-free; this module owns only the window, the event loop, and one call into each of those
per frame. That split is load-bearing rather than tidy: this module is uncollectable in CI, which
never installs the viewer extra, so anything that lives here is untestable by construction — which
is how the world builder's `EntityStore` call stayed broken through a whole release (#110).
Rendering technology: pygame (CLAUDE.md §3.3), chosen over matplotlib because pause/step/speed
need a real per-frame event loop and immediate-mode blitting rather than a plotting library's
redraw-the-whole-figure model.

**Everything it knows about the world comes through `api`** (#35). It holds a session and the
values the session hands back, and nothing else, which is what lets the same view run against a
world in another process: `--connect URL` opens one the service is serving (#36).

**A served world is watched, never driven.** Over the protocol the view holds the world's event
stream open, which is what makes the service wake on its short interval (gate 2 on #35), and it
fetches a frame each time the stream says the world moved. Pause, step and speed do nothing,
because the world isn't the view's to run (#32). Everything else is the same code path.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

import api
from clients.viewer.charts import stack_charts, world_charts
from clients.viewer.export import export_history
from clients.viewer.playback import Playback
from clients.viewer.render import (
    CONDITION_MODES,
    FIELD_LAYERS,
    apply_water_overlay,
    barrier_segments,
    casing_color,
    condition_colors,
    describe_entity,
    drag_rectangle,
    elevation_shading,
    field_overlay,
    live_positions,
    pick_entity,
    screen_to_world,
    species_colors,
    world_to_screen,
)

_SCREEN_SIZE = (900, 900)
_ENTITY_RADIUS = 4

# World units within which a click selects an animal. A little larger than an entity looks,
# because the target is a few pixels across and a diagnostic tool that demands precision is a
# diagnostic tool nobody uses.
_PICK_RADIUS = 3.0
# The one hue nothing else in this view uses: terrain is green and brown, water blue, the condition
# ramp orange, carrion red. A fence is the player's own mark on the world and has to be unmistakably
# theirs rather than something the ecology did.
_FENCE_COLOR = (74, 58, 167)
_PANEL_BACKGROUND = (12, 12, 14)
_PANEL_TEXT = (232, 232, 226)

# The chart panel: wide enough that a few hundred samples are one pixel column apart, and short
# enough that four stacked charts leave most of the world visible. It sits bottom-left, away from
# the inspection panel at the top.
_CHART_SIZE = (320, 54)
_EXPORT_DIRECTORY = Path("exports")
# One slot per world and seed. A save browser is a UI this diagnostic instrument does not need
# (§3.3), and a file per slot is what stops a second run of the same world silently overwriting the
# first's. The world name is part of it because the demo and temperate worlds (#268) share seeds.
_SAVE_DIRECTORY = Path("saves")

# What one keypress removes, and the floor it will not go below. Both are the viewer's choice
# rather than the world's: §2.7 forbids eradicating a species by a single action and `Cull` is what
# enforces that, so this is only how large a bite this particular button takes.
_CULL_SIZE = 10
_CULL_SURVIVORS = 20

# What the overlay key cycles through. `None` is terrain alone, and it leads so that the view
# opens on the same picture it always has.
_LAYER_CYCLE: tuple[str | None, ...] = (None, *FIELD_LAYERS)


def _scaled_surface(rgb, screen_size: tuple[int, int]) -> pygame.Surface:
    """A window-sized surface from a (height, width, 3) uint8 grid."""
    # pygame's surfarray convention is (width, height, 3); ours is (height, width, 3).
    surface = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    return pygame.transform.smoothscale(surface, screen_size)


def _draw_panel(screen, font, lines: tuple[str, ...]) -> None:
    """Blit an inspection readout in the corner. Layout only — every decision about *what* the
    numbers mean lives in `render.describe_entity`, which CI can actually collect."""
    rendered = [font.render(line, True, _PANEL_TEXT) for line in lines]
    width = max((surface.get_width() for surface in rendered), default=0) + 16
    height = sum(surface.get_height() for surface in rendered) + 16
    panel = pygame.Surface((width, height))
    panel.set_alpha(225)
    panel.fill(_PANEL_BACKGROUND)
    screen.blit(panel, (8, 28))
    y = 36
    for surface in rendered:
        screen.blit(surface, (16, y))
        y += surface.get_height()


def _draw_charts(screen, font, samples, gene, gene_names) -> None:
    """Blit the metric panel bottom-left. Layout only — `charts.py` decides where a point goes,
    because that module is collectable in CI and this one is not (#110)."""
    charts = world_charts(samples, gene, gene_names)
    width, chart_height = _CHART_SIZE
    image, labels = stack_charts(charts, width, chart_height)
    if not image.size:
        return
    top = screen.get_height() - image.shape[0] - 8
    screen.blit(pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2))), (8, top))
    for index, (label, detail, color) in enumerate(labels):
        y = top + index * (chart_height + 7)
        screen.blit(font.render(label, True, color), (width + 16, y))
        screen.blit(font.render(detail, True, _PANEL_TEXT), (width + 16, y + 14))


def _cull(session: api.Session, entity_id: int) -> str:
    """Queue a cull of the selected animal's species, and say what will be attempted.

    Only *queued* — it lands at the next tick boundary and may be refused there (§2.7's rule, or an
    empty budget), which is why the outcome is read back out of the ledger rather than reported
    here. Saying "done" now would be reporting something that has not happened.
    """
    report = session.inspect(entity_id)
    if report is None:
        return f"entity {entity_id} is gone, so there is no species to cull"
    session.request(
        api.CullRequest(species_id=report.species, count=_CULL_SIZE, survivors=_CULL_SURVIVORS)
    )
    return f"cull of {_CULL_SIZE} from species {report.species} queued"


def _last_outcome(ledger: api.Ledger) -> str:
    """The most recent intervention's result, for the status line. Empty until one has run.

    The status line is the only place a viewer can tell the player anything, and a refusal that
    reached nobody is the failure §2.7's whole design exists to avoid.
    """
    if not ledger.records:
        return ""
    last = ledger.records[-1]
    return f"{last.name}: {last.refusal or f'done, {ledger.balance:g} left'}"


def _drag_to_world(info: api.WorldInfo, start, end) -> tuple[float, float, float, float]:
    """The two screen corners of a drag, as world-unit corners for a fence.

    Clamped into the world, because a drag that leaves the window is an ordinary thing a player
    does and the fence normalises the corners itself — so the clamp is about staying inside the map,
    not about which corner came first.
    """
    xs, ys = screen_to_world(
        np.array([start[0], end[0]], dtype=np.float64),
        np.array([start[1], end[1]], dtype=np.float64),
        info.width,
        info.height,
        *_SCREEN_SIZE,
    )
    xs = np.clip(xs, 0.0, info.width)
    ys = np.clip(ys, 0.0, info.height)
    return float(xs[0]), float(ys[0]), float(xs[1]), float(ys[1])


def _export(session: api.Session, slot: str, tick: int) -> str:
    """Write the recorded history beside the working directory, and say where it went.

    Returns the message rather than printing it: the status line is the only place a viewer can
    tell the user anything, and a `print` behind a game window is a message nobody reads (§8.7
    applies to success too — an export that says nothing is one the user repeats).
    """
    try:
        csv_path, _ = export_history(session.history(), _EXPORT_DIRECTORY, f"{slot}-tick{tick}")
    except ValueError as refused:
        return str(refused)
    return f"wrote {csv_path} (+ .json)"


def _save(session: api.Session, slot: str) -> str:
    """Write the world to its slot, and say where it went, or why not: a served world saves its
    own snapshot and refuses to be saved by a client."""
    try:
        path = session.save(str(_SAVE_DIRECTORY / f"{slot}.npz"))
    except api.Refused as refused:
        return f"refused: {refused}"
    return f"saved {path} at tick {session.status().tick}"


def _watch(session: api.RemoteSession, moved: threading.Event) -> None:
    """Hold a served world's event stream open, and flag every catch-up chunk.

    On a thread of its own, because the stream blocks between events and the view must keep
    drawing. Holding it open is also what makes the world *watched*: the service shortens its wake
    interval for as long as a stream is open (#36), so this one call is both the notification and
    the attendance.
    """
    for _progress in session.events():
        moved.set()


def _restore(session: api.Session, slot: str) -> str:
    """Load the slot into the running world, or say why not.

    A refusal is reported rather than raised because refusing a load is a normal outcome — a
    snapshot taken under different rules (§2.8) is exactly what it exists to prevent, and the
    status line is the only place a viewer can say so.
    """
    path = _SAVE_DIRECTORY / f"{slot}.npz"
    try:
        status = session.load(str(path))
    except api.Refused as refused:
        return f"refused: {refused}"
    return f"loaded {path} at tick {status.tick}"


def run(
    seed: int = 0,
    n_entities: int = 200,
    world_name: str = api.DEFAULT_WORLD,
    connect: Optional[str] = None,
) -> None:
    remote = connect is not None
    session = api.connect(connect) if connect is not None else api.create(
        world_name, seed=seed, founders=n_entities
    )
    info = session.describe()
    slot = f"{info.name}-seed{info.seed}"
    playback = Playback(ticks_per_second=1.0)

    pygame.init()
    screen = pygame.display.set_mode(_SCREEN_SIZE)
    pygame.display.set_caption(
        f"Sanctuary -- {info.name} world" + (f" served at {connect}" if remote else "")
    )
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()

    moved = threading.Event()
    if isinstance(session, api.RemoteSession):
        threading.Thread(target=_watch, args=(session, moved), name="watch", daemon=True).start()
    # When the frame on screen arrived, and how long the one before it took to come. A served
    # world's frames arrive when the service wakes, so they are blended over the gap actually
    # observed rather than over an interval this view would have to be told.
    arrived_ms = previous_gap_ms = 1_000

    # Terrain and water never change, so this is rendered once and every overlay is blended onto
    # a copy of it rather than recomputing the hillshade.
    terrain = session.terrain()
    terrain_rgb = apply_water_overlay(elevation_shading(terrain), terrain.water_depth.to_numpy())

    # Two frames, blended between (§2.1: the renderer interpolates between ticks). Kept here
    # rather than asked of the world, because how a view smooths motion is the view's business.
    previous = current = session.frame()

    layer_index = 0
    condition_index = 0
    fencing_from: tuple[int, int] | None = None
    selected: int | None = None
    charts_shown = True
    gene_names = info.gene_names
    gene_index = gene_names.index("speed") if "speed" in gene_names else len(gene_names)
    exported = ""
    background = _scaled_surface(terrain_rgb, _SCREEN_SIZE)
    redraw_background = False
    readings_key: Optional[tuple] = None

    running = True
    while running:
        elapsed_seconds = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    playback.toggle_pause()
                elif event.key in (pygame.K_RIGHT, pygame.K_PERIOD):
                    playback.request_step()
                elif event.key in (pygame.K_UP, pygame.K_EQUALS):
                    playback.set_speed(playback.speed * 2.0)
                elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                    playback.set_speed(playback.speed / 2.0)
                elif event.key in (pygame.K_TAB, pygame.K_f):
                    layer_index = (layer_index + 1) % len(_LAYER_CYCLE)
                    redraw_background = True
                elif event.key == pygame.K_v:
                    # What an animal's colour means. With one species, colouring by species is a
                    # field of identical dots — every mechanic this view exists to show is a
                    # *condition*, and none of it was visible until this key existed (#39).
                    condition_index = (condition_index + 1) % len(CONDITION_MODES)
                elif event.key == pygame.K_c:
                    charts_shown = not charts_shown
                elif event.key == pygame.K_g:
                    # Cycle which trait is plotted, ending on `None` — population and condition
                    # alone. Which gene is under selection is the question being asked, and it
                    # changes run to run, so it is a key rather than a constant.
                    gene_index = (gene_index + 1) % (len(gene_names) + 1)
                elif event.key == pygame.K_e:
                    exported = _export(session, slot, current.tick)
                elif event.key == pygame.K_x and selected is not None:
                    # Only meaningful with an animal selected: a cull names a *species*, and
                    # clicking one is how the player says which. Silently doing nothing without a
                    # selection is better than culling an arbitrary species (§8.7 cuts both ways —
                    # the loud failure here would be culling the wrong thing).
                    exported = _cull(session, selected)
                elif event.key == pygame.K_s:
                    exported = _save(session, slot)
                elif event.key == pygame.K_l:
                    exported = _restore(session, slot)
                    # The loaded world's animals are nothing like the ones on screen, so there is
                    # nothing to blend from, and the overlay must match the field that just arrived.
                    previous = current = session.frame()
                    redraw_background = True
                    readings_key = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                fencing_from = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3 and fencing_from:
                # Requested, never applied here: an intervention is recorded data drained at a tick
                # boundary, and a client reaching into the world would be the second mutation path
                # §2.7 exists to forbid. The refusal — a box thinner than a cell, or an empty
                # balance — comes back through the status line like every other outcome.
                session.request(api.FenceRequest(*_drag_to_world(info, fencing_from, event.pos)))
                fencing_from = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Pick against the frame on screen, which holds only the living: a gestating young
                # and a dead animal are both unselectable, because neither is in it (#119).
                wx, wy = screen_to_world(
                    np.array([event.pos[0]], dtype=np.float64),
                    np.array([event.pos[1]], dtype=np.float64),
                    info.width,
                    info.height,
                    *_SCREEN_SIZE,
                )
                selected = pick_entity(
                    float(wx[0]),
                    float(wy[0]),
                    current.x.to_numpy(),
                    current.y.to_numpy(),
                    current.ids.to_numpy(),
                    _PICK_RADIUS,
                )

        if remote:
            n_ticks = 0
            if moved.is_set():
                moved.clear()
                arrival = session.frame()
                if arrival.tick != current.tick:
                    n_ticks = arrival.tick - current.tick
                    previous, current = current, arrival
                    now_ms = pygame.time.get_ticks()
                    previous_gap_ms, arrived_ms = max(now_ms - arrived_ms, 1), now_ms
            alpha = min(1.0, (pygame.time.get_ticks() - arrived_ms) / previous_gap_ms)
        else:
            n_ticks, alpha = playback.advance(elapsed_seconds)
            if n_ticks > 0:
                session.advance(n_ticks)
                previous, current = current, session.frame()

        mode = CONDITION_MODES[condition_index]
        # Everything below the positions changes at most once a tick, so it is read once a tick
        # rather than once a frame. Over the protocol that is the difference between a request a
        # second and several hundred.
        if readings_key != (current.tick, mode, selected):
            if readings_key is None or readings_key[0] != current.tick:
                status = session.status()
                fence_segments = barrier_segments(
                    session.barriers(), info.cell_size, info.width, info.height, *_SCREEN_SIZE
                )
                samples = session.history()
                outcome = _last_outcome(session.ledger())
            colors = (
                species_colors(current.species.to_numpy())
                if mode == "species"
                else condition_colors(session.condition(mode).values.to_numpy())
            )
            panel = (
                describe_entity(selected, session.inspect(selected)) if selected is not None else ()
            )
            readings_key = (current.tick, mode, selected)

        layer = _LAYER_CYCLE[layer_index]
        # The fields move once per tick, not once per frame, so the overlay is rebuilt on a tick
        # boundary rather than every pass — a smoothscale to the window size per frame would cost
        # far more than the field read that motivates it.
        if redraw_background or (n_ticks > 0 and layer is not None):
            rgb = terrain_rgb if layer is None else field_overlay(terrain_rgb, session.field(layer))
            background = _scaled_surface(rgb, _SCREEN_SIZE)
            redraw_background = False

        screen.blit(background, (0, 0))

        x, y, _z = live_positions(previous, current, alpha)
        px, py = world_to_screen(x, y, info.width, info.height, *_SCREEN_SIZE)
        casing = casing_color()
        for screen_x, screen_y, color in zip(px.tolist(), py.tolist(), colors.tolist()):
            # Cased, so the mark reads against terrain that ranges from near-black valley to pale
            # peak: no single fill clears 3:1 across that, and two tones always leave one of them
            # separating. See `render.casing_color`.
            pygame.draw.circle(screen, casing, (screen_x, screen_y), _ENTITY_RADIUS)
            pygame.draw.circle(screen, color, (screen_x, screen_y), _ENTITY_RADIUS - 1)

        for x0, y0, x1, y1 in fence_segments:
            pygame.draw.line(screen, _FENCE_COLOR, (x0, y0), (x1, y1), 3)

        if fencing_from:
            # Drawn while the button is held so the player can see what they are about to buy —
            # a fence is priced by its perimeter and cannot be undone by taking it back.
            pygame.draw.rect(
                screen, _FENCE_COLOR, drag_rectangle(fencing_from, pygame.mouse.get_pos()), 2
            )

        if charts_shown:
            _draw_charts(
                screen,
                font,
                samples,
                None if gene_index == len(gene_names) else gene_names[gene_index],
                gene_names,
            )
        if panel:
            _draw_panel(screen, font, panel)

        clock_text = (
            f"served at {connect}"
            if remote
            else f"{'PAUSED' if playback.paused else 'playing'} | speed x{playback.speed:g}"
        )
        status_line = (
            # Collapse first and unmissable: a run that has ended must not read as one that is
            # merely paused (§2.7). Everything after it is still true and still worth showing —
            # the charts are how you find out *why* it ended.
            ("*** COLLAPSED -- every animal is dead *** | " if status.collapsed else "")
            + f"{clock_text} | tick {status.tick} | "
            f"[tab] {layer or 'terrain'} | [v] {mode}"
            f" | [c] charts [g] trait [e] export"
            f" | right-drag to fence"
            f" [s] save [l] load [x] cull | click an animal to inspect"
            + (f" | {exported}" if exported else "")
            + (f" | {outcome}" if outcome else "")
        )
        # On a dark strip rather than straight onto terrain. White text over a pale peak is
        # unreadable, and the status line is the only place this view can tell the player anything
        # -- including that a cull was refused. Text carries meaning, so it gets a surface.
        text = font.render(status_line, True, (255, 255, 255))
        backing = pygame.Surface((text.get_width() + 12, text.get_height() + 6))
        backing.set_alpha(180)
        backing.fill((20, 20, 15))
        screen.blit(backing, (2, 5))
        screen.blit(text, (8, 8))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanctuary diagnostic world view")
    parser.add_argument("--world", choices=api.worlds(), default=api.DEFAULT_WORLD)
    parser.add_argument(
        "--connect", metavar="URL", help="view a world the service is serving, e.g. http://127.0.0.1:8765"
    )
    arguments = parser.parse_args()
    run(world_name=arguments.world, connect=arguments.connect)
