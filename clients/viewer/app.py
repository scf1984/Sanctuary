"""Thin pygame wiring for the diagnostic world view (CLAUDE.md §3.3).

All the logic worth testing lives in render.py, playback.py and demo_world.py, every one of them
pygame-free; this module owns only the window, the event loop, and one call into each of those per
frame. That split is load-bearing rather than tidy: this module is uncollectable in CI, which
never installs the viewer extra, so anything that lives here is untestable by construction — which
is how the world builder's `EntityStore` call stayed broken through a whole release (#110).
Rendering technology: pygame (CLAUDE.md §3.3), chosen over matplotlib because pause/step/speed
need a real per-frame event loop and immediate-mode blitting rather than a plotting library's
redraw-the-whole-figure model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from clients.viewer.charts import stack_charts, world_charts
from clients.viewer.demo_world import build_demo_world
from clients.viewer.export import export_history
from clients.viewer.playback import Playback
from clients.viewer.render import (
    PLANT_LAYERS,
    apply_water_overlay,
    describe_entity,
    elevation_shading,
    live_positions,
    pick_entity,
    plant_overlay,
    plant_overlay_references,
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
_PANEL_BACKGROUND = (12, 12, 14)
_PANEL_TEXT = (232, 232, 226)

# The chart panel: wide enough that a few hundred samples are one pixel column apart, and short
# enough that four stacked charts leave most of the world visible. It sits bottom-left, away from
# the inspection panel at the top.
_CHART_SIZE = (320, 54)
_EXPORT_DIRECTORY = Path("exports")

# What the overlay key cycles through. `None` is terrain alone, and it leads so that the view
# opens on the same picture it always has.
_LAYER_CYCLE: tuple[str | None, ...] = (None, *PLANT_LAYERS)


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


def _draw_charts(screen, font, history, gene) -> None:
    """Blit the metric panel bottom-left. Layout only — `charts.py` decides where a point goes,
    because that module is collectable in CI and this one is not (#110)."""
    charts = world_charts(history, gene)
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


def _export(world, seed: int) -> str:
    """Write the recorded history beside the working directory, and say where it went.

    Returns the message rather than printing it: the status line is the only place a viewer can
    tell the user anything, and a `print` behind a game window is a message nobody reads (§8.7
    applies to success too — an export that says nothing is one the user repeats).
    """
    stem = f"seed{seed}-tick{world.loop.tick_count}"
    try:
        csv_path, _ = export_history(world.loop.metrics, _EXPORT_DIRECTORY, stem)
    except ValueError as refused:
        return str(refused)
    return f"wrote {csv_path} (+ .json)"


def run(seed: int = 0, n_entities: int = 200) -> None:
    pygame.init()
    screen = pygame.display.set_mode(_SCREEN_SIZE)
    pygame.display.set_caption("Sanctuary -- world view")
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()

    world = build_demo_world(seed, n_entities)
    playback = Playback(ticks_per_second=1.0)

    # Terrain and water never change, so this is rendered once and every overlay is blended onto
    # a copy of it rather than recomputing the hillshade.
    terrain_rgb = apply_water_overlay(elevation_shading(world.terrain), world.water)
    # Fixed for the run, deliberately: a ramp that rescales itself hides the slow decline this
    # view exists to show. See `render.plant_overlay_references`.
    references = plant_overlay_references(world.plants)

    layer_index = 0
    selected: int | None = None
    charts_shown = True
    gene_names = world.genes.vocabulary.names
    gene_index = gene_names.index("speed") if "speed" in gene_names else len(gene_names)
    exported = ""
    background = _scaled_surface(terrain_rgb, _SCREEN_SIZE)
    redraw_background = False

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
                elif event.key == pygame.K_c:
                    charts_shown = not charts_shown
                elif event.key == pygame.K_g:
                    # Cycle which trait is plotted, ending on `None` — population and condition
                    # alone. Which gene is under selection is the question being asked, and it
                    # changes run to run, so it is a key rather than a constant.
                    gene_index = (gene_index + 1) % (len(gene_names) + 1)
                elif event.key == pygame.K_e:
                    exported = _export(world, seed)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Pick against the *living*, so a gestating row and a freed one are both
                # unselectable — the first has not been born and the second is not there (#119).
                alive = world.store.alive & (world.store.age >= 0)
                rows = np.flatnonzero(alive)
                wx, wy = screen_to_world(
                    np.array([event.pos[0]], dtype=np.float64),
                    np.array([event.pos[1]], dtype=np.float64),
                    world.terrain.world_width,
                    world.terrain.world_height,
                    *_SCREEN_SIZE,
                )
                selected = pick_entity(
                    float(wx[0]),
                    float(wy[0]),
                    world.store.x[alive],
                    world.store.y[alive],
                    rows,
                    _PICK_RADIUS,
                )

        n_ticks, alpha = playback.advance(elapsed_seconds)
        if n_ticks > 0:
            world.loop.advance(n_ticks)

        layer = _LAYER_CYCLE[layer_index]
        # The fields move once per tick, not once per frame, so the overlay is rebuilt on a tick
        # boundary rather than every pass — a smoothscale to the window size per frame would cost
        # far more than the field read that motivates it.
        if redraw_background or (n_ticks > 0 and layer is not None):
            rgb = (
                terrain_rgb
                if layer is None
                else plant_overlay(terrain_rgb, world.plants, layer, references)
            )
            background = _scaled_surface(rgb, _SCREEN_SIZE)
            redraw_background = False

        screen.blit(background, (0, 0))

        x, y, _z, drawn = live_positions(
            world.loop.previous_positions,
            world.loop.previous_row_ids,
            world.loop.current_positions,
            world.loop.current_row_ids,
            alpha,
        )
        px, py = world_to_screen(
            x, y, world.terrain.world_width, world.terrain.world_height, *_SCREEN_SIZE
        )
        colors = species_colors(world.store.species_id[drawn])
        for screen_x, screen_y, color in zip(px.tolist(), py.tolist(), colors.tolist()):
            pygame.draw.circle(screen, color, (screen_x, screen_y), _ENTITY_RADIUS)

        if charts_shown:
            _draw_charts(
                screen,
                font,
                world.loop.metrics,
                None if gene_index == len(gene_names) else gene_names[gene_index],
            )
        if selected is not None:
            _draw_panel(screen, font, describe_entity(world, selected))

        status = (
            f"{'PAUSED' if playback.paused else 'playing'} | "
            f"speed x{playback.speed:g} | tick {world.loop.tick_count} | "
            f"[tab] {layer or 'terrain'} | [c] charts [g] trait [e] export"
            f" | click an animal to inspect"
            + (f" | {exported}" if exported else "")
        )
        screen.blit(font.render(status, True, (255, 255, 255)), (8, 8))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
