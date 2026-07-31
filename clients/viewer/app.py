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

import pygame

from clients.viewer.demo_world import build_demo_world
from clients.viewer.playback import Playback
from clients.viewer.render import (
    PLANT_LAYERS,
    apply_water_overlay,
    elevation_shading,
    live_positions,
    plant_overlay,
    plant_overlay_references,
    species_colors,
    world_to_screen,
)

_SCREEN_SIZE = (900, 900)
_ENTITY_RADIUS = 4

# What the overlay key cycles through. `None` is terrain alone, and it leads so that the view
# opens on the same picture it always has.
_LAYER_CYCLE: tuple[str | None, ...] = (None, *PLANT_LAYERS)


def _scaled_surface(rgb, screen_size: tuple[int, int]) -> pygame.Surface:
    """A window-sized surface from a (height, width, 3) uint8 grid."""
    # pygame's surfarray convention is (width, height, 3); ours is (height, width, 3).
    surface = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    return pygame.transform.smoothscale(surface, screen_size)


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

        status = (
            f"{'PAUSED' if playback.paused else 'playing'} | "
            f"speed x{playback.speed:g} | tick {world.loop.tick_count} | "
            f"[tab] {layer or 'terrain'}"
        )
        screen.blit(font.render(status, True, (255, 255, 255)), (8, 8))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
