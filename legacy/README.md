# legacy/

This is the 2017–2023 prototype. **It does not run.** It is kept for reference only —
do not import anything under `legacy/` from new code, and do not treat any file here as a
starting point to extend.

Three ideas from this prototype are worth carrying forward into the new core (see
`CLAUDE.md` §1 for how):

- **Trait genetics** (`traits.py`) — Gaussian inheritance around the parental mean, clamped
  to a bounded drift range.
- **Entity indirection by id** — entities referenced by id and resolved through a central
  store (`Blackboard` / `EntityFetcher`), never by direct object pointer.
- **Spatial hashing** (`InteractionGrid` in `world.py`) — cell size derived from the maximum
  sensing range.

Everything else here — the singleton metaclasses, the `Event` dataclass with an overridden
`Callable` field, base classes living in package `__init__.py` with subclasses in a sibling
module — is exactly the pattern the new core must not reproduce. See `CLAUDE.md` §1 for the
full list of what not to carry forward.

## Known defects

These are catalogued here so the new core can be checked against them — none of these bugs
should reappear.

- **Reversed `atan2` argument order.** `Vector.angle` in `utils.py` computes
  `atan2(*self.coords)`, i.e. `atan2(x, y)`, on a vector whose `coords` are stored as
  `(x, y)`. `atan2` takes `(y, x)`; every heading and bearing computed from this property is
  wrong.
- **`SightAngle` compared in degrees against radians.** Trait values such as
  `SightAngle(35)` in `entities.py` are authored as plain degrees, but `Entity.can_see`
  compares that value directly against `Vector.angle`, which returns radians from `atan2`.
  The comparison silently "works" (both are just floats) and always evaluates the same way
  regardless of the actual angle.
- **Live-view iteration in `World.update`.** `World.entities` (`world.py`) returns a live
  iterator over `Blackboard`'s backing dict. `World.update` iterates it directly while
  processing entities; as soon as an update mutates the entity set (e.g. a birth or death),
  this raises `RuntimeError: dictionary changed size during iteration` or silently skips
  entities.
- **Unreachable `stats` package.** `stats/stats.py` imports `CreateEntityEvent` and
  `ChangeEntityEvent` from `events`, but the `events` package only defines `Event` — those
  two names do not exist anywhere. `stats/stats.py` is also never imported by
  `stats/__init__.py` or anything else, so the module is dead and was never exercised.
- **Decorative `StateTransitions` / `FoodChain`.** `Entity.__init__` builds a
  `StateTransitions` network in `states.py`, but `Entity.change_state` just assigns
  `self.state = new_state` directly and never calls `Network.set_current_state`, so the
  declared transition graph is never consulted and illegal transitions are never rejected.
  `FoodChain` in `entities.py` is declared the same way and never referenced anywhere;
  `can_eat` / `is_eaten_by` are implemented ad hoc via the `eats` class attribute instead.
- **Zero-size render `bbox(0.00)`.** `World.render` in `world.py` calls
  `entity.location.bbox(0.00)`, passing a size of zero, so every entity's bounding box has
  zero area and nothing actually renders as a visible shape on the canvas.

## Do not import from here

Nothing outside `legacy/` may import from `legacy/`. This is enforced by
`tools/check_legacy_imports.py` (see the repository root for how it is wired into linting).
