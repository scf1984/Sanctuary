"""Named worlds: the content a client asks for by name, never by building config itself (#35)."""

from worlds.demo import DEFAULT_WORLD, WORLDS, build_named_world

__all__ = ["DEFAULT_WORLD", "WORLDS", "build_named_world"]
