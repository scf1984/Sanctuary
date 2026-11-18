"""The process wrapper around `api`: one world, served over HTTP, ticking on its own (#36).

Stage 2 of §3.1. One process, one world, no accounts. `python -m service` runs it.
"""

from service.server import Service
from service.state import Founding, close_world, open_world

__all__ = ["Founding", "Service", "close_world", "open_world"]
