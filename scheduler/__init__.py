"""Offline advancement: when a world wakes, and catching it up when it does (#32, §2.4).

Server-side. It reaches a world through `api` like any other caller, so catch-up drives exactly the
`advance` a client would, and it never reads a clock: the process that serves the world passes the
time in.
"""

from scheduler.catchup import Advanceable, Anchor, CatchUp, Progress
from scheduler.schedule import Pacing, Stage, WakeSchedule

__all__ = ["Advanceable", "Anchor", "CatchUp", "Pacing", "Progress", "Stage", "WakeSchedule"]
