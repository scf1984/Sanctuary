"""The boundary between a world and any client: the only package `clients/` may import (#35).

Read `api.values` for what crosses and `api.session` for the rules a session keeps. In short:
every return value is plain, entities are named by id, interventions are requests made of data,
and nothing a client holds can reach back into a running world.
"""

from api.session import CONDITIONS, FIELD_LAYERS, Refused, Session, create, worlds
from api.values import (
    BarrierView,
    Column,
    ConditionView,
    CullRequest,
    DriveShare,
    EntityReport,
    FenceRequest,
    FieldView,
    Frame,
    GeneReading,
    Ledger,
    Pace,
    PaceRequest,
    Progress,
    Quote,
    Record,
    Request,
    Sample,
    Status,
    TerrainView,
    WorldInfo,
)

# Imported after the values it carries, which it reaches through `api.wire`.
from api.remote import RemoteSession, connect

# The world a client opens when it asks for nothing in particular. Every measurement so far was
# taken on it, so a view opened with no arguments must still show it.
from worlds import DEFAULT_WORLD

__all__ = [
    "BarrierView",
    "CONDITIONS",
    "DEFAULT_WORLD",
    "Column",
    "ConditionView",
    "CullRequest",
    "DriveShare",
    "EntityReport",
    "FIELD_LAYERS",
    "FenceRequest",
    "FieldView",
    "Frame",
    "GeneReading",
    "Ledger",
    "Pace",
    "PaceRequest",
    "Progress",
    "Quote",
    "Record",
    "Refused",
    "RemoteSession",
    "Request",
    "Sample",
    "Session",
    "Status",
    "TerrainView",
    "WorldInfo",
    "connect",
    "create",
    "worlds",
]
