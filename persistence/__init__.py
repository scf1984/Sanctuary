"""Persistence: a world's state, durable across a process (CLAUDE.md §3.2, issue #31).

**The snapshot is the only copy of a world in existence.** §2.2 makes the simulation
non-deterministic by design, so a world cannot be regenerated from its seed; §3.2 draws the
consequence — losing one destroys something no amount of compute recovers, and restore is a
correctness requirement rather than ops hygiene. That is why the checks here refuse rather than
repair, and why nothing is written that could be silently wrong on the way back in.

**A snapshot carries state, and the config carries the rules.** `load` restores into a world that
has already been assembled from a config, and refuses if that config is not the one the snapshot
was taken under — see `fingerprint`. §2.8 requires a world to keep running under *its own* rules
forever, and this is what makes the alternative impossible rather than merely discouraged: a
snapshot cannot be opened into a differently-tuned world at all.

Serialising the config itself is the other half of #31 and is **not** here. Until it is, a snapshot
is portable between processes running the same code and not between versions of it — the
fingerprint is what turns that limitation from a silent hazard into a refused load.
"""

from persistence.snapshot import (
    SCHEMA_VERSION,
    SnapshotError,
    fingerprint,
    load,
    save,
)

__all__ = ["SCHEMA_VERSION", "SnapshotError", "fingerprint", "load", "save"]
