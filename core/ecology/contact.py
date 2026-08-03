"""Who is touching whom: the one spatial question two ecological verbs both ask (issue #179).

Mating (#20) and predation (#179) are the same query with different consequences — *find the pairs
of animals that are close enough to interact* — and the answer is the same bucket-and-check in both
places. This module holds it once.

**Extracted at the second caller rather than the third**, against §8.3's default, and the reason is
specific: the algorithm is subtle in a way that a duplicate would not survive. It packs two cell
coordinates into one sortable key by a `1 << 32` shift, it ranks within a run to keep a pair from
straddling two cells, and it re-checks the true distance because sharing a cell only bounds a pair
by `contact_range × √2`. A second copy would drift on any of those three and nothing would fail
loudly (§8.7) — it would merely pair slightly wrong animals, which is invisible in every test that
does not already know the answer.

**The pairing is returned as row-index arrays, not selections.** A `Selection` is a boolean mask,
so it carries no order, and two masks can only express pairings whose couples do not cross in row
space — which a pairing built from *position* does constantly. #191 hit exactly this and moved
`Genetics.inherit` to index arrays for it; the same reasoning binds here.
"""

from __future__ import annotations

import numpy as np


def pair_by_contact(
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    contact_range: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-index arrays of the entities in `rows` that are within `contact_range`, paired elementwise.

    x, y: the store's full position columns, `(capacity,) float32, world units`. Indexed by row,
        so they are the whole column rather than a selection's slice.
    rows: `(m,) int64`, the candidate rows. Each appears in at most one returned pair.

    Candidates are bucketed into contact-sized cells and paired with whoever else is in the bucket;
    the bucket is a cheap way to find candidates and the distance is what decides. Two animals
    therefore interact because they are in the same place, which is the whole of the spatial rule —
    density is what makes both mates and prey findable, so crowding and dispersal matter without
    either being written down.

    **Shuffled before bucketing**, so pairing within a cell is not biased by row order. Rows come
    from a free list that hands neighbouring indices to entities allocated together, so without this
    a lineage's own offspring would preferentially pair with each other.

    Vectorized throughout: there is no per-entity search, because a pairwise nearest-neighbour query
    over the whole population is the 6.3 s/tick cost that ruled out pairwise sensing in #96.
    """
    rows = rows[rng.permutation(rows.shape[0])]
    cell_x = np.floor(x[rows] / contact_range).astype(np.int64)
    cell_y = np.floor(y[rows] / contact_range).astype(np.int64)
    # Two cell coordinates into one sortable key; sorting by it groups a cell's occupants together,
    # which is all the pairing needs. The key's arithmetic value means nothing.
    key = cell_x * np.int64(1 << 32) + cell_y

    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    n = rows.shape[0]

    # Rank within each cell, so pairing consecutive entries never straddles two cells.
    starts_cell = np.empty(n, dtype=bool)
    starts_cell[0] = True
    starts_cell[1:] = sorted_key[1:] != sorted_key[:-1]
    cell_start = np.maximum.accumulate(np.where(starts_cell, np.arange(n), 0))
    rank = np.arange(n) - cell_start

    has_partner = np.zeros(n, dtype=bool)
    has_partner[:-1] = sorted_key[1:] == sorted_key[:-1]
    leads = np.nonzero((rank % 2 == 0) & has_partner)[0]

    first = rows[order[leads]]
    second = rows[order[leads + 1]]
    # A shared cell puts two entities within `contact_range × √2`, not within `contact_range`, so
    # the distance is checked rather than assumed.
    touching = np.hypot(x[first] - x[second], y[first] - y[second]) <= contact_range
    return first[touching], second[touching]
