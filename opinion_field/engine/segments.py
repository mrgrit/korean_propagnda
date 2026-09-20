"""P1 strategist groups: region block (5) × age group (6) × edu bucket (3) = 90 groups.

Strategists never see individuals — only these group aggregates (§10 "유형별 효과 집계만")."""
from __future__ import annotations

import numpy as np

from ..schema import (AGE_GROUP_LABELS, CHANNELS, EDU_BUCKET_LABELS, N_AGE_GROUPS, N_BLOCKS, N_CELLS,
                      N_CHANNELS, N_EDU_BUCKETS, REGION_BLOCKS, SIDO_TO_BLOCK, decode_cell)

N_GROUPS = N_BLOCKS * N_AGE_GROUPS * N_EDU_BUCKETS


def _cell_group_table() -> np.ndarray:
    t = np.zeros(N_CELLS, dtype=np.int32)
    for c in range(N_CELLS):
        sido, age, _sex, edu = decode_cell(c)
        t[c] = (int(SIDO_TO_BLOCK[sido]) * N_AGE_GROUPS + age) * N_EDU_BUCKETS + edu
    return t


CELL_GROUP = _cell_group_table()


def group_label(g: int) -> str:
    edu = g % N_EDU_BUCKETS
    age = (g // N_EDU_BUCKETS) % N_AGE_GROUPS
    block = g // (N_EDU_BUCKETS * N_AGE_GROUPS)
    return f"{REGION_BLOCKS[block]}/{AGE_GROUP_LABELS[age]}/{EDU_BUCKET_LABELS[edu]}"


def group_age(g: int) -> int:
    return (g // N_EDU_BUCKETS) % N_AGE_GROUPS


def group_stats(pop, cell_pers: np.ndarray, cell_lit: np.ndarray, cell_chan: np.ndarray, movable_cell: np.ndarray,
                yield_cell: np.ndarray, extra_cell: dict[str, np.ndarray] | None = None) -> list[dict]:
    """Aggregate per-cell stats to groups (size-weighted means). Returns a list of dicts (one per group)."""
    size = pop.cell_size.astype(np.float64)
    gsize = np.bincount(CELL_GROUP, weights=size, minlength=N_GROUPS)
    def wmean(v):
        return np.divide(np.bincount(CELL_GROUP, weights=v * size, minlength=N_GROUPS), gsize,
                         out=np.zeros(N_GROUPS), where=gsize > 0)
    pers, lit, mov, yld = wmean(cell_pers), wmean(cell_lit), wmean(movable_cell), wmean(yield_cell)
    chan = np.stack([wmean(cell_chan[:, k]) for k in range(N_CHANNELS)], axis=1)
    extras = {k: np.bincount(CELL_GROUP, weights=v, minlength=N_GROUPS) for k, v in (extra_cell or {}).items()}
    out = []
    for g in range(N_GROUPS):
        if gsize[g] <= 0:
            continue
        top = np.argsort(-chan[g])[:2]
        d = {"id": g, "label": group_label(g), "size": int(gsize[g]), "persuadability": round(float(pers[g]), 3),
             "literacy": round(float(lit[g]), 3), "movable": round(float(mov[g]), 3), "yield": round(float(yld[g]), 2),
             "vulnerability": round(float(pers[g] * (1 - lit[g]) * mov[g] * yld[g]), 4),
             "top_channels": [CHANNELS[k] for k in top], "age": group_age(g)}
        for k, v in extras.items():
            d[k] = int(v[g])
        out.append(d)
    return out


def group_weights_to_cells(target: list[tuple[int, float]], cell_weight: np.ndarray, pop) -> np.ndarray:
    """Expand {group: weight} into a per-cell weight vector: within a group, cells split the
    group's weight ∝ cell_weight (vulnerability × size). Returns (N_CELLS,) summing to 1."""
    w = np.zeros(N_CELLS)
    for g, gw in target:
        mask = CELL_GROUP == g
        cw = cell_weight * mask
        if cw.sum() > 0:
            w += gw * cw / cw.sum()
    return w / w.sum() if w.sum() > 0 else w
