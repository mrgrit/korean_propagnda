"""L1→L2 promotion (§1.1): ① borderline transition prob ② centrality ③ per-cell representation, top-K.

Guarantee: every exposed cell gets one representative (its best-scoring voter) before the
remaining budget is filled by score. If there are more exposed cells than K, the K
best-scoring representatives are taken.
"""
from __future__ import annotations

import numpy as np

from ..config import PromotionParams
from .population import Population


def select_promoted(pop: Population, prm: PromotionParams, idx: np.ndarray, p: np.ndarray,
                    k: int, rng: np.random.Generator) -> np.ndarray:
    """Return positions (into idx) of the promoted voters, best first."""
    m = len(idx)
    if m == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    # closeness to the decision boundary: 1 at p=band_center, e^-1 at one band away — smooth, so the
    # ranking still prefers the most uncertain voters when nobody sits inside the hard 0.4~0.6 window
    border = np.exp(-np.abs(p - prm.band_center) / prm.band).astype(np.float32)
    cent = pop.centrality[idx]
    score = prm.w_border * border + prm.w_centrality * cent + rng.random(m) * 1e-3   # jitter = reproducible tie-break
    cells = pop.cell[idx]
    order = np.lexsort((-score, cells))                      # by cell, best score first
    first = np.ones(m, dtype=bool)
    first[1:] = cells[order][1:] != cells[order][:-1]
    reps = order[first]                                       # one representative per cell
    rep_score = score[reps] + prm.w_cell
    if len(reps) >= k:
        top = reps[np.argsort(-rep_score, kind="stable")[:k]]
        return top.astype(np.int64)
    is_rep = np.zeros(m, dtype=bool)
    is_rep[reps] = True
    rest = np.flatnonzero(~is_rep)
    need = k - len(reps)
    if need >= len(rest):
        fill = rest
    else:
        part = np.argpartition(-score[rest], need - 1)[:need]
        fill = rest[part]
    chosen = np.concatenate([reps, fill])
    final_score = score[chosen].copy()
    final_score[:len(reps)] += prm.w_cell
    return chosen[np.argsort(-final_score, kind="stable")].astype(np.int64)
