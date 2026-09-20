"""Channel table (§4.2) and the shared exposure function used by manipulator & defender."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import load_yaml
from ..schema import CHANNELS, FRAMES, N_AGE_GROUPS, N_CELLS, N_CHANNELS


@dataclass
class ChannelTable:
    cpi: np.ndarray              # (C,)
    speed: np.ndarray
    meme: np.ndarray
    closedness: np.ndarray
    oneway: np.ndarray
    correction_trust: np.ndarray
    frame_affinity: np.ndarray   # (F, N_AGE_GROUPS)

    @classmethod
    def load(cls, path: str) -> "ChannelTable":
        raw = load_yaml(path)
        ch = raw["channels"]
        g = lambda k, d=None: np.array([ch[c][k] for c in CHANNELS], dtype=np.float32)
        fa = np.array([raw["frame_affinity"][f] for f in FRAMES], dtype=np.float32)
        assert fa.shape == (len(FRAMES), N_AGE_GROUPS)
        return cls(cpi=g("cost_per_impression"), speed=g("speed"), meme=g("meme"),
                   closedness=g("closedness"), oneway=g("oneway"), correction_trust=g("correction_trust"),
                   frame_affinity=fa)


@dataclass
class Exposure:
    idx: np.ndarray          # voter indices exposed
    dose: np.ndarray         # per exposed voter, Σ_k λ_k reach_k
    trust_eff: np.ndarray    # dose-weighted trust
    channel_share: np.ndarray  # (C,) share of delivered dose by channel
    n_candidates: int        # voters in targeted cells


def expose(pop, cell_impressions: np.ndarray, rng: np.random.Generator,
           trust_mult: np.ndarray | None = None) -> Exposure:
    """cell_impressions: (N_CELLS, C) impressions bought per cell per channel.

    λ_ik = impressions_ck / N_c (per-capita), P(exposed) = 1 − exp(−Σ_k λ_ik·reach_ik).
    Only voters in cells with any impressions are touched (vectorised gather).
    """
    active_cells = cell_impressions.sum(axis=1) > 0
    cand = np.flatnonzero(active_cells[pop.cell])
    if len(cand) == 0:
        return Exposure(np.empty(0, np.int64), np.empty(0, np.float32), np.empty(0, np.float32),
                        np.zeros(N_CHANNELS, np.float32), 0)
    per_capita = cell_impressions / np.maximum(pop.cell_size, 1)[:, None]      # (N_CELLS, C)
    lam = per_capita[pop.cell[cand]]                                            # (m, C)
    d = lam * pop.reach[cand]                                                   # (m, C)
    dose = d.sum(axis=1)
    p = 1.0 - np.exp(-dose)
    hit = rng.random(len(cand)) < p
    idx = cand[hit]
    d_hit = d[hit]
    dose_hit = dose[hit].astype(np.float32)
    trust = pop.trust[idx]
    if trust_mult is not None:
        trust = np.clip(trust * trust_mult[None, :], 0, 1)
    trust_eff = ((d_hit * trust).sum(axis=1) / np.maximum(dose_hit, 1e-6)).astype(np.float32)
    ch_share = d_hit.sum(axis=0)
    ch_share = (ch_share / max(ch_share.sum(), 1e-9)).astype(np.float32)
    return Exposure(idx=idx, dose=dose_hit, trust_eff=trust_eff, channel_share=ch_share, n_candidates=len(cand))
