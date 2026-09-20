"""L1 statistical reaction (§5.1–5.2): transition probabilities over exposed voters, numpy only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import L1Params
from .population import STEP_DOWN, STEP_UP, Population


@dataclass
class L1Result:
    idx: np.ndarray        # exposed voters
    p_up: np.ndarray       # A-ward transition prob per exposed voter
    p_down: np.ndarray     # backlash prob
    up: np.ndarray         # bool masks over idx
    down: np.ndarray


def transition_probs(pop: Population, prm: L1Params, idx: np.ndarray, dose: np.ndarray, trust_eff: np.ndarray,
                     frame_fit: np.ndarray, falsehood: float, neighbor_lean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = pop.state[idx]
    pers = pop.persuadability[idx]
    lit = pop.literacy[idx]
    ladder_up = np.asarray(prm.ladder_up, dtype=np.float32)[s]
    ladder_down = np.asarray(prm.ladder_down, dtype=np.float32)[s]

    resist = 1.0 - lit * (prm.resist_w + prm.falsehood_resist * falsehood)
    resist = resist * (1.0 - pop.inoculation[idx] * prm.inoculation_w)
    punch = 1.0 + prm.falsehood_punch * falsehood
    r = pop.exposure_mem[idx] + dose
    gain = (1.0 + prm.repeat_boost * np.log1p(r)) * np.exp(-r / prm.repeat_r0)
    social = 1.0 + prm.social_w * neighbor_lean[idx]

    p_up = prm.base_rate * pers * frame_fit * trust_eff * resist * punch * gain * social * ladder_up
    p_up = np.clip(p_up, 0.0, prm.p_cap).astype(np.float32)
    over = np.maximum(0.0, r - prm.backlash_start) / prm.backlash_start
    p_down = prm.backlash_rate * over * (0.3 + lit) * (1.0 + falsehood) * ladder_down
    p_down = np.clip(p_down, 0.0, 0.5).astype(np.float32)
    return p_up, p_down


def sample_moves(p_up: np.ndarray, p_down: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.random(len(p_up))
    up = u < p_up
    down = (~up) & (u < p_up + p_down)
    return up, down


def apply_moves(state_new: np.ndarray, idx: np.ndarray, up: np.ndarray, down: np.ndarray):
    if up.any():
        j = idx[up]
        state_new[j] = STEP_UP[state_new[j]]
    if down.any():
        j = idx[down]
        state_new[j] = STEP_DOWN[state_new[j]]
