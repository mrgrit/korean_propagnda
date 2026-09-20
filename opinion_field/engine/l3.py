"""L3 network propagation (§1.1): sharers push neighbours; borderline neighbours may be promoted."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import L1Params, L3Params, PromotionParams
from .population import LEAN, STEP_DOWN, STEP_UP, Population


@dataclass
class L3Result:
    n_seeds: int
    n_reached: int
    idx: np.ndarray            # reached neighbours (unique)
    p: np.ndarray              # social transition prob (signed direction applied separately)
    direction: np.ndarray      # +1 toward A, −1 toward B
    up: np.ndarray
    down: np.ndarray

    def summary(self) -> dict:
        return {"n_seeds": self.n_seeds, "n_reached": self.n_reached,
                "n_moved_up": int(self.up.sum()), "n_moved_down": int(self.down.sum())}


def propagate(pop: Population, prm: L3Params, l1prm: L1Params, seeds: np.ndarray, state_new: np.ndarray,
              speed_eff: float, neighbor_lean: np.ndarray, rng: np.random.Generator) -> L3Result:
    empty = np.empty(0, np.int64)
    if len(seeds) == 0:
        return L3Result(0, 0, empty, np.empty(0, np.float32), np.empty(0, np.int8),
                        np.empty(0, bool), np.empty(0, bool))
    if len(seeds) > prm.max_seeds:
        seeds = rng.choice(seeds, prm.max_seeds, replace=False)
    seed_dir = np.sign(LEAN[state_new[seeds]]).astype(np.int8)
    seeds = seeds[seed_dir != 0]
    seed_dir = seed_dir[seed_dir != 0]
    if len(seeds) == 0:
        return L3Result(0, 0, empty, np.empty(0, np.float32), np.empty(0, np.int8),
                        np.empty(0, bool), np.empty(0, bool))
    src, nb = pop.neighbors_of(seeds)
    dir_map = np.zeros(pop.n, dtype=np.int8)
    dir_map[seeds] = seed_dir
    edge_dir = dir_map[src]
    t = prm.transmit_base * speed_eff * (0.5 + 0.5 * pop.persuadability[nb])
    hit = rng.random(len(nb)) < t
    nb, edge_dir = nb[hit], edge_dir[hit]
    if len(nb) == 0:
        return L3Result(len(seeds), 0, empty, np.empty(0, np.float32), np.empty(0, np.int8),
                        np.empty(0, bool), np.empty(0, bool))
    # net signed pressure per reached neighbour
    pressure = np.bincount(nb, weights=edge_dir.astype(np.float64), minlength=pop.n)
    idx = np.flatnonzero(pressure != 0)
    direction = np.sign(pressure[idx]).astype(np.int8)
    strength = np.minimum(np.abs(pressure[idx]), 3.0) / 3.0
    s = state_new[idx]
    lad_up = np.asarray(l1prm.ladder_up, np.float32)[s]
    lad_down = np.asarray(l1prm.ladder_down, np.float32)[s]
    ladder = np.where(direction > 0, lad_up, lad_down)
    social = 1.0 + l1prm.social_w * neighbor_lean[idx] * direction
    p = (prm.social_base * pop.persuadability[idx] * (1.0 - 0.5 * pop.literacy[idx])
         * (1.0 - pop.inoculation[idx] * l1prm.inoculation_w) * (0.4 + 0.6 * strength) * ladder * social)
    p = np.clip(p, 0.0, l1prm.p_cap).astype(np.float32)
    mv = rng.random(len(idx)) < p
    up = mv & (direction > 0)
    down = mv & (direction < 0)
    return L3Result(len(seeds), len(idx), idx, p, direction, up, down)


def apply_l3(state_new: np.ndarray, res: L3Result):
    if res.up.any():
        j = res.idx[res.up]
        state_new[j] = STEP_UP[state_new[j]]
    if res.down.any():
        j = res.idx[res.down]
        state_new[j] = STEP_DOWN[state_new[j]]
