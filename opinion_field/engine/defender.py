"""Rule-based defender (§5.4, P0): detect → fact-check/correction exposure → revert + inoculate.

Finite resources: correction impressions ≤ strength·budget_ratio·(manipulator impressions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DefenderParams
from ..schema import CHANNELS, N_CELLS, N_CHANNELS
from .channels import ChannelTable, expose
from .manipulator import Plan
from .population import STEP_DOWN, Population


@dataclass
class DefenseResult:
    detected: bool
    p_detect: float
    impressions: float = 0.0
    cost: float = 0.0
    n_corr_exposed: int = 0
    n_reverted: int = 0
    n_inoculated: int = 0
    reverted_idx: np.ndarray | None = None

    def summary(self) -> dict:
        return {"detected": self.detected, "p_detect": round(self.p_detect, 3),
                "impressions": round(self.impressions, 1), "cost": round(self.cost, 1),
                "n_corr_exposed": self.n_corr_exposed, "n_reverted": self.n_reverted,
                "n_inoculated": self.n_inoculated}


class RuleDefender:
    def __init__(self, prm: DefenderParams, chan: ChannelTable):
        self.prm = prm
        self.chan = chan
        # corrections travel on high-correction-trust channels (portal, TV, kakao …)
        w = chan.correction_trust.astype(np.float64)
        self.channel_mix = w / w.sum()

    def act(self, pop: Population, plan: Plan, exposed_idx: np.ndarray, state_new: np.ndarray,
            moved_up_mask: np.ndarray, round_no: int, rng: np.random.Generator) -> DefenseResult:
        prm = self.prm
        if not prm.enabled or prm.strength <= 0:
            return DefenseResult(False, 0.0)
        volume = min(len(exposed_idx) / max(pop.n, 1) * 5.0, 1.0)
        logit = (prm.detect_bias + prm.detect_falsehood_w * plan.falsehood
                 + prm.detect_strength_w * prm.strength + prm.detect_volume_w * volume)
        p_detect = float(1.0 / (1.0 + np.exp(-logit)))
        if rng.random() >= p_detect:
            return DefenseResult(False, p_detect)

        # allocate corrections to targeted cells ∝ exposures there
        exp_c = np.bincount(pop.cell[exposed_idx], minlength=N_CELLS).astype(np.float64)
        alloc = np.zeros(N_CELLS)
        alloc[plan.target_cells] = exp_c[plan.target_cells]
        if alloc.sum() <= 0:
            return DefenseResult(True, p_detect)
        alloc /= alloc.sum()
        total = prm.strength * prm.budget_ratio * plan.impressions_total
        cell_impr = alloc[:, None] * self.channel_mix[None, :] * total
        cost = float((cell_impr.sum(axis=0) * self.chan.cpi).sum())
        ex = expose(pop, cell_impr, rng, trust_mult=self.chan.correction_trust)
        if len(ex.idx) == 0:
            return DefenseResult(True, p_detect, total, cost)

        lit = pop.literacy[ex.idx]
        recent = (pop.last_move[ex.idx] >= round_no - prm.recency_rounds) & (pop.move_dir[ex.idx] > 0)
        recent |= moved_up_mask[ex.idx]
        p_rev = (prm.revert_base * prm.strength * ex.trust_eff * (0.5 + 0.5 * lit)
                 * (0.5 + 0.5 * plan.falsehood))
        rev = recent & (rng.random(len(ex.idx)) < p_rev)
        rev_idx = ex.idx[rev]
        state_new[rev_idx] = STEP_DOWN[state_new[rev_idx]]
        # inoculation (prebunk) for everyone reached by the correction
        gain = prm.inoculation_gain * ex.trust_eff
        pop.inoculation[ex.idx] = np.minimum(1.0, pop.inoculation[ex.idx] + gain).astype(np.float32)
        return DefenseResult(True, p_detect, float(total), cost, int(len(ex.idx)), int(rev.sum()),
                             int(len(ex.idx)), rev_idx)
