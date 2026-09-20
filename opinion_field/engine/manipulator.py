"""Rule-based manipulator (§5.4, P0): target vulnerable cells, buy impressions, pick frame/claim tier.

Outputs a Plan with slot-labelled message *descriptions* (templates) — no real names,
no free-form persuasive copy. Learning = EMA of per-cell yield from previous rounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import ManipulatorParams
from ..schema import (CHANNEL_LABELS_KO, CHANNELS, CLAIM_TIER_LABELS_KO, CLAIM_TIERS, FRAME_LABELS_KO,
                      FRAMES, N_CELLS, N_CHANNELS, N_STATES, SupportState, cell_label, decode_cell)
from .channels import ChannelTable
from .population import Population

# (frame, tier) → abstract, slot-labelled description of the message content
TEMPLATES = {
    ("economy", 0): "후보 A의 경제·민생 공약을 유권자 생활 개선과 연결해 소개하는 긍정 메시지",
    ("economy", 1): "후보 B 관련 경제 지표를 맥락 없이 선택적으로 부각해 실패로 보이게 하는 편집물",
    ("economy", 2): "후보 B가 이슈 I와 관련해 유권자 경제 부담을 숨겼다는 출처 불명의 허위 의혹",
    ("security", 0): "후보 A의 안보·질서 공약과 경력을 안정감 프레임으로 소개",
    ("security", 1): "후보 B의 과거 발언 일부를 잘라 안보 불안을 조장하는 오도적 클립",
    ("security", 2): "후보 B가 이슈 I에서 외부 세력과 결탁했다는 근거 없는 허위 주장",
    ("fairness", 0): "후보 A의 공정·반부패 공약을 사례 중심으로 소개",
    ("fairness", 1): "후보 B 주변 인물의 무관한 사건을 후보 B의 책임처럼 연결하는 오도적 메시지",
    ("fairness", 2): "후보 B가 이슈 I 관련 특혜를 받았다는 조작된 문서 기반 허위 의혹",
    ("welfare", 0): "후보 A의 복지·돌봄 공약이 해당 세대에 주는 혜택을 설명하는 메시지",
    ("welfare", 1): "후보 B가 복지를 축소할 것이라고 단정하는 근거 부족한 경고성 메시지",
    ("welfare", 2): "후보 B가 이슈 I 관련 복지 예산을 유용했다는 허위 주장",
    ("competence", 0): "후보 A의 행정 경험과 성과를 강조하는 능력 프레임 메시지",
    ("competence", 1): "후보 B의 실수 장면만 모아 무능하게 보이도록 편집한 영상",
    ("competence", 2): "후보 B의 학력·경력이 위조됐다는 근거 없는 허위 의혹",
}
MOVABLE = np.array([1, 0, 1, 1, 1, 0], dtype=np.float32)  # abstain, B_weak, undecided, A_weak count as movable


@dataclass
class Plan:
    round_no: int
    frame: str
    claim_tier: int                       # 0 factual, 1 misleading, 2 fabricated
    falsehood: float                      # 0..1
    target_cells: np.ndarray
    cell_impressions: np.ndarray          # (N_CELLS, C)
    impressions_total: float
    cost: float
    message: str                          # slot-labelled description
    channel_mix: np.ndarray               # (C,) share of impressions

    def summary(self) -> dict:
        return {
            "frame": self.frame, "frame_ko": FRAME_LABELS_KO[self.frame],
            "claim_tier": CLAIM_TIERS[self.claim_tier], "claim_tier_ko": CLAIM_TIER_LABELS_KO[CLAIM_TIERS[self.claim_tier]],
            "falsehood": round(self.falsehood, 3), "n_target_cells": int(len(self.target_cells)),
            "target_cells": [int(c) for c in self.target_cells[:20]],
            "target_cells_label_top5": [cell_label(int(c)) for c in self.target_cells[:5]],
            "impressions_total": round(float(self.impressions_total), 1), "cost": round(float(self.cost), 1),
            "channel_mix": {c: round(float(s), 3) for c, s in zip(CHANNELS, self.channel_mix)},
            "message": self.message,
        }


class RuleManipulator:
    def __init__(self, prm: ManipulatorParams, chan: ChannelTable, pop: Population, ethics_level: float,
                 budget_per_round: float):
        self.prm = prm
        self.chan = chan
        self.ethics_level = float(ethics_level)
        self.budget_per_round = float(budget_per_round)
        self.yield_ema = np.ones(N_CELLS, dtype=np.float32)
        # static cell stats
        cs = np.maximum(pop.cell_size, 1)
        self.cell_pers = np.bincount(pop.cell, weights=pop.persuadability, minlength=N_CELLS) / cs
        self.cell_lit = np.bincount(pop.cell, weights=pop.literacy, minlength=N_CELLS) / cs
        # channel attractiveness per cell = mean(reach·trust)
        rt = pop.reach * pop.trust
        self.cell_chan = np.stack([np.bincount(pop.cell, weights=rt[:, k], minlength=N_CELLS) / cs
                                   for k in range(N_CHANNELS)], axis=1).astype(np.float32)
        self.cell_age = np.zeros(N_CELLS, dtype=np.int64)
        for c in range(N_CELLS):
            self.cell_age[c] = decode_cell(c)[1]

    @property
    def claim_tier(self) -> int:
        return int(min(2, np.floor(self.ethics_level * 3)))

    def plan(self, pop: Population, round_no: int) -> Plan:
        prm = self.prm
        counts = pop.cell_state_counts().astype(np.float32)
        movable = (counts * MOVABLE[None, :]).sum(axis=1) / np.maximum(pop.cell_size, 1)
        vulnerability = self.cell_pers * (1.0 - self.cell_lit) * movable * self.yield_ema
        vulnerability[pop.cell_size < prm.min_cell_size] = 0.0
        rank_score = vulnerability * np.power(np.maximum(pop.cell_size, 1), prm.size_weight_exp)
        k = min(prm.n_target_cells, int((rank_score > 0).sum()))
        target = np.argsort(-rank_score, kind="stable")[:k]

        # frame: best average affinity over targeted population
        tgt_size = pop.cell_size[target].astype(np.float64)
        aff = self.chan.frame_affinity[:, self.cell_age[target]]           # (F, k)
        frame_idx = int(np.argmax((aff * tgt_size[None, :]).sum(axis=1)))
        frame = FRAMES[frame_idx]

        # impressions ∝ vulnerability × size, capped per-capita; channel split by top-k attractiveness
        weight = vulnerability[target] * tgt_size
        weight = weight / max(weight.sum(), 1e-9)
        cell_impr = np.zeros((N_CELLS, N_CHANNELS), dtype=np.float64)
        chan_w = self.cell_chan[target].copy()                              # (k, C)
        if prm.channel_top_k < N_CHANNELS:
            thresh = -np.sort(-chan_w, axis=1)[:, prm.channel_top_k - 1][:, None]
            chan_w[chan_w < thresh] = 0.0
        chan_w = chan_w / np.maximum(chan_w.sum(axis=1, keepdims=True), 1e-9)
        cpi_cell = (chan_w * self.chan.cpi[None, :]).sum(axis=1)            # blended cost per impression
        impressions = weight * self.budget_per_round / np.maximum(cpi_cell, 1e-6)
        impressions = np.minimum(impressions, prm.impressions_per_capita_cap * tgt_size)
        cost = float((impressions * cpi_cell).sum())
        if cost > self.budget_per_round:                                    # renormalise after the cap
            impressions *= self.budget_per_round / cost
            cost = float((impressions * cpi_cell).sum())
        cell_impr[target] = impressions[:, None] * chan_w
        total = float(cell_impr.sum())
        mix = cell_impr.sum(axis=0) / max(total, 1e-9)
        tier = self.claim_tier
        msg = (f"[{FRAME_LABELS_KO[frame]}] ({CLAIM_TIER_LABELS_KO[CLAIM_TIERS[tier]]}) "
               f"{TEMPLATES[(frame, tier)]}")
        return Plan(round_no=round_no, frame=frame, claim_tier=tier, falsehood=tier / 2.0,
                    target_cells=target, cell_impressions=cell_impr, impressions_total=total, cost=cost,
                    message=msg, channel_mix=mix.astype(np.float32))

    def learn(self, exposed_cells: np.ndarray, moved_up_cells: np.ndarray):
        """EMA of yield (A-ward movers per exposed) per cell."""
        exp_c = np.bincount(exposed_cells, minlength=N_CELLS).astype(np.float32)
        mv_c = np.bincount(moved_up_cells, minlength=N_CELLS).astype(np.float32)
        touched = exp_c > 0
        y = np.zeros(N_CELLS, dtype=np.float32)
        y[touched] = mv_c[touched] / exp_c[touched]
        # normalise to mean 1 over touched cells so vulnerability scale stays stable
        if touched.any() and y[touched].mean() > 0:
            y[touched] = y[touched] / y[touched].mean()
        lr = self.prm.learn_rate
        self.yield_ema[touched] = (1 - lr) * self.yield_ema[touched] + lr * np.clip(y[touched], 0.2, 3.0)

    def state_dict(self) -> dict:
        return {"yield_ema": self.yield_ema.tolist()}

    def load_state_dict(self, d: dict):
        self.yield_ema = np.array(d["yield_ema"], dtype=np.float32)
