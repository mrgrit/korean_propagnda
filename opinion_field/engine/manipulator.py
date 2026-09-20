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
    llm: dict | None = None               # P1: strategist decision (groups, rationale) or fallback info

    def summary(self) -> dict:
        return {
            "llm": self.llm,
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

    def observe(self, rec: dict) -> None:   # P1 hook (rule version keeps no history)
        pass

    def state_dict(self) -> dict:
        return {"yield_ema": self.yield_ema.tolist()}

    def load_state_dict(self, d: dict):
        self.yield_ema = np.array(d["yield_ema"], dtype=np.float32)


# ============================================================================ P1: LLM strategist
MANIP_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "frame": {"type": "string", "enum": FRAMES},
        "claim_tier": {"type": "integer", "enum": [0, 1, 2]},
        "target_groups": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "weight": {"type": "number"}},
            "required": ["id", "weight"], "additionalProperties": False}},
        "channel_mix": {"type": "object", "properties": {c: {"type": "number"} for c in CHANNELS},
                        "required": list(CHANNELS), "additionalProperties": False},
        "intensity": {"type": "number", "description": "라운드 예산 사용 비율 0.2~1.0"},
        "rationale": {"type": "string"},
    },
    "required": ["frame", "claim_tier", "target_groups", "channel_mix", "intensity", "rationale"],
    "additionalProperties": False,
}


class LLMManipulator(RuleManipulator):
    """P1 manipulator: an LLM strategist chooses frame / claim tier / target groups / channel mix /
    intensity from GROUP-LEVEL aggregates and its own round history (in-context learning). The
    rule-based planner remains the baseline and the fallback when the LLM answer is unusable.
    It never writes message text — content stays the abstract slot-labelled template."""

    def __init__(self, prm, chan, pop, ethics_level, budget_per_round, backend, goal_margin: float, rounds: int):
        super().__init__(prm, chan, pop, ethics_level, budget_per_round)
        from .segments import CELL_GROUP, N_GROUPS
        self.backend = backend
        self.goal_margin = goal_margin
        self.rounds = rounds
        self.history: list[dict] = []
        self.last_exposed_cells = np.zeros(N_CELLS)
        self.last_up_cells = np.zeros(N_CELLS)
        self.cell_group = CELL_GROUP
        self.n_groups = N_GROUPS
        self.memory = None                  # StrategyMemory, attached by the loop (cross-run learning)
        self.run_name: str | None = None

    # ---- brief -------------------------------------------------------------
    def _brief(self, pop: Population, round_no: int, rule_plan: Plan) -> tuple[str, dict]:
        from ..schema import CLAIM_TIER_LABELS_KO
        from .segments import group_stats
        counts = pop.cell_state_counts().astype(np.float32)
        movable = (counts * MOVABLE[None, :]).sum(axis=1) / np.maximum(pop.cell_size, 1)
        groups = group_stats(pop, self.cell_pers, self.cell_lit, self.cell_chan, movable, self.yield_ema,
                             {"exposed_last": self.last_exposed_cells, "moved_up_last": self.last_up_cells})
        shares = pop.support_shares()
        max_tier = self.claim_tier
        hist = self.history[-self.prm.history_rounds:]
        lines = [f"# 라운드 {round_no}/{self.rounds} 조작자 전략 브리핑",
                 f"목표: 후보 A 결정층 지지율 우위 마진 ≥ {self.goal_margin:.3f}. 현재 A {shares['A']:.4f} / B {shares['B']:.4f} (마진 {shares['A']-shares['B']:+.4f}), 부동층(전체) {shares['undecided_all']:.3f}.",
                 f"윤리 상한: claim_tier ≤ {max_tier} ({CLAIM_TIER_LABELS_KO[CLAIM_TIERS[max_tier]]}). 0=사실기반, 1=오도, 2=허위. 높을수록 설득력↑, 방어 탐지·리터러시 반발↑.",
                 f"라운드 예산 {self.budget_per_round:,.0f} (채널 노출 단가: " + ", ".join(f"{c} {v:.1f}" for c, v in zip(CHANNELS, self.chan.cpi)) + "). intensity 는 예산 사용 비율.",
                 "채널 성격: youtube/portal/tv 는 일방·도달 넓음, kakao/instagram/wom 은 양방·폐쇄·전파 빠름. 방어자 정정은 portal/tv 신뢰가 높음.",
                 "", "## 지난 라운드 기록 (최근)", "| R | 프레임 | 유형 | 표적 집단(상위3) | 채널 상위 | 노출 | A방향 이동 | 방어 탐지 | 되돌림 | ΔA |", "|---|---|---|---|---|---|---|---|---|---|"]
        if not hist:
            lines.append("| - | (기록 없음 — 첫 라운드) | | | | | | | | |")
        for h in hist:
            lines.append(f"| {h['round']} | {h['frame']} | {h['tier']} | {', '.join(h['groups'][:3])} | {h['channels']} | {h['exposed']:,} | {h['moved_up']:,} | {'Y' if h['detected'] else '-'} | {h['reverted']:,} | {h['dA']:+.4f} |")
        lines += ["", f"## 인구 집단 ({len(groups)}개) — id | 집단 | 규모 | 설득가능성 | 리터러시 | 이동가능비율 | 최근수율 | 선호채널 | 지난 노출 | 지난 A이동"]
        for g in groups:
            lines.append(f"{g['id']} | {g['label']} | {g['size']:,} | {g['persuadability']:.2f} | {g['literacy']:.2f} | {g['movable']:.2f} | {g['yield']:.2f} | {'/'.join(g['top_channels'])} | {g['exposed_last']:,} | {g['moved_up_last']:,}")
        lines += ["", "## 프레임별 세대 친화도 (19-29/30/40/50/60/70+)"]
        for fi, f in enumerate(FRAMES):
            lines.append(f"{f} ({FRAME_LABELS_KO[f]}): " + " ".join(f"{v:.2f}" for v in self.chan.frame_affinity[fi]))
        mem = self.memory.manip_summary(exclude_run=self.run_name) if (self.memory is not None and self.prm.use_memory) else None
        if mem:
            lines += ["", mem]
        lines += ["", f"## 규칙 기반 기본안 (참고): frame={rule_plan.frame}, tier={rule_plan.claim_tier}, 상위 셀 {', '.join(rule_plan.summary()['target_cells_label_top5'][:3])}",
                  "", f"## 출력: JSON — frame, claim_tier(≤{max_tier}), target_groups(최대 {self.prm.max_groups}개, id·weight>0), channel_mix(6채널 비중, 합 1), intensity(0.2~1.0), rationale(한 문장)."]
        # context for the mock backend (deterministic plan without parsing text)
        best_frame = rule_plan.frame
        hint = {c: round(float(v), 3) for c, v in zip(CHANNELS, rule_plan.channel_mix)}
        ctx = {"kind": "manipulator", "groups": groups, "best_frame": best_frame, "max_tier": max_tier,
               "channel_hint": hint, "max_groups": self.prm.max_groups}
        return "\n".join(lines), ctx

    # ---- plan ----------------------------------------------------------------
    def plan(self, pop: Population, round_no: int) -> Plan:
        from ..agents.prompts import STRATEGY_MANIP_SYSTEM
        rule_plan = super().plan(pop, round_no)
        brief, ctx = self._brief(pop, round_no, rule_plan)
        data, err = self.backend.generate(STRATEGY_MANIP_SYSTEM, brief, MANIP_PLAN_SCHEMA, context=ctx)
        plan = self._apply(data, pop, round_no, rule_plan) if isinstance(data, dict) else None
        if plan is None:
            rule_plan.llm = {"used": False, "error": err or "invalid-plan", "fallback": "rule"}
            return rule_plan
        return plan

    def _apply(self, d: dict, pop: Population, round_no: int, rule_plan: Plan) -> Plan | None:
        from .segments import group_label, group_weights_to_cells
        prm = self.prm
        frame = d.get("frame") if d.get("frame") in FRAMES else rule_plan.frame
        try:
            tier = int(d.get("claim_tier", self.claim_tier))
        except (TypeError, ValueError):
            tier = self.claim_tier
        tier = max(0, min(tier, self.claim_tier))                      # ethics ceiling is hard
        groups: list[tuple[int, float]] = []
        for it in (d.get("target_groups") or [])[: prm.max_groups]:
            try:
                gid, w = int(it["id"]), float(it["weight"])
            except (TypeError, ValueError, KeyError):
                continue
            if 0 <= gid < self.n_groups and w > 0:
                groups.append((gid, w))
        if not groups:
            return None
        mix = np.array([max(0.0, float((d.get("channel_mix") or {}).get(c, 0.0) or 0.0)) for c in CHANNELS])
        if mix.sum() <= 0:
            mix = rule_plan.channel_mix.astype(np.float64)
        mix = mix / mix.sum()
        try:
            intensity = float(d.get("intensity", 1.0))
        except (TypeError, ValueError):
            intensity = 1.0
        intensity = float(np.clip(intensity, 0.2, 1.0))
        # cells inside the chosen groups split weight ∝ vulnerability × size; eligible cells only
        counts = pop.cell_state_counts().astype(np.float32)
        movable = (counts * MOVABLE[None, :]).sum(axis=1) / np.maximum(pop.cell_size, 1)
        vulnerability = self.cell_pers * (1.0 - self.cell_lit) * movable * self.yield_ema
        vulnerability[pop.cell_size < prm.min_cell_size] = 0.0
        cell_w = group_weights_to_cells(groups, vulnerability * pop.cell_size, pop)
        if cell_w.sum() <= 0:
            return None
        budget = self.budget_per_round * intensity
        cpi = float((mix * self.chan.cpi).sum())
        impressions = cell_w * budget / max(cpi, 1e-6)
        impressions = np.minimum(impressions, prm.impressions_per_capita_cap * pop.cell_size)
        cost = float(impressions.sum() * cpi)
        cell_impr = impressions[:, None] * mix[None, :]
        target = np.flatnonzero(cell_w > 0)
        target = target[np.argsort(-cell_w[target], kind="stable")]
        msg = f"[{FRAME_LABELS_KO[frame]}] ({CLAIM_TIER_LABELS_KO[CLAIM_TIERS[tier]]}) {TEMPLATES[(frame, tier)]}"
        llm = {"used": True, "groups": [{"id": g, "label": group_label(g), "weight": round(w, 3)} for g, w in groups],
               "intensity": intensity, "rationale": str(d.get("rationale", ""))[:300], "model": getattr(self.backend, "strategy_model", None),
               "memory": ({"runs": self.memory.n_runs - (1 if self.run_name in self.memory.data["runs"] else 0), "rounds": self.memory.n_rounds}
                          if (self.memory is not None and self.prm.use_memory) else None)}
        return Plan(round_no=round_no, frame=frame, claim_tier=tier, falsehood=tier / 2.0, target_cells=target,
                    cell_impressions=cell_impr, impressions_total=float(cell_impr.sum()), cost=cost, message=msg,
                    channel_mix=mix.astype(np.float32), llm=llm)

    # ---- learning hooks ---------------------------------------------------
    def learn(self, exposed_cells: np.ndarray, moved_up_cells: np.ndarray):
        super().learn(exposed_cells, moved_up_cells)
        self.last_exposed_cells = np.bincount(exposed_cells, minlength=N_CELLS).astype(np.float64)
        self.last_up_cells = np.bincount(moved_up_cells, minlength=N_CELLS).astype(np.float64)

    def observe(self, rec: dict) -> None:
        from .segments import group_label
        m = rec["manipulator"]
        llm = m.get("llm") or {}
        groups = [g["label"] for g in llm.get("groups", [])] if llm.get("used") else m.get("target_cells_label_top5", [])[:3]
        mix = sorted(m["channel_mix"].items(), key=lambda kv: -kv[1])[:3]
        prev_a = self.history[-1]["A"] if self.history else None
        # per-group yields of this round (targeted groups only) for cross-run memory
        ex_g = np.bincount(self.cell_group, weights=self.last_exposed_cells, minlength=self.n_groups)
        up_g = np.bincount(self.cell_group, weights=self.last_up_cells, minlength=self.n_groups)
        group_yield = {group_label(g): [int(ex_g[g]), int(up_g[g])] for g in np.flatnonzero(ex_g > 0)}
        self.history.append({
            "group_yield": group_yield,
            "round": rec["round_no"], "frame": m["frame"], "tier": m["claim_tier"], "groups": groups,
            "channels": "/".join(f"{c}:{v:.2f}" for c, v in mix), "exposed": rec["exposure"]["n_exposed"],
            "moved_up": rec["n_moved_up_total"], "detected": rec["defense"]["detected"], "reverted": rec["defense"]["n_reverted"],
            "A": rec["support"]["A"], "dA": (rec["support"]["A"] - prev_a) if prev_a is not None else 0.0,
            "llm_used": bool(llm.get("used")), "rationale": llm.get("rationale", "")})

    def state_dict(self) -> dict:
        return {"yield_ema": self.yield_ema.tolist(), "history": self.history,
                "last_exposed_cells": self.last_exposed_cells.tolist(), "last_up_cells": self.last_up_cells.tolist()}

    def load_state_dict(self, d: dict):
        self.yield_ema = np.array(d["yield_ema"], dtype=np.float32)
        self.history = list(d.get("history", []))
        if "last_exposed_cells" in d:
            self.last_exposed_cells = np.array(d["last_exposed_cells"], dtype=np.float64)
            self.last_up_cells = np.array(d["last_up_cells"], dtype=np.float64)
