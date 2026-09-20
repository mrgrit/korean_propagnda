"""Defender (§5.4). RuleDefender = P0 rule-based detect → correct → revert + inoculate.
LLMDefender (P1) keeps rule-based detection but lets an LLM strategist allocate the correction
budget across population groups / channels and choose a prebunk (inoculation-only) share.

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
    llm: dict | None = None

    def summary(self) -> dict:
        return {"detected": self.detected, "p_detect": round(self.p_detect, 3),
                "impressions": round(self.impressions, 1), "cost": round(self.cost, 1),
                "n_corr_exposed": self.n_corr_exposed, "n_reverted": self.n_reverted,
                "n_inoculated": self.n_inoculated, "llm": self.llm}


class RuleDefender:
    def __init__(self, prm: DefenderParams, chan: ChannelTable):
        self.prm = prm
        self.chan = chan
        w = chan.correction_trust.astype(np.float64)
        self.channel_mix = w / w.sum()          # corrections travel on high-correction-trust channels

    # ---- pieces (shared with the LLM defender) --------------------------
    def detect(self, pop: Population, plan: Plan, exposed_idx: np.ndarray, rng: np.random.Generator) -> tuple[bool, float]:
        prm = self.prm
        volume = min(len(exposed_idx) / max(pop.n, 1) * 5.0, 1.0)
        logit = (prm.detect_bias + prm.detect_falsehood_w * plan.falsehood
                 + prm.detect_strength_w * prm.strength + prm.detect_volume_w * volume)
        p = float(1.0 / (1.0 + np.exp(-logit)))
        return bool(rng.random() < p), p

    def budget(self, plan: Plan) -> float:
        return self.prm.strength * self.prm.budget_ratio * plan.impressions_total

    def correct(self, pop: Population, plan: Plan, cell_impr: np.ndarray, state_new: np.ndarray, moved_up_mask: np.ndarray,
                round_no: int, rng: np.random.Generator, revert: bool = True, inoc_mult: float = 1.0) -> DefenseResult:
        prm = self.prm
        cost = float((cell_impr.sum(axis=0) * self.chan.cpi).sum())
        total = float(cell_impr.sum())
        ex = expose(pop, cell_impr, rng, trust_mult=self.chan.correction_trust)
        if len(ex.idx) == 0:
            return DefenseResult(True, 0.0, total, cost)
        rev_idx = np.empty(0, dtype=np.int64)
        if revert:
            lit = pop.literacy[ex.idx]
            recent = (pop.last_move[ex.idx] >= round_no - prm.recency_rounds) & (pop.move_dir[ex.idx] > 0)
            recent |= moved_up_mask[ex.idx]
            p_rev = (prm.revert_base * prm.strength * ex.trust_eff * (0.5 + 0.5 * lit)
                     * (0.5 + 0.5 * plan.falsehood))
            rev = recent & (rng.random(len(ex.idx)) < p_rev)
            rev_idx = ex.idx[rev]
            state_new[rev_idx] = STEP_DOWN[state_new[rev_idx]]
        gain = prm.inoculation_gain * inoc_mult * ex.trust_eff
        pop.inoculation[ex.idx] = np.minimum(1.0, pop.inoculation[ex.idx] + gain).astype(np.float32)
        return DefenseResult(True, 0.0, total, cost, int(len(ex.idx)), int(len(rev_idx)), int(len(ex.idx)), rev_idx)

    def allocate_rule(self, pop: Population, plan: Plan, exposed_idx: np.ndarray, total: float) -> np.ndarray:
        exp_c = np.bincount(pop.cell[exposed_idx], minlength=N_CELLS).astype(np.float64)
        alloc = np.zeros(N_CELLS)
        alloc[plan.target_cells] = exp_c[plan.target_cells]
        if alloc.sum() <= 0:
            return np.zeros((N_CELLS, N_CHANNELS))
        alloc /= alloc.sum()
        return alloc[:, None] * self.channel_mix[None, :] * total

    # ---- P0 behaviour ---------------------------------------------------------
    def act(self, pop: Population, plan: Plan, exposed_idx: np.ndarray, state_new: np.ndarray,
            moved_up_mask: np.ndarray, round_no: int, rng: np.random.Generator) -> DefenseResult:
        if not self.prm.enabled or self.prm.strength <= 0:
            return DefenseResult(False, 0.0)
        detected, p = self.detect(pop, plan, exposed_idx, rng)
        if not detected:
            return DefenseResult(False, p)
        cell_impr = self.allocate_rule(pop, plan, exposed_idx, self.budget(plan))
        if cell_impr.sum() <= 0:
            return DefenseResult(True, p)
        res = self.correct(pop, plan, cell_impr, state_new, moved_up_mask, round_no, rng)
        res.p_detect = p
        return res

    def observe(self, rec: dict) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, d: dict) -> None:
        pass


# ============================================================================ P1: LLM strategist
DEF_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "target_groups": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "weight": {"type": "number"}},
            "required": ["id", "weight"], "additionalProperties": False}},
        "channel_mix": {"type": "object", "properties": {c: {"type": "number"} for c in CHANNELS},
                        "required": list(CHANNELS), "additionalProperties": False},
        "prebunk_share": {"type": "number", "description": "예방접종(사전)에 쓸 예산 비율 0~0.8"},
        "intensity": {"type": "number", "description": "정정 예산 사용 비율 0.2~1.0"},
        "rationale": {"type": "string"},
    },
    "required": ["target_groups", "channel_mix", "prebunk_share", "intensity", "rationale"],
    "additionalProperties": False,
}


class LLMDefender(RuleDefender):
    def __init__(self, prm: DefenderParams, chan: ChannelTable, backend, pop: Population, manip):
        super().__init__(prm, chan)
        from .segments import CELL_GROUP, N_GROUPS
        self.backend = backend
        self.manip = manip                 # for static cell stats (pers/lit/chan)
        self.history: list[dict] = []
        self.cell_group = CELL_GROUP
        self.n_groups = N_GROUPS

    def _brief(self, pop: Population, plan: Plan, exposed_idx: np.ndarray, moved_up_mask: np.ndarray, round_no: int, total: float) -> tuple[str, dict]:
        from .segments import group_stats
        from .manipulator import MOVABLE
        counts = pop.cell_state_counts().astype(np.float32)
        movable = (counts * MOVABLE[None, :]).sum(axis=1) / np.maximum(pop.cell_size, 1)
        exp_c = np.bincount(pop.cell[exposed_idx], minlength=N_CELLS).astype(np.float64)
        up_c = np.bincount(pop.cell[np.flatnonzero(moved_up_mask)], minlength=N_CELLS).astype(np.float64)
        inoc_c = np.bincount(pop.cell, weights=pop.inoculation, minlength=N_CELLS)
        groups = group_stats(pop, self.manip.cell_pers, self.manip.cell_lit, self.manip.cell_chan, movable, np.ones(N_CELLS, dtype=np.float32),
                             {"exposed": exp_c, "moved_up": up_c, "inoculation_sum": inoc_c})
        for g in groups:
            g["inoculated_frac"] = round(g["inoculation_sum"] / max(g["size"], 1), 3)
        shares = pop.support_shares()
        lines = [f"# 라운드 {round_no} 방어자 브리핑 — 조작 노출 탐지됨",
                 f"탐지된 조작: 프레임 {plan.frame}, 주장 유형 {plan.claim_tier} (허위도 {plan.falsehood}), 노출 {len(exposed_idx):,}명, 채널 비중 " + ", ".join(f"{c} {v:.2f}" for c, v in zip(CHANNELS, plan.channel_mix)),
                 f"현재 A {shares['A']:.4f} / B {shares['B']:.4f}. 이번 라운드 A방향 이동 {int(moved_up_mask.sum()):,}명.",
                 f"정정 예산(노출 수) {total:,.0f}. 채널별 정정 신뢰 계수: " + ", ".join(f"{c} {v:.1f}" for c, v in zip(CHANNELS, self.chan.correction_trust)),
                 "사후 정정(debunk)은 최근 A방향으로 움직인 사람을 되돌리고, 예방접종(prebunk)은 아직 노출되지 않은 취약 집단의 다음 라운드 저항을 높입니다.",
                 "", "## 지난 정정 기록", "| R | 표적 집단(상위3) | 예방접종 비율 | 정정 노출 | 되돌림 | 다음 라운드 A방향 이동 |", "|---|---|---|---|---|---|"]
        if not self.history:
            lines.append("| - | (기록 없음) | | | | |")
        for h in self.history[-8:]:
            lines.append(f"| {h['round']} | {', '.join(h['groups'][:3])} | {h['prebunk']:.2f} | {h['corr_exposed']:,} | {h['reverted']:,} | {h.get('next_up', '?')} |")
        lines += ["", f"## 인구 집단 ({len(groups)}개) — id | 집단 | 규모 | 리터러시 | 이동가능비율 | 이번 노출 | 이번 A이동 | 면역비율 | 선호채널"]
        for g in groups:
            lines.append(f"{g['id']} | {g['label']} | {g['size']:,} | {g['literacy']:.2f} | {g['movable']:.2f} | {g['exposed']:,} | {g['moved_up']:,} | {g['inoculated_frac']:.2f} | {'/'.join(g['top_channels'])}")
        lines += ["", f"## 출력: JSON — target_groups(최대 {self.prm.max_groups}개, id·weight>0), channel_mix(6채널 비중), prebunk_share(0~0.8), intensity(0.2~1.0), rationale(한 문장)."]
        hint = {c: round(float(v), 3) for c, v in zip(CHANNELS, self.channel_mix)}
        ctx = {"kind": "defender", "groups": groups, "channel_hint": hint, "max_groups": self.prm.max_groups}
        return "\n".join(lines), ctx

    def act(self, pop: Population, plan: Plan, exposed_idx: np.ndarray, state_new: np.ndarray,
            moved_up_mask: np.ndarray, round_no: int, rng: np.random.Generator) -> DefenseResult:
        from ..agents.prompts import STRATEGY_DEF_SYSTEM
        from .segments import group_label, group_weights_to_cells
        if not self.prm.enabled or self.prm.strength <= 0:
            return DefenseResult(False, 0.0)
        detected, p = self.detect(pop, plan, exposed_idx, rng)
        if not detected:
            return DefenseResult(False, p)
        total = self.budget(plan)
        brief, ctx = self._brief(pop, plan, exposed_idx, moved_up_mask, round_no, total)
        data, err = self.backend.generate(STRATEGY_DEF_SYSTEM, brief, DEF_PLAN_SCHEMA, context=ctx)
        groups: list[tuple[int, float]] = []
        if isinstance(data, dict):
            for it in (data.get("target_groups") or [])[: self.prm.max_groups]:
                try:
                    gid, w = int(it["id"]), float(it["weight"])
                except (TypeError, ValueError, KeyError):
                    continue
                if 0 <= gid < self.n_groups and w > 0:
                    groups.append((gid, w))
        if not groups:                                   # fallback to the rule allocation
            cell_impr = self.allocate_rule(pop, plan, exposed_idx, total)
            if cell_impr.sum() <= 0:
                return DefenseResult(True, p)
            res = self.correct(pop, plan, cell_impr, state_new, moved_up_mask, round_no, rng)
            res.p_detect = p
            res.llm = {"used": False, "error": err or "invalid-plan", "fallback": "rule"}
            return res
        mix = np.array([max(0.0, float((data.get("channel_mix") or {}).get(c, 0.0) or 0.0)) for c in CHANNELS])
        mix = mix / mix.sum() if mix.sum() > 0 else self.channel_mix
        try:
            prebunk = float(np.clip(float(data.get("prebunk_share", 0.0)), 0.0, 0.8))
            intensity = float(np.clip(float(data.get("intensity", 1.0)), 0.2, 1.0))
        except (TypeError, ValueError):
            prebunk, intensity = 0.0, 1.0
        budget = total * intensity
        exp_c = np.bincount(pop.cell[exposed_idx], minlength=N_CELLS).astype(np.float64)
        # debunk: within chosen groups, cells ∝ this round's exposure; prebunk: cells ∝ size (reach the unexposed)
        w_debunk = group_weights_to_cells(groups, exp_c, pop)
        w_prebunk = group_weights_to_cells(groups, pop.cell_size.astype(np.float64), pop)
        results = []
        if (1 - prebunk) > 0 and w_debunk.sum() > 0:
            results.append(self.correct(pop, plan, w_debunk[:, None] * mix[None, :] * budget * (1 - prebunk),
                                        state_new, moved_up_mask, round_no, rng, revert=True))
        if prebunk > 0 and w_prebunk.sum() > 0:
            results.append(self.correct(pop, plan, w_prebunk[:, None] * mix[None, :] * budget * prebunk,
                                        state_new, moved_up_mask, round_no, rng, revert=False, inoc_mult=1.3))
        if not results:
            return DefenseResult(True, p)
        rev = np.concatenate([r.reverted_idx for r in results if r.reverted_idx is not None]) if any(r.reverted_idx is not None for r in results) else np.empty(0, np.int64)
        out = DefenseResult(True, p, sum(r.impressions for r in results), sum(r.cost for r in results),
                            sum(r.n_corr_exposed for r in results), int(len(rev)), sum(r.n_inoculated for r in results), rev)
        out.llm = {"used": True, "groups": [{"id": g, "label": group_label(g), "weight": round(w, 3)} for g, w in groups],
                   "prebunk_share": prebunk, "intensity": intensity, "rationale": str(data.get("rationale", ""))[:300],
                   "model": getattr(self.backend, "strategy_model", None)}
        return out

    def observe(self, rec: dict) -> None:
        d = rec["defense"]
        if self.history and "next_up" not in self.history[-1]:
            self.history[-1]["next_up"] = rec["n_moved_up_total"]
        if not d["detected"]:
            return
        llm = d.get("llm") or {}
        self.history.append({"round": rec["round_no"], "groups": [g["label"] for g in llm.get("groups", [])] if llm.get("used") else ["(규칙 배분)"],
                             "prebunk": float(llm.get("prebunk_share", 0.0)) if llm.get("used") else 0.0,
                             "corr_exposed": d["n_corr_exposed"], "reverted": d["n_reverted"], "llm_used": bool(llm.get("used"))})

    def state_dict(self) -> dict:
        return {"history": self.history}

    def load_state_dict(self, d: dict) -> None:
        self.history = list(d.get("history", []))
