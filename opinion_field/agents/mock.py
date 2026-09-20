"""Deterministic mock backend: L2 reactions sampled from the L1 prior; strategist plans from the
engine-supplied context (top groups by vulnerability / exposure). No LLM, fully reproducible."""
from __future__ import annotations

from typing import Any

import numpy as np

from .backend import AgentBackend, L2Request


class MockBackend(AgentBackend):
    name = "mock"

    def __init__(self, seed: int = 0, share_rate: float = 0.25, batch_size: int = 1, fail: bool = False):
        self.seed = seed
        self.share_rate = share_rate
        self.batch_size = max(1, int(batch_size))
        self.fail = fail                      # simulate a broken strategist → engine must fall back
        self.calls = 0
        self.n_requests = 0

    def generate(self, system: str, user: str, schema: dict[str, Any], *, model: str | None = None,
                 context: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
        self.calls += 1
        ctx = context or {}
        kind = ctx.get("kind")
        if kind == "l2_batch":
            reqs: list[L2Request] = ctx["requests"]
            self.n_requests += len(reqs)
            out = []
            for i, r in enumerate(reqs):
                rng = np.random.default_rng([self.seed, r.round_no, r.voter_id])
                u = rng.random()
                delta = 1 if u < r.prior_p else 0
                if delta == 0 and "허위" in r.message and rng.random() < 0.05:
                    delta = -1
                share = rng.random() < self.share_rate * (0.5 + r.prior_p)
                out.append({"idx": i, "support_delta": delta, "share": bool(share), "reaction": ""})
            return {"reactions": out}, None
        if self.fail:
            return None, "mock-fail"
        if kind == "manipulator":
            groups = sorted(ctx["groups"], key=lambda g: -g["vulnerability"])[: ctx.get("max_groups", 8)]
            return {"frame": ctx["best_frame"], "claim_tier": ctx["max_tier"],
                    "target_groups": [{"id": g["id"], "weight": round(float(g["vulnerability"]), 4) or 1.0} for g in groups],
                    "channel_mix": ctx["channel_hint"], "intensity": 1.0,
                    "rationale": "mock: 취약도 상위 집단·친화도 최고 프레임·허용 최대 유형"}, None
        if kind == "defender":
            groups = sorted(ctx["groups"], key=lambda g: -g["exposed"])[: ctx.get("max_groups", 8)]
            return {"target_groups": [{"id": g["id"], "weight": float(g["exposed"]) or 1.0} for g in groups],
                    "channel_mix": ctx["channel_hint"], "prebunk_share": 0.3, "intensity": 1.0,
                    "rationale": "mock: 노출 상위 집단에 정정, 30%는 예방접종"}, None
        if kind == "tone":
            return {"tone": f"mock 톤 (age {ctx.get('age')}, block {ctx.get('block')})", "interests": "mock 관심사"}, None
        return None, f"mock: unknown kind {kind}"

    def usage(self) -> dict:
        return {"calls": self.calls, "requests": self.n_requests, "cost_usd": 0.0}
