"""Deterministic mock backend: samples from the L1 prior so the loop can be tested end-to-end."""
from __future__ import annotations

import numpy as np

from .backend import AgentBackend, L2Request, L2Response


class MockBackend(AgentBackend):
    name = "mock"

    def __init__(self, seed: int = 0, share_rate: float = 0.25, latency_s: float = 0.0):
        self.seed = seed
        self.share_rate = share_rate
        self.latency_s = latency_s
        self.calls = 0
        self.n_requests = 0

    def react(self, requests: list[L2Request]) -> list[L2Response]:
        self.calls += 1
        self.n_requests += len(requests)
        out = []
        for r in requests:
            rng = np.random.default_rng([self.seed, r.round_no, r.voter_id])
            u = rng.random()
            delta = 1 if u < r.prior_p else 0
            # a little contrarian backlash for skeptical personas facing fabricated claims
            if delta == 0 and "허위" in r.message and rng.random() < 0.05:
                delta = -1
            share = rng.random() < self.share_rate * (0.5 + r.prior_p)
            out.append(L2Response(r.voter_id, delta, bool(share), "", "mock"))
        return out

    def usage(self) -> dict:
        return {"calls": self.calls, "requests": self.n_requests, "cost_usd": 0.0}
