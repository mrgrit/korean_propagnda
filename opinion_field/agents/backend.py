"""AgentBackend abstraction (§2.1). The engine only talks to this interface.

Implementations:
  MockBackend          — deterministic, no LLM (tests, §8-4)
  CCSessionBackend     — Claude Code headless (`claude -p`) on the account logged into THIS machine
  APIBackend           — Anthropic API (Messages / Message Batches), prompt-cached (§2.1 scale-out path)

Policy note (verified against code.claude.com/docs/en/legal-and-compliance): a subscriber
may drive their own Claude Code headlessly for ordinary use; pooling several subscription
accounts or routing around their usage limits is not permitted, so no multi-account
dispatcher exists here and none should be added. Higher throughput = APIBackend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class L2Request:
    voter_id: int
    round_no: int
    persona: dict[str, Any]        # slot-labelled persona card fields (no names, no display names)
    state: int                     # SupportState code
    state_label: str
    message: str                   # slot-labelled message description
    channel: str
    prior_p: float                 # L1 transition prob (context for mock / logging)
    kind: str = "l1"               # "l1" | "l3" (second wave)


@dataclass
class L2Response:
    voter_id: int
    support_delta: int             # −1, 0, +1
    share: bool
    reaction: str = ""
    source: str = "mock"           # mock | cc | api | fallback
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def react(self, requests: list[L2Request]) -> list[L2Response]:
        """Return one response per request, same order."""

    def usage(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "support_delta": {"type": "integer", "enum": [-1, 0, 1],
                          "description": "-1: 후보 B 쪽으로 이동, 0: 변화 없음, +1: 후보 A 쪽으로 이동"},
        "share": {"type": "boolean", "description": "이 메시지를 지인·SNS에 공유/전달할지"},
        "reaction": {"type": "string", "description": "80자 이내의 짧은 반응 (선택)"},
    },
    "required": ["support_delta", "share", "reaction"],
    "additionalProperties": False,
}


def coerce_response(voter_id: int, data: dict[str, Any] | None, source: str) -> L2Response:
    if not isinstance(data, dict):
        return L2Response(voter_id, 0, False, "", "fallback", error="no-json")
    try:
        delta = int(data.get("support_delta", 0))
    except (TypeError, ValueError):
        delta = 0
    delta = max(-1, min(1, delta))
    share = bool(data.get("share", False))
    reaction = str(data.get("reaction", ""))[:200]
    return L2Response(voter_id, delta, share, reaction, source)


def _kwargs_for(cls, spec: dict[str, Any]) -> dict[str, Any]:
    """Keep only the keys the constructor accepts (a YAML may carry keys for another backend kind)."""
    import inspect
    allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return {k: v for k, v in spec.items() if k in allowed}


def build_backend(spec: dict[str, Any], seed: int) -> AgentBackend:
    spec = dict(spec or {})
    kind = spec.pop("kind", "mock")
    if kind == "mock":
        from .mock import MockBackend
        return MockBackend(**{"seed": seed, **_kwargs_for(MockBackend, spec)})
    if kind == "cc":
        from .cc_session import CCSessionBackend
        return CCSessionBackend(**_kwargs_for(CCSessionBackend, spec))
    if kind == "api":
        from .api import APIBackend
        return APIBackend(**_kwargs_for(APIBackend, spec))
    raise ValueError(f"unknown backend kind: {kind}")
