"""AgentBackend abstraction (§2.1, extended in P1).

Primitive: `generate(system, user, schema, model=, context=)` → structured JSON. Everything else
(L2 persona reactions, manipulator/defender strategists) is built on it.

L2 batching (P1 "L2 확대"): `react()` packs `batch_size` persona cards into ONE call and maps the
returned array back by index — 8 personas per call cuts subscription usage ~8×.

Implementations:
  MockBackend          — deterministic, no LLM (tests)
  CCSessionBackend     — Claude Code headless (`claude -p`) on the account logged into THIS machine
  APIBackend           — Anthropic API (Messages / Message Batches)

Policy note (code.claude.com/docs/en/legal-and-compliance): a subscriber may drive their own
Claude Code headlessly for ordinary use; pooling several subscription accounts or routing around
their usage limits is not permitted, so no multi-account dispatcher exists here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
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


_REACTION_PROPS = {
    "support_delta": {"type": "integer", "enum": [-1, 0, 1],
                      "description": "-1: 후보 B 쪽으로 이동, 0: 변화 없음, +1: 후보 A 쪽으로 이동"},
    "share": {"type": "boolean", "description": "이 메시지를 지인·SNS에 공유/전달할지"},
    "reaction": {"type": "string", "description": "80자 이내의 짧은 반응 (선택)"},
}
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object", "properties": dict(_REACTION_PROPS),
    "required": ["support_delta", "share", "reaction"], "additionalProperties": False,
}
BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"reactions": {"type": "array", "items": {
        "type": "object",
        "properties": {"idx": {"type": "integer", "description": "페르소나 번호"}, **_REACTION_PROPS},
        "required": ["idx", "support_delta", "share", "reaction"], "additionalProperties": False}}},
    "required": ["reactions"], "additionalProperties": False,
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


class AgentBackend(ABC):
    name: str = "abstract"
    batch_size: int = 1
    strategy_model: str | None = None
    exhausted: bool = False            # usage limit persisted beyond max_wait_s → loop pauses at the round boundary

    def probe(self) -> bool:
        """Cheap liveness check used while paused on a usage limit. True → clear `exhausted`."""
        self.exhausted = False
        return True

    @abstractmethod
    def generate(self, system: str, user: str, schema: dict[str, Any], *, model: str | None = None,
                 context: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
        """One structured-JSON completion. Returns (data, error). `context` is engine-side metadata
        that a non-LLM backend (mock) may use; real backends ignore it."""

    # ---- L2 reactions built on generate() ---------------------------------
    def _react_batch(self, batch: list[L2Request]) -> list[L2Response]:
        from .prompts import SYSTEM_PROMPT, batch_prompt, user_prompt
        if len(batch) == 1:
            data, err = self.generate(SYSTEM_PROMPT, user_prompt(batch[0]), RESPONSE_SCHEMA,
                                      context={"kind": "l2_batch", "requests": batch})
            if isinstance(data, dict) and "reactions" in data:      # mock answers in batch shape
                data = (data["reactions"] or [None])[0]
            r = coerce_response(batch[0].voter_id, data, self.name)
            if err and r.error is None:
                r.error = err
            return [r]
        data, err = self.generate(SYSTEM_PROMPT, batch_prompt(batch), BATCH_SCHEMA,
                                  context={"kind": "l2_batch", "requests": batch})
        by_idx: dict[int, dict[str, Any]] = {}
        if isinstance(data, dict):
            for item in data.get("reactions") or []:
                if isinstance(item, dict) and isinstance(item.get("idx"), int):
                    by_idx.setdefault(item["idx"], item)
        out = []
        for i, req in enumerate(batch):
            item = by_idx.get(i)
            r = coerce_response(req.voter_id, item, self.name)
            if item is None:
                r.error = err or "missing-in-batch"
            out.append(r)
        return out

    def react(self, requests: list[L2Request]) -> list[L2Response]:
        """Sequential default; concurrent backends override."""
        out: list[L2Response] = []
        bs = max(1, int(self.batch_size))
        for i in range(0, len(requests), bs):
            out.extend(self._react_batch(requests[i:i + bs]))
        return out

    def usage(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


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
