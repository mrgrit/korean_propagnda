"""APIBackend — Anthropic API path (§2.1 scale-out alternative). Not selected by default.

generate(): synchronous Messages call with structured JSON output (strategists, single reactions).
react():   mode="batch" → Message Batches API (50% price, ≤100k requests/batch), one request per
           persona batch; mode="realtime" → AsyncAnthropic with a concurrency semaphore.
Frozen, cache-marked system prompt. Default models follow the plan (§2.1: 저비용 모델 우선).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .backend import BATCH_SCHEMA, RESPONSE_SCHEMA, AgentBackend, L2Request, L2Response, coerce_response
from .prompts import SYSTEM_PROMPT, batch_prompt, user_prompt

_PRICE = {  # USD per MTok (input, output) — informational estimate only
    "claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-5": (2.0, 10.0), "claude-opus-5": (5.0, 25.0),
}


class APIBackend(AgentBackend):
    name = "api"

    def __init__(self, model: str = "claude-haiku-4-5", mode: str = "batch", concurrency: int = 16,
                 max_tokens: int = 2048, poll_s: int = 20, batch_timeout_s: int = 6 * 3600,
                 batch_size: int = 8, strategy_model: str | None = "claude-sonnet-5"):
        import anthropic  # deferred so mock/cc runs never need the SDK configured
        self._anthropic = anthropic
        self.model = model
        self.mode = mode
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.poll_s = poll_s
        self.batch_timeout_s = batch_timeout_s
        self.batch_size = max(1, int(batch_size))
        self.strategy_model = strategy_model or model
        self.client = anthropic.Anthropic()
        self.aclient = anthropic.AsyncAnthropic()
        self.stats = {"requests": 0, "calls": 0, "ok": 0, "fallback": 0, "input_tokens": 0, "output_tokens": 0,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "batches": [], "strategy_calls": 0}

    def _params(self, system: str, user: str, schema: dict[str, Any], model: str, max_tokens: int | None = None) -> dict[str, Any]:
        return {
            "model": model, "max_tokens": max_tokens or self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }

    def _account(self, usage) -> None:
        if usage is None:
            return
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.stats[k] += int(getattr(usage, k, 0) or 0)

    def _parse(self, message) -> tuple[dict[str, Any] | None, str | None]:
        self._account(getattr(message, "usage", None))
        if getattr(message, "stop_reason", None) == "refusal":
            self.stats["fallback"] += 1
            return None, "refusal"
        text = next((b.text for b in message.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            self.stats["fallback"] += 1
            return None, "no-json"
        self.stats["ok"] += 1
        return (data if isinstance(data, dict) else None), (None if isinstance(data, dict) else "no-json")

    def generate(self, system: str, user: str, schema: dict[str, Any], *, model: str | None = None,
                 context: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
        A = self._anthropic
        is_strategy = (context or {}).get("kind") in ("manipulator", "defender")
        mdl = model or (self.strategy_model if is_strategy else self.model)
        if is_strategy:
            self.stats["strategy_calls"] += 1
        self.stats["calls"] += 1
        try:
            msg = self.client.messages.create(**self._params(system, user, schema, mdl, 4096 if is_strategy else None))
            return self._parse(msg)
        except A.RateLimitError as e:
            self.stats["fallback"] += 1
            return None, f"rate_limit:{e.status_code}"
        except A.APIStatusError as e:
            self.stats["fallback"] += 1
            return None, f"api:{e.status_code}"
        except A.APIConnectionError:
            self.stats["fallback"] += 1
            return None, "connection"

    # ---- L2 reactions --------------------------------------------------------
    def _batch_params(self, batch: list[L2Request]) -> dict[str, Any]:
        if len(batch) == 1:
            return self._params(SYSTEM_PROMPT, user_prompt(batch[0]), RESPONSE_SCHEMA, self.model)
        return self._params(SYSTEM_PROMPT, batch_prompt(batch), BATCH_SCHEMA, self.model)

    def _map(self, batch: list[L2Request], data: dict[str, Any] | None, err: str | None) -> list[L2Response]:
        if len(batch) == 1:
            r = coerce_response(batch[0].voter_id, data, "api")
            if err and r.error is None:
                r.error = err
            return [r]
        by_idx: dict[int, dict[str, Any]] = {}
        for item in (data or {}).get("reactions") or []:
            if isinstance(item, dict) and isinstance(item.get("idx"), int):
                by_idx.setdefault(item["idx"], item)
        out = []
        for i, req in enumerate(batch):
            r = coerce_response(req.voter_id, by_idx.get(i), "api")
            if i not in by_idx:
                r.error = err or "missing-in-batch"
            out.append(r)
        return out

    async def _one(self, sem: asyncio.Semaphore, batch: list[L2Request]) -> list[L2Response]:
        A = self._anthropic
        async with sem:
            try:
                msg = await self.aclient.messages.create(**self._batch_params(batch))
                data, err = self._parse(msg)
                return self._map(batch, data, err)
            except (A.APIStatusError, A.APIConnectionError) as e:
                self.stats["fallback"] += 1
                return self._map(batch, None, f"api:{getattr(e, 'status_code', 'conn')}")

    async def _realtime(self, batches: list[list[L2Request]]) -> list[L2Response]:
        sem = asyncio.Semaphore(self.concurrency)
        res = await asyncio.gather(*(self._one(sem, b) for b in batches))
        return [r for batch in res for r in batch]

    def _batch_api(self, batches: list[list[L2Request]]) -> list[L2Response]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        by_id = {f"r{b[0].round_no}-b{i}": b for i, b in enumerate(batches)}
        job = self.client.messages.batches.create(
            requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**self._batch_params(b)))
                      for cid, b in by_id.items()])
        t0 = time.time()
        while True:
            st = self.client.messages.batches.retrieve(job.id)
            if st.processing_status == "ended":
                break
            if time.time() - t0 > self.batch_timeout_s:
                self.client.messages.batches.cancel(job.id)
                break
            time.sleep(self.poll_s)
        self.stats["batches"].append({"id": job.id, "n": len(by_id), "elapsed_s": round(time.time() - t0)})
        out: dict[str, list[L2Response]] = {}
        try:
            for res in self.client.messages.batches.results(job.id):
                b = by_id.get(res.custom_id)
                if b is None:
                    continue
                if res.result.type == "succeeded":
                    data, err = self._parse(res.result.message)
                    out[res.custom_id] = self._map(b, data, err)
                else:
                    self.stats["fallback"] += 1
                    out[res.custom_id] = self._map(b, None, res.result.type)
        except Exception:  # noqa: BLE001 - results unavailable (cancelled/expired)
            pass
        return [r for cid, b in by_id.items() for r in (out.get(cid) or self._map(b, None, "missing"))]

    def react(self, requests: list[L2Request]) -> list[L2Response]:
        if not requests:
            return []
        self.stats["requests"] += len(requests)
        bs = self.batch_size
        batches = [requests[i:i + bs] for i in range(0, len(requests), bs)]
        if self.mode == "batch":
            return self._batch_api(batches)
        return asyncio.run(self._realtime(batches))

    def usage(self) -> dict:
        p_in, p_out = _PRICE.get(self.model, (0.0, 0.0))
        est = (self.stats["input_tokens"] * p_in + self.stats["output_tokens"] * p_out
               + self.stats["cache_read_input_tokens"] * p_in * 0.1
               + self.stats["cache_creation_input_tokens"] * p_in * 1.25) / 1e6
        if self.mode == "batch":
            est *= 0.5
        return dict(self.stats, model=self.model, strategy_model=self.strategy_model, mode=self.mode,
                    batch_size=self.batch_size, est_cost_usd=round(est, 4))
