"""APIBackend — Anthropic API path (§2.1 scale-out alternative). Not selected by default.

mode="batch"    : Message Batches API (50% price, ≤100k requests/batch) — best fit for a
                  round's L2 cohort, which is not latency-sensitive.
mode="realtime" : AsyncAnthropic with a concurrency semaphore.
Both use a frozen, cache-marked system prompt and structured JSON output.
Default model follows the plan (§2.1: 저비용 모델 우선): claude-haiku-4-5.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .backend import RESPONSE_SCHEMA, AgentBackend, L2Request, L2Response, coerce_response
from .prompts import SYSTEM_PROMPT, user_prompt

_PRICE = {  # USD per MTok (input, output) — informational estimate only
    "claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-5": (2.0, 10.0), "claude-opus-5": (5.0, 25.0),
}


class APIBackend(AgentBackend):
    name = "api"

    def __init__(self, model: str = "claude-haiku-4-5", mode: str = "batch", concurrency: int = 16,
                 max_tokens: int = 256, poll_s: int = 20, batch_timeout_s: int = 6 * 3600):
        import anthropic  # deferred so mock/cc runs never need the SDK configured
        self._anthropic = anthropic
        self.model = model
        self.mode = mode
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.poll_s = poll_s
        self.batch_timeout_s = batch_timeout_s
        self.client = anthropic.Anthropic()
        self.aclient = anthropic.AsyncAnthropic()
        self.stats = {"requests": 0, "ok": 0, "fallback": 0, "input_tokens": 0, "output_tokens": 0,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "batches": []}

    # ---- shared request params ------------------------------------------
    def _params(self, req: L2Request) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_prompt(req)}],
            "output_config": {"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        }

    def _account(self, usage) -> None:
        if usage is None:
            return
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.stats[k] += int(getattr(usage, k, 0) or 0)

    def _parse(self, req: L2Request, message) -> L2Response:
        self._account(getattr(message, "usage", None))
        if getattr(message, "stop_reason", None) == "refusal":
            self.stats["fallback"] += 1
            return L2Response(req.voter_id, 0, False, "", "fallback", error="refusal")
        text = next((b.text for b in message.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        resp = coerce_response(req.voter_id, data, "api")
        self.stats["ok" if resp.error is None else "fallback"] += 1
        return resp

    # ---- realtime --------------------------------------------------------
    async def _one(self, sem: asyncio.Semaphore, req: L2Request) -> L2Response:
        A = self._anthropic
        async with sem:
            try:
                msg = await self.aclient.messages.create(**self._params(req))
                return self._parse(req, msg)
            except A.RateLimitError as e:
                self.stats["fallback"] += 1
                return L2Response(req.voter_id, 0, False, "", "fallback", error=f"rate_limit:{e.status_code}")
            except A.APIStatusError as e:
                self.stats["fallback"] += 1
                return L2Response(req.voter_id, 0, False, "", "fallback", error=f"api:{e.status_code}")
            except A.APIConnectionError:
                self.stats["fallback"] += 1
                return L2Response(req.voter_id, 0, False, "", "fallback", error="connection")

    async def _realtime(self, requests: list[L2Request]) -> list[L2Response]:
        sem = asyncio.Semaphore(self.concurrency)
        return list(await asyncio.gather(*(self._one(sem, r) for r in requests)))

    # ---- batch -----------------------------------------------------------
    def _batch(self, requests: list[L2Request]) -> list[L2Response]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        by_id = {f"r{r.round_no}-v{r.voter_id}": r for r in requests}
        batch = self.client.messages.batches.create(
            requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**self._params(r)))
                      for cid, r in by_id.items()])
        t0 = time.time()
        while True:
            b = self.client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            if time.time() - t0 > self.batch_timeout_s:
                self.client.messages.batches.cancel(batch.id)
                break
            time.sleep(self.poll_s)
        self.stats["batches"].append({"id": batch.id, "n": len(by_id), "elapsed_s": round(time.time() - t0)})
        out: dict[str, L2Response] = {}
        try:
            for res in self.client.messages.batches.results(batch.id):
                req = by_id.get(res.custom_id)
                if req is None:
                    continue
                if res.result.type == "succeeded":
                    out[res.custom_id] = self._parse(req, res.result.message)
                else:
                    self.stats["fallback"] += 1
                    out[res.custom_id] = L2Response(req.voter_id, 0, False, "", "fallback", error=res.result.type)
        except Exception as e:  # results unavailable (cancelled/expired)
            pass
        return [out.get(cid) or L2Response(r.voter_id, 0, False, "", "fallback", error="missing")
                for cid, r in by_id.items()]

    def react(self, requests: list[L2Request]) -> list[L2Response]:
        if not requests:
            return []
        self.stats["requests"] += len(requests)
        if self.mode == "batch":
            return self._batch(requests)
        return asyncio.run(self._realtime(requests))

    def usage(self) -> dict:
        p_in, p_out = _PRICE.get(self.model, (0.0, 0.0))
        est = (self.stats["input_tokens"] * p_in + self.stats["output_tokens"] * p_out
               + self.stats["cache_read_input_tokens"] * p_in * 0.1
               + self.stats["cache_creation_input_tokens"] * p_in * 1.25) / 1e6
        if self.mode == "batch":
            est *= 0.5
        return dict(self.stats, model=self.model, mode=self.mode, est_cost_usd=round(est, 4))
