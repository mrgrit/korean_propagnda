"""CCSessionBackend — drives the Claude Code CLI headlessly on the account logged into this machine.

    claude -p "<prompt>" --output-format json --json-schema <schema> --tools "" --max-turns 3
           --model <alias> --system-prompt <text> --no-session-persistence --strict-mcp-config
           --disable-slash-commands

Verified against code.claude.com/docs/en/headless.md and cli-reference.md (2026-09):
the JSON result carries `result`, `structured_output`, `is_error`, `usage`, `total_cost_usd`.
`--json-schema` is served by an internal structured-output tool turn (stop_reason "tool_use",
num_turns 2), so --max-turns must be ≥ 2. `--bare` is NOT used: it disables OAuth auth.

Scope, deliberately: ONE account — whatever `claude` resolves to in this environment
(optionally a CLAUDE_CONFIG_DIR passed through by the operator). Per Anthropic's
legal-and-compliance page, a subscriber may automate their own Claude Code for ordinary
use; pooling several subscription accounts or routing around usage limits is not
permitted, so this backend has no account list and no cross-account dispatch.
When the account's limit is reached the backend backs off and, if the wait exceeds
`max_wait_s`, returns fallback responses (delta 0) flagged `source="fallback"` so the
round can checkpoint and be resumed later.

P1: `batch_size` personas per call (default 8), `strategy_model` for the manipulator/defender
strategists (default sonnet; 2 calls per round), `MAX_THINKING_TOKENS` passthrough.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .backend import AgentBackend, L2Request, L2Response

_LIMIT_MARKERS = ("rate limit", "usage limit", "limit reached", "429", "overloaded", "too many requests",
                  "exceeded", "quota")


class CCSessionBackend(AgentBackend):
    name = "cc"

    def __init__(self, model: str = "haiku", concurrency: int = 4, timeout_s: int = 180,
                 claude_bin: str = "claude", config_dir: str | None = None, max_wait_s: int = 900,
                 backoff_s: int = 30, max_budget_usd: float | None = None, max_turns: int = 3,
                 debug_dir: str | None = None, thinking_tokens: int = 0, batch_size: int = 8,
                 strategy_model: str | None = "sonnet", strategy_thinking_tokens: int = 2048):
        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.timeout_s = timeout_s
        self.claude_bin = shutil.which(claude_bin) or claude_bin
        self.config_dir = config_dir
        self.max_wait_s = max_wait_s
        self.backoff_s = backoff_s
        self.max_budget_usd = max_budget_usd
        self.max_turns = max_turns
        self.debug_dir = debug_dir
        self.thinking_tokens = int(thinking_tokens)   # 0 → MAX_THINKING_TOKENS=0 (Haiku ~5s instead of ~10s per call)
        self.batch_size = max(1, int(batch_size))
        self.strategy_model = strategy_model or model
        self.strategy_thinking_tokens = int(strategy_thinking_tokens)
        self.stats = {"calls": 0, "ok": 0, "fallback": 0, "limit_hits": 0, "cost_usd_reported": 0.0,
                      "input_tokens": 0, "output_tokens": 0, "strategy_calls": 0, "personas": 0}
        self._limited_until = 0.0

    def _env(self, thinking: int) -> dict[str, str]:
        env = dict(os.environ)
        if self.config_dir:
            env["CLAUDE_CONFIG_DIR"] = self.config_dir
        env["MAX_THINKING_TOKENS"] = str(thinking)
        env.pop("CLAUDECODE", None)   # allow launching from inside an interactive Claude Code session
        return env

    def _cmd(self, system: str, user: str, schema: dict[str, Any], model: str) -> list[str]:
        cmd = [self.claude_bin, "-p", user, "--output-format", "json",
               "--json-schema", json.dumps(schema, ensure_ascii=False),
               "--tools", "", "--max-turns", str(self.max_turns), "--model", model,
               "--system-prompt", system,
               "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"]
        if self.max_budget_usd is not None:
            cmd += ["--max-budget-usd", str(self.max_budget_usd)]
        return cmd

    def _dump(self, tag: str, proc, err_text: str) -> None:
        if not self.debug_dir:
            return
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            with open(os.path.join(self.debug_dir, f"{tag}_{int(time.time()*1000)}.txt"), "w", encoding="utf-8") as f:
                f.write(f"ERR: {err_text}\nRC: {proc.returncode}\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n")
        except OSError:
            pass

    def generate(self, system: str, user: str, schema: dict[str, Any], *, model: str | None = None,
                 context: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
        is_strategy = (context or {}).get("kind") in ("manipulator", "defender")
        mdl = model or (self.strategy_model if is_strategy else self.model)
        thinking = self.strategy_thinking_tokens if is_strategy else self.thinking_tokens
        if is_strategy:
            self.stats["strategy_calls"] += 1
        tag = (context or {}).get("kind", "call")
        deadline = time.time() + self.max_wait_s
        while True:
            wait = self._limited_until - time.time()
            if wait > 0:
                if time.time() + wait > deadline:
                    self.stats["fallback"] += 1
                    return None, "limit-wait-exceeded"
                time.sleep(min(wait, 5.0))
                continue
            self.stats["calls"] += 1
            try:
                proc = subprocess.run(self._cmd(system, user, schema, mdl), capture_output=True, text=True,
                                      timeout=self.timeout_s, env=self._env(thinking))
            except subprocess.TimeoutExpired:
                self.stats["fallback"] += 1
                return None, "timeout"
            except FileNotFoundError:
                self.stats["fallback"] += 1
                return None, "claude-cli-not-found"
            try:
                data = json.loads(proc.stdout) if proc.stdout.strip() else None
            except json.JSONDecodeError:
                data = None
            err_text = ""
            if isinstance(data, dict) and data.get("is_error"):
                err_text = str(data.get("result") or data.get("terminal_reason") or "is_error")
                if data.get("api_error_status") == 429:
                    err_text += " rate limit 429"
            elif not isinstance(data, dict):
                err_text = (f"rc={proc.returncode} stdout={proc.stdout.strip()[:200]!r} "
                            f"stderr={proc.stderr.strip()[:300]!r}")
            if err_text:
                self._dump(tag, proc, err_text)
            if err_text and any(m in err_text.lower() for m in _LIMIT_MARKERS):
                self.stats["limit_hits"] += 1
                self._limited_until = time.time() + self.backoff_s
                self.backoff_s = min(self.backoff_s * 2, 600)
                continue
            if isinstance(data, dict) and not data.get("is_error"):
                usage = data.get("usage") or {}
                self.stats["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                self.stats["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                self.stats["cost_usd_reported"] += float(data.get("total_cost_usd", 0.0) or 0.0)
                so = data.get("structured_output")
                if so is None and isinstance(data.get("result"), str):
                    try:
                        so = json.loads(data["result"])
                    except json.JSONDecodeError:
                        so = None
                if isinstance(so, dict):
                    self.stats["ok"] += 1
                    return so, None
                self.stats["fallback"] += 1
                return None, "no-json"
            self.stats["fallback"] += 1
            return None, (err_text or "unknown")[:200]

    def react(self, requests: list[L2Request]) -> list[L2Response]:
        if not requests:
            return []
        self.stats["personas"] += len(requests)
        bs = self.batch_size
        batches = [requests[i:i + bs] for i in range(0, len(requests), bs)]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            results = list(ex.map(self._react_batch, batches))
        return [r for batch in results for r in batch]

    def usage(self) -> dict:
        return dict(self.stats, model=self.model, strategy_model=self.strategy_model,
                    concurrency=self.concurrency, batch_size=self.batch_size)
