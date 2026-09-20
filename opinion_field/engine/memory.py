"""P1 cross-run strategy memory ("학습"): aggregate, group-level outcomes of past runs that the LLM
strategists read as '누적 경험'. Stores only aggregates (frame / tier / group / channel yields,
defender prebunk effect) — never individuals (§10)."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


_TIER_NAMES = ["factual", "misleading", "fabricated"]


def _tier_code(t) -> int:
    """History stores the tier as its name ('fabricated') or code (2); normalise to 0..2."""
    if isinstance(t, str):
        return _TIER_NAMES.index(t) if t in _TIER_NAMES else 0
    try:
        return max(0, min(2, int(t)))
    except (TypeError, ValueError):
        return 0


class StrategyMemory:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.data: dict[str, Any] = {"version": 1, "runs": {}}
        if self.path and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - corrupt memory must not kill a run
                self.data = {"version": 1, "runs": {}}
        self.data.setdefault("runs", {})

    # ---- bookkeeping ---------------------------------------------------------
    @property
    def n_runs(self) -> int:
        return len(self.data["runs"])

    @property
    def n_rounds(self) -> int:
        return sum(len(r.get("manipulator", [])) for r in self.data["runs"].values())

    def record_run(self, run_name: str, manip_history: list[dict], defender_history: list[dict], meta: dict[str, Any]) -> None:
        if not manip_history and not defender_history:
            return
        self.data["runs"][run_name] = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "meta": meta,
                                       "manipulator": manip_history, "defender": defender_history}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")

    def _rounds(self, key: str, exclude_run: str | None = None) -> list[dict]:
        out = []
        for name, r in self.data["runs"].items():
            if name == exclude_run:
                continue
            out.extend(r.get(key, []))
        return out

    # ---- summaries for the briefs ---------------------------------------------
    def manip_summary(self, exclude_run: str | None = None, max_groups: int = 12) -> str | None:
        rows = self._rounds("manipulator", exclude_run)
        if not rows:
            return None
        by_frame: dict[str, list] = defaultdict(lambda: [0, 0, 0.0, 0])     # exposed, up, dA sum, n
        by_tier: dict[int, list] = defaultdict(lambda: [0, 0, 0, 0, 0])     # exposed, up, detected, reverted, n
        by_group: dict[str, list] = defaultdict(lambda: [0, 0])             # exposed, up
        by_chan: dict[str, list] = defaultdict(lambda: [0, 0])              # exposed, up (dominant channel)
        for h in rows:
            f = by_frame[h["frame"]]; f[0] += h["exposed"]; f[1] += h["moved_up"]; f[2] += h.get("dA", 0.0); f[3] += 1
            t = by_tier[_tier_code(h.get("tier"))]; t[0] += h["exposed"]; t[1] += h["moved_up"]; t[2] += int(bool(h["detected"])); t[3] += h["reverted"]; t[4] += 1
            for lbl, (ex, up) in (h.get("group_yield") or {}).items():
                g = by_group[lbl]; g[0] += ex; g[1] += up
            dom = (h.get("channels") or "").split("/")[0].split(":")[0]
            if dom:
                c = by_chan[dom]; c[0] += h["exposed"]; c[1] += h["moved_up"]
        y = lambda ex, up: (up / ex) if ex else 0.0
        lines = [f"## 누적 경험 (지난 실행 {self.n_runs - (1 if exclude_run in self.data['runs'] else 0)}회, {len(rows)}라운드 집계)",
                 "프레임 | 라운드 | 노출 | A방향 이동 | 수율 | 평균 ΔA"]
        for k, v in sorted(by_frame.items(), key=lambda kv: -y(kv[1][0], kv[1][1])):
            lines.append(f"{k} | {v[3]} | {v[0]:,} | {v[1]:,} | {y(v[0], v[1]):.4f} | {v[2]/max(v[3],1):+.4f}")
        lines.append("주장유형 | 라운드 | 수율 | 방어 탐지율 | 되돌림/이동")
        for k, v in sorted(by_tier.items()):
            lines.append(f"{k} | {v[4]} | {y(v[0], v[1]):.4f} | {v[2]/max(v[4],1):.2f} | {v[3]/max(v[1],1):.2f}")
        top = sorted(((k, v) for k, v in by_group.items() if v[0] >= 200), key=lambda kv: -y(kv[1][0], kv[1][1]))[:max_groups]
        if top:
            lines.append("집단(표적 경험) | 노출 | 수율")
            for k, v in top:
                lines.append(f"{k} | {v[0]:,} | {y(v[0], v[1]):.4f}")
        if by_chan:
            lines.append("주채널 | 노출 | 수율")
            for k, v in sorted(by_chan.items(), key=lambda kv: -y(kv[1][0], kv[1][1])):
                lines.append(f"{k} | {v[0]:,} | {y(v[0], v[1]):.4f}")
        return "\n".join(lines)

    def defender_summary(self, exclude_run: str | None = None) -> str | None:
        rows = [h for h in self._rounds("defender", exclude_run) if h.get("corr_exposed")]
        if not rows:
            return None
        buckets: dict[str, list] = defaultdict(lambda: [0, 0, 0.0, 0])   # corr_exposed, reverted, next/now ratio sum, n
        def b(p):
            return "0" if p <= 0 else ("0~0.3" if p <= 0.3 else ("0.3~0.6" if p <= 0.6 else "0.6+"))
        for h in rows:
            k = buckets[b(float(h.get("prebunk", 0.0)))]
            k[0] += h["corr_exposed"]; k[1] += h["reverted"]; k[3] += 1
            if h.get("next_up") is not None and h.get("moved_up_now"):
                k[2] += h["next_up"] / max(h["moved_up_now"], 1)
        lines = [f"## 누적 경험 (지난 정정 {len(rows)}라운드)", "예방접종 비율 | 라운드 | 정정 노출 | 되돌림률 | 다음 라운드 이동/이번 이동"]
        for k, v in sorted(buckets.items()):
            lines.append(f"{k} | {v[3]} | {v[0]:,} | {v[1]/max(v[0],1):.3f} | {v[2]/max(v[3],1):.2f}")
        return "\n".join(lines)
