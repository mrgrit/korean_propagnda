"""Round loop (§1.3), world state, checkpoint/resume (§8-8). Orchestrates L1 → promote → L2 → L3 → defender → commit."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.backend import AgentBackend, L2Request, L2Response
from ..agents.prompts import make_requests
from ..config import CampaignConfig
from ..observe.logging import RunLogger
from ..schema import CHANNELS, FRAMES, N_CELLS, STATE_NAMES
from .channels import ChannelTable, expose
from .defender import LLMDefender, RuleDefender
from .l1 import apply_moves, sample_moves, transition_probs
from .l3 import apply_l3, propagate
from .manipulator import LLMManipulator, RuleManipulator
from .memory import StrategyMemory
from .population import LEAN, STEP_DOWN, STEP_UP, Population
from .promote import select_promoted


class UsageLimitPaused(Exception):
    """Raised inside step() when the backend's usage window closed; the round is discarded."""


class Simulation:
    def __init__(self, cfg: CampaignConfig, pop: Population, chan: ChannelTable, backend: AgentBackend,
                 out_dir: str | Path, data_manifest: dict[str, Any] | None = None, log=print):
        self.cfg = cfg
        self.pop = pop
        self.chan = chan
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.logger = RunLogger(self.out_dir)
        self.log = log
        self.data_manifest = data_manifest or {}
        self.rng = np.random.default_rng(cfg.seed)
        budget_per_round = cfg.budget / max(cfg.rounds, 1)
        if cfg.manipulator.mode == "llm":
            self.manip = LLMManipulator(cfg.manipulator, chan, pop, cfg.ethics_level, budget_per_round, backend,
                                        cfg.goal_margin, cfg.rounds)
        else:
            self.manip = RuleManipulator(cfg.manipulator, chan, pop, cfg.ethics_level, budget_per_round)
        if cfg.defender.mode == "llm":
            self.defender = LLMDefender(cfg.defender, chan, backend, pop, self.manip)
        else:
            self.defender = RuleDefender(cfg.defender, chan)
        # cross-run strategy memory (aggregates only) — read by LLM strategists, written at run end
        self.memory = None
        if cfg.manipulator.use_memory and (cfg.manipulator.mode == "llm" or cfg.defender.mode == "llm"):
            self.memory = StrategyMemory(cfg.manipulator.memory_path)
            run_name = str(Path(out_dir).resolve().relative_to(Path("runs").resolve())) if Path("runs").resolve() in Path(out_dir).resolve().parents else Path(out_dir).name
            for agent in (self.manip, self.defender):
                if hasattr(agent, "memory"):
                    agent.memory = self.memory
                    agent.run_name = run_name
            self.run_name = run_name
        self.round_no = 0
        self.cost_cum = 0.0
        self.cost_def_cum = 0.0
        self.goal_round: int | None = None

    # ------------------------------------------------------------------ run
    def run(self, resume: bool = False, should_stop=None, on_round=None) -> list[dict[str, Any]]:
        """Run to cfg.rounds. `should_stop()` is polled between rounds (cooperative stop → checkpoint);
        `on_round(rec)` is called after every round (progress hook for the web UI)."""
        if resume and self.load_checkpoint():
            self.log(f"[sim] resumed at round {self.round_no}")
        else:
            self.logger.write_manifest(self.manifest())
        records = self.logger.read_rounds()
        self.stopped = False
        self.paused = False
        while self.round_no < self.cfg.rounds:
            if should_stop is not None and should_stop():
                self.stopped = True
                self.save_checkpoint()
                self.logger.flush_segments()
                self._record_memory()
                self.logger.write_manifest(dict(self.manifest(), stopped=True))
                self.log(f"[sim] stopped after round {self.round_no} (checkpoint saved; resume to continue)")
                return records
            try:
                rec = self.step()
            except UsageLimitPaused as e:
                self.paused = True
                self.logger.write_manifest(dict(self.manifest(), paused=True, paused_reason="usage_limit",
                                                paused_at_round=self.round_no + 1))
                self.log(f"[sim] usage limit reached during round {self.round_no + 1}: round discarded, "
                         f"state = checkpoint of round {self.round_no}; resume when the window reopens ({e})")
                return records
            records.append(rec)
            if on_round is not None:
                on_round(rec)
            self.log(f"[sim] R{rec['round_no']:>3} A={rec['support']['A']:.4f} B={rec['support']['B']:.4f} "
                     f"exp={rec['exposure']['n_exposed']:>7,} L1↑{rec['l1']['n_up']:>6,} L2={rec['l2']['n_promoted']:>5} "
                     f"L3↑{rec['l3']['n_moved_up']:>5,} def={'Y' if rec['defense']['detected'] else '-'} "
                     f"rev={rec['defense']['n_reverted']:>6,} {rec['elapsed_s']:.2f}s")
            if self.cfg.checkpoint_every and self.round_no % self.cfg.checkpoint_every == 0:
                self.save_checkpoint()
        self.logger.flush_segments()
        self._record_memory()
        self.logger.write_manifest(self.manifest(final=True))
        return records

    def reset_to_start(self) -> None:
        """Back to round 0 (used when a pause happens before any checkpoint exists)."""
        self.pop.reset_dynamic(self.pop.initial_state)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.round_no = 0
        self.cost_cum = self.cost_def_cum = 0.0
        self.goal_round = None
        self.manip.yield_ema[:] = 1.0
        if hasattr(self.manip, "history"):
            self.manip.history = []
        if hasattr(self.defender, "history"):
            self.defender.history = []
        self.logger.truncate_after(0)

    def rewind_to_checkpoint(self) -> None:
        """After a pause: restore the last checkpoint (or the start) so the discarded round re-runs identically."""
        if not self.load_checkpoint():
            self.reset_to_start()

    def _record_memory(self) -> None:
        """Write this run's group-level strategist history into the cross-run memory (aggregates only)."""
        if self.memory is None:
            return
        self.memory.record_run(self.run_name, list(getattr(self.manip, "history", [])), list(getattr(self.defender, "history", [])),
                               {"ethics_level": self.cfg.ethics_level, "defender_strength": self.cfg.defender.strength,
                                "rounds": self.round_no, "seed": self.cfg.seed, "n_voters": self.pop.n,
                                "manip_mode": self.cfg.manipulator.mode, "defender_mode": self.cfg.defender.mode})

    # ----------------------------------------------------------------- step
    def step(self) -> dict[str, Any]:
        t0 = time.time()
        cfg, pop, chan, rng = self.cfg, self.pop, self.chan, self.rng
        rno = self.round_no + 1
        n = pop.n
        pop.inoculation *= np.float32(cfg.l1.inoculation_decay)
        state_new = pop.state.copy()
        neighbor_lean = pop.neighbor_lean()

        # 1. manipulator targets & buys impressions ------------------------------
        t_s = time.time()
        plan = self.manip.plan(pop, rno)
        strategy_elapsed = time.time() - t_s
        # 2. exposure (L1) --------------------------------------------------------
        ex = expose(pop, plan.cell_impressions, rng)
        # 3. L1 transition probabilities + provisional sampling -------------------
        frame_fit = 0.5 + 0.5 * chan.frame_affinity[FRAMES.index(plan.frame)][pop.age_group[ex.idx]]
        p_up, p_down = transition_probs(pop, cfg.l1, ex.idx, ex.dose, ex.trust_eff, frame_fit, plan.falsehood, neighbor_lean)
        up, down = sample_moves(p_up, p_down, rng)
        # 4. promotion L1→L2 -------------------------------------------------------
        pos = select_promoted(pop, cfg.promotion, ex.idx, p_up, cfg.promotion.k, rng)
        promoted = ex.idx[pos]
        ch_idx = np.argmax(pop.reach[promoted] * plan.channel_mix[None, :], axis=1) if len(promoted) else np.empty(0, int)
        reqs = make_requests(pop, rno, promoted, p_up[pos], plan.message, [CHANNELS[c] for c in ch_idx], kind="l1",
                             neighbor_lean=neighbor_lean, exposure_mem=pop.exposure_mem)
        # 5. L2 agent reactions (backend call, batched) ----------------------------
        t_l2 = time.time()
        resps = self.backend.react(reqs) if reqs else []
        l2_elapsed = time.time() - t_l2
        if getattr(self.backend, "exhausted", False):
            raise UsageLimitPaused("during L2 wave 1")
        up[pos] = False
        down[pos] = False
        apply_moves(state_new, ex.idx, up, down)
        deltas = np.array([r.support_delta for r in resps], dtype=np.int8) if resps else np.empty(0, np.int8)
        shares = np.array([r.share for r in resps], dtype=bool) if resps else np.empty(0, bool)
        l2_up_idx = promoted[deltas > 0]
        l2_down_idx = promoted[deltas < 0]
        state_new[l2_up_idx] = STEP_UP[state_new[l2_up_idx]]
        state_new[l2_down_idx] = STEP_DOWN[state_new[l2_down_idx]]
        # 6. L3 propagation ---------------------------------------------------------
        meme_eff = float((plan.channel_mix * chan.meme).sum())
        speed_eff = float((plan.channel_mix * chan.speed).sum())
        l1_movers = ex.idx[up]
        extra_seeds = l1_movers[rng.random(len(l1_movers)) < cfg.promotion.share_default * (0.5 + meme_eff)]
        seeds = np.unique(np.concatenate([promoted[shares], extra_seeds])).astype(np.int64)
        l3 = propagate(pop, cfg.l3, cfg.l1, seeds, state_new, speed_eff, neighbor_lean, rng)
        resps3: list[L2Response] = []
        promoted3 = np.empty(0, dtype=np.int64)
        if cfg.promotion.k_l3 > 0 and len(l3.idx):
            pos3 = select_promoted(pop, cfg.promotion, l3.idx, l3.p, cfg.promotion.k_l3, rng)
            promoted3 = l3.idx[pos3]
            social_ch = ["kakao" if pop.reach[v, CHANNELS.index("kakao")] >= pop.reach[v, CHANNELS.index("wom")] else "wom"
                         for v in promoted3]
            shared_texts = [r.reaction for r in resps if r.share and r.reaction]
            reqs3 = make_requests(pop, rno, promoted3, l3.p[pos3], plan.message + " (지인이 공유한 내용)", social_ch, kind="l3",
                                  neighbor_lean=neighbor_lean, exposure_mem=pop.exposure_mem, shared_texts=shared_texts, rng=rng)
            t_l2 = time.time()
            resps3 = self.backend.react(reqs3)
            l2_elapsed += time.time() - t_l2
            l3.up[pos3] = False
            l3.down[pos3] = False
            apply_l3(state_new, l3)
            d3 = np.array([r.support_delta for r in resps3], dtype=np.int8)
            # the neighbour's own answer decides direction; keep sign consistent with pressure
            j_up = promoted3[d3 > 0]
            j_dn = promoted3[d3 < 0]
            state_new[j_up] = STEP_UP[state_new[j_up]]
            state_new[j_dn] = STEP_DOWN[state_new[j_dn]]
        else:
            apply_l3(state_new, l3)
        if getattr(self.backend, "exhausted", False):
            raise UsageLimitPaused("during L2 wave 2")
        # 7. defender -----------------------------------------------------------------
        moved_up_mask = state_new > pop.state
        t_s = time.time()
        dres = self.defender.act(pop, plan, ex.idx, state_new, moved_up_mask, rno, rng)
        strategy_elapsed += time.time() - t_s
        if getattr(self.backend, "exhausted", False):
            raise UsageLimitPaused("during defender strategy")
        # 8. commit ---------------------------------------------------------------------
        changed = state_new != pop.state
        pop.move_dir[changed] = np.sign(state_new[changed].astype(np.int16) - pop.state[changed].astype(np.int16)).astype(np.int8)
        pop.last_move[changed] = rno
        all_up = np.flatnonzero(state_new > pop.state)
        all_down = np.flatnonzero(state_new < pop.state)
        pop.state = state_new
        pop.exposure_mem *= np.float32(cfg.l1.exposure_decay)
        pop.exposure_mem[ex.idx] += ex.dose
        self.manip.learn(pop.cell[ex.idx], pop.cell[all_up])
        self.cost_cum += plan.cost
        self.cost_def_cum += dres.cost
        self.round_no = rno

        # 9. log ----------------------------------------------------------------------------
        shares_d = pop.support_shares()
        goal = (shares_d["A"] - shares_d["B"]) >= cfg.goal_margin
        if goal and self.goal_round is None:
            self.goal_round = rno
        n_fallback = sum(1 for r in list(resps) + list(resps3) if r.source == "fallback")
        rec = {
            "round_no": rno, "elapsed_s": round(time.time() - t0, 3), "l2_elapsed_s": round(l2_elapsed, 3),
            "strategy_elapsed_s": round(strategy_elapsed, 3),
            "support": shares_d, "state_counts": dict(zip(STATE_NAMES, pop.state_counts().tolist())),
            "goal_reached": bool(goal), "goal_round": self.goal_round,
            "manipulator": plan.summary(),
            "exposure": {"n_candidates": ex.n_candidates, "n_exposed": int(len(ex.idx)),
                         "channel_share": {c: round(float(s), 3) for c, s in zip(CHANNELS, ex.channel_share)},
                         "mean_dose": round(float(ex.dose.mean()), 4) if len(ex.dose) else 0.0},
            "l1": {"n_up": int(up.sum()), "n_down": int(down.sum()),
                   "mean_p_up": round(float(p_up.mean()), 4) if len(p_up) else 0.0,
                   "borderline": int(((p_up > 0.5 - cfg.promotion.band) & (p_up < 0.5 + cfg.promotion.band)).sum())},
            "l2": {"n_promoted": int(len(promoted)), "n_delta_up": int((deltas > 0).sum()), "n_delta_down": int((deltas < 0).sum()),
                   "n_share": int(shares.sum()), "n_l3_promoted": int(len(promoted3)),
                   "n_l3_delta_up": int(sum(1 for r in resps3 if r.support_delta > 0)), "n_fallback": n_fallback},
            "l3": l3.summary(),
            "defense": dres.summary(),
            "cost": {"round": round(plan.cost, 1), "cumulative": round(self.cost_cum, 1),
                     "defender_round": round(dres.cost, 1), "defender_cumulative": round(self.cost_def_cum, 1)},
            "n_moved_up_total": int(len(all_up)), "n_moved_down_total": int(len(all_down)),
            "backend_usage": self.backend.usage(),
        }
        self.logger.log_round(rec)
        self.manip.observe(rec)
        self.defender.observe(rec)
        if resps or resps3:
            allr = list(resps) + list(resps3)
            allv = np.concatenate([promoted, promoted3]).astype(np.int64)
            self.logger.log_l2(rno, [r.to_dict() for r in allr], pop.cell[allv])
        rev_idx = dres.reverted_idx if dres.reverted_idx is not None else np.empty(0, np.int64)
        self.logger.log_segments(rno, pop.cell_state_counts(), pop.cell[ex.idx], pop.cell[all_up], pop.cell[all_down],
                                 pop.cell[np.concatenate([promoted, promoted3]).astype(np.int64)], pop.cell[rev_idx],
                                 self.manip.cell_lit, self.manip.cell_pers)
        return rec

    # ---------------------------------------------------------- checkpoint
    def manifest(self, final: bool = False) -> dict[str, Any]:
        return {"config": self.cfg.to_dict(), "data": self.data_manifest,
                "backend": {"name": self.backend.name, "spec": self.cfg.backend, "usage": self.backend.usage()},
                "strategy_memory": ({"path": str(self.memory.path), "runs_seen": self.memory.n_runs, "rounds_seen": self.memory.n_rounds}
                                    if self.memory is not None else None),
                "started_at": getattr(self, "_started", time.strftime("%Y-%m-%dT%H:%M:%S")),
                "finished": final, "rounds_done": self.round_no, "goal_round": self.goal_round,
                "note": "engine/log outputs use anonymous slot labels only (§4.4/§10)"}

    def save_checkpoint(self):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.ckpt_dir / f"round_{self.round_no:04d}.npz", **self.pop.dynamic_snapshot())
        meta = {"round_no": self.round_no, "cost_cum": self.cost_cum, "cost_def_cum": self.cost_def_cum,
                "goal_round": self.goal_round, "rng_state": self.rng.bit_generator.state,
                "budget_per_round": self.manip.budget_per_round,
                "manipulator": self.manip.state_dict(), "defender": self.defender.state_dict(),
                "backend_usage": self.backend.usage(),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.ckpt_dir / f"round_{self.round_no:04d}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        with open(self.ckpt_dir / "latest.json", "w", encoding="utf-8") as f:
            json.dump({"round_no": self.round_no}, f)
        self.logger.flush_segments()

    def load_checkpoint(self, round_no: int | None = None) -> bool:
        latest = self.ckpt_dir / "latest.json"
        if round_no is None:
            if not latest.exists():
                return False
            round_no = json.loads(latest.read_text())["round_no"]
        npz = self.ckpt_dir / f"round_{round_no:04d}.npz"
        meta_path = self.ckpt_dir / f"round_{round_no:04d}.json"
        if not npz.exists() or not meta_path.exists():
            return False
        snap = dict(np.load(npz))
        self.pop.restore_dynamic(snap)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.round_no = meta["round_no"]
        self.cost_cum = meta["cost_cum"]
        self.cost_def_cum = meta.get("cost_def_cum", 0.0)
        self.goal_round = meta.get("goal_round")
        self.rng.bit_generator.state = meta["rng_state"]
        self.manip.load_state_dict(meta["manipulator"])
        self.defender.load_state_dict(meta.get("defender", {}))
        if "budget_per_round" in meta:            # keep the per-round budget stable even if `rounds` was extended
            self.manip.budget_per_round = float(meta["budget_per_round"])
        self.logger.truncate_after(self.round_no)
        return True


def run_until_done(sim: "Simulation", resume: bool = False, should_stop=None, on_round=None,
                   retry_s: int = 600, on_wait=None) -> list[dict[str, Any]]:
    """Run a simulation to completion, waiting out backend usage-limit windows.

    On a pause: the discarded round is re-run from the last checkpoint once `backend.probe()`
    succeeds. Between probes we sleep `retry_s` (polling `should_stop` every 5 s)."""
    records = sim.run(resume=resume, should_stop=should_stop, on_round=on_round)
    while getattr(sim, "paused", False):
        if should_stop is not None and should_stop():
            return records
        waited = 0
        while waited < retry_s:
            if should_stop is not None and should_stop():
                return records
            time.sleep(5)
            waited += 5
            if on_wait is not None:
                on_wait(retry_s - waited)
        if not sim.backend.probe():
            sim.log("[sim] usage window still closed — will retry")
            continue
        sim.log("[sim] usage window open — resuming from checkpoint")
        sim.rewind_to_checkpoint()
        records = sim.run(resume=True, should_stop=should_stop, on_round=on_round)
    return records
