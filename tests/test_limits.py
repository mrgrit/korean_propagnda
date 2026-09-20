import json

from conftest import make_cfg
from opinion_field.agents.mock import MockBackend
from opinion_field.engine.loop import Simulation, run_until_done
from opinion_field.engine.population import Population


def test_usage_limit_pause_and_auto_resume(paths, chan, tmp_path):
    # reference: uninterrupted 3 rounds
    cfg = make_cfg(paths, rounds=3)
    ref = Simulation(cfg, Population.load(paths), chan, MockBackend(seed=1), tmp_path / "ref", {}, log=lambda *a: None).run()
    # limit hits during round 2's L2 calls; probe recovers on the 2nd attempt
    b = MockBackend(seed=1, exhaust_after=1, probes_to_recover=2)   # 1 call per round (batch=1 → many calls)… use batch so round 1 = 1 call
    b.batch_size = 10_000
    sim = Simulation(cfg, Population.load(paths), chan, b, tmp_path / "lim", {}, log=lambda *a: None)
    recs = run_until_done(sim, retry_s=5)
    assert b.probes == 2 and not sim.paused
    assert [r["round_no"] for r in recs] == [1, 2, 3]
    # the discarded round re-ran identically → trajectories match the reference exactly
    assert [r["support"] for r in recs] == [r["support"] for r in ref]
    lines = [json.loads(l) for l in (tmp_path / "lim" / "rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [l["round_no"] for l in lines] == [1, 2, 3]                 # no duplicate/discarded round in the log
    assert all(l["l2"]["n_fallback"] == 0 for l in lines)


def test_pause_before_first_checkpoint_resets_to_start(paths, chan, tmp_path):
    cfg = make_cfg(paths, rounds=2)
    ref = Simulation(cfg, Population.load(paths), chan, MockBackend(seed=3), tmp_path / "ref2", {}, log=lambda *a: None).run()
    b = MockBackend(seed=3, exhaust_after=0, probes_to_recover=1)
    b.batch_size = 10_000
    sim = Simulation(cfg, Population.load(paths), chan, b, tmp_path / "lim2", {}, log=lambda *a: None)
    recs = run_until_done(sim, retry_s=5)
    assert [r["support"] for r in recs] == [r["support"] for r in ref]


def test_stop_while_paused_returns(paths, chan, tmp_path):
    cfg = make_cfg(paths, rounds=2)
    b = MockBackend(seed=3, exhaust_after=0, probes_to_recover=99)
    b.batch_size = 10_000
    sim = Simulation(cfg, Population.load(paths), chan, b, tmp_path / "lim3", {}, log=lambda *a: None)
    recs = run_until_done(sim, retry_s=5, should_stop=lambda: sim.paused)   # operator stops once we are paused
    assert recs == [] and sim.paused and b.probes == 0
    manifest = json.loads((tmp_path / "lim3" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["paused"] and manifest["paused_reason"] == "usage_limit"
