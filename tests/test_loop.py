import json
from pathlib import Path

import numpy as np

from opinion_field.agents.mock import MockBackend
from opinion_field.engine.loop import Simulation
from opinion_field.engine.population import Population
from opinion_field.observe.report import render_markdown, compare_runs
from opinion_field.observe.display import DisplayMap
from conftest import make_cfg


def _run(paths, chan, out, **kw):
    cfg = make_cfg(paths, **kw)
    pop = Population.load(paths)
    sim = Simulation(cfg, pop, chan, MockBackend(seed=cfg.seed), out, {"n_voters": pop.n, "population_source": "synthetic", "priors_provenance": {}}, log=lambda *a: None)
    return sim.run(), sim


def test_loop_runs_and_logs(paths, chan, tmp_path):
    recs, sim = _run(paths, chan, tmp_path / "r1")
    assert len(recs) == 3
    assert (tmp_path / "r1" / "rounds.jsonl").exists()
    assert (tmp_path / "r1" / "segments.parquet").exists()
    assert (tmp_path / "r1" / "l2_responses.jsonl").exists()
    for r in recs:
        assert r["l2"]["n_promoted"] <= 40
        assert r["exposure"]["n_exposed"] > 0
        assert abs(sum(r["state_counts"].values()) - sim.pop.n) == 0
    # movement happened
    assert sum(r["n_moved_up_total"] for r in recs) > 0
    md = render_markdown(tmp_path / "r1")
    assert "지지율 궤적" in md and "후보 A" in md


def test_determinism(paths, chan, tmp_path):
    a, _ = _run(paths, chan, tmp_path / "a")
    b, _ = _run(paths, chan, tmp_path / "b")
    for x, y in zip(a, b):
        assert x["support"] == y["support"]
        assert x["state_counts"] == y["state_counts"]


def test_checkpoint_resume(paths, chan, tmp_path):
    full, _ = _run(paths, chan, tmp_path / "full")
    # simulate an interruption: 2 rounds with checkpoints, then a fresh process resumes the third
    cfg = make_cfg(paths, rounds=3)
    pop = Population.load(paths)
    sim = Simulation(cfg, pop, chan, MockBackend(seed=cfg.seed), tmp_path / "part", {}, log=lambda *a: None)
    sim.logger.write_manifest(sim.manifest())
    sim.step(); sim.save_checkpoint(); sim.step(); sim.save_checkpoint()
    cfg2 = make_cfg(paths, rounds=3)
    pop2 = Population.load(paths)
    sim2 = Simulation(cfg2, pop2, chan, MockBackend(seed=cfg2.seed), tmp_path / "part", {}, log=lambda *a: None)
    recs = sim2.run(resume=True)
    assert len(recs) == 3
    assert recs[2]["support"] == full[2]["support"]
    assert recs[2]["state_counts"] == full[2]["state_counts"]
    # extending `rounds` on resume must not change the per-round budget
    cfg3 = make_cfg(paths, rounds=5)
    sim3 = Simulation(cfg3, Population.load(paths), chan, MockBackend(seed=cfg3.seed), tmp_path / "part", {}, log=lambda *a: None)
    assert sim3.load_checkpoint() and sim3.manip.budget_per_round == sim2.manip.budget_per_round


def test_defender_reduces_manipulation(paths, chan, tmp_path):
    on, _ = _run(paths, chan, tmp_path / "on", **{"defender.enabled": True, "defender.strength": 0.9, "ethics_level": 0.9, "rounds": 4})
    off, _ = _run(paths, chan, tmp_path / "off", **{"defender.enabled": False, "ethics_level": 0.9, "rounds": 4})
    assert any(r["defense"]["detected"] for r in on)
    assert sum(r["defense"]["n_reverted"] for r in on) > 0
    assert off[-1]["support"]["A"] >= on[-1]["support"]["A"] - 0.002
    cmp_df = compare_runs(tmp_path / "on", tmp_path / "off")
    assert cmp_df.height == 4


def test_no_display_names_in_engine_outputs(paths, chan, tmp_path):
    recs, _ = _run(paths, chan, tmp_path / "dn")
    text = (tmp_path / "dn" / "rounds.jsonl").read_text(encoding="utf-8")
    assert ("후보 A" in text) or ("후보 B" in text)  # slot labels present in engine logs
    dm = DisplayMap({"A": "표시명甲", "B": "표시명乙"})
    md = render_markdown(tmp_path / "dn", dm)
    assert ("표시명甲" in md or "표시명乙" in md) and "후보 A" not in md and "후보 B" not in md
    assert "표시명甲" not in text
