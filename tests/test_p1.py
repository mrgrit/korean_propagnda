import numpy as np

from conftest import make_cfg
from opinion_field.agents.backend import L2Request
from opinion_field.agents.mock import MockBackend
from opinion_field.agents.prompts import batch_prompt, make_requests
from opinion_field.engine.loop import Simulation
from opinion_field.engine.population import Population
from opinion_field.engine.segments import CELL_GROUP, N_GROUPS, group_label, group_weights_to_cells
from opinion_field.schema import N_CELLS


def test_groups_cover_cells():
    assert CELL_GROUP.shape == (N_CELLS,) and CELL_GROUP.min() >= 0 and CELL_GROUP.max() < N_GROUPS
    assert N_GROUPS == 90 and "/" in group_label(0)
    w = group_weights_to_cells([(0, 1.0), (5, 3.0)], np.ones(N_CELLS), None)
    assert abs(w.sum() - 1) < 1e-9 and np.isclose(w[CELL_GROUP == 5].sum(), 0.75)


def test_batch_react_maps_by_idx(pop):
    ids = np.arange(0, 60, 3)
    reqs = make_requests(pop, 2, ids, np.linspace(0.1, 0.9, len(ids)), "후보 A 메시지", ["kakao"] * len(ids))
    assert "idx 3" in batch_prompt(reqs[:5]) and all("말투·관심 힌트" in r.persona for r in reqs)
    one = MockBackend(seed=9, batch_size=1).react(reqs)
    eight = MockBackend(seed=9, batch_size=8).react(reqs)
    assert [r.voter_id for r in eight] == ids.tolist()
    assert [r.support_delta for r in one] == [r.support_delta for r in eight]   # batching must not change outcomes
    assert all(r.source == "mock" and r.error is None for r in eight)


def _run(paths, chan, out, backend, **kw):
    cfg = make_cfg(paths, **kw)
    pop = Population.load(paths)
    sim = Simulation(cfg, pop, chan, backend, out, {}, log=lambda *a: None)
    return sim.run(), sim


def test_llm_modes_with_mock_strategist(paths, chan, tmp_path):
    kw = {"manipulator.mode": "llm", "defender.mode": "llm", "defender.strength": 0.9, "ethics_level": 0.9, "rounds": 3}
    recs, sim = _run(paths, chan, tmp_path / "llm", MockBackend(seed=123, batch_size=8), **kw)
    assert all(r["manipulator"]["llm"]["used"] for r in recs)
    assert all(g["id"] < N_GROUPS for r in recs for g in r["manipulator"]["llm"]["groups"])
    assert all(r["manipulator"]["claim_tier"] == "fabricated" for r in recs)       # ethics ceiling 0.9 → tier 2 allowed
    detected = [r for r in recs if r["defense"]["detected"]]
    assert detected and all(r["defense"]["llm"]["used"] and r["defense"]["llm"]["prebunk_share"] == 0.3 for r in detected)
    assert len(sim.manip.history) == 3 and sim.manip.history[-1]["llm_used"]
    # determinism + checkpoint carries strategist history
    recs2, sim2 = _run(paths, chan, tmp_path / "llm2", MockBackend(seed=123, batch_size=8), **kw)
    assert [r["support"] for r in recs] == [r["support"] for r in recs2]
    assert sim2.load_checkpoint() and len(sim2.manip.history) == 3 and len(sim2.defender.history) == len(detected)


def test_ethics_ceiling_and_fallback(paths, chan, tmp_path):
    recs, _ = _run(paths, chan, tmp_path / "eth", MockBackend(seed=1), **{"manipulator.mode": "llm", "ethics_level": 0.2, "rounds": 2})
    assert all(r["manipulator"]["claim_tier"] == "factual" for r in recs)          # mock asks for max tier → clipped to 0
    recs, _ = _run(paths, chan, tmp_path / "fb", MockBackend(seed=1, fail=True),
                   **{"manipulator.mode": "llm", "defender.mode": "llm", "defender.strength": 0.9, "rounds": 2})
    assert all(r["manipulator"]["llm"]["used"] is False and r["manipulator"]["llm"]["fallback"] == "rule" for r in recs)
    assert all(r["exposure"]["n_exposed"] > 0 for r in recs)                       # rule fallback still runs the round
    det = [r for r in recs if r["defense"]["detected"]]
    assert all(r["defense"]["llm"]["used"] is False for r in det)
