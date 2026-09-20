import json

import numpy as np
import yaml

from conftest import make_cfg
from opinion_field.agents.mock import MockBackend
from opinion_field.agents.prompts import load_tone_bank, make_requests, social_context
from opinion_field.data.tone_bank import build_tone_bank
from opinion_field.engine.loop import Simulation
from opinion_field.engine.memory import StrategyMemory
from opinion_field.engine.population import Population


def test_social_context_and_shared_text(pop):
    assert social_context(0.5, 0.0)["주변 지인 성향"] == "후보 A 쪽이 많음"
    assert social_context(-0.5, 5.0)["이 메시지 접촉"].startswith("최근")
    ids = np.arange(5)
    reqs = make_requests(pop, 1, ids, np.full(5, 0.5), "메시지", ["kakao"] * 5, kind="l3",
                         neighbor_lean=np.full(pop.n, -0.3, np.float32), exposure_mem=np.ones(pop.n, np.float32),
                         shared_texts=["경제가 문제네요"], rng=np.random.default_rng(0))
    assert all("지인이 공유하며 한 말" in r.message and r.persona["주변 지인 성향"] == "후보 B 쪽이 많음" for r in reqs)


def test_memory_accumulates_and_feeds_brief(paths, chan, tmp_path):
    mem_path = tmp_path / "mem.json"
    kw = {"manipulator.mode": "llm", "defender.mode": "llm", "defender.strength": 0.9, "ethics_level": 0.9, "rounds": 2,
          "manipulator.memory_path": str(mem_path)}
    for name in ("m1", "m2"):
        cfg = make_cfg(paths, **kw)
        sim = Simulation(cfg, Population.load(paths), chan, MockBackend(seed=5, batch_size=8), tmp_path / "runs" / name, {}, log=lambda *a: None)
        sim.run()
    m = StrategyMemory(mem_path)
    assert m.n_runs == 2 and m.n_rounds == 4
    assert "누적 경험" in m.manip_summary() and "주장유형" in m.manip_summary()
    d = json.loads(mem_path.read_text(encoding="utf-8"))
    assert all("group_yield" in h for r in d["runs"].values() for h in r["manipulator"])
    # a third run sees the first two in its brief (excluding itself)
    cfg = make_cfg(paths, **kw)
    sim = Simulation(cfg, Population.load(paths), chan, MockBackend(seed=5, batch_size=8), tmp_path / "runs" / "m3", {}, log=lambda *a: None)
    rule_plan = sim.manip.__class__.__mro__[1].plan(sim.manip, sim.pop, 1)
    brief, _ = sim.manip._brief(sim.pop, 1, rule_plan)
    assert "누적 경험 (지난 실행 2회" in brief
    manifest = json.loads((tmp_path / "runs" / "m1" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["strategy_memory"]["path"].endswith("mem.json")


def test_tone_bank_builder_with_mock(paths, tmp_path):
    out = tmp_path / "tone.yaml"
    res = build_tone_bank(paths, MockBackend(seed=1), out, samples=5, log=lambda *a: None)
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert res["groups_ok"] == len(doc["by_group"]) >= 25 and doc["source"].startswith("NEMOTRON")
    load_tone_bank.cache_clear()
    tone = load_tone_bank(str(out))
    from opinion_field.agents.prompts import persona_card
    card = persona_card(__import__("opinion_field.engine.population", fromlist=["Population"]).Population.load(paths), 0, None, tone)
    assert "말투·관심 힌트" in card and "mock 톤" in card["말투·관심 힌트"]
