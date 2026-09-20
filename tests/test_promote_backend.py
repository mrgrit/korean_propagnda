import numpy as np

from opinion_field.agents.backend import L2Request, coerce_response, RESPONSE_SCHEMA
from opinion_field.agents.mock import MockBackend
from opinion_field.agents.prompts import make_requests, user_prompt, SYSTEM_PROMPT
from opinion_field.config import PromotionParams
from opinion_field.engine.promote import select_promoted


def test_promotion_topk_unique_and_representative(pop):
    rng = np.random.default_rng(1)
    idx = np.arange(0, pop.n, 3)
    p = rng.random(len(idx)).astype(np.float32)
    prm = PromotionParams(k=100)
    pos = select_promoted(pop, prm, idx, p, 100, rng)
    assert len(pos) == 100 and len(np.unique(pos)) == 100
    # representation: with k large, every exposed cell has at least one promoted voter
    pos_all = select_promoted(pop, prm, idx, p, len(np.unique(pop.cell[idx])), rng)
    assert len(np.unique(pop.cell[idx[pos_all]])) == len(np.unique(pop.cell[idx]))
    # fewer slots than cells → still unique, k of them
    pos_small = select_promoted(pop, prm, idx, p, 5, rng)
    assert len(pos_small) == 5 and len(np.unique(pop.cell[idx[pos_small]])) == 5
    assert len(select_promoted(pop, prm, idx[:0], p[:0], 10, rng)) == 0


def test_mock_backend_deterministic_and_requests(pop):
    ids = np.arange(10)
    reqs = make_requests(pop, 1, ids, np.linspace(0, 1, 10), "후보 A 메시지 (허위)", ["kakao"] * 10)
    assert all("이름" not in r.persona for r in reqs)
    txt = user_prompt(reqs[0])
    assert "페르소나 카드" in txt and "후보" in reqs[0].state_label or "부동" in reqs[0].state_label or "기권" in reqs[0].state_label
    assert "합성" in SYSTEM_PROMPT and "집계" in SYSTEM_PROMPT
    b = MockBackend(seed=5)
    r1 = b.react(reqs)
    r2 = MockBackend(seed=5).react(reqs)
    assert [x.support_delta for x in r1] == [x.support_delta for x in r2]
    assert all(x.support_delta in (-1, 0, 1) for x in r1)


def test_coerce_response():
    r = coerce_response(3, {"support_delta": 5, "share": "yes", "reaction": "x" * 500}, "api")
    assert r.support_delta == 1 and r.share is True and len(r.reaction) == 200
    assert coerce_response(3, None, "api").source == "fallback"
    assert set(RESPONSE_SCHEMA["required"]) == {"support_delta", "share", "reaction"}
