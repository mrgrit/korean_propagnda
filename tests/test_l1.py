import numpy as np

from opinion_field.config import L1Params
from opinion_field.engine.channels import expose
from opinion_field.engine.l1 import apply_moves, sample_moves, transition_probs
from opinion_field.engine.population import STEP_DOWN, STEP_UP
from opinion_field.schema import N_CELLS, N_CHANNELS, SupportState


def test_ladder_steps():
    assert STEP_UP[SupportState.ABSTAIN] == SupportState.UNDECIDED
    assert STEP_UP[SupportState.A_STRONG] == SupportState.A_STRONG
    assert STEP_DOWN[SupportState.B_STRONG] == SupportState.B_STRONG
    assert STEP_DOWN[SupportState.ABSTAIN] == SupportState.ABSTAIN
    assert STEP_UP[SupportState.B_WEAK] == SupportState.UNDECIDED


def test_exposure_and_probs(pop):
    rng = np.random.default_rng(0)
    cell_impr = np.zeros((N_CELLS, N_CHANNELS))
    cells = np.unique(pop.cell)[:30]
    cell_impr[cells] = pop.cell_size[cells][:, None] * 0.5
    ex = expose(pop, cell_impr, rng)
    assert ex.n_candidates == int(np.isin(pop.cell, cells).sum())
    assert 0 < len(ex.idx) < ex.n_candidates
    assert np.all(np.isin(pop.cell[ex.idx], cells))
    assert np.all((ex.trust_eff > 0) & (ex.trust_eff < 1))
    prm = L1Params()
    fit = np.full(len(ex.idx), 0.8, dtype=np.float32)
    p_up, p_down = transition_probs(pop, prm, ex.idx, ex.dose, ex.trust_eff, fit, 0.5, pop.neighbor_lean())
    assert p_up.min() >= 0 and p_up.max() <= prm.p_cap
    assert np.all(p_up[pop.state[ex.idx] == SupportState.A_STRONG] == 0)
    strong_b = pop.state[ex.idx] == SupportState.B_STRONG
    und = pop.state[ex.idx] == SupportState.UNDECIDED
    if strong_b.any() and und.any():
        assert p_up[strong_b].mean() < p_up[und].mean()
    up, down = sample_moves(p_up, p_down, rng)
    assert not np.any(up & down)
    state_new = pop.state.copy()
    apply_moves(state_new, ex.idx, up, down)
    assert np.all(state_new[ex.idx[up]] == STEP_UP[pop.state[ex.idx[up]]])
    assert np.all(state_new[~np.isin(np.arange(pop.n), ex.idx)] == pop.state[~np.isin(np.arange(pop.n), ex.idx)])
