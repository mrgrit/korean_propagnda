import numpy as np

from opinion_field.schema import N_CELLS, cell_id, decode_cell, cell_label


def test_cell_roundtrip():
    for c in range(N_CELLS):
        s, a, x, e = decode_cell(c)
        assert int(cell_id(np.array([s]), np.array([a]), np.array([x]), np.array([e]))[0]) == c
    assert "/" in cell_label(0)


def test_network_csr_valid(pop):
    assert pop.indptr[0] == 0 and pop.indptr[-1] == len(pop.indices)
    assert np.all(np.diff(pop.indptr) >= 0)
    assert pop.indices.min() >= 0 and pop.indices.max() < pop.n
    # symmetric: edge (a,b) implies (b,a)
    src = pop.edge_src
    pair = set(zip(src[:5000].tolist(), pop.indices[:5000].tolist()))
    rev_ok = 0
    for a, b in list(pair)[:500]:
        nb = pop.indices[pop.indptr[b]:pop.indptr[b + 1]]
        rev_ok += int(a in nb)
    assert rev_ok == min(500, len(pair))
    assert 0 < pop.centrality.max() <= 1.0


def test_population_vectors(pop):
    assert pop.literacy.min() > 0 and pop.literacy.max() < 1
    assert pop.persuadability.min() > 0 and pop.persuadability.max() < 1
    assert pop.reach.shape == (pop.n, 6) and pop.trust.shape == (pop.n, 6)
    counts = pop.state_counts()
    assert counts.sum() == pop.n and counts[3] > 0 and counts[0] > 0
