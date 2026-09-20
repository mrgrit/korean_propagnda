"""Population = struct-of-arrays over all voters (L1 layer), plus dynamic state.

Static arrays come from voters.parquet; dynamic arrays are what a checkpoint saves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ..config import DataPaths
from ..schema import CHANNELS, N_CELLS, N_CHANNELS, N_STATES, SupportState

# lean of each state on the B(-1)…A(+1) axis; abstain/undecided = 0
LEAN = np.array([0.0, -1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
# next state when pushed toward A / toward B (index = current state)
STEP_UP = np.array([3, 2, 3, 4, 5, 5], dtype=np.uint8)
STEP_DOWN = np.array([0, 1, 1, 2, 3, 4], dtype=np.uint8)


@dataclass
class Population:
    n: int
    sido: np.ndarray
    sigungu: np.ndarray
    age: np.ndarray
    age_group: np.ndarray
    sex: np.ndarray
    edu: np.ndarray
    edu_bucket: np.ndarray
    occ_group: np.ndarray
    cell: np.ndarray
    persuadability: np.ndarray
    literacy: np.ndarray
    centrality: np.ndarray
    reach: np.ndarray            # (n, C)
    trust: np.ndarray            # (n, C)
    indptr: np.ndarray
    indices: np.ndarray
    # dynamic
    state: np.ndarray = field(default=None)
    exposure_mem: np.ndarray = field(default=None)
    inoculation: np.ndarray = field(default=None)
    last_move: np.ndarray = field(default=None)
    move_dir: np.ndarray = field(default=None)
    initial_state: np.ndarray = field(default=None)
    # derived caches
    cell_size: np.ndarray = field(default=None)
    edge_src: np.ndarray = field(default=None)
    degree: np.ndarray = field(default=None)
    personas_path: Path | None = None
    _personas: pl.DataFrame | None = None

    DYNAMIC = ("state", "exposure_mem", "inoculation", "last_move", "move_dir")

    def __post_init__(self):
        self.cell_size = np.bincount(self.cell, minlength=N_CELLS).astype(np.int64)
        self.degree = np.diff(self.indptr).astype(np.int32)
        self.edge_src = np.repeat(np.arange(self.n, dtype=np.int32), self.degree)
        if self.state is None:
            self.state = np.full(self.n, 3, dtype=np.uint8)
        if self.exposure_mem is None:
            self.reset_dynamic()
        if self.initial_state is None:
            self.initial_state = self.state.copy()

    def reset_dynamic(self, initial_state: np.ndarray | None = None):
        if initial_state is not None:
            self.state = initial_state.astype(np.uint8).copy()
        self.exposure_mem = np.zeros(self.n, dtype=np.float32)
        self.inoculation = np.zeros(self.n, dtype=np.float32)
        self.last_move = np.full(self.n, -999, dtype=np.int16)
        self.move_dir = np.zeros(self.n, dtype=np.int8)

    # ---- stats -----------------------------------------------------------
    def state_counts(self) -> np.ndarray:
        return np.bincount(self.state, minlength=N_STATES)

    def support_shares(self) -> dict[str, float]:
        c = self.state_counts().astype(np.float64)
        decided = c[1] + c[2] + c[4] + c[5]
        n = c.sum()
        return {
            "A": float((c[4] + c[5]) / max(decided, 1)), "B": float((c[1] + c[2]) / max(decided, 1)),
            "undecided_all": float(c[3] / n), "abstain_all": float(c[0] / n),
            "A_all": float((c[4] + c[5]) / n), "B_all": float((c[1] + c[2]) / n),
        }

    def cell_state_counts(self) -> np.ndarray:
        """(N_CELLS, N_STATES) counts."""
        key = self.cell.astype(np.int64) * N_STATES + self.state
        return np.bincount(key, minlength=N_CELLS * N_STATES).reshape(N_CELLS, N_STATES)

    def neighbor_lean(self) -> np.ndarray:
        """Mean lean of neighbours (social proof); 0 for isolated nodes."""
        s = np.bincount(self.edge_src, weights=LEAN[self.state[self.indices]], minlength=self.n)
        return (s / np.maximum(self.degree, 1)).astype(np.float32)

    def neighbors_of(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (src_repeated, neighbour) for all edges out of idx."""
        deg = self.degree[idx]
        total = int(deg.sum())
        if total == 0:
            return np.empty(0, np.int32), np.empty(0, np.int32)
        starts = self.indptr[idx]
        offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(deg) - deg, deg)
        nb = self.indices[np.repeat(starts, deg) + offs]
        return np.repeat(idx.astype(np.int32), deg), nb

    # ---- personas (L2 cards) --------------------------------------------
    def personas(self, voter_ids: np.ndarray) -> pl.DataFrame:
        if self._personas is None:
            if self.personas_path is None or not Path(self.personas_path).exists():
                return pl.DataFrame({"voter_id": voter_ids.astype(np.uint32)})
            self._personas = pl.read_parquet(self.personas_path)
        return self._personas.filter(pl.col("voter_id").is_in(voter_ids.astype(np.uint32).tolist()))

    # ---- IO ----------------------------------------------------------------
    @classmethod
    def load(cls, paths: DataPaths) -> "Population":
        df = pl.read_parquet(paths.voters_parquet)
        net = np.load(paths.network_npz)
        reach = np.stack([df[f"reach_{c}"].to_numpy() for c in CHANNELS], axis=1).astype(np.float32)
        trust = np.stack([df[f"trust_{c}"].to_numpy() for c in CHANNELS], axis=1).astype(np.float32)
        pop = cls(
            n=df.height, sido=df["sido"].to_numpy(), sigungu=df["sigungu"].to_numpy(),
            age=df["age"].to_numpy(), age_group=df["age_group"].to_numpy(), sex=df["sex"].to_numpy(),
            edu=df["edu"].to_numpy(), edu_bucket=df["edu_bucket"].to_numpy(),
            occ_group=df["occ_group"].to_numpy(), cell=df["cell"].to_numpy(),
            persuadability=df["persuadability"].to_numpy().astype(np.float32),
            literacy=df["literacy"].to_numpy().astype(np.float32),
            centrality=df["centrality"].to_numpy().astype(np.float32),
            reach=reach, trust=trust, indptr=net["indptr"], indices=net["indices"],
            state=df["support_state"].to_numpy().astype(np.uint8).copy(),
            personas_path=paths.personas_parquet,
        )
        return pop

    def dynamic_snapshot(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k).copy() for k in self.DYNAMIC}

    def restore_dynamic(self, snap: dict[str, np.ndarray]):
        for k in self.DYNAMIC:
            setattr(self, k, snap[k].copy())
