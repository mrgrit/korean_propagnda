"""Structured run logging (§6): rounds.jsonl, segments.parquet, l2_responses.jsonl, manifest.json."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..schema import N_CELLS, N_STATES, STATE_NAMES, cell_label


class RunLogger:
    def __init__(self, out_dir: str | Path):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rounds_path = self.out / "rounds.jsonl"
        self.l2_path = self.out / "l2_responses.jsonl"
        self.segments_path = self.out / "segments.parquet"
        self._segments: list[pl.DataFrame] = []
        if self.segments_path.exists():
            self._segments.append(pl.read_parquet(self.segments_path))

    def write_manifest(self, data: dict[str, Any]):
        with open(self.out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)

    def log_round(self, record: dict[str, Any]):
        with open(self.rounds_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def log_l2(self, round_no: int, responses: list[dict[str, Any]], cells: np.ndarray):
        with open(self.l2_path, "a", encoding="utf-8") as f:
            for r, c in zip(responses, cells.tolist()):
                r = dict(r, round_no=round_no, cell=int(c))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def log_segments(self, round_no: int, cell_counts: np.ndarray, exposed_cells: np.ndarray,
                     up_cells: np.ndarray, down_cells: np.ndarray, promoted_cells: np.ndarray,
                     reverted_cells: np.ndarray, cell_lit: np.ndarray, cell_pers: np.ndarray):
        df = pl.DataFrame({
            "round_no": np.full(N_CELLS, round_no, dtype=np.int32),
            "cell": np.arange(N_CELLS, dtype=np.int32),
            "n": cell_counts.sum(axis=1).astype(np.int64),
            "exposed": np.bincount(exposed_cells, minlength=N_CELLS).astype(np.int64),
            "moved_up": np.bincount(up_cells, minlength=N_CELLS).astype(np.int64),
            "moved_down": np.bincount(down_cells, minlength=N_CELLS).astype(np.int64),
            "l2_promoted": np.bincount(promoted_cells, minlength=N_CELLS).astype(np.int64),
            "reverted": np.bincount(reverted_cells, minlength=N_CELLS).astype(np.int64),
            "mean_literacy": cell_lit.astype(np.float32),
            "mean_persuadability": cell_pers.astype(np.float32),
        })
        for s, name in enumerate(STATE_NAMES):
            df = df.with_columns(pl.Series(f"n_{name}", cell_counts[:, s].astype(np.int64)))
        df = df.filter(pl.col("n") > 0)
        self._segments.append(df)

    def flush_segments(self):
        if self._segments:
            pl.concat(self._segments).unique(subset=["round_no", "cell"], keep="last").sort(
                ["round_no", "cell"]).write_parquet(self.segments_path)

    def read_rounds(self) -> list[dict[str, Any]]:
        if not self.rounds_path.exists():
            return []
        with open(self.rounds_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def truncate_after(self, round_no: int):
        """Drop log lines for rounds > round_no (used on resume from an older checkpoint)."""
        for path in (self.rounds_path, self.l2_path):
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    keep = [l for l in f if l.strip() and json.loads(l).get("round_no", 0) <= round_no]
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(keep)
        self._segments = [d.filter(pl.col("round_no") <= round_no) for d in self._segments]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
