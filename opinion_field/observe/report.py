"""P0 metrics (§6): trajectory, goal attainment, segment contribution, literacy correlation,
defender on/off comparison. Display names injected only here via DisplayMap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from ..schema import STATE_NAMES, cell_label
from .display import DisplayMap


def load_run(run_dir: str | Path) -> tuple[dict, list[dict], pl.DataFrame | None]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rounds = [json.loads(l) for l in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seg_path = run_dir / "segments.parquet"
    segments = pl.read_parquet(seg_path) if seg_path.exists() else None
    return manifest, rounds, segments


def trajectory_table(rounds: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "round": [r["round_no"] for r in rounds],
        "A": [r["support"]["A"] for r in rounds],
        "B": [r["support"]["B"] for r in rounds],
        "undecided": [r["support"]["undecided_all"] for r in rounds],
        "abstain": [r["support"]["abstain_all"] for r in rounds],
        "exposed": [r["exposure"]["n_exposed"] for r in rounds],
        "l2": [r["l2"]["n_promoted"] for r in rounds],
        "defended": [r["defense"]["detected"] for r in rounds],
        "reverted": [r["defense"]["n_reverted"] for r in rounds],
        "cost_cum": [r["cost"]["cumulative"] for r in rounds],
    })


def goal_summary(manifest: dict, rounds: list[dict]) -> dict:
    margin = manifest["config"]["goal_margin"]
    first = next((r for r in rounds if r["support"]["A"] - r["support"]["B"] >= margin), None)
    last = rounds[-1] if rounds else None
    return {
        "goal_margin": margin,
        "achieved": first is not None,
        "achieved_round": first["round_no"] if first else None,
        "cost_at_achievement": first["cost"]["cumulative"] if first else None,
        "final_A": last["support"]["A"] if last else None,
        "final_B": last["support"]["B"] if last else None,
        "final_margin": (last["support"]["A"] - last["support"]["B"]) if last else None,
    }


def segment_contribution(segments: pl.DataFrame, top: int = 15) -> pl.DataFrame:
    """Net A-ward movement by cell over the whole run (which cells moved)."""
    agg = (segments.group_by("cell")
           .agg(pl.col("n").max().alias("n"), pl.col("exposed").sum(), pl.col("moved_up").sum(),
                pl.col("moved_down").sum(), pl.col("reverted").sum(), pl.col("l2_promoted").sum(),
                pl.col("mean_literacy").first(), pl.col("mean_persuadability").first())
           .with_columns((pl.col("moved_up") - pl.col("moved_down") - pl.col("reverted")).alias("net_up"))
           .with_columns((pl.col("net_up") / pl.col("n")).alias("net_up_rate"))
           .sort("net_up", descending=True))
    agg = agg.with_columns(pl.col("cell").map_elements(lambda c: cell_label(int(c)), return_dtype=pl.Utf8).alias("label"))
    return agg.head(top)


def literacy_vulnerability(segments: pl.DataFrame) -> dict:
    agg = (segments.group_by("cell")
           .agg(pl.col("n").max(), pl.col("moved_up").sum(), pl.col("moved_down").sum(),
                pl.col("reverted").sum(), pl.col("mean_literacy").first())
           .filter(pl.col("n") >= 50)
           .with_columns(((pl.col("moved_up") - pl.col("moved_down") - pl.col("reverted")) / pl.col("n")).alias("net_rate")))
    if agg.height < 3:
        return {"pearson_r": None, "n_cells": agg.height}
    x = agg["mean_literacy"].to_numpy()
    y = agg["net_rate"].to_numpy()
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    return {"pearson_r": r, "n_cells": agg.height}


def compare_runs(run_on: str | Path, run_off: str | Path) -> pl.DataFrame:
    _, r_on, _ = load_run(run_on)
    _, r_off, _ = load_run(run_off)
    n = min(len(r_on), len(r_off))
    return pl.DataFrame({
        "round": [r_on[i]["round_no"] for i in range(n)],
        "A_defended": [r_on[i]["support"]["A"] for i in range(n)],
        "A_undefended": [r_off[i]["support"]["A"] for i in range(n)],
        "delta_A": [r_off[i]["support"]["A"] - r_on[i]["support"]["A"] for i in range(n)],
        "reverted": [r_on[i]["defense"]["n_reverted"] for i in range(n)],
    })


def render_markdown(run_dir: str | Path, display_map: DisplayMap | None = None, compare_with: str | Path | None = None) -> str:
    dm = display_map or DisplayMap()
    manifest, rounds, segments = load_run(run_dir)
    traj = trajectory_table(rounds)
    goal = goal_summary(manifest, rounds)
    A, B = dm.name("A"), dm.name("B")
    lines = [f"# 여론장 P0 리포트 — {manifest['config']['name']}", "",
             f"- 인구: {manifest['data']['n_voters']:,} ({manifest['data']['population_source']})",
             f"- 사전분포 출처: {manifest['data']['priors_provenance']}",
             f"- 라운드: {len(rounds)} / 시드: {manifest['config']['seed']} / 윤리수위: {manifest['config']['ethics_level']} "
             f"/ 방어자: {'on' if manifest['config']['defender']['enabled'] else 'off'} (강도 {manifest['config']['defender']['strength']})",
             f"- 백엔드: {manifest.get('backend', {}).get('name')}", "",
             f"## 지지율 궤적 ({A} / {B}, 결정층 기준)", "",
             "| round | " + A + " | " + B + " | 부동(전체) | 노출 | L2 | 방어탐지 | 되돌림 | 누적비용 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in traj.iter_rows(named=True):
        lines.append(f"| {row['round']} | {row['A']:.4f} | {row['B']:.4f} | {row['undecided']:.4f} | {row['exposed']:,} "
                     f"| {row['l2']} | {'Y' if row['defended'] else '-'} | {row['reverted']:,} | {row['cost_cum']:,.0f} |")
    lines += ["", "## 목표 달성", "",
              f"- 목표: {A} 우위 마진 ≥ {goal['goal_margin']}",
              f"- 달성: {'예' if goal['achieved'] else '아니오'}"
              + (f" (라운드 {goal['achieved_round']}, 비용 {goal['cost_at_achievement']:,.0f})" if goal['achieved'] else ""),
              f"- 최종: {A} {goal['final_A']:.4f} / {B} {goal['final_B']:.4f} / 마진 {goal['final_margin']:+.4f}", ""]
    if segments is not None and segments.height:
        contrib = segment_contribution(segments)
        lines += ["## 세그먼트별 기여 (순 A방향 이동 상위)", "",
                  "| 셀 | n | 노출 | ↑ | ↓ | 되돌림 | 순이동 | 순이동률 | 리터러시 | 설득가능성 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in contrib.iter_rows(named=True):
            lines.append(f"| {r['label']} | {r['n']:,} | {r['exposed']:,} | {r['moved_up']:,} | {r['moved_down']:,} | {r['reverted']:,} "
                         f"| {r['net_up']:,} | {r['net_up_rate']:.4f} | {r['mean_literacy']:.2f} | {r['mean_persuadability']:.2f} |")
        lv = literacy_vulnerability(segments)
        lines += ["", "## 취약성: 리터러시 vs 순이동률",
                  f"- 셀 단위 Pearson r = {lv['pearson_r']:.3f} (n_cells={lv['n_cells']})" if lv["pearson_r"] is not None else "- 계산 불가", ""]
    lines += ["## 라운드별 조작 vs 방어 로그", ""]
    for r in rounds:
        m = r["manipulator"]
        d = r["defense"]
        lines.append(f"- R{r['round_no']}: 조작 [{m['frame_ko']}/{m['claim_tier_ko']}] 셀 {m['n_target_cells']}개, "
                     f"노출 {r['exposure']['n_exposed']:,}, L1↑{r['l1']['n_up']:,} L2↑{r['l2']['n_delta_up']} L3↑{r['l3']['n_moved_up']} "
                     f"→ 방어 {'탐지' if d['detected'] else '미탐지'}(p={d['p_detect']}), 정정노출 {d['n_corr_exposed']:,}, 되돌림 {d['n_reverted']:,}")
    if compare_with:
        cmp_df = compare_runs(run_dir, compare_with)
        lines += ["", "## 방어자 on/off 대조 (같은 시드)", "", f"| round | {A} 방어on | {A} 방어off | 차이(off−on) | 되돌림 |", "|---|---|---|---|---|"]
        for r in cmp_df.iter_rows(named=True):
            lines.append(f"| {r['round']} | {r['A_defended']:.4f} | {r['A_undefended']:.4f} | {r['delta_A']:+.4f} | {r['reverted']:,} |")
        lines += ["", f"방어자가 막아낸 최종 {A} 지지율 차이: {cmp_df['delta_A'][-1]:+.4f}"]
    text = "\n".join(lines) + "\n"
    return dm.render(text)
