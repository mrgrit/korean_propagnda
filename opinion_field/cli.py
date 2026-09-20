"""CLI: build-data | run | compare-defender | report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import BuildConfig, DataPaths, load_build_config, load_campaign


def _parse_set(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def cmd_build_data(a):
    from .data.build import build
    cfg = load_build_config(a.config)
    if a.limit:
        cfg.limit = a.limit
    if a.out:
        cfg.paths.processed_dir = a.out
    if a.seed is not None:
        cfg.seed = a.seed
    m = build(cfg, synthetic_n=a.synthetic)
    print(json.dumps(m, ensure_ascii=False, indent=2))


def _setup(a, overrides: dict | None = None):
    from .agents.backend import build_backend
    from .engine.channels import ChannelTable
    from .engine.population import Population
    cfg = load_campaign(a.campaign, overrides)
    if a.rounds:
        cfg.rounds = a.rounds
    if a.backend:
        cfg.backend = {"kind": a.backend, **{k: v for k, v in cfg.backend.items() if k != "kind"}}
    if a.data:
        cfg.paths.processed_dir = a.data
    pop = Population.load(cfg.paths)
    chan = ChannelTable.load(cfg.paths.channels_path)
    manifest_path = cfg.paths.manifest_json
    data_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8")) if Path(manifest_path).exists() else {}
    backend = build_backend(cfg.backend, cfg.seed)
    return cfg, pop, chan, backend, data_manifest


def cmd_run(a):
    from .engine.loop import Simulation
    cfg, pop, chan, backend, dm = _setup(a, _parse_set(a.set))
    sim = Simulation(cfg, pop, chan, backend, a.out, dm)
    sim.run(resume=a.resume)
    backend.close()
    print(f"run complete → {a.out}")


def cmd_compare_defender(a):
    from .engine.loop import Simulation
    from .observe.report import render_markdown
    from .observe.display import DisplayMap
    overrides = _parse_set(a.set)
    cfg, pop, chan, backend, dm = _setup(a, overrides)
    out = Path(a.out)
    for flag, sub in ((True, "defender_on"), (False, "defender_off")):
        c = load_campaign(a.campaign, {**overrides, "defender.enabled": flag})
        c.rounds, c.backend, c.paths = cfg.rounds, cfg.backend, cfg.paths
        pop.reset_dynamic(pop.initial_state)
        Simulation(c, pop, chan, backend, out / sub, dm).run()
    md = render_markdown(out / "defender_on", DisplayMap.load(a.display_map), compare_with=out / "defender_off")
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_web(a):
    from .web.app import serve
    serve(host=a.host, port=a.port, root=a.root)


def cmd_report(a):
    from .observe.display import DisplayMap
    from .observe.report import render_markdown
    md = render_markdown(a.run, DisplayMap.load(a.display_map), compare_with=a.compare)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    print(md)


def main(argv=None):
    p = argparse.ArgumentParser(prog="opinion-field", description="여론장 P0 engine")
    sp = p.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build-data", help="Nemotron + priors → voters.parquet/personas.parquet/network.npz")
    b.add_argument("--config", default=None)
    b.add_argument("--limit", type=int, default=None, help="use only the first N personas")
    b.add_argument("--synthetic", type=int, default=None, help="build a synthetic test population of N (no Nemotron)")
    b.add_argument("--out", default=None)
    b.add_argument("--seed", type=int, default=None)
    b.set_defaults(fn=cmd_build_data)

    def common(x):
        x.add_argument("--campaign", default="configs/campaign.yaml")
        x.add_argument("--out", required=True)
        x.add_argument("--rounds", type=int, default=None)
        x.add_argument("--backend", choices=["mock", "cc", "api"], default=None)
        x.add_argument("--data", default=None, help="processed data dir")
        x.add_argument("--set", action="append", help="dotted config override, e.g. defender.strength=0.8")

    r = sp.add_parser("run", help="run a campaign")
    common(r)
    r.add_argument("--resume", action="store_true")
    r.set_defaults(fn=cmd_run)

    c = sp.add_parser("compare-defender", help="same seed, defender on vs off, comparison report")
    common(c)
    c.add_argument("--display-map", default=None)
    c.set_defaults(fn=cmd_compare_defender)

    rp = sp.add_parser("report", help="render markdown report for a run")
    rp.add_argument("--run", required=True)
    rp.add_argument("--compare", default=None, help="second run dir (defender off) for A/B table")
    rp.add_argument("--display-map", default=None, help="configs/display_map.yaml — render-time only")
    rp.add_argument("--out", default=None)
    rp.set_defaults(fn=cmd_report)

    w = sp.add_parser("web", help="start the local web UI (everything else is done in the browser)")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--root", default=".", help="project root (configs/, data/, runs/)")
    w.set_defaults(fn=cmd_web)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
