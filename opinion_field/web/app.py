"""Web UI backend (FastAPI). Everything the CLI does, from the browser: build data, edit configs,
run / stop / resume campaigns, defender on/off comparison, live progress, trajectories, diffusion
heatmap, reports. Password login (configs/web_auth.yaml) because the UI may be exposed via a tunnel."""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..config import CampaignConfig, _merge_dataclass, load_campaign
from ..schema import AGE_GROUP_LABELS, N_AGE_GROUPS, N_EDU_BUCKETS, SIDO_CANON, STATE_NAMES, cell_label
from .jobs import Job, JobManager

STATIC = Path(__file__).parent / "static"
ROOT = Path(os.environ.get("OPINION_FIELD_ROOT", ".")).resolve()
RUN_NAME_RX = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
FILE_KINDS = {"campaign": "configs/campaign.yaml", "priors": "configs/priors.yaml",
              "channels": "configs/channels.yaml", "display_map": "configs/display_map.yaml",
              "tone_bank": "configs/tone_bank.yaml"}

app = FastAPI(title="K-Propaganda", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
jobs = JobManager()

# ----------------------------------------------------------------------------- auth
_sessions: set[str] = set()
_fail: dict[str, list[float]] = {}


def _auth_path() -> Path:
    return ROOT / "configs" / "web_auth.yaml"


def _password() -> str:
    p = _auth_path()
    if p.exists():
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if d.get("password"):
            return str(d["password"])
    pw = secrets.token_urlsafe(12)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# 웹 UI 로그인 비밀번호. 바꾸려면 이 값을 수정하고 서비스를 재시작.\npassword: {pw}\n", encoding="utf-8")
    os.chmod(p, 0o600)
    return pw


def _client_ip(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip() \
        or (request.client.host if request.client else "?")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path in ("/login", "/healthz", "/favicon.ico") or path.startswith("/static/"):
        return await call_next(request)
    tok = request.cookies.get("of_session")
    if tok and tok in _sessions:
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return RedirectResponse("/login", status_code=302)


@app.get("/favicon.ico")
def favicon():
    svg = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='7' fill='#2a78d6'/><path d='M6 22 L13 13 L19 18 L26 8' fill='none' stroke='#fff' stroke-width='3.5' stroke-linecap='round' stroke-linejoin='round'/></svg>"
    from fastapi.responses import Response
    return Response(svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/healthz")
def healthz():
    return {"ok": True, "time": time.time()}


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return (STATIC / "login.html").read_text(encoding="utf-8")


@app.post("/login")
async def login(request: Request):
    form = await request.form()
    ip = _client_ip(request)
    now = time.time()
    recent = [t for t in _fail.get(ip, []) if now - t < 600]
    _fail[ip] = recent
    if len(recent) >= 8:
        return HTMLResponse("<p>시도 횟수 초과. 10분 후 다시 시도하세요.</p>", status_code=429)
    if hmac.compare_digest(str(form.get("password", "")), _password()):
        tok = secrets.token_urlsafe(32)
        _sessions.add(tok)
        resp = RedirectResponse("/", status_code=303)
        secure = request.headers.get("x-forwarded-proto", "") == "https"
        resp.set_cookie("of_session", tok, httponly=True, samesite="lax", secure=secure, max_age=30 * 24 * 3600)
        return resp
    _fail.setdefault(ip, []).append(now)
    time.sleep(1.0)
    return RedirectResponse("/login?err=1", status_code=303)


@app.post("/logout")
def logout(request: Request):
    tok = request.cookies.get("of_session")
    _sessions.discard(tok)
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("of_session")
    return resp


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- status / data
def _claude_status() -> dict[str, Any]:
    exe = shutil.which("claude")
    if not exe:
        return {"available": False}
    try:
        v = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001
        v = "?"
    creds = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / ".credentials.json"
    return {"available": True, "version": v, "credentials": creds.exists()}


_claude_cache: dict[str, Any] = {}


def _data_dirs() -> list[dict[str, Any]]:
    out = []
    for m in sorted((ROOT / "data").glob("*/manifest.json")):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        out.append({"dir": str(m.parent.relative_to(ROOT)), "n_voters": d.get("n_voters"),
                    "source": d.get("population_source"), "priors": d.get("priors_provenance"),
                    "built_at": d.get("built_at"), "n_edges": d.get("n_edges")})
    return out


@app.get("/api/status")
def status():
    if not _claude_cache or time.time() - _claude_cache.get("t", 0) > 300:
        _claude_cache.update(_claude_status(), t=time.time())
    raw = sorted(p.name for p in (ROOT / "data" / "raw" / "nemotron").glob("train-*.parquet")) if (ROOT / "data/raw/nemotron").exists() else []
    return {"root": str(ROOT), "data": _data_dirs(), "raw_shards": len(raw),
            "claude": {k: v for k, v in _claude_cache.items() if k != "t"},
            "job_current": jobs.current.to_dict(tail=0) if jobs.current else None, "busy": jobs.busy(),
            "runs": _list_runs()}


# ----------------------------------------------------------------------------- configs
@app.get("/api/config")
def get_config():
    text = (ROOT / FILE_KINDS["campaign"]).read_text(encoding="utf-8")
    cfg = load_campaign(ROOT / FILE_KINDS["campaign"])
    return {"yaml": text, "config": cfg.to_dict()}


def _validate(fn, *args):
    """Turn config/YAML validation errors into a 400 with the message instead of a 500."""
    try:
        return fn(*args)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - yaml.YAMLError, ValueError, TypeError, KeyError …
        raise HTTPException(400, f"{type(e).__name__}: {e}") from e


@app.put("/api/config")
async def put_config(request: Request):
    body = await request.json()
    if "yaml" in body:
        text = body["yaml"]
        _validate(lambda: _merge_dataclass(CampaignConfig, yaml.safe_load(text) or {}))
        (ROOT / FILE_KINDS["campaign"]).write_text(text, encoding="utf-8")
    elif "overrides" in body:
        data = yaml.safe_load((ROOT / FILE_KINDS["campaign"]).read_text(encoding="utf-8")) or {}
        _apply_overrides(data, body["overrides"])
        _validate(lambda: _merge_dataclass(CampaignConfig, json.loads(json.dumps(data))))
        (ROOT / FILE_KINDS["campaign"]).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return get_config()


def _apply_overrides(data: dict, overrides: dict[str, Any]) -> None:
    for k, v in overrides.items():
        node = data
        parts = k.split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = v


@app.get("/api/files/{kind}")
def get_file(kind: str):
    if kind not in FILE_KINDS:
        raise HTTPException(404)
    return {"kind": kind, "path": FILE_KINDS[kind], "text": (ROOT / FILE_KINDS[kind]).read_text(encoding="utf-8")}


@app.put("/api/files/{kind}")
async def put_file(kind: str, request: Request):
    if kind not in FILE_KINDS:
        raise HTTPException(404)
    text = (await request.json())["text"]

    def check():
        parsed = yaml.safe_load(text)
        if kind == "campaign":
            _merge_dataclass(CampaignConfig, parsed or {})
        elif kind == "priors":
            from ..data.priors import load_priors
            tmp = ROOT / "configs" / ".priors_check.yaml"
            tmp.write_text(text, encoding="utf-8")
            try:
                load_priors(str(tmp))
            finally:
                tmp.unlink(missing_ok=True)
        elif kind == "channels":
            from ..engine.channels import ChannelTable
            tmp = ROOT / "configs" / ".channels_check.yaml"
            tmp.write_text(text, encoding="utf-8")
            try:
                ChannelTable.load(str(tmp))
            finally:
                tmp.unlink(missing_ok=True)
        elif kind == "display_map" and not isinstance((parsed or {}).get("slots"), dict):
            raise HTTPException(400, "display_map.yaml needs a top-level 'slots' mapping")

    _validate(check)
    (ROOT / FILE_KINDS[kind]).write_text(text, encoding="utf-8")
    return {"ok": True}


# ----------------------------------------------------------------------------- jobs
def _job_build(job: Job) -> None:
    from ..config import BuildConfig, DataPaths
    from ..data.build import build
    p = job.params
    out = p.get("out") or ("data/processed" if not p.get("synthetic") else f"data/processed_synth{p['synthetic']}")
    cfg = BuildConfig(seed=int(p.get("seed") or 20260920), limit=int(p["limit"]) if p.get("limit") else None,
                      paths=DataPaths(processed_dir=out))
    job.progress = {"phase": "build", "out": out}
    m = build(cfg, synthetic_n=int(p["synthetic"]) if p.get("synthetic") else None, log=job.log)
    job.progress.update(done=True, n_voters=m["n_voters"], elapsed_s=m["elapsed_s"])


def _setup_sim(job: Job):
    from ..agents.backend import build_backend
    from ..engine.channels import ChannelTable
    from ..engine.population import Population
    p = job.params
    cfg = load_campaign(ROOT / FILE_KINDS["campaign"], p.get("overrides") or {})
    if p.get("backend_kind"):
        cfg.backend = {**cfg.backend, "kind": p["backend_kind"]}
    if p.get("data_dir"):
        cfg.paths.processed_dir = p["data_dir"]
    job.log(f"[web] loading population from {cfg.paths.processed_dir}")
    pop = Population.load(cfg.paths)
    chan = ChannelTable.load(cfg.paths.channels_path)
    mp = Path(cfg.paths.manifest_json)
    dm = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    backend = build_backend(cfg.backend, cfg.seed)
    job.log(f"[web] n={pop.n:,} backend={backend.name} rounds={cfg.rounds} defender={'on' if cfg.defender.enabled else 'off'}")
    return cfg, pop, chan, backend, dm


def _progress_hook(job: Job, total: int, label: str = ""):
    def on_round(rec):
        job.progress.update(round_no=rec["round_no"], rounds=total, label=label,
                            A=rec["support"]["A"], B=rec["support"]["B"])
    return on_round


def _wait_hook(job: Job):
    def on_wait(remaining: int):
        job.progress["label"] = f"사용 한도 대기 (다음 확인 {remaining}s 후)"
    return on_wait


def _job_run(job: Job) -> None:
    from ..engine.loop import Simulation, run_until_done
    p = job.params
    cfg, pop, chan, backend, dm = _setup_sim(job)
    out = ROOT / "runs" / p["name"]
    sim = Simulation(cfg, pop, chan, backend, out, dm, log=job.log)
    run_until_done(sim, resume=bool(p.get("resume")), should_stop=job.stop_event.is_set,
                   on_round=_progress_hook(job, cfg.rounds), retry_s=int(p.get("retry_s") or 600), on_wait=_wait_hook(job))
    backend.close()
    job.progress["done"] = True


def _job_compare(job: Job) -> None:
    from ..engine.loop import Simulation
    from ..observe.display import DisplayMap
    from ..observe.report import render_markdown
    p = job.params
    cfg, pop, chan, backend, dm = _setup_sim(job)
    out = ROOT / "runs" / p["name"]
    for flag, sub in ((True, "defender_on"), (False, "defender_off")):
        if job.stop_event.is_set():
            return
        c = load_campaign(ROOT / FILE_KINDS["campaign"], {**(p.get("overrides") or {}), "defender.enabled": flag})
        c.backend, c.paths, c.rounds = cfg.backend, cfg.paths, cfg.rounds
        pop.reset_dynamic(pop.initial_state)
        job.log(f"[web] === {sub} ===")
        from ..engine.loop import run_until_done
        run_until_done(Simulation(c, pop, chan, backend, out / sub, dm, log=job.log),
                       resume=bool(p.get("resume")), should_stop=job.stop_event.is_set,
                       on_round=_progress_hook(job, cfg.rounds, sub), retry_s=int(p.get("retry_s") or 600), on_wait=_wait_hook(job))
    backend.close()
    md = render_markdown(out / "defender_on", DisplayMap(), compare_with=out / "defender_off")
    (out / "report.md").write_text(md, encoding="utf-8")
    job.progress["done"] = True


def _job_tone(job: Job) -> None:
    from ..agents.backend import build_backend
    from ..config import DataPaths
    from ..data.tone_bank import build_tone_bank
    p = job.params
    cfg = load_campaign(ROOT / FILE_KINDS["campaign"])
    if p.get("backend_kind"):
        cfg.backend = {**cfg.backend, "kind": p["backend_kind"]}
    paths = DataPaths(processed_dir=p.get("data_dir") or cfg.paths.processed_dir)
    backend = build_backend(cfg.backend, cfg.seed)
    job.progress = {"phase": "tone_bank"}
    res = build_tone_bank(paths, backend, ROOT / FILE_KINDS["tone_bank"], samples=int(p.get("samples") or 20), log=job.log)
    job.progress.update(done=True, groups_ok=res["groups_ok"], failed=len(res["failed"]))


@app.post("/api/tone-bank")
async def api_tone_bank(request: Request):
    p = await request.json()
    d = ROOT / (p.get("data_dir") or "data/processed")
    if not (d / "personas.parquet").exists():
        raise HTTPException(400, f"{d} 에 personas.parquet 가 없습니다. 먼저 데이터를 빌드하세요.")
    job = jobs.submit("tone_bank", p, _job_tone)
    return job.to_dict()


@app.get("/api/memory")
def api_memory():
    from ..engine.memory import StrategyMemory
    cfg = load_campaign(ROOT / FILE_KINDS["campaign"])
    m = StrategyMemory(ROOT / cfg.manipulator.memory_path)
    return {"path": cfg.manipulator.memory_path, "runs": m.n_runs, "rounds": m.n_rounds,
            "manip_summary": m.manip_summary(), "defender_summary": m.defender_summary(),
            "run_names": list(m.data["runs"].keys())}


@app.delete("/api/memory")
def api_memory_clear():
    cfg = load_campaign(ROOT / FILE_KINDS["campaign"])
    p = ROOT / cfg.manipulator.memory_path
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.post("/api/build")
async def api_build(request: Request):
    p = await request.json()
    if not p.get("synthetic") and not (ROOT / "data/raw/nemotron").exists():
        raise HTTPException(400, "data/raw/nemotron 에 Nemotron 샤드가 없습니다. 합성 인구(synthetic)로 빌드하세요.")
    job = jobs.submit("build", p, _job_build)
    return job.to_dict()


@app.post("/api/runs")
async def api_run(request: Request):
    p = await request.json()
    name = str(p.get("name", "")).strip()
    if not RUN_NAME_RX.match(name):
        raise HTTPException(400, "실행 이름은 영문·숫자·-_ 1~40자")
    out = ROOT / "runs" / name
    mode = p.get("mode", "run")
    exists = (out / "manifest.json").exists() or (out / "defender_on" / "manifest.json").exists()
    if exists and not p.get("resume"):
        raise HTTPException(409, f"runs/{name} 이(가) 이미 있습니다. '재개'를 켜거나 다른 이름을 쓰세요.")
    if p.get("data_dir") and not (ROOT / p["data_dir"] / "voters.parquet").exists():
        raise HTTPException(400, f"{p['data_dir']} 에 voters.parquet 가 없습니다. 먼저 데이터를 빌드하세요.")
    job = jobs.submit("compare" if mode == "compare" else "run", p, _job_compare if mode == "compare" else _job_run)
    return job.to_dict()


@app.get("/api/jobs")
def api_jobs():
    return {"jobs": jobs.list(), "current": jobs.current.id if jobs.current else None}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str, since: int = 0, tail: int | None = None):
    job = jobs.jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    return job.to_dict(since=since, tail=tail)


@app.post("/api/jobs/{job_id}/stop")
def api_job_stop(job_id: str):
    if not jobs.stop(job_id):
        raise HTTPException(404)
    return {"ok": True}


# ----------------------------------------------------------------------------- runs
def _run_summary(d: Path) -> dict[str, Any] | None:
    mp = d / "manifest.json"
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    last = None
    rp = d / "rounds.jsonl"
    n_rounds = 0
    if rp.exists():
        lines = [l for l in rp.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_rounds = len(lines)
        if lines:
            last = json.loads(lines[-1])
    return {"rounds_done": n_rounds, "rounds_total": m.get("config", {}).get("rounds"), "finished": m.get("finished"),
            "stopped": m.get("stopped", False), "goal_round": m.get("goal_round"),
            "A": last["support"]["A"] if last else None, "B": last["support"]["B"] if last else None,
            "backend": (m.get("backend") or {}).get("name"), "started_at": m.get("started_at"),
            "ethics": m.get("config", {}).get("ethics_level"), "defender": (m.get("config", {}).get("defender") or {}),
            "n_voters": (m.get("data") or {}).get("n_voters"), "seed": m.get("config", {}).get("seed")}


def _list_runs() -> list[dict[str, Any]]:
    out = []
    rdir = ROOT / "runs"
    if not rdir.exists():
        return out
    for d in sorted(rdir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        s = _run_summary(d)
        if s:
            out.append({"name": d.name, "kind": "run", "summary": s})
        elif (d / "defender_on" / "manifest.json").exists():
            out.append({"name": d.name, "kind": "compare", "summary": _run_summary(d / "defender_on"),
                        "summary_off": _run_summary(d / "defender_off")})
    return out


@app.get("/api/runs")
def api_runs():
    return {"runs": _list_runs()}


def _run_dir(name: str, sub: str | None = None) -> Path:
    if not RUN_NAME_RX.match(name):
        raise HTTPException(400)
    d = ROOT / "runs" / name
    if sub in ("defender_on", "defender_off"):
        d = d / sub
    if not (d / "manifest.json").exists():
        raise HTTPException(404, "run not found")
    return d


def _heatmap(segments: pl.DataFrame) -> dict[str, Any]:
    """Per round: (sido × age_group) cumulative net A-ward movement rate + exposed rate."""
    seg = segments.with_columns([
        (pl.col("cell") // (N_AGE_GROUPS * 2 * N_EDU_BUCKETS)).alias("sido"),
        ((pl.col("cell") // (2 * N_EDU_BUCKETS)) % N_AGE_GROUPS).alias("age"),
        (pl.col("moved_up") - pl.col("moved_down") - pl.col("reverted")).alias("net"),
    ])
    g = (seg.group_by(["round_no", "sido", "age"])
         .agg(pl.col("n").sum(), pl.col("net").sum(), pl.col("exposed").sum(), pl.col("l2_promoted").sum())
         .sort(["round_no", "sido", "age"]))
    rounds = sorted(g["round_no"].unique().to_list())
    n_s, n_a = len(SIDO_CANON), N_AGE_GROUPS
    cum = np.zeros((n_s, n_a))
    net_rate, exp_rate, n_arr = [], [], np.zeros((n_s, n_a))
    for r in rounds:
        sub = g.filter(pl.col("round_no") == r)
        net = np.zeros((n_s, n_a)); ex = np.zeros((n_s, n_a)); n_arr = np.zeros((n_s, n_a))
        for row in sub.iter_rows(named=True):
            net[row["sido"], row["age"]] = row["net"]; ex[row["sido"], row["age"]] = row["exposed"]; n_arr[row["sido"], row["age"]] = row["n"]
        cum += net
        with np.errstate(divide="ignore", invalid="ignore"):
            net_rate.append(np.where(n_arr > 0, cum / np.maximum(n_arr, 1), 0.0).round(5).tolist())
            exp_rate.append(np.where(n_arr > 0, ex / np.maximum(n_arr, 1), 0.0).round(4).tolist())
    return {"rounds": rounds, "sido": SIDO_CANON, "age": AGE_GROUP_LABELS, "net_rate": net_rate,
            "exposed_rate": exp_rate, "n": n_arr.tolist()}


def _run_detail(d: Path) -> dict[str, Any]:
    from ..observe.report import goal_summary, literacy_vulnerability, load_run, segment_contribution
    manifest, rounds, segments = load_run(d)
    out: dict[str, Any] = {"manifest": manifest, "rounds": rounds, "goal": goal_summary(manifest, rounds) if rounds else None}
    if segments is not None and segments.height:
        contrib = segment_contribution(segments, top=15)
        out["contribution"] = contrib.to_dicts()
        out["literacy"] = literacy_vulnerability(segments)
        out["heatmap"] = _heatmap(segments)
    return out


@app.get("/api/runs/{name}")
def api_run_detail(name: str, sub: str | None = None):
    d = ROOT / "runs" / name
    if sub is None and not (d / "manifest.json").exists() and (d / "defender_on" / "manifest.json").exists():
        sub = "defender_on"
    out = _run_detail(_run_dir(name, sub))
    out["name"] = name
    out["sub"] = sub
    if (d / "defender_off" / "manifest.json").exists() and (d / "defender_on" / "manifest.json").exists():
        from ..observe.report import compare_runs
        out["compare"] = compare_runs(d / "defender_on", d / "defender_off").to_dicts()
    return out


@app.get("/api/runs/{name}/report", response_class=PlainTextResponse)
def api_run_report(name: str, display: int = 0, sub: str | None = None):
    from ..observe.display import DisplayMap
    from ..observe.report import render_markdown
    d = ROOT / "runs" / name
    compare_with = None
    if (d / "defender_on" / "manifest.json").exists():
        run_dir = d / (sub or "defender_on")
        if (d / "defender_off" / "manifest.json").exists() and (sub in (None, "defender_on")):
            compare_with = d / "defender_off"
    else:
        run_dir = _run_dir(name)
    dm = DisplayMap.load(ROOT / FILE_KINDS["display_map"]) if display else DisplayMap()
    return render_markdown(run_dir, dm, compare_with=compare_with)


@app.get("/api/compare")
def api_compare(on: str, off: str):
    from ..observe.report import compare_runs
    return {"rows": compare_runs(_resolve_any(on), _resolve_any(off)).to_dicts()}


def _resolve_any(spec: str) -> Path:
    name, _, sub = spec.partition("/")
    return _run_dir(name, sub or None)


@app.delete("/api/runs/{name}")
def api_run_delete(name: str):
    if not RUN_NAME_RX.match(name):
        raise HTTPException(400)
    d = ROOT / "runs" / name
    if not d.exists():
        raise HTTPException(404)
    if jobs.current and jobs.current.params.get("name") == name:
        raise HTTPException(409, "실행 중인 run 은 삭제할 수 없습니다. 먼저 중지하세요.")
    shutil.rmtree(d)
    return {"ok": True}


@app.get("/api/runs/{name}/download/{fname}")
def api_run_download(name: str, fname: str, sub: str | None = None):
    if fname not in ("rounds.jsonl", "segments.parquet", "l2_responses.jsonl", "manifest.json", "report.md"):
        raise HTTPException(404)
    d = ROOT / "runs" / name / (sub or "")
    f = d / fname
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(str(f), filename=f"{name}_{fname}")


# ----------------------------------------------------------------------------- entry
def serve(host: str = "127.0.0.1", port: int = 8765, root: str = ".") -> None:
    global ROOT
    ROOT = Path(root).resolve()
    os.chdir(ROOT)                      # engine paths (configs/, data/, runs/) are project-relative
    (ROOT / "runs").mkdir(exist_ok=True)
    pw = _password()
    print(f"[web] K-Propaganda UI  http://{host}:{port}   root={ROOT}")
    print(f"[web] login password is in {_auth_path()} (current: {pw})")
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info", proxy_headers=True, forwarded_allow_ips="*")
