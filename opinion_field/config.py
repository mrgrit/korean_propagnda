"""Configuration dataclasses + YAML loading (§9: campaign/knobs/backend live in YAML)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DataPaths:
    raw_nemotron_dir: str = "data/raw/nemotron"
    processed_dir: str = "data/processed"
    priors_path: str = "configs/priors.yaml"
    channels_path: str = "configs/channels.yaml"

    @property
    def voters_parquet(self) -> Path:
        return Path(self.processed_dir) / "voters.parquet"

    @property
    def personas_parquet(self) -> Path:
        return Path(self.processed_dir) / "personas.parquet"

    @property
    def network_npz(self) -> Path:
        return Path(self.processed_dir) / "network.npz"

    @property
    def manifest_json(self) -> Path:
        return Path(self.processed_dir) / "manifest.json"


@dataclass
class BuildConfig:
    seed: int = 20260920
    limit: int | None = None            # cap rows (dev runs); None = all 1M
    mean_degree: float = 8.0            # synthetic relation network
    hub_sigma: float = 0.6              # lognormal degree spread → hubs
    p_same_district: float = 0.55       # neighbour drawn from same 시군구 & age band
    p_same_sido: float = 0.30           # else same 시도
    paths: DataPaths = field(default_factory=DataPaths)


@dataclass
class L1Params:
    base_rate: float = 0.30             # base A-ward transition scale for an exposed voter
    resist_w: float = 0.8               # literacy resistance weight
    falsehood_punch: float = 0.5        # lies are more compelling (×(1+punch·tier))
    falsehood_resist: float = 0.4       # ...but literate voters resist them more
    repeat_boost: float = 0.6           # gain = (1+boost·log1p(r))·exp(-r/r0)
    repeat_r0: float = 6.0
    backlash_start: float = 8.0         # cumulative exposure beyond which backlash appears
    backlash_rate: float = 0.06
    social_w: float = 0.5               # social proof weight on neighbour A-lean
    exposure_decay: float = 0.7         # per-round decay of cumulative exposure memory
    inoculation_decay: float = 0.8
    inoculation_w: float = 0.6
    p_cap: float = 0.95
    # ladder multipliers by current state (index = SupportState code)
    ladder_up: tuple[float, ...] = (0.35, 0.12, 0.6, 1.0, 0.5, 0.0)
    ladder_down: tuple[float, ...] = (0.0, 0.0, 0.5, 1.0, 0.6, 0.12)


@dataclass
class ManipulatorParams:
    n_target_cells: int = 60            # cells targeted per round
    min_cell_size: int = 50
    learn_rate: float = 0.5             # EMA on per-cell yield
    channel_top_k: int = 3              # channels per cell
    size_weight_exp: float = 0.5        # cell ranking = vulnerability × size^exp (0 = pure vulnerability)
    impressions_per_capita_cap: float = 3.0


@dataclass
class DefenderParams:
    enabled: bool = True
    strength: float = 0.5               # 0..1 (§1.2 absolute-power input)
    budget_ratio: float = 0.6           # correction impressions ≤ ratio × manipulator impressions
    detect_bias: float = -1.2
    detect_falsehood_w: float = 2.2
    detect_strength_w: float = 1.5
    detect_volume_w: float = 3.0
    revert_base: float = 0.7
    recency_rounds: int = 2
    inoculation_gain: float = 0.5


@dataclass
class PromotionParams:
    k: int = 500                        # L2 budget per round (K)
    k_l3: int = 100                     # second-wave promotions from propagation
    band: float = 0.15                  # width of the borderline window around band_center
    band_center: float = 0.5            # §1.1: "갈리는 유권자" ≈ p≈0.5
    w_border: float = 1.0
    w_centrality: float = 0.6
    w_cell: float = 0.8
    share_default: float = 0.15         # share prob for non-L2 movers


@dataclass
class L3Params:
    transmit_base: float = 0.35
    social_base: float = 0.25
    max_seeds: int = 20000


@dataclass
class CampaignConfig:
    name: str = "p0-default"
    goal_target: str = "A"              # slot label only
    goal_margin: float = 0.05           # A share − B share among decided
    budget: float = 3_000_000.0         # total impressions-cost units over the run
    ethics_level: float = 0.5           # 0 = factual only … 1 = fabricated claims allowed
    rounds: int = 20
    seed: int = 42
    checkpoint_every: int = 1
    l1: L1Params = field(default_factory=L1Params)
    manipulator: ManipulatorParams = field(default_factory=ManipulatorParams)
    defender: DefenderParams = field(default_factory=DefenderParams)
    promotion: PromotionParams = field(default_factory=PromotionParams)
    l3: L3Params = field(default_factory=L3Params)
    backend: dict[str, Any] = field(default_factory=lambda: {"kind": "mock"})
    paths: DataPaths = field(default_factory=DataPaths)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls, data: dict[str, Any] | None):
    data = dict(data or {})
    kwargs = {}
    for f_name, f_type in cls.__dataclass_fields__.items():
        if f_name not in data:
            continue
        v = data.pop(f_name)
        sub = _SUBCONFIGS.get((cls.__name__, f_name))
        if sub is not None and isinstance(v, dict):
            v = _merge_dataclass(sub, v)
        elif isinstance(v, list) and f_name.startswith("ladder"):
            v = tuple(float(x) for x in v)
        kwargs[f_name] = v
    if data:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(data)}")
    return cls(**kwargs)


_SUBCONFIGS = {
    ("CampaignConfig", "l1"): L1Params,
    ("CampaignConfig", "manipulator"): ManipulatorParams,
    ("CampaignConfig", "defender"): DefenderParams,
    ("CampaignConfig", "promotion"): PromotionParams,
    ("CampaignConfig", "l3"): L3Params,
    ("CampaignConfig", "paths"): DataPaths,
    ("BuildConfig", "paths"): DataPaths,
}


def load_campaign(path: str | Path, overrides: dict[str, Any] | None = None) -> CampaignConfig:
    data = load_yaml(path)
    for k, v in (overrides or {}).items():       # dotted overrides e.g. defender.enabled
        node = data
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return _merge_dataclass(CampaignConfig, data)


def load_build_config(path: str | Path | None) -> BuildConfig:
    if path is None:
        return BuildConfig()
    return _merge_dataclass(BuildConfig, load_yaml(path))
