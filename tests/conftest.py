import json
from pathlib import Path

import numpy as np
import pytest

from opinion_field.config import BuildConfig, CampaignConfig, DataPaths
from opinion_field.data.build import build
from opinion_field.engine.channels import ChannelTable
from opinion_field.engine.population import Population

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("processed")
    cfg = BuildConfig(seed=7, paths=DataPaths(processed_dir=str(d), priors_path=str(ROOT / "configs/priors.yaml"),
                                            channels_path=str(ROOT / "configs/channels.yaml")))
    build(cfg, synthetic_n=20000, log=lambda *a, **k: None)
    return d


@pytest.fixture(scope="session")
def paths(data_dir):
    return DataPaths(processed_dir=str(data_dir), priors_path=str(ROOT / "configs/priors.yaml"),
                     channels_path=str(ROOT / "configs/channels.yaml"))


@pytest.fixture
def pop(paths):
    return Population.load(paths)


@pytest.fixture(scope="session")
def chan(paths):
    return ChannelTable.load(paths.channels_path)


def make_cfg(paths, **kw) -> CampaignConfig:
    cfg = CampaignConfig(name="test", rounds=3, seed=123, budget=60000.0, paths=paths)
    cfg.promotion.k = 40
    cfg.promotion.k_l3 = 10
    cfg.manipulator.n_target_cells = 40
    cfg.manipulator.min_cell_size = 10
    for k, v in kw.items():
        if "." in k:
            a, b = k.split(".")
            setattr(getattr(cfg, a), b, v)
        else:
            setattr(cfg, k, v)
    return cfg
