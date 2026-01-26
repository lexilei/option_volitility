"""Universe and sector mapping helpers."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import yaml


def load_universe(config_path: str) -> List[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return list(cfg["universe"]["symbols"])


def load_sector_map(source: str, path: str) -> Dict[str, str]:
    if source == "csv":
        df = pd.read_csv(path)
        if "symbol" not in df.columns or "sector" not in df.columns:
            raise ValueError("CSV must include symbol and sector columns")
        return dict(zip(df["symbol"], df["sector"]))
    raise NotImplementedError(f"Unsupported sector map source: {source}")
