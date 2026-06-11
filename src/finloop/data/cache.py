"""本地 parquet 缓存：减少 API 请求、抵抗限流、提供离线可用性。

键 = (ticker, interval, period)。TTL 策略：
  日线：当天首次拉取后缓存 6 小时（盘中数据会更新，不能缓存一整天）
  分钟线：30 分钟
缓存目录 data_cache/ 已加入 .gitignore。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

from ..config import repo_root

TTL_SECONDS = {"1d": 6 * 3600, "default": 30 * 60}


def _cache_dir() -> Path:
    d = repo_root() / "data_cache"
    d.mkdir(exist_ok=True)
    return d


def _key_path(ticker: str, interval: str, period: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{ticker}_{interval}_{period}")
    return _cache_dir() / f"{safe}.parquet"


def load(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    """命中且未过期返回 DataFrame，否则 None。缓存损坏视为未命中。"""
    path = _key_path(ticker, interval, period)
    if not path.exists():
        return None
    ttl = TTL_SECONDS.get(interval, TTL_SECONDS["default"])
    if time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        path.unlink(missing_ok=True)
        return None


def save(df: pd.DataFrame, ticker: str, interval: str, period: str) -> None:
    if df is None or df.empty:
        return
    try:
        df.to_parquet(_key_path(ticker, interval, period))
    except Exception:
        pass  # 缓存写失败不应影响主流程
