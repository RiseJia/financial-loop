"""读取 config/watchlist.yaml 等配置。"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "tickers": ["AAPL", "MSFT", "NVDA"],
    "benchmarks": {"SPY": "标普500", "QQQ": "纳斯达克100"},
    "sectors": {},
    "macro": {"^VIX": "恐慌指数 VIX", "^TNX": "美债10年期收益率"},
}


def repo_root() -> Path:
    """从包位置向上定位 repo 根目录（包含 config/ 的目录）。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "watchlist.yaml").exists():
            return parent
    return Path.cwd()


def load_watchlist(path: str | Path | None = None) -> dict:
    """加载自选股配置；文件缺失时回退到内置默认值。"""
    if path is None:
        path = repo_root() / "config" / "watchlist.yaml"
    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    merged = dict(DEFAULT_CONFIG)
    merged.update({k: v for k, v in cfg.items() if v})
    return merged


def reports_dir() -> Path:
    d = repo_root() / "reports"
    d.mkdir(exist_ok=True)
    return d
