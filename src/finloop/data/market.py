"""行情数据入口（薄委托层，向后兼容）。

实际工作由 service.DataService 完成：缓存 → 主源(yfinance) → 备源(stooq)
→ 质量校验。上层（指标/信号/策略/报告）只依赖这里的三个函数，
契约不变：标准 OHLCV（open/high/low/close/volume，DatetimeIndex）。
"""

from __future__ import annotations

import pandas as pd

from .providers.base import STANDARD_COLS  # noqa: F401  保持旧导入路径可用
from .service import default_service


def get_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """日线数据。period 如 '6mo'、'2y'、'max'。失败返回空 DataFrame。"""
    return default_service.get_daily(ticker, period)


def get_intraday(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """分钟线数据（仅主源支持）。雅虎限制：1m 最近 7 天、5m 最近 60 天。"""
    return default_service.get_intraday(ticker, interval, period)


def get_last_quote(ticker: str) -> dict:
    """最近收盘价与涨跌幅快照（基于最近两根日线）。"""
    df = get_daily(ticker, period="10d")
    if len(df) < 2:
        return {"ticker": ticker, "price": None, "change_pct": None}
    last, prev = df["close"].iloc[-1], df["close"].iloc[-2]
    return {
        "ticker": ticker,
        "price": round(float(last), 2),
        "change_pct": round(float(last / prev - 1) * 100, 2),
        "date": df.index[-1].date().isoformat(),
    }
