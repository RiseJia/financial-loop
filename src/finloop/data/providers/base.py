"""数据源 Provider 协议。

任何数据源只要实现 fetch_daily / fetch_intraday 并返回标准 OHLCV
（列：open/high/low/close/volume，DatetimeIndex，升序），就能接入 DataService。
新增 Polygon / Alpha Vantage / IBKR 等付费源时实现本协议即可。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

STANDARD_COLS = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """规整任意来源的 DataFrame 为标准 OHLCV。"""
    if df is None or df.empty:
        return empty_ohlcv()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df[[c for c in STANDARD_COLS if c in df.columns]]
    df = df.dropna(subset=["close"])
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLS)


@runtime_checkable
class Provider(Protocol):
    name: str

    def fetch_daily(self, ticker: str, period: str) -> pd.DataFrame: ...

    def fetch_intraday(self, ticker: str, interval: str, period: str) -> pd.DataFrame: ...
