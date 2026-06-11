"""行情数据层：统一封装 yfinance。

所有函数返回标准化的 OHLCV DataFrame，列名固定为：
open / high / low / close / volume，索引为 DatetimeIndex。
上层（指标、信号、策略）只依赖这个契约，更换数据源时只需改这一层。
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

STANDARD_COLS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """把 yfinance 返回的 DataFrame 规整为标准 OHLCV。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLS)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df[[c for c in STANDARD_COLS if c in df.columns]]
    df = df.dropna(subset=["close"])
    df.index = pd.to_datetime(df.index)
    return df


def get_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """日线数据。period 如 '6mo'、'2y'、'max'。获取失败返回空 DataFrame。"""
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception:
        return pd.DataFrame(columns=STANDARD_COLS)
    return _normalize(df)


def get_intraday(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """分钟线数据。获取失败返回空 DataFrame。

    雅虎限制：interval='1m' 仅最近 7 天；'5m'/'15m' 最近 60 天。
    返回带时区（美东交易时段）的 OHLCV。
    """
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception:
        return pd.DataFrame(columns=STANDARD_COLS)
    return _normalize(df)


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
