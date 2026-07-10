"""动量类指标：RSI / Stochastic / ROC。

动量指标回答的问题是：「当前涨跌的速度有多快，是否到了极端」。
"""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI（相对强弱指数），0~100。

    RSI = 100 - 100 / (1 + 平均涨幅 / 平均跌幅)，使用 Wilder 平滑。
    >70 超买、<30 超卖；趋势市中可长期钝化在极端区，需结合市场状态解读。
    """
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    # loss=0 且 gain>0 时 rs=inf → RSI=100（单边上涨的正确极值，曾被错置为中性 50）；
    # gain=loss=0（横盘）→ 0/0=NaN → 50
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0).astype(float)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """随机指标 KD：收盘价在最近 N 日高低区间中的相对位置（0~100）。

    %K = (收盘 - N日最低) / (N日最高 - N日最低) * 100
    %D = %K 的 3 日均线。>80 超买、<20 超卖，K 上穿/下穿 D 是常用触发。
    """
    lowest = low.rolling(k_window).min()
    highest = high.rolling(k_window).max()
    span = highest - lowest
    k = 100 * (close - lowest) / span
    # 仅在"区间已定义但为零"（一字横盘 highest==lowest）处置中性 50——避免除零；
    # 暖机期（前 k_window-1 根 highest/lowest 为 NaN）必须保留 NaN，不能捏造成 50，
    # 否则会在第 k_window 根出现 50→真实值跳变，诱发 KD 虚假金叉/死叉。
    flat = span.notna() & (span.abs() < 1e-12)
    k = k.mask(flat, 50.0)
    d = k.rolling(d_window).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def roc(close: pd.Series, window: int = 20) -> pd.Series:
    """变动率 Rate of Change：N 日收益率（%），最朴素的动量定义。

    ROC 由负转正/由正转负是动量方向切换的直接证据。
    """
    return close.pct_change(window) * 100
