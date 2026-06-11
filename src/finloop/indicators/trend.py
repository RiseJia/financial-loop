"""趋势类指标：SMA / EMA / MACD / ADX。

趋势指标回答的问题是：「价格在朝哪个方向走，走得有多坚决」。
详细教学解释见 docs/indicators.md 与 explain.py。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """简单移动平均：最近 window 根收盘价的算术平均。"""
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """指数移动平均：对近期价格赋予更高权重，比 SMA 反应更快。"""
    return close.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD（指数平滑异同移动平均线）。

    macd_line  = EMA(12) - EMA(26)      快慢均线的差，衡量短期动能相对长期的强弱
    signal     = EMA(macd_line, 9)      对 macd_line 再做一次平滑，作为触发线
    histogram  = macd_line - signal     柱状图，动能加速/减速的直接读数
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    """ADX（平均趋向指数）：衡量趋势强度（不分方向）。

    +DI / -DI 分别衡量上行 / 下行方向的力量；
    ADX 是两者差值占比的平滑，>25 通常认为存在有效趋势，<20 为震荡市。
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    # Wilder 平滑（等价于 alpha=1/window 的 EMA）
    atr_ = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / window, adjust=False).mean()
    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di})
