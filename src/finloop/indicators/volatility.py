"""波动率类指标：布林带 / ATR。

波动率指标回答的问题是：「市场现在有多躁动，价格偏离均值有多远」。
"""

from __future__ import annotations

import pandas as pd


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """布林带：中轨 = SMA(20)，上下轨 = 中轨 ± 2 倍标准差。

    bb_pctb     %B = (价格 - 下轨) / (上轨 - 下轨)，>1 突破上轨，<0 跌破下轨
    bb_width    带宽 = (上轨 - 下轨) / 中轨，极低带宽（挤压）往往孕育大行情
    """
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_pctb": (close - lower) / (upper - lower),
        "bb_width": (upper - lower) / mid,
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR（平均真实波幅）：衡量单根 K 线的典型波动绝对幅度。

    真实波幅 TR = max(当日高低差, |高-昨收|, |低-昨收|)，再做 Wilder 平滑。
    常用于设定止损距离（如 2×ATR）与仓位大小（波动越大仓位越小）。
    """
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()
