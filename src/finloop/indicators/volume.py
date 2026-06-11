"""量能类指标：OBV / VWAP / 量比。

量能指标回答的问题是：「这波价格变化背后有没有真金白银在推动」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV（能量潮）：上涨日加成交量、下跌日减成交量的累计值。

    OBV 与价格的背离（价格新高而 OBV 不创新高）是经典的顶部预警。
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP（成交量加权均价）：日内交易的「公允价格」基准线。

    典型价 = (高+低+收)/3，VWAP = Σ(典型价×量) / Σ量。
    注意：VWAP 只对「单个交易日的日内数据」有意义，应按交易日分组计算
    （enrich 中对日内数据已按日期分组）。价格在 VWAP 上方视为多头掌控。
    """
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum()


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """量比：当前成交量 / 近 N 日平均成交量。>2 为显著放量，<0.5 为缩量。"""
    return volume / volume.rolling(window).mean()
