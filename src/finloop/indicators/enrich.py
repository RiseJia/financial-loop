"""enrich：在标准 OHLCV DataFrame 上一次性叠加全部指标列。

这是指标层对上层（signals / strategy / report）的统一出口。
"""

from __future__ import annotations

import pandas as pd

from .trend import sma, ema, macd, adx
from .momentum import rsi, stochastic, roc
from .volatility import bollinger, atr
from .volume import obv, vwap, volume_ratio


def enrich(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    """返回带全部指标列的副本。

    intraday=True 时额外按交易日分组计算 VWAP（VWAP 仅对日内数据有意义）。
    """
    out = df.copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]

    out["sma20"] = sma(c, 20)
    out["sma50"] = sma(c, 50)
    out["sma200"] = sma(c, 200)
    out["ema9"] = ema(c, 9)
    out["ema21"] = ema(c, 21)

    out = out.join(macd(c))
    out = out.join(adx(h, l, c))

    out["rsi14"] = rsi(c, 14)
    out = out.join(stochastic(h, l, c))
    out["roc20"] = roc(c, 20)

    out = out.join(bollinger(c))
    out["atr14"] = atr(h, l, c)

    out["obv"] = obv(c, v)
    out["vol_ratio"] = volume_ratio(v)

    if intraday:
        out["vwap"] = (
            out.groupby(out.index.date, group_keys=False)
            .apply(lambda g: vwap(g["high"], g["low"], g["close"], g["volume"]))
        )
    return out
