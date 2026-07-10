import numpy as np
import pandas as pd

from finloop.indicators import (adx, atr, bollinger, ema, enrich, macd, obv,
                                roc, rsi, sma, stochastic, volume_ratio, vwap)
from tests.conftest import make_ohlcv


def test_sma_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert sma(s, 3).iloc[-1] == 4.0
    assert pd.isna(sma(s, 3).iloc[0])


def test_ema_converges_to_constant():
    s = pd.Series([10.0] * 50)
    assert abs(ema(s, 9).iloc[-1] - 10.0) < 1e-9


def test_rsi_bounds_and_direction(trending_up, trending_down):
    r_up = rsi(trending_up["close"]).dropna()
    r_down = rsi(trending_down["close"]).dropna()
    assert ((r_up >= 0) & (r_up <= 100)).all()
    assert r_up.iloc[-1] > 60   # 持续上涨 → RSI 偏高
    assert r_down.iloc[-1] < 40


def test_macd_sign_in_trend(trending_up, trending_down):
    assert macd(trending_up["close"])["macd"].iloc[-1] > 0
    assert macd(trending_down["close"])["macd"].iloc[-1] < 0


def test_bollinger_contains_price_mostly(flat_range):
    bb = bollinger(flat_range["close"])
    valid = bb["bb_upper"].notna()
    inside = ((flat_range["close"] <= bb["bb_upper"]) &
              (flat_range["close"] >= bb["bb_lower"]))[valid]
    assert inside.mean() > 0.85  # ±2σ 应覆盖绝大多数价格


def test_atr_positive(trending_up):
    a = atr(trending_up["high"], trending_up["low"], trending_up["close"]).dropna()
    assert (a > 0).all()


def test_adx_strong_in_trend_weak_in_range(trending_up, flat_range):
    adx_trend = adx(trending_up["high"], trending_up["low"], trending_up["close"])
    adx_range = adx(flat_range["high"], flat_range["low"], flat_range["close"])
    assert adx_trend["adx"].iloc[-1] > adx_range["adx"].iloc[-1]
    assert adx_trend["plus_di"].iloc[-1] > adx_trend["minus_di"].iloc[-1]


def test_stochastic_bounds(trending_up):
    st = stochastic(trending_up["high"], trending_up["low"], trending_up["close"]).dropna()
    assert ((st >= 0) & (st <= 100)).all().all()


def test_roc_sign(trending_up, trending_down):
    assert roc(trending_up["close"]).iloc[-1] > 0
    assert roc(trending_down["close"]).iloc[-1] < 0


def test_obv_rises_with_uptrend(trending_up):
    o = obv(trending_up["close"], trending_up["volume"])
    assert o.iloc[-1] > o.iloc[20]


def test_vwap_between_low_and_high():
    df = make_ohlcv(np.linspace(100, 110, 50))
    v = vwap(df["high"], df["low"], df["close"], df["volume"])
    assert (v <= df["high"].cummax()).all()
    assert (v >= df["low"].cummin()).all()


def test_volume_ratio_centered_at_one(trending_up):
    vr = volume_ratio(trending_up["volume"]).dropna()
    assert abs(vr.mean() - 1.0) < 0.01  # 恒定成交量 → 量比恒为 1


def test_enrich_adds_all_columns(trending_up):
    df = enrich(trending_up)
    for col in ["sma20", "sma50", "sma200", "macd", "macd_hist", "rsi14",
                "adx", "bb_pctb", "atr14", "obv", "vol_ratio", "roc20"]:
        assert col in df.columns, col
    assert len(df) == len(trending_up)


# ---------------------------------------------------------------- 对抗审计：指标定义修复

def test_stochastic_warmup_preserved():
    """暖机期(前13根)保留 NaN，不捏造成中性50——避免第14根跳变引发虚假金叉。"""
    import pandas as pd
    from finloop.indicators import stochastic
    closes = pd.Series(range(1, 61), dtype=float)  # 严格递增
    df = make_ohlcv(closes.to_numpy())
    k = stochastic(df["high"], df["low"], df["close"])["stoch_k"]
    assert k.iloc[:13].isna().all()      # 暖机期 NaN
    assert k.iloc[13:].notna().all()     # 之后有效


def test_volume_ratio_excludes_current_day():
    """量比分母为过去N日均量，不含当日——放量日不自我稀释。"""
    import numpy as np, pandas as pd
    from finloop.indicators import volume_ratio
    v = pd.Series([1e6] * 20 + [3e6])    # 前20日恒定，第21日放量3倍
    vr = volume_ratio(v, window=20)
    # 标准定义：3e6 / mean(前20日=1e6) = 3.0（含当日会被稀释到 < 3）
    assert abs(vr.iloc[-1] - 3.0) < 1e-9


def test_bollinger_flat_pctb_neutral():
    """一字横盘带宽为零，%B 置中性 0.5，不产生 NaN。"""
    import numpy as np, pandas as pd
    from finloop.indicators import bollinger
    bb = bollinger(pd.Series([100.0] * 40), window=20)
    post = bb["bb_pctb"].iloc[20:]
    assert post.notna().all() and (post == 0.5).all()
