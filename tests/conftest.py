import numpy as np
import pandas as pd
import pytest


def make_ohlcv(closes: np.ndarray, start="2023-01-02") -> pd.DataFrame:
    """从收盘价序列构造合成 OHLCV（交易日索引）。"""
    idx = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(len(closes), 1_000_000.0),
    })


@pytest.fixture
def trending_up():
    """稳定上升趋势：300 根，日涨约 0.3% 加小噪声。"""
    rng = np.random.default_rng(42)
    rets = 0.003 + rng.normal(0, 0.005, 300)
    return make_ohlcv(100 * np.cumprod(1 + rets))


@pytest.fixture
def trending_down():
    rng = np.random.default_rng(7)
    rets = -0.003 + rng.normal(0, 0.005, 300)
    return make_ohlcv(100 * np.cumprod(1 + rets))


@pytest.fixture
def v_reversal():
    """先跌 150 根再涨 150 根的 V 型反转。"""
    rng = np.random.default_rng(3)
    down = -0.004 + rng.normal(0, 0.004, 150)
    up = 0.005 + rng.normal(0, 0.004, 150)
    return make_ohlcv(100 * np.cumprod(1 + np.concatenate([down, up])))


@pytest.fixture
def flat_range():
    """围绕 100 的均值回归震荡。"""
    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.normal(0, 0.5, 300))
    closes = 100 + (closes - 100) * 0.3  # 压缩漂移，保持区间
    return make_ohlcv(closes)
