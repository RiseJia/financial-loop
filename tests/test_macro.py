"""多资产总览（finloop macro）——纯函数离线测试。"""

import numpy as np
import pandas as pd
import pytest

from finloop.macro import (asset_snapshot, build_macro_report, relative_strength,
                           risk_score)


def _df(closes, start="2025-01-02", with_hl=True):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    d = {"close": c}
    if with_hl:
        d["high"] = c * 1.01
        d["low"] = c * 0.99
    d["open"] = c
    d["volume"] = 1000
    return pd.DataFrame(d)


# ---------------------------------------------------------------- 快照数学

def test_snapshot_price_windows_and_ytd_base():
    """窗口收益用 iloc[-1-w]；YTD 基准 = 上一年最后一个收盘，而非今年首根。"""
    n = 520
    closes = np.linspace(100, 200, n)             # 单调上行
    df = _df(closes, start="2024-12-02")
    s = asset_snapshot(df, "price")
    c = df["close"]
    assert s["level"] == pytest.approx(float(c.iloc[-1]))
    assert s["1d"] == pytest.approx((c.iloc[-1] / c.iloc[-2] - 1) * 100)
    assert s["1m"] == pytest.approx((c.iloc[-1] / c.iloc[-22] - 1) * 100)
    last_year = df.index[-1].year
    prior_close = c[c.index.year < last_year].iloc[-1]
    assert s["ytd"] == pytest.approx((c.iloc[-1] / prior_close - 1) * 100)
    assert s["above_200dma"] is True and s["vs_200dma_pct"] > 0
    hi = df["high"].iloc[-252:].max(); lo = df["low"].iloc[-252:].min()
    assert s["pct_of_52w_range"] == pytest.approx((c.iloc[-1] - lo) / (hi - lo) * 100)
    assert s["off_52w_high_pct"] == pytest.approx((c.iloc[-1] / hi - 1) * 100)
    assert s["rvol20_pct"] is not None and s["rvol20_pct"] >= 0


def test_snapshot_yield_in_bp_not_percent():
    """收益率 4.00 → 4.40 是 +40bp；按价格算会得到 +10%（错误口径）。"""
    closes = [4.0] * 30 + [4.4]
    s = asset_snapshot(_df(closes, with_hl=False), "yield")
    assert s["1d"] == pytest.approx(40.0)
    assert s["vs_200dma_pct"] is None and s["rvol20_pct"] is None
    assert s["off_52w_high_pct"] is None


def test_snapshot_level_in_points():
    closes = [16.0] * 10 + [20.0]
    s = asset_snapshot(_df(closes, with_hl=False), "level")
    assert s["1d"] == pytest.approx(4.0)
    assert s["1w"] == pytest.approx(4.0)


def test_snapshot_insufficient_data_gives_none_not_zero():
    s = asset_snapshot(_df([100, 101, 102]), "price")
    assert s["1d"] is not None
    assert s["1m"] is None and s["3m"] is None
    assert s["above_200dma"] is None and s["vs_200dma_pct"] is None
    assert s["ytd"] is None                          # 无上一年数据
    assert asset_snapshot(pd.DataFrame(), "price") is None
    with pytest.raises(ValueError):
        asset_snapshot(_df([1, 2]), "ratio")


def test_relative_strength_is_ratio_return():
    """分子涨 10%、分母涨 0% → 比值 1 月 +10%；日期用内连接对齐。"""
    a = _df(np.r_[np.full(30, 100.0), np.full(21, 110.0)])   # iloc[-22] 仍是 100
    b = _df(np.full(51, 50.0))
    s = relative_strength(a, b)
    assert s["1m"] == pytest.approx(10.0)
    assert relative_strength(a, pd.DataFrame()) is None


# ---------------------------------------------------------------- 风险偏好打分

def _snap(**kw):
    base = {"kind": "price", "level": 100.0, "1m": 0.0, "above_200dma": True,
            "vs_200dma_pct": 5.0}
    base.update(kw)
    return base


def test_risk_score_risk_on_and_off_labels():
    on = risk_score(
        {"SPY": _snap(), "QQQ": _snap(), "XLP": _snap(**{"1m": -3.0}), "XLV": _snap(**{"1m": -3.0}),
         "^VIX": _snap(level=14.0), "UUP": _snap(**{"1m": 0.5})},
        {"HYG/IEF": {"1m": 1.2}}, vix_is_proxy=False)
    assert on["label"] == "risk-on" and on["score"] == 5

    off = risk_score(
        {"SPY": _snap(above_200dma=False, vs_200dma_pct=-6.0),
         "QQQ": _snap(above_200dma=False, vs_200dma_pct=-8.0),
         "XLP": _snap(**{"1m": 4.0}), "XLV": _snap(**{"1m": 2.0}),
         "^VIX": _snap(level=34.0), "UUP": _snap(**{"1m": 3.0})},
        {"HYG/IEF": {"1m": -2.0}}, vix_is_proxy=False)
    assert off["label"] == "risk-off" and off["score"] == -6


def test_risk_score_skips_vix_when_proxy():
    """只有 VIXY 代理时波动率规则不计分（代理含展期损耗，水平无意义）。"""
    r = risk_score({"SPY": _snap(), "QQQ": _snap(), "^VIX": _snap(level=12.0)}, {}, vix_is_proxy=True)
    vix_rule = [x for x in r["rules"] if x[0].startswith("波动率")][0]
    assert vix_rule[1] == 0 and "代理" in vix_rule[2]


# ---------------------------------------------------------------- 报告与代理回退

def test_report_falls_back_to_proxy_and_labels_it():
    n = 300
    data = {
        "SPY": _df(np.linspace(100, 120, n), start="2025-01-02"),
        "VIXY": _df(np.linspace(50, 40, n), start="2025-01-02"),
    }

    def fetch(sym, period):
        return data.get(sym, pd.DataFrame())

    cfg = {"groups": {"测试": [
        {"symbol": "SPY", "name": "标普500"},
        {"symbol": "^VIX", "name": "VIX", "kind": "level",
         "proxy": {"symbol": "VIXY", "name": "VIX期货ETF", "kind": "price"}},
        {"symbol": "^IRX", "name": "T-bill", "kind": "yield"},
    ]}, "relative_pairs": [["VIXY", "SPY", "测试比值"]]}
    md = build_macro_report(cfg, fetch=fetch)
    assert "⚠️代理:VIX期货ETF" in md            # 回退被显式标注
    assert "T-bill | 不可得" in md               # 无代理 → 诚实标不可得
    assert "^VIX→VIXY" in md
    assert "无真实 VIX" in md                    # 代理不进打分
    assert "VIXY/SPY" in md and "风险偏好读数" in md
