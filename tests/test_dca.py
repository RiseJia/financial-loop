"""定投/分批入场回测（backtest/dca.py）——合成数据离线测试。"""

import numpy as np
import pandas as pd
import pytest

from finloop.backtest.dca import (bucket_by_start_state, rolling_dca_study, simulate_dca,
                                  start_state, summarize_study)


def _series(vals, start="2015-01-02"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


def test_lump_sum_multiple_equals_price_ratio():
    c = _series(np.linspace(100, 150, 300))
    r = simulate_dca(c, c.index[0], 1.0, 12, "lump_sum", cash_rate_annual=0.0)
    assert r["multiple"] == pytest.approx(1.5)
    assert r["deploy_months"] == 1 and r["cash_left"] == pytest.approx(0.0)
    assert r["breakeven"] == c.index[1]          # 一次性投入、价格不跌 → 次日即「回本」（投完当日不计）
    assert r["truncated"] is False


def test_linear_flat_price_multiple_is_one_and_cash_earns_interest():
    c = _series(np.full(400, 100.0))
    r0 = simulate_dca(c, c.index[0], 1.0, 12, "linear", cash_rate_annual=0.0)
    assert r0["multiple"] == pytest.approx(1.0)
    assert r0["deploy_months"] == 12
    r1 = simulate_dca(c, c.index[0], 1.0, 12, "linear", cash_rate_annual=0.05)
    assert r1["multiple"] > 1.0                  # 未投现金计息 → 高于 1
    assert r1["multiple"] < 1.05                 # 但远低于全额吃一年 5%
    assert r1["cash_left"] == pytest.approx(0.0, abs=1e-9)   # 利息不残留：最后一期扫空
    assert r1["fully_deployed"] is not None and r1["deploy_months"] == 12


def test_drawdown_tilt_front_loads_in_bear():
    """先涨后跌 25% 再横盘：回撤加速应比线性更早投完。"""
    up = np.linspace(100, 120, 260)
    down = np.linspace(120, 90, 60)
    flat = np.full(400, 90.0)
    c = _series(np.r_[up, down, flat])
    start = c.index[290]                         # 已跌 ~17%（>10% 档）
    lin = simulate_dca(c, start, 1.0, 12, "drawdown_tilt" if False else "linear")
    tilt = simulate_dca(c, start, 1.0, 12, "drawdown_tilt")
    assert tilt["deploy_months"] < lin["deploy_months"]
    assert tilt["cash_left"] == pytest.approx(0.0, abs=1e-9)


def test_trend_gate_waits_below_200dma_and_accumulates_capped():
    """长期下行（在 200 日线下）→ 不投；转为上行穿越后以累积份额（上限 3×）补投。"""
    down = np.linspace(200, 100, 300)
    up = np.linspace(100, 260, 400)
    c = _series(np.r_[down, up])
    start = c.index[250]
    r = simulate_dca(c, start, 1.0, 12, "trend_gate")
    inv = r["invested"]
    # 起点处在 200 日线下方：前几个月累计投入应为 0
    assert inv.iloc[0] == 0.0
    assert inv.iloc[40] == 0.0
    assert r["cash_left"] == pytest.approx(0.0, abs=1e-9)       # 最终投完（上行期足够长）
    # 单月最大投入 ≤ 3 × 基础份额
    monthly = inv.diff().fillna(inv.iloc[0])
    assert monthly.max() <= 3 / 12 + 1e-9


def test_breakeven_defined_after_full_deployment():
    """V 形：投完时仍亏，之后回到预算 → 回本日期在投完之后。"""
    down = np.linspace(100, 60, 260)
    up = np.linspace(60, 130, 300)
    c = _series(np.r_[down, up])
    r = simulate_dca(c, c.index[0], 1.0, 12, "linear")
    assert r["fully_deployed"] is not None
    assert r["breakeven"] is not None and r["breakeven"] > r["fully_deployed"]
    assert r["breakeven_years"] > 0


def test_invalid_inputs():
    c = _series(np.full(50, 1.0))
    with pytest.raises(ValueError):
        simulate_dca(c, c.index[0], mode="magic")
    with pytest.raises(ValueError):
        simulate_dca(c, "2030-01-01")


def test_start_state_uses_only_past_data():
    c = _series(np.r_[np.linspace(100, 200, 300), np.linspace(200, 150, 100)])
    st = start_state(c, c.index[-1])
    assert st["off_high_pct"] == pytest.approx((150 / 200 - 1) * 100, abs=0.5)
    assert st["above_200dma"] in (True, False)
    early = start_state(c, c.index[10])
    assert early["above_200dma"] is None          # 不足 200 根 → None 而非假装知道


def test_rolling_study_and_summary_shapes():
    rng = np.random.default_rng(0)
    c = _series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 252 * 4))))
    df = rolling_dca_study(c, horizon_years=1, months=6, cash_rate_annual=0.02, step_months=3)
    assert set(df["mode"]) == {"lump_sum", "linear", "drawdown_tilt", "trend_gate"}
    assert {"multiple", "off_high_pct", "above_200dma", "breakeven_years"} <= set(df.columns)
    summ = summarize_study(df)
    assert summ.loc["lump_sum", "beat_lump_rate"] == 0.0
    assert 0 <= summ.loc["linear", "beat_lump_rate"] <= 1
    buckets = bucket_by_start_state(df, "linear")
    assert {"bucket", "trend", "n", "median"} <= set(buckets.columns)


def test_tz_aware_index_accepted():
    """yfinance 返回 tz-aware 索引；与 '2000-03-10' 这类 naive 起点比较不能炸。"""
    c = _series(np.linspace(100, 120, 300))
    c.index = c.index.tz_localize("America/New_York")
    r = simulate_dca(c, "2015-06-01", 1.0, 6, "linear")
    assert r["deploy_months"] == 6
    assert start_state(c, "2016-01-04")["off_high_pct"] <= 0


def test_truncated_horizon_is_flagged():
    c = _series(np.linspace(100, 110, 100))
    r = simulate_dca(c, c.index[0], 1.0, 3, "linear", horizon_days=500)
    assert r["truncated"] is True
    r2 = simulate_dca(c, c.index[0], 1.0, 3, "linear", horizon_days=50)
    assert r2["truncated"] is False


def test_trend_gate_keeps_trying_until_deployed():
    """长熊后牛：门控模式不会因「买点用完」而留下现金；span 记录从起点到投完的月数。"""
    down = np.linspace(200, 100, 900)        # ~3.5 年下行
    up = np.linspace(100, 300, 500)
    c = _series(np.r_[down, up])
    r = simulate_dca(c, c.index[300], 1.0, 12, "trend_gate")
    assert r["cash_left"] == pytest.approx(0.0, abs=1e-9)
    assert r["deploy_span_months"] is not None and r["deploy_span_months"] > 24
