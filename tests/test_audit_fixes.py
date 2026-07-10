"""红队审计（M1/M2/m1-m8）修复的回归测试。每个测试对应一条审计发现。"""

import numpy as np
import pandas as pd

from finloop.backtest.event_study import event_study, format_event_study
from finloop.backtest.metrics import compute_metrics
from finloop.backtest.scenarios import generate
from finloop.data.quality import validate_ohlcv
from finloop.data.service import DataService
from finloop.indicators import bollinger, enrich, rsi
from finloop.signals import detect_momentum_switches
from finloop.signals.turning_points import detect_turning_points
from tests.conftest import make_ohlcv
from tests.test_data_quality import FakeProvider


# ---------------------------------------------------------------- M2：校验器盲区

def test_M2_nan_close_rejected(trending_up):
    df = trending_up.copy()
    df.iloc[50:130, df.columns.get_loc("close")] = np.nan
    rep = validate_ohlcv(df, "NAN")
    assert not rep.ok and any("缺失" in e for e in rep.errors)


def test_M2_negative_volume_rejected(trending_up):
    df = trending_up.copy()
    df.iloc[20:40, df.columns.get_loc("volume")] = -1e6
    rep = validate_ohlcv(df, "NEGVOL")
    assert not rep.ok and any("负成交量" in e for e in rep.errors)


def test_M2_negative_open_low_rejected(trending_up):
    df = trending_up.copy()
    df.iloc[7, df.columns.get_loc("open")] = -50.0
    df.iloc[7, df.columns.get_loc("low")] = -60.0
    rep = validate_ohlcv(df, "NEGOHL")
    assert not rep.ok and any("open/high/low" in e for e in rep.errors)


# ---------------------------------------------------------------- M1：事件研究全样本

def test_M1_cap_none_returns_full_sample():
    # 箱体震荡情景：均线反复缠绕，中期金叉次数远超 10
    df = enrich(generate("range_bound", n_paths=1, base_seed=3)[0])
    capped = detect_turning_points(df, lookback=len(df))           # 默认 cap=10
    full = detect_turning_points(df, lookback=len(df), cap=None)   # 全样本
    by_type_capped = pd.Series([e["type"] for e in capped]).value_counts()
    by_type_full = pd.Series([e["type"] for e in full]).value_counts()
    assert by_type_capped.max() <= 10
    assert by_type_full.max() > 10           # 截断确实曾在丢样本
    assert len(full) >= len(capped)


def test_M1_event_study_uses_full_sample():
    df = enrich(generate("range_bound", n_paths=1, base_seed=3)[0])
    result = event_study(df)
    ns = [e["horizons"][h]["n"] for k, e in result.items() if k != "_baseline"
          for h in e["horizons"]]
    assert max(ns) > 10  # 至少有一类信号样本数超过旧上限 10


# ---------------------------------------------------------------- m1：回撤含初始资本

def test_m1_first_day_loss_counts_as_drawdown():
    m = compute_metrics(pd.Series([-0.40] + [0.0] * 49))
    assert abs(m["max_drawdown"] - (-0.40)) < 1e-12


# ---------------------------------------------------------------- m2：RSI 单边极值

def test_m2_rsi_monotonic_up_is_100():
    r = rsi(pd.Series(100 * 1.01 ** np.arange(60)))
    assert r.iloc[-1] > 99.9


def test_m2_rsi_flat_is_50():
    r = rsi(pd.Series([100.0] * 60))
    assert (r == 50.0).all()


# ---------------------------------------------------------------- m3：布林总体标准差

def test_m3_bollinger_population_std():
    closes = pd.Series(np.random.default_rng(0).normal(100, 2, 60))
    bb = bollinger(closes, window=20)
    manual_std = closes.rolling(20).std(ddof=0)
    expected_upper = closes.rolling(20).mean() + 2 * manual_std
    pd.testing.assert_series_equal(bb["bb_upper"], expected_upper, check_names=False)


# ---------------------------------------------------------------- m4：暖机期伪切换

def test_m4_pure_uptrend_no_spurious_switch():
    """从未出现过空头动量的序列，不应检出『空转多』伪事件。"""
    df = enrich(make_ohlcv(100 * 1.003 ** np.arange(300)))
    events = detect_momentum_switches(df, lookback=len(df))
    assert events == []


# ---------------------------------------------------------------- m5：NaN t 值渲染

def test_m5_nan_tstat_not_marked_significant():
    fake = {
        "_baseline": {5: {"mean": 0.001, "median": 0.001, "std": 0.01, "pos_rate": 0.5}},
        "单样本信号": {"direction": "bullish", "n": 1, "horizons": {
            5: {"n": 1, "mean": 0.10, "median": 0.10, "hit_rate": 1.0,
                "baseline_mean": 0.001, "excess": 0.099, "t_stat": float("nan")},
        }},
    }
    text = format_event_study(fake)
    row = [l for l in text.splitlines() if "单样本信号" in l][0]
    assert "✱" not in row and "nan" not in row


# ---------------------------------------------------------------- m6：暖机期跳变检测

def test_m6_jump_in_first_bars_detected(trending_up):
    df = trending_up.copy()
    loc = df.columns.get_loc("close")
    df.iloc[5, loc] = df.iloc[4, loc] * 3.0  # 第 5 根 +200%
    hi = df.columns.get_loc("high")
    df.iloc[5, hi] = df.iloc[5, loc] * 1.01
    df.iloc[6, df.columns.get_loc("open")] = df.iloc[5, loc]  # 保持后续 OHLC 自洽
    df.iloc[6, hi] = max(df.iloc[6]["open"], df.iloc[6]["close"]) * 1.005
    rep = validate_ohlcv(df, "EARLYJUMP")
    assert any("跳变" in w for w in rep.warnings)


# ---------------------------------------------------------------- m7：缓存过质量门

def test_m7_corrupted_cache_treated_as_miss(trending_up, monkeypatch):
    bad = trending_up.copy()
    bad.iloc[10, bad.columns.get_loc("close")] = -9.0
    import finloop.data.service as service_mod
    monkeypatch.setattr(service_mod.cache, "load", lambda *a, **k: bad)
    monkeypatch.setattr(service_mod.cache, "save", lambda *a, **k: None)
    svc = DataService(FakeProvider("p", trending_up),
                      FakeProvider("f", trending_up), use_cache=True)
    df = svc.get_daily("X")
    assert not df.empty
    assert svc.last_source == "p"  # 坏缓存被拒，落到主源而不是直接交付


# ---------------------------------------------------------------- m8：负 PE 不是便宜

def test_m8_negative_pe_not_ranked_cheapest():
    from finloop.screener import score_universe
    base = {"revenueGrowth": 0.40, "earningsGrowth": 0.40, "grossMargins": 0.50}
    # 亏损公司负 PE → NaN，但高 P/S 施加估值惩罚（不再逃到中性 z=0）
    fundamentals = {
        "LOSS_MAKER": {**base, "forwardPE": -40.0, "priceToSalesTrailing12Months": 25.0},
        "FAIR": {**base, "forwardPE": 25.0, "priceToSalesTrailing12Months": 6.0},
        "PRICEY": {**base, "forwardPE": 90.0, "priceToSalesTrailing12Months": 20.0},
    }
    scored = score_universe(fundamentals)
    assert scored.index[0] == "FAIR"          # 真低估值第一
    assert scored.loc["LOSS_MAKER", "gap"] <= scored.loc["FAIR", "gap"]  # P/S 惩罚亏损高估值


# ---------------------------------------------------------------- 规则③：增速强背离

def test_growth_divergence_nittobo_pattern():
    """同号但差距>150pp（卖楼收益假便宜）→ 触发。"""
    from finloop.data.fundamentals import validate_fundamentals
    warns = validate_fundamentals({"revenueGrowth": 0.10, "earningsGrowth": 1.90})
    assert any("强背离" in w for w in warns)


def test_growth_divergence_kyec_pattern():
    """异号且差距>30pp（基期卖厂收益假恶化）→ 触发。"""
    from finloop.data.fundamentals import validate_fundamentals
    warns = validate_fundamentals({"revenueGrowth": 0.39, "earningsGrowth": -0.47})
    assert any("强背离" in w for w in warns)


def test_growth_divergence_operating_leverage_passes():
    """正常经营杠杆（同号、差距46pp）→ 不触发。"""
    from finloop.data.fundamentals import validate_fundamentals
    warns = validate_fundamentals({"revenueGrowth": 0.39, "earningsGrowth": 0.85})
    assert not any("强背离" in w for w in warns)
