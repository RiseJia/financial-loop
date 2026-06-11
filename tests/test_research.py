import numpy as np
import pandas as pd

from finloop.backtest.batch import ACTIVE_STRATEGIES, run_scenario_matrix
from finloop.backtest.scenarios import SCENARIOS, generate, make_ohlcv_from_closes
from finloop.data.demo import run_demo
from finloop.data.quality import validate_ohlcv
from finloop.screener import score_universe


# ---------------------------------------------------------------- 情景生成器

def test_scenarios_reproducible_and_valid():
    for name in SCENARIOS:
        a = generate(name, n_paths=2, base_seed=1)
        b = generate(name, n_paths=2, base_seed=1)
        pd.testing.assert_frame_equal(a[0], b[0])  # 同 seed 可复现
        assert not a[0].equals(a[1])               # 不同 seed 路径独立
        rep = validate_ohlcv(a[0], name)            # 合成数据必须过自家校验
        assert rep.ok, rep.summary()


def test_scenario_shapes():
    bull = generate("secular_bull", n_paths=1)[0]
    assert len(bull) == 5 * 252
    bear = generate("grinding_bear", n_paths=1)[0]
    assert bear["close"].iloc[-1] == bear["close"].iloc[-1]  # 非 NaN


def test_make_ohlcv_volume_correlates_with_moves():
    closes = np.array([100.0] * 50 + [110.0] + [110.0] * 49)
    df = make_ohlcv_from_closes(closes)
    assert df["volume"].iloc[50] > df["volume"].iloc[10]  # 跳变日放量


# ---------------------------------------------------------------- 批量矩阵

def test_scenario_matrix_smoke():
    m = run_scenario_matrix(n_paths=1, cost_bps=10)
    assert set(m["scenario"]) == set(SCENARIOS)
    assert set(m["strategy"]) == set(ACTIVE_STRATEGIES)
    for col in ("cagr", "sharpe", "max_dd", "excess_cagr", "dd_saved"):
        assert m[col].notna().all()
    # 结构性预期：阴跌熊市中 sma200 的回撤应明显浅于买入持有
    bear = m[(m.scenario == "grinding_bear") & (m.strategy == "sma200")]
    assert (bear["dd_saved"] > 0.10).all()


# ---------------------------------------------------------------- 筛选器

def make_fund(rev, earn, margin, fpe, peg):
    return {"revenueGrowth": rev, "earningsGrowth": earn, "grossMargins": margin,
            "forwardPE": fpe, "pegRatio": peg}


def test_screener_ranks_demand_vs_valuation():
    fundamentals = {
        "CHEAP_GROWTH": make_fund(0.60, 0.80, 0.55, 18.0, 0.6),   # 高需求低估值
        "HOT_EXPENSIVE": make_fund(0.55, 0.70, 0.60, 95.0, 3.5),  # 高需求高估值
        "NO_GROWTH": make_fund(0.02, -0.05, 0.30, 30.0, 2.8),     # 低需求
    }
    scored = score_universe(fundamentals)
    assert scored.index[0] == "CHEAP_GROWTH"
    assert scored.index[-1] == "NO_GROWTH"
    assert scored.loc["CHEAP_GROWTH", "gap"] > scored.loc["HOT_EXPENSIVE", "gap"]


def test_screener_missing_fields_neutral():
    fundamentals = {
        "FULL": make_fund(0.30, 0.30, 0.50, 25.0, 1.0),
        "EMPTY": {},
        "HALF": {"revenueGrowth": 0.50, "forwardPE": 20.0},
    }
    scored = score_universe(fundamentals)
    assert scored.loc["EMPTY", "coverage"] == "0/5"
    assert scored.loc["FULL", "coverage"] == "5/5"
    assert np.isfinite(scored["gap"]).all()  # 缺数据不产生 NaN 传染


def test_screener_constant_column_no_nan():
    fundamentals = {f"T{i}": make_fund(0.2, 0.2, 0.5, 30.0, 1.5) for i in range(4)}
    scored = score_universe(fundamentals)  # 零方差列 → z 全 0，不除零
    assert (scored["gap"] == 0).all()


# ---------------------------------------------------------------- 质量演示

def test_quality_demo_runs_and_catches_all():
    text = run_demo()
    assert "拒绝" in text and "警告" in text
    assert "clean-fallback" in text          # 降级到了备源
    assert text.count("🚫") >= 3              # 至少三类结构性故障被拒绝
    assert "⚠️" in text                       # 统计类故障被警告
