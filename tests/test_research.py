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

def make_fund(rev, earn, margin, fpe, ps):
    """估值端字段现为 forwardPE + P/S（PEG 已弃用，避免增长双计）。"""
    return {"revenueGrowth": rev, "earningsGrowth": earn, "grossMargins": margin,
            "forwardPE": fpe, "priceToSalesTrailing12Months": ps}


def test_screener_ranks_demand_vs_valuation():
    fundamentals = {
        "CHEAP_GROWTH": make_fund(0.60, 0.80, 0.55, 18.0, 3.0),   # 高需求低估值
        "HOT_EXPENSIVE": make_fund(0.55, 0.70, 0.60, 95.0, 25.0), # 高需求高估值
        "NO_GROWTH": make_fund(0.02, -0.05, 0.30, 30.0, 8.0),     # 低需求
    }
    scored = score_universe(fundamentals)
    assert scored.index[0] == "CHEAP_GROWTH"
    assert scored.index[-1] == "NO_GROWTH"
    assert scored.loc["CHEAP_GROWTH", "gap"] > scored.loc["HOT_EXPENSIVE", "gap"]


def test_screener_missing_fields_neutral():
    fundamentals = {
        "FULL": make_fund(0.30, 0.30, 0.50, 25.0, 6.0),
        "EMPTY": {},
        "HALF": {"revenueGrowth": 0.50, "forwardPE": 20.0},
    }
    scored = score_universe(fundamentals)
    assert scored.loc["EMPTY", "coverage"] == "0/5"
    assert scored.loc["FULL", "coverage"] == "5/5"
    assert np.isfinite(scored["gap"]).all()  # 缺数据不产生 NaN 传染


def test_screener_constant_column_no_nan():
    fundamentals = {f"T{i}": make_fund(0.2, 0.2, 0.5, 30.0, 6.0) for i in range(4)}
    scored = score_universe(fundamentals)  # 零方差列 → z 全 0，不除零
    assert (scored["gap"] == 0).all()


# ---------------------------------------------------------------- 质量演示

def test_quality_demo_runs_and_catches_all():
    text = run_demo()
    assert "拒绝" in text and "警告" in text
    assert "clean-fallback" in text          # 降级到了备源
    assert text.count("🚫") >= 3              # 至少三类结构性故障被拒绝
    assert "⚠️" in text                       # 统计类故障被警告


# ---------------------------------------------------------------- universe 加载

def test_load_universe_nested_flatten(tmp_path, monkeypatch):
    import finloop.config as cfg
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "watchlist.yaml").write_text("tickers: [SPY]\n")
    (tmp_path / "config" / "universe_test.yaml").write_text(
        "upstream:\n"
        "  封装:\n"
        "    '4062.T': Ibiden\n"
        "    AMKR: Amkor\n"
        "  flat_direct: 直接标签\n"
        "midstream:\n"
        "  MSFT: Azure\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "repo_root", lambda: tmp_path)
    uni = cfg.load_universe("test")
    assert uni["upstream"]["4062.T"] == "封装·Ibiden"
    assert uni["upstream"]["AMKR"] == "封装·Amkor"
    assert uni["upstream"]["flat_direct"] == "直接标签"  # 新旧格式可混用
    assert uni["midstream"]["MSFT"] == "Azure"


# ---------------------------------------------------------------- 筛选器金融正确性（对抗审计修复）

def test_screener_sector_neutralization():
    """组内需求信号相同、仅商业模式中枢不同 → 行业中性化后 gap 应全 0。"""
    fund, sectors = {}, {}
    for i in range(4):
        fund[f"SW{i}"] = make_fund(0.25, 0.25, 0.85, 30.0, 12.0); sectors[f"SW{i}"] = "software"
        fund[f"HW{i}"] = make_fund(0.25, 0.25, 0.25, 25.0, 4.0); sectors[f"HW{i}"] = "hardware"
    s = score_universe(fund, sectors=sectors)
    assert s["gap"].abs().max() < 1e-9          # 组内相同 → 无行业偏置
    s_bare = score_universe(fund)               # 裸横截面 → 出现行业偏置
    assert s_bare["gap"].abs().max() > 0.1


def test_screener_ps_penalizes_loss_maker():
    """负 PE 的高增长股靠高 P/S 承受估值惩罚，不再逃到中性登顶。"""
    fund = {
        "LOSS_HYPE": make_fund(0.6, 0.6, 0.5, -40.0, 25.0),   # 亏损+高P/S
        "CHEAP_GROW": make_fund(0.5, 0.5, 0.5, 18.0, 3.0),    # 盈利+便宜
        "FAIR": make_fund(0.3, 0.3, 0.5, 25.0, 6.0),
    }
    s = score_universe(fund)
    assert s.loc["LOSS_HYPE", "valuation_score"] > 0          # 被判"贵"
    assert s.index[0] == "CHEAP_GROW"                          # 真便宜的登顶


def test_screener_no_growth_double_count():
    """估值端不含 PEG：两只需求相同、仅增速不同的股，增速只影响需求端一次。

    验证方式：估值字段完全相同（同 PE/PS），gap 差异应等于需求分差异。
    """
    fund = {
        "HI_G": make_fund(0.8, 0.8, 0.5, 25.0, 6.0),
        "LO_G": make_fund(0.1, 0.1, 0.5, 25.0, 6.0),
    }
    s = score_universe(fund)
    gap_diff = s.loc["HI_G", "gap"] - s.loc["LO_G", "gap"]
    dem_diff = s.loc["HI_G", "demand_score"] - s.loc["LO_G", "demand_score"]
    # 估值分相同 → gap 差完全来自需求分差（增长不被估值端二次计入）
    assert abs(gap_diff - dem_diff) < 1e-9
    assert abs(s.loc["HI_G", "valuation_score"] - s.loc["LO_G", "valuation_score"]) < 1e-9


def test_screener_sides_equal_scale():
    """需求/估值两侧均为字段 z 的均值 → 同 ±3 量纲，gap 不结构性偏向需求。"""
    fund = {f"T{i}": make_fund(0.1 * i, 0.1 * i, 0.3 + 0.1 * i, 20 + 5 * i, 3 + i)
            for i in range(5)}
    s = score_universe(fund)
    assert s["demand_score"].abs().max() <= 3.0 + 1e-9
    assert s["valuation_score"].abs().max() <= 3.0 + 1e-9
