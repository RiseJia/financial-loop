import numpy as np
import pandas as pd
import pytest

from finloop.backtest import STRATEGIES, get_strategy, run_backtest
from finloop.backtest.event_study import event_study, format_event_study
from finloop.backtest.metrics import compute_metrics
from finloop.indicators import enrich
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------- 引擎会计正确性

def test_engine_accounting_exact():
    """逐 bar 手工核对会计恒等式：ret[t] = pos[t-1]*asset_ret[t] - turnover[t]*cost。"""
    closes = np.array([100.0, 110.0, 99.0, 108.9])  # 收益 +10%, -10%, +10%
    df = make_ohlcv(closes)
    pos = pd.Series([1.0, 0.0, 1.0, 1.0], index=df.index)
    cost_bps = 100.0  # 1% 便于手算

    r = run_backtest(df, pos, cost_bps=cost_bps)
    # t0: 建仓成本 1*1% = -1%
    # t1: pos[0]=1 吃 +10%，换手|0-1|=1 再扣 1% → +9%
    # t2: pos[1]=0 吃 0，换手 1 扣 1% → -1%
    # t3: pos[2]=1 吃 +10%，无换手 → +10%
    expected = (1 - 0.01) * (1 + 0.09) * (1 - 0.01) * (1 + 0.10)
    assert abs(r.equity.iloc[-1] - expected) < 1e-12
    assert r.n_trades == 3  # 建仓、清仓、再建仓


def test_engine_no_lookahead():
    """完美预知策略（仓位=当日收益符号）经过 shift 后不应获得完美收益。"""
    rng = np.random.default_rng(0)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 300))
    df = make_ohlcv(closes)
    rets = df["close"].pct_change().fillna(0)
    cheating_pos = (rets > 0).astype(float)  # 偷看当天收益

    r = run_backtest(df, cheating_pos, cost_bps=0.0)
    perfect = (1 + rets.clip(lower=0)).prod()  # 只吃上涨日的完美净值
    assert r.equity.iloc[-1] < perfect * 0.5  # shift 后作弊优势必须消失大半


def test_flat_position_zero_return():
    df = make_ohlcv(100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.02, 100)))
    r = run_backtest(df, pd.Series(0.0, index=df.index), cost_bps=10)
    assert abs(r.equity.iloc[-1] - 1.0) < 1e-12


def test_buy_hold_zero_cost_matches_benchmark():
    df = make_ohlcv(100 * np.cumprod(1 + np.random.default_rng(2).normal(0.001, 0.02, 200)))
    r = run_backtest(df, pd.Series(1.0, index=df.index), cost_bps=0.0)
    assert abs(r.equity.iloc[-1] - r.benchmark_equity.iloc[-1]) < 1e-9


# ---------------------------------------------------------------- 指标计算

def test_metrics_known_drawdown():
    # 净值 1 → 2 → 1 → 3：最大回撤 = -50%
    rets = pd.Series([1.0, -0.5, 2.0])
    m = compute_metrics(rets)
    assert abs(m["max_drawdown"] - (-0.5)) < 1e-12
    assert abs(m["total_return"] - 2.0) < 1e-12


def test_metrics_empty_safe():
    m = compute_metrics(pd.Series(dtype=float))
    assert m["sharpe"] == 0.0 and m["max_underwater_days"] == 0


# ---------------------------------------------------------------- 内置策略行为

def test_sma200_invested_in_uptrend_flat_in_downtrend(trending_up, trending_down):
    up = enrich(trending_up)
    down = enrich(trending_down)
    pos_up = STRATEGIES["sma200"](up)
    pos_down = STRATEGIES["sma200"](down)
    valid_up = pos_up[up["sma200"].notna()]
    valid_down = pos_down[down["sma200"].notna()]
    assert valid_up.mean() > 0.9   # 上升趋势中几乎一直持仓
    assert valid_down.mean() < 0.1  # 下降趋势中几乎一直空仓


def test_sma200_cuts_drawdown_in_downtrend(trending_down):
    df = enrich(trending_down)
    r = run_backtest(df, STRATEGIES["sma200"](df), cost_bps=10)
    assert r.metrics["max_drawdown"] > r.benchmark_metrics["max_drawdown"]  # 回撤更浅（数值更大）


def test_rsi_mr_state_machine():
    """构造 RSI 从超卖到中性的路径，验证持仓的进出时点。"""
    df = enrich(make_ohlcv(np.concatenate([
        100 * 0.99 ** np.arange(60),          # 持续下跌 → RSI 超卖
        100 * 0.99 ** 59 * 1.012 ** np.arange(60),  # 反弹 → RSI 回升
    ])))
    pos = STRATEGIES["rsi_mr"](df)
    assert pos.max() == 1.0          # 超卖时进了场
    assert pos.iloc[-1] == 0.0       # RSI 回升过 55 后离了场
    entered = pos.idxmax()
    assert df.loc[entered, "rsi14"] < 30  # 进场那天确实超卖


def test_momentum_strategy_hysteresis(v_reversal):
    df = enrich(v_reversal)
    pos = STRATEGIES["momentum"](df)
    assert set(pos.unique()) <= {0.0, 1.0}
    assert pos.iloc[-1] == 1.0  # V 型反转后段应为持仓状态


def test_get_strategy_unknown():
    with pytest.raises(KeyError):
        get_strategy("nope")


# ---------------------------------------------------------------- 事件研究

def test_event_study_structure(v_reversal):
    df = enrich(v_reversal)
    result = event_study(df)
    assert "_baseline" in result
    sig_types = [k for k in result if k != "_baseline"]
    assert sig_types  # V 型反转必然产生信号
    for st in sig_types:
        entry = result[st]
        assert entry["direction"] in ("bullish", "bearish")
        for h, s in entry["horizons"].items():
            assert s["n"] >= 1
            assert "excess" in s and "hit_rate" in s
    text = format_event_study(result, "TEST")
    assert "基线" in text and "|" in text
