"""组合与风险管理层（第五部件）测试——纯函数离线覆盖。"""

import pytest

from finloop.portfolio import (DEFAULT_LIMITS, evaluate, load_portfolio,
                               node_alerts_from_state, position_size)


def make_pf(positions, cash=10000.0, **limit_overrides):
    limits = dict(DEFAULT_LIMITS)
    limits.update(limit_overrides)
    return {"cash": cash, "limits": limits, "positions": positions}


# ---------------------------------------------------------------- 仓位计算

def test_position_size_risk_binding():
    """风险约束生效：股数 = 风险额/(2×ATR)，止损=现价-2ATR。"""
    s = position_size(price=100.0, atr=2.0, total_capital=100_000,
                      risk_frac=0.005, stop_atr_mult=2.0, max_position_pct=0.20)
    # 风险额 500，止损距离 4 → 125 股；市值 12,500 < 20% 上限 → risk 约束生效
    assert s["shares"] == 125
    assert s["stop"] == 96.0
    assert s["binding"] == "risk"
    assert abs(s["risk_amount"] - 500.0) < 1e-9


def test_position_size_cap_binding():
    """低波动票风险约束给出的仓位过大 → 单标的上限约束生效。"""
    s = position_size(price=100.0, atr=0.2, total_capital=100_000,
                      risk_frac=0.005, max_position_pct=0.20)
    # 风险法 500/0.4=1250 股(市值 125k>20%上限) → 上限法 200 股
    assert s["shares"] == 200
    assert s["binding"] == "max_position"
    assert s["position_pct"] == 20.0


def test_position_size_rejects_bad_inputs():
    with pytest.raises(ValueError):
        position_size(price=0, atr=1, total_capital=1000)
    with pytest.raises(ValueError):
        position_size(price=10, atr=0, total_capital=1000)


# ---------------------------------------------------------------- 组合体检

def test_evaluate_stop_breach_critical():
    pf = make_pf([{"ticker": "A", "shares": 100, "cost": 100, "stop": 95}])
    rep = evaluate(pf, prices={"A": 94.0})
    assert any(lvl == "critical" and "止损已触发" in msg
               for lvl, msg in rep["findings"])


def test_evaluate_concentration_warnings():
    pf = make_pf([
        {"ticker": "A", "shares": 100, "cost": 100},   # 10,000
        {"ticker": "B", "shares": 100, "cost": 100},   # 10,000 同子链
    ], cash=2000)
    rep = evaluate(pf, prices={"A": 100.0, "B": 100.0},
                   groups={"A": "光互联", "B": "光互联"})
    msgs = [m for _, m in rep["findings"]]
    assert any("超单标的上限" in m for m in msgs)      # 各 ~45% > 20%
    assert any("光互联" in m and "超上限" in m for m in msgs)  # 合计 ~91% > 40%


def test_evaluate_cash_floor():
    pf = make_pf([{"ticker": "A", "shares": 10, "cost": 100}], cash=10.0)
    rep = evaluate(pf, prices={"A": 100.0})
    assert any("现金占比" in m for _, m in rep["findings"])


def test_evaluate_unpriced_falls_back_to_cost():
    pf = make_pf([{"ticker": "A", "shares": 10, "cost": 100}])
    rep = evaluate(pf, prices={})
    row = rep["rows"][0]
    assert row["priced"] is False and row["value"] == 1000.0
    assert any("按成本估值" in m for _, m in rep["findings"])


def test_evaluate_node_alert_linkage():
    """持仓关联的论点节点有黄灯触发器 → 体检提醒。"""
    pf = make_pf([{"ticker": "2449.TW", "shares": 100, "cost": 250,
                   "node": "test_equipment"}], cash=50_000)
    rep = evaluate(pf, prices={"2449.TW": 280.0},
                   node_alerts={"test_equipment": ["AI 测试论点失效风险"]})
    assert any("论点节点 test_equipment" in m and "🟡" in m
               for _, m in rep["findings"])


def test_evaluate_clean_portfolio_no_warnings():
    """纪律内的组合：无 warning/critical（info 允许）。"""
    pf = make_pf([{"ticker": "A", "shares": 10, "cost": 100, "stop": 80}],
                 cash=10_000)   # 权重 ~9%，现金充足
    rep = evaluate(pf, prices={"A": 110.0})
    assert not [1 for lvl, _ in rep["findings"] if lvl in ("critical", "warning")]


def test_node_alerts_from_state_extracts_yellow():
    state = {"nodes": {
        "n1": {"triggers": [{"condition": "c1", "status": "warning"},
                            {"condition": "c2", "status": "ok"}]},
        "n2": {"triggers": [{"condition": "c3", "status": "ok"}]},
    }}
    alerts = node_alerts_from_state(state)
    assert alerts == {"n1": ["c1"]}


def test_load_portfolio_defaults(tmp_path):
    """空文件/缺失文件返回安全默认值；limits 与用户覆盖合并。"""
    missing = load_portfolio(tmp_path / "nope.yaml")
    assert missing["positions"] == [] and missing["limits"]["max_position_pct"] == 0.20
    f = tmp_path / "p.yaml"
    f.write_text("cash: 5000\nlimits:\n  max_position_pct: 0.10\npositions: []\n",
                 encoding="utf-8")
    pf = load_portfolio(f)
    assert pf["cash"] == 5000.0
    assert pf["limits"]["max_position_pct"] == 0.10     # 用户覆盖
    assert pf["limits"]["min_cash_pct"] == 0.05          # 默认保留


# ---------------------------------------------------------------- 红队二审修复

def test_C1_limits_string_and_unknown_key_safe(tmp_path):
    """字符串值/拼错键不再崩溃或静默——强转+警告。"""
    f = tmp_path / "p.yaml"
    f.write_text('cash: 1000\nlimits:\n  max_position_pct: "0.2"\n'
                 '  max_postion_pct: 0.5\npositions: []\n', encoding="utf-8")
    pf = load_portfolio(f)
    assert pf["limits"]["max_position_pct"] == 0.20      # 字符串被 float 强转
    assert any("未知键" in w for w in pf["config_warnings"])  # 拼错键显形
    rep = evaluate(pf, prices={})                          # 不崩溃
    assert any("配置" in m for _, m in rep["findings"])


def test_C2_split_lots_aggregate_for_position_limit():
    """同 ticker 分批建仓（各 15%）合计 30% → 触发单标的上限警告。"""
    pf = make_pf([
        {"ticker": "NVDA", "shares": 15, "cost": 100},
        {"ticker": "NVDA", "shares": 15, "cost": 100},
    ], cash=7000)   # 各 ~15%，合计 ~30% > 20%
    rep = evaluate(pf, prices={"NVDA": 100.0})
    assert any("NVDA" in m and "合计权重" in m and "超单标的上限" in m
               for _, m in rep["findings"])


def test_M2_negative_stop_rejected():
    """2×ATR ≥ 现价 → 拒绝出仓位而非给出负止损价。"""
    s = position_size(price=10.0, atr=6.0, total_capital=100_000)
    assert s["viable"] is False and s["stop"] is None
    assert "不适用" in s["reason"]


def test_m1_zero_shares_flagged():
    """资金不足 1 股 → viable=False 并说明原因。"""
    s = position_size(price=700_000.0, atr=10_000.0, total_capital=50_000)
    assert s["viable"] is False and s["shares"] == 0


def test_M4_short_position_refused_not_miscalculated():
    """做空记录被拒算并警告，不再按错误语义计算 pnl/止损/权重。"""
    pf = make_pf([{"ticker": "X", "shares": -100, "cost": 50, "stop": 60}],
                 cash=10_000)
    rep = evaluate(pf, prices={"X": 40.0})
    assert any("做空" in m for _, m in rep["findings"])
    assert not any(lvl == "critical" for lvl, _ in rep["findings"])  # 无错误的止损误报
    assert rep["rows"][0]["value"] == 0.0                             # 不污染权重分母


def test_M3_invalid_node_flagged():
    """node 拼错时显形提示联动未生效，而非静默。"""
    pf = make_pf([{"ticker": "A", "shares": 10, "cost": 100,
                   "node": "ai_capex"}], cash=10_000)
    rep = evaluate(pf, prices={"A": 100.0},
                   node_alerts={"ai-capex": ["fired!"]},
                   valid_nodes={"ai-capex"})
    assert any("联动未生效" in m for _, m in rep["findings"])
