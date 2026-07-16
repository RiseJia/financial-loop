"""个人组合与风险管理层——workflow.md 第五部件（风险管理）的代码实现。

此前状态：仓位纪律只存在于文档（docs/long_term.md 的原则、docs/chain/08 的
水位规则），没有代码强制执行。本模块补齐：

  load_portfolio    读取个人持仓（config/portfolio.local.yaml 优先，gitignore 保护隐私）
  position_size     ATR 仓位法：股数 = 单笔风险额 ÷ (N×ATR)，受单标的上限约束
  evaluate          组合体检：权重/集中度/止损触发/现金下限/持仓关联的论点触发器

设计原则：
  - 纯函数核心（价格/分组/触发器作为参数传入）→ 可离线测试，联网只是外壳；
  - 检查只输出 findings 不阻断——工具提示纪律，执行纪律的是人；
  - 无价格时按成本价估值并显式标注（宁可显形的降级，不要静默的假精确）。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import repo_root

DEFAULT_LIMITS = {
    "max_position_pct": 0.20,   # 单标的上限（docs/long_term.md：15-20%）
    "max_group_pct": 0.40,      # 单子链/行业上限
    "min_cash_pct": 0.05,       # 现金下限（熊市的廉价筹码属于有现金的人）
    "risk_per_trade": 0.005,    # 单笔风险占总资产（docs/intraday.md：≤0.5%）
    "stop_atr_mult": 2.0,       # 止损距离 = N×ATR
}


def load_portfolio(path: str | Path | None = None) -> dict:
    """读取组合文件。优先 config/portfolio.local.yaml（gitignored，放真实持仓），
    退回 config/portfolio.yaml（模板）。返回 {cash, limits, positions}。"""
    if path is None:
        local = repo_root() / "config" / "portfolio.local.yaml"
        path = local if local.exists() else repo_root() / "config" / "portfolio.yaml"
    path = Path(path)
    if not path.exists():
        return {"cash": 0.0, "limits": dict(DEFAULT_LIMITS), "positions": []}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # limits 合并：白名单 + float 强转（红队 C1：字符串值曾直接崩掉全部风控；
    # 拼错的键曾被静默收录，用户以为放宽了上限实则默认值仍生效）
    limits = dict(DEFAULT_LIMITS)
    config_warnings: list[str] = []
    for k, v in (raw.get("limits") or {}).items():
        if k not in DEFAULT_LIMITS:
            config_warnings.append(f"limits 未知键 '{k}' 被忽略（可用：{'/'.join(DEFAULT_LIMITS)}）")
            continue
        try:
            limits[k] = float(v)
        except (TypeError, ValueError):
            config_warnings.append(f"limits.{k}={v!r} 非数值，改用默认 {DEFAULT_LIMITS[k]}")
    return {
        "cash": float(raw.get("cash") or 0.0),
        "limits": limits,
        "positions": list(raw.get("positions") or []),
        "config_warnings": config_warnings,
    }


def position_size(price: float, atr: float, total_capital: float,
                  risk_frac: float = 0.005, stop_atr_mult: float = 2.0,
                  max_position_pct: float = 0.20) -> dict:
    """ATR 仓位法（docs/indicators.md ATR 节的代码化）。

    股数 = (总资产 × 单笔风险) ÷ (N×ATR)   ——波动大的票自动小仓位
    同时受单标的市值上限约束，两者取小；binding 标注哪条约束生效。
    止损价 = 现价 − N×ATR。
    """
    if price <= 0 or atr <= 0 or total_capital <= 0:
        raise ValueError("price/atr/total_capital 必须为正")
    stop_dist = stop_atr_mult * atr
    if stop_dist >= price:
        # 红队 M2：高波动票 stop=price-N×ATR 会算出负数（不可成交的止损）
        return {"shares": 0, "stop": None, "viable": False,
                "reason": f"{stop_atr_mult}×ATR({stop_dist:.2f}) ≥ 现价({price:.2f})——"
                          "波动过大，ATR 止损法对该标的不适用（或调小止损倍数）"}
    shares_by_risk = (total_capital * risk_frac) / stop_dist
    shares_by_cap = (total_capital * max_position_pct) / price
    shares = int(min(shares_by_risk, shares_by_cap))
    if shares == 0:
        return {"shares": 0, "stop": round(price - stop_dist, 2), "viable": False,
                "reason": "该风险参数下不足以买入 1 股（资金过小或股价过高）"}
    value = shares * price
    return {
        "shares": shares, "viable": True,
        "stop": round(price - stop_dist, 2),
        "stop_dist_pct": round(stop_dist / price * 100, 2),
        "risk_amount": round(shares * stop_dist, 2),   # 实际承担的风险额
        "position_value": round(value, 2),
        "position_pct": round(value / total_capital * 100, 2),
        "binding": "risk" if shares_by_risk <= shares_by_cap else "max_position",
    }


def evaluate(portfolio: dict, prices: dict[str, float],
             groups: dict[str, str] | None = None,
             node_alerts: dict[str, list[str]] | None = None,
             valid_nodes: set[str] | None = None) -> dict:
    """组合体检（纯函数）。

    prices:      {ticker: 最新价}；缺失的持仓按成本价估值并标注
    groups:      {ticker: 子链/行业标签}（集中度检查用，来自 universe）
    node_alerts: {node名: [触发器警告文本]}（来自 research_state 的 warning/fired）
    返回 {rows, total_value, cash, cash_pct, findings:[(level, msg)]}
      level: critical（止损已触发/破产级） / warning（纪律越线） / info（提示）
    """
    groups = groups or {}
    node_alerts = node_alerts or {}
    limits = portfolio["limits"]
    findings: list[tuple[str, str]] = []
    rows = []

    total_pos_value = 0.0
    for p in portfolio["positions"]:
        ticker = p["ticker"]
        shares = float(p["shares"])
        if shares < 0:
            # 红队 M4：做空记录的 pnl/止损方向/权重语义与多头全反，本层不支持——
            # 拒算并显形，胜过按错误语义静默计算
            findings.append(("warning",
                             f"{ticker}：shares<0（做空记录）不被组合体检支持，该行按 0 计入"
                             "——空头的止损在上方、盈亏方向相反，请单独人工管理"))
            rows.append({"ticker": ticker, "shares": shares, "cost": 0.0,
                         "price": None, "value": 0.0, "pnl_pct": None, "stop": None,
                         "group": groups.get(ticker, "?"), "node": p.get("node"),
                         "priced": False})
            continue
        cost = float(p.get("cost") or 0.0)
        price = prices.get(ticker)
        priced = price is not None and price > 0
        mv = shares * (price if priced else cost)
        total_pos_value += mv
        pnl_pct = (price / cost - 1) * 100 if (priced and cost > 0) else None
        stop = p.get("stop")
        rows.append({
            "ticker": ticker, "shares": shares, "cost": cost,
            "price": price if priced else None, "value": mv,
            "pnl_pct": round(pnl_pct, 1) if pnl_pct is not None else None,
            "stop": stop, "group": groups.get(ticker, "?"),
            "node": p.get("node"), "priced": priced,
        })
        if not priced:
            findings.append(("info", f"{ticker}：无最新价，按成本估值（数值仅供参考）"))
        if priced and stop and price <= float(stop):
            findings.append(("critical",
                             f"{ticker}：现价 {price:.2f} ≤ 止损 {float(stop):.2f}——止损已触发，"
                             "按纪律执行离场（不执行止损的止损只是心理安慰）"))

    total = total_pos_value + portfolio["cash"]
    if total <= 0:
        return {"rows": rows, "total_value": 0.0, "cash": portfolio["cash"],
                "cash_pct": 0.0, "findings": [("info", "组合为空")]}

    # 权重与集中度——单标的检查按 **ticker 聚合**而非按行：
    # 红队 C2：分批建仓记成多条 lot（各 15%）曾绕过 20% 上限检查
    group_weight: dict[str, float] = {}
    ticker_weight: dict[str, float] = {}
    for r in rows:
        w = r["value"] / total
        r["weight_pct"] = round(w * 100, 1)
        ticker_weight[r["ticker"]] = ticker_weight.get(r["ticker"], 0.0) + w
        group_weight[r["group"]] = group_weight.get(r["group"], 0.0) + w
    for tkr, w in ticker_weight.items():
        if w > limits["max_position_pct"]:
            findings.append(("warning",
                             f"{tkr}：合计权重 {w:.0%} 超单标的上限 "
                             f"{limits['max_position_pct']:.0%}——信仰再强也要给'自己错了'留余地"))
    for g, w in group_weight.items():
        if g != "?" and w > limits["max_group_pct"]:
            findings.append(("warning",
                             f"子链『{g}』合计权重 {w:.0%} 超上限 {limits['max_group_pct']:.0%}"
                             "——子链系统性风险=组合风险"))

    cash_pct = portfolio["cash"] / total
    if cash_pct < limits["min_cash_pct"]:
        findings.append(("warning",
                         f"现金占比 {cash_pct:.0%} 低于下限 {limits['min_cash_pct']:.0%}"
                         "——熊市的廉价筹码只属于有现金的人"))

    # 持仓关联的论点触发器（research_state 联动）
    for r in rows:
        node = r.get("node")
        # 红队 M3：node 拼错时曾静默失联——用户以为挂了联动实则没挂
        if node and valid_nodes is not None and node not in valid_nodes:
            findings.append(("info",
                             f"{r['ticker']}：论点节点 '{node}' 在 research_state 中不存在"
                             "（拼写检查——联动未生效）"))
            continue
        if node and node in node_alerts:
            for alert in node_alerts[node]:
                findings.append(("warning",
                                 f"{r['ticker']}（论点节点 {node}）：🟡 {alert}——"
                                 "持仓的研究前提出现黄灯，复核后决定持有/减仓"))

    for w in portfolio.get("config_warnings") or []:
        findings.append(("warning", f"配置：{w}"))
    return {"rows": rows, "total_value": round(total, 2), "cash": portfolio["cash"],
            "cash_pct": round(cash_pct * 100, 1), "findings": findings}


def node_alerts_from_state(state: dict) -> dict[str, list[str]]:
    """从 research_state 提取各节点的 warning/fired 触发器文本（组合体检联动用）。"""
    out: dict[str, list[str]] = {}
    for key, node in (state.get("nodes") or {}).items():
        alerts = [t.get("condition", "?") for t in (node.get("triggers") or [])
                  if t.get("status") in ("warning", "fired")]
        if alerts:
            out[key] = alerts
    return out
