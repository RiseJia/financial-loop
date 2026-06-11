"""批量回测组织器：把单次回测升级为研究级矩阵。

量化标准的三个要求，对应三种矩阵维度：
  1. 横截面稳健性：策略 × 多标的（缓解单票运气与幸存者偏差）
     —— run_real_matrix(tickers)
  2. 状态稳健性：策略 × 多情景合成路径分布（明确回答「什么市场会亏」）
     —— run_scenario_matrix()
  3. 时间稳健性：分年度绩效（一个只靠某一年发财的策略不可信）
     —— 真实模式自动附带 per-year 分解

输出统一为研究报告 Markdown，写入 reports/research/。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .. import DISCLAIMER
from ..config import reports_dir
from ..indicators import enrich
from .engine import run_backtest
from .metrics import compute_metrics
from .scenarios import SCENARIOS, generate
from .strategy import STRATEGIES

ACTIVE_STRATEGIES = [n for n in STRATEGIES if n != "buy_hold"]


# ---------------------------------------------------------------- 情景矩阵

def run_scenario_matrix(n_paths: int = 5, cost_bps: float = 10.0) -> pd.DataFrame:
    """全部策略 × 全部情景 × n 条路径 → 长表。"""
    rows = []
    for scen in SCENARIOS:
        for i, raw in enumerate(generate(scen, n_paths)):
            df = enrich(raw)
            bench = compute_metrics(df["close"].pct_change().fillna(0.0))
            for name in ACTIVE_STRATEGIES:
                r = run_backtest(df, STRATEGIES[name](df),
                                 strategy_name=name, cost_bps=cost_bps)
                rows.append({
                    "scenario": scen, "path": i, "strategy": name,
                    "cagr": r.metrics["cagr"], "sharpe": r.metrics["sharpe"],
                    "max_dd": r.metrics["max_drawdown"],
                    "exposure": r.metrics["exposure"],
                    "bh_cagr": bench["cagr"], "bh_max_dd": bench["max_drawdown"],
                    "excess_cagr": r.metrics["cagr"] - bench["cagr"],
                    "dd_saved": r.metrics["max_drawdown"] - bench["max_drawdown"],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 真实数据矩阵

def run_real_matrix(tickers: list[str], period: str = "5y",
                    cost_bps: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """策略 × 标的矩阵 + 分年度分解。返回 (matrix, yearly)。"""
    from ..data import get_daily

    rows, yearly_rows = [], []
    for ticker in tickers:
        raw = get_daily(ticker, period=period)
        if len(raw) < 260:
            continue
        df = enrich(raw)
        bench_ret = df["close"].pct_change().fillna(0.0)
        bench = compute_metrics(bench_ret)
        for name in ACTIVE_STRATEGIES:
            r = run_backtest(df, STRATEGIES[name](df),
                             strategy_name=name, ticker=ticker, cost_bps=cost_bps)
            rows.append({
                "ticker": ticker, "strategy": name,
                "cagr": r.metrics["cagr"], "sharpe": r.metrics["sharpe"],
                "max_dd": r.metrics["max_drawdown"],
                "bh_cagr": bench["cagr"],
                "excess_cagr": r.metrics["cagr"] - bench["cagr"],
                "dd_saved": r.metrics["max_drawdown"] - bench["max_drawdown"],
                "beats_bh": r.metrics["cagr"] > bench["cagr"],
            })
            # 分年度
            strat_ret = r.equity.pct_change().fillna(0.0)
            for year, grp in strat_ret.groupby(strat_ret.index.year):
                if len(grp) < 40:
                    continue
                b_grp = bench_ret.loc[grp.index]
                yearly_rows.append({
                    "ticker": ticker, "strategy": name, "year": int(year),
                    "ret": float((1 + grp).prod() - 1),
                    "bh_ret": float((1 + b_grp).prod() - 1),
                })
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows)


# ---------------------------------------------------------------- 汇总与报告

def _agg_table(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    return df.groupby(group).agg(
        n=("sharpe", "size"),
        sharpe_med=("sharpe", "median"),
        cagr_med=("cagr", "median"),
        excess_med=("excess_cagr", "median"),
        maxdd_med=("max_dd", "median"),
        dd_saved_med=("dd_saved", "median"),
    ).reset_index()


def _md_table(df: pd.DataFrame, pct_cols: tuple[str, ...]) -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in pct_cols:
                cells.append(f"{v:+.1%}")
            elif isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


SCENARIO_EXPECTATIONS = {
    "secular_bull": "预期：趋势类接近买入持有；过度交易的策略被成本拖累。",
    "grinding_bear": "预期：趋势过滤类靠空仓少亏（看 dd_saved）；这是它们存在的意义。",
    "crash_recovery": "预期：检验崩盘时多快离场、修复时多快回场；过慢的策略两头挨打。",
    "range_bound": "预期：均值回归占优；趋势类被反复止损，亏损应有界且小。",
    "whipsaw": "预期：所有趋势策略的最坏情景。关注：最坏亏损是否可承受（生存性检验）。",
}


def write_scenario_report(matrix: pd.DataFrame, cost_bps: float = 10.0) -> str:
    today = dt.date.today().isoformat()
    agg = _agg_table(matrix, ["scenario", "strategy"])
    lines = [
        f"# 策略情景压力测试报告 — {today}",
        "",
        f"方法：{len(SCENARIOS)} 个市场情景 × 每情景 {matrix['path'].nunique()} 条独立路径 × "
        f"{len(ACTIVE_STRATEGIES)} 个策略，成本 {cost_bps}bps/边。"
        "报告中位数统计（分布而非单点）。情景为合成行情（参数校准到美股量级），"
        "用途是检验**状态依赖性与生存性**，不预测真实收益水平。",
        "",
    ]
    for scen, (_, label) in SCENARIOS.items():
        sub = agg[agg["scenario"] == scen].drop(columns="scenario")
        sub = sub.sort_values("sharpe_med", ascending=False)
        bh = matrix[matrix["scenario"] == scen][["bh_cagr", "bh_max_dd"]].median()
        lines += [
            f"## 情景：{label}（`{scen}`）",
            "",
            f"买入持有基线（中位）：CAGR {bh['bh_cagr']:+.1%}，最大回撤 {bh['bh_max_dd']:.1%}",
            "",
            *_md_table(sub, pct_cols=("cagr_med", "excess_med", "maxdd_med", "dd_saved_med")),
            "",
            f"> {SCENARIO_EXPECTATIONS[scen]}",
            "",
        ]

    # 全情景生存性总表：每个策略最坏情景的表现
    worst = (agg.sort_values("excess_med").groupby("strategy").first().reset_index()
             [["strategy", "scenario", "excess_med", "maxdd_med"]])
    worst.columns = ["strategy", "最坏情景", "最坏超额(中位)", "该情景回撤(中位)"]
    lines += ["## 生存性总表：每个策略的最坏情景", "",
              *_md_table(worst, pct_cols=("最坏超额(中位)", "该情景回撤(中位)")), "",
              "> 量化纪律：先确认最坏情景亏得起，再谈最好情景赚多少。",
              "", "---", DISCLAIMER, ""]

    report = "\n".join(lines)
    out_dir = reports_dir() / "research"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{today}_scenario_stress.md"
    path.write_text(report, encoding="utf-8")
    return str(path)


def write_real_report(matrix: pd.DataFrame, yearly: pd.DataFrame,
                      period: str, cost_bps: float = 10.0) -> str:
    today = dt.date.today().isoformat()
    lines = [
        f"# 策略横截面回测报告 — {today}",
        "",
        f"方法：{matrix['ticker'].nunique()} 只标的 × {len(ACTIVE_STRATEGIES)} 个策略，"
        f"区间 {period}，成本 {cost_bps}bps/边。",
        "",
        "⚠️ 解读警告：标的池是当前的关注列表（幸存者+选择偏差），"
        "横截面统计缓解单票运气，但不消除池子本身的偏差。",
        "",
        "## 策略横截面汇总",
        "",
    ]
    agg = matrix.groupby("strategy").agg(
        n=("ticker", "size"),
        beat_rate=("beats_bh", "mean"),
        sharpe_med=("sharpe", "median"),
        excess_med=("excess_cagr", "median"),
        dd_saved_med=("dd_saved", "median"),
    ).reset_index().sort_values("sharpe_med", ascending=False)
    lines += _md_table(agg, pct_cols=("beat_rate", "excess_med", "dd_saved_med"))

    lines += ["", "## 分年度稳健性（策略收益 − 买入持有，按年中位）", ""]
    if not yearly.empty:
        yearly = yearly.assign(excess=yearly["ret"] - yearly["bh_ret"])
        pivot = yearly.pivot_table(index="strategy", columns="year",
                                   values="excess", aggfunc="median")
        cols = ["strategy"] + [str(c) for c in pivot.columns]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for strat, row in pivot.iterrows():
            cells = [strat] + [f"{v:+.1%}" if pd.notna(v) else "-" for v in row]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "> 看的不是哪年最亮眼，而是有没有「全靠某一年」的策略——那是脆弱的信号。"]

    lines += ["", "## 明细（按标的）", ""]
    detail = matrix[["ticker", "strategy", "cagr", "bh_cagr", "excess_cagr",
                     "max_dd", "sharpe"]].sort_values(["ticker", "sharpe"],
                                                      ascending=[True, False])
    lines += _md_table(detail, pct_cols=("cagr", "bh_cagr", "excess_cagr", "max_dd"))
    lines += ["", "---", DISCLAIMER, ""]

    report = "\n".join(lines)
    out_dir = reports_dir() / "research"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{today}_cross_section.md"
    path.write_text(report, encoding="utf-8")
    return str(path)
