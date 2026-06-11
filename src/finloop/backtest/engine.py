"""向量化回测引擎。

================================================================
回测在测什么？
================================================================
回测 = 把策略函数放回历史，逐 bar 模拟「如果当时按规则执行会怎样」，
得到一条策略净值曲线，然后回答两个问题：

  1. 收益问题：扣除成本后，它赢过『买入持有』了吗？赢多少？
  2. 风险问题：拿到这个收益的路径有多痛苦（回撤/波动/最长水下期）？

核心会计恒等式（本引擎逐 bar 执行的全部逻辑）：

  strategy_ret[t] = position[t-1] × asset_ret[t] − turnover[t] × cost

  - position[t-1]：昨天收盘时决定的仓位，吃今天的涨跌。
    这一个 shift(1) 是回测正确性的命脉——没有它，策略在用
    今天的信息吃今天的收益（前视偏差），结果必然虚高。
  - turnover[t] = |position[t] − position[t-1]|：换手量
  - cost：单边交易成本（佣金+滑点），以 bps 计，默认 10bps=0.1%

执行假设（已知的简化，解读结果时要记得）：
  - 在信号 bar 的收盘价成交（实际盘中执行有滑点，靠 cost_bps 补偿）
  - 无市场冲击：假设我们的单子不影响价格（小资金近似成立）
  - 仓位可以是任意小数（不取整到股数）
================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import compute_metrics


@dataclass
class BacktestResult:
    strategy_name: str
    ticker: str
    equity: pd.Series          # 策略净值曲线（起点 1.0）
    benchmark_equity: pd.Series  # 买入持有净值
    positions: pd.Series       # 实际生效仓位（已 shift）
    metrics: dict              # 策略指标
    benchmark_metrics: dict    # 基准指标
    n_trades: int
    cost_bps: float

    def summary(self) -> str:
        m, b = self.metrics, self.benchmark_metrics
        rows = [
            ("总收益", f"{m['total_return']:+.1%}", f"{b['total_return']:+.1%}"),
            ("年化收益 CAGR", f"{m['cagr']:+.2%}", f"{b['cagr']:+.2%}"),
            ("年化波动", f"{m['vol']:.2%}", f"{b['vol']:.2%}"),
            ("夏普比率", f"{m['sharpe']:.2f}", f"{b['sharpe']:.2f}"),
            ("最大回撤", f"{m['max_drawdown']:.1%}", f"{b['max_drawdown']:.1%}"),
            ("Calmar(收益/回撤)", f"{m['calmar']:.2f}", f"{b['calmar']:.2f}"),
            ("最长水下期(交易日)", f"{m['max_underwater_days']}", f"{b['max_underwater_days']}"),
            ("日胜率", f"{m['win_rate']:.1%}", f"{b['win_rate']:.1%}"),
            ("持仓暴露", f"{m['exposure']:.1%}", f"{b['exposure']:.1%}"),
        ]
        lines = [
            f"回测：{self.ticker} × {self.strategy_name}"
            f"（{self.equity.index[0].date()} ~ {self.equity.index[-1].date()}，"
            f"成本 {self.cost_bps}bps/边，交易 {self.n_trades} 次）",
            "",
            f"{'指标':<14}{'策略':>12}{'买入持有':>12}",
        ]
        lines += [f"{name:<14}{s:>12}{bm:>12}" for name, s, bm in rows]
        return "\n".join(lines)


def run_backtest(df: pd.DataFrame, positions: pd.Series, *,
                 strategy_name: str = "strategy", ticker: str = "?",
                 cost_bps: float = 10.0) -> BacktestResult:
    """df：enrich 后的日线；positions：策略输出的目标仓位（与 df 对齐）。

    指标的 warmup 期（前 200 根 sma200 为 NaN 等）由策略自己处理
    （NaN 条件比较结果为 False → 空仓），这天然等价于「上市初期不交易」。
    """
    positions = positions.reindex(df.index).fillna(0.0).clip(-1, 1)

    asset_ret = df["close"].pct_change().fillna(0.0)
    effective_pos = positions.shift(1).fillna(0.0)   # 因果性：昨天的仓位吃今天的收益
    turnover = positions.diff().abs().fillna(positions.abs())
    cost = turnover * cost_bps / 1e4

    strat_ret = effective_pos * asset_ret - cost
    equity = (1 + strat_ret).cumprod()
    bench_equity = (1 + asset_ret).cumprod()

    n_trades = int((turnover > 1e-9).sum())
    return BacktestResult(
        strategy_name=strategy_name,
        ticker=ticker,
        equity=equity,
        benchmark_equity=bench_equity,
        positions=effective_pos,
        metrics=compute_metrics(strat_ret, effective_pos),
        benchmark_metrics=compute_metrics(asset_ret, pd.Series(1.0, index=df.index)),
        n_trades=n_trades,
        cost_bps=cost_bps,
    )
