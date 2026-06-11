"""事件研究（event study）：验证信号本身，而不是验证完整策略。

与回测的分工：
  回测       检验「一套完整的进出场规则」端到端的净值表现
  事件研究   检验「单一信号」是否携带信息：信号发生后 N 天的收益
            分布，是否系统性地不同于随便挑一天？

方法：
  1. 在全部历史上扫描某类信号的每次发生时点 t_i
  2. 计算每个时点的前向收益 fwd_i = close[t_i + N] / close[t_i] - 1
  3. 与「无条件基线」（同周期所有交易日的 N 日前向收益）比较
  4. 输出：均值差、命中率、t 统计量近似

t 统计量读法：|t| > 2 约对应 95% 置信度「均值差不是碰巧」。
但务必记住两个陷阱（详见 docs/backtesting.md）：
  - 样本量：一只票几年历史里黄金交叉可能只有 3 次，n<10 的结论无意义；
  - 多重假设：同时检验 15 种信号，按 5% 显著水平，纯随机也会有约 1 种
    «显著»。跨多只票稳定显著的信号才值得相信。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..signals import detect_momentum_switches, detect_turning_points

HORIZONS = (5, 20, 60)


def _forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1


def event_study(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> dict:
    """对 enrich 后日线的全部信号类型做事件研究。

    返回 {signal_type: {direction, n, horizons: {N: {...统计量}}}}，
    以及 '_baseline'：无条件前向收益基线。
    """
    # 全历史扫描：lookback 取数据长度，cap=None 取每类信号的全部样本
    # （默认 cap=10 是日报防噪声用的，统计上会造成近因偏差）
    events = detect_turning_points(df, lookback=len(df), cap=None) + \
        detect_momentum_switches(df, lookback=len(df))

    close = df["close"]
    date_to_pos = {d.date().isoformat(): i for i, d in enumerate(df.index)}
    fwd = {h: _forward_returns(close, h) for h in horizons}

    baseline = {
        h: {"mean": float(fwd[h].mean()), "median": float(fwd[h].median()),
            "std": float(fwd[h].std()), "pos_rate": float((fwd[h] > 0).mean())}
        for h in horizons
    }

    out: dict = {"_baseline": baseline}
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)

    for sig_type, evs in by_type.items():
        positions = [date_to_pos[e["date"]] for e in evs if e["date"] in date_to_pos]
        entry = {"direction": evs[0]["direction"], "n": len(positions), "horizons": {}}
        for h in horizons:
            rets = np.array([fwd[h].iloc[p] for p in positions
                             if p + h < len(df) and not np.isnan(fwd[h].iloc[p])])
            if len(rets) == 0:
                continue
            base = baseline[h]
            excess = rets.mean() - base["mean"]
            # 均值差的 t 统计量近似（用事件样本的标准误）
            se = rets.std(ddof=1) / np.sqrt(len(rets)) if len(rets) > 1 else np.nan
            entry["horizons"][h] = {
                "n": int(len(rets)),
                "mean": float(rets.mean()),
                "median": float(np.median(rets)),
                "hit_rate": float((rets > 0).mean()),
                "baseline_mean": base["mean"],
                "excess": float(excess),
                "t_stat": float(excess / se) if se and se > 1e-12 else float("nan"),
            }
        # 背离类信号日期记在序列末根，前向收益必为 NaN → horizons 为空，剔除
        if entry["horizons"]:
            out[sig_type] = entry
    return out


def format_event_study(result: dict, ticker: str = "") -> str:
    lines = [f"# 信号事件研究{'：' + ticker if ticker else ''}", ""]
    base = result["_baseline"]
    lines += ["无条件基线（随便挑一天的前向收益均值）：",
              "  " + " | ".join(f"{h}日 {base[h]['mean']:+.2%}" for h in sorted(base)), ""]
    lines += ["| 信号 | 方向 | 次数 | 周期 | 信号后均值 | 基线 | 超额 | 命中率 | t值 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for sig_type, entry in result.items():
        if sig_type == "_baseline":
            continue
        for h, s in entry["horizons"].items():
            t = s.get("t_stat")
            # NaN（n=1 等）不渲染数值更不标显著（nan 的任何比较都为 False，曾被误标 ✱）
            if t is None or not np.isfinite(t):
                t_cell = "-"
            else:
                t_cell = f"{t:.1f}" + (" ✱" if abs(t) > 2 else "")
            lines.append(
                f"| {sig_type} | {'🟢' if entry['direction']=='bullish' else '🔴'} "
                f"| {s['n']} | {h}日 | {s['mean']:+.2%} | {s['baseline_mean']:+.2%} "
                f"| {s['excess']:+.2%} | {s['hit_rate']:.0%} | {t_cell} |"
            )
    lines += ["", "✱ = |t|>2（约 95% 置信）。三个必须记住的折扣因子：",
              "1. 单票样本少 + 多重假设检验：孤立的显著不可信，跨多票稳定显著才算数；",
              "2. 事件窗口重叠：相邻信号间隔常小于前向周期（20/60日），样本序列相关，t 值被系统性高估；",
              "3. 信号扎堆出现在同一段行情时，n 次事件 ≈ 1 次独立观察。"]
    return "\n".join(lines)
