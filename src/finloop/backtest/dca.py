"""定投（DCA）与分批入场的回测：把「要不要定投纳指」变成可以用历史回答的问题。

回答三个问题：
  1. 同一笔预算，一次性投入 / 线性分批 / 回撤加速分批 / 趋势门控分批，
     在给定起点、给定持有期后，终值差多少？
  2. 从历史大顶（2000-03、2007-10、2021-11）开始，各种节奏多久回本？
  3. 滚动所有起点（每月一个），按「起点状态」（距 52 周高点、是否在 200 日线上）分组，
     10 年终值倍数的分布是什么？——这才是「现在这种位置开始定投」该看的证据。

金融口径（每条都是常见的回测错误的反面）：
  - 未投出的现金**计息**（cash_rate_annual，按日复利近似）。不计息会系统性高估一次性投入
    相对分批的优势；2000 年现金收益 ~6%，2026 年 ~4%，差异不可忽略。
  - 终值倍数 = 期末总价值 / 总预算，所有模式**同一预算、同一起点、同一期末日**，
    是公平的总回报比较。不用「年化收益」掩盖投入节奏的差异；给出的 ann_on_budget
    只是倍数的几何年化，**不是 IRR**（IRR 会因为后投的钱占用时间短而显得更高）。
  - 价格用**全复权**收盘（含分红再投资）；只允许 adjustment == full 的数据源进入长周期
    定投研究（仓库 batch 模块同一纪律）。
  - 回本定义：预算**全部投出之后**，组合总价值首次 ≥ 总预算的日期。分批尚未投完时
    价值 ≥ 已投金额是常态，不算回本。
  - 回撤加速用的是**截至当日**的 252 日滚动高点（因果），不是事后知道的顶部。
  - 月度买点 = 每月同一自然日或其后第一个交易日；起点若非交易日顺延。

局限（写在这里，报告也会提）：
  - 现金利率为常数，不是历史 T-bill 序列（FRED 接入在 frontier）。
  - 不含交易成本与税；ETF 定投成本极低，省略对结论无实质影响。
  - QQQ 历史从 1999-03 开始，只含一次「科网泡沫」级别的 -83% 回撤——
    10 年滚动起点样本之间高度重叠，分布的尾部不要当成概率读。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MODES = ("lump_sum", "linear", "drawdown_tilt", "trend_gate")
DEFAULT_TILT = ((0.10, 2.0), (0.20, 3.0))   # (回撤阈值, 当月投入倍数)，取满足的最高档


def _naive(close: pd.Series) -> pd.Series:
    """数值化 + 去时区（yfinance 返回 tz-aware 索引，与 pd.Timestamp("2000-03-10") 不可比）。"""
    c = pd.to_numeric(close, errors="coerce").dropna()
    if getattr(c.index, "tz", None) is not None:
        c = c.copy()
        c.index = c.index.tz_localize(None)
    return c


def _month_marks(index: pd.DatetimeIndex, start: pd.Timestamp, months: int | None) -> list[pd.Timestamp]:
    """起点及其后每月同一日（顺延到首个交易日），最多 months 个买点；None = 直到数据末尾。"""
    marks = []
    k = -1
    while months is None or k + 1 < months:
        k += 1
        target = start + pd.DateOffset(months=k)
        pos = index.searchsorted(target)
        if pos >= len(index):
            break
        ts = index[pos]
        if not marks or ts != marks[-1]:
            marks.append(ts)
    return marks


def simulate_dca(close: pd.Series, start, total_budget: float = 1.0, months: int = 12,
                 mode: str = "linear", horizon_days: int | None = None,
                 cash_rate_annual: float = 0.0, tilt=DEFAULT_TILT,
                 gate_window: int = 200, gate_cap: float = 3.0) -> dict:
    """单起点、单模式的定投路径。

    返回 {path(总价值序列), invested(累计投入序列), units, multiple, ann_on_budget,
          fully_deployed(日期), breakeven(日期或 None), deploy_months, max_dd_post}
    close：全复权收盘（DatetimeIndex 升序）。
    """
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES}")
    c = _naive(close)
    if c.empty:
        raise ValueError("close 为空")
    start = pd.Timestamp(start)
    pos0 = c.index.searchsorted(start)
    if pos0 >= len(c):
        raise ValueError("起点晚于数据末尾")
    start = c.index[pos0]
    end_pos = len(c) - 1 if horizon_days is None else min(len(c) - 1, pos0 + horizon_days)
    truncated = bool(horizon_days is not None and pos0 + horizon_days > len(c) - 1)
    seg = c.iloc[pos0:end_pos + 1]
    idx = seg.index
    n = len(seg)

    base = total_budget / months
    # 因果的 252 日滚动高点（含当日）与 200 日均线，用全序列算再切片
    roll_hi = c.rolling(252, min_periods=1).max().reindex(idx)
    sma = c.rolling(gate_window, min_periods=gate_window).mean().reindex(idx)
    # 月度买点集合；drawdown_tilt / trend_gate 在预算未投完时**一直**按月尝试（不设上限——
    # 设 3×months 上限会让 2000 年顶部的趋势门控在熊市里「买点用完」而永远投不完）
    max_months = months if mode in ("lump_sum", "linear") else None
    marks_list = _month_marks(c.index, start, max_months)
    marks = set(marks_list)
    marks_left = {ts: len(marks_list) - k for k, ts in enumerate(marks_list)}   # 含本期的剩余买点数

    daily_r = (1 + cash_rate_annual) ** (1 / 252) - 1
    cash = float(total_budget)
    units = 0.0
    carry = 0.0                    # trend_gate 跳过的份额累积
    invested = np.zeros(n)
    path = np.zeros(n)
    deploy_dates: list[pd.Timestamp] = []
    inv_cum = 0.0

    for i, (ts, px) in enumerate(seg.items()):
        if i > 0:
            cash *= 1 + daily_r
        amt = 0.0
        if cash > 1e-12:
            if mode == "lump_sum":
                amt = cash if i == 0 else 0.0
            elif ts in marks:
                if mode == "linear":
                    amt = cash / marks_left[ts]      # 等分剩余现金（含已计利息），最后一期自然扫空
                elif mode == "drawdown_tilt":
                    dd = 1 - px / float(roll_hi.iloc[i]) if roll_hi.iloc[i] else 0.0
                    mult = 1.0
                    for thr, m in sorted(tilt):
                        if dd >= thr:
                            mult = m
                    amt = base * mult
                elif mode == "trend_gate":
                    above = pd.notna(sma.iloc[i]) and px > float(sma.iloc[i])
                    if above:
                        amt = min(base + carry, base * gate_cap)
                        carry = max(0.0, base + carry - amt)
                    else:
                        carry += base
                        amt = 0.0
            amt = min(amt, cash)
            if mode != "lump_sum" and amt > 0 and cash - amt < base * 0.5:
                amt = cash                      # 零头不足半个份额 → 并入本期，避免利息残留造成「永远未投完」
            if amt > 0:
                units += amt / px
                cash -= amt
                inv_cum += amt
                deploy_dates.append(ts)
        invested[i] = inv_cum
        path[i] = units * px + cash

    path_s = pd.Series(path, index=idx)
    inv_s = pd.Series(invested, index=idx)
    fully = deploy_dates[-1] if cash <= 1e-9 and deploy_dates else None
    span_months = round((fully - idx[0]).days / 30.44) if fully is not None else None
    breakeven = None
    if fully is not None:
        after = path_s.loc[fully:].iloc[1:]      # 不含投完当日（当日价值≈预算是平凡成立）
        hit = after[after >= total_budget * (1 - 1e-9)]
        breakeven = hit.index[0] if len(hit) else None
    years = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    multiple = float(path_s.iloc[-1] / total_budget)
    post = path_s.loc[fully:] if fully is not None else path_s
    peak = post.cummax()
    max_dd_post = float((post / peak - 1).min()) if len(post) else float("nan")
    return {
        "mode": mode, "start": idx[0], "end": idx[-1], "path": path_s, "invested": inv_s,
        "units": units, "cash_left": cash, "multiple": multiple,
        "ann_on_budget": multiple ** (1 / years) - 1 if multiple > 0 else -1.0,
        "fully_deployed": fully, "deploy_months": len(deploy_dates), "truncated": truncated,
        "deploy_span_months": span_months,
        "breakeven": breakeven,
        "breakeven_years": ((breakeven - idx[0]).days / 365.25) if breakeven is not None else None,
        "max_dd_post": max_dd_post,
    }


def start_state(close: pd.Series, start) -> dict:
    """起点状态标签：距 252 日高点 %、是否在 200 日线上（都只用起点之前的数据）。"""
    c = _naive(close)
    pos = c.index.searchsorted(pd.Timestamp(start))
    pos = min(pos, len(c) - 1)
    hist = c.iloc[max(0, pos - 251):pos + 1]
    px = float(c.iloc[pos])
    off_high = (px / float(hist.max()) - 1) * 100
    above = None
    if pos >= 199:
        above = px > float(c.iloc[pos - 199:pos + 1].mean())
    return {"off_high_pct": off_high, "above_200dma": above}


def rolling_dca_study(close: pd.Series, horizon_years: int = 10, months: int = 12,
                      modes=MODES, cash_rate_annual: float = 0.03,
                      step_months: int = 1) -> pd.DataFrame:
    """每 step_months 一个起点，所有模式跑同一持有期，返回长表。"""
    c = _naive(close)
    horizon_days = int(round(horizon_years * 252))
    rows = []
    start = c.index[0]
    # 起点从首根开始：前 200 根的起点没有趋势标签（分桶里单列「不足200日历史」），
    # 但不能丢——1999 年的高位起点正是最需要被看见的坏样本。
    last_ok = c.index[-1] - pd.DateOffset(years=horizon_years)
    while start <= last_ok:
        st = start_state(c, start)
        for mode in modes:
            r = simulate_dca(c, start, 1.0, months, mode, horizon_days, cash_rate_annual)
            rows.append({"start": r["start"], "mode": mode, "multiple": r["multiple"],
                         "ann_on_budget": r["ann_on_budget"], "max_dd_post": r["max_dd_post"],
                         "breakeven_years": r["breakeven_years"],
                         "off_high_pct": st["off_high_pct"], "above_200dma": st["above_200dma"]})
        start = start + pd.DateOffset(months=step_months)
    return pd.DataFrame(rows)


def summarize_study(df: pd.DataFrame) -> pd.DataFrame:
    """按模式汇总：倍数分位、相对一次性投入的胜率、回本年数中位数。"""
    if df.empty:
        return df
    lump = df[df["mode"] == "lump_sum"].set_index("start")["multiple"]
    out = []
    for mode, g in df.groupby("mode"):
        g = g.set_index("start")
        common = g.index.intersection(lump.index)
        beat = float((g.loc[common, "multiple"] > lump.loc[common]).mean()) if len(common) else float("nan")
        out.append({
            "mode": mode, "n": len(g),
            "p10": g["multiple"].quantile(0.10), "median": g["multiple"].median(),
            "p90": g["multiple"].quantile(0.90), "min": g["multiple"].min(),
            "beat_lump_rate": beat,
            "median_max_dd_post": g["max_dd_post"].median(),
            "median_breakeven_y": g["breakeven_years"].median(),
            "worst_breakeven_y": g["breakeven_years"].max(),
            "never_breakeven": int(g["breakeven_years"].isna().sum()),
        })
    return pd.DataFrame(out).set_index("mode").loc[[m for m in MODES if m in df["mode"].unique()]]


def bucket_by_start_state(df: pd.DataFrame, mode: str = "linear") -> pd.DataFrame:
    """按起点状态分桶：距高点 <5% / 5-15% / 15-30% / >30%，× 是否在 200 日线上。"""
    g = df[df["mode"] == mode].copy()
    if g.empty:
        return g
    bins = [-1e9, -30, -15, -5, 1e-9]
    labels = [">30%回撤", "15-30%回撤", "5-15%回撤", "<5%回撤(高位)"]
    g["bucket"] = pd.cut(g["off_high_pct"], bins=bins, labels=labels)
    g["trend"] = g["above_200dma"].map({True: "200日线上", False: "200日线下"}).fillna("不足200日历史")
    res = g.groupby(["bucket", "trend"], observed=True)["multiple"].agg(
        n="count", p10=lambda s: s.quantile(0.10), median="median", p90=lambda s: s.quantile(0.90))
    return res.reset_index()


def format_dca_report(ticker: str, close: pd.Series, cases: dict[str, str],
                      horizon_years: int = 10, months: int = 12,
                      cash_rate_annual: float = 0.03, source_note: str = "") -> str:
    """生成 Markdown：历史大顶案例 + 滚动起点分布 + 起点状态分桶。"""
    close = _naive(close)
    lines = [f"# {ticker} 定投/分批入场回测", "",
             f"数据：{close.index[0].date()} → {close.index[-1].date()}（{len(close)} 根，全复权）{source_note}",
             f"参数：预算分 {months} 个月；持有期 {horizon_years} 年；未投现金年化 {cash_rate_annual:.1%}；"
             f"回撤加速档位 {DEFAULT_TILT}；趋势门控 200 日线、跳过份额累积上限 3×。", ""]

    lines += ["## 一、从历史大顶开始（最不利起点）", "",
              "| 起点 | 模式 | 投完用时(月,自起点) | 回本(年) | 5年倍数 | 10年倍数 | 投完后最大回撤 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for label, start in cases.items():
        if pd.Timestamp(start) < close.index[0]:
            continue
        for mode in MODES:
            r5 = simulate_dca(close, start, 1.0, months, mode, 5 * 252, cash_rate_annual)
            r10 = simulate_dca(close, start, 1.0, months, mode, 10 * 252, cash_rate_annual)
            full = simulate_dca(close, start, 1.0, months, mode, None, cash_rate_annual)
            if full["fully_deployed"] is None:
                span, be = "未投完", f"未投完(剩{full['cash_left']:.0%})"
            else:
                span = str(full["deploy_span_months"])
                be = f"{full['breakeven_years']:.1f}" if full["breakeven_years"] is not None else "未回本"
            m5 = f"{r5['multiple']:.2f}" + ("*" if r5["truncated"] else "")
            m10 = f"{r10['multiple']:.2f}" + ("*" if r10["truncated"] else "")
            lines.append(f"| {label} {pd.Timestamp(start).date()} | {mode} | {span} | {be} "
                         f"| {m5} | {m10} | {r10['max_dd_post']:.0%} |")
    lines += ["", "> 回本 = 预算全部投出后组合总价值首次回到总预算。倍数含未投现金的利息。"
              "`*` = 持有期未满，数字为截至数据末尾的倍数。", ""]

    study = rolling_dca_study(close, horizon_years, months, MODES, cash_rate_annual)
    summ = summarize_study(study)
    lines += [f"## 二、滚动起点（每月一个，共 {study['start'].nunique()} 个起点，持有 {horizon_years} 年）", "",
              "| 模式 | 10%分位 | 中位 | 90%分位 | 最差 | 跑赢一次性投入的比例 | 投完后最大回撤(中位) | 回本年数(中位/最差) | 从未回本 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, r in summ.iterrows():
        be = f"{r['median_breakeven_y']:.1f} / {r['worst_breakeven_y']:.1f}" if pd.notna(r["median_breakeven_y"]) else "—"
        lines.append(f"| {mode} | {r['p10']:.2f} | {r['median']:.2f} | {r['p90']:.2f} | {r['min']:.2f} "
                     f"| {r['beat_lump_rate']:.0%} | {r['median_max_dd_post']:.0%} | {be} | {int(r['never_breakeven'])} |")
    lines += ["", "> 倍数 = 期末价值/总预算。一次性投入的「跑赢比例」恒为 0（与自己比）。"
              "起点样本高度重叠（同一轮牛熊被计入几十个起点），分位数是**描述**不是概率。", ""]

    lines += [f"## 三、按起点状态分桶（linear 模式，{horizon_years} 年倍数）", "",
              "| 起点距 252 日高点 | 起点趋势 | 样本 | 10%分位 | 中位 | 90%分位 |", "|---|---|---:|---:|---:|---:|"]
    for _, r in bucket_by_start_state(study, "linear").iterrows():
        lines.append(f"| {r['bucket']} | {r['trend']} | {int(r['n'])} | {r['p10']:.2f} | {r['median']:.2f} | {r['p90']:.2f} |")
    cur = start_state(close, close.index[-1])
    lines += ["", f"> 当前起点状态：距 252 日高点 {cur['off_high_pct']:+.1f}%，"
              f"{'在' if cur['above_200dma'] else '不在'} 200 日线上方——对照上表读「高位起点」那一行。", ""]
    return "\n".join(lines)
