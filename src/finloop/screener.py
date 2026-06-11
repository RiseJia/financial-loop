"""产业链筛选器：「需求 vs 估值」横截面打分。

研究命题（用户主线）：在 AI 上游寻找需求远大于估值定价的标的。

================================================================
诚实的方法论声明（量化标准要求先讲清数据的边界）
================================================================
「需求」的一手证据是：下游资本开支指引（超大规模云厂商 capex）、
订单积压(backlog)、交期(lead time)、分析师预期上修(estimate revisions)。
这些要么在财报电话会里（文本），要么在付费数据源里（IBES/Visible Alpha）。

免费数据（yfinance）只能拿到二手代理：
  需求代理：营收增速、盈利增速、毛利率（涨价能力=供需紧张的直接证据）
  估值：forward PE、PEG、P/S
  确认信号：相对强度（价格是聪明钱投票的结果）

因此本筛选器的定位是【漏斗的第一层】：用同一把尺子横向排序，
把值得人工深挖（读财报、查交期、对 capex 指引）的名单从 40 个缩到 5 个。
它不是答案，是研究队列的生成器。
================================================================

打分模型（横截面 z-score 合成）：
  demand_score   = z(营收增速) + z(盈利增速) + z(毛利率)
  valuation_pen  = z(forward PE) + z(PEG)        # 越高越贵
  gap = demand_score − valuation_pen             # 「需求 − 估值定价」缺口
  缺字段时按该字段横截面中位数填充（不奖励也不惩罚缺数据者，但标记覆盖率）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_universe

# (字段, 中文名, 方向)  方向 +1 = 越大越代表需求强；-1 = 越大越贵
DEMAND_FIELDS = [
    ("revenueGrowth", "营收增速", 1),
    ("earningsGrowth", "盈利增速", 1),
    ("grossMargins", "毛利率", 1),
]
VALUATION_FIELDS = [
    ("forwardPE", "前瞻PE", -1),
    ("pegRatio", "PEG", -1),
]
ALL_FIELDS = DEMAND_FIELDS + VALUATION_FIELDS


def _zscore(s: pd.Series) -> pd.Series:
    """横截面 z-score；缺失值填中位数（z=0 中性）；剪裁 ±3 抗离群。"""
    filled = s.fillna(s.median())
    std = filled.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((filled - filled.median()) / std).clip(-3, 3)


def score_universe(fundamentals: dict[str, dict]) -> pd.DataFrame:
    """纯函数打分（可离线测试）。

    fundamentals: {ticker: {field: value}}，value 为原始数值。
    返回按 gap 降序的 DataFrame。
    """
    # reindex 保留全字段缺失的标的（from_dict 会丢空行），它们应显示 0/5 覆盖而非消失
    raw = pd.DataFrame.from_dict(fundamentals, orient="index").reindex(fundamentals.keys())
    for field, _, _ in ALL_FIELDS:
        if field not in raw.columns:
            raw[field] = np.nan
    # 负 PE/PEG = 亏损公司，不是「便宜」——置 NaN 落到中性，而非被 z-score 当成最低估值
    for field, _, _ in VALUATION_FIELDS:
        raw[field] = raw[field].where(raw[field] > 0)

    demand = sum(_zscore(raw[f]) for f, _, _ in DEMAND_FIELDS)
    valuation = sum(_zscore(raw[f]) for f, _, _ in VALUATION_FIELDS)

    out = pd.DataFrame({
        "demand_score": demand.round(2),
        "valuation_score": valuation.round(2),
        "gap": (demand - valuation).round(2),
    }, index=raw.index)
    # 数据覆盖率：5 个字段里有几个是真实值
    out["coverage"] = raw[[f for f, _, _ in ALL_FIELDS]].notna().sum(axis=1).astype(str) + "/5"
    for field, label, _ in ALL_FIELDS:
        out[label] = raw[field]
    return out.sort_values("gap", ascending=False)


def run_screen(universe: str = "ai", tier: str = "upstream") -> pd.DataFrame:
    """联网模式：抓取 universe 基本面并打分。tier='all' 时全产业链一起比。"""
    from .data.fundamentals import get_fundamentals

    cfg = load_universe(universe)
    if not cfg:
        raise FileNotFoundError(f"universe '{universe}' 不存在")
    tiers = cfg if tier == "all" else {tier: cfg.get(tier, {})}

    fundamentals, labels = {}, {}
    for t, members in tiers.items():
        for ticker, label in (members or {}).items():
            raw = get_fundamentals(ticker)
            fundamentals[ticker] = {k: v["value"] for k, v in raw.items()
                                    if not k.startswith("_")}
            labels[ticker] = f"{label}（{t}）" if tier == "all" else label
    if not fundamentals:
        return pd.DataFrame()

    scored = score_universe(fundamentals)
    scored.insert(0, "标签", [labels.get(t, "") for t in scored.index])
    return scored


def format_screen(scored: pd.DataFrame, tier: str) -> str:
    pct = {"营收增速", "盈利增速", "毛利率"}
    lines = [
        f"# AI 产业链筛选：需求 vs 估值（tier={tier}）",
        "",
        "gap = 需求得分 − 估值得分（横截面 z-score）。**gap 高 ≠ 买入**，",
        "它生成的是人工深挖队列：去读财报确认 backlog/交期/下游 capex 指引。",
        "",
        "| 代码 | 标签 | gap | 需求分 | 估值分 | 覆盖 | 营收增速 | 盈利增速 | 毛利率 | 前瞻PE | PEG |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ticker, row in scored.iterrows():
        cells = [ticker, str(row.get("标签", "")), f"{row['gap']:+.2f}",
                 f"{row['demand_score']:+.2f}", f"{row['valuation_score']:+.2f}",
                 row["coverage"]]
        for _, label, _ in ALL_FIELDS:
            v = row.get(label)
            if pd.isna(v):
                cells.append("-")
            elif label in pct:
                cells.append(f"{v * 100:.0f}%")
            else:
                cells.append(f"{v:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "解读纪律：",
        "1. 高 gap + 低覆盖率（缺数据）= 先补数据再下结论；",
        "2. 毛利率扩张是上游供需紧张最硬的免费证据（涨价能力）；",
        "3. 盈利增速远高于营收增速 = 经营杠杆释放期，常见于周期上行的设备/存储；",
        "4. 下一步人工验证：超大规模云厂 capex 指引（MSFT/GOOGL/AMZN/META 财报）",
        "   是全部上游需求的总闸门——上游的「需求」就是下游的「开支」。",
    ]
    return "\n".join(lines)
