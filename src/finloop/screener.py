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

打分模型（行业中性横截面 z-score）：
  demand_score   = mean[ z(营收增速), z(盈利增速), z(毛利率) ]        # 均值→±3 量纲
  valuation_score= mean[ z(forward PE), z(P/S) ]                       # 纯价格倍数，越高越贵
  gap = demand_score − valuation_score                                # 「需求 − 估值定价」缺口

金融正确性要点（2026-06 对抗审计后重构，见 docs/data_quality.md）：
  1. 行业中性化：z-score 在**可比组内**计算（sector/子链），因为 PE 与毛利率的
     结构性中枢由商业模式决定（软件毛利 80% vs EMS 10%、公用事业 PE 15 vs GPU 35）；
     跨行业裸横截面会让 gap 被行业成分而非真实错价主导。
  2. 增长不双计：估值端用 P/S 而非 PEG——PEG=PE/增长，会把已在需求端的增长
     再从估值端奖励一次。纯价格倍数（PE、P/S）与增长正交，gap 本身即"增长 vs 价格"。
  3. 亏损公司的估值兜底：负 PE 置 NaN 后由 P/S 施加估值惩罚，而非填中性 z=0
     让无盈利高增长泡沫逃过估值。
  4. 等权：demand/valuation 各取字段 z 的均值（而非求和），两侧同为 ±3 量纲，
     gap 不再结构性偏向字段更多的一侧。
  5. 离散度无偏：z 的中心与标准差只用**观测值**计算，缺失值事后填 0（中性），
     避免"先中位数填充再算 std"压缩离散度、放大少数真实值。
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
    ("priceToSalesTrailing12Months", "P/S", -1),
]
ALL_FIELDS = DEMAND_FIELDS + VALUATION_FIELDS
MIN_GROUP = 4  # 组内成员少于此值无法稳健 z-score，并入 "_other" 池


def _zscore_within_groups(s: pd.Series, group: pd.Series) -> pd.Series:
    """按 group 分组做组内 z-score。

    - 中心与标准差只用组内**观测值**（非 NaN）计算，缺失值最后填 0（中性），
      避免填充压缩离散度；
    - 成员 < MIN_GROUP 的小组合并进 "_other" 一起标准化（否则 1-2 人组恒为 0）；
    - 组内零方差 → 该组全 0。剪裁 ±3 抗离群。
    """
    g = group.reindex(s.index).fillna("_other").astype(str)
    counts = g.value_counts()
    g = g.where(g.map(counts) >= MIN_GROUP, "_other")

    out = pd.Series(0.0, index=s.index)
    for name, idx in g.groupby(g).groups.items():
        vals = s.loc[idx]
        obs = vals.dropna()
        if len(obs) < 2:
            continue
        med, std = obs.median(), obs.std(ddof=0)
        if not np.isfinite(std) or std < 1e-12:
            continue
        z = ((vals - med) / std).clip(-3, 3)
        out.loc[idx] = z.fillna(0.0)
    return out


def score_universe(fundamentals: dict[str, dict],
                   sectors: dict[str, str] | None = None) -> pd.DataFrame:
    """纯函数打分（可离线测试）。

    fundamentals: {ticker: {field: value}}，value 为原始数值。
    sectors:      {ticker: 可比组标签}（sector/子链）。None 时全体为单一组
                  （退化为跨行业裸横截面——仅用于同质小样本测试，生产必传）。
    返回按 gap 降序的 DataFrame。
    """
    raw = pd.DataFrame.from_dict(fundamentals, orient="index").reindex(fundamentals.keys())
    for field, _, _ in ALL_FIELDS:
        if field not in raw.columns:
            raw[field] = np.nan
    # 负/零 PE、P/S 无意义（亏损或数据错误）→ 置 NaN，由另一估值字段承担惩罚
    for field, _, _ in VALUATION_FIELDS:
        raw[field] = raw[field].where(raw[field] > 0)

    group = (pd.Series(sectors).reindex(raw.index) if sectors
             else pd.Series("_all", index=raw.index))

    def mean_z(fields):
        zs = [_zscore_within_groups(raw[f], group) for f, _, _ in fields]
        return sum(zs) / len(zs)  # 字段 z 的均值 → ±3 量纲，两侧同量纲可比

    demand = mean_z(DEMAND_FIELDS)        # 越高 = 需求越强
    valuation = mean_z(VALUATION_FIELDS)  # 越高 = 估值越贵（PE/PS 越大）

    out = pd.DataFrame({
        "demand_score": demand.round(2),
        "valuation_score": valuation.round(2),
        "gap": (demand - valuation).round(2),
    }, index=raw.index)
    out["coverage"] = raw[[f for f, _, _ in ALL_FIELDS]].notna().sum(axis=1).astype(str) + "/5"
    for field, label, _ in ALL_FIELDS:
        out[label] = raw[field]
    return out.sort_values("gap", ascending=False)


def run_screen(universe: str = "ai", tier: str = "upstream",
               archive: bool = True) -> pd.DataFrame:
    """联网模式：抓取 universe 基本面并打分。tier='all' 时全产业链一起比。

    archive=True 时把本次抓到的原始基本面快照存档到
    reports/research/data/（带抓取时间戳）——估值结论的半衰期以季度计，
    没有快照就无法回答"当时看到的是什么数据"。
    """
    from .data.fundamentals import get_fundamentals, validate_fundamentals

    cfg = load_universe(universe)
    if not cfg:
        raise FileNotFoundError(f"universe '{universe}' 不存在")
    tiers = cfg if tier == "all" else {tier: cfg.get(tier, {})}

    fundamentals, labels, sectors = {}, {}, {}
    for t, members in tiers.items():
        for ticker, label in (members or {}).items():
            raw = get_fundamentals(ticker)
            fundamentals[ticker] = {k: v["value"] for k, v in raw.items()
                                    if not k.startswith("_")}
            labels[ticker] = f"{label}（{t}）" if tier == "all" else label
            # 可比组用于行业中性化：优先 yfinance sector，缺失则退化到子链标签
            # （label 形如 "子链·名称"），保证跨行业不裸混算（审计 critical 修复）
            sub = label.split("·")[0] if "·" in label else t
            sectors[ticker] = raw.get("_sector", sub)
    if not fundamentals:
        return pd.DataFrame()

    warnings = {t: validate_fundamentals(vals) for t, vals in fundamentals.items()}
    scored = score_universe(fundamentals, sectors=sectors)
    scored.insert(2, "可比组", [sectors.get(t, "") for t in scored.index])
    scored.insert(0, "标签", [labels.get(t, "") for t in scored.index])
    scored.insert(1, "数据预警", ["；".join(warnings[t]) for t in scored.index])
    if archive:
        scored.attrs["snapshot_path"] = str(
            archive_snapshot(fundamentals, labels, warnings, universe, tier))
    return scored


def archive_snapshot(fundamentals: dict, labels: dict, warnings: dict,
                     universe: str, tier: str, out_dir=None):
    """把原始基本面快照写成带时间戳的 JSON，返回路径。

    目的：可回溯性。排名表是加工品，快照才是证据——人工核验、事后复盘、
    跨期对比（"上季度市场给它的 fPE 是多少"）都需要原始值。
    """
    import datetime as dt
    import json

    from .config import reports_dir

    if out_dir is None:
        out_dir = reports_dir() / "research" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "universe": universe,
        "tier": tier,
        "source": "yfinance（预期类字段有静默错误前科，采信前过数据预警+人工核验）",
        "labels": labels,
        "warnings": {t: w for t, w in warnings.items() if w},
        "fundamentals": fundamentals,
    }
    path = out_dir / f"{now.date().isoformat()}_screen_{universe}_{tier}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                    encoding="utf-8")
    return path


def format_screen(scored: pd.DataFrame, tier: str) -> str:
    pct = {"营收增速", "盈利增速", "毛利率"}
    lines = [
        f"# AI 产业链筛选：需求 vs 估值（tier={tier}）",
        "",
        "gap = 需求分 − 估值分（**行业内** z-score，纯价格倍数 PE/PS 不含增长）。",
        "**gap 高 ≠ 买入**，它生成人工深挖队列：读财报确认 backlog/交期/capex 指引。",
        "",
        "| 代码 | 标签 | 可比组 | gap | 需求分 | 估值分 | 覆盖 | 数据预警 | 营收增速 | 盈利增速 | 毛利率 | 前瞻PE | P/S |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ticker, row in scored.iterrows():
        cells = [ticker, str(row.get("标签", "")), str(row.get("可比组", "")),
                 f"{row['gap']:+.2f}",
                 f"{row['demand_score']:+.2f}", f"{row['valuation_score']:+.2f}",
                 row["coverage"], str(row.get("数据预警", "")) or "-"]
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
        "2. **数据预警列非空的标的，先人工核验 forward 字段再采信**——",
        "   缺数据显形于覆盖列，错数据显形于预警列（2026-06 实测：yfinance 对",
        "   日台中小盘的预期字段有静默错误，核验 3 个错 2 个）；",
        "3. 毛利率扩张是上游供需紧张最硬的免费证据（涨价能力）；",
        "4. 盈利增速远高于营收增速 = 经营杠杆释放期，常见于周期上行的设备/存储；",
        "5. 下一步人工验证：超大规模云厂 capex 指引（MSFT/GOOGL/AMZN/META 财报）",
        "   是全部上游需求的总闸门——上游的「需求」就是下游的「开支」。",
    ]
    return "\n".join(lines)
