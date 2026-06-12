"""基本面快照：从 yfinance 的 info 字段提取长线投资关心的核心指标。"""

from __future__ import annotations

import yfinance as yf

# 字段 -> (中文名, 解释)
FIELDS = {
    "trailingPE": ("市盈率 TTM", "过去12个月利润对应的估值倍数，越高代表市场定价的增长预期越高"),
    "forwardPE": ("预期市盈率", "基于分析师下一年盈利预测的估值，低于 TTM PE 通常意味着盈利预期增长"),
    "pegRatio": ("PEG", "PE / 盈利增速，约 1 视为增长与估值匹配，<1 可能低估，>2 偏贵"),
    "priceToBook": ("市净率 PB", "股价相对每股净资产，适合重资产/金融类公司比较"),
    "profitMargins": ("净利率", "净利润 / 营收，衡量盈利质量"),
    "grossMargins": ("毛利率", "反映产品定价能力与护城河"),
    "returnOnEquity": ("ROE", "股东权益回报率，长期 >15% 通常是优质企业的特征"),
    "revenueGrowth": ("营收增速 YoY", "最近季度营收同比增长"),
    "earningsGrowth": ("盈利增速 YoY", "最近季度盈利同比增长"),
    "debtToEquity": ("负债权益比", "杠杆水平，行业间差异大，需横向比较"),
    "freeCashflow": ("自由现金流", "公司真实可支配现金，长线投资的核心关注点"),
    "dividendYield": ("股息率", "年化分红 / 股价"),
    "marketCap": ("市值", "公司规模"),
    "totalRevenue": ("营收 TTM", "过去12个月总营收，用于交叉校验估值字段的合理性"),
    "beta": ("Beta", "相对大盘的波动放大倍数，>1 比大盘波动更大"),
}


def get_fundamentals(ticker: str) -> dict:
    """返回 {字段: {label, value, note}}；拿不到的字段跳过。"""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    out = {}
    for key, (label, note) in FIELDS.items():
        val = info.get(key)
        if val is None:
            continue
        out[key] = {"label": label, "value": val, "note": note}
    if info.get("longName"):
        out["_name"] = info["longName"]
    if info.get("sector"):
        out["_sector"] = info["sector"]
    return out


# ---------------------------------------------------------------- 一致性校验
#
# 背景（2026-06 实测教训）：yfinance 的预期类字段存在**静默错误**——
# 错的数据长得和对的一模一样，coverage 列只能暴露"缺"暴露不了"错"。
# 实锤案例：3110.T 的 forwardPE 给 58.5（共识实为 ~18）；MU 的 forwardPE 8.2
# 用的是两年外峰值盈利。本校验把当时人工抓错的两个方法自动化：
#   ① forward PE 与 trailing PE 偏离过大 → 预期口径过时/远期/将崩，先核验；
#   ② 市值÷forwardPE 倒推的隐含净利 对照 营收 TTM → 隐含净利率荒谬即必有一错。
# 校验只产生警告（研究纪律是"先核验再采信"），不修改、不丢弃数据。

PE_RATIO_BAND = (1 / 3, 3.0)     # fPE/tPE 容忍区间
IMPLIED_MARGIN_SOFT = 0.60       # 隐含净利率 > 60% 提示核验
IMPLIED_MARGIN_HARD = 1.00       # > 100%（净利超营收）必有一错


def validate_fundamentals(vals: dict) -> list[str]:
    """对单个标的的基本面快照做交叉一致性检查，返回警告列表（可为空）。

    vals: {field: value} 原始数值（get_fundamentals 输出再取 value）。
    缺字段时跳过相应检查——本函数管"错"，"缺"由 coverage 列负责。
    """
    warns: list[str] = []
    fpe, tpe = vals.get("forwardPE"), vals.get("trailingPE")
    if fpe and tpe and fpe > 0 and tpe > 0:
        ratio = fpe / tpe
        if ratio > PE_RATIO_BAND[1]:
            warns.append(f"fPE/tPE={ratio:.1f}x：前瞻盈利预估远低于已实现——疑似过时共识或盈利将崩")
        elif ratio < PE_RATIO_BAND[0]:
            warns.append(f"fPE/tPE={ratio:.2f}x：前瞻盈利为已实现的 {1/ratio:.1f} 倍——疑似远期财年口径或周期顶外推")
    mc, rev = vals.get("marketCap"), vals.get("totalRevenue")
    if mc and rev and fpe and fpe > 0 and mc > 0 and rev > 0:
        implied_margin = (mc / fpe) / rev
        if implied_margin > IMPLIED_MARGIN_HARD:
            warns.append(f"隐含净利率 {implied_margin:.0%}＞100%：fPE 与市值/营收必有一错")
        elif implied_margin > IMPLIED_MARGIN_SOFT:
            warns.append(f"隐含净利率 {implied_margin:.0%}：超 60%，先核验 fPE 口径")
    return warns


def format_value(key: str, value) -> str:
    """把原始数值格式化为可读文本。"""
    if key in ("profitMargins", "grossMargins", "returnOnEquity",
               "revenueGrowth", "earningsGrowth", "dividendYield"):
        return f"{value * 100:.1f}%"
    if key in ("marketCap", "freeCashflow", "totalRevenue"):
        for unit, div in (("万亿", 1e12), ("亿", 1e8)):
            if abs(value) >= div:
                return f"{value / div:.2f} {unit}美元"
        return f"{value:,.0f} 美元"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
