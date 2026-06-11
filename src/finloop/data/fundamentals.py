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


def format_value(key: str, value) -> str:
    """把原始数值格式化为可读文本。"""
    if key in ("profitMargins", "grossMargins", "returnOnEquity",
               "revenueGrowth", "earningsGrowth", "dividendYield"):
        return f"{value * 100:.1f}%"
    if key in ("marketCap", "freeCashflow"):
        for unit, div in (("万亿", 1e12), ("亿", 1e8)):
            if abs(value) >= div:
                return f"{value / div:.2f} {unit}美元"
        return f"{value:,.0f} 美元"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
