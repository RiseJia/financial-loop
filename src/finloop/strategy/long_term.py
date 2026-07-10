"""长线投资决策支撑。

输出一个结构化的「长线视角」：
  1. 趋势状态：200 日线过滤 + 均线结构 + 距 52 周高低点位置
  2. 估值与基本面快照（PE/PEG/ROE/增速/现金流等，带解读）
  3. 综合检查清单：每项给出 通过/警示/不通过
  4. 操作参考：基于规则的分批/定投提示（非投资建议）

方法论详见 docs/long_term.md。
"""

from __future__ import annotations

import pandas as pd

from ..data.fundamentals import (get_fundamentals, format_value,
                                 validate_fundamentals)


def long_term_view(df: pd.DataFrame, ticker: str, with_fundamentals: bool = True) -> dict:
    """df 需为 enrich 后的日线（建议 period>=2y）。"""
    row = df.iloc[-1]
    price = float(row["close"])
    out: dict = {"ticker": ticker, "price": price}

    # --- 趋势状态 ---
    # 52 周高/低点用日内 high/low（市场惯例，如 "52-week high"），而非收盘价——
    # 收盘价口径会系统性低估真实高点/高估真实低点，使"距52周高点"偏乐观
    win = df.iloc[-252:] if len(df) >= 252 else df
    hi52 = float(win["high"].max()) if "high" in win else float(win["close"].max())
    lo52 = float(win["low"].min()) if "low" in win else float(win["close"].min())
    out["trend"] = {
        "above_200d": bool(price > row["sma200"]) if pd.notna(row["sma200"]) else None,
        "sma200_slope_up": bool(row["sma200"] > df["sma200"].iloc[-21]) if pd.notna(row["sma200"]) and len(df) > 21 else None,
        "bull_stack": bool(row["sma20"] > row["sma50"] > row["sma200"]) if pd.notna(row["sma200"]) else None,
        "off_52w_high_pct": round((price / hi52 - 1) * 100, 1),
        "off_52w_low_pct": round((price / lo52 - 1) * 100, 1),
        "ret_1y_pct": round((price / df["close"].iloc[-252] - 1) * 100, 1) if len(df) >= 252 else None,
    }

    # --- 基本面 ---
    fund = get_fundamentals(ticker) if with_fundamentals else {}
    out["name"] = fund.pop("_name", ticker) if fund else ticker
    out["sector"] = fund.pop("_sector", "") if fund else ""
    # 基本面一致性预警（一次性损益/口径错配等）——此前只在 screener 用，
    # 单票 analyze 从不调用，导致 Nittobo/KYEC 那类污染数据在个股报告里裸奔
    out["data_warnings"] = validate_fundamentals(
        {k: v["value"] for k, v in fund.items()})
    out["fundamentals"] = {
        k: {**v, "formatted": format_value(k, v["value"])} for k, v in fund.items()
    }

    # --- 检查清单 ---
    checks = []

    def check(name, ok, note):
        status = "✅" if ok else ("⚠️" if ok is None else "❌")
        checks.append({"item": name, "status": status, "note": note})

    t = out["trend"]
    check("价格位于 200 日均线上方（长线趋势过滤）", t["above_200d"],
          "200 日线是长线多头最重要的单一过滤条件；线下买入需要明确的左侧逻辑。")
    check("200 日均线本身向上", t["sma200_slope_up"],
          "均线斜率比价格位置更稳定地反映趋势方向。")
    check("均线多头排列（20>50>200）", t["bull_stack"],
          "结构健康度：排列整齐说明各周期资金方向一致。")
    check("距 52 周高点回撤小于 20%", t["off_52w_high_pct"] > -20,
          f"当前距 52 周高点 {t['off_52w_high_pct']}%；回撤超过 20% 进入技术性熊市区域。")

    f = out["fundamentals"]
    if "returnOnEquity" in f:
        check("ROE > 15%（盈利质量）", f["returnOnEquity"]["value"] > 0.15,
              f"当前 ROE {f['returnOnEquity']['formatted']}。")
    if "revenueGrowth" in f:
        check("营收正增长", f["revenueGrowth"]["value"] > 0,
              f"最近季度营收同比 {f['revenueGrowth']['formatted']}。")
    if "profitMargins" in f:
        check("净利率 > 10%", f["profitMargins"]["value"] > 0.10,
              f"当前净利率 {f['profitMargins']['formatted']}。")
    if "pegRatio" in f:
        peg = f["pegRatio"]["value"]
        # PEG<2 只对"正 PEG"有意义；负 PEG（亏损或增长为负）不是"便宜"，不能判 ✅
        check("PEG 在 (0, 2)（估值与增长匹配度）", 0 < peg < 2,
              f"当前 PEG {f['pegRatio']['formatted']}；"
              + ("负值=亏损或增长为负，PEG 失效。" if peg <= 0 else ">2 说明估值大幅透支增长。"))
    out["checklist"] = checks

    passed = sum(1 for c in checks if c["status"] == "✅")
    out["score"] = f"{passed}/{len(checks)}"

    # --- 操作参考 ---
    if t["above_200d"] and t["bull_stack"]:
        action = ("趋势与结构俱佳：符合「顺势持有/分批买入」的环境。常见执行方式是回调至 50 日均线"
                  "附近分批，而非一次性追入；以跌破 200 日线作为长线减仓的客观纪律。")
    elif t["above_200d"]:
        action = ("长线结构尚可但短中期纠缠：适合小额定投积累、等结构修复后再加大力度。")
    else:
        action = ("价格位于 200 日线下方：长线趋势过滤未通过。纪律化的做法是观察等待，"
                  "或仅以极小仓位定投；左侧抄底需要基本面强支撑且接受更大浮亏。")
    out["action_note"] = action
    return out
