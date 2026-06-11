"""指标解释引擎。

两个能力：
1. explain_snapshot(df)：对 enrich 后 DataFrame 的最新一行，逐指标生成中文实时解读。
2. LESSONS / get_lesson(name)：教学式详解（原理、公式、参数、解读、误区），
   供 `finloop explain rsi` 这类命令使用。
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------- 实时解读

def explain_snapshot(df: pd.DataFrame) -> list[dict]:
    """返回 [{indicator, value, reading, detail}]，按指标逐条解读最新读数。"""
    row = df.iloc[-1]
    price = row["close"]
    items: list[dict] = []

    def add(indicator: str, value: str, reading: str, detail: str):
        items.append({"indicator": indicator, "value": value,
                      "reading": reading, "detail": detail})

    # --- 均线结构 ---
    if pd.notna(row.get("sma200")):
        above50 = price > row["sma50"]
        above200 = price > row["sma200"]
        stack = row["sma20"] > row["sma50"] > row["sma200"]
        if stack and above200:
            reading, detail = "多头排列", "20>50>200 日均线依次向上排列，且价格站在 200 日均线上方，是典型的上升趋势结构，长线持有的顺风环境。"
        elif not above200 and row["sma20"] < row["sma50"] < row["sma200"]:
            reading, detail = "空头排列", "均线依次向下排列且价格在 200 日线下方，下降趋势结构，长线买入需要格外谨慎，等待趋势修复信号。"
        elif above200:
            reading, detail = "趋势偏多但有纠缠", "价格在 200 日线上方但中短期均线相互纠缠，趋势在休整或酝酿方向，关注能否重新走出多头排列。"
        else:
            reading, detail = "趋势偏弱", "价格在 200 日线下方，长期趋势承压；短期均线若率先拐头向上，是趋势修复的早期迹象。"
        add("均线结构 (20/50/200日)",
            f"价格 {price:.2f} | SMA50 {row['sma50']:.2f} | SMA200 {row['sma200']:.2f}",
            reading, detail)

    # --- MACD ---
    if pd.notna(row.get("macd")):
        hist = row["macd_hist"]
        prev_hist = df["macd_hist"].iloc[-2] if len(df) > 1 else hist
        if row["macd"] > row["macd_signal"]:
            trend = "金叉状态（MACD 在信号线上方）"
            momentum = "且柱体在放大，多头动能在加速" if hist > prev_hist else "但柱体在收缩，多头动能正在衰减——这是动量见顶的早期预警"
        else:
            trend = "死叉状态（MACD 在信号线下方）"
            momentum = "且柱体在加深，空头动能在加速" if hist < prev_hist else "但柱体在收窄，空头动能正在衰减——下跌可能进入尾声"
        zero = "零轴上方（多头主导区）" if row["macd"] > 0 else "零轴下方（空头主导区）"
        add("MACD (12,26,9)",
            f"MACD {row['macd']:.2f} | 信号线 {row['macd_signal']:.2f} | 柱 {hist:.2f}",
            trend, f"当前处于{zero}，{momentum}。")

    # --- RSI ---
    if pd.notna(row.get("rsi14")):
        r = row["rsi14"]
        if r >= 70:
            reading, detail = "超买", "RSI≥70。强趋势中 RSI 可长期钝化于超买区，不能机械做空；但若伴随顶背离（价格新高、RSI 不新高）则警惕回调。"
        elif r <= 30:
            reading, detail = "超卖", "RSI≤30。恐慌性下跌后的超卖往往酝酿反弹，但下跌趋势中超卖同样会钝化，需等待 RSI 回升穿过 30 确认。"
        elif r >= 55:
            reading, detail = "偏强", "RSI 处于 55-70 的多头区间，上涨动能健康。回调中 RSI 若守住 40-50 一带，通常是强势股的特征。"
        elif r <= 45:
            reading, detail = "偏弱", "RSI 处于 30-45 的空头区间，反弹乏力。若 RSI 持续无法突破 60，说明每次反弹都被卖压消化。"
        else:
            reading, detail = "中性", "RSI 在 45-55 之间，多空力量均衡，方向待选择。"
        add("RSI (14)", f"{r:.1f}", reading, detail)

    # --- ADX ---
    if pd.notna(row.get("adx")):
        a = row["adx"]
        direction = "上行方向占优 (+DI > -DI)" if row["plus_di"] > row["minus_di"] else "下行方向占优 (-DI > +DI)"
        if a >= 40:
            reading, detail = "极强趋势", f"ADX≥40，趋势非常强劲，{direction}。强趋势中逆势抄底/摸顶的胜率很低，顺势策略占优。"
        elif a >= 25:
            reading, detail = "有效趋势", f"ADX 在 25-40，存在可交易的趋势，{direction}。均线、MACD 等趋势类信号在此环境下可信度较高。"
        else:
            reading, detail = "震荡市", f"ADX<25（当前{direction}），方向性不足。震荡市中趋势信号容易反复打脸，RSI/KD 这类摆动指标的高抛低吸更适用。"
        add("ADX (14)", f"ADX {a:.1f} | +DI {row['plus_di']:.1f} | -DI {row['minus_di']:.1f}",
            reading, detail)

    # --- 布林带 ---
    if pd.notna(row.get("bb_pctb")):
        b = row["bb_pctb"]
        width = row["bb_width"]
        squeeze = width < df["bb_width"].rolling(120).quantile(0.15).iloc[-1] if len(df) >= 120 else False
        if b > 1:
            reading, detail = "突破上轨", "价格收在布林上轨之外，短期极强。若伴随放量是突破行情的特征；缩量则可能是冲高乏力。"
        elif b < 0:
            reading, detail = "跌破下轨", "价格收在布林下轨之外，短期极弱或恐慌，统计上存在均值回归的反弹倾向，但接刀需要等企稳信号。"
        elif b > 0.8:
            reading, detail = "贴近上轨", "价格运行在带宽上沿，多头掌控；强趋势中沿上轨爬升（band walking）是健康表现。"
        elif b < 0.2:
            reading, detail = "贴近下轨", "价格运行在带宽下沿，空头掌控。"
        else:
            reading, detail = "带内中性", "价格在布林带中部运行，无极端状态。"
        if squeeze:
            detail += " ⚠️ 当前带宽处于近半年最窄的 15% 分位（布林挤压），波动率极度压缩，往往预示着即将选择方向的大行情。"
        add("布林带 (20,2)", f"%B {b:.2f} | 带宽 {width:.3f}", reading, detail)

    # --- 量能 ---
    if pd.notna(row.get("vol_ratio")):
        vr = row["vol_ratio"]
        up_day = price >= df["close"].iloc[-2] if len(df) > 1 else True
        if vr >= 2:
            reading = "显著放量"
            detail = ("放量上涨，资金积极介入，价格变化的可信度高。" if up_day
                      else "放量下跌，主动卖压沉重，需警惕趋势性下跌的开端。")
        elif vr <= 0.5:
            reading, detail = "明显缩量", "成交极度萎缩，市场参与度低；缩量回调通常杀伤力有限，但缩量上涨的持续性也存疑。"
        else:
            reading, detail = "量能正常", "成交量处于近期均值附近，无异常资金行为。"
        add("量比 (vs 20日均量)", f"{vr:.2f}x", reading, detail)

    # --- ATR 波动率 ---
    if pd.notna(row.get("atr14")):
        atr_pct = row["atr14"] / price * 100
        add("ATR (14)", f"{row['atr14']:.2f}（{atr_pct:.1f}% 价格）",
            "波动幅度参考",
            f"该标的当前日均真实波幅约为价格的 {atr_pct:.1f}%。常用 2×ATR（约 {2*atr_pct:.1f}%）作为止损距离参考；波动越大，单笔仓位应越小。")

    return items


# ---------------------------------------------------------------- 教学详解

LESSONS: dict[str, dict] = {
    "sma": {
        "name": "SMA 简单移动平均线",
        "category": "趋势",
        "formula": "SMA(N) = 最近 N 根收盘价之和 / N",
        "principle": (
            "移动平均把噪声滤掉，留下趋势。N 越大越平滑、越滞后。美股语境下：\n"
            "  20 日 ≈ 一个月的交易日，刻画短期趋势；\n"
            "  50 日 ≈ 一个季度，机构常用的中期趋势线；\n"
            "  200 日 ≈ 一年，长线牛熊分界线，价格在其上方为牛市结构。"
        ),
        "usage": (
            "1) 趋势过滤：只在价格位于 200 日线上方时做多，是最经典的长线规则之一；\n"
            "2) 多头/空头排列：短中长均线依次向上(向下)排列代表趋势健康(恶化)；\n"
            "3) 动态支撑阻力：上升趋势中回踩 50 日线常是加仓点。"
        ),
        "pitfalls": "震荡市中均线信号反复无效（whipsaw）；均线本质是滞后指标，只能确认趋势、不能预测拐点。",
    },
    "ema": {
        "name": "EMA 指数移动平均线",
        "category": "趋势",
        "formula": "EMA_t = α × 收盘价_t + (1-α) × EMA_{t-1}，α = 2/(N+1)",
        "principle": "对越近的价格赋予越高权重，对新信息反应比 SMA 快，代价是更多噪声。日内交易常用 9/21 EMA 组合。",
        "usage": "短线/日内更偏好 EMA；9EMA 上穿 21EMA 常作为日内趋势启动的确认。",
        "pitfalls": "更快也意味着更多假信号，需配合量能或更高周期过滤。",
    },
    "macd": {
        "name": "MACD 指数平滑异同移动平均线",
        "category": "趋势 + 动量",
        "formula": (
            "MACD 线 = EMA(12) - EMA(26)\n"
            "信号线  = MACD 线的 9 日 EMA\n"
            "柱状图  = MACD 线 - 信号线"
        ),
        "principle": (
            "快慢两条 EMA 的差值衡量「短期动能相对长期趋势的强弱」。\n"
            "柱状图是动量的二阶导：柱体放大=动能加速，柱体收缩=动能衰减。\n"
            "零轴是多空分界：零轴上方的金叉远比零轴下方的金叉可靠。"
        ),
        "usage": (
            "1) 金叉/死叉：MACD 上穿/下穿信号线，基础的趋势触发信号；\n"
            "2) 柱体拐点：柱体由放大转收缩往往领先于金叉死叉，是更早的动量切换证据；\n"
            "3) 背离：价格新高而 MACD 高点降低（顶背离），是趋势衰竭的重要预警，反之为底背离。"
        ),
        "pitfalls": "震荡市中金叉死叉频繁且无效；背离可以持续很久才兑现，背离本身不是入场信号，而是「降低仓位、提高警惕」的信号。",
    },
    "rsi": {
        "name": "RSI 相对强弱指数",
        "category": "动量",
        "formula": "RSI = 100 - 100/(1+RS)，RS = N 日平均涨幅 / N 日平均跌幅（Wilder 平滑，N 默认 14）",
        "principle": (
            "把涨跌力量之比压缩到 0-100。70/30 是传统超买/超卖阈值。\n"
            "更重要的是「区间」概念：牛市中 RSI 往往在 40-90 运行（40-50 即是回调支撑），\n"
            "熊市中往往在 10-60 运行（55-60 即是反弹阻力）。RSI 运行区间的迁移本身就是趋势转换的证据。"
        ),
        "usage": (
            "1) 极端反转：震荡市中 <30 买、>70 卖的均值回归打法；\n"
            "2) 区间判定：观察 RSI 高低点的区间属于牛市型还是熊市型；\n"
            "3) 背离：价格创新高/新低而 RSI 不再跟随，警示动量衰竭。"
        ),
        "pitfalls": "强趋势中 RSI 长期钝化在极端区，机械按 70/30 反向操作会在大牛股上反复止损。先判断市场状态（ADX），再决定 RSI 用法。",
    },
    "stochastic": {
        "name": "随机指标 KD (Stochastic)",
        "category": "动量",
        "formula": "%K = (收盘 - N日最低)/(N日最高 - N日最低) × 100；%D = %K 的 3 日均线",
        "principle": "刻画收盘价在近期高低区间中的位置。比 RSI 更敏感，适合震荡市的高抛低吸节奏。",
        "usage": "K 在 20 以下上穿 D 是超卖反弹触发；80 以上下穿 D 是超买回落触发。",
        "pitfalls": "在趋势市中信号噪声极大，通常只作为次级确认而非主信号。",
    },
    "roc": {
        "name": "ROC 变动率",
        "category": "动量",
        "formula": "ROC(N) = (今收 / N 日前收 - 1) × 100%",
        "principle": "最朴素的动量定义。学术上的「动量效应」（过去 3-12 个月强者恒强）正是建立在这个量上。",
        "usage": "ROC 上穿/下穿零轴是动量方向切换最直接的定义；多周期 ROC（20/60/120日）同向时趋势可信度高。",
        "pitfalls": "单周期 ROC 噪声大；N 日前恰好是异常值时读数会失真（基期效应）。",
    },
    "bollinger": {
        "name": "布林带 Bollinger Bands",
        "category": "波动率",
        "formula": "中轨 = SMA(20)；上/下轨 = 中轨 ± 2×标准差(20)；%B = (价-下轨)/(上轨-下轨)；带宽 = (上轨-下轨)/中轨",
        "principle": (
            "统计上约 95% 的价格落在 ±2σ 之内，触及轨道即为统计意义上的极端。\n"
            "核心洞察是「波动率聚类与循环」：低波动（挤压）之后往往爆发大行情，高波动之后回归收敛。"
        ),
        "usage": (
            "1) 挤压突破：带宽收缩至历史低分位后，向上/向下放量突破往往是新趋势起点；\n"
            "2) 均值回归：震荡市中触及下轨买、上轨卖；\n"
            "3) band walking：强趋势中价格沿上轨爬升，此时触及上轨不是卖出理由。"
        ),
        "pitfalls": "突破方向无法由挤压本身预测；轨道是动态的，「跌破下轨」在恐慌中可以连续发生。",
    },
    "atr": {
        "name": "ATR 平均真实波幅",
        "category": "波动率",
        "formula": "TR = max(高-低, |高-昨收|, |低-昨收|)；ATR = TR 的 Wilder 平滑（N=14）",
        "principle": "用绝对价格单位衡量「这只票一天正常会动多少」，跳空也被计入。它不指示方向，只刻画波动环境。",
        "usage": (
            "1) 止损设定：入场价 ∓ 2×ATR，让止损距离适配该标的的呼吸幅度；\n"
            "2) 仓位管理：单笔风险固定时，仓位 = 风险额 / (2×ATR)，波动大的票自动减仓；\n"
            "3) 异动检测：ATR 突然飙升说明市场进入高波动状态，需降杠杆。"
        ),
        "pitfalls": "ATR 是滞后的平滑值，黑天鹅当天的真实波动远超 ATR 读数。",
    },
    "adx": {
        "name": "ADX 平均趋向指数",
        "category": "趋势强度",
        "formula": "+DI/-DI 衡量上/下方向力量；DX = |+DI - -DI|/(+DI + -DI)×100；ADX = DX 的平滑",
        "principle": (
            "ADX 不告诉你方向，只告诉你「有没有趋势、趋势有多强」。\n"
            "它是策略选择器：ADX>25 用趋势跟随策略，ADX<20 用均值回归策略。\n"
            "用错策略类型比用错参数的代价大得多。"
        ),
        "usage": "ADX 从低位拐头上行 = 新趋势萌芽；ADX 高位回落 = 趋势进入衰竭/休整（不代表反转）。方向看 +DI 与 -DI 的相对位置。",
        "pitfalls": "ADX 严重滞后，高位读数出现时趋势往往已走了很远；ADX 回落常被误读为反转信号，实际只代表趋势强度减弱。",
    },
    "obv": {
        "name": "OBV 能量潮",
        "category": "量能",
        "formula": "上涨日 OBV += 当日量；下跌日 OBV -= 当日量",
        "principle": "「量在价先」：聪明资金的进出会先体现在量能累计上。OBV 的绝对值无意义，形态和背离才有意义。",
        "usage": "价格新高 + OBV 同步新高 = 趋势健康；价格新高 + OBV 走低 = 顶背离，上涨缺乏资金支持。",
        "pitfalls": "对涨跌 0.01% 与 5% 一视同仁（只看方向不看幅度）；除权、巨量异常日会扭曲累计值。",
    },
    "vwap": {
        "name": "VWAP 成交量加权均价",
        "category": "量能 / 日内",
        "formula": "VWAP = Σ(典型价 × 成交量) / Σ成交量，典型价 = (高+低+收)/3，每个交易日重新累计",
        "principle": (
            "VWAP 是当日所有成交的真实平均成本，机构算法（执行基准）大量锚定它，\n"
            "因此它是日内最重要的多空分界与磁力线：价格在 VWAP 上方=日内买方占优。"
        ),
        "usage": (
            "1) 方向过滤：日内只在 VWAP 上方做多、下方做空；\n"
            "2) 回踩入场：强势股回踩 VWAP 获支撑是经典日内买点；\n"
            "3) 偏离回归：价格远离 VWAP 过多（如 >2 倍日内标准差）时有回归倾向。"
        ),
        "pitfalls": "VWAP 只在日内有意义，跨日累计的 VWAP（未分组）是错误用法；午后 VWAP 因累计量巨大而几乎不动，敏感度下降。",
    },
}

ALIASES = {
    "kd": "stochastic", "stoch": "stochastic", "boll": "bollinger",
    "bb": "bollinger", "ma": "sma", "均线": "sma",
}


def get_lesson(name: str) -> dict | None:
    key = name.lower().strip()
    key = ALIASES.get(key, key)
    return LESSONS.get(key)


def format_lesson(lesson: dict) -> str:
    return (
        f"# {lesson['name']}（{lesson['category']}类）\n\n"
        f"## 公式\n{lesson['formula']}\n\n"
        f"## 原理\n{lesson['principle']}\n\n"
        f"## 用法\n{lesson['usage']}\n\n"
        f"## 常见误区\n{lesson['pitfalls']}\n"
    )
