"""宏观/多资产总览（`finloop macro`）：把「各类投资市场的走势」放到一张可核对的表里。

这是第四部件（策略/信号）的**顶层视角**：个股/产业链信号之前，先回答
「现在的水位在哪、钱在往哪流、风险偏好开还是关」。

金融口径（每一条都是常见错误的反面）：
  1. 资产分三种 kind：
       price —— 股票/ETF/商品/加密，算 % 收益；
       yield —— 收益率（^TNX 报价 5.14 = 5.14%），算 **bp 变动**，把收益率当价格算 %
                会把 4.0→4.4 说成「涨 10%」，实际是 +40bp；
       level —— VIX 这类波动率指数，算点数变动（VIX 16→20 是 +4 点，不是「涨 25%」
                那么直观）。
  2. YTD 基准 = **上一自然年最后一个交易日收盘**（行业惯例），不是今年第一根 K 线的收盘
     ——后者会漏掉 1 月第一个交易日的涨跌。
  3. 200 日均线距离只对 price 口径有意义；yield/level 只给「在均线上/下方」。
  4. 首选真实标的（^VIX / GC=F / ^TNX，yfinance 可得）；不可得时回退到 ETF **代理**
     并在表中显式标注。代理≠标的：VIXY/USO 有展期损耗、IEF 是价格而收益率是反向。
  5. 相对强弱 = 两标的收盘价**比值**的窗口收益（比值下行 = 分子跑输分母），用于把
     「芯片热度在降」这类观感变成可证伪的数字。
  6. 风险偏好读数是**规则打分**而不是预测：每条规则的读数都列出来，供逐条反驳。
     打分不含黄金——黄金既可能是避险需求也可能是央行结构性购买，方向不唯一。
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

WINDOWS = {"1d": 1, "1w": 5, "1m": 21, "3m": 63}   # 交易日口径
KINDS = ("price", "yield", "level")


# ------------------------------------------------------------------ 纯函数

def _change(last: float, prev: float, kind: str) -> float | None:
    """按 kind 计算变动：price → %；yield → bp（报价单位为 %）；level → 点。"""
    if prev is None or (isinstance(prev, float) and math.isnan(prev)):
        return None
    if kind == "price":
        return None if prev == 0 else (last / prev - 1) * 100
    if kind == "yield":
        return (last - prev) * 100
    return last - prev


def asset_snapshot(df: pd.DataFrame, kind: str = "price") -> dict | None:
    """单资产快照：水平、各窗口变动、YTD、200 日线距离、52 周位置、20 日年化波动。

    df 需含 close（可选 high/low），DatetimeIndex 升序。数据不足的字段给 None，
    绝不用 0 顶替（0 会被读成「没涨没跌」）。
    """
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS}，收到 {kind!r}")
    if df is None or df.empty or "close" not in df:
        return None
    c = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(c) < 2:
        return None
    last = float(c.iloc[-1])
    date = c.index[-1]
    out: dict = {"kind": kind, "level": last, "n": int(len(c)),
                 "date": date.date().isoformat() if hasattr(date, "date") else str(date)}

    for key, w in WINDOWS.items():
        out[key] = _change(last, float(c.iloc[-1 - w]), kind) if len(c) > w else None

    # YTD：上一自然年最后一个收盘
    year = getattr(date, "year", None)
    prior = c[c.index.year < year] if year is not None else c.iloc[:0]
    out["ytd"] = _change(last, float(prior.iloc[-1]), kind) if len(prior) else None

    # 200 日均线
    if len(c) >= 200:
        sma200 = float(c.iloc[-200:].mean())
        out["sma200"] = sma200
        out["above_200dma"] = bool(last > sma200)
        out["vs_200dma_pct"] = (last / sma200 - 1) * 100 if kind == "price" and sma200 else None
    else:
        out["sma200"] = None
        out["above_200dma"] = None
        out["vs_200dma_pct"] = None

    # 52 周区间位置（price 用日内高低，其他用收盘）
    win = df.loc[c.index[-252:]] if len(c) >= 252 else df.loc[c.index]
    if kind == "price" and {"high", "low"} <= set(win.columns):
        hi = float(pd.to_numeric(win["high"], errors="coerce").max())
        lo = float(pd.to_numeric(win["low"], errors="coerce").min())
    else:
        seg = c.iloc[-252:]
        hi, lo = float(seg.max()), float(seg.min())
    out["hi_52w"], out["lo_52w"] = hi, lo
    out["pct_of_52w_range"] = (last - lo) / (hi - lo) * 100 if hi > lo else None
    out["off_52w_high_pct"] = (last / hi - 1) * 100 if kind == "price" and hi > 0 else None

    # 20 日已实现波动率（年化，仅 price）
    if kind == "price" and len(c) > 21:
        r = c.pct_change().iloc[-20:]
        out["rvol20_pct"] = float(r.std(ddof=1) * math.sqrt(252) * 100)
    else:
        out["rvol20_pct"] = None
    return out


def relative_strength(num: pd.DataFrame, den: pd.DataFrame) -> dict | None:
    """比值序列 num/den 的 price 口径快照（内连接对齐日期）。"""
    if num is None or den is None or num.empty or den.empty:
        return None
    j = pd.concat([num["close"].rename("a"), den["close"].rename("b")],
                  axis=1, join="inner").dropna()
    if len(j) < 2:
        return None
    ratio = (j["a"] / j["b"]).to_frame("close")
    return asset_snapshot(ratio, kind="price")


def risk_score(snaps: dict[str, dict], rs: dict[str, dict] | None = None,
               vix_is_proxy: bool = True) -> dict:
    """规则化风险偏好读数。返回 {score, label, rules:[(name, vote, evidence)]}.

    规则（每条 ±1/0，阈值写死且公开，便于反驳）：
      趋势：SPY 在 200 日线上方 +1 / 下方 -1；QQQ 同理。
      信用：HYG/IEF 比值 1 月变动 > +0.5% 为 +1，< -0.5% 为 -1（高收益债相对国债的风险偏好）。
      波动：VIX 水平 <20 +1 / 20-30 0 / >30 -1（仅真实 VIX；代理 ETF 跳过）。
      防御轮动：(XLP, XLV 均值 − SPY) 的 1 月收益差 > +2pp 为 -1（防御领涨），< -2pp 为 +1。
      美元：UUP/DXY 1 月涨幅 > +2% 为 -1（美元走强≈全球流动性收紧），其余 0。
    标签：总分 ≥3 risk-on；≤-2 risk-off；其余 mixed。
    """
    rules: list[tuple[str, int, str]] = []

    def trend(sym: str):
        s = snaps.get(sym)
        if not s or s.get("above_200dma") is None:
            rules.append((f"{sym} 趋势", 0, "数据不足"))
            return
        v = 1 if s["above_200dma"] else -1
        rules.append((f"{sym} 趋势", v,
                      f"{'高于' if v > 0 else '低于'} 200 日线 {s['vs_200dma_pct']:+.1f}%"))

    trend("SPY")
    trend("QQQ")

    cr = (rs or {}).get("HYG/IEF")
    if cr and cr.get("1m") is not None:
        v = 1 if cr["1m"] > 0.5 else (-1 if cr["1m"] < -0.5 else 0)
        rules.append(("信用 HYG/IEF", v, f"比值 1 月 {cr['1m']:+.2f}%"))
    else:
        rules.append(("信用 HYG/IEF", 0, "数据不足"))

    vix = snaps.get("^VIX")
    if vix and not vix_is_proxy:
        lvl = vix["level"]
        v = 1 if lvl < 20 else (-1 if lvl > 30 else 0)
        rules.append(("波动率 VIX", v, f"VIX = {lvl:.1f}"))
    else:
        rules.append(("波动率 VIX", 0, "无真实 VIX（代理 ETF 不计分）"))

    spy, xlp, xlv = snaps.get("SPY"), snaps.get("XLP"), snaps.get("XLV")
    if spy and xlp and xlv and None not in (spy.get("1m"), xlp.get("1m"), xlv.get("1m")):
        gap = (xlp["1m"] + xlv["1m"]) / 2 - spy["1m"]
        v = -1 if gap > 2 else (1 if gap < -2 else 0)
        rules.append(("防御轮动", v, f"防御(XLP,XLV)−SPY 1 月收益差 {gap:+.1f}pp"))
    else:
        rules.append(("防御轮动", 0, "数据不足"))

    usd = snaps.get("DX-Y.NYB") or snaps.get("UUP")
    if usd and usd.get("1m") is not None:
        v = -1 if usd["1m"] > 2 else 0
        rules.append(("美元", v, f"1 月 {usd['1m']:+.2f}%"))
    else:
        rules.append(("美元", 0, "数据不足"))

    score = sum(v for _, v, _ in rules)
    label = "risk-on" if score >= 3 else ("risk-off" if score <= -2 else "mixed")
    return {"score": score, "label": label, "rules": rules}


# ------------------------------------------------------------------ 配置与取数

def load_macro_config(path: Path | None = None) -> dict:
    from .config import config_dir
    p = path or (config_dir() / "macro.yaml")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("groups", {})
    cfg.setdefault("relative_pairs", [])
    return cfg


def _fetch_with_proxy(entry: dict, fetch: Callable[[str, str], pd.DataFrame],
                      period: str = "2y", min_rows: int = 30) -> tuple[pd.DataFrame, dict, bool]:
    """先取真实标的，不足 min_rows 时回退代理。返回 (df, 生效的条目, 是否代理)。"""
    try:
        df = fetch(entry["symbol"], period)
    except Exception:
        df = pd.DataFrame()
    if df is not None and len(df) >= min_rows:
        return df, entry, False
    proxy = entry.get("proxy")
    if proxy:
        try:
            pdf = fetch(proxy["symbol"], period)
        except Exception:
            pdf = pd.DataFrame()
        if pdf is not None and len(pdf) >= min_rows:
            eff = {**entry, **proxy, "kind": proxy.get("kind", entry.get("kind", "price"))}
            return pdf, eff, True
    return pd.DataFrame(), entry, False


# ------------------------------------------------------------------ 报告

def _fmt(v: float | None, kind: str, pct_sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if kind == "price":
        return f"{v:+.1f}%"
    if kind == "yield":
        return f"{v:+.0f}bp"
    return f"{v:+.2f}"


def _fmt_level(v: float, kind: str) -> str:
    if kind == "yield":
        return f"{v:.2f}%"
    return f"{v:,.2f}" if v < 1000 else f"{v:,.0f}"


def build_macro_report(cfg: dict | None = None,
                       fetch: Callable[[str, str], pd.DataFrame] | None = None,
                       period: str = "2y") -> str:
    """生成多资产总览 Markdown。fetch 可注入（测试/离线）。"""
    from . import DISCLAIMER
    if fetch is None:
        from .data import get_daily
        fetch = get_daily
    cfg = cfg or load_macro_config()

    frames: dict[str, pd.DataFrame] = {}
    snaps: dict[str, dict] = {}
    proxied: dict[str, str] = {}
    missing: list[str] = []
    asof = None

    lines = ["# 多资产市场总览", ""]
    for group, entries in cfg["groups"].items():
        lines += [f"## {group}", "",
                  "| 资产 | 水平 | 1日 | 1周 | 1月 | 3月 | YTD | vs200日线 | 52周位置 | 20日波动 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for entry in entries:
            kind = entry.get("kind", "price")
            df, eff, is_proxy = _fetch_with_proxy(entry, fetch, period)
            snap = asset_snapshot(df, eff.get("kind", kind)) if not df.empty else None
            if snap is None:
                missing.append(f"{entry['name']}({entry['symbol']})")
                lines.append(f"| {entry['name']} | 不可得 | — | — | — | — | — | — | — | — |")
                continue
            key = entry["symbol"]
            frames[key] = df
            frames[eff["symbol"]] = df
            snaps[key] = snap
            snaps[eff["symbol"]] = snap
            if is_proxy:
                proxied[key] = eff["symbol"]
            asof = max(asof or snap["date"], snap["date"])
            k = snap["kind"]
            name = entry["name"] + (f" ⚠️代理:{eff['name']}" if is_proxy else "")
            if k == "price":
                v200 = _fmt(snap["vs_200dma_pct"], "price")
            elif snap["above_200dma"] is None:
                v200 = "—"
            else:
                v200 = "上方" if snap["above_200dma"] else "下方"
            pos = "—" if snap["pct_of_52w_range"] is None else f"{snap['pct_of_52w_range']:.0f}%"
            vol = "—" if snap["rvol20_pct"] is None else f"{snap['rvol20_pct']:.0f}%"
            lines.append(
                f"| {name} | {_fmt_level(snap['level'], k)} | {_fmt(snap['1d'], k)} | {_fmt(snap['1w'], k)} "
                f"| {_fmt(snap['1m'], k)} | {_fmt(snap['3m'], k)} | {_fmt(snap['ytd'], k)} | {v200} | {pos} | {vol} |")
        lines.append("")

    lines += ["> 口径：price 资产为 % 收益；收益率为 **bp** 变动；VIX 为点数。YTD 基准为上年末收盘。"
              "52 周位置 = (现价−52周低)/(52周高−52周低)，0% 在年内最低、100% 在年内最高。"
              "20 日波动为年化已实现波动率。", ""]
    if proxied:
        lines += ["> ⚠️ 代理说明：" + "；".join(f"{k}→{v}" for k, v in proxied.items())
                  + "。代理与标的不等价：VIXY/USO 含期货展期损耗，IEF 是债券价格（与收益率反向）。", ""]
    if missing:
        lines += ["> 不可得：" + "、".join(missing) + "（当前数据源不覆盖，且无代理或代理亦失败）。", ""]

    # 相对强弱
    rs: dict[str, dict] = {}
    pairs = list(cfg.get("relative_pairs", []))
    if ("HYG", "IEF") not in {(p[0], p[1]) for p in pairs} and "HYG" in frames and "IEF" in frames:
        pairs.append(["HYG", "IEF", "高收益债/国债(信用风险偏好)"])
    if pairs:
        lines += ["## 相对强弱（比值收益：正 = 分子跑赢分母）", "",
                  "| 比值 | 含义 | 1周 | 1月 | 3月 | YTD | 比值vs200日线 |", "|---|---|---:|---:|---:|---:|---:|"]
        for a, b, label in pairs:
            fa, fb = frames.get(a), frames.get(b)
            if fa is None:
                fa = _safe_fetch(fetch, a, period); frames[a] = fa
            if fb is None:
                fb = _safe_fetch(fetch, b, period); frames[b] = fb
            s = relative_strength(fa, fb)
            if s is None:
                lines.append(f"| {a}/{b} | {label} | — | — | — | — | — |")
                continue
            rs[f"{a}/{b}"] = s
            lines.append(f"| {a}/{b} | {label} | {_fmt(s['1w'], 'price')} | {_fmt(s['1m'], 'price')} "
                         f"| {_fmt(s['3m'], 'price')} | {_fmt(s['ytd'], 'price')} | {_fmt(s['vs_200dma_pct'], 'price')} |")
        lines.append("")

    # 风险偏好
    score = risk_score(snaps, rs, vix_is_proxy="^VIX" in proxied or "^VIX" not in snaps)
    lines += [f"## 风险偏好读数：**{score['label']}**（{score['score']:+d}）", "",
              "| 规则 | 票 | 读数 |", "|---|:-:|---|"]
    for name, v, ev in score["rules"]:
        mark = {1: "🟢 +1", 0: "⚪ 0", -1: "🔴 -1"}[v]
        lines.append(f"| {name} | {mark} | {ev} |")
    lines += ["", "> 读数是规则打分不是预测。risk-on ≥ +3，risk-off ≤ -2，其余 mixed。"
              "规则与阈值见 `finloop.macro.risk_score`，欢迎逐条反驳。", ""]

    lines += [f"_数据截至 {asof or '未知'}，生成于 {dt.date.today().isoformat()}。_", "", "---", DISCLAIMER, ""]
    return "\n".join(lines)


def _safe_fetch(fetch, symbol: str, period: str) -> pd.DataFrame:
    try:
        return fetch(symbol, period)
    except Exception:
        return pd.DataFrame()
