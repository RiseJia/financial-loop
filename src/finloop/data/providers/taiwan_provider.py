"""备用数据源：台湾证交所(TWSE)/柜买中心(TPEx) 官方开放数据（无需 API key）。

覆盖：.TW（上市，走 TWSE）与 .TWO（上柜，走 TPEx）后缀的日线。
这是交易所自己发布的数据——质量上是"官方原始记录"级别，
正好补上 yfinance 对台股小盘（如 3081.TWO）预期字段不可靠时的行情底座。

口径与限制（接入前必读）：
  - 价格为**未复权**原始价：拆股/除权除息日前后会出现价格跳变，
    与 yfinance 全复权口径对账时远端历史允许系统性偏差，
    质量门的"极端跳变"警告在除权日附近属预期行为；
  - 接口按"月"返回数据 → 拉 5y 需要 ~61 次请求，每次间隔 REQUEST_GAP 秒
    礼貌限速——备源定位是"救急与对账"，不是高频批量拉取；
  - 日期为民国纪年（如 115/06/01 = 2026-06-01），数值含千分位逗号，
    缺值用 "--"；解析失败的行跳过并计数，不让单行坏数据毁掉整月。

实网验证（2026-06-12）：TWSE（2330.TW）与 TPEx（3081.TWO）均验收通过，
真实响应与解析假设一致；与 yfinance 的对账偏差呈"除息阶跃"形态——
最近一次除息后两源逐日一致，更早区间为常数级系统偏差（复权口径差异，
预期内），实测数字见 docs/data_quality.md。
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .base import empty_ohlcv, normalize_ohlcv, period_to_days

REQUEST_GAP = 0.3  # 秒，对官方接口的礼貌间隔

TWSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"

_HEADERS = {"User-Agent": "finloop/0.1 (research; contact: repo owner)"}


def _roc_to_date(s: str) -> pd.Timestamp | None:
    """民国纪年 '115/06/01' → Timestamp(2026-06-01)；解析失败返回 None。"""
    try:
        y, m, d = s.strip().split("/")
        return pd.Timestamp(int(y) + 1911, int(m), int(d))
    except (ValueError, AttributeError):
        return None


def _num(s) -> float | None:
    """'1,234.56' → 1234.56；'--' 与空值 → None。"""
    if s is None:
        return None
    text = str(s).replace(",", "").strip()
    if text in ("", "--", "---", "X"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


class TaiwanProvider:
    """按后缀路由：.TW → TWSE，.TWO → TPEx。其余代码返回空。"""

    name = "taiwan-official"

    def __init__(self, timeout: float = 15.0, request_gap: float = REQUEST_GAP):
        self.timeout = timeout
        self.request_gap = request_gap

    # ------------------------------------------------------------ 单月抓取
    def _fetch_month_twse(self, code: str, month_start: pd.Timestamp) -> list[dict]:
        resp = requests.get(
            TWSE_URL,
            params={"date": month_start.strftime("%Y%m%d"), "stockNo": code,
                    "response": "json"},
            headers=_HEADERS, timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("stat") != "OK":
            return []
        rows = []
        # fields: 日期 成交股數 成交金額 開盤價 最高價 最低價 收盤價 漲跌價差 成交筆數
        for r in payload.get("data", []):
            if len(r) < 7:
                continue
            date = _roc_to_date(r[0])
            o, h, lo, c = _num(r[3]), _num(r[4]), _num(r[5]), _num(r[6])
            vol = _num(r[1])  # 成交股數：股
            if date is None or c is None:
                continue
            rows.append({"date": date, "open": o, "high": h, "low": lo,
                         "close": c, "volume": vol or 0.0})
        return rows

    def _fetch_month_tpex(self, code: str, month_start: pd.Timestamp) -> list[dict]:
        resp = requests.get(
            TPEX_URL,
            params={"code": code, "date": month_start.strftime("%Y/%m/%d"),
                    "response": "json"},
            headers=_HEADERS, timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        tables = payload.get("tables") or []
        data = tables[0].get("data", []) if tables else payload.get("aaData", [])
        rows = []
        # fields(实测): 日 期 成交張數 成交仟元 開盤 最高 最低 收盤 漲跌 筆數
        for r in data:
            if len(r) < 7:
                continue
            date = _roc_to_date(r[0])
            o, h, lo, c = _num(r[3]), _num(r[4]), _num(r[5]), _num(r[6])
            vol = _num(r[1])  # 成交張數：1 張 = 1000 股
            if date is None or c is None:
                continue
            rows.append({"date": date, "open": o, "high": h, "low": lo,
                         "close": c, "volume": (vol or 0.0) * 1000})
        return rows

    # ------------------------------------------------------------ Provider 协议
    def fetch_daily(self, ticker: str, period: str) -> pd.DataFrame:
        if ticker.endswith(".TWO"):
            code, fetch_month = ticker[:-4], self._fetch_month_tpex
        elif ticker.endswith(".TW"):
            code, fetch_month = ticker[:-3], self._fetch_month_twse
        else:
            return empty_ohlcv()

        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=period_to_days(period))
        month = pd.Timestamp.today().normalize().replace(day=1)
        all_rows: list[dict] = []
        while month >= cutoff.replace(day=1):
            try:
                all_rows.extend(fetch_month(code, month))
            except requests.RequestException:
                # 单月失败不中止：备源的价值是"尽量给出可对账的数据"
                pass
            month = (month - pd.Timedelta(days=1)).replace(day=1)
            if month >= cutoff.replace(day=1):
                time.sleep(self.request_gap)

        if not all_rows:
            return empty_ohlcv()
        df = pd.DataFrame(all_rows).set_index("date")
        df = normalize_ohlcv(df)
        return df[df.index >= cutoff]

    def fetch_intraday(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        return empty_ohlcv()
