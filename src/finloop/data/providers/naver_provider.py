"""备用数据源：Naver 金融（韩股日线，无需 API key）。

背景：yfinance 的韩股日线出过 OHLC 自洽性坏行（实测 000660.KS 有 3 处，
导致整票被质量门拒收），而 Stooq 不覆盖韩股。Naver 是韩国事实标准的
行情门户，其图表接口长期公开可用、无反爬验证。

口径与限制：
  - siseJson 接口返回的是**修正股价**（韩语"수정주가"，拆股调整），
    分红是否调整未经官方文档确认——与 yfinance 对账时按"拆股调整、
    分红存疑"的容差对待（同 Stooq 时代的口径注意事项）；
  - 响应是 Python 风格的列表文本（单引号），需做容错解析；
  - 仅日线。

⚠️ 仍未经实网验证（2026-06-12 第二次尝试失败）：执行环境的 egress 代理
对 api.finance.naver.com 有域名白名单之外的整域拦截（任意路径/方法均
返回代理 403 "Blocked by egress policy"），解析逻辑仍只过了离线单测。
降级管线对该故障的行为已实测：`finloop quality 000660.KS` 显式报
"备源失败 403"并标明路由，不静默。在不经此代理的环境首次启用时，
先跑 `finloop quality 000660.KS` 确认解析正确。
"""

from __future__ import annotations

import ast
import json

import pandas as pd
import requests

from .base import empty_ohlcv, normalize_ohlcv, period_to_days

NAVER_URL = "https://api.finance.naver.com/siseJson.naver"

_HEADERS = {"User-Agent": "Mozilla/5.0 (research; finloop)"}


def _parse_sise(text: str) -> list[list]:
    """Naver 返回 Python 风格列表文本（单引号、可能含多余空白）。

    先尝试 JSON（替换引号后），再退到 ast.literal_eval（仅字面量，安全）。
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    try:
        return json.loads(cleaned.replace("'", '"'))
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(cleaned)
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []


class NaverProvider:
    name = "naver"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def fetch_daily(self, ticker: str, period: str) -> pd.DataFrame:
        if not ticker.endswith(".KS"):
            return empty_ohlcv()
        code = ticker[:-3]
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=period_to_days(period))
        resp = requests.get(
            NAVER_URL,
            params={"symbol": code, "requestType": 1,
                    "startTime": start.strftime("%Y%m%d"),
                    "endTime": end.strftime("%Y%m%d"), "timeframe": "day"},
            headers=_HEADERS, timeout=self.timeout,
        )
        resp.raise_for_status()
        rows = _parse_sise(resp.text)
        if len(rows) < 2:
            return empty_ohlcv()
        # rows[0] 是表头：날짜(日期) 시가(开) 고가(高) 저가(低) 종가(收) 거래량(量) ...
        records = []
        for r in rows[1:]:
            if not isinstance(r, (list, tuple)) or len(r) < 6:
                continue
            try:
                records.append({
                    "date": pd.Timestamp(str(r[0])),
                    "open": float(r[1]), "high": float(r[2]),
                    "low": float(r[3]), "close": float(r[4]),
                    "volume": float(r[5]),
                })
            except (ValueError, TypeError):
                continue  # 单行坏数据跳过，不毁整票
        if not records:
            return empty_ohlcv()
        df = pd.DataFrame(records).set_index("date")
        return normalize_ohlcv(df)

    def fetch_intraday(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        return empty_ohlcv()
