"""数据防线回归测试（2026-06 估值验证轮次的工程化产物）。

四道防线 + 备源换代，每条对应当轮实测抓到的一类事故：
  1. validate_fundamentals：fPE/tPE 偏离（3110.T 静默错误）、
     隐含净利率荒谬（MU 远期口径）
  2. repair_ohlcv：孤立坏行剔除（000660.KS 三行坏数据毁整票）
  3. DataService 修复编排：备源优先于修复、修复显形于 warnings
  4. 快照存档：排名表是加工品，快照才是证据
  5. RegionalFallbackProvider 路由 + 各 provider 离线解析
"""

import json

import numpy as np
import pandas as pd
import pytest

from finloop.data.fundamentals import validate_fundamentals
from finloop.data.providers.base import empty_ohlcv
from finloop.data.providers.naver_provider import NaverProvider, _parse_sise
from finloop.data.providers.regional import RegionalFallbackProvider
from finloop.data.providers.taiwan_provider import TaiwanProvider, _num, _roc_to_date
from finloop.data.providers.tiingo_provider import TiingoProvider
from finloop.data.quality import repair_ohlcv, validate_ohlcv
from finloop.data.service import DataService
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------- 防线1+2：基本面一致性

def test_fundamentals_fpe_far_above_tpe_warns():
    """3110.T 案例：fPE 58.5 vs 共识 ~18 → fPE/tPE 偏离暴露过时口径。"""
    warns = validate_fundamentals({"forwardPE": 58.5, "trailingPE": 17.0})
    assert any("fPE/tPE" in w and "过时" in w for w in warns)


def test_fundamentals_fpe_far_below_tpe_warns():
    """MU 案例：fPE 8.2 vs trailing 43.8 → 远期峰值口径。"""
    warns = validate_fundamentals({"forwardPE": 8.2, "trailingPE": 43.8})
    assert any("远期" in w or "周期顶" in w for w in warns)


def test_fundamentals_implied_margin_impossible():
    """MU 案例：市值 $1.04T / fPE 8.2 隐含净利 $127B > 营收 → 必有一错。"""
    warns = validate_fundamentals({
        "forwardPE": 8.2, "marketCap": 1.04e12, "totalRevenue": 6.0e10,
    })
    assert any("必有一错" in w for w in warns)


def test_fundamentals_implied_margin_soft_warn():
    warns = validate_fundamentals({
        "forwardPE": 10.0, "marketCap": 7.0e10, "totalRevenue": 1.0e10,
    })  # 隐含净利率 70%
    assert any("先核验" in w for w in warns)


def test_fundamentals_clean_and_missing_no_warn():
    assert validate_fundamentals({"forwardPE": 20.0, "trailingPE": 25.0,
                                  "marketCap": 2.0e10, "totalRevenue": 1.0e10}) == []
    assert validate_fundamentals({}) == []
    assert validate_fundamentals({"forwardPE": 20.0}) == []  # 缺对照字段→跳过
    # 负值（亏损）不参与比值检查
    assert validate_fundamentals({"forwardPE": -5.0, "trailingPE": 30.0}) == []


# ---------------------------------------------------------------- 防线3：坏行修复

def _corrupt_rows(df, positions):
    """把指定行破坏成 high < low（结构性错误）。"""
    bad = df.copy()
    for i in positions:
        bad.iloc[i, bad.columns.get_loc("high")] = bad.iloc[i]["low"] - 1.0
    return bad


def test_repair_few_bad_rows(trending_up):
    """000660.KS 案例：1219 行里 3 处坏行不应毁掉整票。"""
    bad = _corrupt_rows(trending_up, [10, 50, 100])
    assert not validate_ohlcv(bad, "KS").ok  # 修复前确实被拒
    repaired, notes = repair_ohlcv(bad, "KS")
    assert len(repaired) == len(bad) - 3
    assert notes and "3/" in notes[0]
    assert validate_ohlcv(repaired, "KS").ok


def test_repair_refuses_mass_corruption(trending_up):
    """大面积污染（>max(5,0.5%)）不修——掩盖比拒绝危险。"""
    bad = _corrupt_rows(trending_up, list(range(0, 60)))
    repaired, notes = repair_ohlcv(bad, "MASS")
    assert notes == [] and len(repaired) == len(bad)


def test_repair_duplicate_timestamps(trending_up):
    dup = pd.concat([trending_up, trending_up.iloc[[50]]]).sort_index()
    repaired, notes = repair_ohlcv(dup, "DUP")
    assert len(repaired) == len(trending_up) and notes
    assert validate_ohlcv(repaired, "DUP").ok


def test_repair_clean_data_untouched(trending_up):
    repaired, notes = repair_ohlcv(trending_up, "CLEAN")
    assert notes == [] and repaired is trending_up


# ---------------------------------------------------------------- 防线3：DataService 编排

class FakeProvider:
    def __init__(self, name, df=None, raises=False):
        self.name = name
        self.df = df if df is not None else empty_ohlcv()
        self.raises = raises

    def fetch_daily(self, ticker, period):
        if self.raises:
            raise ConnectionError("boom")
        return self.df

    def fetch_intraday(self, ticker, interval, period):
        return self.fetch_daily(ticker, period)


def test_service_repairs_when_all_sources_rejected(trending_up):
    """主源坏行可修 + 备源死 → 修复采用，来源标记 +repair，修复显形。"""
    bad = _corrupt_rows(trending_up, [10, 50, 100])
    svc = DataService(FakeProvider("p", bad), FakeProvider("f", raises=True),
                      use_cache=False)
    df = svc.get_daily("X")
    assert len(df) == len(bad) - 3
    assert svc.last_source == "p+repair"
    assert any("剔除" in w for w in svc.last_report.warnings)


def test_service_prefers_clean_fallback_over_repair(trending_up):
    """干净的备源数据优先于修补过的主源数据（保持既有降级语义）。"""
    bad = _corrupt_rows(trending_up, [10])
    svc = DataService(FakeProvider("p", bad), FakeProvider("f", trending_up),
                      use_cache=False)
    svc.get_daily("X")
    assert svc.last_source == "f"


def test_service_mass_corruption_still_fails(trending_up):
    bad = _corrupt_rows(trending_up, list(range(0, 60)))
    svc = DataService(FakeProvider("p", bad), FakeProvider("f", raises=True),
                      use_cache=False)
    df = svc.get_daily("X")
    assert df.empty and not svc.last_report.ok


# ---------------------------------------------------------------- 防线4：快照存档

def test_screen_archive_snapshot(tmp_path):
    from finloop.screener import archive_snapshot

    path = archive_snapshot(
        fundamentals={"AAA": {"forwardPE": 10.0}},
        labels={"AAA": "测试"},
        warnings={"AAA": ["示例预警"], "BBB": []},
        universe="ai", tier="upstream", out_dir=tmp_path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["fundamentals"]["AAA"]["forwardPE"] == 10.0
    assert payload["warnings"] == {"AAA": ["示例预警"]}  # 空预警不存档
    assert "fetched_at" in payload and "screen_ai_upstream" in path.name


def test_format_screen_shows_warning_column():
    from finloop.screener import format_screen, score_universe

    scored = score_universe({"AAA": {"forwardPE": 58.5, "revenueGrowth": 0.1},
                             "BBB": {"forwardPE": 20.0, "revenueGrowth": 0.3}})
    scored.insert(0, "标签", ["a", "b"])
    scored.insert(1, "数据预警", ["fPE/tPE=3.4x：疑似过时", ""])
    text = format_screen(scored, "upstream")
    assert "数据预警" in text and "疑似过时" in text
    assert "错数据显形于预警列" in text  # 解读纪律已更新


# ---------------------------------------------------------------- 防线5：备源路由与解析

def test_regional_routing(trending_up):
    tw = FakeProvider("taiwan-official", trending_up)
    kr = FakeProvider("naver", trending_up)
    us = FakeProvider("tiingo", trending_up)
    us.available = True
    router = RegionalFallbackProvider(tiingo=us, taiwan=tw, naver=kr,
                                      stooq=FakeProvider("stooq"))
    assert not router.fetch_daily("3081.TWO", "1y").empty
    assert router.last_route == "taiwan-official"
    assert not router.fetch_daily("2330.TW", "1y").empty
    assert router.last_route == "taiwan-official"
    assert not router.fetch_daily("000660.KS", "1y").empty
    assert router.last_route == "naver"
    assert not router.fetch_daily("MU", "1y").empty
    assert router.last_route == "tiingo"
    # 无备源市场诚实返回空，不假装有数据
    assert router.fetch_daily("4062.T", "1y").empty
    assert router.last_route is None


def test_regional_us_falls_to_stooq_without_tiingo_key(trending_up):
    router = RegionalFallbackProvider(
        tiingo=TiingoProvider(api_key=""),  # 无 key → available=False
        taiwan=FakeProvider("tw"), naver=FakeProvider("kr"),
        stooq=FakeProvider("stooq", trending_up))
    assert not router.fetch_daily("MU", "1y").empty
    assert router.last_route == "stooq"


def test_tiingo_without_key_makes_no_request(monkeypatch):
    def explode(*a, **kw):
        raise AssertionError("不应发起网络请求")
    monkeypatch.setattr("finloop.data.providers.tiingo_provider.requests.get", explode)
    assert TiingoProvider(api_key="").fetch_daily("MU", "1y").empty


def test_taiwan_parsers():
    assert _roc_to_date("115/06/01") == pd.Timestamp(2026, 6, 1)
    assert _roc_to_date("garbage") is None
    assert _num("1,234.56") == 1234.56
    assert _num("--") is None and _num(None) is None


def test_taiwan_month_parsing(monkeypatch):
    """TWSE 月度 JSON（民国纪年+千分位）→ 标准 OHLCV。"""
    payload = {"stat": "OK", "data": [
        ["115/06/01", "33,743,629", "x", "960.00", "970.00", "955.00", "965.00", "+5", "n"],
        ["115/06/02", "30,000,000", "x", "966.00", "980.00", "--", "975.00", "+10", "n"],
        ["garbage-date", "1", "x", "1", "1", "1", "1", "1", "n"],  # 坏行跳过
    ]}

    class FakeResp:
        def raise_for_status(self): ...
        def json(self): return payload

    monkeypatch.setattr("finloop.data.providers.taiwan_provider.requests.get",
                        lambda *a, **kw: FakeResp())
    rows = TaiwanProvider()._fetch_month_twse("2330", pd.Timestamp(2026, 6, 1))
    assert len(rows) == 2
    assert rows[0]["close"] == 965.0 and rows[0]["volume"] == 33743629.0
    assert rows[1]["low"] is None  # "--" → None，由 normalize/质量门处置


def test_naver_parse_and_fetch(monkeypatch):
    text = ("[['날짜', '시가', '고가', '저가', '종가', '거래량', '외국인소진율'], \n"
            '["20260601", 126000, 126500, 124000, 126000, 3270936, 49.99], \n'
            '["20260602", 126500, 130000, 126000, 129500, 4100000, 50.01]]')
    assert len(_parse_sise(text)) == 3

    class FakeResp:
        def __init__(self): self.text = text
        def raise_for_status(self): ...

    monkeypatch.setattr("finloop.data.providers.naver_provider.requests.get",
                        lambda *a, **kw: FakeResp())
    df = NaverProvider().fetch_daily("000660.KS", "1y")
    assert list(df.index) == [pd.Timestamp(2026, 6, 1), pd.Timestamp(2026, 6, 2)]
    assert df.iloc[1]["close"] == 129500.0
    assert validate_ohlcv(df, "000660.KS").ok
    # 非 .KS 代码不归它管
    assert NaverProvider().fetch_daily("MU", "1y").empty
