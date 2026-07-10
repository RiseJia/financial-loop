import numpy as np
import pandas as pd

from finloop.data.providers.base import empty_ohlcv
from finloop.data.quality import reconcile, validate_ohlcv
from finloop.data.service import DataService
from finloop.indicators import stochastic
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------- 校验管道

def test_validate_good_data(trending_up):
    rep = validate_ohlcv(trending_up, "GOOD")
    assert rep.ok and not rep.errors


def test_validate_empty():
    assert not validate_ohlcv(empty_ohlcv(), "EMPTY").ok


def test_validate_nonpositive_close(trending_up):
    df = trending_up.copy()
    df.iloc[10, df.columns.get_loc("close")] = -5.0
    rep = validate_ohlcv(df, "BAD")
    assert not rep.ok and any("非正" in e for e in rep.errors)


def test_validate_ohlc_inconsistency(trending_up):
    df = trending_up.copy()
    df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["low"] - 1  # high < low
    rep = validate_ohlcv(df, "BAD")
    assert not rep.ok and any("自洽" in e for e in rep.errors)


def test_validate_duplicate_timestamps(trending_up):
    df = pd.concat([trending_up, trending_up.iloc[[50]]]).sort_index()
    rep = validate_ohlcv(df, "DUP")
    assert not rep.ok and any("重复" in e for e in rep.errors)


def test_validate_duplicate_columns_rejected(trending_up):
    """重复列名按结构性错误拒绝，不允许崩溃。

    真实事故（2026-06 Tiingo 实网验收）：响应里原始列与 adj* 列并存，
    provider rename 后出现两个 close 列，df["close"] 取出 DataFrame，
    后续所有标量判断抛 ValueError——质量门自己先崩了。
    """
    df = pd.concat([trending_up, trending_up[["close"]]], axis=1)
    rep = validate_ohlcv(df, "DUPCOL")
    assert not rep.ok and any("重复列名" in e for e in rep.errors)


def test_validate_extreme_jump_warns(trending_up):
    df = trending_up.copy()
    df.iloc[100, df.columns.get_loc("close")] *= 3  # 单日 +200%
    # 修高低价保持自洽，确保只触发 warning 不触发 error
    df.iloc[100, df.columns.get_loc("high")] = df.iloc[100]["close"] * 1.01
    rep = validate_ohlcv(df, "JUMP")
    assert rep.ok and any("跳变" in w for w in rep.warnings)


def test_reconcile_identical_and_shifted(trending_up):
    same = reconcile(trending_up, trending_up.copy())
    assert same["overlap_days"] == len(trending_up)
    assert same["mean_abs_dev"] < 1e-12

    shifted = trending_up.copy()
    shifted["close"] *= 1.05  # 系统性 5% 偏差
    rep = reconcile(trending_up, shifted, tolerance=0.01)
    assert rep["n_breaches"] == rep["overlap_days"]


# ---------------------------------------------------------------- DataService 编排

class FakeProvider:
    def __init__(self, name, df=None, raises=False):
        self.name = name
        self.df = df if df is not None else empty_ohlcv()
        self.raises = raises
        self.calls = 0

    def fetch_daily(self, ticker, period):
        self.calls += 1
        if self.raises:
            raise ConnectionError("boom")
        return self.df

    def fetch_intraday(self, ticker, interval, period):
        return self.fetch_daily(ticker, period)


def test_service_uses_primary(trending_up):
    primary = FakeProvider("p", trending_up)
    fallback = FakeProvider("f", trending_up)
    svc = DataService(primary, fallback, use_cache=False)
    df = svc.get_daily("X")
    assert not df.empty and svc.last_source == "p"
    assert primary.calls == 1 and fallback.calls == 0


def test_service_falls_back_on_exception(trending_up):
    svc = DataService(FakeProvider("p", raises=True),
                      FakeProvider("f", trending_up), use_cache=False)
    df = svc.get_daily("X")
    assert not df.empty and svc.last_source == "f"


def test_service_falls_back_on_bad_quality(trending_up):
    bad = trending_up.copy()
    bad.iloc[10, bad.columns.get_loc("close")] = -1.0
    svc = DataService(FakeProvider("p", bad),
                      FakeProvider("f", trending_up), use_cache=False)
    df = svc.get_daily("X")
    assert not df.empty and svc.last_source == "f"


def test_service_all_fail():
    svc = DataService(FakeProvider("p", raises=True),
                      FakeProvider("f", raises=True), use_cache=False)
    df = svc.get_daily("X")
    assert df.empty and not svc.last_report.ok


# ---------------------------------------------------------------- 边界输入修复回归

def test_stochastic_flat_price_no_division_error():
    """一字横盘：区间为 0 → 暖机后置中性 50（不 inf）；暖机期保留 NaN（不捏造）。"""
    df = make_ohlcv(np.full(60, 100.0))
    df["high"] = 100.0
    df["low"] = 100.0
    k = stochastic(df["high"], df["low"], df["close"])["stoch_k"]
    assert not np.isinf(k).any()               # 无除零 inf
    post = k.iloc[13:]                          # 暖机期后（14日窗口已满）
    assert (post == 50.0).all()                # 区间为零 → 中性 50
    assert k.iloc[:13].isna().all()            # 暖机期保留 NaN，不捏造成 50


# ---------------------------------------------------------------- 对抗审计：复权口径

def test_provider_adjustment_declared():
    """每个 provider 声明复权口径，供下游判断收益率是否可比。"""
    from finloop.data.providers import (YFinanceProvider, TiingoProvider,
                                        StooqProvider, TaiwanProvider, NaverProvider)
    assert YFinanceProvider.adjustment == "full"
    assert TiingoProvider.adjustment == "full"
    assert StooqProvider.adjustment == "split_only"
    assert TaiwanProvider.adjustment == "none"
    assert NaverProvider.adjustment == "split_only"


def test_service_records_adjustment(trending_up):
    """DataService 记录本次数据的复权口径。"""
    class P:
        name = "p"; adjustment = "full"
        def fetch_daily(self, t, period): return trending_up
        def fetch_intraday(self, *a): return trending_up
    svc = DataService(P(), P(), use_cache=False)
    svc.get_daily("X")
    assert svc.last_adjustment == "full"


def test_reconcile_recent_window_limits_breaches(trending_up):
    """recent_days 限制对账窗口——只看近端可减少口径分歧的除息 breach 累积。"""
    a = trending_up.copy()
    b = trending_up.copy()
    b.iloc[:100, b.columns.get_loc("close")] *= 1.05   # 远端 100 天系统性偏差
    full = reconcile(a, b, tolerance=0.01)
    recent = reconcile(a, b, tolerance=0.01, recent_days=50)
    assert recent["n_breaches"] < full["n_breaches"]   # 近端窗口 breach 更少
