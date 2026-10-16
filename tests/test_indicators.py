# -*- coding: utf-8 -*-
"""模块 3：技术指标单元测试"""
import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    sma, ema, rsi, macd, obv, volatility, momentum, lag_returns, build_features,
)


def series(vals):
    return pd.Series(vals, dtype="float64")


# ============ SMA ============
def test_sma_basic():
    out = sma(series([1, 2, 3, 4, 5]), 2)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(1.5)
    assert out.iloc[2] == pytest.approx(2.5)
    assert out.iloc[4] == pytest.approx(4.5)


def test_sma_constant():
    out = sma(series([5.0] * 10), 3)
    assert out.dropna().eq(5.0).all()


def test_sma_window_larger_than_data():
    out = sma(series([1.0, 2.0]), 10)
    assert len(out) == 2
    assert out.isna().all()


# ============ EMA ============
def test_ema_first_value_equals_input():
    out = ema(series([1.0, 2.0, 3.0]), 2)
    assert out.iloc[0] == pytest.approx(1.0)


def test_ema_constant():
    out = ema(series([7.0] * 10), 5)
    assert out.dropna().eq(7.0).all()


def test_ema_reacts_faster_than_sma():
    s = series([10.0] * 20 + [20.0])
    assert ema(s, 5).iloc[-1] > sma(s, 5).iloc[-1]


# ============ RSI ============
def test_rsi_range_0_to_100():
    rng = np.random.default_rng(42)
    close = series(100 + np.cumsum(rng.normal(0, 1, 100)))
    out = rsi(close, 14).dropna()
    assert (out >= 0).all() and (out <= 100).all()


def test_rsi_all_up_is_high():
    out = rsi(series(np.arange(1.0, 60.0)), 14).dropna()
    assert out.iloc[-1] >= 99, f"全涨 RSI 应接近 100，实际 {out.iloc[-1]}"


def test_rsi_all_down_is_low():
    out = rsi(series(np.arange(60.0, 1.0, -1.0)), 14).dropna()
    assert out.iloc[-1] <= 1, f"全跌 RSI 应接近 0，实际 {out.iloc[-1]}"


def test_rsi_downtrend_is_low():
    s = series(list(np.arange(1.0, 30.0)) + list(np.arange(30.0, 1.0, -1.0)))
    out = rsi(s, 14).dropna()
    assert out.iloc[-1] < 50


# ============ MACD ============
def test_macd_returns_three_series():
    dif, dea, hist = macd(series(np.arange(1.0, 100.0)))
    assert len(dif) == len(dea) == len(hist)


def test_macd_hist_equals_dif_minus_dea():
    s = series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 80)))
    dif, dea, hist = macd(s)
    pd.testing.assert_series_equal(hist, dif - dea, check_names=False)


def test_macd_constant_price_is_zero():
    s = series([50.0] * 60)
    dif, dea, hist = macd(s)
    assert abs(dif.dropna()).max() < 1e-8
    assert abs(hist.dropna()).max() < 1e-8


def test_macd_uptrend_dif_positive():
    dif, _, _ = macd(series(np.arange(1.0, 120.0)))
    assert dif.dropna().iloc[-1] > 0


# ============ OBV ============
def test_obv_known_values():
    close = series([10.0, 11.0, 10.0, 12.0])
    volume = series([100.0, 200.0, 300.0, 400.0])
    out = obv(close, volume)
    # diff=[NaN,+1,-1,+2] -> cumsum=[0,200,-100,300]
    assert out.iloc[1] == pytest.approx(200.0)
    assert out.iloc[2] == pytest.approx(-100.0)
    assert out.iloc[3] == pytest.approx(300.0)


def test_obv_all_up_accumulates_all_volume():
    out = obv(series([1.0, 2.0, 3.0]), series([10.0, 20.0, 30.0]))
    assert out.iloc[-1] == pytest.approx(50.0)


# ============ 波动率 ============
def test_volatility_constant_is_zero():
    out = volatility(series([100.0] * 60), 30)
    assert out.dropna().abs().max() < 1e-8


def test_volatility_positive_for_random():
    rng = np.random.default_rng(7)
    s = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    out = volatility(s, 30)
    assert out.dropna().min() > 0


def test_volatility_annualized_magnitude():
    rng = np.random.default_rng(3)
    s = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    out = volatility(s, 30).dropna()
    assert out.mean() > 0.01 and out.mean() < 5.0


# ============ 动量 ============
def test_momentum_basic():
    out = momentum(series([1.0, 2.0, 3.0, 4.0, 5.0]), 2)
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(2.0)
    assert pd.isna(out.iloc[0])


def test_momentum_uptrend_positive():
    out = momentum(series(np.arange(1.0, 50.0)), 10)
    assert out.dropna().iloc[-1] > 0


# ============ 滞后收益 ============
def test_lag_returns_shifts_correctly():
    s = series([100.0, 110.0, 121.0])
    out = lag_returns(s, 1)
    assert pd.isna(out.iloc[0]) and pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(0.1)


def test_lag_returns_zero_shift_equals_pct_change():
    s = series([100.0, 110.0, 121.0])
    pd.testing.assert_series_equal(lag_returns(s, 0), s.pct_change(), check_names=False)


# ============ build_features ============
def _make_ohlcv(n=300):
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
        "amount": rng.integers(1e7, 1e8, n).astype(float),
        "turnover": rng.uniform(0.01, 0.05, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


def test_build_features_returns_dataframe():
    out = build_features(_make_ohlcv())
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 300


def test_build_features_has_expected_columns():
    out = build_features(_make_ohlcv())
    for col in ["ma_5","ma_10","ma_30","ma_60","ema_12","ema_26","rsi_14",
    "macd_dif","macd_dea","macd_hist","rsi_14", "obv", "volatility_30", "mom_10","lag_ret_1","lag_ret_5"]:
        assert col in out.columns, f"缺少因子 {col}"


def test_build_features_no_future_data():
    """防泄漏核心：改未来数据，历史因子值不应变"""
    df = _make_ohlcv(300)
    out1 = build_features(df)
    df2 = df.copy()
    df2.iloc[250:, df2.columns.get_loc("close")] *= 1.5
    out2 = build_features(df2)
    pd.testing.assert_frame_equal(out1.iloc[:240], out2.iloc[:240])


def test_build_features_does_not_mutate_input():
    df = _make_ohlcv(100)
    before = df["close"].copy()
    build_features(df)
    pd.testing.assert_series_equal(df["close"], before)