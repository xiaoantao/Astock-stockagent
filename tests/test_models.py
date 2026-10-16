# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from src.models.backtest import backtest, metrics
from src.models.train import get_X_y, compute_ic


def _mk_dataset(n=300, seed=1):
    """构造一个带 y 和若干特征列的数据集"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    X = pd.DataFrame({
        "ma_5": close, "ma_30": close * 1.01, "rsi_14": rng.uniform(30, 70, n),
        "mom_10": rng.normal(0, 1, n),
    }, index=idx)
    fwd = close * (1 + rng.normal(0, 0.01, n)) - close
    data = X.copy()
    data["fwd_ret"] = rng.normal(0, 0.01, n)
    data["y"] = (data["fwd_ret"] > 0).astype(int)
    return data


# ============ get_X_y ============
def test_get_X_y_excludes_label_columns():
    data = _mk_dataset()
    X, y = get_X_y(data)
    assert "y" not in X.columns
    assert "fwd_ret" not in X.columns, "fwd_ret 是答案，绝不能进 X"


def test_get_X_y_shapes():
    data = _mk_dataset()
    X, y = get_X_y(data)
    assert len(X) == len(y)
    assert set(y.unique()) <= {0, 1}


# ============ IC ============
def test_ic_perfect_positive():
    """预测值与实际收益完全同序 → IC = 1"""
    fwd = pd.Series([0.01, 0.02, 0.03, 0.04])
    proba = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert compute_ic(fwd, proba) == pytest.approx(1.0, abs=1e-6)


def test_ic_perfect_negative():
    fwd = pd.Series([0.04, 0.03, 0.02, 0.01])
    proba = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert compute_ic(fwd, proba) == pytest.approx(-1.0, abs=1e-6)


def test_ic_range():
    rng = np.random.default_rng(0)
    fwd = pd.Series(rng.normal(0, 0.01, 100))
    proba = pd.Series(rng.uniform(0, 1, 100))
    ic = compute_ic(fwd, proba)
    assert -1 <= ic <= 1


# ============ 回测 ============
def test_backtest_returns_series():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    fwd = pd.Series(np.zeros(100), index=idx)
    proba = pd.Series(np.ones(100), index=idx)
    nav = backtest(fwd, proba)
    assert isinstance(nav, pd.Series)
    assert len(nav) == 100


def test_backtest_zero_return_gives_flat_nav():
    """收益恒为 0 且无成本 → 净值恒为 1"""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    fwd = pd.Series(np.zeros(100), index=idx)
    proba = pd.Series(np.ones(100), index=idx)
    nav = backtest(fwd, proba, threshold=0.5, cost_bp=0)
    assert nav.iloc[-1] == pytest.approx(1.0)


def test_backtest_cost_reduces_nav():
    """计入成本后，净值必须低于不计成本"""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(2)
    fwd = pd.Series(rng.normal(0.001, 0.01, 100), index=idx)
    proba = pd.Series(rng.uniform(0, 1, 100), index=idx)
    nav_free = backtest(fwd, proba, threshold=0.5, cost_bp=0)
    nav_cost = backtest(fwd, proba, threshold=0.5, cost_bp=10)
    assert nav_cost.iloc[-1] < nav_free.iloc[-1]


def test_backtest_position_uses_same_day_signal():
    """
    修正后的语义：proba[t] 预测 fwd_ret[t]（t→t+1），
    因此全持仓信号下，0 成本净值应等于买入持有曲线。
    """
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    rng = np.random.default_rng(0)
    fwd = pd.Series(rng.normal(0, 0.01, 50), index=idx)
    proba = pd.Series(np.ones(50), index=idx)      # 全持仓

    nav = backtest(fwd, proba, threshold=0.5, cost_bp=0)
    expected = (1 + fwd).cumprod()
    pd.testing.assert_series_equal(nav, expected, check_names=False)


def test_backtest_zero_cost_full_position_tracks_buyhold():
    """全持仓 + 0 成本 = 买入持有（验证成本与仓位计算正确）"""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(1)
    fwd = pd.Series(rng.normal(0.0005, 0.01, 100), index=idx)
    proba = pd.Series(np.ones(100), index=idx)
    nav = backtest(fwd, proba, threshold=0.5, cost_bp=0)
    assert nav.iloc[-1] == pytest.approx((1 + fwd).prod(), rel=1e-9)


# ============ 回测指标 ============
def test_metrics_has_required_keys():
    idx = pd.date_range("2024-01-01", periods=252, freq="D")
    nav = pd.Series(np.linspace(1.0, 1.2, 252), index=idx)
    m = metrics(nav)
    for k in ["ann_ret", "ann_vol", "sharpe", "max_drawdown"]:
        assert k in m, f"缺少指标 {k}"


def test_metrics_max_drawdown_range():
    idx = pd.date_range("2024-01-01", periods=252, freq="D")
    nav = pd.Series(np.linspace(1.0, 1.2, 252), index=idx)
    m = metrics(nav)
    assert 0 <= m["max_drawdown"] <= 1


def test_metrics_uptrend_positive_return():
    idx = pd.date_range("2024-01-01", periods=252, freq="D")
    nav = pd.Series(np.linspace(1.0, 1.3, 252), index=idx)
    m = metrics(nav)
    assert m["ann_ret"] > 0


def test_metrics_flat_nav_zero_vol():
    """净值恒定 → 波动率 0，回撤 0"""
    idx = pd.date_range("2024-01-01", periods=252, freq="D")
    nav = pd.Series(np.ones(252), index=idx)
    m = metrics(nav)
    assert m["ann_vol"] == pytest.approx(0.0, abs=1e-8)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-8)


def test_metrics_known_drawdown():
    """净值 1 → 1.5 → 0.75，最大回撤应为 50%"""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    nav = pd.Series([1.0, 1.5, 0.75], index=idx)
    m = metrics(nav)
    assert m["max_drawdown"] == pytest.approx(0.5, abs=1e-6)