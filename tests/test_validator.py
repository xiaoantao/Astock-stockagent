# -*- coding: utf-8 -*-
"""双源一致性校验测试"""
import pandas as pd
from src.data.validator import compare_sources


def _df(prices, dates=None):
    if dates is None:
        dates = pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 2 + len(prices))])
    return pd.DataFrame({"close": prices}, index=dates)


def test_identical_sources_are_consistent():
    a = _df([100.0, 101.0, 102.0])
    b = _df([100.0, 101.0, 102.0])
    r = compare_sources(a, b)
    assert r["matched"] == 3
    assert r["diff_ratio"] == 0.0
    assert r["message"] == "一致"


def test_tiny_float_error_is_tolerated():
    """浮点误差不应被判为不一致"""
    a = _df([100.0, 101.0])
    b = _df([100.0 + 1e-9, 101.0 - 1e-9])
    r = compare_sources(a, b)
    assert r["diff_ratio"] == 0.0


def test_big_diff_is_detected():
    a = _df([100.0, 101.0])
    b = _df([100.0, 105.0])
    r = compare_sources(a, b)
    assert r["diff_ratio"] > 0
    assert "差异" in r["message"]


def test_partial_diff_ratio_is_fraction():
    """3 天里有 1 天不同 → 差异率应约 1/3"""
    a = _df([100.0, 101.0, 102.0])
    b = _df([100.0, 105.0, 102.0])
    r = compare_sources(a, b)
    assert abs(r["diff_ratio"] - 1 / 3) < 1e-6


def test_no_overlap_returns_zero_matched():
    a = _df([100.0], dates=pd.to_datetime(["2024-01-02"]))
    b = _df([100.0], dates=pd.to_datetime(["2025-01-02"]))
    r = compare_sources(a, b)
    assert r["matched"] == 0
    assert r["message"] == "两个源的日期无交集"


def test_report_has_required_keys():
    r = compare_sources(_df([1.0]), _df([1.0]))
    for key in ["matched", "diff_ratio", "max_abs_diff", "message"]:
        assert key in r, f"报告缺少字段 {key}"