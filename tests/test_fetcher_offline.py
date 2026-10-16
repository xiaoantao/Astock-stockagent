# -*- coding: utf-8 -*-
"""模块 2 离线测试：不联网、不读写真实数据，秒级完成"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.fetcher import (
    AK_COLUMN_MAP,
    _cache_path,
    parse_baostock_rows,
    standardize_ak,
)


# ============ 构造假数据的工具函数 ============
def make_ak_df(rows=None):
    """模拟 AKShare stock_zh_a_hist 的原始返回（中文列名）"""
    if rows is None:
        rows = [
            ["2024-01-02", "600519", 1.0, 1.5, 1.6, 0.9, 100, 1000.0, 0.1, 0.2, 0.3, 0.01],
            ["2024-01-03", "600519", 1.5, 2.5, 2.6, 1.9, 200, 2000.0, 0.2, 0.3, 0.4, 0.02],
        ]
    return pd.DataFrame(rows, columns=[
        "日期", "股票代码", "开盘", "收盘", "最高", "最低",
        "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率",
    ])


REQUIRED_COLS = ["open", "high", "low", "close", "volume", "amount", "turnover"]


# ============ 一、_cache_path ============
def test_cache_path_returns_path_object():
    p = _cache_path("600519.SH", "20240101", "20260901", "hfq")
    assert isinstance(p, Path)


def test_cache_path_is_parquet_under_raw():
    p = _cache_path("600519.SH", "20240101", "20260901", "hfq")
    assert p.suffix == ".parquet"
    assert "raw" in str(p)


def test_cache_path_normalizes_code_formats():
    """同一只股票的不同写法必须指向同一个缓存文件"""
    a = _cache_path("600519", "20240101", "20260901", "hfq")
    b = _cache_path("sh.600519", "20240101", "20260901", "hfq")
    c = _cache_path("600519.SH", "20240101", "20260901", "hfq")
    d = _cache_path(" 600519 ", "20240101", "20260901", "hfq")
    assert a == b == c == d


def test_cache_path_differs_by_adjust():
    hfq = _cache_path("600519.SH", "20240101", "20260901", "hfq")
    qfq = _cache_path("600519.SH", "20240101", "20260901", "qfq")
    assert hfq != qfq, "后复权与前复权不能用同一个缓存文件"


def test_cache_path_differs_by_date_range():
    a = _cache_path("600519.SH", "20240101", "20260901", "hfq")
    b = _cache_path("600519.SH", "20200101", "20260901", "hfq")
    assert a != b


def test_cache_path_is_deterministic():
    a = _cache_path("600519.SH", "20240101", "20260901", "hfq")
    b = _cache_path("600519.SH", "20240101", "20260901", "hfq")
    assert a == b


def test_cache_path_no_illegal_chars():
    """Windows 文件名非法字符检查"""
    illegal = set('\\/:*?"<>|')
    for adj in ["hfq", "qfq", ""]:
        p = _cache_path("600519.SH", "20240101", "20260901", adj)
        assert not (set(p.name) & illegal), f"文件名含非法字符: {p.name}"


def test_cache_path_handles_empty_adjust():
    """adjust 为空时不能生成 '..parquet' 这种怪名字"""
    p = _cache_path("600519.SH", "20240101", "20260901", "")
    assert ".." not in p.name
    assert p.suffix == ".parquet"


# ============ 二、standardize_ak ============
def test_standardize_ak_renames_columns():
    out = standardize_ak(make_ak_df())
    for col in REQUIRED_COLS:
        assert col in out.columns, f"缺少列 {col}"


def test_standardize_ak_sets_datetime_index():
    out = standardize_ak(make_ak_df())
    assert out.index.name == "date"
    assert pd.api.types.is_datetime64_any_dtype(out.index)


def test_standardize_ak_sorts_by_date():
    rows = [
        ["2024-01-05", "600519", 3.0, 3.5, 3.6, 2.9, 300, 3000.0, 0.3, 0.4, 0.5, 0.03],
        ["2024-01-02", "600519", 1.0, 1.5, 1.6, 0.9, 100, 1000.0, 0.1, 0.2, 0.3, 0.01],
        ["2024-01-03", "600519", 2.0, 2.5, 2.6, 1.9, 200, 2000.0, 0.2, 0.3, 0.4, 0.02],
    ]
    out = standardize_ak(pd.DataFrame(rows, columns=[
        "日期", "股票代码", "开盘", "收盘", "最高", "最低",
        "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率",
    ]))
    assert out.index.is_monotonic_increasing, "日期索引必须升序"
    assert str(out.index[0].date()) == "2024-01-02"


def test_standardize_ak_numeric_dtypes():
    """输入是字符串也要转成数值"""
    rows = [["2024-01-02", "600519", "1.0", "1.5", "1.6", "0.9",
             "100", "1000.0", "0.1", "0.2", "0.3", "0.01"]]
    df = pd.DataFrame(rows, columns=[
        "日期", "股票代码", "开盘", "收盘", "最高", "最低",
        "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率",
    ])
    out = standardize_ak(df)
    for col in REQUIRED_COLS:
        assert pd.api.types.is_numeric_dtype(out[col]), f"{col} 不是数值类型"
    assert out["close"].iloc[0] == 1.5


def test_standardize_ak_does_not_mutate_input():
    """不能修改传入的原始 DataFrame（避免副作用）"""
    df = make_ak_df()
    before = df.columns.tolist()
    standardize_ak(df)
    assert df.columns.tolist() == before, "standardize_ak 不应原地修改入参"


# ============ 三、parse_baostock_rows ============
BS_FIELDS = "date,open,high,low,close,volume,amount"


def test_parse_baostock_rows_converts_types():
    rows = [
        ["2024-01-02", "1.0", "1.6", "0.9", "1.5", "100", "1000.0"],
        ["2024-01-03", "2.0", "2.6", "1.9", "2.5", "200", "2000.0"],
    ]
    out = parse_baostock_rows(rows, BS_FIELDS)
    assert len(out) == 2
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        assert pd.api.types.is_numeric_dtype(out[col]), f"{col} 未转成数值"
    assert out["close"].iloc[0] == 1.5


def test_parse_baostock_rows_sets_datetime_index():
    rows = [["2024-01-02", "1.0", "1.6", "0.9", "1.5", "100", "1000.0"]]
    out = parse_baostock_rows(rows, BS_FIELDS)
    assert out.index.name == "date"
    assert pd.api.types.is_datetime64_any_dtype(out.index)


def test_parse_baostock_empty_returns_empty():
    out = parse_baostock_rows([], BS_FIELDS)
    assert len(out) == 0
    assert "close" in out.columns


def test_parse_baostock_handles_missing_turnover():
    """Baostock 不返回换手率，schema 要能对齐"""
    out = parse_baostock_rows(
        [["2024-01-02", "1.0", "1.6", "0.9", "1.5", "100", "1000.0"]],
        BS_FIELDS,
    )
    if "turnover" in out.columns:
        assert pd.api.types.is_numeric_dtype(out["turnover"])


# ============ 四、AK_COLUMN_MAP 常量 ============
def test_ak_column_map_covers_required():
    mapped = set(AK_COLUMN_MAP.values())
    for col in REQUIRED_COLS:
        assert col in mapped, f"AK_COLUMN_MAP 未覆盖 {col}"