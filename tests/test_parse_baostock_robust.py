# tests/test_parse_baostock_robust.py
# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from src.data.fetcher import parse_baostock_rows

BS_FIELDS = "date,open,high,low,close,volume,amount"
ROW = ["2024-01-02", "1.0", "1.6", "0.9", "1.5", "100", "1000.0"]


def test_returns_dataframe():
    out = parse_baostock_rows([ROW], BS_FIELDS)
    assert isinstance(out, pd.DataFrame)


def test_required_columns_present():
    """这些列必须在（Baostock 源会补齐 turnover）"""
    out = parse_baostock_rows([ROW], BS_FIELDS)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        assert col in out.columns, f"缺少列 {col}"


def test_numeric_dtypes():
    """数值列必须是 float/int，不能还是字符串"""
    out = parse_baostock_rows([ROW], BS_FIELDS)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        assert pd.api.types.is_numeric_dtype(out[col]), f"{col} 仍是 {out[col].dtype}"


def test_close_value_correct():
    out = parse_baostock_rows([ROW], BS_FIELDS)
    assert float(out["close"].iloc[0]) == 1.5


def test_index_is_datetime():
    out = parse_baostock_rows([ROW], BS_FIELDS)
    assert pd.api.types.is_datetime64_any_dtype(out.index)


def test_sorted_ascending():
    rows = [
        ["2024-01-05", "3.0", "3.6", "2.9", "3.5", "300", "3000.0"],
        ["2024-01-02", "1.0", "1.6", "0.9", "1.5", "100", "1000.0"],
    ]
    out = parse_baostock_rows(rows, BS_FIELDS)
    assert out.index.is_monotonic_increasing, "日期必须升序"


def test_empty_rows_no_crash():
    """空输入不能崩"""
    out = parse_baostock_rows([], BS_FIELDS)
    assert len(out) == 0


def test_empty_string_becomes_nan():
    """空字符串应转 NaN 而不是崩"""
    out = parse_baostock_rows(
        [["2024-01-02", "1.0", "1.6", "0.9", "1.5", "", "1000.0"]],
        BS_FIELDS,
    )
    assert pd.isna(out["volume"].iloc[0]), "空值应为 NaN"


def test_turn_field_renamed_if_present():
    """传 turn 字段时，若实现支持改名则校验；不支持则跳过"""
    out = parse_baostock_rows(
        [["2024-01-02", "1.0", "1.6", "0.9", "1.5", "100", "1000.0", "0.12"]],
        BS_FIELDS + ",turn",
    )
    # 只断言不崩，turnover 的具体处理由你的实现决定
    assert isinstance(out, pd.DataFrame)


def test_schema_matches_akshare():
    """两个源的列集合必须一致（这是 schema 契约的核心）"""
    from src.data.fetcher import standardize_ak

    ak_df = standardize_ak(pd.DataFrame(
        [["2024-01-02", "600519", 1.0, 1.5, 1.6, 0.9, 100, 1000.0, 0.1, 0.2, 0.3, 0.01]],
        columns=["日期", "股票代码", "开盘", "收盘", "最高", "最低",
                 "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"],
    ))
    bs_df = parse_baostock_rows([ROW], BS_FIELDS)
    assert set(ak_df.columns) == set(bs_df.columns), (
        f"列不一致\nAK: {sorted(ak_df.columns)}\nBS: {sorted(bs_df.columns)}"
    )