# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from src.features.dataset import make_label, build_dataset, walk_forward_split


def _mk(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
        "amount": rng.integers(1e7, 1e8, n).astype(float),
        "turnover": rng.uniform(0.01, 0.05, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


# ============ 标签 ============
def test_label_is_binary():
    out = make_label(_mk())
    assert set(out["y"].dropna().unique()) <= {0, 1}


def test_label_uses_next_day_return():
    """标签必须等于"明日收盘 > 今日收盘"，最后一天无明日 → NaN"""
    df = _mk(50)
    out = make_label(df)

    fwd = df["close"].shift(-1) / df["close"] - 1
    expected = (fwd > 0).astype(float)
    expected[fwd.isna()] = np.nan          # ← 最后一天显式置 NaN

    pd.testing.assert_series_equal(
        out["y"].dropna(),
        expected.dropna(),
        check_names=False,
        check_dtype=False,
    )


def test_label_last_row_is_nan():
    """最后一天没有"明天"，标签必须是 NaN（不能编造）"""
    out = make_label(_mk(30))
    assert pd.isna(out["y"].iloc[-1])


def test_label_has_fwd_ret():
    out = make_label(_mk())
    assert "fwd_ret" in out.columns


def test_label_does_not_mutate_input():
    df = _mk(50)
    before = df["close"].copy()
    make_label(df)
    pd.testing.assert_series_equal(df["close"], before)


# ============ 数据集 ============
def test_build_dataset_shapes_match():
    from src.features.indicators import build_features
    df = _mk(300)
    feat = build_features(df)
    ds = build_dataset(df, feat)
    assert len(ds) > 0
    assert "y" in ds.columns


def test_build_dataset_no_nan_after_dropna():
    from src.features.indicators import build_features
    df = _mk(300)
    ds = build_dataset(df, build_features(df))
    assert not ds.isna().any().any(), "dropna 后不应还有 NaN"


def test_build_dataset_shorter_than_input():
    """对齐 + dropna 后行数必然减少"""
    from src.features.indicators import build_features
    df = _mk(300)
    ds = build_dataset(df, build_features(df))
    assert len(ds) < len(df)


def test_build_dataset_no_future_leakage():
    """
    核心防泄漏测试：
    把最后 50 天的收盘价改掉，改动点之前的特征不应变化
    """
    from src.features.indicators import build_features
    df1 = _mk(300)
    ds1 = build_dataset(df1, build_features(df1))

    MOD_POS = 250                      # 从这里开始改收盘价
    df2 = df1.copy()
    df2.iloc[MOD_POS:, df2.columns.get_loc("close")] *= 1.5
    ds2 = build_dataset(df2, build_features(df2))

    # 关键修正：只比较"改动日之前"的行
    # 因子 shift(1) 后，改动会影响第 MOD_POS+1 行，
    # 所以严格取 < df1.index[MOD_POS] 的行，留足安全边界
    cutoff = df1.index[MOD_POS]
    common = ds1.index.intersection(ds2.index)
    common = common[common < cutoff]

    assert len(common) > 0, "比较区间为空，测试无效"

    feat_cols = [c for c in ds1.columns if c not in ("y", "fwd_ret")]
    pd.testing.assert_frame_equal(
        ds1.loc[common, feat_cols],
        ds2.loc[common, feat_cols],
    )

def test_build_dataset_feature_shifted():
    """
    验证因子确实后移了一格：
    第 t 行的 ma_5，应该等于原始第 t-1 行的 ma_5
    """
    from src.features.indicators import build_features
    df = _mk(300)
    raw_feat = build_features(df)
    ds = build_dataset(df, raw_feat)

    # 取一个有效日期
    idx = ds.index[10]
    pos = raw_feat.index.get_loc(idx)
    prev = raw_feat.index[pos - 1]

    assert ds.loc[idx, "ma_5"] == pytest.approx(raw_feat.loc[prev, "ma_5"])


# ============ 时序切分 ============
def test_split_returns_train_test_pairs():
    df = _mk(300)
    from src.features.indicators import build_features
    ds = build_dataset(df, build_features(df))
    splits = list(walk_forward_split(ds, n_splits=5))
    assert len(splits) > 0
    for tr, te in splits:
        assert len(tr) > 0 and len(te) > 0


def test_split_train_before_test():
    """训练集的时间必须全部早于测试集"""
    from src.features.indicators import build_features
    ds = build_dataset(_mk(300), build_features(_mk(300)))
    for tr, te in walk_forward_split(ds, n_splits=5):
        assert tr.index.max() < te.index.min(), "训练集必须早于测试集"


def test_split_has_embargo_gap():
    """训练末尾与测试开头之间必须留 embargo 间隔"""
    from src.features.indicators import build_features
    ds = build_dataset(_mk(300), build_features(_mk(300)))
    for tr, te in walk_forward_split(ds, n_splits=5, embargo=5):
        gap = ds.index.get_loc(te.index.min()) - ds.index.get_loc(tr.index.max())
        assert gap > 1, f"应有 embargo 间隔，实际 gap={gap}"


def test_split_no_overlap():
    """训练与测试的索引不能重叠"""
    from src.features.indicators import build_features
    ds = build_dataset(_mk(300), build_features(_mk(300)))
    for tr, te in walk_forward_split(ds, n_splits=5):
        assert len(tr.index.intersection(te.index)) == 0


def test_split_covers_most_data():
    """切分应覆盖大部分数据，不能漏太多"""
    from src.features.indicators import build_features
    ds = build_dataset(_mk(300), build_features(_mk(300)))
    covered = set()
    for tr, te in walk_forward_split(ds, n_splits=5):
        covered |= set(te.index)
    assert len(covered) > len(ds) * 0.3