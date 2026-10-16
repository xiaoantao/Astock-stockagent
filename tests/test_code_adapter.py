import pytest
from src.data.code_adapter import normalize, to_baostock, to_akshare, to_tushare

# (输入, 期望的 normalize 输出)
NORMALIZE_CASES = [
    ("600519",    "600519.SH"),
    ("sh.600519", "600519.SH"),
    ("sh600519",  "600519.SH"),
    ("600519.SH", "600519.SH"),
    ("600519.sh", "600519.SH"),
    ("SH.600519", "600519.SH"),
    (" 600519 ",  "600519.SH"),
    ("000001",    "000001.SZ"),
    ("sz.000001", "000001.SZ"),
    ("000001.SZ", "000001.SZ"),
    ("300750",    "300750.SZ"),
    ("688981",    "688981.SH"),
    ("601398",    "601398.SH"),
    ("002594",    "002594.SZ"),
]

@pytest.mark.parametrize("raw, expected", NORMALIZE_CASES)
def test_normalize(raw, expected):
    assert normalize(raw) == expected

@pytest.mark.parametrize("raw, expected", [
    ("600519",    "sh.600519"),
    ("sh.600519", "sh.600519"),
    ("600519.SH", "sh.600519"),
    ("000001",    "sz.000001"),
    ("300750",    "sz.300750"),
    ("688981",    "sh.688981"),
])
def test_to_baostock(raw, expected):
    assert to_baostock(raw) == expected

@pytest.mark.parametrize("raw, expected", [
    ("600519",    "600519"),
    ("sh.600519", "600519"),
    ("600519.SH", "600519"),
    ("000001.SZ", "000001"),
])
def test_to_akshare(raw, expected):
    assert to_akshare(raw) == expected

@pytest.mark.parametrize("raw, expected", [
    ("600519",    "600519.SH"),
    ("sh.600519", "600519.SH"),
    ("sh600519",  "600519.SH"),
    ("000001",    "000001.SZ"),
])
def test_to_tushare(raw, expected):
    assert to_tushare(raw) == expected

def test_idempotent():
    for raw, _ in NORMALIZE_CASES:
        once = normalize(raw)
        assert normalize(once) == once

@pytest.mark.parametrize("bad", [
    "", "   ", "12345", "1234567", "abcdef",
    "999999", "jp.600519", "600519.JP",
])
def test_invalid_raises(bad):
    with pytest.raises(ValueError):
        normalize(bad)

def test_none_raises():
    with pytest.raises((ValueError, TypeError)):
        normalize(None)