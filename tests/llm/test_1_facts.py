# -*- coding: utf-8 -*-
"""测试 1：build_facts   用法: python tests/llm/test_1_facts.py [tag]"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CODE, START, END = "600519.SH", "20200101", "20260901"
PROCESS = PROJECT_ROOT / "data" / "process"
TAG = sys.argv[1] if len(sys.argv) > 1 else "sent17"

from src.llm.report import build_facts
from src.models.registry import load_model
from src.data.fetcher import fetch_daily

print("=" * 68)
print(f"测试 1: build_facts   TAG={TAG}")
print("=" * 68)

model, meta = load_model(TAG)
summary_df = pd.read_parquet(PROCESS / f"summary_{TAG}.parquet")
nav = pd.read_parquet(PROCESS / f"nav_{TAG}.parquet")["nav"]
ds = pd.read_parquet(PROCESS / f"dataset_{TAG}.parquet")
df = fetch_daily(CODE, START, END)

data_info = {
    "n_raw": len(df), "n_valid": len(ds), "n_lost": len(df) - len(ds),
    "pos": int((ds["y"] == 1).sum()), "neg": int((ds["y"] == 0).sum()),
    "n_splits": 5, "embargo": 5,
}
print(f"data_info: {data_info}\n")

facts = build_facts(meta, summary_df, nav, data_info)

print(f"返回类型: {type(facts).__name__}   字段数: {len(facts)}")
print("-" * 68)
for k, v in facts.items():
    print(f"  {k:<18} = {v!r:<22} ({type(v).__name__})")

print("\n[验收]")
print(f"  {'[PASS]' if isinstance(facts, dict) else '[FAIL]'} 返回 dict")
bad = [k for k, v in facts.items()
       if type(v).__name__.startswith(("int64", "float64", "ndarray"))]
print(f"  {'[WARN] ' + str(bad) if bad else '[PASS]'} 无 numpy 原生类型")
for key in ("auc", "ic"):
    print(f"  {'[PASS]' if key in facts else '[WARN]'} 含字段 {key}"
          + (f" = {facts[key]}" if key in facts else ""))