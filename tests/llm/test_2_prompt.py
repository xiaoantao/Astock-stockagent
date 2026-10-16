# -*- coding: utf-8 -*-
"""测试 2：两个 prompt   用法: python tests/llm/test_2_prompt.py [tag]"""
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

from src.llm.report import build_facts, build_system_prompt, build_user_prompt
from src.models.registry import load_model
from src.data.fetcher import fetch_daily

print("=" * 68)
print(f"测试 2: prompt   TAG={TAG}")
print("=" * 68)

model, meta = load_model(TAG)
summary_df = pd.read_parquet(PROCESS / f"summary_{TAG}.parquet")
nav = pd.read_parquet(PROCESS / f"nav_{TAG}.parquet")["nav"]
ds = pd.read_parquet(PROCESS / f"dataset_{TAG}.parquet")
df = fetch_daily(CODE, START, END)
data_info = {"n_raw": len(df), "n_valid": len(ds), "n_lost": len(df) - len(ds),
             "pos": int((ds["y"] == 1).sum()), "neg": int((ds["y"] == 0).sum()),
             "n_splits": 5, "embargo": 5}

facts = build_facts(meta, summary_df, nav, data_info)
sp = build_system_prompt()
up = build_user_prompt(facts)

print("\n---- system prompt ----\n")
print(sp)
print("\n---- user prompt ----\n")
print(up)

print("\n[验收]")
for m in ("不得", "约束", "AUC", "四、改进方向"):
    print(f"  {'[PASS]' if m in sp else '[WARN]'} system 含「{m}」")
hit = sum(1 for v in facts.values() if isinstance(v, (int, float)) and str(v) in up)
print(f"  {'[PASS]' if hit else '[FAIL]'} user 中命中 {hit} 个事实数字")
print(f"  [INFO] system {len(sp)} 字 / user {len(up)} 字")