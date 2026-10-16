# -*- coding: utf-8 -*-
"""测试 3：check_facts（造假测试）   用法: python tests/llm/test_3_check.py [tag]"""
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

from src.llm.report import build_facts, check_facts
from src.models.registry import load_model
from src.data.fetcher import fetch_daily

print("=" * 68)
print(f"测试 3: check_facts（造假测试）   TAG={TAG}")
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

real_key, real_val = None, None
for k, v in facts.items():
    if isinstance(v, float):
        real_key, real_val = k, v
        break
if real_val is None:
    print("[FAIL] facts 里没有 float，无法测试"); sys.exit(1)

FAKE = "0.8734"
fake_md = f"""## 一、结果解读
模型的 {real_key} 为 {real_val:.4f}，表现一般。
另外辅助指标达到 {FAKE}，说明还有提升空间。

## 二、归因分析
样本区间为 2020 年至 2026 年。

## 三、局限性
1. 样本单一
2. 因子同质化

## 四、改进方向
扩充标的池。
"""

print(f"真值: {real_key} = {real_val}")
print(f"编造: {FAKE}\n" + "-" * 68)

fc = check_facts(fake_md, facts)
print("\n返回:")
for k, v in fc.items():
    print(f"  {k}: {v}")

print("\n[验收]")
susp = fc.get("suspicious", [])
s = " ".join(str(x) for x in susp)
print(f"  {'[PASS]' if FAKE in s else '[FAIL]'} 抓出编造数字 {FAKE}")
print(f"  {'[PASS]' if f'{real_val:.4f}' not in s else '[FAIL]'} 未误报真值")
print(f"  {'[PASS]' if '2020' not in s and '2026' not in s else '[WARN]'} 未误报年份")