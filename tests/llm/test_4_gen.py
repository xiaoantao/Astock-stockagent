# -*- coding: utf-8 -*-
"""测试 4：generate_report（★ 调 GLM 花钱）   用法: python tests/llm/test_4_gen.py [tag]"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TAG = sys.argv[1] if len(sys.argv) > 1 else "sent17"

from src.llm.report import generate_report

print("=" * 68)
print(f"测试 4: generate_report（★ 调用 GLM）   TAG={TAG}")
print("=" * 68)

if input("确认调用 GLM？输入 y 继续: ").strip().lower() != "y":
    print("已取消"); sys.exit(0)

md, fc = generate_report(TAG)

print("\n---- 生成的报告 ----\n")
print(md)
print("\n---- 数字校验 ----")
for k, v in fc.items():
    print(f"  {k}: {v}")

print("\n[验收]")
for sec in ("一、结果解读", "二、归因分析", "三、局限性", "四、改进方向"):
    print(f"  {'[PASS]' if sec in md else '[FAIL]'} 含小节「{sec}」")
neg = any(w in md for w in ("无预测能力", "随机", "不具备", "无实质差异", "亏损"))
print(f"  {'[PASS]' if neg else '[FAIL]'} 承认负面结论（★ 硬标准）")
print(f"  [INFO] suspicious: {len(fc.get('suspicious', []))}")