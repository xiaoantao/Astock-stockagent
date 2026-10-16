# -*- coding: utf-8 -*-
"""
L7 可视化：Streamlit 单页总览
用法：streamlit run src/app/streamlit_app.py
"""
import sys
from pathlib import Path

# ★ 文件在 src/app/ 下，parents[2] 才是项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.agent.graph import run

PROCESS = PROJECT_ROOT / "data" / "process"


# ============================================================
# 兜底读盘（State 里没有的，才从文件读）
# ============================================================
def _load_nav(tag: str):
    """读净值曲线"""
    p = PROCESS / f"nav_{tag}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    col = "nav" if "nav" in df.columns else df.columns[0]
    s = df[col]
    if "date" in df.columns:
        s.index = pd.to_datetime(df["date"])
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _load_summary(tag: str) -> dict:
    """读回测指标，兜底用"""
    p = PROCESS / f"summary_{tag}.parquet"
    if not p.exists():
        return {}
    sm = pd.read_parquet(p)
    out = {}
    for k in sm.index:
        row = sm.loc[k]
        try:
            v = row["value"] if "value" in sm.columns else row.iloc[0]
            out[str(k)] = float(v)
        except Exception:
            pass
    return out


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(page_title="A股投研 Agent", layout="wide")
st.title("A股量化投研智能体 StockAgent")


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.header("参数")
    code  = st.text_input("股票代码", value="600519.SH")
    start = st.text_input("开始日期", value="20200101")
    end   = st.text_input("结束日期", value="20260901")
    tag   = st.selectbox("模型版本", options=["sent17", "base15"])
    go    = st.button("运行分析", type="primary")


# ============================================================
# 运行（★ 缓存，避免每次交互重跑 30 秒）
# ============================================================
@st.cache_data(show_spinner="正在运行七节点工作流...")
def _run(code: str, start: str, end: str, tag: str) -> dict:
    return run(code=code, start=start, end=end, tag=tag)


if go:
    with st.spinner("正在运行七节点工作流..."):
        st.session_state["state"] = _run(code, start, end, tag)
    st.session_state["done"] = True

if not st.session_state.get("done"):
    st.info("请在左侧点击「运行分析」")
    st.stop()

state = st.session_state["state"]


# ============================================================
# ① 节点错误（★ 有错也要显示，不白屏）
# ============================================================
errs = state.get("errors") or []
if errs:
    with st.expander(f"⚠️ 节点错误 {len(errs)} 条", expanded=False):
        for e in errs:
            st.text(str(e))


# ============================================================
# ② 指标卡片
# ============================================================
facts = state.get("facts") or {}
if not facts:
    facts = _load_summary(tag)          # 兜底

st.subheader("核心指标")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("AUC",     f"{facts.get('auc', 0):.4f}")
c2.metric("IC",      f"{facts.get('ic', 0):.4f}")
c3.metric("年化收益", f"{facts.get('ann_ret', 0):.2%}")
c4.metric("夏普比率", f"{facts.get('sharpe', 0):.4f}")
c5.metric("最大回撤", f"{facts.get('max_drawdown', 0):.2%}")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("期末净值",  f"{facts.get('nav_final', 0):.4f}")
c7.metric("年化波动",  f"{facts.get('ann_vol', 0):.2%}")
c8.metric("特征数",    f"{int(facts.get('n_features', 0))}")
c9.metric("启用情绪",   "是" if facts.get("with_sentiment") else "否")
c10.metric("折数/Embargo",
           f"{int(facts.get('n_splits', 0))} / {int(facts.get('embargo', 0))}")


# ============================================================
# ③ 净值曲线
# ============================================================
st.subheader("净值曲线")
nav = _load_nav(tag)
if nav is not None and len(nav) > 0:
    st.line_chart(nav)
    st.caption(f"区间：{nav.index.min().date()} ~ {nav.index.max().date()}"
               f"期末：{nav.iloc[-1]:.4f}")
else:
    st.warning("无净值数据")


# ============================================================
# ④ 样本分布
# ============================================================
st.subheader("样本分布")
di = state.get("data_info") or {}

if di:
    st.write(f"原始 **{di.get('n_raw')}** → 有效 **{di.get('n_valid')}**"
             f"（丢失 **{di.get('n_lost')}**)|"
             f"训练区间：{facts.get('train_range')}")

    pos, neg = int(di.get("pos", 0)), int(di.get("neg", 0))
    dist = pd.DataFrame({"样本数": [pos, neg]}, index=["涨(1)", "跌(0)"])
    st.bar_chart(dist)

    left, right = st.columns(2)
    left.metric("正样本 pos", pos)
    right.metric("负样本 neg", neg)
else:
    st.warning("State 中没有 data_info")


# ============================================================
# ⑤ 预测概率（若 State 里有）
# ============================================================
proba = state.get("proba")
if proba is not None and len(proba) > 0:
    with st.expander("预测概率序列", expanded=False):
        st.line_chart(proba)
        st.caption(f"长度：{len(proba)}　均值：{proba.mean():.4f}")


# ============================================================
# ⑥ LLM 研报
# ============================================================
st.subheader("LLM 研报")
md = state.get("report_md") or ""
if md:
    st.markdown(md)

    fc = state.get("fact_check") or {}
    st.caption(f"数字校验：total={fc.get('total')}"
               f"pass={fc.get('pass')}"
               f"疑似幻觉={len(fc.get('suspicious', []))}")
    susp = fc.get("suspicious") or []
    if susp:
        with st.expander("疑似幻觉明细", expanded=False):
            st.write(susp)
else:
    st.warning("未生成研报")


# ============================================================
# ⑦ 底部提示
# ============================================================
st.divider()
st.caption("L1 数据接入 → L2 因子工程 → L3 数据集 → L4 模型训练 → "
           "L5 LangGraph 编排 → L6 LLM 解读 → L7 可视化")