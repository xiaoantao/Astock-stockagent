import time
import traceback
import pandas as pd
from pathlib import Path
from langgraph.graph import StateGraph, END

from .state import AgentState
from src.data.acquire_data import generate_data_info
from src.config import PROCESSED_DIR,PROCESS_DIR
from src.data.notices import fetch_notices
from src.data.fetcher import fetch_daily
from src.features.sentiment import score_notices,build_sentiment_features
from src.features.indicators import build_features
from src.models.registry import load_model, predict_proba
from src.llm.report import build_facts, generate_report
#========节点一:拉行情==========
def fetch_data(state:AgentState):
    try:
        df = fetch_daily(state["code"],state["start"],state["end"],adjust="hfq",use_cache=True)
        if df is None or len(df) == 0:
            return {"errors":["fetch_data:行情为空!"]}
        return {"df":df}
    except Exception as e:
        return {"errors":[f"fetch_data:调用失败:{e}!\n{traceback.format_exc()}"]}
#========节点二:算因子===========
def build_feature(state:AgentState):
    try:
        feat = build_features(state["df"])
        notice_df = fetch_notices(state["code"],state["start"],state["end"],limit=1000,use_cache=True)
        sent_df = score_notices(notice_df,limit=1000)
        sentiment = build_sentiment_features(sent_df,notice_df,index=state["df"].index)
        feat = pd.merge(feat,sentiment,on=state["df"].index,how="left")
        return {"feat":feat}
    except Exception as e:
        return {"errors":[f"build_feature失败:{e}"]}
#==========节点三:加载模型，只加载不训练========
def load_model_node(state:AgentState):
    try:
        model,meta = load_model(state["tag"])
        return {"model":model,"meta":meta}
    except Exception as e:
        return {"errors":[f"load_model_node调用失败{e}!"]}
#=========节点四:推理======================
def predict(state:AgentState):
    try:
        arr = predict_proba(state["model"],state["meta"],state["feat"])
        proba = pd.Series(data=arr,index=state["feat"].index)
        return {"proba":proba}
    except Exception as e:
        return {"errors":[f"predict方法调用失败{e}!"]}
#========节点五:造data_info=============
def build_data_info(state:AgentState):
    try:
        data_df = generate_data_info(state["code"],state["start"],state["end"],adjust="hfq",use_cache=True)
        summary_path = f"{PROCESS_DIR}/summary_sent17.parquet"
        nav_path = f"{PROCESS_DIR}/nav_sent17.parquet"
        summary_p = Path(summary_path)
        nav_p = Path(nav_path)
        summary_df = pd.read_parquet(summary_p)
        nav_df = pd.read_parquet(nav_p)
        data_info = build_facts(state["meta"],summary_df,nav_df,data_df)
        return {"data_info":data_info}
    except Exception as e:
        return {"errors":[f"build_data_info无法调用{e}!"]}

#==========节点六:LLM研报===============
def llm_report(state:AgentState):
    try:
        md,fc = generate_report(state["tag"])
        if not md:
            return {"errors":[f"生成文本内容为空!"]}
        return {"report_md":md,"fact_check":fc}
    except Exception as e:
        return {"errors":[f"llm_report失败:{e}"]}

#============节点七:汇总================
def summarize(state:AgentState):
    out = {"errors":[f"[汇总]流程结束,共{len(state.get('errors',[]))}个错误!"]}
    print(state.get("report_md"))
    return out

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("fetch_data",fetch_data)
    g.add_node("build_feature",build_feature)
    g.add_node("load_model",load_model_node)
    g.add_node("predict",predict)
    g.add_node("build_data_info",build_data_info)
    g.add_node("llm_report",llm_report)
    g.add_node("summarize",summarize)
    g.set_entry_point("fetch_data")

    g.add_edge("fetch_data","build_feature")
    g.add_edge("build_feature","load_model")
    g.add_edge("load_model","predict")
    g.add_edge("predict","build_data_info")
    g.add_edge("build_data_info","llm_report")
    g.add_edge("llm_report","summarize")
    g.add_edge("summarize",END)
    return g.compile()


def run(code="600519.SH",start="20200101",end="20260901",tag="sent_17"):
    app = build_graph()
    return app.invoke({"code":code,"start":start,"end":end,"tag":tag,"errors":[]})
