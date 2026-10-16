"""L5 编排层：节点之间传递的状态"""
from typing import TypedDict, Any, Annotated
from operator import add


class AgentState(TypedDict):
    # ---- 输入 ----
    code:str
    start:str
    end:str
    tag:str                      # "sent17" / "base15"

    # ---- 各节点产物 ----
    df:Any                 # 行情 DataFrame
    feat:Any                 # 因子 DataFrame
    model:Any
    meta:dict
    proba:Any                 # 预测概率 Series
    data_info:dict                # 样本统计
    facts:dict
    report_md:str
    fact_check:dict

    # ---- 容错 ----
    errors: Annotated[list, add]    # 累加器，不互相覆盖