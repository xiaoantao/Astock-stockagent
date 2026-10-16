import pandas as pd
from src.config import PROCESSED_DIR
from src.llm.client import chat_json
def score_notices(notice_df : pd,limit : int):
    system_prompt = f"""你是资深 A 股分析师。
    请判断公告标题对股价的情绪倾向。
    只输出 JSON,不要任何解释文字,不要 markdown 代码块。"""
    
    user_template = (
    "【任务】判断下面这则 A 股上市公司公告对股价的情绪倾向。\n"
    "\n"
    "【公告信息】\n"
    "股票：{name}({code})\n"
    "公告日期：{date}\n"
    "公告标题：{title}\n"
    "\n"
    "【输出格式】只输出一行 JSON,不要任何解释文字、不要 markdown 代码块：\n"
    '{{"score": <浮点数>, "label": "<字符串>", "reason": "<字符串>", "confidence": <浮点数>}}\n'
    "\n"
    "【字段定义】\n"
    "- score:情绪强度,取值范围 -1.0 ~ 1.0。\n"
    "  利好为正、利空为负、中性为 0。绝对值越大表示影响越强。\n"
    "- label:只能取「利好」「利空」「中性」三者之一。\n"
    "- reason:判断依据,不超过 30 个汉字。\n"
    "- confidence:你对本次判断的把握程度,0.0 ~ 1.0。\n"
    "\n"
    "【评分标准】\n"
    "利好(score 0.3 ~ 1.0):\n"
    "  大额分红派息、股份回购、业绩预增/超预期、重大合同中标、\n"
    "  股东或高管增持、股权激励、重大资产重组利好、新产品获批。\n"
    "  金额越大、超预期幅度越大,score 越接近 1.0。\n"
    "\n"
    "利空J(score -1.0 ~ -0.3):\n"
    "  业绩预减/亏损、股东或高管减持、诉讼仲裁、监管处罚或问询、\n"
    "  重大合同违约、资产减值、核心人员离职、退市风险警示。\n"
    "  影响越严重,score 越接近 -1.0。\n"
    "\n"
    "中性(score -0.2 ~ 0.2):\n"
    "  董事会/监事会/股东会决议公告、公司章程或管理制度修订、\n"
    "  人事变动（非核心岗位）、日常经营公告、会议通知、\n"
    "  定期报告的例行披露、担保额度、关联交易的常规说明。\n"
    "\n"
    "【重要提醒】\n"
    "1. 大量上市公司公告属于程序性、例行性披露，\n"
    "   这类公告对股价几乎无影响,必须判为「中性」,score 取 0.0。\n"
    "2. 不要因为公告标题看起来正面就判利好——\n"
    "   只有真正影响未来现金流或市场预期的信息才算利好。\n"
    "3. 标题信息不足时，倾向判「中性」并降低 confidence,不要臆测。\n"
)
    rows = []
    fail_count = 0
    columns = ["notice_id","score","label","reason","confidence"]
    filename = f"sentiment.parquet"
    file_path = PROCESSED_DIR / filename
    if file_path.exists():
        df_score = pd.read_parquet(file_path)
        return df_score
    else:
        df_score = pd.DataFrame(columns=columns)
    wait_score = notice_df.loc[~notice_df["notice_id"].astype(str).isin(df_score["notice_id"].astype(str))]
    
    for _,row in wait_score.iterrows():
        user = user_template.format(
            name=row["name"],
            code=row["code"],
            date=row["date"],
            title=row["title"]
        )
        try:

            result = chat_json(system_prompt,user,cache_key=row["notice_id"],temperature=0.0,use_cache=True)
            score = float(result.get("score"))
            label = result.get("label")
            reason = result.get("reason")
            confidence = result.get("confidence")
            if score > 1 or score < -1:
                score = 0.0
            if label not in ("利好","利空","中性"):
                label = None
            rows.append([row["notice_id"],score,label,reason,confidence])
        except Exception as e:
            fail_count += 1
            print(f"[失败]{row['notice_id']}:{type(e).__name__}:{str(e)[:120]}")
            if fail_count > 3:
                print(f"失败3次,疑似限流/停止输出!")
                break
            continue
        if len(rows) % 5 == 0:
            new = pd.concat([df_score,pd.DataFrame(rows,columns=columns)],ignore_index=True)
            new = new.drop_duplicates("notice_id")
            new.to_parquet(file_path)
        new_df = pd.concat([df_score,pd.DataFrame(rows,columns=columns)],ignore_index=True)
        new_df = new_df.drop_duplicates("notice_id")
        new_df.to_parquet(file_path).head(limit)
    return new_df



def build_sentiment_features(sent_df : pd.DataFrame,notice_df : pd.DataFrame,index : pd.DatetimeIndex):
    df = sent_df.merge(notice_df[["notice_id","date"]],on="notice_id",how="left")
    df["date"] = pd.to_datetime(df["date"],errors="coerce")
    df = df.dropna(subset=["date","score"])
    daily = df.groupby("date")["score"].mean().sort_index()
    pos = index.searchsorted(daily.index,side="left")
    valid = pos < len(index)
    mapped = pd.Series(daily.values[valid],index=index[pos[valid]])
    mapped = mapped.groupby(level=0).mean()
    sentiment = mapped.reindex(index).fillna(0.0)
    sentiment_ma5 = sentiment.rolling(5).mean().fillna(0.0)
    out = pd.DataFrame({"sentiment":sentiment,"sentiment_ma5":sentiment_ma5},index=index)
    return out



    

    
    

