import time
import pandas as pd
import baostock as bs
from src import config
from pathlib import Path
import akshare as ak
from src.features import indicators
from src.data.fetcher import fetch_daily 
from src.data.code_adapter import to_akshare
from tenacity import (retry,stop_after_attempt,wait_exponential)
def fetch_notices(code : str,start : str,end : str,limit : int,use_cache=True):
    hits = []
    columns_name = {"代码":"code","名称":"name","公告标题":"title",
    "公告类型":"type","公告日期":"date","网址":"url"}
    num_code = to_akshare(code)
    if "-" in start:
        start = start.replace("-","")
    if "-" in end:
        end = end.replace("-","")
    filename = f"{num_code}_{start}_{end}_notices.parquet"
    file_path = config.RAW_DIR / filename
    p = Path(file_path)
    if use_cache and p.exists():
        return pd.read_parquet(p)
    else:
        date_df = fetch_daily(num_code,start,end)
        fmt_date = date_df.index
        fail_count = 0
        for date in fmt_date:
            try:
                if fail_count > 3:break
                str_date = date.strftime("%Y%m%d")
                report_df = ak.stock_notice_report(symbol="全部",date=str_date)
                if report_df is None or len(report_df) == 0:
                    continue
                report_df = report_df.rename(columns=columns_name)
                sub = report_df[report_df["code"].astype(str).str.strip() == str(num_code)]
                if len(sub) > 0:
                    hits.append(sub)
            except Exception as e:
                    fail_count += 1
                    print(f"{date}_{type(e)}")
                    continue
    if hits:
        df_title = pd.concat(hits,axis=0)
    else:
        df = pd.DataFrame(["code","name","title","type","date","url"])
    df_title["notice_id"] = (df_title["code"].astype(str) + df_title["date"].astype(str) + df_title["title"].astype(str))
    df_title = df_title.drop_duplicates(
    subset=["notice_id"],        
    keep="first",       
    inplace=False,      
    ignore_index=False  
    )
    df_title = df_title.sort_values(
        by="date",
        ascending=False,
        ignore_index=False
    )
    data = df_title.dropna(subset=["date","title"]).head(limit)
    data.to_parquet(p) 
    return data

def build_sentiment_features(notice_df : pd.DataFrame):
    score_mean = notice_df.groupby("date")["score"].agg(
        score_mean="mean"
    )
    notice_df["score"] = score_mean
    df = notice_df.reindex(notice_df.index)
    if df["score"].isna():
        df["score"] = 0.0
    df["sentiment_ma5"] = sma(df["sentiment"],n=5)
    return df 
    


    

            

        