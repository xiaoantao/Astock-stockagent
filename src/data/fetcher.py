import re, time
from pathlib import Path
import pandas as pd
import akshare as ak
import baostock as bs
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data.code_adapter import to_akshare, to_baostock
from src import config
FINAL_COLS = ["date","open", "high", "low", "close", "volume", "amount", "turnover"]

AK_COLUMN_MAP = {
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume",
    "成交额": "amount", "换手率": "turnover",
}

def _cache_path(code : str,start : str,end : str,adjust : str):
    num_code = ""
    adjust_map = {
        "hfq":1,
        "qfq":2,
        "":3
    }
    if "-" in code:
        num_code = to_baostock(code)
        norm_adjust = adjust_map.get(adjust,3)
    else:
        num_code = to_akshare(code)
        norm_adjust = adjust if adjust else "raw"
    filename = f"{num_code}_{start}_{end}_{norm_adjust}.parquet"
    return config.RAW_DIR / filename

def standardize_ak(df : pd.DataFrame):
    new_df = df.rename(columns=AK_COLUMN_MAP)
    columns = ['open','close','high','low','volume','amount','turnover']
    new_df = new_df.dropna(subset=['date'])
    new_df['date'] = pd.to_datetime(new_df['date'])
    new_df[columns] = new_df[columns].astype("float64")
    select_df = new_df[['date']+columns]
    index_df = select_df.set_index("date").sort_index(ascending=True)
    return index_df

def parse_baostock_rows(rows,fields):
    columns = fields.split(",")
    df = pd.DataFrame(rows,columns=columns)
    df['date'] = pd.to_datetime(df['date'],errors="coerce")
    df = df.dropna(subset=['date'])
    rest_columns = ["open","close","high","low","volume","amount","turn"]
    for col in rest_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col],errors='coerce')
    new_df = df.reindex(columns=FINAL_COLS)
    index_df = new_df.set_index('date').sort_index(ascending=True)
    modify_df = index_df.rename(columns={"turn":"turnover"})
    return modify_df

@retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1,min=2,max=30),reraise=True)
def fetch_akshare(code,start,end,adjust='hfq'):
    norm_code = to_akshare(code)
    time.sleep(config.REQUEST_SLEEP)
    df = ak.stock_zh_a_hist(symbol=norm_code,period="daily",start_date=start,end_date=end,adjust=adjust)
    new_df = standardize_ak(df)
    return new_df

def fetch_baostock(code,start,end,adjust="hfq"):
    rows = []
    fields = "date,open,close,high,low,volume,amount,turn"
    norm_code = to_baostock(code)
    start_bs = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    end_bs = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    adjust_map = {
        "hfq":"1",
        "qfq":"2",
        "":"3"
    }
    adjust_flag = adjust_map.get(adjust,"3")
    lg = bs.login()
    rs = bs.query_history_k_data_plus(
        code=norm_code,
        start_date=start_bs,
        end_date=end_bs,
        frequency='d',
        adjustflag=adjust_flag,
        fields=fields
    )
    while rs.next():
        row = rs.get_row_data()
        rows.append(row)
    bs.logout()
    df = parse_baostock_rows(rows,fields)
    return df

def fetch_daily(code,start,end,adjust="hfq",use_cache=True):
    num_code = to_akshare(code)
    if start.find("-"):
        start = start.replace("-","")
        start = f"{start[:4]}{start[4:6]}{start[6:8]}"
    if end.find("-"):
        end = end.replace("-","")
        end = f"{end[:4]}{end[4:6]}{end[6:8]}"
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path =_cache_path(code,start,end,adjust)
    p = Path(cache_path)
    if p.is_file() and use_cache:
        cache_df = pd.read_parquet(p)
        return cache_df
    else:
        try:
            new_df = fetch_akshare(code,start,end,adjust)
            norm_df = standardize_ak(new_df)
            norm_df.to_parquet(p)
        except Exception as e:
            print(f"[降级]拉取失败{type(e).__name__},改用baostock去拉取!")
            norm_df = fetch_baostock(code,start,end,adjust)
            norm_df.to_parquet(p)
    return norm_df

