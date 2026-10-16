import pandas as pd
from src.config import TMP_DIR
from pathlib import Path
from src.data.fetcher import fetch_daily
def generate_data_info(code,start,end,adjust="hfq",use_cache=True):
    file_path = f"{TMP_DIR}/data_info.parquet"
    p = Path(file_path)
    if p.exists():
        return pd.read_parquet(p)
    else:
        stock_df = fetch_daily(code,start,end,adjust="hfq",use_cache=True)
        fwd = stock_df["close"].shift(-1) / stock_df["close"] - 1
        stock_df["fwd_ret"] = fwd
        y = fwd
        stock_df["y"] = y
        stock_df.to_parquet(p)
        return stock_df

