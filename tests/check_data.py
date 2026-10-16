import sys
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import RAW_DIR,PROJECT_ROOT,TMP_DIR
raw_path = f"{RAW_DIR}/600519_20200101_20260901_hfq.parquet"
report_path = f"{TMP_DIR}/data_info.parquet"
p = Path(raw_path)
if p.exists():
    data = pd.read_parquet(p)
def make_y(raw_df : pd.DataFrame):
    raw_df = raw_df.copy()
    fwd = raw_df["close"].shift(-1) / raw_df["close"] - 1
    raw_df["fwd_ret"] = fwd
    raw_df["y"] = fwd
    return raw_df
process_p = Path(report_path)
process_data = make_y(data)
process_data.to_parquet(process_p)
