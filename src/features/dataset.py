from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
def make_label(df : pd.DataFrame):
    df = df.copy()
    fwd = df["close"].shift(-1) / df["close"] - 1
    df["fwd_ret"] = fwd
    y = (fwd > 0).astype(float)
    y[fwd.isna()] = np.nan
    df["y"] = y
    return df

def build_dataset(df : pd.DataFrame,feats_df : pd.DataFrame):
    df = df.copy()
    labels = make_label(df)
    df_unit = pd.concat([feats_df.shift(1),labels[["y","fwd_ret"]]],axis=1)
    df_unit_clean = df_unit.dropna()
    return df_unit_clean

def walk_forward_split(data : pd.DataFrame,n_splits=5,embargo=5):
    tscv = TimeSeriesSplit(n_splits = n_splits)
    for tr_idx,te_idx in tscv.split(data):
        te_idx = te_idx[te_idx >= tr_idx[-1] + 1 + embargo]
        if len(tr_idx) == 0:
            continue
        yield data.iloc[tr_idx],data.iloc[te_idx]



