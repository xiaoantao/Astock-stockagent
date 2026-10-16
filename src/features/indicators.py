import pandas as pd
import numpy as np
def sma(s : pd.Series,n : int):
    return s.rolling(n).mean()

def ema(s : pd.Series,n : int):
    return s.ewm(span=n,adjust=False).mean()

def rsi(close : pd.Series,n=14):
    diff = close.diff().fillna(0)
    up = diff.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs = up / down
    out = 100 - 100/(1 + rs)
    rsi = out.where(down!=0,100)
    rsi = out.where(up!=0,0)
    return rsi.clip(0,100)

def macd(close : pd.Series):
    dif = ema(close,12) - ema(close,26)
    dea = ema(dif,9)
    hist = dif - dea
    return (dif,dea,hist)

def obv(close : pd.Series,volume : pd.Series):
    diff = close.diff().fillna(0)
    sign_volume = pd.Series(np.where(diff > 0 ,1,np.where(diff < 0,-1,0)),index = close.index) * volume
    return sign_volume.cumsum()

def volatility(close : pd.Series,n=30):
    std_v_month = close.pct_change().rolling(n).std()
    std_v_year = std_v_month * np.sqrt(252)
    return std_v_year

def momentum(close : pd.Series,n=10):
    return close - close.shift(n)

def lag_returns(close : pd.Series,k : int):
    return close.pct_change().shift(k)

def build_features(df : pd.DataFrame):
    feats = {}
    close = df["close"]
    volume = df["volume"]
    feats["ma_5"] = sma(close,n=5)
    feats["ma_10"] = sma(close,n=10)
    feats["ma_30"] = sma(close,n=30)
    feats["ma_60"] = sma(close,n=60)
    feats["ema_12"] = ema(close,n=12)
    feats["ema_26"] = ema(close,n=26)
    feats["rsi_14"] = rsi(close,n=14)
    macd_dif,macd_dea,macd_hist = macd(close)
    feats["macd_dif"] = macd_dif
    feats["macd_dea"] = macd_dea
    feats["macd_hist"] = macd_hist
    feats["mom_10"] = momentum(close,n=10)
    feats["obv"] = obv(close,volume)
    feats["volatility_30"] = volatility(close,30)
    feats["lag_ret_1"] = lag_returns(close,k=1)
    feats["lag_ret_5"] = lag_returns(close,k=5)
    out_indicator = pd.DataFrame(feats,index=df.index)
    return out_indicator

    




