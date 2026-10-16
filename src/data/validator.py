# src/data/validator.py
import pandas as pd

RETURN_TOL = 1e-4

def compare_sources(a: pd.DataFrame, b: pd.DataFrame, tol: float = RETURN_TOL) -> dict:
    """比对两个源一致性。归一化消除复权因子差异后比较价格形态。"""
    a.index = pd.to_datetime(a.index)
    b.index = pd.to_datetime(b.index)

    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return {
            "matched": 0, "n_common": 0,
            "max_abs_diff": 0.0, "mean_abs_diff": 0.0,
            "diff_ratio": 0.0, "passed": False,
            "message": "两个源的日期无交集",
        }

    # 关键：各自除以自身首值 → 消除复权基准差异
    ca = a.loc[idx, "close"].astype(float)
    cb = b.loc[idx, "close"].astype(float)
    na = ca / ca.iloc[0]
    nb = cb / cb.iloc[0]

    abs_ser = (na - nb).abs()
    n_common     = int(len(abs_ser))
    max_abs_diff = float(abs_ser.max())
    mean_abs_diff= float(abs_ser.mean())
    diff_ratio   = float((abs_ser > tol).mean())
    matched      = int((abs_ser <= tol).sum())

    if max_abs_diff <= tol:
        message = "一致"
    else:
        message = (f"存在差异：共同交易日 {n_common} 天 | "
                   f"最大偏差 {max_abs_diff:.3e} | 超标天数占比 {diff_ratio:.2%}")

    return {
        "matched": matched,
        "n_common": n_common,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "diff_ratio": diff_ratio,
        "passed": bool(max_abs_diff <= tol),
        "message": message,
    }