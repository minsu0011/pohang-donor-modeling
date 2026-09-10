from __future__ import annotations

from dataclasses import dataclass
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


@dataclass(frozen=True)
class BiasCandidate:
    family: str
    months: int = 0
    half_life: float = 0.0

    @property
    def id(self) -> str:
        if self.family == "NO_BIAS":
            return "NO_BIAS"
        if self.family == "EWMA":
            return f"EWMA_HL{self.half_life:g}"
        return f"{self.family}_{self.months}M"


def bias_candidates(quick: bool=False) -> list[BiasCandidate]:
    if quick:
        return [
            BiasCandidate("NO_BIAS"),
            BiasCandidate("RECENT_MEAN", 4),
            BiasCandidate("RECENT_MEAN", 8),
            BiasCandidate("ROBUST_MEAN", 8),
            BiasCandidate("EWMA", half_life=4.0),
        ]
    return [
        BiasCandidate("NO_BIAS"),
        *[BiasCandidate("RECENT_MEAN", m) for m in (2,4,6,8,12)],
        *[BiasCandidate("ROBUST_MEAN", m) for m in (4,8,12)],
        BiasCandidate("EWMA", half_life=2.0),
        BiasCandidate("EWMA", half_life=4.0),
        BiasCandidate("EWMA", half_life=8.0),
    ]


def _month_means(train: pd.DataFrame) -> pd.DataFrame:
    z=train[["year_month","__residual"]].copy()
    z["year_month"]=pd.to_numeric(z.year_month,errors="coerce").astype("Int64")
    z["__residual"]=pd.to_numeric(z.__residual,errors="coerce")
    z=z.dropna()
    return z.groupby("year_month",as_index=False).__residual.mean().sort_values("year_month")


def _recent_rows(train: pd.DataFrame, months: int) -> pd.DataFrame:
    mm=sorted(pd.to_numeric(train.year_month,errors="coerce").dropna().astype(int).unique())
    if not mm:
        return train.iloc[0:0]
    keep=set(mm[-max(1,int(months)):])
    return train[pd.to_numeric(train.year_month,errors="coerce").isin(keep)].copy()


def _robust_mean(x: np.ndarray) -> float:
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0: return 0.0
    if len(x)<8: return float(np.mean(x))
    lo,hi=np.quantile(x,[0.10,0.90])
    return float(np.mean(np.clip(x,lo,hi)))


def estimate_bias(cand: BiasCandidate, train: pd.DataFrame) -> tuple[float, dict]:
    resid=pd.to_numeric(train["__residual"],errors="coerce").to_numpy(float)
    resid=resid[np.isfinite(resid)]
    if cand.family=="NO_BIAS" or len(resid)==0:
        return 0.0, {"bias_raw":0.0,"recent_mean_3":0.0,"recent_mean_12":0.0,"residual_sd":float(np.nanstd(resid)) if len(resid) else 0.0,"drift_sigma":0.0}
    if cand.family in {"RECENT_MEAN","ROBUST_MEAN"}:
        rr=_recent_rows(train,cand.months)
        vals=pd.to_numeric(rr.__residual,errors="coerce").to_numpy(float)
        vals=vals[np.isfinite(vals)]
        raw=float(np.mean(vals)) if cand.family=="RECENT_MEAN" and len(vals) else _robust_mean(vals)
    elif cand.family=="EWMA":
        mm=_month_means(train)
        if mm.empty:
            raw=0.0
        else:
            age=np.arange(len(mm)-1,-1,-1,dtype=float)
            w=np.exp(-math.log(2.0)*age/max(float(cand.half_life),1e-6))
            raw=float(np.average(mm.__residual.to_numpy(float),weights=w))
    else:
        raise ValueError(cand.family)
    r3=_recent_rows(train,3); r12=_recent_rows(train,12)
    m3=float(pd.to_numeric(r3.__residual,errors="coerce").mean()) if len(r3) else 0.0
    m12=float(pd.to_numeric(r12.__residual,errors="coerce").mean()) if len(r12) else 0.0
    sd=float(np.nanstd(resid)) if len(resid) else 0.0
    drift=abs(m3-m12)/max(sd,1e-9)
    return raw,{"bias_raw":raw,"recent_mean_3":m3,"recent_mean_12":m12,"residual_sd":sd,"drift_sigma":drift}


def bias_metrics(y: np.ndarray, stage1: np.ndarray, bias: float, lam: float) -> dict:
    pred=stage1+float(lam)*float(bias)
    b=float(r2_score(y,stage1)); f=float(r2_score(y,pred))
    return {
        "stage1_r2":b,
        "bias_r2":f,
        "delta_r2":f-b,
        "rmse":float(np.sqrt(mean_squared_error(y,pred))),
        "valid_residual_mean_after_bias":float(np.mean(y-pred)),
    }


def aggregate_bias(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gcfg=cfg["stage2_bias_structure"]["bias_gate"]
    rows=[]
    for (cid,lam),g in raw.groupby(["bias_candidate_id","bias_lambda"],sort=True):
        gains=pd.to_numeric(g.delta_r2,errors="coerce").dropna()
        if gains.empty: continue
        med=float(gains.median()); worst=float(gains.min()); std=float(gains.std(ddof=0)); pos=float((gains>0).mean())
        drift=float(pd.to_numeric(g.drift_sigma,errors="coerce").median())
        score=med-float(gcfg["std_penalty"])*std-float(gcfg["negative_worst_penalty"])*abs(min(0.0,worst))-float(gcfg["drift_penalty"])*drift
        if cid=="NO_BIAS":
            eligible=True; score=0.0
        else:
            eligible=(med>=float(gcfg["median_gain_min"]) and pos>=float(gcfg["positive_share_min"]) and worst>=float(gcfg["worst_gain_min"]) and drift<=float(gcfg["drift_sigma_max"]))
        rows.append({"bias_candidate_id":cid,"bias_lambda":float(lam),"window_count":int(len(gains)),"median_gain":med,"worst_gain":worst,"gain_std":std,"positive_share":pos,"median_drift_sigma":drift,"selection_score":score,"eligible":int(eligible)})
    return pd.DataFrame(rows).sort_values(["eligible","selection_score","median_gain"],ascending=[False,False,False]).reset_index(drop=True)
