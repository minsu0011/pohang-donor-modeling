from __future__ import annotations
import pandas as pd
import numpy as np

from pohang_stage1_v23.selection import _aggregate
from pohang_stage1_v23.ablation import fold_diagnostics


def test_feasibility_first_does_not_promote_infeasible():
    rows=[]
    for w in [1,2,3]:
        rows.append({"name":"A","stage2_gain":0.01,"category_reentry_gain":0.0,"strict_r2":0.8,"age_industry_drop":0.0,"interaction_excess_over_identity":0.0})
        rows.append({"name":"B","stage2_gain":-0.001 if w==2 else 0.02,"category_reentry_gain":0.0,"strict_r2":0.9,"age_industry_drop":0.0,"interaction_excess_over_identity":0.0})
    cfg={"stage1_v23_selection":{"feasible_median_stage2_gain_min":0.003,"feasible_worst_stage2_gain_min":0.0,"feasible_stage2_positive_share_min":1.0,"feasible_category_reentry_median_max":0.005,"feasible_category_reentry_worst_max":0.01}}
    out=_aggregate(pd.DataFrame(rows),cfg)
    a=out[out.name.eq("A")].iloc[0]; b=out[out.name.eq("B")].iloc[0]
    assert int(a.feasible)==1
    assert int(b.feasible)==0


def test_no_feasible_possible():
    rows=[]
    for name in ["A","B"]:
        for _ in range(3):
            rows.append({"name":name,"stage2_gain":-0.01,"category_reentry_gain":0.02,"strict_r2":0.9,"age_industry_drop":0.0,"interaction_excess_over_identity":0.0})
    cfg={"stage1_v23_selection":{"feasible_median_stage2_gain_min":0.003,"feasible_worst_stage2_gain_min":0.0,"feasible_stage2_positive_share_min":1.0,"feasible_category_reentry_median_max":0.005,"feasible_category_reentry_worst_max":0.01}}
    out=_aggregate(pd.DataFrame(rows),cfg)
    assert int(out.feasible.sum())==0


def test_fold2_harmful_diagnostic():
    rows=[]
    for backend in ["CPU","GPU"]:
        for fold,drop in [(1,0.002),(2,-0.003),(3,0.001),(4,0.002)]:
            for seed in [42,137]:
                rows.append({"status":"PASS","kind":"individual","unit":"x","omitted_columns":"x","backend":backend,"fold":fold,"seed":seed,"r2_drop":drop})
    out=fold_diagnostics(pd.DataFrame(rows),harmful_threshold=0.0005)
    r=out.iloc[0]
    assert int(r.fold2_harmful_consensus)==1
    assert r.diagnostic_class=="REGIME_SPECIFIC_CONFLICT"
