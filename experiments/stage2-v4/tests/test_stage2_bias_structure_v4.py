import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from pohang_stage2_bias_structure_v4.bias import BiasCandidate,estimate_bias,bias_metrics,aggregate_bias
from pohang_stage2_bias_structure_v4.models import StructureCandidate,fit_predict_centered
from pohang_stage2_bias_structure_v4.structure import category_gate_from_windows
from pohang_stage2_bias_structure_v4.tournament import _loo_category_gated
from pohang_main_v2.io import verify_extracted_core_tree
from pohang_main_v2.utils import atomic_write_json


def _cfg():
    return {"stage2_bias_structure":{"bias_gate":{"median_gain_min":0.0005,"positive_share_min":0.8,"worst_gain_min":-0.0015,"drift_sigma_max":0.45,"std_penalty":0.5,"negative_worst_penalty":2.0,"drift_penalty":0.0005},"category_abstention_gate":{"min_windows":3,"min_rows_total":24,"positive_share_min":0.75,"median_sse_gain_ratio_min":0.002,"worst_sse_gain_ratio_min":-0.02}}}


def test_bias_recent_mean_past_only_value():
    rows=[]
    for i,m in enumerate([202301,202302,202303,202304,202305,202306]):
        for j in range(4): rows.append({"year_month":m,"__residual":float(i)})
    df=pd.DataFrame(rows)
    v,d=estimate_bias(BiasCandidate("RECENT_MEAN",4),df)
    assert abs(v-np.mean([2,3,4,5]))<1e-12
    assert d["drift_sigma"]>=0


def test_no_bias_is_zero_score_control():
    raw=pd.DataFrame([
        {"bias_candidate_id":"NO_BIAS","bias_lambda":0.0,"delta_r2":0.0,"drift_sigma":0.0},
        {"bias_candidate_id":"NO_BIAS","bias_lambda":0.0,"delta_r2":0.0,"drift_sigma":0.0},
    ])
    a=aggregate_bias(raw,_cfg())
    assert int(a.iloc[0].eligible)==1 and float(a.iloc[0].selection_score)==0.0


def test_centered_structure_train_predictions_are_zero_mean():
    rng=np.random.default_rng(1)
    tr=pd.DataFrame({"x":rng.normal(size=200),"__centered_residual":rng.normal(size=200)})
    va=pd.DataFrame({"x":rng.normal(size=30)})
    c=StructureCandidate.make("RIDGE","P",alpha=1.0)
    ptr,pv,center=fit_predict_centered(c,tr,va,["x"],[],backend="CPU",threads=1)
    assert abs(float(np.mean(ptr)))<1e-12
    assert len(pv)==len(va)


def test_category_gate_requires_stability():
    rows=[]
    for w in ["A","B","C","D"]:
        rows.append({"window":w,"category":"GOOD","rows":10,"sse_gain_ratio":0.05})
        rows.append({"window":w,"category":"BAD","rows":10,"sse_gain_ratio":-0.03 if w in {"A","B"} else 0.01})
    g=category_gate_from_windows(pd.DataFrame(rows),_cfg())
    m=dict(zip(g.category,g.structure_enabled))
    assert m["GOOD"]==1 and m["BAD"]==0


def test_loo_category_gate_reads_corr_column_not_dataframe_method():
    detail=pd.DataFrame([
        {"window":w,"category":"GOOD","y":y,"base":base,"corr":0.1}
        for w in ["A","B","C","D"] for y,base in [(0.9,0.8),(1.1,1.0)]
    ])
    summary=pd.DataFrame([
        {"window":w,"category":"GOOD","rows":10,"sse_gain_ratio":0.05}
        for w in ["A","B","C","D"]
    ])
    result=_loo_category_gated(detail,summary,_cfg())
    assert len(result)==4 and np.isfinite(result["delta_r2"]).all()


def test_extracted_core_fallback_rejects_hash_mismatch(tmp_path):
    payload=tmp_path/"model.csv"; payload.write_text("a\n1\n",encoding="utf-8")
    pd.DataFrame([{"relative_path":"model.csv","size_bytes":payload.stat().st_size,"sha256":"0"*64}]).to_csv(tmp_path/"PACKAGE_CONTENT_MANIFEST.csv",index=False)
    try:
        verify_extracted_core_tree(tmp_path)
    except RuntimeError as exc:
        assert "CORE_EXTRACTED_TREE_VERIFICATION_FAILED" in str(exc)
    else:
        raise AssertionError("hash mismatch was accepted")


def test_atomic_json_concurrent_writes(tmp_path):
    destination=tmp_path/"shared.json"
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda value: atomic_write_json(destination,{"value":value}),range(32)))
    assert json.loads(destination.read_text(encoding="utf-8"))["value"] in range(32)
    assert not list(tmp_path.glob("*.tmp"))
