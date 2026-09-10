import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from pohang_stage2_adaptive_v3.windows import build_adaptive_windows
from pohang_stage2_adaptive_v3.models import Candidate, fit_predict_candidate
from pohang_stage2_adaptive_v3.tournament import aggregate_candidate_rows
from pohang_main_v2.io import verify_extracted_core_tree
from pohang_main_v2.utils import atomic_write_json


def _cfg():
    return {"stage2_adaptive":{"selection_gate":{"median_gain_min":0.002,"positive_share_min":0.75,"worst_gain_min":-0.002,"std_penalty":0.5,"negative_worst_penalty":1.5}}}


def test_adaptive_windows_have_stability_and_long_horizon():
    periods=pd.period_range('2020-08',periods=24,freq='M')
    f=pd.DataFrame({'year_month':[p.year*100+p.month for p in periods]})
    w=build_adaptive_windows(f,short_count=4,short_months=4,long_months=8,min_train_months=8)
    assert len(w)==5
    assert sum(x.horizon_months==4 for x in w)==4
    assert sum(x.horizon_months==8 for x in w)==1
    assert all(x.train_end < x.valid_start for x in w)


def test_unstable_candidate_is_ineligible_and_no_stage2_is_eligible():
    rows=[]
    for i,g in enumerate([0.01,0.008,-0.006,0.009,0.007]):
        rows.append({'candidate_id':'c','family':'RIDGE','profile':'FULL_ORTHO','params_json':'{}','backend':'CPU','lambda':1.0,'delta_r2':g,'leakage_violations':0})
    for i in range(5):
        rows.append({'candidate_id':'n','family':'NO_STAGE2','profile':'FULL_ORTHO','params_json':'{}','backend':'CPU','lambda':0.0,'delta_r2':0.0,'leakage_violations':0})
    a=aggregate_candidate_rows(pd.DataFrame(rows),_cfg())
    assert int(a[a.family.eq('RIDGE')].eligible.iloc[0])==0
    assert int(a[a.family.eq('NO_STAGE2')].eligible.iloc[0])==1


def test_stable_small_positive_candidate_is_eligible():
    rows=[]
    for g in [0.003,0.004,0.0025,0.0035,0.003]:
        rows.append({'candidate_id':'c','family':'RIDGE','profile':'FULL_ORTHO','params_json':'{}','backend':'CPU','lambda':0.5,'delta_r2':g,'leakage_violations':0})
    a=aggregate_candidate_rows(pd.DataFrame(rows),_cfg())
    r=a.iloc[0]
    assert int(r.eligible)==1
    assert r.positive_share==1.0
    assert r.worst_gain>=0.002


def test_ridge_pipeline_runs_with_categorical_and_missing():
    rng=np.random.default_rng(2)
    n=120
    tr=pd.DataFrame({'x':rng.normal(size=n),'z':rng.normal(size=n),'district':np.where(np.arange(n)%2,'A','B')})
    tr.loc[0,'x']=np.nan
    tr['__residual']=1.5*tr['z'].to_numpy()+rng.normal(scale=.1,size=n)
    va=tr.iloc[:20].copy(); tr=tr.iloc[20:].copy()
    c=Candidate.make('RIDGE','FULL_ORTHO',alpha=1.0)
    p=fit_predict_candidate(c,tr,va,['x','z','district'],['district'],backend='CPU')
    assert len(p)==len(va)
    assert np.isfinite(p).all()

from pohang_stage2_adaptive_v3.stage1_freeze import FrozenStage1, FALLBACK_SCOPE

def test_frozen_scope_snapshot_never_used_before_effective_date():
    sel=pd.DataFrame({
        'fold':[1,1,2,2], 'category':['A','B','A','B'],
        'season_hybrid_v2_scope':['CATEGORY_POOLED','DISTRICT_LEVEL_SHRUNK','FULL_DISTRICT_SEASON_STRICT','CATEGORY_POOLED']
    })
    met=pd.DataFrame({'fold':[1,2],'profile':['SEASON_HYBRID_V2','SEASON_HYBRID_V2'],'r2':[.8,.9]})
    cfg={'validation':{'rolling_test_blocks':[[202208,202307],[202308,202407]]}}
    f=FrozenStage1(sel,met,cfg)
    m,src,eff=f.scope_map_asof(202201,['A','B'])
    assert src=='EARLY_FALLBACK' and eff==-1 and set(m.values())=={FALLBACK_SCOPE}
    m,src,eff=f.scope_map_asof(202210,['A','B'])
    assert src=='REFERENCE_FOLD_1' and eff==202208 and m['A']=='CATEGORY_POOLED'
    m,src,eff=f.scope_map_asof(202309,['A','B'])
    assert src=='REFERENCE_FOLD_2' and eff==202308 and m['A']=='FULL_DISTRICT_SEASON_STRICT'


def test_extracted_core_fallback_rejects_hash_mismatch(tmp_path):
    payload=tmp_path/'model.csv'; payload.write_text('a\n1\n',encoding='utf-8')
    pd.DataFrame([{'relative_path':'model.csv','size_bytes':payload.stat().st_size,'sha256':'0'*64}]).to_csv(tmp_path/'PACKAGE_CONTENT_MANIFEST.csv',index=False)
    try:
        verify_extracted_core_tree(tmp_path)
    except RuntimeError as exc:
        assert 'CORE_EXTRACTED_TREE_VERIFICATION_FAILED' in str(exc)
    else:
        raise AssertionError('hash mismatch was accepted')


def test_atomic_json_concurrent_writes(tmp_path):
    destination=tmp_path/'shared.json'
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda value: atomic_write_json(destination,{'value':value}),range(32)))
    assert json.loads(destination.read_text(encoding='utf-8'))['value'] in range(32)
    assert not list(tmp_path.glob('*.tmp'))
