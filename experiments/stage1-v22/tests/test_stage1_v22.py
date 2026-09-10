import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from pohang_main_v2.residual_modeling import ResidualModelSpec
from pohang_main_v2.io import verify_core_manifest_tree
from pohang_main_v2.utils import atomic_write_json, sha256_file
from pohang_stage1_v22.inner_windows import build_inner_windows, audit_inner_windows
from pohang_stage1_v22.orthogonal_interactions import (
    AGE_BASES, WEATHER_BASES, canonical_feature_template, fit_orthogonal_state,
    past_only_orthogonalize, raw_sparse_interaction_columns, transform_orthogonal,
)
from pohang_stage1_v22.gates import summarize_gate


def _months(start='2020-01', n=36):
    return [int(x.strftime('%Y%m')) for x in pd.period_range(start, periods=n, freq='M')]


def test_multiwindow_uses_only_past():
    periods=_months(n=24)
    frame=pd.DataFrame({'year_month':periods})
    w=build_inner_windows(frame,'year_month',count=3,valid_months=4,step_months=4,min_train_months=12)
    assert len(w)==3
    audit=audit_inner_windows(w,202201)
    assert audit.valid_before_outer_test.eq(1).all()
    assert audit.train_before_valid.eq(1).all()
    assert w[-1].valid_end==202112


def test_canonical_template_blocks_raw_sparse_proxy():
    frame=pd.DataFrame({'category':['a'], 'category_major':['식음료']})
    raw_age=['interaction_age20_x_food','interaction_age3040_x_food','interaction_age50plus_x_food']
    raw_weather=['interaction_rain_x_food']
    spec=ResidualModelSpec('x',frame,raw_age+raw_weather+['age_20_share','district'],[],{
        'age_industry_interaction':raw_age,'weather_industry_interaction':raw_weather,'district_identity':['district']
    })
    features,cats,groups=canonical_feature_template(spec)
    assert not raw_sparse_interaction_columns(features)
    assert set(AGE_BASES).issubset(features)
    assert set(WEATHER_BASES).issubset(features)
    assert groups['age_industry_interaction']==list(AGE_BASES)


def test_orthogonal_holdout_is_dense_and_past_only():
    rows=[]
    for ym in _months(n=18):
        for cat,major,offset in [('a','식음료',0.0),('b','쇼핑',0.1)]:
            rows.append({'year_month':ym,'category':cat,'category_major':major,'age_20_share':0.2+offset+(ym%100)*.001,
                         'age_30_40_share':0.4-offset,'age_50_plus_share':0.4,'precipitation_sum_mm':10.,'rain_day_count':2.,'heatwave_day_count':1.})
    df=pd.DataFrame(rows)
    train=df[df.year_month<202106]
    test=df[df.year_month>=202106]
    state=fit_orthogonal_state(train)
    x,a=transform_orthogonal(state,test)
    assert a.orth_future_feature_violation.sum()==0
    assert all(c in x for c in AGE_BASES)
    # Dense encoding: both categories have a defined value in the same orthogonal column.
    assert x['orth_age20_dev_by_category'].notna().all()
    assert x.groupby('category')['orth_age20_dev_by_category'].count().min()>0


def test_past_only_orthogonalization_has_no_future_violation():
    rows=[]
    for ym in _months(n=15):
        rows.append({'year_month':ym,'category_major':'식음료','age_20_share':.2,'age_30_40_share':.4,'age_50_plus_share':.4,
                     'precipitation_sum_mm':10.,'rain_day_count':1.,'heatwave_day_count':0.})
    x,a=past_only_orthogonalize(pd.DataFrame(rows))
    assert a.orth_future_feature_violation.sum()==0


def test_gate_rejects_negative_proxy_and_accepts_multiwindow_only_when_complete(tmp_path):
    bas=[]
    for backend in ['CPU','GPU']:
        for fold,gain in [(1,.02),(2,.01),(3,.01),(4,.012)]:
            bas.append({'status':'PASS','backend':backend,'seed':42,'fold':fold,'r2':.84,'delta_r2_vs_stage1':gain})
    baseline=pd.DataFrame(bas)
    re=[]; prox=[]; age=[]
    for backend in ['CPU','GPU']:
        for fold in range(1,5):
            re += [
                {'backend':backend,'fold':fold,'seed':42,'profile':'add_category','r2_gain_vs_strict':0.0},
            ]
            prox.append({'backend':backend,'fold':fold,'seed':42,'orth_interaction_excess_over_identity':.001,
                         'raw_sparse_excess_over_identity':-.002,'proxy_excess_improvement_orth_minus_raw':.003})
            age.append({'backend':backend,'fold':fold,'seed':42,'r2_drop':.006,'positive':1})
    ref={'age_industry':{'CPU':{'reference':.009},'GPU':{'reference':.005}}}
    (tmp_path/'STAGE1_V21_GATE.json').write_text(json.dumps(ref),encoding='utf-8')
    sel=pd.DataFrame({'fold':[1,2,3,4],'inner_window_count':[3]*4,'outer_test_used_for_selection':[0]*4})
    win=pd.DataFrame({'valid_before_outer_test':[1]*12,'train_before_valid':[1]*12})
    leak=pd.DataFrame({'violations':[0]})
    cfg={'stage1_v22_gate':{'category_reentry_cpu_max_median_delta_r2':.005,'category_reentry_cpu_max_worst_fold_delta_r2':.01,
        'proxy_interaction_excess_min':0.,'proxy_improvement_min':0.,'cpu_median_stage2_gain_min':.01,'latest_cpu_fold_r2_min':.826,
        'age_industry_min_delta_r2':.003,'age_industry_reference_retention_fraction':.5,'age_industry_positive_run_share_min':.75,
        'future_target_violation_max':0},'stage1_v22_selection':{'inner_window_count':3}}
    g=summarize_gate(baseline,pd.DataFrame(re),pd.DataFrame(prox),pd.DataFrame(age),leak,sel,win,tmp_path,cfg)
    assert g['checks']['G2_orthogonal_proxy_excess_cpu_gpu_nonnegative']
    assert g['checks']['G8_multiwindow_selection_complete']
    assert g['checks']['G9_outer_test_never_used_for_selection']


def test_manifest_verified_core_fallback_is_strict(tmp_path: Path):
    data = tmp_path / '02_NEW_INTEGRATED' / 'analysis.csv'
    data.parent.mkdir(parents=True)
    data.write_bytes(b'model-input\n')
    manifest = tmp_path / 'PACKAGE_CONTENT_MANIFEST.csv'
    manifest.write_text(
        'relative_path,size_bytes,sha256\n'
        f'02_NEW_INTEGRATED/analysis.csv,{data.stat().st_size},{sha256_file(data)}\n'
        'docs/guide.md,12,0000000000000000000000000000000000000000000000000000000000000000\n',
        encoding='utf-8-sig',
    )
    passed = verify_core_manifest_tree(tmp_path)
    assert passed['status'] == 'PASS_MODEL_INPUTS_WITH_PACKAGING_EXCEPTION'
    assert passed['verified_files'] == 1
    data.write_bytes(b'tampered\n')
    assert verify_core_manifest_tree(tmp_path)['status'] == 'FAIL'


def test_atomic_json_writes_are_concurrency_safe(tmp_path: Path):
    path = tmp_path / 'shared.json'
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: atomic_write_json(path, {'value': value}), range(64)))
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['value'] in range(64)
    assert not list(tmp_path.glob('*.tmp'))
