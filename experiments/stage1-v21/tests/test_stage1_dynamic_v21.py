import json
from pathlib import Path
import numpy as np
import pandas as pd

from pohang_stage1_v21.dynamic_baseline import Stage1Candidate, fit_dynamic_stage1, predict_dynamic_stage1, past_only_dynamic_stage1
from pohang_stage1_v21.gates import summarize_gate


def synthetic_frame():
    rows=[]
    for ym in [202001,202002,202003,202004,202005,202006,202007,202008]:
        month=ym%100
        for cat,base in [('A',5.0),('B',7.0)]:
            for district in ['N','S']:
                rows.append({'year_month':ym,'month':month,'category':cat,'district':district,'target_log_spend':base+0.02*(ym-202000)+0.1*(district=='S')})
    return pd.DataFrame(rows)


def test_past_only_never_uses_same_or_future_period():
    frame=synthetic_frame()
    cand=Stage1Candidate('test',recent_window=3,recent_weight=0.5,trend_window=4,trend_weight=0.2,bias_window=2,bias_weight=0.3)
    pred=past_only_dynamic_stage1(frame,cand)
    assert int(pred.stage1_future_target_violation.sum())==0
    valid=pred.stage1_max_history_period.ge(0)
    assert (pred.loc[valid,'stage1_max_history_period'] < pred.loc[valid,'stage1_target_period']).all()


def test_frozen_state_is_strictly_earlier_than_future():
    frame=synthetic_frame()
    train=frame[frame.year_month<=202006]
    future=frame[frame.year_month>=202007]
    cand=Stage1Candidate('test',recent_window=3,recent_weight=.5)
    state=fit_dynamic_stage1(train,cand)
    pred=predict_dynamic_stage1(state,future)
    assert pred.stage1_future_target_violation.sum()==0
    assert pred.stage1_max_history_period.max()==202006


def test_gate_logic(tmp_path: Path):
    baseline=[]
    for backend in ['CPU','GPU']:
        for fold in range(1,5):
            for seed in [42,137]:
                baseline.append({'backend':backend,'status':'PASS','fold':fold,'seed':seed,'r2':.83 if fold==4 else .87,'delta_r2_vs_stage1':.012,'stage1_test_leakage_violations':0,'stage1_train_leakage_violations':0})
    baseline=pd.DataFrame(baseline)
    re=[]
    for backend in ['CPU','GPU']:
        for fold in range(1,5):
            for seed in [42,137]:
                re += [
                    {'backend':backend,'fold':fold,'seed':seed,'profile':'strict_residual','r2_gain_vs_strict':0.0},
                    {'backend':backend,'fold':fold,'seed':seed,'profile':'add_category','r2_gain_vs_strict':0.003},
                ]
    re=pd.DataFrame(re)
    proxy=pd.DataFrame([{'backend':b,'seed':42,'fold':f,'interaction_excess_over_identity':.001} for b in ['CPU','GPU'] for f in range(1,5)])
    age=pd.DataFrame([{'backend':b,'seed':42,'fold':f,'r2_drop':.005,'positive':1} for b in ['CPU','GPU'] for f in range(1,5)])
    leak=pd.DataFrame([{'violations':0}])
    for b,v in [('CPU',.009),('GPU',.005)]:
        pd.DataFrame([{'unit':'age_industry_interaction','median_r2_drop':v}]).to_csv(tmp_path/f'V2_GROUP_ABLATION_{b}.csv',index=False,encoding='utf-8-sig')
    cfg={'stage1_v21_gate':{}}
    gate=summarize_gate(baseline,re,proxy,age,leak,tmp_path,cfg)
    assert gate['status']=='PASS'
    assert all(gate['checks'].values())
