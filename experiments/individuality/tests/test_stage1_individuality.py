import numpy as np, pandas as pd
import csv, hashlib
from pohang_stage1_individuality.profiles import prepare_panel, past_only_row_labels
from pohang_stage1_individuality.atlas import composite_similarity, distance_from_similarity
from pohang_stage1_individuality.io import verify_manifest_tree

def toy():
    rows=[]
    for ym in [202001,202002,202003,202004,202005,202006]:
        for d,mult in [('포항시 남구',1),('포항시 북구',2)]:
            rows.append({'district':d,'year_month':ym,'category':'호텔','spend_thousand_krw':100*mult+(ym%100)})
    return prepare_panel(pd.DataFrame(rows),[])

def test_past_only_never_uses_current_target():
    x=toy(); lab=past_only_row_labels(x,{'minimum_history_months':2,'recent_months':2,'trend_months':3,'slope_stable_abs':0.01})
    assert lab.current_target_used.sum()==0
    assert lab[lab.year_month.eq(202001)].past_history_count.eq(0).all()

def test_same_category_keeps_two_series():
    x=toy(); assert x.series_id.nunique()==2

def test_distance_properties():
    m=pd.DataFrame([[1,.2],[.2,1]],index=['a','b'],columns=['a','b']); d=distance_from_similarity(m)
    assert np.allclose(np.diag(d),0); assert np.allclose(d,d.T)

def test_composite_weak_floor():
    m=pd.DataFrame([[1,.01],[.01,1]],index=['a','b'],columns=['a','b']); c=composite_similarity({'x':m},{'x':1},.05)
    assert c.loc['a','b']==0

def test_fold_safe_scope_config_accepts_atlas_weights():
    from pohang_stage1_individuality.profiles import fold_safe_scope_recommendations
    from pohang_stage1_individuality.atlas import build_peer_divergence
    x=toy()
    cfg={'minimum_train_months':2,'recent_months':2,'trend_months':3,'slope_stable_abs':0.01,
         'peer_divergence_low_quantile':0.33,'peer_divergence_high_quantile':0.67,
         'pooled_scope_max_individuality':0.3,'district_scope_min_individuality':0.6,
         'minimum_pair_months':2,'weak_similarity_floor':0.05,'composite_weights':{'past_only_residual':.35,'yoy':.3,'recent_residual':.2,'seasonal_shape':.15}}
    out=fold_safe_scope_recommendations(x,[[202005,202006]],cfg,build_peer_divergence)
    assert isinstance(out,pd.DataFrame)

def test_manifest_fallback_rejects_model_file_mismatch(tmp_path):
    data=tmp_path/'integrated'; data.mkdir()
    target=data/'spend.csv'; target.write_text('x\n1\n',encoding='utf-8')
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    with (tmp_path/'PACKAGE_CONTENT_MANIFEST.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['relative_path','size_bytes','sha256'])
        writer.writeheader(); writer.writerow({'relative_path':'integrated/spend.csv','size_bytes':target.stat().st_size,'sha256':digest})
    assert verify_manifest_tree(tmp_path)['status']=='PASS_MODEL_INPUTS_WITH_PACKAGING_EXCEPTION'
    target.write_text('x\n2\n',encoding='utf-8')
    result=verify_manifest_tree(tmp_path)
    assert result['status']=='FAIL' and len(result['mismatched_files'])==1
