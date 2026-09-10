from __future__ import annotations
import numpy as np
import pandas as pd


def build_gate(stage1_repro,selection,metrics,oof,spec,cfg,bias_raw=None,structure_raw=None):
    canon=[]
    for fold,g in metrics.groupby("fold"):
        fam=str(selection[selection.fold.eq(fold)].structure_family.iloc[0])
        if fam.startswith("CATBOOST"):
            z=g[g.backend.eq("CPU")]
            canon.append({"fold":int(fold),"stage1_r2":float(z.stage1_r2.mean()),"bias_only_r2":float(z.bias_only_r2.mean()),"final_r2":float(z.final_r2.mean()),"bias_delta_r2":float(z.bias_delta_r2.mean()),"structure_delta_r2":float(z.structure_delta_r2.mean()),"total_delta_r2":float(z.total_delta_r2.mean())})
        else:
            z=g.iloc[0]
            canon.append({k:(int(fold) if k=="fold" else float(z[k])) for k in ["fold","stage1_r2","bias_only_r2","final_r2","bias_delta_r2","structure_delta_r2","total_delta_r2"]})
    cm=pd.DataFrame(canon).sort_values("fold"); latest=cm.iloc[-1]; gg=cfg["stage2_bias_structure_gates"]
    blocked={"category","category_major","target_log_spend","spend_thousand_krw"}
    checks={
        "stage1_reference_reproduced":bool((pd.to_numeric(stage1_repro.abs_diff,errors="coerce")<=float(gg["stage1_repro_abs_tolerance"])).all()),
        "stage1_future_target_leakage_zero":int(pd.to_numeric(stage1_repro.future_target_violations,errors="coerce").fillna(0).sum())==0,
        "bias_abstention_operational":bool(bias_raw is not None and not bias_raw.empty and bias_raw.bias_family.eq("NO_BIAS").any()),
        "structure_abstention_operational":bool(structure_raw is not None and not structure_raw.empty and structure_raw.family.eq("NO_STRUCTURE").any()),
        "final_median_noninferior_to_stage1":float(cm.total_delta_r2.median())>=-float(gg["median_noninferiority_tolerance"]),
        "no_fold_total_harm_over_limit":float(cm.total_delta_r2.min())>=-float(gg["worst_fold_total_harm_limit"]),
        "fold2_stress_harm_guard":float(cm.loc[cm.fold.eq(2),"total_delta_r2"].iloc[0])>=-float(gg["fold2_harm_limit"]) if 2 in cm.fold.values else True,
        "latest_fold_r2_gate":float(latest.final_r2)>=float(gg["latest_fold_r2_min"]),
        "future_feature_leakage_zero":int(pd.to_numeric(metrics.leakage_violations,errors="coerce").fillna(0).sum())==0,
        "blocked_identity_target_features_absent":not any(x in blocked for x in spec.features),
        "canonical_oof_complete":len(oof)==int(gg["expected_oof_rows"]) and oof.row_index.nunique()==len(oof),
        "structure_zero_mean_audit":float(pd.to_numeric(oof.stage2b_structure_raw_log,errors="coerce").groupby(oof.fold).mean().abs().median())<=float(gg["outer_structure_mean_abs_max"]),
    }
    backend=[]; applicable=0; gpu_ok=True
    for fold in sorted(selection.fold.unique()):
        fam=str(selection[selection.fold.eq(fold)].structure_family.iloc[0])
        if not fam.startswith("CATBOOST"): continue
        applicable+=1; g=metrics[(metrics.fold.eq(fold)) & metrics.backend.isin(["CPU","GPU"])]
        med=g.groupby("backend").total_delta_r2.mean(); ok=("CPU" in med.index and "GPU" in med.index and float(med["CPU"])*float(med["GPU"])>=0)
        gpu_ok &= bool(ok); backend.append({"fold":int(fold),"cpu_total_gain":float(med.get("CPU",np.nan)),"gpu_total_gain":float(med.get("GPU",np.nan)),"sign_agree":bool(ok)})
    checks["catboost_cpu_gpu_sign_agreement"] = bool(gpu_ok) if applicable else True
    backend_status="PASS" if applicable and gpu_ok else ("FAIL" if applicable else "NOT_APPLICABLE")
    status="PASS" if all(checks.values()) else "PARTIAL_PASS"
    return {"status":status,"checks":checks,"canonical_fold_metrics":cm.to_dict("records"),"catboost_backend_audit_status":backend_status,"catboost_backend_audit":backend,"selected_by_fold":selection.to_dict("records"),"median_final_r2":float(cm.final_r2.median()),"median_stage1_r2":float(cm.stage1_r2.median()),"median_total_gain":float(cm.total_delta_r2.median()),"worst_total_gain":float(cm.total_delta_r2.min()),"latest_final_r2":float(latest.final_r2),"consumption_gap_use_allowed":int(status=="PASS"),"policy_use_allowed":0}
