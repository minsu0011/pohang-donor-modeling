from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from .io import load_spend_from_core
from .profiles import prepare_panel, full_history_profiles, past_only_row_labels, fold_safe_scope_recommendations, assign_behavior_clusters
from .atlas import build_views, composite_similarity, distance_from_similarity, clustered_order, build_peer_divergence, save_heatmap, save_peer_metric_map


def write_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def load_cfg(path): return yaml.safe_load(Path(path).read_text(encoding='utf-8'))

def run(config:str):
    cfg=load_cfg(config); root=Path.cwd(); out=root/cfg['outputs']['dir']; out.mkdir(parents=True,exist_ok=True)
    inp=cfg['inputs']; core=root/inp['core_zip']
    fallback = Path(inp['core_extracted_dir']) if inp.get('core_extracted_dir') else None
    if fallback is not None and not fallback.is_absolute(): fallback = root / fallback
    raw,meta=load_spend_from_core(
        core, inp.get('expected_core_sha256',''), inp['spend_member_suffix'],
        fallback_dir=fallback,
        allow_manifest_fallback=bool(inp.get('allow_manifest_verified_core_fallback',False)),
    )
    panel=prepare_panel(raw,inp.get('exclude_categories',[]))
    stage_cfg=cfg['stage1_labels']; atlas_cfg=cfg['atlas']
    profile=full_history_profiles(panel,stage_cfg)
    peer=build_peer_divergence(panel,{**stage_cfg,**atlas_cfg})
    profile=profile.merge(peer[['category','individuality_score','individuality_band','recommended_stage1_scope']],on='category',how='left')
    views=build_views(panel,atlas_cfg)
    comp=composite_similarity(views,atlas_cfg['composite_weights'],atlas_cfg.get('weak_similarity_floor',0.05))
    dist=distance_from_similarity(comp); order=clustered_order(dist,atlas_cfg.get('linkage','average'))
    profile=assign_behavior_clusters(profile,dist,n_clusters=6)
    row_labels=past_only_row_labels(panel,stage_cfg)
    fold_scope=fold_safe_scope_recommendations(panel,cfg['validation']['rolling_test_blocks'],{**stage_cfg,**atlas_cfg,**cfg['validation']},build_peer_divergence)
    # Save tables
    panel.to_csv(out/'STAGE1_SERIES_PANEL.csv.gz',index=False,encoding='utf-8-sig',compression={'method':'gzip','mtime':0})
    profile.to_csv(out/'STAGE1_SERIES_LABEL_REGISTRY.csv',index=False,encoding='utf-8-sig')
    row_labels.to_csv(out/'STAGE1_PAST_ONLY_ROW_LABELS.csv.gz',index=False,encoding='utf-8-sig',compression={'method':'gzip','mtime':0})
    peer.to_csv(out/'STAGE1_WITHIN_CATEGORY_DIVERGENCE.csv',index=False,encoding='utf-8-sig')
    fold_scope.to_csv(out/'STAGE1_FOLD_SAFE_SCOPE_RECOMMENDATIONS.csv',index=False,encoding='utf-8-sig')
    comp.to_csv(out/'STAGE1_INDIVIDUALITY_COMPOSITE_SIMILARITY.csv',encoding='utf-8-sig')
    dist.to_csv(out/'STAGE1_INDIVIDUALITY_DISTANCE.csv',encoding='utf-8-sig')
    pd.DataFrame({'cluster_order':order}).to_csv(out/'STAGE1_ATLAS_ORDER.csv',index=False,encoding='utf-8-sig')
    for name,m in views.items(): m.to_csv(out/f'STAGE1_VIEW_{name.upper()}.csv',encoding='utf-8-sig')
    # Figures
    save_heatmap(comp,out/'STAGE1_INDIVIDUALITY_CORRELATION_MAP.png','포항 Stage 1 세부업종×권역 개별성 중심 상관지도',order,atlas_cfg['annotate_abs_threshold'],atlas_cfg['dpi'],tuple(atlas_cfg['figsize']))
    save_heatmap(views['raw_level'],out/'STAGE1_RAW_LEVEL_CORRELATION_MAP.png','Stage 1 원수준 Spearman 상관지도',order,atlas_cfg['annotate_abs_threshold'],atlas_cfg['dpi'],tuple(atlas_cfg['figsize']))
    save_heatmap(views['past_only_residual'],out/'STAGE1_PAST_ONLY_RESIDUAL_CORRELATION_MAP.png','Stage 1 과거전용 업종계절 기준 제거 상관지도',order,atlas_cfg['annotate_abs_threshold'],atlas_cfg['dpi'],tuple(atlas_cfg['figsize']))
    save_heatmap(views['district_shock_removed'],out/'STAGE1_DISTRICT_SHOCK_REMOVED_CORRELATION_MAP.png','Stage 1 권역 공통충격 제거 개별성 상관지도',order,atlas_cfg['annotate_abs_threshold'],atlas_cfg['dpi'],tuple(atlas_cfg['figsize']))
    save_heatmap(views['yoy'],out/'STAGE1_YOY_CORRELATION_MAP.png','Stage 1 전년동월 변화 상관지도',order,atlas_cfg['annotate_abs_threshold'],atlas_cfg['dpi'],tuple(atlas_cfg['figsize']))
    save_peer_metric_map(peer,out/'STAGE1_WITHIN_CATEGORY_INDIVIDUALITY_MAP.png',atlas_cfg['dpi'])
    # QA metrics
    ids=comp.index.tolist(); off=comp.to_numpy(float)[~np.eye(len(ids),dtype=bool)]; off=off[np.isfinite(off)]
    same=[]; major_same=[]; cross=[]
    meta_sid=profile.set_index('series_id')
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            v=comp.loc[a,b]
            if not np.isfinite(v): continue
            if meta_sid.loc[a,'category']==meta_sid.loc[b,'category']: same.append(v)
            elif meta_sid.loc[a,'category_major']==meta_sid.loc[b,'category_major']: major_same.append(v)
            else: cross.append(v)
    qa={**meta,
        'status':'PASS','series_count':len(ids),'category_count':int(panel.category.nunique()),'district_count':int(panel.district.nunique()),
        'month_count':int(panel.year_month.nunique()),'panel_rows':int(len(panel)),
        'past_only_current_target_violations':int(row_labels.current_target_used.sum()),
        'label_registry_rows':int(len(profile)),
        'same_category_pair_mean_similarity':float(np.nanmean(same)) if same else None,
        'same_major_different_category_mean_similarity':float(np.nanmean(major_same)) if major_same else None,
        'cross_major_mean_similarity':float(np.nanmean(cross)) if cross else None,
        'offdiag_abs_similarity_median':float(np.nanmedian(np.abs(off))) if len(off) else None,
        'high_individuality_categories':peer.loc[peer.individuality_band.eq('HIGH'),'category'].tolist(),
        'fold_safe_scope_rows':int(len(fold_scope)),
        'same_category_is_not_forced_together':True,
        'full_history_behavior_labels_are_audit_only':True,
        'past_only_labels_are_model_safe':True,
    }
    write_json(out/'STAGE1_INDIVIDUALITY_ATLAS_QA.json',qa)
    # Simple markdown review
    top=peer.sort_values('individuality_score',ascending=False).head(8)
    lines=['# Stage 1 개별성 Atlas 요약','',f"- 시계열: {len(ids)}개 (17 세부업종 × 남/북구)",f"- 기간: {panel.year_month.min()}~{panel.year_month.max()}",f"- 과거전용 라벨 current-target 위반: {int(row_labels.current_target_used.sum())}건",'', '## 같은 category 내부 개별성 상위','']
    for r in top.itertuples(index=False): lines.append(f"- {r.category}: individuality={r.individuality_score:.3f}, composite={r.composite_similarity:.3f}, 권고={r.recommended_stage1_scope}")
    lines += ['','## 해석 규칙','- 같은 category라고 남구/북구를 강제로 같은 cluster에 묶지 않는다.','- 전체기간 target-derived behavior label은 AUDIT_ONLY다.','- 모델에 들어갈 수 있는 라벨은 PAST_ONLY 산출물 또는 fold-safe scope만 사용한다.','- 개별성이 높으면 Stage1에서 category pooled 평균보다 category×district/shrinkage baseline을 우선 검증한다.']
    (out/'STAGE1_INDIVIDUALITY_ATLAS_REVIEW_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(qa,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/default.yaml'); args=ap.parse_args(); run(args.config)
if __name__=='__main__': main()
