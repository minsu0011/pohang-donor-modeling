from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .association import run_panel_atlas
from .category_baseline import past_only_category_baseline
from .config import load_config, resolve_path
from .feature_registry import active_features, build_feature_registry
from .feature_store import build_feature_bundle
from .io import PreparedInputs, prepare_inputs
from .residual_ablation import (
    merge_backend_consensus,
    redundancy_units,
    run_residual_ablation_units,
    run_residual_baselines,
    shard_units,
    single_feature_units,
    summarize_residual_ablation,
)
from .residual_diagnostics import (
    aggregate_oof_predictions,
    build_fold_drift_report,
    build_model_gate,
    build_residual_feature_gate,
    run_category_reentry_audit,
    run_challenger_profiles,
    write_v2_review,
)
from .residual_modeling import ResidualModelSpec
from .screening import build_screening_scores
from .separation import build_separation_representative_map, run_target_separation
from .utils import atomic_write_json, ensure_dir, write_csv, dataframe_identity


def project_root() -> Path:
    return Path.cwd().resolve()


def _run_paths(cfg: dict[str, Any], root: Path, run_name: str) -> tuple[Path, Path, Path]:
    work = ensure_dir(resolve_path(root, cfg["paths"]["work_dir"]) / run_name)
    out = ensure_dir(resolve_path(root, cfg["paths"]["output_dir"]) / run_name)
    cache = ensure_dir(resolve_path(root, cfg["paths"]["cache_dir"]) / run_name)
    return work, out, cache


def _input_context(cfg: dict[str, Any], root: Path, out: Path, force: bool = False) -> PreparedInputs:
    prepared = prepare_inputs(root, cfg, force=force)
    expected = str(cfg["inputs"].get("expected_core_sha256", ""))
    if expected and prepared.core_source_mode == "ZIP" and prepared.core_sha256.lower() != expected.lower():
        raise RuntimeError(f"CORE_SHA256_MISMATCH expected={expected} actual={prepared.core_sha256}")
    expected_donor = str(cfg["inputs"].get("expected_donor_sha256", ""))
    if expected_donor and prepared.donor_sha256.lower() != expected_donor.lower():
        raise RuntimeError(f"DONOR_SHA256_MISMATCH expected={expected_donor} actual={prepared.donor_sha256}")
    donor_required = ["donor_feature_gate_v21.csv", "group_ablation_consensus_v21.csv"]
    missing = [name for name in donor_required if not (prepared.donor_reference_dir / name).exists()]
    input_status = (
        "PASS_MODEL_INPUTS_WITH_PACKAGING_EXCEPTION"
        if prepared.core_source_mode == "MANIFEST_VERIFIED_EXTRACTED_TREE" and not missing
        else ("PASS" if not missing else "FAIL")
    )
    verification = {
        "status": input_status,
        "canonical_zip_status": "PASS" if prepared.core_source_mode == "ZIP" else "UNAVAILABLE",
        "expected_core_sha256": expected,
        "core_source_mode": prepared.core_source_mode,
        "core_zip": str(prepared.core_zip),
        "core_sha256": prepared.core_sha256,
        "core_manifest_verification": prepared.core_manifest_verification,
        "donor_zip": str(prepared.donor_zip),
        "donor_sha256": prepared.donor_sha256,
        "integration_root": str(prepared.integration_root),
        "donor_reference_dir": str(prepared.donor_reference_dir),
        "donor_reference_missing": missing,
        "category_normalization_is_past_only": True,
        "donor_is_secondary_only": True,
    }
    atomic_write_json(out / "INPUT_VERIFICATION.json", verification)
    if missing:
        raise RuntimeError("DONOR_REFERENCE_MISSING: " + ", ".join(missing))
    return prepared


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False, encoding="utf-8-sig", compression={"method": "gzip", "mtime": 0})


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", compression="gzip", low_memory=False)


def _store_paths(work: Path) -> dict[str, Path]:
    base = work / "feature_store"
    return {
        "main": base / "main_panel.csv.gz",
        "screening": base / "screening_panel.csv.gz",
        "site": base / "site_panel.csv.gz",
        "corridor": base / "corridor_panel.csv.gz",
        "registry": base / "feature_registry.csv",
    }


def _write_feature_labeling_artifacts(registry: pd.DataFrame, out: Path) -> dict[str, Any]:
    write_csv(registry, out / "POHANG_FEATURE_REGISTRY_FULL_V2.csv")
    required = [
        "feature_id", "panel", "column_name", "korean_label", "english_label",
        "concept_group", "data_type", "source_dataset", "spatial_grain",
        "temporal_grain", "availability", "model_role", "leakage_status",
        "active_residual",
    ]
    audit = registry[[c for c in required if c in registry]].copy()
    for column in required:
        audit[f"check_{column}"] = (
            registry[column].fillna("").astype(str).str.strip().ne("") if column in registry else False
        )
    checks = [c for c in audit if c.startswith("check_")]
    audit["labeling_complete"] = audit[checks].all(axis=1)
    write_csv(audit, out / "POHANG_FEATURE_LABEL_AUDIT_V2.csv")
    lines = [
        "# 포항 Category-normalized Expected Spend V2 피처 사전", "",
        "`category`와 `category_major`는 Stage 1 기준선 전용이며 canonical Stage 2 predictor가 아니다.", "",
    ]
    for (panel, concept), part in registry.sort_values(["panel", "concept_group", "column_name"]).groupby(["panel", "concept_group"], dropna=False):
        lines += [f"## {panel} · {concept}", "", "|피처|한글 라벨|역할|Residual 활성|누출 상태|donor|", "|---|---|---|---:|---|---|"]
        for row in part.itertuples(index=False):
            lines.append(
                f"|`{row.column_name}`|{str(row.korean_label).replace('|','/')}|{row.model_role}|"
                f"{int(getattr(row,'active_residual',0))}|{row.leakage_status}|{getattr(row,'donor_decision','')}|"
            )
        lines.append("")
    (out / "POHANG_FEATURE_DICTIONARY_KO_V2.md").write_text("\n".join(lines), encoding="utf-8")
    incomplete = int((~audit.labeling_complete).sum())
    summary = {
        "status": "PASS" if incomplete == 0 else "FAIL",
        "feature_count": int(len(registry)),
        "labeling_incomplete": incomplete,
        "active_residual": int(pd.to_numeric(registry.get("active_residual", 0), errors="coerce").fillna(0).sum()),
        "stage1_baseline_keys": registry[registry.model_role.eq("STAGE1_BASELINE_KEY")].column_name.tolist(),
    }
    atomic_write_json(out / "FEATURE_LABELING_SUMMARY_V2.json", summary)
    if incomplete:
        raise RuntimeError(f"FEATURE_LABELING_INCOMPLETE: {incomplete}")
    return summary


def _augment_category_normalized_target(main: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    mode = str(cfg["category_baseline"].get("atlas_mode", "category_month"))
    kwargs = {
        "category_col": str(cfg["category_baseline"].get("category_col", "category")),
        "district_col": str(cfg["category_baseline"].get("district_col", "district")),
        "month_col": str(cfg["category_baseline"].get("month_col", "month")),
        "alpha_category": float(cfg["category_baseline"].get("alpha_category", 12.0)),
        "alpha_month": float(cfg["category_baseline"].get("alpha_month", 8.0)),
        "alpha_district": float(cfg["category_baseline"].get("alpha_district", 8.0)),
        "alpha_leaf": float(cfg["category_baseline"].get("alpha_leaf", 5.0)),
    }
    baseline = past_only_category_baseline(
        main,
        mode,
        target_col="target_log_spend",
        period_col="year_month",
        **kwargs,
    )
    out = main.copy()
    out["category_baseline_past_only_log"] = baseline["category_baseline_log"]
    out["category_baseline_history_count"] = baseline["category_history_count"]
    out["target_category_normalized"] = (
        pd.to_numeric(out["target_log_spend"], errors="coerce")
        - pd.to_numeric(out["category_baseline_past_only_log"], errors="coerce")
    )
    out["category_baseline_atlas_mode"] = mode
    return out


def _ensure_prepared(cfg: dict[str, Any], root: Path, run_name: str) -> tuple[Path, Path, Path, PreparedInputs]:
    work, out, cache = _run_paths(cfg, root, run_name)
    paths = _store_paths(work)
    prepared = _input_context(cfg, root, out)
    if not all(paths[k].exists() for k in ["main", "screening", "site", "corridor", "registry"]):
        _prepare(cfg, root, run_name, prepared)
    return work, out, cache, prepared


def _prepare(cfg: dict[str, Any], root: Path, run_name: str, prepared: PreparedInputs | None = None) -> dict[str, Any]:
    work, out, _ = _run_paths(cfg, root, run_name)
    prepared = prepared or _input_context(cfg, root, out)
    bundle = build_feature_bundle(prepared.integration_root, cfg)
    main = _augment_category_normalized_target(bundle.main_panel, cfg)
    panels = {
        "main": main,
        "screening": bundle.screening_panel,
        "site": bundle.site_panel,
        "corridor": bundle.corridor_panel,
    }
    registry = build_feature_registry(panels, prepared.donor_reference_dir)
    store = ensure_dir(work / "feature_store")
    for name, frame in panels.items():
        _save_frame(frame, store / f"{name}_panel.csv.gz")
    write_csv(registry, store / "feature_registry.csv")
    write_csv(bundle.build_audit, store / "feature_build_audit.csv")
    label_summary = _write_feature_labeling_artifacts(registry, out)
    ranked, rank_ablation = build_screening_scores(bundle.screening_panel, cfg)
    _save_frame(ranked, store / "screening_ranked.csv.gz")
    write_csv(rank_ablation, store / "screening_ranking_ablation.csv")
    summary = {
        "status": "PASS",
        "panels": {k: {"rows": int(len(v)), "columns": int(len(v.columns))} for k, v in panels.items()},
        "feature_registry_rows": int(len(registry)),
        "feature_labeling": label_summary,
        "category_normalized_nonmissing_rows": int(main.target_category_normalized.notna().sum()),
        "category_baseline_atlas_mode": str(cfg["category_baseline"].get("atlas_mode")),
        "same_month_target_leakage_allowed": False,
    }
    atomic_write_json(out / "PREPARE_SUMMARY_V2.json", summary)
    return summary


def verify_cmd(args: argparse.Namespace) -> None:
    root = project_root()
    cfg = load_config(args.config)
    _, out, _ = _run_paths(cfg, root, args.run_name)
    _input_context(cfg, root, out, force=args.force_extract)
    print((out / "INPUT_VERIFICATION.json").read_text(encoding="utf-8"))


def prepare_cmd(args: argparse.Namespace) -> None:
    root = project_root()
    cfg = load_config(args.config)
    _, out, _ = _run_paths(cfg, root, args.run_name)
    prepared = _input_context(cfg, root, out, force=args.force_extract)
    print(json.dumps(_prepare(cfg, root, args.run_name, prepared), ensure_ascii=False, indent=2))


def correlate_cmd(args: argparse.Namespace) -> None:
    root = project_root()
    cfg = load_config(args.config)
    work, out, _, _ = _ensure_prepared(cfg, root, args.run_name)
    store = _store_paths(work)
    registry = pd.read_csv(store["registry"], encoding="utf-8-sig")
    main = _load_frame(store["main"])

    residual_registry = registry.copy()
    residual_registry["active_structural"] = pd.to_numeric(
        residual_registry.get("active_residual", 0), errors="coerce"
    ).fillna(0).astype(int)

    rows: list[dict[str, Any]] = []
    if not args.quick:
        raw = run_panel_atlas(
            "main", main, registry, out / "correlation_atlas_raw",
            "target_log_spend", "year_month", cfg, active_only=False
        )
        raw["atlas_family"] = "raw_target"
        rows.append(raw)

    residual = run_panel_atlas(
        "main", main, residual_registry, out / "correlation_atlas_residual",
        "target_category_normalized", "year_month", cfg, active_only=bool(args.quick)
    )
    residual["atlas_family"] = "category_normalized_target"
    rows.append(residual)

    if not args.quick:
        for panel, target, period in [
            ("screening", "latest_tourism_spend_share_pct", None),
            ("site", "nav_search_window_sum", "period_end"),
            ("corridor", "screening_priority_score_0_100", None),
        ]:
            frame = _load_frame(store[panel])
            context_row = run_panel_atlas(
                panel, frame, registry, out / "correlation_atlas_context",
                target if target in frame else None,
                period if period and period in frame else None,
                cfg, active_only=False
            )
            context_row["atlas_family"] = "context"
            rows.append(context_row)

    separation = run_target_separation(
        main, residual_registry, cfg, out / "target_separation_residual",
        panel="main", target_col="target_category_normalized",
        period_col="year_month", structural_only=True, quick=args.quick
    )
    build_separation_representative_map(
        out / "correlation_atlas_residual" / "main", separation.summary,
        residual_registry, cfg, panel="main", target="target_category_normalized"
    )
    summary = pd.DataFrame(rows)
    write_csv(summary, out / "CORRELATION_ATLAS_SUMMARY_V2.csv")
    print(summary.to_string(index=False))


def _spec(cfg: dict[str, Any], root: Path, run_name: str) -> tuple[ResidualModelSpec, pd.DataFrame, Path, Path, Path, PreparedInputs]:
    work, out, cache, prepared = _ensure_prepared(cfg, root, run_name)
    store = _store_paths(work)
    frame = _load_frame(store["main"])
    registry = pd.read_csv(store["registry"], encoding="utf-8-sig")
    features, categorical, groups = active_features(registry, "main", "residual")
    features = [c for c in features if c in frame and frame[c].nunique(dropna=True) > 1 and frame[c].isna().mean() < 0.90]
    categorical = [c for c in categorical if c in features]
    groups = {g: [c for c in cols if c in features] for g, cols in groups.items()}
    groups = {g: cols for g, cols in groups.items() if cols}
    if "category" in features or "category_major" in features:
        raise RuntimeError("CATEGORY_IDENTITY_ENTERED_CANONICAL_STAGE2")
    spec = ResidualModelSpec(
        name="pohang_category_residual_expected_spend",
        frame=frame,
        features=features,
        categorical=categorical,
        groups=groups,
        profile="residual",
        data_identity=dataframe_identity(frame),
    )
    return spec, registry, work, out, cache, prepared


def baseline_cmd(args: argparse.Namespace) -> None:
    root = project_root()
    cfg = load_config(args.config)
    spec, _, _, out, _, _ = _spec(cfg, root, args.run_name)
    backend = args.backend.upper()
    threads = int(cfg["hardware"]["cpu_threads_per_model"] if backend == "CPU" else cfg["hardware"]["gpu_host_threads"])
    metrics, predictions = run_residual_baselines(
        spec,
        cfg,
        backend,
        args.mode == "quick",
        threads,
        sample_weight_mode=str(cfg["residual_model"].get("canonical_sample_weight_mode", "none")),
        calibration_mode=str(cfg["residual_model"].get("canonical_calibration_mode", "none")),
    )
    dest = ensure_dir(out / "residual_model" / backend.lower())
    write_csv(metrics, dest / "baseline.csv")
    if not predictions.empty:
        predictions.to_csv(dest / "oof_predictions.csv.gz", index=False, encoding="utf-8-sig", compression={"method": "gzip", "mtime": 0})
    print(metrics.to_string(index=False))


def ablate_cmd(args: argparse.Namespace) -> None:
    root = project_root()
    cfg = load_config(args.config)
    spec, _, _, out, cache, _ = _spec(cfg, root, args.run_name)
    backend = args.backend.upper()
    dest = ensure_dir(out / "residual_model" / backend.lower())
    baseline_path = dest / "baseline.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Run baseline first: {baseline_path}")
    baseline = pd.read_csv(baseline_path, encoding="utf-8-sig")
    pair_path = out / "correlation_atlas_residual" / "main" / "association_pairs.csv"
    pairs = pd.read_csv(pair_path, encoding="utf-8-sig") if pair_path.exists() else pd.DataFrame()
    units: dict[str, dict[str, list[str]]] = {}
    if args.kind in {"group", "all"}:
        units["group"] = spec.groups
    if args.kind in {"feature", "all"}:
        units["feature"] = single_feature_units(spec.features)
    if args.kind in {"redundancy", "all"}:
        units["redundancy"] = redundancy_units(spec.features, pairs, float(cfg["ablation"]["redundancy_cluster_threshold"]))
    if "feature" in units and int(args.unit_shard_count) > 1:
        units["feature"] = shard_units(units["feature"], int(args.unit_shard_count), int(args.unit_shard_index))
    quick = args.mode == "quick"
    workers = int(cfg["hardware"]["cpu_workers"] if backend == "CPU" else 1)
    threads = int(cfg["hardware"]["cpu_threads_per_model"] if backend == "CPU" else cfg["hardware"]["gpu_host_threads"])
    for kind, named_units in units.items():
        write_csv(pd.DataFrame([{"kind": kind, "unit": n, "columns": "|".join(c), "count": len(c)} for n, c in named_units.items()]), dest / f"{kind}_ablation_units.csv")
        raw = run_residual_ablation_units(spec, cfg, backend, baseline, named_units, quick, workers, threads, cache / "residual_ablation_tasks", kind)
        write_csv(raw, dest / f"ablation_raw_{kind}.csv")
        write_csv(summarize_residual_ablation(raw), dest / f"ablation_summary_{kind}.csv")


def branch_cmd(args: argparse.Namespace) -> None:
    common = dict(vars(args))
    for kind in ["group", "redundancy", "feature"]:
        local = argparse.Namespace(**common)
        local.kind = kind
        local.unit_shard_count = int(args.feature_shard_count) if kind == "feature" else 1
        local.unit_shard_index = int(args.feature_shard_index) if kind == "feature" else 0
        ablate_cmd(local)


def diagnose_cmd(args: argparse.Namespace) -> None:
    root = project_root()
    cfg = load_config(args.config)
    spec, _, _, out, _, _ = _spec(cfg, root, args.run_name)
    backend = args.backend.upper()
    threads = int(cfg["hardware"]["cpu_threads_per_model"] if backend == "CPU" else cfg["hardware"]["gpu_host_threads"])
    baseline_path = out / "residual_model" / backend.lower() / "baseline.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Run baseline first: {baseline_path}")
    baseline = pd.read_csv(baseline_path, encoding="utf-8-sig")
    diag = ensure_dir(out / "diagnostics")
    run_category_reentry_audit(spec, cfg, baseline, backend, threads, diag)
    run_challenger_profiles(spec, cfg, backend, threads, diag, quick=args.mode == "quick")
    if backend == "CPU":
        build_fold_drift_report(spec, cfg, diag)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def _competition_coverage(registry: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    structural, _, _ = active_features(registry, "main", "structural")
    structural = [c for c in structural if c in frame and frame[c].nunique(dropna=True) > 1 and frame[c].isna().mean() < 0.90]
    residual, _, _ = active_features(registry, "main", "residual")
    residual = [c for c in residual if c in frame and frame[c].nunique(dropna=True) > 1 and frame[c].isna().mean() < 0.90]
    rows = []
    for feature in structural:
        route = "stage1_category_reentry_audit" if feature in {"category", "category_major"} else "stage2_full_ablation"
        rows.append({"feature": feature, "competition_route": route, "covered": int(feature in residual or feature in {"category", "category_major"})})
    coverage = pd.DataFrame(rows)
    summary = coverage.groupby("competition_route", as_index=False).agg(feature_count=("feature", "size"), covered_count=("covered", "sum"))
    return coverage, summary


def finalize_cmd(args: argparse.Namespace) -> None:
    root = project_root()
    cfg = load_config(args.config)
    spec, registry, work, out, _, prepared = _spec(cfg, root, args.run_name)
    summaries: list[pd.DataFrame] = []
    baselines: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    reentry: list[pd.DataFrame] = []
    challengers: list[pd.DataFrame] = []
    for backend in ["cpu", "gpu"]:
        dest = out / "residual_model" / backend
        for kind in ["group", "feature", "redundancy"]:
            part = _read_optional_csv(dest / f"ablation_summary_{kind}.csv")
            if not part.empty:
                summaries.append(part)
        base = _read_optional_csv(dest / "baseline.csv")
        if not base.empty:
            baselines.append(base)
        pred_path = dest / "oof_predictions.csv.gz"
        if pred_path.exists():
            predictions.append(pd.read_csv(pred_path, encoding="utf-8-sig", compression="gzip"))
        re = _read_optional_csv(out / "diagnostics" / f"category_reentry_raw_{backend}.csv")
        if not re.empty:
            reentry.append(re)
        ch = _read_optional_csv(out / "diagnostics" / f"challenger_profiles_raw_{backend}.csv")
        if not ch.empty:
            challengers.append(ch)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    consensus = merge_backend_consensus(summary)
    baseline = pd.concat(baselines, ignore_index=True) if baselines else pd.DataFrame()
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    reentry_df = pd.concat(reentry, ignore_index=True) if reentry else pd.DataFrame()
    challenger_df = pd.concat(challengers, ignore_index=True) if challengers else pd.DataFrame()
    final_dir = ensure_dir(out / "residual_model")
    write_csv(summary, final_dir / "ablation_summary_all_backends.csv")
    write_csv(consensus, final_dir / "ablation_consensus.csv")
    write_csv(baseline, final_dir / "baseline_all_backends.csv")
    write_csv(reentry_df, out / "diagnostics" / "category_reentry_raw_all_backends.csv")
    write_csv(challenger_df, out / "diagnostics" / "challenger_profiles_raw_all_backends.csv")

    separation = _read_optional_csv(out / "target_separation_residual" / "main" / "target_separation_summary.csv")
    feature_gate = build_residual_feature_gate(registry, consensus, separation, prepared.donor_reference_dir, reentry_df, cfg)
    write_csv(feature_gate, out / "FINAL_RESIDUAL_FEATURE_GATE.csv")
    keep_states = {"KEEP_RESIDUAL_FEATURE", "KEEP_RESIDUAL_GROUP", "CONDITIONAL_RESIDUAL", "CONTROL_KEEP"}
    feature_set = feature_gate[feature_gate.final_residual_decision.isin(keep_states)].column_name.tolist()
    atomic_write_json(out / "FINAL_RESIDUAL_FEATURE_SET.json", {
        "status": "EVIDENCE_GATED",
        "canonical_stage2_features": spec.features,
        "evidence_priority_features": feature_set,
        "stage1_baseline_keys": ["category", "category_major"],
        "donor_is_secondary_only": True,
    })

    reference_path = resolve_path(root, cfg["inputs"]["v1_reference_baseline"])
    v1_reference = _read_optional_csv(reference_path)
    gate = build_model_gate(baseline, reentry_df, challenger_df, cfg, v1_reference, out)
    aggregate_oof_predictions(
        pred,
        out,
        gate["status"],
        canonical_backend=str(cfg["residual_model"].get("canonical_prediction_backend", "CPU")),
    )

    frame = _load_frame(_store_paths(work)["main"])
    coverage, coverage_summary = _competition_coverage(registry, frame)
    write_csv(coverage, out / "FEATURE_COMPETITION_COVERAGE_95.csv")
    write_csv(coverage_summary, out / "FEATURE_COMPETITION_COVERAGE_SUMMARY.csv")
    if not coverage.empty and (coverage.covered != 1).any():
        raise RuntimeError("FEATURE_COMPETITION_COVERAGE_FAIL")

    write_v2_review(out, gate, baseline, reentry_df, challenger_df, feature_gate, coverage_summary)
    run_summary = {
        "status": gate["status"],
        "model": spec.name,
        "stage1": "strict past-only hierarchical category baseline",
        "stage2": "CatBoost on category-normalized residual",
        "canonical_stage2_feature_count": len(spec.features),
        "original_v1_feature_competition_coverage": int(len(coverage)),
        "baseline_runs": int(len(baseline)),
        "ablation_units": int(consensus.unit.nunique()) if not consensus.empty else 0,
        "category_reentry_audit_runs": int(len(reentry_df)),
        "challenger_runs": int(len(challenger_df)),
        "model_gate": gate,
        "donor_is_secondary_only": True,
        "oof_gap_is_preliminary": True,
    }
    atomic_write_json(out / "MAIN_RUN_SUMMARY_V2.json", run_summary)
    (resolve_path(root, cfg["paths"]["output_dir"]) / "LATEST_RESIDUAL_V2_RUN.txt").write_text(str(out), encoding="utf-8")
    print((out / "CATEGORY_RESIDUAL_V2_REVIEW_KO.md").read_text(encoding="utf-8"))


def gpu_check_cmd(args: argparse.Namespace) -> None:
    from catboost import CatBoostRegressor
    rng = np.random.default_rng(42)
    X = rng.normal(size=(1600, 12))
    y = 2 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(size=1600)
    try:
        model = CatBoostRegressor(iterations=40, depth=5, task_type="GPU", devices="0", verbose=False, allow_writing_files=False)
        model.fit(X, y)
        result = {"status": "PASS", "backend": "GPU"}
    except Exception as exc:
        result = {"status": "FAIL", "backend": "GPU", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS" and bool(load_config(args.config)["hardware"]["fail_if_gpu_requested_but_unavailable"]):
        raise SystemExit(2)


def quick_cmd(args: argparse.Namespace) -> None:
    root = project_root()
    cfg = load_config(args.config)
    _, out, _ = _run_paths(cfg, root, args.run_name)
    prepared = _input_context(cfg, root, out)
    _prepare(cfg, root, args.run_name, prepared)
    correlate_cmd(argparse.Namespace(**vars(args), quick=True))
    baseline_cmd(argparse.Namespace(**vars(args), backend="CPU", mode="quick"))
    ablate_cmd(argparse.Namespace(**vars(args), backend="CPU", mode="quick", kind="group", unit_shard_count=1, unit_shard_index=0))
    diagnose_cmd(argparse.Namespace(**vars(args), backend="CPU", mode="quick"))
    finalize_cmd(args)


def full_cmd(args: argparse.Namespace) -> None:
    root = project_root()
    cfg = load_config(args.config)
    _, out, _ = _run_paths(cfg, root, args.run_name)
    prepared = _input_context(cfg, root, out)
    _prepare(cfg, root, args.run_name, prepared)
    correlate_cmd(argparse.Namespace(**vars(args), quick=False))
    for backend in ["CPU", "GPU"]:
        if backend == "GPU":
            gpu_check_cmd(args)
        baseline_cmd(argparse.Namespace(**vars(args), backend=backend, mode="full"))
        ablate_cmd(argparse.Namespace(**vars(args), backend=backend, mode="full", kind="all", unit_shard_count=1, unit_shard_index=0))
        diagnose_cmd(argparse.Namespace(**vars(args), backend=backend, mode="full"))
    finalize_cmd(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pohang category-normalized residual Expected Tourism Spend V2")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--run-name", default="residual_v2")
    parser.add_argument("--force-extract", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [("verify", verify_cmd), ("prepare", prepare_cmd), ("gpu-check", gpu_check_cmd), ("quick", quick_cmd), ("full", full_cmd), ("finalize", finalize_cmd)]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
    sp = sub.add_parser("correlate")
    sp.add_argument("--quick", action="store_true")
    sp.set_defaults(func=correlate_cmd)
    sp = sub.add_parser("baseline")
    sp.add_argument("--backend", choices=["CPU", "GPU"], required=True)
    sp.add_argument("--mode", choices=["quick", "full"], default="full")
    sp.set_defaults(func=baseline_cmd)
    sp = sub.add_parser("ablate")
    sp.add_argument("--backend", choices=["CPU", "GPU"], required=True)
    sp.add_argument("--mode", choices=["quick", "full"], default="full")
    sp.add_argument("--kind", choices=["group", "feature", "redundancy", "all"], default="all")
    sp.add_argument("--unit-shard-count", type=int, default=1)
    sp.add_argument("--unit-shard-index", type=int, default=0)
    sp.set_defaults(func=ablate_cmd)
    sp = sub.add_parser("branch")
    sp.add_argument("--backend", choices=["CPU", "GPU"], required=True)
    sp.add_argument("--mode", choices=["quick", "full"], default="full")
    sp.add_argument("--feature-shard-count", type=int, default=2)
    sp.add_argument("--feature-shard-index", type=int, required=True)
    sp.set_defaults(func=branch_cmd)
    sp = sub.add_parser("diagnose")
    sp.add_argument("--backend", choices=["CPU", "GPU"], required=True)
    sp.add_argument("--mode", choices=["quick", "full"], default="full")
    sp.set_defaults(func=diagnose_cmd)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
