from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import gc
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .utils import ensure_dir, write_json


@dataclass
class RegressionArtifacts:
    model: Any
    backend: str
    metrics: dict[str, float]
    importance: pd.DataFrame
    predictions: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]


def _prepare_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    result = frame[feature_columns].copy()
    for column in categorical_columns:
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    for column in set(feature_columns) - set(categorical_columns):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_regressor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    output_dir: str | Path,
    config: dict[str, Any],
    sample_weight_column: str | None = None,
    key_columns: list[str] | None = None,
    quick: bool = False,
    persist_model: bool = True,
    compute_importance: bool = True,
    write_artifacts: bool = True,
) -> RegressionArtifacts:
    output_path = ensure_dir(output_dir)
    x_train = _prepare_features(train, feature_columns, categorical_columns)
    x_test = _prepare_features(test, feature_columns, categorical_columns)
    y_train = pd.to_numeric(train[target_column], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test[target_column], errors="coerce").to_numpy(dtype=float)
    valid_train = np.isfinite(y_train)
    valid_test = np.isfinite(y_test)
    x_train = x_train.loc[valid_train].reset_index(drop=True)
    x_test = x_test.loc[valid_test].reset_index(drop=True)
    y_train = y_train[valid_train]
    y_test = y_test[valid_test]

    weights = None
    if sample_weight_column and sample_weight_column in train.columns:
        weights = pd.to_numeric(train.loc[valid_train, sample_weight_column], errors="coerce")
        weights = weights.fillna(weights.median()).clip(lower=0).to_numpy(dtype=float)

    model_config = config["model"]
    requested_backend = str(model_config.get("backend", "catboost")).lower()
    backend = requested_backend
    executed_task_type = "CPU"
    model: Any
    importance_values: np.ndarray

    if requested_backend == "catboost":
        try:
            from catboost import CatBoostRegressor

            iterations = int(
                model_config.get("quick_iterations", 100)
                if quick
                else model_config.get("iterations", 450)
            )
            params = {
                "iterations": iterations,
                "learning_rate": float(model_config.get("learning_rate", 0.05)),
                "depth": int(model_config.get("depth", 7)),
                "l2_leaf_reg": float(model_config.get("l2_leaf_reg", 5.0)),
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": int(config["project"].get("random_seed", 42)),
                "verbose": bool(model_config.get("verbose", False)),
                "allow_writing_files": False,
                "task_type": str(model_config.get("task_type", "CPU")),
                "thread_count": int(model_config.get("thread_count", -1)),
            }
            if model_config.get("used_ram_limit"):
                params["used_ram_limit"] = str(model_config["used_ram_limit"])
            if params["task_type"].upper() == "GPU" and model_config.get("devices") is not None:
                params["devices"] = str(model_config["devices"])
                params["gpu_ram_part"] = float(model_config.get("gpu_ram_part", 0.90))
                params["pinned_memory_size"] = str(
                    model_config.get("pinned_memory_size", "2gb")
                )
            model = CatBoostRegressor(**params)
            fit_kwargs: dict[str, Any] = {
                "X": x_train,
                "y": y_train,
                "cat_features": categorical_columns,
                "eval_set": (x_test, y_test),
                "early_stopping_rounds": int(model_config.get("early_stopping_rounds", 50)),
            }
            if weights is not None:
                fit_kwargs["sample_weight"] = weights
            try:
                model.fit(**fit_kwargs)
            except Exception:
                if params["task_type"].upper() != "CPU":
                    params["task_type"] = "CPU"
                    params.pop("devices", None)
                    params.pop("gpu_ram_part", None)
                    params.pop("pinned_memory_size", None)
                    model = CatBoostRegressor(**params)
                    model.fit(**fit_kwargs)
                else:
                    raise
            predictions = model.predict(x_test)
            importance_values = (
                np.asarray(model.get_feature_importance(), dtype=float)
                if compute_importance
                else np.zeros(len(feature_columns), dtype=float)
            )
            if persist_model:
                model.save_model(str(output_path / "model.cbm"))
        except Exception:
            backend = "sklearn_hist_gradient_boosting"
        else:
            backend = "catboost"
            executed_task_type = str(params["task_type"]).upper()
    if backend != "catboost":
        numeric_columns = [column for column in feature_columns if column not in categorical_columns]
        transformer = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                    encoded_missing_value=-1,
                                ),
                            ),
                        ]
                    ),
                    categorical_columns,
                ),
                (
                    "numeric",
                    Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                    numeric_columns,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        model = Pipeline(
            steps=[
                ("preprocess", transformer),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=100 if quick else 300,
                        learning_rate=float(model_config.get("learning_rate", 0.05)),
                        max_leaf_nodes=31,
                        l2_regularization=float(model_config.get("l2_leaf_reg", 5.0)),
                        random_state=int(config["project"].get("random_seed", 42)),
                    ),
                ),
            ]
        )
        fit_params = {}
        if weights is not None:
            fit_params["model__sample_weight"] = weights
        model.fit(x_train, y_train, **fit_params)
        predictions = model.predict(x_test)
        if compute_importance:
            max_rows = int(config["validation"].get("max_permutation_rows", 5000))
            if len(x_test) > max_rows:
                rng = np.random.default_rng(int(config["project"].get("random_seed", 42)))
                sample_index = rng.choice(len(x_test), max_rows, replace=False)
                x_perm = x_test.iloc[sample_index]
                y_perm = y_test[sample_index]
            else:
                x_perm = x_test
                y_perm = y_test
            perm = permutation_importance(
                model,
                x_perm,
                y_perm,
                n_repeats=3 if quick else 5,
                random_state=int(config["project"].get("random_seed", 42)),
                scoring="neg_root_mean_squared_error",
            )
            importance_values = np.asarray(perm.importances_mean, dtype=float)
        else:
            importance_values = np.zeros(len(feature_columns), dtype=float)
        if persist_model:
            joblib.dump(model, output_path / "model.joblib")

    metrics = regression_metrics(y_test, predictions)
    metrics["train_rows"] = int(len(x_train))
    metrics["test_rows"] = int(len(x_test))
    metrics["backend"] = backend
    metrics["requested_task_type"] = str(model_config.get("task_type", "CPU")).upper()
    metrics["executed_task_type"] = executed_task_type
    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": importance_values}
    ).sort_values("importance", ascending=False, ignore_index=True)
    total = importance["importance"].clip(lower=0).sum()
    importance["importance_normalized"] = (
        importance["importance"].clip(lower=0) / total if total > 0 else 0.0
    )

    prediction_frame = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": predictions,
            "residual": y_test - predictions,
        }
    )
    if key_columns:
        key_data = test.loc[valid_test, key_columns].reset_index(drop=True)
        prediction_frame = pd.concat([key_data, prediction_frame], axis=1)

    if write_artifacts:
        write_json(metrics, output_path / "metrics.json")
        importance.to_csv(output_path / "feature_importance.csv", index=False, encoding="utf-8-sig")
        prediction_frame.to_csv(output_path / "test_predictions.csv", index=False, encoding="utf-8-sig")
        write_json(
            {
                "feature_columns": feature_columns,
                "categorical_columns": categorical_columns,
                "target_column": target_column,
                "backend": backend,
                "requested_task_type": str(model_config.get("task_type", "CPU")).upper(),
                "executed_task_type": executed_task_type,
            },
            output_path / "model_spec.json",
        )
    return RegressionArtifacts(
        model=model,
        backend=backend,
        metrics=metrics,
        importance=importance,
        predictions=prediction_frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )


def importance_sum(importance: pd.DataFrame, features: list[str]) -> float:
    lookup = importance.set_index("feature")["importance_normalized"]
    return float(sum(float(lookup.get(feature, 0.0)) for feature in features))


def run_feature_ablation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    output_dir: str | Path,
    config: dict[str, Any],
    baseline_metrics: dict[str, Any],
    sample_weight_column: str | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Retrain once per input feature on the unchanged chronological holdout."""
    output = ensure_dir(output_dir)
    task_type = str(config["model"].get("task_type", "CPU")).upper()
    requested_jobs = int(config["model"].get("ablation_parallel_jobs", 1))
    parallel_jobs = 1 if task_type == "GPU" else max(1, min(requested_jobs, len(feature_columns)))
    logical_cpus = os.cpu_count() or 1

    def evaluate(omitted: str) -> dict[str, Any]:
        remaining = [column for column in feature_columns if column != omitted]
        remaining_categorical = [
            column for column in categorical_columns if column != omitted
        ]
        local_config = deepcopy(config)
        if parallel_jobs > 1:
            local_config["model"]["thread_count"] = max(1, logical_cpus // parallel_jobs)
        row: dict[str, Any] = {
            "omitted_feature": omitted,
            "feature_type": "categorical" if omitted in categorical_columns else "numeric",
            "status": "PASS",
        }
        try:
            artifacts = fit_regressor(
                train,
                test,
                remaining,
                remaining_categorical,
                target_column,
                output,
                local_config,
                sample_weight_column=sample_weight_column,
                quick=quick,
                persist_model=False,
                compute_importance=False,
                write_artifacts=False,
            )
            metrics = artifacts.metrics
            row.update(
                {
                    "backend": artifacts.backend,
                    "ablated_mae": float(metrics["mae"]),
                    "mae_increase": float(metrics["mae"] - baseline_metrics["mae"]),
                    "ablated_rmse": float(metrics["rmse"]),
                    "rmse_increase": float(metrics["rmse"] - baseline_metrics["rmse"]),
                    "relative_rmse_increase_pct": float(
                        100.0
                        * (metrics["rmse"] - baseline_metrics["rmse"])
                        / max(abs(float(baseline_metrics["rmse"])), np.finfo(float).eps)
                    ),
                    "ablated_r2": float(metrics["r2"]),
                    "r2_drop": float(baseline_metrics["r2"] - metrics["r2"]),
                    "train_rows": int(metrics["train_rows"]),
                    "test_rows": int(metrics["test_rows"]),
                    "error": "",
                }
            )
            del artifacts
        except Exception as exc:  # pragma: no cover - retained as a complete audit row
            row.update(
                {
                    "status": "FAILED",
                    "backend": "",
                    "ablated_mae": np.nan,
                    "mae_increase": np.nan,
                    "ablated_rmse": np.nan,
                    "rmse_increase": np.nan,
                    "relative_rmse_increase_pct": np.nan,
                    "ablated_r2": np.nan,
                    "r2_drop": np.nan,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        return row

    if parallel_jobs == 1:
        rows = [evaluate(feature) for feature in feature_columns]
    else:
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            rows = list(executor.map(evaluate, feature_columns))
    gc.collect()

    result = pd.DataFrame(rows).sort_values(
        ["status", "r2_drop", "rmse_increase"],
        ascending=[False, False, False],
        na_position="last",
        ignore_index=True,
    )
    result.to_csv(output / "feature_ablation.csv", index=False, encoding="utf-8-sig")
    write_json(dict(baseline_metrics), output / "baseline_metrics.json")
    passed = result[result["status"] == "PASS"]
    summary = {
        "tested_feature_count": int(len(feature_columns)),
        "passed_count": int(len(passed)),
        "failed_count": int((result["status"] == "FAILED").sum()),
        "parallel_jobs": parallel_jobs,
        "threads_per_job": int(
            config["model"].get("thread_count", -1)
            if parallel_jobs == 1
            else max(1, logical_cpus // parallel_jobs)
        ),
        "baseline_metrics": dict(baseline_metrics),
        "most_harmful_removal": (
            str(passed.iloc[0]["omitted_feature"]) if not passed.empty else None
        ),
        "features_with_positive_r2_contribution": int((passed["r2_drop"] > 0).sum()),
        "interpretation": (
            "r2_drop>0 또는 rmse_increase>0이면 해당 피처 제거 시 시간 홀드아웃 성능이 악화됨을 뜻한다. "
            "이는 donor 내부 예측 기여도이며 포항의 효과크기나 인과효과가 아니다."
        ),
    }
    write_json(summary, output / "summary.json")
    return summary
