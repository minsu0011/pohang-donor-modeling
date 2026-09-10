from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def setup_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("pohang_donor_models")
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_path is not None:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"JSON 직렬화 불가 타입: {type(value)!r}")


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    result = np.full_like(num, np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (den != 0)
    result[valid] = num[valid] / den[valid]
    return result


def minmax_score(values: Iterable[float], neutral: float = 0.5) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    finite = np.isfinite(arr)
    result = np.full(arr.shape, neutral, dtype=float)
    if not finite.any():
        return result
    lo = np.nanmin(arr[finite])
    hi = np.nanmax(arr[finite])
    if math.isclose(lo, hi):
        result[finite] = neutral
    else:
        result[finite] = (arr[finite] - lo) / (hi - lo)
    return np.clip(result, 0.0, 1.0)


def parse_ym(series: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce").astype("Int64")
    year = values // 100
    month = values % 100
    return pd.DataFrame({"year": year.astype(float), "month": month.astype(float)})


def quarter_to_index(value: int | str) -> int:
    text = str(int(value))
    if len(text) != 5:
        raise ValueError(f"예상하지 못한 분기코드: {value}")
    year = int(text[:4])
    quarter = int(text[4])
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"예상하지 못한 분기값: {value}")
    return year * 4 + (quarter - 1)


def chronological_split(
    frame: pd.DataFrame,
    period_col: str,
    test_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Any]]:
    periods = sorted(pd.Series(frame[period_col].dropna().unique()).tolist())
    if len(periods) <= test_periods:
        raise ValueError(
            f"기간 수({len(periods)})가 테스트 기간 수({test_periods})보다 작거나 같습니다."
        )
    test_values = periods[-test_periods:]
    train = frame[~frame[period_col].isin(test_values)].copy()
    test = frame[frame[period_col].isin(test_values)].copy()
    return train, test, test_values


def weighted_entropy(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    probabilities = arr[arr > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def hhi(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    shares = arr / total
    return float(np.square(shares).sum())


def yearly_sign_stability(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    year_col: str = "year",
    minimum_rows: int = 30,
) -> dict[str, float]:
    correlations: list[float] = []
    for _, group in frame.groupby(year_col, dropna=True):
        subset = group[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(subset) < minimum_rows or subset[feature].nunique() < 2 or subset[target].nunique() < 2:
            continue
        corr = spearmanr(subset[feature], subset[target]).correlation
        if np.isfinite(corr):
            correlations.append(float(corr))
    if not correlations:
        return {"stability": 0.5, "mean_abs_correlation": 0.0, "year_count": 0}
    signs = np.sign(correlations)
    positive = float((signs >= 0).mean())
    negative = float((signs <= 0).mean())
    stability = max(positive, negative)
    return {
        "stability": stability,
        "mean_abs_correlation": float(np.mean(np.abs(correlations))),
        "year_count": len(correlations),
    }


def write_markdown_table(frame: pd.DataFrame, path: str | Path, title: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        if frame.empty:
            handle.write("결과가 없습니다.\n")
        else:
            handle.write(frame.to_markdown(index=False))
            handle.write("\n")
