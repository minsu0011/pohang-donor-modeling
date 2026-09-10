from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def setup_logging(path: Path) -> logging.Logger:
    ensure_dir(path.parent)
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def normalize_district(value: Any) -> str:
    s = str(value).strip()
    if s in {"남구", "포항 남구", "경상북도 포항시 남구"}:
        return "포항시 남구"
    if s in {"북구", "포항 북구", "경상북도 포항시 북구"}:
        return "포항시 북구"
    return s


def ym_to_period(value: Any) -> pd.Period:
    s = re.sub(r"[^0-9]", "", str(value))[:6]
    if len(s) != 6:
        return pd.NaT
    return pd.Period(f"{s[:4]}-{s[4:]}", freq="M")


def period_to_ym(value: pd.Period) -> int:
    return int(value.strftime("%Y%m"))


def period_range(start: pd.Period, end: pd.Period) -> list[pd.Period]:
    return list(pd.period_range(start, end, freq="M"))


def robust_minmax(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    lo = float(valid.quantile(lower))
    hi = float(valid.quantile(upper))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=series.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).clip(0.0, 1.0)


def normalized_entropy(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size <= 1:
        return 0.0
    p = arr / arr.sum()
    h = -float(np.sum(p * np.log(p)))
    return h / float(np.log(len(p)))


def safe_divide(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    out = np.full_like(aa, np.nan, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb) & (bb != 0)
    out[mask] = aa[mask] / bb[mask]
    return out


def choose_font(candidates: list[str]) -> str:
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in installed:
                return name
    except Exception:
        pass
    return "DejaVu Sans"


def safe_entropy(values: Iterable[float]) -> float:
    """Normalized Shannon entropy for non-negative counts/shares."""
    return normalized_entropy(values)


def safe_hhi(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if arr.size == 0 or arr.sum() <= 0:
        return float("nan")
    p = arr / arr.sum()
    return float(np.sum(p ** 2))


def month_distance(left_yyyymm: pd.Series, right_yyyymm: pd.Series) -> pd.Series:
    left = pd.to_numeric(left_yyyymm, errors="coerce")
    right = pd.to_numeric(right_yyyymm, errors="coerce")
    ly = np.floor(left / 100)
    lm = left % 100
    ry = np.floor(right / 100)
    rm = right % 100
    return pd.Series((ly - ry) * 12 + (lm - rm), index=left_yyyymm.index, dtype="float64")


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically to avoid partial run metadata."""
    ensure_dir(path.parent)
    tmp = path.parent / f".atomic.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _replace_with_windows_retry(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.parent / f".atomic.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        _replace_with_windows_retry(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _replace_with_windows_retry(tmp: Path, path: Path) -> None:
    for attempt in range(100):
        try:
            os.replace(tmp, path); return
        except PermissionError:
            if attempt == 99: raise
            time.sleep(min(0.005 * (attempt + 1), 0.05))


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataframe_identity(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Stable data identity independent of gzip timestamps and file metadata."""
    cols = list(columns) if columns is not None else list(frame.columns)
    x = frame.loc[:, cols]
    h = hashlib.sha256()
    metadata = [(str(c), str(x[c].dtype)) for c in cols]
    h.update(json.dumps(metadata, ensure_ascii=False, sort_keys=False).encode("utf-8"))
    row_hash = pd.util.hash_pandas_object(x, index=True, categorize=True).to_numpy(dtype="uint64", copy=False)
    h.update(row_hash.tobytes())
    return h.hexdigest()


def minmax_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    rank = x.rank(pct=True, method="average", ascending=not higher_is_better)
    return rank.fillna(0.5)
