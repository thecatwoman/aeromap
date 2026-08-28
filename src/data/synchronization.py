from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_REFERENCE_COLUMNS = [
    "carspeed_art",
    "pair",
    "pitot_c",
    "pitch_rh",
]


@dataclass(frozen=True)
class SynchronizationResult:
    synced_df: pd.DataFrame
    method: str
    axis_method: str
    reference_columns_used: tuple[str, ...]
    lag_steps: int
    lag_progress: float
    real_points_used: int
    simu_points_used: int
    grid_size: int


def _normalized_axis_from_sample_index(df: pd.DataFrame) -> pd.Series:
    if len(df) <= 1:
        return pd.Series([0.0] * len(df), index=df.index, dtype="float64")

    return pd.Series(
        np.linspace(0.0, 1.0, len(df)),
        index=df.index,
        dtype="float64",
    )


def _normalized_axis_from_column(df: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in df.columns:
        return None

    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.dropna()
    if len(valid) < 2:
        return None

    value_min = valid.min()
    value_max = valid.max()
    if not np.isfinite(value_min) or not np.isfinite(value_max) or value_max <= value_min:
        return None

    normalized = (values - value_min) / (value_max - value_min)
    return normalized.clip(0.0, 1.0)


def build_normalized_sync_axis(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for column in ["distancelap", "distance"]:
        axis = _normalized_axis_from_column(df, column)
        if axis is not None:
            return axis, f"normalized_{column}"

    return _normalized_axis_from_sample_index(df), "normalized_sample_index"


def _interpolate_series(
    axis: pd.Series,
    values: pd.Series,
    grid: np.ndarray,
) -> tuple[np.ndarray, int]:
    clean = pd.DataFrame({"axis": axis, "value": values}).dropna()
    clean = clean[np.isfinite(clean["axis"]) & np.isfinite(clean["value"])]
    clean = clean.sort_values("axis")
    clean = clean.drop_duplicates(subset="axis", keep="first")

    if len(clean) < 2:
        return np.full_like(grid, np.nan, dtype="float64"), int(len(clean))

    interpolated = np.interp(
        grid,
        clean["axis"].to_numpy(dtype="float64"),
        clean["value"].to_numpy(dtype="float64"),
    )
    return interpolated, int(len(clean))


def _zscore(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.full_like(values, np.nan, dtype="float64")

    centered = values.copy()
    mean = np.nanmean(centered)
    std = np.nanstd(centered)
    if not np.isfinite(std) or std == 0:
        return np.full_like(values, np.nan, dtype="float64")

    centered[valid] = (centered[valid] - mean) / std
    centered[~valid] = np.nan
    return centered


def _build_reference_trace(
    df: pd.DataFrame,
    axis: pd.Series,
    grid: np.ndarray,
    reference_columns: list[str],
) -> tuple[np.ndarray, tuple[str, ...], int]:
    traces: list[np.ndarray] = []
    columns_used: list[str] = []
    max_points_used = 0

    for column in reference_columns:
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        interpolated, points_used = _interpolate_series(axis, values, grid)
        standardized = _zscore(interpolated)
        if np.isfinite(standardized).sum() < 3:
            continue

        traces.append(standardized)
        columns_used.append(column)
        max_points_used = max(max_points_used, points_used)

    if not traces:
        return np.full_like(grid, np.nan, dtype="float64"), tuple(), 0

    stacked = np.vstack(traces)
    trace = np.nanmean(stacked, axis=0)
    return trace, tuple(columns_used), max_points_used


def _shift_array(values: np.ndarray, lag_steps: int) -> np.ndarray:
    shifted = np.full_like(values, np.nan, dtype="float64")

    if lag_steps == 0:
        shifted[:] = values
        return shifted

    if lag_steps > 0:
        shifted[lag_steps:] = values[:-lag_steps]
        return shifted

    shifted[:lag_steps] = values[-lag_steps:]
    return shifted


def _correlation_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    mask = np.isfinite(reference) & np.isfinite(candidate)
    if mask.sum() < 3:
        return -np.inf

    ref = reference[mask]
    cand = candidate[mask]
    ref_std = np.std(ref)
    cand_std = np.std(cand)
    if ref_std == 0 or cand_std == 0:
        return -np.inf

    return float(np.corrcoef(ref, cand)[0, 1])


def _best_lag_via_cross_correlation(
    real_trace: np.ndarray,
    simu_trace: np.ndarray,
    max_lag_steps: int,
) -> tuple[int, float]:
    best_lag = 0
    best_score = _correlation_score(real_trace, simu_trace)

    for lag_steps in range(-max_lag_steps, max_lag_steps + 1):
        shifted = _shift_array(simu_trace, lag_steps)
        score = _correlation_score(real_trace, shifted)
        if score > best_score:
            best_score = score
            best_lag = lag_steps

    return best_lag, best_score


def synchronize_dataframes(
    real_df: pd.DataFrame,
    simu_df: pd.DataFrame,
    columns: list[str],
    grid_size: int = 200,
    reference_columns: list[str] | None = None,
    max_lag_fraction: float = 0.15,
) -> SynchronizationResult:
    if reference_columns is None:
        reference_columns = DEFAULT_REFERENCE_COLUMNS

    real_axis, real_axis_method = build_normalized_sync_axis(real_df)
    simu_axis, simu_axis_method = build_normalized_sync_axis(simu_df)
    grid = np.linspace(0.0, 1.0, grid_size)

    real_reference, real_reference_cols, real_ref_points = _build_reference_trace(
        real_df,
        real_axis,
        grid,
        reference_columns,
    )
    simu_reference, simu_reference_cols, simu_ref_points = _build_reference_trace(
        simu_df,
        simu_axis,
        grid,
        reference_columns,
    )

    shared_reference_cols = tuple(
        column for column in real_reference_cols if column in simu_reference_cols
    )
    max_lag_steps = max(1, int(round(grid_size * max_lag_fraction)))

    if shared_reference_cols and np.isfinite(real_reference).sum() >= 3 and np.isfinite(simu_reference).sum() >= 3:
        lag_steps, _ = _best_lag_via_cross_correlation(
            real_trace=real_reference,
            simu_trace=simu_reference,
            max_lag_steps=max_lag_steps,
        )
        method = "cross_correlation"
    else:
        lag_steps = 0
        method = "no_reference_fallback"

    synced = pd.DataFrame({"sync_progress": grid})
    real_points_used = 0
    simu_points_used = 0

    for column in columns:
        if column in real_df.columns:
            real_values = pd.to_numeric(real_df[column], errors="coerce")
        else:
            real_values = pd.Series(dtype="float64")

        if column in simu_df.columns:
            simu_values = pd.to_numeric(simu_df[column], errors="coerce")
        else:
            simu_values = pd.Series(dtype="float64")

        real_interp, real_used = _interpolate_series(real_axis, real_values, grid)
        simu_interp, simu_used = _interpolate_series(simu_axis, simu_values, grid)
        simu_shifted = _shift_array(simu_interp, lag_steps)

        synced[f"real__{column}"] = real_interp
        synced[f"simu__{column}"] = simu_shifted

        real_points_used = max(real_points_used, real_used)
        simu_points_used = max(simu_points_used, simu_used)

    axis_method = f"real={real_axis_method}, simu={simu_axis_method}"
    return SynchronizationResult(
        synced_df=synced,
        method=method,
        axis_method=axis_method,
        reference_columns_used=shared_reference_cols,
        lag_steps=lag_steps,
        lag_progress=lag_steps / max(grid_size - 1, 1),
        real_points_used=max(real_points_used, real_ref_points),
        simu_points_used=max(simu_points_used, simu_ref_points),
        grid_size=grid_size,
    )
