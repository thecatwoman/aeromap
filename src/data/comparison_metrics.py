from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

ROLLING_MEDIAN_WINDOW = 9
STABLE_SIGN_MIN_FRACTION = 0.80
STABLE_RUN_MIN_FRACTION = 0.25
DTW_MAX_POINTS = 80


@dataclass(frozen=True)
class SignalComparisonMetrics:
    sample_count: int
    median_real: float | None
    median_sim: float | None
    median_delta: float | None
    mean_error: float | None
    rmse: float | None
    nrmse: float | None
    mae: float | None
    pearson_r: float | None


@dataclass(frozen=True)
class ThresholdComparisonEvidence:
    state: str
    sample_count: int
    median_delta: float | None
    raw_median_delta: float | None
    mean_error: float | None
    rmse: float | None
    nrmse: float | None
    mae: float | None
    pearson_r: float | None
    within_fraction: float | None
    above_fraction: float | None
    below_fraction: float | None
    positive_fraction: float | None
    negative_fraction: float | None
    longest_above_run: int | None
    longest_below_run: int | None
    ci_low: float | None
    ci_high: float | None
    iqr: float | None
    mad: float | None
    hodges_lehmann_shift: float | None
    real_q25: float | None
    real_q50: float | None
    real_q75: float | None
    simu_q25: float | None
    simu_q50: float | None
    simu_q75: float | None
    iqr_overlap_ratio: float | None
    detail: str


@dataclass(frozen=True)
class TimeSeriesAgreementEvidence:
    sample_count: int
    best_lag_steps: int | None
    best_correlation: float | None
    normalized_area_between_curves: float | None
    mean_signed_error: float | None
    mean_absolute_error: float | None
    dtw_distance: float | None
    detail: str


@dataclass(frozen=True)
class FitQualityEvidence:
    sample_count: int
    r_squared: float | None
    residual_median: float | None
    residual_iqr: float | None
    residual_mad: float | None
    within_tolerance_fraction: float | None
    detail: str


def _to_aligned_numeric_arrays(
    real_series: pd.Series,
    simu_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    aligned = pd.DataFrame(
        {
            "real": pd.to_numeric(real_series, errors="coerce"),
            "simu": pd.to_numeric(simu_series, errors="coerce"),
        }
    ).dropna()

    if aligned.empty:
        return np.array([], dtype="float64"), np.array([], dtype="float64")

    return (
        aligned["real"].to_numpy(dtype="float64"),
        aligned["simu"].to_numpy(dtype="float64"),
    )


def _rolling_median_series(
    series: pd.Series,
    window: int = ROLLING_MEDIAN_WINDOW,
) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .rolling(window=window, center=True, min_periods=1)
        .median()
    )


def compute_signal_comparison_metrics(
    real_series: pd.Series,
    simu_series: pd.Series,
) -> SignalComparisonMetrics:
    real_values, simu_values = _to_aligned_numeric_arrays(real_series, simu_series)
    sample_count = int(len(real_values))

    if sample_count == 0:
        return SignalComparisonMetrics(
            sample_count=0,
            median_real=None,
            median_sim=None,
            median_delta=None,
            mean_error=None,
            rmse=None,
            nrmse=None,
            mae=None,
            pearson_r=None,
        )

    error = real_values - simu_values
    median_real = float(np.median(real_values))
    median_sim = float(np.median(simu_values))
    median_delta = float(median_real - median_sim)
    mean_error = float(np.mean(error))
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))

    operating_window = float(np.nanmax(simu_values) - np.nanmin(simu_values))
    if not np.isfinite(operating_window) or operating_window == 0:
        nrmse = None
    else:
        nrmse = float(rmse / operating_window)

    if sample_count < 2:
        pearson_r = None
    else:
        real_std = float(np.std(real_values))
        simu_std = float(np.std(simu_values))
        if real_std == 0.0 or simu_std == 0.0:
            pearson_r = None
        else:
            pearson_r = float(np.corrcoef(real_values, simu_values)[0, 1])

    return SignalComparisonMetrics(
        sample_count=sample_count,
        median_real=median_real,
        median_sim=median_sim,
        median_delta=median_delta,
        mean_error=mean_error,
        rmse=rmse,
        nrmse=nrmse,
        mae=mae,
        pearson_r=pearson_r,
    )


def format_metric_value(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    return f"{value:.4f}{suffix}"


def _aligned_numeric_frame(
    real_series: pd.Series,
    simu_series: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "real": pd.to_numeric(real_series, errors="coerce"),
            "simu": pd.to_numeric(simu_series, errors="coerce"),
        }
    ).dropna()


def _iqr(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, 75.0) - np.percentile(values, 25.0))


def _median_absolute_deviation(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _quantiles(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    return (
        float(np.percentile(values, 25.0)),
        float(np.percentile(values, 50.0)),
        float(np.percentile(values, 75.0)),
    )


def _iqr_overlap_ratio(
    real_values: np.ndarray,
    simu_values: np.ndarray,
) -> float | None:
    real_q25, _, real_q75 = _quantiles(real_values)
    simu_q25, _, simu_q75 = _quantiles(simu_values)
    if None in {real_q25, real_q75, simu_q25, simu_q75}:
        return None

    overlap_low = max(real_q25, simu_q25)
    overlap_high = min(real_q75, simu_q75)
    union_low = min(real_q25, simu_q25)
    union_high = max(real_q75, simu_q75)
    union_span = union_high - union_low
    if union_span <= 0:
        return 1.0
    return float(max(0.0, overlap_high - overlap_low) / union_span)


def _hodges_lehmann_shift(
    real_values: np.ndarray,
    simu_values: np.ndarray,
) -> float | None:
    if real_values.size == 0 or simu_values.size == 0:
        return None
    diffs = real_values[:, None] - simu_values[None, :]
    return float(np.median(diffs))


def _longest_true_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0

    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)


def _resample_to_fixed_length(values: np.ndarray, target_size: int) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype="float64")
    if values.size == target_size:
        return values.astype("float64")
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, target_size)
    return np.interp(x_new, x_old, values).astype("float64")


def _dtw_distance(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0:
        return None
    n = min(DTW_MAX_POINTS, x.size)
    m = min(DTW_MAX_POINTS, y.size)
    x_small = _resample_to_fixed_length(x, n)
    y_small = _resample_to_fixed_length(y, m)
    cost = np.full((n + 1, m + 1), np.inf, dtype="float64")
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = abs(x_small[i - 1] - y_small[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / max(n, m))


def compute_time_series_agreement_evidence(
    real_series: pd.Series,
    simu_series: pd.Series,
    max_lag_fraction: float = 0.25,
) -> TimeSeriesAgreementEvidence:
    real_values, simu_values = _to_aligned_numeric_arrays(real_series, simu_series)
    sample_count = int(len(real_values))
    if sample_count == 0:
        return TimeSeriesAgreementEvidence(
            sample_count=0,
            best_lag_steps=None,
            best_correlation=None,
            normalized_area_between_curves=None,
            mean_signed_error=None,
            mean_absolute_error=None,
            dtw_distance=None,
            detail="time-series agreement unavailable (missing valid values)",
        )

    max_lag = max(0, int(ceil(sample_count * max_lag_fraction)))
    best_lag_steps = 0
    best_correlation = None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            real_slice = real_values[:lag]
            simu_slice = simu_values[-lag:]
        elif lag > 0:
            real_slice = real_values[lag:]
            simu_slice = simu_values[:-lag]
        else:
            real_slice = real_values
            simu_slice = simu_values
        if real_slice.size < 2 or simu_slice.size < 2:
            continue
        real_std = float(np.std(real_slice))
        simu_std = float(np.std(simu_slice))
        if real_std == 0.0 or simu_std == 0.0:
            continue
        corr = float(np.corrcoef(real_slice, simu_slice)[0, 1])
        if best_correlation is None or corr > best_correlation:
            best_correlation = corr
            best_lag_steps = lag

    error = real_values - simu_values
    mean_signed_error = float(np.mean(error))
    mean_absolute_error = float(np.mean(np.abs(error)))
    operating_window = float(max(np.nanmax(real_values), np.nanmax(simu_values)) - min(np.nanmin(real_values), np.nanmin(simu_values)))
    if not np.isfinite(operating_window) or operating_window == 0.0:
        normalized_area_between_curves = None
    else:
        normalized_area_between_curves = float(np.mean(np.abs(error)) / operating_window)

    dtw_distance = _dtw_distance(real_values, simu_values)
    return TimeSeriesAgreementEvidence(
        sample_count=sample_count,
        best_lag_steps=best_lag_steps,
        best_correlation=best_correlation,
        normalized_area_between_curves=normalized_area_between_curves,
        mean_signed_error=mean_signed_error,
        mean_absolute_error=mean_absolute_error,
        dtw_distance=dtw_distance,
        detail=(
            f"time-series: lag_steps={best_lag_steps}, "
            f"best_corr={format_metric_value(best_correlation)}, "
            f"normalized_area={format_metric_value(normalized_area_between_curves)}, "
            f"mse_signed={format_metric_value(mean_signed_error)}, "
            f"mae={format_metric_value(mean_absolute_error)}, "
            f"dtw={format_metric_value(dtw_distance)}"
        ),
    )


def compute_fit_quality_evidence(
    x: pd.Series,
    y: pd.Series,
    slope: float | None,
    intercept: float | None,
    tolerance: float,
) -> FitQualityEvidence:
    clean = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    clean = clean[np.isfinite(clean["x"]) & np.isfinite(clean["y"])]
    sample_count = int(len(clean))
    if sample_count < 3 or slope is None or intercept is None:
        return FitQualityEvidence(
            sample_count=sample_count,
            r_squared=None,
            residual_median=None,
            residual_iqr=None,
            residual_mad=None,
            within_tolerance_fraction=None,
            detail="fit quality unavailable (insufficient fit data)",
        )

    predicted = slope * clean["x"].to_numpy(dtype="float64") + intercept
    observed = clean["y"].to_numpy(dtype="float64")
    residuals = observed - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    residual_median = float(np.median(residuals))
    residual_iqr = _iqr(residuals)
    residual_mad = _median_absolute_deviation(residuals)
    within_tolerance_fraction = float(np.mean(np.abs(residuals) <= tolerance))
    return FitQualityEvidence(
        sample_count=sample_count,
        r_squared=r_squared,
        residual_median=residual_median,
        residual_iqr=residual_iqr,
        residual_mad=residual_mad,
        within_tolerance_fraction=within_tolerance_fraction,
        detail=(
            f"fit_quality: r2={format_metric_value(r_squared)}, "
            f"residual_median={format_metric_value(residual_median)}, "
            f"residual_iqr={format_metric_value(residual_iqr)}, "
            f"residual_mad={format_metric_value(residual_mad)}, "
            f"within_tol={within_tolerance_fraction:.1%}"
        ),
    )

def evaluate_absolute_threshold_comparison(
    real_series: pd.Series,
    simu_series: pd.Series,
    threshold: float,
    channel_name: str,
    higher_state: str,
    lower_state: str,
    within_min_fraction: float = 0.65,
) -> ThresholdComparisonEvidence:
    aligned = _aligned_numeric_frame(real_series, simu_series)
    metrics = compute_signal_comparison_metrics(aligned["real"], aligned["simu"])

    if aligned.empty:
        return ThresholdComparisonEvidence(
            state="unavailable",
            sample_count=0,
            median_delta=None,
            raw_median_delta=None,
            mean_error=None,
            rmse=None,
            nrmse=None,
            mae=None,
            pearson_r=None,
            within_fraction=None,
            above_fraction=None,
            below_fraction=None,
            positive_fraction=None,
            negative_fraction=None,
            longest_above_run=None,
            longest_below_run=None,
            ci_low=None,
            ci_high=None,
            iqr=None,
            mad=None,
            hodges_lehmann_shift=None,
            real_q25=None,
            real_q50=None,
            real_q75=None,
            simu_q25=None,
            simu_q50=None,
            simu_q75=None,
            iqr_overlap_ratio=None,
            detail=f"{channel_name}: comparison unavailable (missing valid values)",
        )

    rolling_real = _rolling_median_series(aligned["real"])
    rolling_simu = _rolling_median_series(aligned["simu"])
    rolling_aligned = pd.DataFrame(
        {
            "real": rolling_real,
            "simu": rolling_simu,
        }
    ).dropna()

    diff = (
        rolling_aligned["real"].to_numpy(dtype="float64")
        - rolling_aligned["simu"].to_numpy(dtype="float64")
    )
    within_fraction = float(np.mean(np.abs(diff) <= threshold))
    above_fraction = float(np.mean(diff > threshold))
    below_fraction = float(np.mean(diff < -threshold))
    positive_fraction = float(np.mean(diff > 0.0))
    negative_fraction = float(np.mean(diff < 0.0))
    longest_above_run = _longest_true_run(diff > threshold)
    longest_below_run = _longest_true_run(diff < -threshold)
    spread_iqr = _iqr(diff)
    spread_mad = _median_absolute_deviation(diff)
    rolling_median_real = float(np.median(rolling_aligned["real"].to_numpy(dtype="float64")))
    rolling_median_simu = float(np.median(rolling_aligned["simu"].to_numpy(dtype="float64")))
    rolling_median_delta = float(rolling_median_real - rolling_median_simu)
    real_values = aligned["real"].to_numpy(dtype="float64")
    simu_values = aligned["simu"].to_numpy(dtype="float64")
    real_q25, real_q50, real_q75 = _quantiles(real_values)
    simu_q25, simu_q50, simu_q75 = _quantiles(simu_values)
    hl_shift = _hodges_lehmann_shift(real_values, simu_values)
    iqr_overlap = _iqr_overlap_ratio(real_values, simu_values)

    if metrics.median_delta is None:
        state = "unavailable"
    elif abs(rolling_median_delta) <= threshold:
        state = "within_threshold"
    elif rolling_median_delta > threshold:
        state = higher_state
    elif rolling_median_delta < -threshold:
        state = lower_state
    else:
        state = "ambiguous"

    return ThresholdComparisonEvidence(
        state=state,
        sample_count=metrics.sample_count,
        median_delta=rolling_median_delta,
        raw_median_delta=metrics.median_delta,
        mean_error=metrics.mean_error,
        rmse=metrics.rmse,
        nrmse=metrics.nrmse,
        mae=metrics.mae,
        pearson_r=metrics.pearson_r,
        within_fraction=within_fraction,
        above_fraction=above_fraction,
        below_fraction=below_fraction,
        positive_fraction=positive_fraction,
        negative_fraction=negative_fraction,
        longest_above_run=longest_above_run,
        longest_below_run=longest_below_run,
        ci_low=None,
        ci_high=None,
        iqr=spread_iqr,
        mad=spread_mad,
        hodges_lehmann_shift=hl_shift,
        real_q25=real_q25,
        real_q50=real_q50,
        real_q75=real_q75,
        simu_q25=simu_q25,
        simu_q50=simu_q50,
        simu_q75=simu_q75,
        iqr_overlap_ratio=iqr_overlap,
        detail=(
            f"{channel_name}: rolling_median_delta={format_metric_value(rolling_median_delta)}, "
            f"raw_median_delta={format_metric_value(metrics.median_delta)}, "
            f"iqr={format_metric_value(spread_iqr)}, "
            f"mad={format_metric_value(spread_mad)}, "
            f"hl_shift={format_metric_value(hl_shift)}, "
            f"mean_error={format_metric_value(metrics.mean_error)}, "
            f"within={within_fraction:.1%}, above={above_fraction:.1%}, "
            f"below={below_fraction:.1%}, positive={positive_fraction:.1%}, "
            f"negative={negative_fraction:.1%}, "
            f"longest_above_run={longest_above_run}, longest_below_run={longest_below_run}, "
            f"iqr_overlap={format_metric_value(None if iqr_overlap is None else 100.0 * iqr_overlap, '%')}, "
            f"pearson_r={format_metric_value(metrics.pearson_r)}"
        ),
    )
    

def evaluate_relative_threshold_comparison(
    real_series: pd.Series,
    reference_series: pd.Series,
    relative_threshold: float,
    channel_name: str,
    lower_state: str,
    higher_state: str,
    within_min_fraction: float = 0.65,
) -> ThresholdComparisonEvidence:
    aligned = _aligned_numeric_frame(real_series, reference_series)
    metrics = compute_signal_comparison_metrics(aligned["real"], aligned["simu"])

    if aligned.empty:
        return ThresholdComparisonEvidence(
            state="unavailable",
            sample_count=0,
            median_delta=None,
            raw_median_delta=None,
            mean_error=None,
            rmse=None,
            nrmse=None,
            mae=None,
            pearson_r=None,
            within_fraction=None,
            above_fraction=None,
            below_fraction=None,
            positive_fraction=None,
            negative_fraction=None,
            longest_above_run=None,
            longest_below_run=None,
            ci_low=None,
            ci_high=None,
            iqr=None,
            mad=None,
            hodges_lehmann_shift=None,
            real_q25=None,
            real_q50=None,
            real_q75=None,
            simu_q25=None,
            simu_q50=None,
            simu_q75=None,
            iqr_overlap_ratio=None,
            detail=f"{channel_name}: comparison unavailable (missing valid values)",
        )

    rolling_real = _rolling_median_series(aligned["real"])
    rolling_simu = _rolling_median_series(aligned["simu"])
    rolling_aligned = pd.DataFrame(
        {
            "real": rolling_real,
            "simu": rolling_simu,
        }
    ).dropna()

    diff = (
        rolling_aligned["real"].to_numpy(dtype="float64")
        - rolling_aligned["simu"].to_numpy(dtype="float64")
    )
    scale = np.maximum(np.abs(rolling_aligned["simu"].to_numpy(dtype="float64")), 1e-9)
    relative_diff = diff / scale

    within_fraction = float(np.mean(np.abs(relative_diff) <= relative_threshold))
    above_fraction = float(np.mean(relative_diff > relative_threshold))
    below_fraction = float(np.mean(relative_diff < -relative_threshold))
    positive_fraction = float(np.mean(relative_diff > 0.0))
    negative_fraction = float(np.mean(relative_diff < 0.0))
    longest_above_run = _longest_true_run(relative_diff > relative_threshold)
    longest_below_run = _longest_true_run(relative_diff < -relative_threshold)
    mean_relative_error = float(np.mean(relative_diff))
    rolling_median_relative_error = float(np.median(relative_diff))
    raw_scale = np.maximum(np.abs(aligned["simu"].to_numpy(dtype="float64")), 1e-9)
    raw_relative_diff = (
        aligned["real"].to_numpy(dtype="float64") - aligned["simu"].to_numpy(dtype="float64")
    ) / raw_scale
    raw_median_relative_error = float(np.median(raw_relative_diff))
    spread_iqr = _iqr(relative_diff)
    spread_mad = _median_absolute_deviation(relative_diff)
    real_values = aligned["real"].to_numpy(dtype="float64")
    simu_values = aligned["simu"].to_numpy(dtype="float64")
    real_q25, real_q50, real_q75 = _quantiles(real_values)
    simu_q25, simu_q50, simu_q75 = _quantiles(simu_values)
    hl_shift = _hodges_lehmann_shift(real_values, simu_values)
    iqr_overlap = _iqr_overlap_ratio(real_values, simu_values)

    if abs(rolling_median_relative_error) <= relative_threshold:
        state = "within_threshold"
    elif rolling_median_relative_error > relative_threshold:
        state = higher_state
    elif rolling_median_relative_error < -relative_threshold:
        state = lower_state
    else:
        state = "ambiguous"

    return ThresholdComparisonEvidence(
        state=state,
        sample_count=metrics.sample_count,
        median_delta=rolling_median_relative_error,
        raw_median_delta=raw_median_relative_error,
        mean_error=metrics.mean_error,
        rmse=metrics.rmse,
        nrmse=metrics.nrmse,
        mae=metrics.mae,
        pearson_r=metrics.pearson_r,
        within_fraction=within_fraction,
        above_fraction=above_fraction,
        below_fraction=below_fraction,
        positive_fraction=positive_fraction,
        negative_fraction=negative_fraction,
        longest_above_run=longest_above_run,
        longest_below_run=longest_below_run,
        ci_low=None,
        ci_high=None,
        iqr=spread_iqr,
        mad=spread_mad,
        hodges_lehmann_shift=hl_shift,
        real_q25=real_q25,
        real_q50=real_q50,
        real_q75=real_q75,
        simu_q25=simu_q25,
        simu_q50=simu_q50,
        simu_q75=simu_q75,
        iqr_overlap_ratio=iqr_overlap,
        detail=(
            f"{channel_name}: rolling_median_relative_error={rolling_median_relative_error:.4%}, "
            f"raw_median_relative_error={raw_median_relative_error:.4%}, "
            f"relative_iqr={format_metric_value(spread_iqr, '%')}, "
            f"relative_mad={format_metric_value(spread_mad, '%')}, "
            f"hl_shift={format_metric_value(hl_shift)}, "
            f"mean_relative_error={mean_relative_error:.4%}, "
            f"within={within_fraction:.1%}, above={above_fraction:.1%}, "
            f"below={below_fraction:.1%}, positive={positive_fraction:.1%}, "
            f"negative={negative_fraction:.1%}, "
            f"longest_above_run={longest_above_run}, longest_below_run={longest_below_run}, "
            f"iqr_overlap={format_metric_value(None if iqr_overlap is None else 100.0 * iqr_overlap, '%')}, "
            f"pearson_r={format_metric_value(metrics.pearson_r)}"
        ),
    )
