import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.comparison_metrics import (
    FitQualityEvidence,
    STABLE_RUN_MIN_FRACTION,
    STABLE_SIGN_MIN_FRACTION,
    ThresholdComparisonEvidence,
    TimeSeriesAgreementEvidence,
    compute_fit_quality_evidence,
    compute_signal_comparison_metrics,
    compute_time_series_agreement_evidence,
    evaluate_absolute_threshold_comparison,
    evaluate_relative_threshold_comparison,
    format_metric_value,
)
from src.data.synchronization import synchronize_dataframes
from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file


@dataclass(frozen=True)
class ThresholdConfig:
    rh_mm: float = 1.5
    damper_mm: float = 1.5
    push_kg: float = 2.5
    pitot_relative: float = 0.03
    tyre_psi: float = 1.5
    rh_prediction_mm: float = 1.5
    push_prediction_kg: float = 2.5
    pitot_prediction_relative: float = 0.03


@dataclass(frozen=True)
class ChannelSet:
    name: str
    ride_height: str
    push_load: str
    damper: str
    tyre_pressure: str


@dataclass(frozen=True)
class ComparisonResult:
    state: str
    track_median: float | None
    reference_median: float | None
    difference: float | None
    threshold: float
    evidence: ThresholdComparisonEvidence | None
    detail: str


@dataclass(frozen=True)
class FitCheckResult:
    status: str
    slope: float | None
    intercept: float | None
    predicted_median: float | None
    observed_median: float | None
    tolerance: float
    quality: FitQualityEvidence | None
    detail: str


@dataclass(frozen=True)
class DecisionResult:
    channel: str
    action: str
    summary: str
    detail: str


@dataclass(frozen=True)
class PlateauEvidence:
    start_index: int | None
    end_index: int | None
    sample_count: int
    speed_gradient_mean: float | None
    accx_mean_abs: float | None
    accy_mean_abs: float | None
    ride_height_variance: float | None
    push_load_variance: float | None
    detail: str


@dataclass(frozen=True)
class HelperAgreementSummary:
    support_count: int
    contradiction_count: int
    neutral_count: int
    total_count: int
    detail: str


@dataclass(frozen=True)
class ConditionComparabilityResult:
    channel_name: str
    state: str
    threshold: float
    sample_count: int
    median_real: float | None
    median_simu: float | None
    median_delta: float | None
    mae: float | None
    pearson_r: float | None
    within_fraction: float | None
    overlap_ratio: float | None
    detail: str


@dataclass(frozen=True)
class SegmentComparabilityResult:
    comparable: bool
    detail: str
    channel_results: tuple[ConditionComparabilityResult, ...]


@dataclass(frozen=True)
class PitGroupSelectionInfo:
    source: str
    segment_label: str
    pit_speed_band: str
    group_id: int
    sample_count: int
    index_start: int | None
    index_end: int | None
    speed_min: float | None
    speed_median: float | None
    speed_max: float | None
    quality_column: str | None
    quality_nonzero_fraction: float | None
    quality_variability: float | None


CHANNEL_SETS = {
    "fl": ChannelSet(
        name="front-left",
        ride_height="rh_f",
        push_load="pushavg_c",
        damper="damper_fl_art",
        tyre_pressure="tpms_p_fr",
    ),
    "fr": ChannelSet(
        name="front-right",
        ride_height="rh_f",
        push_load="pushavd_c",
        damper="damper_fr_art",
        tyre_pressure="tpms_p_fr",
    ),
    "rl": ChannelSet(
        name="rear-left",
        ride_height="rh_r",
        push_load="pusharg_c",
        damper="damper_rl_art",
        tyre_pressure="tpms_p_rl",
    ),
    "rr": ChannelSet(
        name="rear-right",
        ride_height="rh_r",
        push_load="pushard_c",
        damper="damper_rr_art",
        tyre_pressure="tpms_p_rl",
    ),
}


EXTRA_MAP_COLUMNS: list[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment-label",
        choices=["pit", "straight", "corner"],
        default="pit",
        help="Segment type to analyze with the shared decision-tree engine.",
    )
    parser.add_argument(
        "--channel-set",
        choices=sorted(CHANNEL_SETS),
        default="fl",
        help="Corner / channel set to validate.",
    )
    parser.add_argument(
        "--pit-speed-band",
        choices=["slow", "fast", "both", "low", "high"],
        default="slow",
        help="For pit segments, compare either the slow pit band (30-50 km/h), the fast pit band (50-75 km/h), or both sequentially. Legacy aliases: low=slow, high=fast.",
    )
    parser.add_argument(
        "--min-pit-band-samples",
        type=int,
        default=50,
        help="Minimum number of samples required in the selected pit speed band before running comparisons.",
    )
    parser.add_argument(
        "--rh-threshold-mm",
        type=float,
        default=ThresholdConfig.rh_mm,
        help="Ride height equality threshold in mm.",
    )
    parser.add_argument(
        "--damper-threshold-mm",
        type=float,
        default=ThresholdConfig.damper_mm,
        help="Damper equality threshold in mm.",
    )
    parser.add_argument(
        "--push-threshold-kg",
        type=float,
        default=ThresholdConfig.push_kg,
        help="Push-load equality threshold in kg.",
    )
    parser.add_argument(
        "--pitot-relative-threshold",
        type=float,
        default=ThresholdConfig.pitot_relative,
        help="Relative pitot vs pAir equality threshold, e.g. 0.03 for 3%%.",
    )
    parser.add_argument(
        "--tyre-threshold-psi",
        type=float,
        default=ThresholdConfig.tyre_psi,
        help="Tyre-pressure threshold in psi.",
    )
    parser.add_argument(
        "--rh-prediction-threshold-mm",
        type=float,
        default=ThresholdConfig.rh_prediction_mm,
        help="Allowed RH prediction error in mm for the XY evidence check.",
    )
    parser.add_argument(
        "--push-prediction-threshold-kg",
        type=float,
        default=ThresholdConfig.push_prediction_kg,
        help="Allowed push-load prediction error in kg for the XY evidence check.",
    )
    parser.add_argument(
        "--pitot-prediction-relative-threshold",
        type=float,
        default=ThresholdConfig.pitot_prediction_relative,
        help="Allowed relative pitot prediction error for the XY evidence check.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show and save the time-domain and XY evidence plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/plots/segment_decision_tree"),
        help="Directory where evidence plots will be saved.",
    )
    parser.add_argument(
        "--sync-grid-size",
        type=int,
        default=200,
        help="Number of synchronized comparison points across the pit segment.",
    )
    parser.add_argument(
        "--condition-speed-threshold-kmh",
        type=float,
        default=10.0,
        help="Allowed real-vs-sim speed mismatch in km/h for the pitlane comparability pre-check.",
    )
    parser.add_argument(
        "--condition-accx-threshold",
        type=float,
        default=0.08,
        help="Allowed real-vs-sim longitudinal-acceleration mismatch for the pitlane comparability pre-check.",
    )
    parser.add_argument(
        "--condition-accy-threshold",
        type=float,
        default=0.08,
        help="Allowed real-vs-sim lateral-acceleration mismatch for the pitlane comparability pre-check.",
    )
    return parser.parse_args()


def load_segmented_data(source: str) -> pd.DataFrame:
    if source == "simu":
        return pd.read_csv(segmented_simudata_run_file(), low_memory=False)
    return pd.read_csv(segmented_rh_run_file(), low_memory=False)


def canonical_pit_speed_band(pit_speed_band: str) -> str:
    if pit_speed_band == "low":
        return "slow"
    if pit_speed_band == "high":
        return "fast"
    return pit_speed_band


def build_pit_speed_band_mask(speed: pd.Series, pit_speed_band: str) -> pd.Series:
    band = canonical_pit_speed_band(pit_speed_band)
    if band == "slow":
        return speed.ge(30.0) & speed.lt(50.0)
    if band == "fast":
        return speed.ge(50.0) & speed.lt(75.0)
    raise ValueError(f"Unsupported pit speed band: {pit_speed_band}")


def get_first_segment(df: pd.DataFrame, source: str, segment_label: str) -> pd.DataFrame:
    if "segment_final" not in df.columns:
        raise ValueError(f"Missing 'segment_final' in {source} data.")

    segment_df = df[df["segment_final"].astype("string") == segment_label].copy()
    if segment_df.empty:
        raise ValueError(f"No '{segment_label}' rows found in segmented {source} data.")

    segment_df["group"] = (segment_df.index.to_series().diff() != 1).cumsum()
    first_group = segment_df["group"].iloc[0]
    first_segment = segment_df[segment_df["group"] == first_group].copy()
    return first_segment.drop(columns=["group"])


def build_pit_band_groups(
    df: pd.DataFrame,
    source: str,
    segment_label: str,
    pit_speed_band: str,
    min_pit_band_samples: int,
    quality_column: str | None = None,
) -> tuple[pd.DataFrame, list[PitGroupSelectionInfo]]:
    if segment_label != "pit":
        segment = get_first_segment(df, source, segment_label)
        return segment, []

    if "segment_final" not in df.columns:
        raise ValueError(f"Missing 'segment_final' in {source} data.")
    if "carspeed_art" not in df.columns:
        raise ValueError(f"Missing 'carspeed_art' in {source} data.")

    segment_df = df[df["segment_final"].astype("string") == segment_label].copy()
    if segment_df.empty:
        raise ValueError(f"No '{segment_label}' rows found in segmented {source} data.")

    speed = pd.to_numeric(segment_df["carspeed_art"], errors="coerce")
    band_mask = build_pit_speed_band_mask(speed, pit_speed_band)

    band_df = segment_df.loc[band_mask].copy()
    if band_df.empty:
        raise ValueError(
            f"No '{segment_label}' rows found in the requested {pit_speed_band} pit-speed band for {source} data."
        )

    band_df["group"] = (band_df.index.to_series().diff() != 1).cumsum()
    group_sizes = band_df.groupby("group").size().sort_values(ascending=False)
    qualifying_groups = group_sizes[group_sizes >= min_pit_band_samples]

    candidate_groups = qualifying_groups if not qualifying_groups.empty else group_sizes
    candidate_ids = candidate_groups.index.tolist()
    candidate_df = band_df[band_df["group"].isin(candidate_ids)].copy()

    group_infos: list[PitGroupSelectionInfo] = []
    for group_id in candidate_ids:
        group_df = candidate_df[candidate_df["group"] == group_id].copy()
        size = len(group_df)
        speed_series = pd.to_numeric(group_df["carspeed_art"], errors="coerce").dropna()
        quality_nonzero_fraction = 0.0
        quality_variability = 0.0
        if quality_column is not None and quality_column in group_df.columns:
            quality_series = pd.to_numeric(group_df[quality_column], errors="coerce").dropna()
            if not quality_series.empty:
                quality_nonzero_fraction = float((quality_series.abs() > 1e-9).mean())
                quality_variability = (
                    float(quality_series.std(ddof=0)) if len(quality_series) > 1 else 0.0
                )
        group_infos.append(
            PitGroupSelectionInfo(
                source=source,
                segment_label=segment_label,
                pit_speed_band=pit_speed_band,
                group_id=int(group_id),
                sample_count=int(size),
                index_start=int(group_df.index.min()) if not group_df.empty else None,
                index_end=int(group_df.index.max()) if not group_df.empty else None,
                speed_min=float(speed_series.min()) if not speed_series.empty else None,
                speed_median=float(speed_series.median()) if not speed_series.empty else None,
                speed_max=float(speed_series.max()) if not speed_series.empty else None,
                quality_column=quality_column,
                quality_nonzero_fraction=quality_nonzero_fraction,
                quality_variability=quality_variability,
            )
        )

    fallback_group_id = int(
        max(
            group_infos,
            key=lambda info: (
                float(info.quality_nonzero_fraction or 0.0),
                float(info.quality_variability or 0.0),
                int(info.sample_count),
            ),
        ).group_id
    )
    fallback = candidate_df[candidate_df["group"] == fallback_group_id].copy().drop(columns=["group"])
    fallback.attrs["segment_selection"] = next(
        info for info in group_infos if info.group_id == fallback_group_id
    )
    return fallback, group_infos


def select_matched_pit_segments_for_analysis(
    real_df: pd.DataFrame,
    simu_df: pd.DataFrame,
    pit_speed_band: str,
    min_pit_band_samples: int,
    quality_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_fallback, real_group_infos = build_pit_band_groups(
        real_df,
        "real",
        "pit",
        pit_speed_band,
        min_pit_band_samples,
        quality_column=quality_column,
    )
    simu_fallback, simu_group_infos = build_pit_band_groups(
        simu_df,
        "simu",
        "pit",
        pit_speed_band,
        min_pit_band_samples,
        quality_column=quality_column,
    )

    if not real_group_infos or not simu_group_infos:
        return real_fallback, simu_fallback

    def prefer_nonzero_groups(
        group_infos: list[PitGroupSelectionInfo],
    ) -> list[PitGroupSelectionInfo]:
        valid_groups = [
            info
            for info in group_infos
            if (info.quality_nonzero_fraction or 0.0) > 0.0
        ]
        return valid_groups if valid_groups else group_infos

    real_group_infos = prefer_nonzero_groups(real_group_infos)
    simu_group_infos = prefer_nonzero_groups(simu_group_infos)

    best_real_info: PitGroupSelectionInfo | None = None
    best_simu_info: PitGroupSelectionInfo | None = None
    best_pair_key: tuple[float, float, float, int] | None = None

    for real_info in real_group_infos:
        for simu_info in simu_group_infos:
            real_speed = real_info.speed_median
            simu_speed = simu_info.speed_median
            if real_speed is None or simu_speed is None:
                continue
            speed_gap = abs(float(real_speed) - float(simu_speed))
            pair_key = (
                -speed_gap,
                min(
                    float(real_info.quality_nonzero_fraction or 0.0),
                    float(simu_info.quality_nonzero_fraction or 0.0),
                ),
                float(real_info.quality_variability or 0.0)
                + float(simu_info.quality_variability or 0.0),
                int(real_info.sample_count) + int(simu_info.sample_count),
            )
            if best_pair_key is None or pair_key > best_pair_key:
                best_pair_key = pair_key
                best_real_info = real_info
                best_simu_info = simu_info

    if best_real_info is None or best_simu_info is None:
        return real_fallback, simu_fallback

    def select_specific_group(
        df: pd.DataFrame,
        group_id: int,
        info: PitGroupSelectionInfo,
    ) -> pd.DataFrame:
        segment_df = df[df["segment_final"].astype("string") == "pit"].copy()
        speed = pd.to_numeric(segment_df["carspeed_art"], errors="coerce")
        band_mask = build_pit_speed_band_mask(speed, pit_speed_band)
        band_df = segment_df.loc[band_mask].copy()
        band_df["group"] = (band_df.index.to_series().diff() != 1).cumsum()
        selected = band_df[band_df["group"] == group_id].copy().drop(columns=["group"])
        selected.attrs["segment_selection"] = info
        return selected

    return (
        select_specific_group(real_df, best_real_info.group_id, best_real_info),
        select_specific_group(simu_df, best_simu_info.group_id, best_simu_info),
    )


def print_selected_segment_info(df: pd.DataFrame) -> None:
    info = df.attrs.get("segment_selection")
    if not isinstance(info, PitGroupSelectionInfo):
        print("selected segment info unavailable")
        return

    print(
        f"{info.source}: group={info.group_id}, "
        f"band={info.pit_speed_band}, "
        f"samples={info.sample_count}, "
        f"indices={info.index_start}->{info.index_end}, "
        f"speed[min/median/max]={format_metric_value(info.speed_min)}/"
        f"{format_metric_value(info.speed_median)}/"
        f"{format_metric_value(info.speed_max)}"
    )
    if info.quality_column:
        print(
            f"  quality column={info.quality_column}, "
            f"nonzero_fraction={format_metric_value(None if info.quality_nonzero_fraction is None else 100.0 * info.quality_nonzero_fraction, '%')}, "
            f"variability={format_metric_value(info.quality_variability)}"
        )


def get_median(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")
    value = pd.to_numeric(df[column], errors="coerce").median(skipna=True)
    if pd.isna(value):
        raise ValueError(f"Column {column} has no valid numeric values.")
    return float(value)


def get_optional_median(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    value = pd.to_numeric(df[column], errors="coerce").median(skipna=True)
    if pd.isna(value):
        return None
    return float(value)


def get_synced_median(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None

    value = pd.to_numeric(df[column], errors="coerce").median(skipna=True)
    if pd.isna(value):
        return None
    return float(value)


def get_synced_rolling_median(
    df: pd.DataFrame,
    column: str,
    window: int = 9,
) -> float | None:
    if column not in df.columns:
        return None

    value = (
        pd.to_numeric(df[column], errors="coerce")
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .median(skipna=True)
    )
    if pd.isna(value):
        return None
    return float(value)


def build_raw_progress_aligned_df(
    real_df: pd.DataFrame,
    simu_df: pd.DataFrame,
    columns: list[str],
    grid_size: int,
) -> pd.DataFrame:
    if len(real_df) == 0 or len(simu_df) == 0:
        return pd.DataFrame({"progress": np.array([], dtype="float64")})

    size = max(2, min(grid_size, len(real_df), len(simu_df)))
    progress = np.linspace(0.0, 1.0, size)
    real_idx = np.rint(progress * (len(real_df) - 1)).astype(int)
    simu_idx = np.rint(progress * (len(simu_df) - 1)).astype(int)

    aligned = pd.DataFrame({"progress": progress})
    real_rows = real_df.reset_index(drop=True)
    simu_rows = simu_df.reset_index(drop=True)

    for column in columns:
        if column in real_rows.columns:
            real_values = pd.to_numeric(real_rows[column], errors="coerce").to_numpy(dtype="float64")
            aligned[f"real__{column}"] = real_values[real_idx]
        if column in simu_rows.columns:
            simu_values = pd.to_numeric(simu_rows[column], errors="coerce").to_numpy(dtype="float64")
            aligned[f"simu__{column}"] = simu_values[simu_idx]

    return aligned 


def compare_track_vs_sim_higher_is_signal(
    real_series: pd.Series,
    simu_series: pd.Series,
    threshold: float,
    channel_name: str,
) -> ComparisonResult:
    evidence = evaluate_absolute_threshold_comparison(
        real_series=real_series,
        simu_series=simu_series,
        threshold=threshold,
        channel_name=channel_name,
        higher_state="higher_than_sim",
        lower_state="lower_than_sim",
    )
    track_value = get_synced_rolling_median(pd.DataFrame({"value": real_series}), "value")
    sim_value = get_synced_rolling_median(pd.DataFrame({"value": simu_series}), "value")
    if evidence.state == "unavailable":
        return ComparisonResult(
            state="unavailable",
            track_median=track_value,
            reference_median=sim_value,
            difference=None,
            threshold=threshold,
            evidence=evidence,
            detail=evidence.detail,
        )
    difference = None if track_value is None or sim_value is None else track_value - sim_value
    return ComparisonResult(
        state=evidence.state,
        track_median=track_value,
        reference_median=sim_value,
        difference=difference,
        threshold=threshold,
        evidence=evidence,
        detail=evidence.detail,
    )


def compare_track_vs_sim_lower_is_signal(
    real_series: pd.Series,
    simu_series: pd.Series,
    threshold: float,
    channel_name: str,
) -> ComparisonResult:
    evidence = evaluate_absolute_threshold_comparison(
        real_series=real_series,
        simu_series=simu_series,
        threshold=threshold,
        channel_name=channel_name,
        higher_state="higher_than_sim",
        lower_state="lower_than_sim",
    )
    track_value = get_synced_rolling_median(pd.DataFrame({"value": real_series}), "value")
    sim_value = get_synced_rolling_median(pd.DataFrame({"value": simu_series}), "value")
    if evidence.state == "unavailable":
        return ComparisonResult(
            state="unavailable",
            track_median=track_value,
            reference_median=sim_value,
            difference=None,
            threshold=threshold,
            evidence=evidence,
            detail=evidence.detail,
        )
    difference = None if track_value is None or sim_value is None else track_value - sim_value
    return ComparisonResult(
        state=evidence.state,
        track_median=track_value,
        reference_median=sim_value,
        difference=difference,
        threshold=threshold,
        evidence=evidence,
        detail=evidence.detail,
    )


def unavailable_comparison_result(
    channel_name: str,
    threshold: float,
    reason: str,
) -> ComparisonResult:
    return ComparisonResult(
        state="unavailable",
        track_median=None,
        reference_median=None,
        difference=None,
        threshold=threshold,
        evidence=None,
        detail=f"{channel_name}: unavailable ({reason})",
    )


def compare_pitot_vs_pair(
    pitot_series: pd.Series,
    pair_series: pd.Series,
    relative_threshold: float,
) -> ComparisonResult:
    evidence = evaluate_relative_threshold_comparison(
        real_series=pitot_series,
        reference_series=pair_series,
        relative_threshold=relative_threshold,
        channel_name="pitot vs pAir",
        lower_state="lower_than_pair",
        higher_state="higher_than_pair",
    )
    pitot_value = get_synced_median(pd.DataFrame({"value": pitot_series}), "value")
    pair_value = get_synced_median(pd.DataFrame({"value": pair_series}), "value")
    if evidence.state == "unavailable":
        return ComparisonResult(
            state="unavailable",
            track_median=pitot_value,
            reference_median=pair_value,
            difference=None,
            threshold=relative_threshold,
            evidence=evidence,
            detail=evidence.detail,
        )
    difference = (
        pitot_value - pair_value
        if pitot_value is not None and pair_value is not None
        else None
    )
    return ComparisonResult(
        state=evidence.state,
        track_median=pitot_value,
        reference_median=pair_value,
        difference=difference,
        threshold=relative_threshold,
        evidence=evidence,
        detail=evidence.detail,
    )


def fit_linear_relationship(
    x: pd.Series,
    y: pd.Series,
    input_median: float | None,
    observed_median: float | None,
    tolerance: float,
    label: str,
) -> FitCheckResult:
    if input_median is None or observed_median is None:
        return FitCheckResult(
            status="insufficient_data",
            slope=None,
            intercept=None,
            predicted_median=None,
            observed_median=observed_median,
            tolerance=tolerance,
            quality=None,
            detail=f"{label}: missing median input for linear fit",
        )

    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    clean = clean[np.isfinite(clean["x"]) & np.isfinite(clean["y"])]

    if len(clean) < 3 or clean["x"].nunique() < 2:
        return FitCheckResult(
            status="insufficient_data",
            slope=None,
            intercept=None,
            predicted_median=None,
            observed_median=observed_median,
            tolerance=tolerance,
            quality=None,
            detail=f"{label}: not enough distinct points for linear fit",
        )

    fit_coefficients = np.polyfit(clean["x"], clean["y"], 1)
    slope = float(fit_coefficients[0])
    intercept = float(fit_coefficients[1])
    predicted = float(slope * input_median + intercept)
    difference = predicted - observed_median
    status = "match" if abs(difference) <= tolerance else "mismatch"

    quality = compute_fit_quality_evidence(
        x=clean["x"],
        y=clean["y"],
        slope=slope,
        intercept=intercept,
        tolerance=tolerance,
    )

    return FitCheckResult(
        status=status,
        slope=slope,
        intercept=intercept,
        predicted_median=predicted,
        observed_median=observed_median,
        tolerance=tolerance,
        quality=quality,
        detail=(
            f"{label}: predicted median={predicted:.4f}, "
            f"observed median={observed_median:.4f}, delta={difference:.4f}"
        ),
    )


def decide_ride_height_offset(
    rh_cmp: ComparisonResult,
    damper_cmp: ComparisonResult,
    push_cmp: ComparisonResult,
    pitot_cmp: ComparisonResult,
    tyre_cmp: ComparisonResult,
    rh_from_damper_fit: FitCheckResult,
    rh_from_push_fit: FitCheckResult,
) -> DecisionResult:
    comparison_states = {
        rh_cmp.state,
        damper_cmp.state,
        push_cmp.state,
        pitot_cmp.state,
        tyre_cmp.state,
    }
    if "ambiguous" in comparison_states:
        return DecisionResult(
            channel="ride_height",
            action="manual_review",
            summary="Ambiguous RH evidence",
            detail="At least one synchronized comparison is mixed around the threshold band. Manual review recommended before applying RH logic.",
        )
    if rh_cmp.state == "within_threshold":
        return DecisionResult(
            channel="ride_height",
            action="no_offset",
            summary="RH OK",
            detail="Ride height is within +/- threshold of sim. No offset needed.",
        )

    if rh_cmp.state != "higher_than_sim":
        return DecisionResult(
            channel="ride_height",
            action="logic_not_defined",
            summary="RH low",
            detail="Written logic only defines the RH-higher-than-sim branch.",
        )

    # Diamond 1:
    # RH high and damper lower compression and push load lower and pitot lower than pAir?
    if (
        damper_cmp.state == "lower_than_sim"
        and push_cmp.state == "lower_than_sim"
        and pitot_cmp.state == "lower_than_pair"
    ):
        return DecisionResult(
            channel="ride_height",
            action="no_offset",
            summary="No RH offset",
            detail="Data is genuine: RH high with lower damper compression, lower push load, and lower pitot.",
        )

    # Diamond 2:
    # RH high and damper lower compression and push load lower but pitot NOT lower than pAir?
    if (
        damper_cmp.state == "lower_than_sim"
        and push_cmp.state == "lower_than_sim"
        and pitot_cmp.state != "lower_than_pair"
    ):
        return DecisionResult(
            channel="ride_height",
            action="no_offset",
            summary="No RH offset",
            detail="Data seems genuine. Unknown reason may keep RH higher although pitot does not support lower aero load.",
        )

    # Diamond 3:
    # RH high and damper lower compression but push load NOT lower and pitot lower than pAir?
    if (
        damper_cmp.state == "lower_than_sim"
        and push_cmp.state != "lower_than_sim"
        and pitot_cmp.state == "lower_than_pair"
    ):
        return DecisionResult(
            channel="ride_height",
            action="no_offset",
            summary="No RH offset",
            detail=(
                "Damper and pitot support a genuine higher RH, but push load does not. "
                "Do not offset RH based on this combination alone; validate push separately."
            ),
        )

    # Diamond 4:
    # RH high and damper NOT lower compression and push load lower and pitot lower than pAir?
    if (
        damper_cmp.state != "lower_than_sim"
        and push_cmp.state == "lower_than_sim"
        and pitot_cmp.state == "lower_than_pair"
    ):
        if rh_from_push_fit.status == "match":
            return DecisionResult(
                channel="ride_height",
                action="no_offset",
                summary="No RH offset",
                detail=(
                    "Push load and pitot support genuine unloading, and the RH=f(Push load) "
                    "relationship is consistent. Keep RH unchanged and treat damper as the inconsistent helper."
                ),
            )
        if rh_from_push_fit.status == "mismatch":
            return DecisionResult(
                channel="ride_height",
                action="apply_offset",
                summary="Apply RH offset",
                detail=(
                    "Push load and pitot suggest higher RH, but the RH=f(Push load) relationship is inconsistent. "
                    "RH offset is more likely."
                ),
            )
        return DecisionResult(
            channel="ride_height",
            action="manual_review",
            summary="RH / push consistency unavailable",
            detail="RH-high branch reached, but RH=f(Push load) consistency is unavailable. Manual review recommended.",
        )

    # Diamond 5:
    # RH high and damper lower compression but push load NOT lower and pitot NOT lower than pAir?
    if (
        damper_cmp.state == "lower_than_sim"
        and push_cmp.state != "lower_than_sim"
        and pitot_cmp.state != "lower_than_pair"
    ):
        return DecisionResult(
            channel="ride_height",
            action="no_offset",
            summary="No RH offset",
            detail="Data seems genuine. Suspension may be stiffer than expected and keep RH higher.",
        )

    # Diamond 6:
    # RH high but damper NOT lower compression and push load lower and pitot NOT lower than pAir?
    if (
        damper_cmp.state != "lower_than_sim"
        and push_cmp.state == "lower_than_sim"
        and pitot_cmp.state != "lower_than_pair"
    ):
        if rh_from_push_fit.status == "match":
            return DecisionResult(
                channel="ride_height",
                action="no_offset",
                summary="No RH offset",
                detail=(
                    "Push load supports higher RH and the RH=f(Push load) relationship is consistent. "
                    "Keep RH unchanged and investigate damper/pitot separately."
                ),
            )
        if rh_from_push_fit.status == "mismatch":
            return DecisionResult(
                channel="ride_height",
                action="apply_offset",
                summary="Apply RH offset",
                detail=(
                    "Push load suggests higher RH, but neither damper nor pitot supports it and RH=f(Push load) is inconsistent. "
                    "RH offset becomes more likely."
                ),
            )
        return DecisionResult(
            channel="ride_height",
            action="manual_review",
            summary="RH / push consistency unavailable",
            detail="RH-high branch reached, but RH=f(Push load) consistency is unavailable. Manual review recommended.",
        )

    # Diamond 7:
    # RH high but damper NOT lower compression and push load NOT lower and pitot lower than pAir?
    if (
        damper_cmp.state != "lower_than_sim"
        and push_cmp.state != "lower_than_sim"
        and pitot_cmp.state == "lower_than_pair"
    ):
        if tyre_cmp.state == "higher_than_sim":
            return DecisionResult(
                channel="ride_height",
                action="no_offset",
                summary="No RH offset",
                detail=(
                    "RH is high and only pitot supports it, but higher tyre pressure can explain the increased ride height. "
                    "Do not apply RH offset."
                ),
            )
        if (
            rh_from_damper_fit.status == "mismatch"
            and rh_from_push_fit.status == "mismatch"
        ):
            return DecisionResult(
                channel="ride_height",
                action="apply_offset",
                summary="Apply RH offset",
                detail=(
                    "Only pitot supports the higher RH. Tyre pressure does not explain it, and both RH=f(Damper) and "
                    "RH=f(Push load) fail to support the measured RH. RH offset is justified."
                ),
            )
        return DecisionResult(
            channel="ride_height",
            action="manual_review",
            summary="RH offset candidate",
            detail=(
                "Only pitot supports the higher RH. Check tyre pressure first; if it does not explain the RH increase, "
                "validate RH=f(Damper) and RH=f(Push load) before applying RH offset."
            ),
        )

    # Diamond 8:
    # RH high but damper NOT lower compression and push load NOT lower and pitot NOT lower than pAir?
    if (
        damper_cmp.state != "lower_than_sim"
        and push_cmp.state != "lower_than_sim"
        and pitot_cmp.state != "lower_than_pair"
    ):
        if tyre_cmp.state == "higher_than_sim":
            return DecisionResult(
                channel="ride_height",
                action="no_offset",
                summary="No RH offset",
                detail="Higher tyre pressure can explain the higher RH in pit lane.",
            )
        if tyre_cmp.state == "unavailable":
            return DecisionResult(
                channel="ride_height",
                action="manual_review",
                summary="Tyre-pressure check unavailable",
                detail="RH offset branch reached, but tyre-pressure evidence is unavailable. Manual review recommended before applying RH offset.",
            )
        return DecisionResult(
            channel="ride_height",
            action="apply_offset",
            summary="Apply RH offset",
            detail="RH is high but helper channels do not support a genuine physical explanation.",
        )

    return DecisionResult(
        channel="ride_height",
        action="manual_review",
        summary="Ambiguous RH case",
        detail="This RH-high combination does not land on any explicitly defined RH decision path in the written tree. Manual review recommended.",
    )


def decide_push_load_offset(
    push_cmp: ComparisonResult,
    damper_cmp: ComparisonResult,
    rh_cmp: ComparisonResult,
    pitot_cmp: ComparisonResult,
    rh_from_push_fit: FitCheckResult,
    push_from_damper_fit: FitCheckResult,
) -> DecisionResult:
    comparison_states = {
        push_cmp.state,
        damper_cmp.state,
        rh_cmp.state,
        pitot_cmp.state,
    }
    if "ambiguous" in comparison_states:
        return DecisionResult(
            channel="push_load",
            action="manual_review",
            summary="Ambiguous push-load evidence",
            detail="At least one synchronized comparison is mixed around the threshold band. Manual review recommended before applying push-load logic.",
        )
    if push_cmp.state == "within_threshold":
        return DecisionResult(
            channel="push_load",
            action="no_offset",
            summary="Push load OK",
            detail="Push load is within +/- threshold of sim. No offset needed.",
        )

    if push_cmp.state == "higher_than_sim":
        damper_lower = damper_cmp.state == "lower_than_sim"
        damper_higher = damper_cmp.state == "higher_than_sim"
        damper_ok = damper_cmp.state == "within_threshold"
        rh_higher = rh_cmp.state == "higher_than_sim"
        rh_lower = rh_cmp.state == "lower_than_sim"
        rh_ok = rh_cmp.state == "within_threshold"
        pitot_lower = pitot_cmp.state == "lower_than_pair"
        pitot_higher = pitot_cmp.state == "higher_than_pair"
        pitot_ok = pitot_cmp.state == "within_threshold"

        # 1. Push higher + Damper lower + RH higher + Pitot lower
        if damper_lower and rh_higher and pitot_lower:
            if push_from_damper_fit.status == "mismatch" and rh_from_push_fit.status == "mismatch":
                return DecisionResult(
                    channel="push_load",
                    action="apply_offset",
                    summary="Apply push-load offset",
                    detail=(
                        "All helper channels contradict the higher push load, and both Push load = f(Damper) "
                        "and RH = f(Push load) fail to support it. Push offset is justified."
                    ),
                )
            return DecisionResult(
                channel="push_load",
                action="manual_review",
                summary="Strong push offset candidate",
                detail=(
                    "Damper, RH, and pitot all contradict the higher push load. "
                    "Validate Push load = f(Damper) and RH = f(Push load) before applying push offset."
                ),
            )

        # 2. Push higher + Damper lower + RH higher + Pitot not lower
        if damper_lower and rh_higher and not pitot_lower:
            if pitot_ok:
                return DecisionResult(
                    channel="push_load",
                    action="apply_offset",
                    summary="Apply push-load offset",
                    detail="Damper and RH contradict the higher push load while pitot is neutral. No helper channel supports the push increase.",
                )
            if pitot_higher:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail=(
                        "Pitot supports higher loading, but damper and RH contradict the higher push load. "
                        "Validate Push load = f(Damper) and RH = f(Push load)."
                    ),
                )

        # 3. Push higher + Damper lower + RH not higher + Pitot lower
        if damper_lower and not rh_higher and pitot_lower:
            if rh_ok:
                return DecisionResult(
                    channel="push_load",
                    action="apply_offset",
                    summary="Apply push-load offset",
                    detail="Damper and pitot contradict the higher push load while RH is neutral. No helper channel supports the push increase.",
                )
            if rh_lower:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail=(
                        "RH supports higher loading, but damper and pitot contradict the higher push load. "
                        "Validate Push load = f(Damper) and RH = f(Push load)."
                    ),
                )

        # 4. Push higher + Damper lower + RH not higher + Pitot not lower
        if damper_lower and not rh_higher and not pitot_lower:
            if rh_ok and pitot_ok:
                return DecisionResult(
                    channel="push_load",
                    action="apply_offset",
                    summary="Apply push-load offset",
                    detail="Damper contradicts the higher push load while RH and pitot are neutral. Push is not independently supported.",
                )
            if rh_lower and pitot_ok:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail="RH supports higher loading, damper contradicts it, and pitot is neutral. Validate push before applying an offset.",
                )
            if rh_ok and pitot_higher:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail="Pitot supports higher aerodynamic loading, damper contradicts it, and RH is neutral. Validate push before applying an offset.",
                )
            if rh_lower and pitot_higher:
                return DecisionResult(
                    channel="push_load",
                    action="no_offset",
                    summary="No push-load offset",
                    detail="RH and pitot both support the higher push load. Investigate damper separately.",
                )

        # 5. Push higher + Damper not lower + RH higher + Pitot lower
        if not damper_lower and rh_higher and pitot_lower:
            if damper_ok:
                return DecisionResult(
                    channel="push_load",
                    action="apply_offset",
                    summary="Apply push-load offset",
                    detail="RH and pitot contradict the higher push load while damper is neutral. Nothing independently supports the push increase.",
                )
            if damper_higher:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail="Damper supports higher loading, but RH and pitot contradict it. Validate push before applying an offset.",
                )

        # 6. Push higher + Damper not lower + RH higher + Pitot not lower
        if not damper_lower and rh_higher and not pitot_lower:
            if damper_ok and pitot_ok:
                return DecisionResult(
                    channel="push_load",
                    action="apply_offset",
                    summary="Apply push-load offset",
                    detail="RH contradicts the higher push load while damper and pitot are neutral. Push has no independent support.",
                )
            if damper_higher and pitot_ok:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail="Damper supports higher loading, RH contradicts it, and pitot is neutral. Validate push before applying an offset.",
                )
            if damper_ok and pitot_higher:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail="Pitot supports higher loading, RH contradicts it, and damper is neutral. Validate push before applying an offset.",
                )
            if damper_higher and pitot_higher:
                return DecisionResult(
                    channel="push_load",
                    action="no_offset",
                    summary="No push-load offset",
                    detail="Damper and pitot support the higher push load. Investigate RH separately.",
                )

        # 7. Push higher + Damper not lower + RH not higher + Pitot lower
        if not damper_lower and not rh_higher and pitot_lower:
            if damper_ok and rh_ok:
                return DecisionResult(
                    channel="push_load",
                    action="apply_offset",
                    summary="Apply push-load offset",
                    detail="Pitot contradicts the higher push load while damper and RH are neutral. There is no positive support for the push increase.",
                )
            if damper_higher and rh_ok:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail="Damper supports higher loading, pitot contradicts it, and RH is neutral. Validate push before applying an offset.",
                )
            if damper_ok and rh_lower:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail="RH supports higher loading, pitot contradicts it, and damper is neutral. Validate push before applying an offset.",
                )
            if damper_higher and rh_lower:
                return DecisionResult(
                    channel="push_load",
                    action="no_offset",
                    summary="No push-load offset",
                    detail="Damper and RH both support the higher push load. Investigate pitot separately.",
                )

        # 8. Push higher + Damper not lower + RH not higher + Pitot not lower
        if not damper_lower and not rh_higher and not pitot_lower:
            if damper_ok and rh_ok and pitot_ok:
                return DecisionResult(
                    channel="push_load",
                    action="apply_offset",
                    summary="Apply push-load offset",
                    detail="Push is the only channel outside tolerance. Nothing else supports the increased load.",
                )
            if damper_higher and rh_ok and pitot_ok:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail="Damper supports the push increase, but RH and pitot are neutral. Evidence is weak; validate push before applying an offset.",
                )
            if damper_ok and rh_lower and pitot_ok:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail="RH supports the push increase, but damper and pitot are neutral. Evidence is weak; validate push before applying an offset.",
                )
            if damper_ok and rh_ok and pitot_higher:
                return DecisionResult(
                    channel="push_load",
                    action="manual_review",
                    summary="Push offset candidate",
                    detail="Pitot supports the push increase, but neither mechanical channel confirms it. Validate push before applying an offset.",
                )
            if damper_higher and rh_lower and pitot_ok:
                return DecisionResult(
                    channel="push_load",
                    action="no_offset",
                    summary="No push-load offset",
                    detail="Damper and RH both support the higher push load. Pitot is neutral and does not contradict it.",
                )
            if damper_higher and rh_ok and pitot_higher:
                return DecisionResult(
                    channel="push_load",
                    action="no_offset",
                    summary="No push-load offset",
                    detail="Damper and pitot support the higher push load. RH is neutral and does not contradict it.",
                )
            if damper_ok and rh_lower and pitot_higher:
                return DecisionResult(
                    channel="push_load",
                    action="no_offset",
                    summary="No push-load offset",
                    detail="RH and pitot support the higher push load. Damper is neutral and does not contradict it.",
                )
            if damper_higher and rh_lower and pitot_higher:
                return DecisionResult(
                    channel="push_load",
                    action="no_offset",
                    summary="No push-load offset",
                    detail="Damper, RH, and pitot all support the higher push load. Data appears genuine.",
                )

        return DecisionResult(
            channel="push_load",
            action="manual_review",
            summary="Ambiguous push-load case",
            detail="This push-high combination does not land on any explicitly defined push decision path. Manual review recommended.",
        )

    if push_cmp.state != "lower_than_sim":
        return DecisionResult(
            channel="push_load",
            action="logic_not_defined",
            summary="Push load state not defined",
            detail=(
                "The current written push-load logic does not define this push-load state. "
                "Manual review recommended."
            ),
        )

    if (
        damper_cmp.state == "lower_than_sim"
        and rh_cmp.state == "higher_than_sim"
        and pitot_cmp.state == "lower_than_pair"
    ):
        return DecisionResult(
            channel="push_load",
            action="no_offset",
            summary="No push-load offset",
            detail="Data is genuine: push load lower with lower damper compression, higher RH, and lower pitot.",
        )

    if (
        damper_cmp.state == "lower_than_sim"
        and rh_cmp.state == "higher_than_sim"
        and pitot_cmp.state != "lower_than_pair"
    ):
        return DecisionResult(
            channel="push_load",
            action="no_offset",
            summary="No push-load offset",
            detail="Data seems genuine. Unknown reason may keep RH higher although pitot does not support lower aero load.",
        )

    if (
        damper_cmp.state == "lower_than_sim"
        and rh_cmp.state != "higher_than_sim"
        and pitot_cmp.state == "lower_than_pair"
    ):
        return DecisionResult(
            channel="push_load",
            action="no_offset",
            summary="No push-load offset",
            detail=(
                "Damper and pitot support genuine unloading, so keep push unchanged for now. "
                "Ride height is the inconsistent channel and should be investigated separately."
            ),
        )

    if (
        damper_cmp.state == "lower_than_sim"
        and rh_cmp.state != "higher_than_sim"
        and pitot_cmp.state != "lower_than_pair"
    ):
        return DecisionResult(
            channel="push_load",
            action="no_offset",
            summary="No push-load offset",
            detail="Damper and push-load data look genuine, but RH appears to be the issue.",
        )

    damper_ok = damper_cmp.state == "within_threshold"
    damper_higher = damper_cmp.state == "higher_than_sim"
    rh_higher = rh_cmp.state == "higher_than_sim"
    rh_ok = rh_cmp.state == "within_threshold"
    rh_lower = rh_cmp.state == "lower_than_sim"
    pitot_lower = pitot_cmp.state == "lower_than_pair"
    pitot_ok = pitot_cmp.state == "within_threshold"
    pitot_higher = pitot_cmp.state == "higher_than_pair"
    push_fit_supported = push_from_damper_fit.status == "match"
    rh_fit_supported = rh_from_push_fit.status == "match"

    # 1. Push low + RH high + Pitot low + Damper not low
    if damper_cmp.state != "lower_than_sim" and rh_higher and pitot_lower:
        if damper_higher:
            return DecisionResult(
                channel="push_load",
                action="no_offset",
                summary="No push-load offset",
                detail=(
                    "RH and pitot support a genuine lower push load. Damper does not follow the expected "
                    "lower-compression pattern and should be investigated separately."
                ),
            )
        if damper_ok:
            return DecisionResult(
                channel="push_load",
                action="no_offset",
                summary="No push-load offset",
                detail=(
                    "RH and pitot support a genuine lower push load. Damper is within threshold and is treated "
                    "as neutral, so no immediate push offset is needed."
                ),
            )

    # 2. Push low + RH high + Pitot not low + Damper not low
    if damper_cmp.state != "lower_than_sim" and rh_higher and not pitot_lower:
        if not push_fit_supported and not rh_fit_supported:
            return DecisionResult(
                channel="push_load",
                action="apply_offset",
                summary="Apply push-load offset",
                detail=(
                    "Push is low, but damper and pitot do not support a genuine push-low case, and both "
                    "Push load = f(Damper) and RH = f(Push load) fail to support the measured push. "
                    "Push offset is justified."
                ),
            )
        detail = (
            "Push is low with RH high, but damper and pitot do not fully support a genuine push-low case. "
            "Validate Push load = f(Damper) and RH = f(Push load) before applying a push offset."
        )
        if damper_higher or pitot_higher:
            detail = (
                "Push is low with RH high, but at least one helper channel moves in the opposite direction "
                "(damper and/or pitot). Validate Push load = f(Damper) and RH = f(Push load) before "
                "applying a push offset."
            )
        return DecisionResult(
            channel="push_load",
            action="manual_review",
            summary="Push offset candidate",
            detail=detail,
        )

    # 3. Push low + RH not high + Pitot low + Damper not low
    if damper_cmp.state != "lower_than_sim" and not rh_higher and pitot_lower:
        if not push_fit_supported and not rh_fit_supported:
            return DecisionResult(
                channel="push_load",
                action="apply_offset",
                summary="Apply push-load offset",
                detail=(
                    "Only pitot supports the lower push load, while RH and damper do not support it and both "
                    "Push load = f(Damper) and RH = f(Push load) fail the consistency check. Push offset is justified."
                ),
            )
        detail = (
            "Only pitot may support the lower push load. RH and damper do not confirm it strongly enough, "
            "so validate Push load = f(Damper) and RH = f(Push load) before applying a push offset."
        )
        if rh_lower or damper_higher:
            detail = (
                "Only pitot supports the lower push load, while RH and/or damper move in the opposite direction, "
                "which strengthens the push-offset hypothesis. Validate Push load = f(Damper) and RH = f(Push load) "
                "before applying a push offset."
            )
        return DecisionResult(
            channel="push_load",
            action="manual_review",
            summary="Push offset candidate",
            detail=detail,
        )

    # 4. Push low + RH not high + Pitot not low + Damper not low
    if damper_cmp.state != "lower_than_sim" and not rh_higher and not pitot_lower:
        detail = (
            "Push is low while the helper channels do not provide a genuine physical explanation. "
            "Push is isolated, so this is a strong push-offset candidate. Validate before applying correction."
        )
        if not (damper_ok and rh_ok and pitot_ok):
            detail = (
                "Push is low while one or more helper channels move in the opposite direction, which makes the "
                "push-offset hypothesis even stronger. Validate before applying correction."
            )
        return DecisionResult(
            channel="push_load",
            action="manual_review",
            summary="Push offset candidate",
            detail=detail,
        )

    return DecisionResult(
        channel="push_load",
        action="manual_review",
        summary="Ambiguous push-load case",
        detail="This push-load combination is not fully defined in the written tree. Manual review recommended.",
    )


def finalize_actions(
    rh_decision: DecisionResult,
    push_decision: DecisionResult,
    blockers: list[str] | None = None,
) -> DecisionResult:
    blocking_actions = {"manual_review", "logic_not_defined"}
    if rh_decision.action in blocking_actions or push_decision.action in blocking_actions:
        detail_parts: list[str] = []
        if rh_decision.action in blocking_actions:
            detail_parts.append(f"ride height: {rh_decision.summary}")
        if push_decision.action in blocking_actions:
            detail_parts.append(f"push load: {push_decision.summary}")
        if blockers:
            detail_parts.extend(blockers)
        return DecisionResult(
            channel="final",
            action="hold_offsets",
            summary="Hold offsets",
            detail=" | ".join(detail_parts) if detail_parts else "Manual review before applying changes.",
        )

    approved = []
    if rh_decision.action == "apply_offset":
        approved.append("ride_height")
    if push_decision.action == "apply_offset":
        approved.append("push_load")

    if not approved:
        summary = "Approved offsets: none"
        detail = "Both primary channels are treated as genuine / no-offset cases."
    else:
        approved_labels = ", ".join(item.replace("_", " ") for item in approved)
        summary = f"Approved offsets: {approved_labels}"
        detail = "Final consistency loop complete."

    return DecisionResult(
        channel="final",
        action="apply_approved_offsets",
        summary=summary,
        detail=detail,
    )


def build_offset_blockers(
    channel_set: ChannelSet,
    rh_cmp: ComparisonResult,
    damper_cmp: ComparisonResult,
    push_cmp: ComparisonResult,
    pitot_cmp: ComparisonResult,
    tyre_cmp: ComparisonResult,
) -> list[str]:
    blockers: list[str] = []

    if rh_cmp.state == "higher_than_sim":
        if tyre_cmp.state == "higher_than_sim":
            blockers.append(
                "Tyre pressure is also higher than simulation, so it can explain the higher ride height."
            )
        if damper_cmp.state != "lower_than_sim":
            blockers.append(
                f"Damper does not support genuine RH-high: {channel_set.damper} is {damper_cmp.state}, not lower_than_sim."
            )
        if push_cmp.state != "lower_than_sim":
            blockers.append(
                f"Push load does not support genuine RH-high: {channel_set.push_load} is {push_cmp.state}, not lower_than_sim."
            )
        if pitot_cmp.state != "lower_than_pair":
            blockers.append(
                f"Pitot does not support the RH-high branch: pitot is {pitot_cmp.state}, not lower_than_pair."
            )

    if push_cmp.state == "lower_than_sim":
        if rh_cmp.state != "higher_than_sim":
            blockers.append(
                f"Ride height does not support genuine push-low: {channel_set.ride_height} is {rh_cmp.state}, not higher_than_sim."
            )
        if damper_cmp.state != "lower_than_sim":
            blockers.append(
                f"Damper does not support genuine push-low: {channel_set.damper} is {damper_cmp.state}, not lower_than_sim."
            )
        if pitot_cmp.state != "lower_than_pair":
            blockers.append(
                f"Pitot does not support the push-low branch: pitot is {pitot_cmp.state}, not lower_than_pair."
            )

    if push_cmp.state == "higher_than_sim":
        if damper_cmp.state == "lower_than_sim":
            blockers.append(
                f"Damper contradicts genuine push-high: {channel_set.damper} is lower_than_sim."
            )
        if rh_cmp.state == "higher_than_sim":
            blockers.append(
                f"Ride height contradicts genuine push-high: {channel_set.ride_height} is higher_than_sim."
            )
        if pitot_cmp.state == "lower_than_pair":
            blockers.append(
                "Pitot contradicts genuine push-high: pitot is lower_than_pair."
            )

    return blockers


def decision_detail_label(action: str) -> str:
    if action == "apply_offset":
        return "  why offset:"
    if action == "no_offset":
        return "  why no offset:"
    return "  why review is needed:"


def detect_stable_plateau(
    synced_pit: pd.DataFrame,
    channel_set: ChannelSet,
) -> PlateauEvidence:
    required = [
        "real__carspeed_art",
        "real__avg_accx",
        "real__avg_accy",
        f"real__{channel_set.ride_height}",
        f"real__{channel_set.push_load}",
    ]
    if any(column not in synced_pit.columns for column in required):
        return PlateauEvidence(
            start_index=None,
            end_index=None,
            sample_count=0,
            speed_gradient_mean=None,
            accx_mean_abs=None,
            accy_mean_abs=None,
            ride_height_variance=None,
            push_load_variance=None,
            detail="plateau unavailable (missing required channels)",
        )

    sample_count = len(synced_pit)
    if sample_count < 8:
        return PlateauEvidence(
            start_index=0 if sample_count else None,
            end_index=sample_count - 1 if sample_count else None,
            sample_count=sample_count,
            speed_gradient_mean=None,
            accx_mean_abs=None,
            accy_mean_abs=None,
            ride_height_variance=None,
            push_load_variance=None,
            detail="plateau fallback (too few samples to detect a stable subsection)",
        )

    window = max(8, int(np.ceil(sample_count * 0.25)))
    speed = pd.to_numeric(synced_pit["real__carspeed_art"], errors="coerce").reset_index(drop=True)
    accx = pd.to_numeric(synced_pit["real__avg_accx"], errors="coerce").reset_index(drop=True)
    accy = pd.to_numeric(synced_pit["real__avg_accy"], errors="coerce").reset_index(drop=True)
    rh = pd.to_numeric(synced_pit[f"real__{channel_set.ride_height}"], errors="coerce").reset_index(drop=True)
    push = pd.to_numeric(synced_pit[f"real__{channel_set.push_load}"], errors="coerce").reset_index(drop=True)

    candidates: list[tuple[float, int, int, float, float, float, float, float]] = []
    for start in range(0, sample_count - window + 1):
        end = start + window
        speed_window = speed.iloc[start:end].dropna()
        accx_window = accx.iloc[start:end].dropna()
        accy_window = accy.iloc[start:end].dropna()
        rh_window = rh.iloc[start:end].dropna()
        push_window = push.iloc[start:end].dropna()
        if min(len(speed_window), len(accx_window), len(accy_window), len(rh_window), len(push_window)) < max(5, window // 2):
            continue
        speed_gradient_mean = float(speed_window.diff().abs().dropna().mean()) if len(speed_window) > 1 else 0.0
        accx_mean_abs = float(accx_window.abs().mean())
        accy_mean_abs = float(accy_window.abs().mean())
        rh_variance = float(rh_window.var(ddof=0))
        push_variance = float(push_window.var(ddof=0))
        candidates.append(
            (
                speed_gradient_mean + accx_mean_abs + accy_mean_abs + rh_variance + (push_variance / max(abs(push_window.median()), 1.0)),
                start,
                end - 1,
                speed_gradient_mean,
                accx_mean_abs,
                accy_mean_abs,
                rh_variance,
                push_variance,
            )
        )

    if not candidates:
        return PlateauEvidence(
            start_index=None,
            end_index=None,
            sample_count=0,
            speed_gradient_mean=None,
            accx_mean_abs=None,
            accy_mean_abs=None,
            ride_height_variance=None,
            push_load_variance=None,
            detail="plateau unavailable (no valid stable subsection candidate)",
        )

    _, start_index, end_index, speed_gradient_mean, accx_mean_abs, accy_mean_abs, rh_variance, push_variance = min(
        candidates,
        key=lambda item: item[0],
    )
    return PlateauEvidence(
        start_index=start_index,
        end_index=end_index,
        sample_count=end_index - start_index + 1,
        speed_gradient_mean=speed_gradient_mean,
        accx_mean_abs=accx_mean_abs,
        accy_mean_abs=accy_mean_abs,
        ride_height_variance=rh_variance,
        push_load_variance=push_variance,
        detail=(
            f"stable plateau samples={end_index - start_index + 1}, "
            f"indices={start_index}->{end_index}, "
            f"speed_grad_mean={format_optional_value(speed_gradient_mean)}, "
            f"|accx|={format_optional_value(accx_mean_abs)}, "
            f"|accy|={format_optional_value(accy_mean_abs)}, "
            f"rh_var={format_optional_value(rh_variance)}, "
            f"push_var={format_optional_value(push_variance)}"
        ),
    )


def slice_plateau_df(
    synced_pit: pd.DataFrame,
    plateau: PlateauEvidence,
) -> pd.DataFrame | None:
    if plateau.start_index is None or plateau.end_index is None or plateau.sample_count <= 0:
        return None
    return synced_pit.iloc[plateau.start_index : plateau.end_index + 1].copy()


def build_helper_agreement_summary(
    target_channel: str,
    rh_cmp: ComparisonResult,
    damper_cmp: ComparisonResult,
    push_cmp: ComparisonResult,
    pitot_cmp: ComparisonResult,
    tyre_cmp: ComparisonResult | None = None,
) -> HelperAgreementSummary:
    supports = 0
    contradictions = 0
    neutrals = 0
    details: list[str] = []

    def classify_binary(condition: bool, label: str) -> None:
        nonlocal supports, contradictions, neutrals
        if condition:
            supports += 1
            details.append(f"{label}=support")
        else:
            contradictions += 1
            details.append(f"{label}=contradiction")

    def classify_state(actual_state: str, support_state: str, neutral_states: set[str], label: str) -> None:
        nonlocal supports, contradictions, neutrals
        if actual_state == support_state:
            supports += 1
            details.append(f"{label}=support")
        elif actual_state in neutral_states:
            neutrals += 1
            details.append(f"{label}=neutral")
        else:
            contradictions += 1
            details.append(f"{label}=contradiction")

    if target_channel == "ride_height" and rh_cmp.state == "higher_than_sim":
        classify_state(damper_cmp.state, "lower_than_sim", {"within_threshold"}, "damper lower than sim")
        classify_state(push_cmp.state, "lower_than_sim", {"within_threshold"}, "push lower than sim")
        classify_state(pitot_cmp.state, "lower_than_pair", {"within_threshold"}, "pitot lower than pAir")
        if tyre_cmp is not None:
            if tyre_cmp.state == "higher_than_sim":
                contradictions += 1
                details.append("tpms higher than sim=offset blocker")
            elif tyre_cmp.state == "within_threshold":
                neutrals += 1
                details.append("tpms within threshold=neutral")
            else:
                neutrals += 1
                details.append(f"tpms {tyre_cmp.state}=neutral")
    elif target_channel == "push_load" and push_cmp.state == "lower_than_sim":
        classify_state(damper_cmp.state, "lower_than_sim", {"within_threshold"}, "damper lower than sim")
        classify_state(rh_cmp.state, "higher_than_sim", {"within_threshold"}, "ride height higher than sim")
        classify_state(pitot_cmp.state, "lower_than_pair", {"within_threshold"}, "pitot lower than pAir")
    elif target_channel == "push_load" and push_cmp.state == "higher_than_sim":
        classify_state(damper_cmp.state, "higher_than_sim", {"within_threshold"}, "damper higher than sim")
        classify_state(rh_cmp.state, "lower_than_sim", {"within_threshold"}, "ride height lower than sim")
        classify_state(pitot_cmp.state, "higher_than_pair", {"within_threshold"}, "pitot higher than pAir")
    else:
        details.append("helper agreement not applicable for current branch state")

    total = supports + contradictions + neutrals
    return HelperAgreementSummary(
        support_count=supports,
        contradiction_count=contradictions,
        neutral_count=neutrals,
        total_count=total,
        detail=", ".join(details),
    )


def decision_confidence_label(
    decision: DecisionResult,
    comparison: ComparisonResult,
    plateau_comparison: ComparisonResult | None,
    helper_agreement: HelperAgreementSummary,
    fit_results: list[FitCheckResult],
    time_series: TimeSeriesAgreementEvidence | None,
) -> str:
    if decision.action in {"manual_review", "logic_not_defined"}:
        return "low"

    score = 0
    if comparison.evidence is not None:
        evidence = comparison.evidence
        if evidence.within_fraction is not None and evidence.within_fraction >= 0.65:
            score += 1
        if evidence.positive_fraction is not None and evidence.negative_fraction is not None:
            dominant = max(evidence.positive_fraction, evidence.negative_fraction)
            if dominant >= STABLE_SIGN_MIN_FRACTION:
                score += 1
        if evidence.longest_above_run is not None and evidence.longest_below_run is not None:
            longest = max(evidence.longest_above_run, evidence.longest_below_run)
            if longest >= max(1, int(np.ceil(evidence.sample_count * STABLE_RUN_MIN_FRACTION))):
                score += 1
    if plateau_comparison is not None and plateau_comparison.state == comparison.state:
        score += 1
    if helper_agreement.support_count > helper_agreement.contradiction_count:
        score += 1
    if fit_results:
        good_fits = sum(1 for fit in fit_results if fit.status == "match")
        if good_fits >= max(1, len(fit_results) // 2):
            score += 1
    if time_series is not None and time_series.best_correlation is not None and time_series.best_correlation >= 0.7:
        score += 1

    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def evidence_persistence_score(comparison: ComparisonResult) -> int:
    evidence = comparison.evidence
    if evidence is None:
        return 0

    score = 0
    if evidence.positive_fraction is not None and evidence.negative_fraction is not None:
        dominant_fraction = max(evidence.positive_fraction, evidence.negative_fraction)
        if dominant_fraction >= STABLE_SIGN_MIN_FRACTION:
            score += 1
    if evidence.longest_above_run is not None and evidence.longest_below_run is not None:
        longest_run = max(evidence.longest_above_run, evidence.longest_below_run)
        if longest_run >= max(1, int(np.ceil(evidence.sample_count * STABLE_RUN_MIN_FRACTION))):
            score += 1
    if comparison.state == "within_threshold":
        if evidence.within_fraction is not None and evidence.within_fraction >= 0.65:
            score += 1
    else:
        target_fraction = evidence.above_fraction if comparison.state == "higher_than_sim" else evidence.below_fraction
        if target_fraction is not None and target_fraction >= 0.6:
            score += 1
    return score


def unresolved_decision_evidence_summary(
    *,
    comparison: ComparisonResult,
    plateau_comparison: ComparisonResult | None,
    helper_agreement: HelperAgreementSummary,
    fit_results: list[FitCheckResult],
    time_series: TimeSeriesAgreementEvidence | None,
) -> tuple[int, int, list[str]]:
    offset_score = 0
    genuine_score = 0
    reasons: list[str] = []

    persistence = evidence_persistence_score(comparison)
    if persistence > 0:
        reasons.append(f"primary persistence score={persistence}")
    if comparison.state != "within_threshold" and persistence >= 2:
        offset_score += 1
    if comparison.state == "within_threshold" and persistence >= 2:
        genuine_score += 1

    if plateau_comparison is not None:
        if plateau_comparison.state == comparison.state:
            reasons.append(f"plateau agrees with whole-band state ({comparison.state})")
            if comparison.state == "within_threshold":
                genuine_score += 1
            else:
                offset_score += 1
        else:
            reasons.append(
                f"plateau disagrees with whole-band state (whole={comparison.state}, plateau={plateau_comparison.state})"
            )

    if helper_agreement.support_count > helper_agreement.contradiction_count:
        genuine_score += 2
        reasons.append(
            f"helper agreement favors genuine data ({helper_agreement.support_count} support vs {helper_agreement.contradiction_count} contradiction)"
        )
    elif helper_agreement.contradiction_count > helper_agreement.support_count:
        offset_score += 2
        reasons.append(
            f"helper agreement favors offset ({helper_agreement.contradiction_count} contradiction vs {helper_agreement.support_count} support)"
        )
    else:
        reasons.append(
            f"helper agreement is balanced ({helper_agreement.support_count} support, {helper_agreement.contradiction_count} contradiction)"
        )

    fit_match_count = sum(1 for fit in fit_results if fit.status == "match")
    fit_mismatch_count = sum(1 for fit in fit_results if fit.status == "mismatch")
    if fit_match_count or fit_mismatch_count:
        reasons.append(f"XY evidence: {fit_match_count} match / {fit_mismatch_count} mismatch")
    if fit_match_count > fit_mismatch_count:
        genuine_score += 1
    elif fit_mismatch_count > fit_match_count:
        offset_score += 1

    if time_series is not None and time_series.best_correlation is not None:
        if time_series.best_correlation >= 0.7:
            genuine_score += 1
            reasons.append(f"time-series agreement supports genuineness (corr={time_series.best_correlation:.3f})")
        elif time_series.best_correlation <= 0.2:
            offset_score += 1
            reasons.append(f"time-series agreement is weak (corr={time_series.best_correlation:.3f})")

    return offset_score, genuine_score, reasons


def resolve_unsettled_decision_with_evidence_layers(
    *,
    decision: DecisionResult,
    channel: str,
    comparison: ComparisonResult,
    plateau_comparison: ComparisonResult | None,
    helper_agreement: HelperAgreementSummary,
    fit_results: list[FitCheckResult],
    time_series: TimeSeriesAgreementEvidence | None,
) -> DecisionResult:
    if decision.action not in {"manual_review", "logic_not_defined"}:
        return decision

    offset_score, genuine_score, reasons = unresolved_decision_evidence_summary(
        comparison=comparison,
        plateau_comparison=plateau_comparison,
        helper_agreement=helper_agreement,
        fit_results=fit_results,
        time_series=time_series,
    )
    reason_detail = " | ".join(reasons) if reasons else "insufficient layered evidence"

    if offset_score >= genuine_score + 2 and offset_score >= 3:
        summary = "Apply ride-height offset" if channel == "ride_height" else "Apply push-load offset"
        return DecisionResult(
            channel=channel,
            action="apply_offset",
            summary=summary,
            detail=f"Evidence-layer resolution favors offset. {reason_detail}",
        )

    if genuine_score >= offset_score + 2 and genuine_score >= 3:
        summary = "RH OK" if channel == "ride_height" else "Push load OK"
        return DecisionResult(
            channel=channel,
            action="no_offset",
            summary=summary,
            detail=f"Evidence-layer resolution favors genuine / no-offset data. {reason_detail}",
        )

    return DecisionResult(
        channel=channel,
        action="manual_review",
        summary=decision.summary,
        detail=f"{decision.detail} Evidence-layer result remains inconclusive. {reason_detail}",
    )


def build_conflict_checks(
    rh_decision: DecisionResult,
    push_decision: DecisionResult,
    rh_cmp: ComparisonResult,
    push_cmp: ComparisonResult,
) -> list[str]:
    conflicts: list[str] = []
    if rh_decision.action == "no_offset" and push_decision.action == "apply_offset":
        conflicts.append("RH says no offset while push load says apply offset.")
    if rh_decision.action == "apply_offset" and push_decision.action == "no_offset":
        conflicts.append("RH says apply offset while push load says no offset.")
    if rh_cmp.state == "higher_than_sim" and push_cmp.state not in {"lower_than_sim", "within_threshold"}:
        conflicts.append("RH is high but push load does not support the expected RH-high physical pattern.")
    if push_cmp.state in {"lower_than_sim", "higher_than_sim"} and rh_cmp.state == "within_threshold":
        conflicts.append("Push load is out of threshold while ride height stays within threshold.")
    return conflicts


def main_state_summary(
    label: str,
    comparison: ComparisonResult,
) -> str:
    if comparison.state == "within_threshold":
        return f"{label} is within threshold"
    if comparison.state == "higher_than_sim":
        return f"{label} is out of threshold and higher than simulation"
    if comparison.state == "lower_than_sim":
        return f"{label} is out of threshold and lower than simulation"
    return f"{label} comparison state is {comparison.state}"


def select_channel_blockers(
    channel: str,
    blockers: list[str],
) -> list[str]:
    if channel == "ride_height":
        ride_height_markers = (
            "Tyre pressure is also higher than simulation",
            "Damper does not support genuine RH-high",
            "Push load does not support genuine RH-high",
            "Pitot does not support the RH-high branch",
        )
        return [item for item in blockers if item.startswith(ride_height_markers)]
    if channel == "push_load":
        push_markers = (
            "Ride height does not support genuine push-low",
            "Damper does not support genuine push-low",
            "Pitot does not support the push-low branch",
            "Damper contradicts genuine push-high",
            "Ride height contradicts genuine push-high",
            "Pitot contradicts genuine push-high",
        )
        return [item for item in blockers if item.startswith(push_markers)]
    return []


def print_critical_decision_summary(
    rh_cmp: ComparisonResult,
    push_cmp: ComparisonResult,
    rh_decision: DecisionResult,
    push_decision: DecisionResult,
    blockers: list[str],
    plateau_rh_cmp: ComparisonResult | None,
    plateau_push_cmp: ComparisonResult | None,
    rh_helper_agreement: HelperAgreementSummary,
    push_helper_agreement: HelperAgreementSummary,
    rh_confidence: str,
    push_confidence: str,
) -> None:
    print_section("Critical Decision Summary")

    rh_blockers = select_channel_blockers("ride_height", blockers)
    rh_gate_cmp = plateau_rh_cmp if plateau_rh_cmp is not None else rh_cmp
    push_gate_cmp = plateau_push_cmp if plateau_push_cmp is not None else push_cmp

    print(
        "ride height: "
        f"whole pit = {rh_cmp.state}"
        + (
            f", plateau gate = {plateau_rh_cmp.state}"
            if plateau_rh_cmp is not None
            else ", plateau gate unavailable"
        )
    )
    print(
        "  whole-pit rolling medians: "
        f"real={format_optional_value(rh_cmp.track_median)}, "
        f"sim={format_optional_value(rh_cmp.reference_median)}, "
        f"delta={format_optional_value(rh_cmp.difference)}"
    )
    if plateau_rh_cmp is not None:
        print(
            "  plateau-gate rolling medians: "
            f"real={format_optional_value(plateau_rh_cmp.track_median)}, "
            f"sim={format_optional_value(plateau_rh_cmp.reference_median)}, "
            f"delta={format_optional_value(plateau_rh_cmp.difference)}"
        )
    print(f"  actual decision basis: {rh_gate_cmp.state}")
    print(
        "  helper agreement: "
        f"supports={rh_helper_agreement.support_count}, "
        f"contradictions={rh_helper_agreement.contradiction_count}, "
        f"neutral={rh_helper_agreement.neutral_count}"
    )
    print(f"  confidence: {rh_confidence}")
    print(f"  outcome: {rh_decision.summary}")
    print(f"  reason: {rh_decision.detail}")
    if rh_blockers:
        print(decision_detail_label(rh_decision.action))
        for blocker in rh_blockers:
            print(f"- {blocker}")

    print()

    print(
        "push load: "
        f"whole pit = {push_cmp.state}"
        + (
            f", plateau gate = {plateau_push_cmp.state}"
            if plateau_push_cmp is not None
            else ", plateau gate unavailable"
        )
    )
    print(
        "  whole-pit rolling medians: "
        f"real={format_optional_value(push_cmp.track_median)}, "
        f"sim={format_optional_value(push_cmp.reference_median)}, "
        f"delta={format_optional_value(push_cmp.difference)}"
    )
    if plateau_push_cmp is not None:
        print(
            "  plateau-gate rolling medians: "
            f"real={format_optional_value(plateau_push_cmp.track_median)}, "
            f"sim={format_optional_value(plateau_push_cmp.reference_median)}, "
            f"delta={format_optional_value(plateau_push_cmp.difference)}"
        )
    print(f"  actual decision basis: {push_gate_cmp.state}")
    print(
        "  helper agreement: "
        f"supports={push_helper_agreement.support_count}, "
        f"contradictions={push_helper_agreement.contradiction_count}, "
        f"neutral={push_helper_agreement.neutral_count}"
    )
    print(f"  confidence: {push_confidence}")
    print(f"  outcome: {push_decision.summary}")
    print(f"  reason: {push_decision.detail}")
    push_blockers = select_channel_blockers("push_load", blockers)
    if push_blockers:
        print(decision_detail_label(push_decision.action))
        for blocker in push_blockers:
            print(f"- {blocker}")


def print_decision_overview(
    rh_decision: DecisionResult,
    push_decision: DecisionResult,
    final_decision: DecisionResult,
) -> None:
    print_section("Decisions")

    print("ride height")
    print(f"  outcome: {rh_decision.summary}")
    print(f"  action: {rh_decision.action}")
    print(f"  reason: {rh_decision.detail}")
    print()

    print("push load")
    print(f"  outcome: {push_decision.summary}")
    print(f"  action: {push_decision.action}")
    print(f"  reason: {push_decision.detail}")
    print()

    print("final")
    print(f"  outcome: {final_decision.summary}")
    print(f"  action: {final_decision.action}")
    print(f"  reason: {final_decision.detail}")


def select_gate_comparison(
    whole_pit_cmp: ComparisonResult,
    plateau_cmp: ComparisonResult | None,
) -> ComparisonResult:
    if plateau_cmp is not None:
        return plateau_cmp
    return whole_pit_cmp


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def format_optional_value(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.4f}"


def evaluate_condition_comparability(
    real_series: pd.Series,
    simu_series: pd.Series,
    threshold: float,
    channel_name: str,
    min_within_fraction: float = 0.65,
    min_overlap_ratio: float = 0.50,
) -> ConditionComparabilityResult:
    metrics = compute_signal_comparison_metrics(real_series=real_series, simu_series=simu_series)
    aligned = pd.DataFrame(
        {
            "real": pd.to_numeric(real_series, errors="coerce"),
            "simu": pd.to_numeric(simu_series, errors="coerce"),
        }
    ).dropna()

    if aligned.empty:
        return ConditionComparabilityResult(
            channel_name=channel_name,
            state="unavailable",
            threshold=threshold,
            sample_count=0,
            median_real=None,
            median_simu=None,
            median_delta=None,
            mae=None,
            pearson_r=None,
            within_fraction=None,
            overlap_ratio=None,
            detail=f"{channel_name}: no aligned samples available for comparability check",
        )

    diff = aligned["real"].to_numpy(dtype="float64") - aligned["simu"].to_numpy(dtype="float64")
    within_fraction = float(np.mean(np.abs(diff) <= threshold))

    real_low = float(np.nanpercentile(aligned["real"], 10))
    real_high = float(np.nanpercentile(aligned["real"], 90))
    simu_low = float(np.nanpercentile(aligned["simu"], 10))
    simu_high = float(np.nanpercentile(aligned["simu"], 90))

    overlap_low = max(real_low, simu_low)
    overlap_high = min(real_high, simu_high)
    union_low = min(real_low, simu_low)
    union_high = max(real_high, simu_high)
    union_span = union_high - union_low
    if union_span <= 0:
        overlap_ratio = 1.0
    else:
        overlap_ratio = max(0.0, overlap_high - overlap_low) / union_span

    comparable = (
        metrics.mae is not None
        and metrics.mae <= threshold
        and within_fraction >= min_within_fraction
    )
    state = "comparable" if comparable else "not_comparable"

    checks = [
        f"mae {format_metric_value(metrics.mae)} <= {threshold:.4f}",
        f"within {within_fraction:.1%} >= {min_within_fraction:.1%}",
        f"overlap {overlap_ratio:.1%} (diagnostic, target {min_overlap_ratio:.1%})",
    ]
    if comparable:
        outcome = "PASS"
    else:
        outcome = "FAIL"

    return ConditionComparabilityResult(
        channel_name=channel_name,
        state=state,
        threshold=threshold,
        sample_count=metrics.sample_count,
        median_real=metrics.median_real,
        median_simu=metrics.median_sim,
        median_delta=metrics.median_delta,
        mae=metrics.mae,
        pearson_r=metrics.pearson_r,
        within_fraction=within_fraction,
        overlap_ratio=overlap_ratio,
        detail=(
            f"{channel_name} [{outcome}]: "
            f"median_delta={format_metric_value(metrics.median_delta)}, "
            f"mae={format_metric_value(metrics.mae)}, "
            f"within={within_fraction:.1%}, overlap={overlap_ratio:.1%}, "
            f"pearson_r={format_metric_value(metrics.pearson_r)} "
            f"({' ; '.join(checks)})"
        ),
    )


def summarize_segment_comparability(
    results: list[ConditionComparabilityResult],
) -> SegmentComparabilityResult:
    if not results:
        return SegmentComparabilityResult(
            comparable=False,
            detail="No comparability checks were evaluated.",
            channel_results=tuple(),
        )

    unavailable = [result.channel_name for result in results if result.state == "unavailable"]
    failed = [result.channel_name for result in results if result.state == "not_comparable"]
    if unavailable:
        unavailable_details = [
            result.detail for result in results if result.state == "unavailable"
        ]
        detail = (
            "Condition pre-check unavailable for: " + ", ".join(unavailable) +
            ". Manual review before trusting the pitlane comparison. "
            + " | ".join(unavailable_details)
        )
        return SegmentComparabilityResult(
            comparable=False,
            detail=detail,
            channel_results=tuple(results),
        )
    if failed:
        failed_details = [
            result.detail for result in results if result.state == "not_comparable"
        ]
        detail = (
            "Condition mismatch detected for: " + ", ".join(failed) +
            ". Real and sim pit segments are not operating in the same envelope. "
            + " | ".join(failed_details)
        )
        return SegmentComparabilityResult(
            comparable=False,
            detail=detail,
            channel_results=tuple(results),
        )

    return SegmentComparabilityResult(
        comparable=True,
        detail="Speed, longitudinal acceleration, and lateral acceleration are comparable enough to trust the synchronized pitlane signal checks.",
        channel_results=tuple(results),
    )


def print_signal_metrics(
    synced_df: pd.DataFrame,
    label: str,
    real_column: str,
    simu_column: str,
) -> None:
    metrics = compute_signal_comparison_metrics(
        real_series=synced_df[real_column],
        simu_series=synced_df[simu_column],
    )
    print(
        f"{label}: "
        f"n={metrics.sample_count}, "
        f"median {format_metric_value(metrics.median_real)} - "
        f"{format_metric_value(metrics.median_sim)} = "
        f"{format_metric_value(metrics.median_delta)}, "
        f"mean_error={format_metric_value(metrics.mean_error)}, "
        f"rmse={format_metric_value(metrics.rmse)}, "
        f"nrmse={format_metric_value(None if metrics.nrmse is None else 100.0 * metrics.nrmse, '%')}, "
        f"mae={format_metric_value(metrics.mae)}, "
        f"pearson_r={format_metric_value(metrics.pearson_r)}"
    )


def build_normalized_x(df: pd.DataFrame) -> pd.Series:
    if len(df) <= 1:
        return pd.Series([0.0] * len(df), index=df.index, dtype="float64")
    return pd.Series(
        np.linspace(0.0, 1.0, len(df)),
        index=df.index,
        dtype="float64",
    )


def plot_time_domain_evidence(
    synced_pit: pd.DataFrame,
    real_column: str,
    simu_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = pd.to_numeric(synced_pit["sync_progress"], errors="coerce")
    real_y = pd.to_numeric(synced_pit[real_column], errors="coerce")
    simu_y = pd.to_numeric(synced_pit[simu_column], errors="coerce")

    plt.figure(figsize=(12, 5))
    if real_y.notna().any():
        plt.plot(x[real_y.notna()], real_y[real_y.notna()], label=f"track {real_column}")
    if simu_y.notna().any():
        plt.plot(x[simu_y.notna()], simu_y[simu_y.notna()], label=f"sim {simu_column}")
    plt.title(title)
    plt.xlabel("Normalized pit progress")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.show()


def plot_threshold_check(
    synced_pit: pd.DataFrame,
    real_column: str,
    simu_column: str,
    threshold: float,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = pd.to_numeric(synced_pit["sync_progress"], errors="coerce")
    real_y = pd.to_numeric(synced_pit[real_column], errors="coerce")
    simu_y = pd.to_numeric(synced_pit[simu_column], errors="coerce")
    diff = real_y - simu_y

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    if real_y.notna().any():
        axes[0].plot(x[real_y.notna()], real_y[real_y.notna()], label=f"track {real_column}")
    if simu_y.notna().any():
        axes[0].plot(x[simu_y.notna()], simu_y[simu_y.notna()], label=f"sim {simu_column}")
    axes[0].set_title(title)
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if diff.notna().any():
        axes[1].plot(x[diff.notna()], diff[diff.notna()], color="tab:orange", label="track - sim")
    axes[1].axhline(threshold, color="red", linestyle="--", linewidth=1.2, label="+threshold")
    axes[1].axhline(-threshold, color="red", linestyle="--", linewidth=1.2, label="-threshold")
    axes[1].axhline(0.0, color="white", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Normalized pit progress")
    axes[1].set_ylabel("Delta")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.show()


def plot_xy_evidence(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    fit_result: FitCheckResult,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = pd.to_numeric(df[x_column], errors="coerce")
    y = pd.to_numeric(df[y_column], errors="coerce")
    mask = x.notna() & y.notna()

    plt.figure(figsize=(7, 6))
    plt.scatter(x[mask], y[mask], s=18, alpha=0.7, label="track points")

    if fit_result.slope is not None and fit_result.intercept is not None and mask.any():
        x_fit = x[mask].to_numpy(dtype="float64")
        x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
        y_line = fit_result.slope * x_line + fit_result.intercept
        plt.plot(x_line, y_line, color="black", linestyle="--", linewidth=2.0, label="fit")
        plt.text(
            0.04,
            0.96,
            (
                f"y = {fit_result.slope:.4f}x + {fit_result.intercept:.4f}\n"
                f"status = {fit_result.status}"
            ),
            transform=plt.gca().transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.show()


def plot_selected_pit_context(
    real_df: pd.DataFrame,
    simu_df: pd.DataFrame,
    real_pit: pd.DataFrame,
    simu_pit: pd.DataFrame,
    synced_pit: pd.DataFrame,
    channel_set: ChannelSet,
    channel_set_key: str,
    output_path: Path,
) -> None:
    overlap_columns = [
        f"real__{channel_set.ride_height}",
        f"simu__{channel_set.ride_height}",
        f"real__{channel_set.push_load}",
        f"simu__{channel_set.push_load}",
        f"real__{channel_set.damper}",
        f"simu__{channel_set.damper}",
        f"real__{channel_set.tyre_pressure}",
        f"simu__{channel_set.tyre_pressure}",
        "real__carspeed_art",
        "simu__carspeed_art",
        "real__avg_accx",
        "simu__avg_accx",
        "real__avg_accy",
        "simu__avg_accy",
    ]
    present_columns = [column for column in overlap_columns if column in synced_pit.columns]
    overlap_start = None
    overlap_end = None
    if present_columns:
        overlap_mask = synced_pit[present_columns].notna().all(axis=1)
        if overlap_mask.any():
            overlap_progress = pd.to_numeric(
                synced_pit.loc[overlap_mask, "sync_progress"],
                errors="coerce",
            ).dropna()
            if not overlap_progress.empty:
                overlap_start = float(overlap_progress.min())
                overlap_end = float(overlap_progress.max())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for ax, df, pit_df, source in [
        (axes[0], real_df, real_pit, "track"),
        (axes[1], simu_df, simu_pit, "sim"),
    ]:
        x = pd.Series(np.arange(len(df)), index=df.index, dtype="int64")
        speed = pd.to_numeric(df["carspeed_art"], errors="coerce")
        full_mask = speed.notna()
        ax.plot(
            x[full_mask],
            speed[full_mask],
            color="gray",
            linewidth=0.8,
            alpha=0.6,
            label=f"{source} full run",
        )

        pit_x = pd.Series(pit_df.index.to_numpy(dtype="int64"), index=pit_df.index, dtype="int64")
        pit_speed = pd.to_numeric(pit_df["carspeed_art"], errors="coerce")
        pit_mask = pit_speed.notna()
        ax.plot(
            pit_x[pit_mask],
            pit_speed[pit_mask],
            color="magenta",
            linewidth=2.2,
            label=f"{source} compared pit segment",
        )
        start_idx = int(pit_df.index.min())
        end_idx = int(pit_df.index.max())
        ax.axvline(
            start_idx,
            color="magenta",
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            label=f"{source} pit start",
        )
        ax.axvline(
            end_idx,
            color="magenta",
            linestyle=":",
            linewidth=1.4,
            alpha=0.9,
            label=f"{source} pit end",
        )
        if overlap_start is not None and overlap_end is not None:
            pit_progress = build_normalized_x(pit_df)
            overlap_local_mask = (pit_progress >= overlap_start) & (pit_progress <= overlap_end)
            if overlap_local_mask.any():
                overlap_indices = pit_df.index[overlap_local_mask]
                overlap_start_idx = int(overlap_indices.min())
                overlap_end_idx = int(overlap_indices.max())
                ax.axvline(
                    overlap_start_idx,
                    color="deepskyblue",
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.9,
                    label=f"{source} sync overlap start",
                )
                ax.axvline(
                    overlap_end_idx,
                    color="deepskyblue",
                    linestyle=":",
                    linewidth=1.4,
                    alpha=0.9,
                    label=f"{source} sync overlap end",
                )
        ax.set_title(
            f"{source}: full run with compared pit segment "
            f"({start_idx} -> {end_idx})"
        )
        ax.set_xlabel("global_sample")
        ax.set_ylabel("carspeed_art")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"{channel_set_key}: selected pit segments used in current comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.show()


def generate_evidence_plots(
    output_dir: Path,
    segment_label: str,
    channel_set_key: str,
    channel_set: ChannelSet,
    real_df: pd.DataFrame,
    simu_df: pd.DataFrame,
    raw_aligned_pit: pd.DataFrame,
    synced_pit: pd.DataFrame,
    real_pit: pd.DataFrame,
    simu_pit: pd.DataFrame,
    thresholds: ThresholdConfig,
    rh_from_damper_fit: FitCheckResult,
    rh_from_push_fit: FitCheckResult,
    push_from_damper_fit: FitCheckResult,
    pitot_from_pair_fit: FitCheckResult,
    condition_checks: SegmentComparabilityResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_selected_pit_context(
        real_df=real_df,
        simu_df=simu_df,
        real_pit=real_pit,
        simu_pit=simu_pit,
        synced_pit=synced_pit,
        channel_set=channel_set,
        channel_set_key=f"{segment_label}_{channel_set_key}",
        output_path=output_dir / f"{segment_label}_{channel_set_key}_selected_context.png",
    )

    plot_time_domain_evidence(
        synced_pit=synced_pit,
        real_column=f"real__{channel_set.ride_height}",
        simu_column=f"simu__{channel_set.ride_height}",
        title=f"{segment_label} {channel_set_key}: track vs sim ride height",
        ylabel=channel_set.ride_height,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_time_rh.png",
    )
    plot_time_domain_evidence(
        synced_pit=synced_pit,
        real_column=f"real__{channel_set.damper}",
        simu_column=f"simu__{channel_set.damper}",
        title=f"{segment_label} {channel_set_key}: track vs sim damper",
        ylabel=channel_set.damper,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_time_damper.png",
    )
    plot_time_domain_evidence(
        synced_pit=synced_pit,
        real_column=f"real__{channel_set.push_load}",
        simu_column=f"simu__{channel_set.push_load}",
        title=f"{segment_label} {channel_set_key}: track vs sim push load",
        ylabel=channel_set.push_load,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_time_push.png",
    )
    plot_time_domain_evidence(
        synced_pit=synced_pit,
        real_column="real__pitot_c",
        simu_column="real__pair",
        title=f"{segment_label} {channel_set_key}: track pitot vs track pAir",
        ylabel="pressure",
        output_path=output_dir / f"{segment_label}_{channel_set_key}_time_pitot_pair.png",
    )
    for column in EXTRA_MAP_COLUMNS:
        real_column = f"real__{column}"
        simu_column = f"simu__{column}"
        if real_column not in synced_pit.columns or simu_column not in synced_pit.columns:
            continue
        plot_time_domain_evidence(
            synced_pit=synced_pit,
            real_column=real_column,
            simu_column=simu_column,
            title=f"{segment_label} {channel_set_key}: track vs sim {column}",
            ylabel=column,
            output_path=output_dir / f"{segment_label}_{channel_set_key}_time_{column}.png",
        )
    plot_threshold_check(
        synced_pit=raw_aligned_pit.rename(columns={"progress": "sync_progress"}),
        real_column=f"real__{channel_set.ride_height}",
        simu_column=f"simu__{channel_set.ride_height}",
        threshold=thresholds.rh_mm,
        title=f"{segment_label} {channel_set_key}: RH threshold check",
        ylabel=channel_set.ride_height,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_check_rh.png",
    )
    plot_threshold_check(
        synced_pit=raw_aligned_pit.rename(columns={"progress": "sync_progress"}),
        real_column=f"real__{channel_set.damper}",
        simu_column=f"simu__{channel_set.damper}",
        threshold=thresholds.damper_mm,
        title=f"{segment_label} {channel_set_key}: damper threshold check",
        ylabel=channel_set.damper,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_check_damper.png",
    )
    plot_threshold_check(
        synced_pit=raw_aligned_pit.rename(columns={"progress": "sync_progress"}),
        real_column=f"real__{channel_set.push_load}",
        simu_column=f"simu__{channel_set.push_load}",
        threshold=thresholds.push_kg,
        title=f"{segment_label} {channel_set_key}: push-load threshold check",
        ylabel=channel_set.push_load,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_check_push.png",
    )
    tpms_real_column = f"real__{channel_set.tyre_pressure}"
    tpms_simu_column = f"simu__{channel_set.tyre_pressure}"
    if tpms_real_column in raw_aligned_pit.columns and tpms_simu_column in raw_aligned_pit.columns:
        plot_threshold_check(
            synced_pit=raw_aligned_pit.rename(columns={"progress": "sync_progress"}),
            real_column=tpms_real_column,
            simu_column=tpms_simu_column,
            threshold=thresholds.tyre_psi,
            title=f"{segment_label} {channel_set_key}: TPMS threshold check",
            ylabel=channel_set.tyre_pressure,
            output_path=output_dir / f"{segment_label}_{channel_set_key}_check_tpms.png",
        )
    plot_xy_evidence(
        df=real_pit,
        x_column=channel_set.damper,
        y_column=channel_set.ride_height,
        fit_result=rh_from_damper_fit,
        title=f"{segment_label} {channel_set_key}: RH = f(Damper)",
        xlabel=channel_set.damper,
        ylabel=channel_set.ride_height,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_xy_rh_from_damper.png",
    )
    plot_xy_evidence(
        df=real_pit,
        x_column=channel_set.push_load,
        y_column=channel_set.ride_height,
        fit_result=rh_from_push_fit,
        title=f"{segment_label} {channel_set_key}: RH = f(Push load)",
        xlabel=channel_set.push_load,
        ylabel=channel_set.ride_height,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_xy_rh_from_push.png",
    )
    plot_xy_evidence(
        df=real_pit,
        x_column=channel_set.damper,
        y_column=channel_set.push_load,
        fit_result=push_from_damper_fit,
        title=f"{segment_label} {channel_set_key}: Push load = f(Damper)",
        xlabel=channel_set.damper,
        ylabel=channel_set.push_load,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_xy_push_from_damper.png",
    )
    plot_xy_evidence(
        df=real_pit,
        x_column="pair",
        y_column="pitot_c",
        fit_result=pitot_from_pair_fit,
        title=f"{segment_label} {channel_set_key}: Pitot = f(pAir)",
        xlabel="pair",
        ylabel="pitot_c",
        output_path=output_dir / f"{segment_label}_{channel_set_key}_xy_pitot_from_pair.png",
    )


def thresholds_for_plot(
    condition_checks: SegmentComparabilityResult,
    channel_name: str,
) -> float | None:
    for result in condition_checks.channel_results:
        if result.channel_name == channel_name:
            return result.threshold
    return None


def build_segment_placeholder_decisions(
    segment_label: str,
    condition_checks: SegmentComparabilityResult,
) -> tuple[DecisionResult, DecisionResult, DecisionResult]:
    detail = (
        f"Shared engine is active for '{segment_label}', but segment-specific decision rules "
        "are not implemented yet. Review the raw and synchronized evidence manually."
    )
    if not condition_checks.comparable:
        detail = condition_checks.detail
    primary = DecisionResult(
        channel="segment",
        action="manual_review",
        summary=f"{segment_label} logic pending",
        detail=detail,
    )
    secondary = DecisionResult(
        channel="segment_secondary",
        action="manual_review",
        summary=f"{segment_label} logic pending",
        detail=detail,
    )
    final = DecisionResult(
        channel="final",
        action="hold_offsets",
        summary=f"{segment_label} decision tree not implemented",
        detail=detail,
    )
    return primary, secondary, final


def parse_named_decision_block(stdout: str, block_name: str) -> dict[str, str]:
    lines = stdout.splitlines()
    target = block_name.strip().lower()

    for index, line in enumerate(lines):
        if line.strip().lower() != target:
            continue

        parsed: dict[str, str] = {}
        cursor = index + 1
        while cursor < len(lines):
            current = lines[cursor]
            stripped = current.strip()
            if not stripped:
                break
            if not current.startswith("  "):
                break
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                parsed[key.strip().lower()] = value.strip()
            cursor += 1
        return parsed

    return {}


def parse_critical_channel_summary(stdout: str, channel_name: str) -> dict[str, str]:
    lines = stdout.splitlines()
    target = channel_name.strip().lower()

    for index, line in enumerate(lines):
        if not line.lower().startswith(f"{target}: "):
            continue

        parsed: dict[str, str] = {"header": line.strip()}
        cursor = index + 1
        while cursor < len(lines):
            current = lines[cursor]
            stripped = current.strip()
            if not stripped:
                break
            if not current.startswith("  "):
                break
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                parsed[key.strip().lower()] = value.strip()
            cursor += 1
        return parsed

    return {}


def describe_band_consistency(
    slow_block: dict[str, str],
    fast_block: dict[str, str],
    *,
    channel_label: str,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    slow_action = slow_block.get("action", "unknown")
    fast_action = fast_block.get("action", "unknown")
    slow_outcome = slow_block.get("outcome", "unknown")
    fast_outcome = fast_block.get("outcome", "unknown")
    slow_basis = slow_block.get("actual decision basis", "unknown")
    fast_basis = fast_block.get("actual decision basis", "unknown")

    if slow_action == fast_action and slow_basis == fast_basis:
        status = "band_consistent"
    elif slow_action == fast_action:
        status = "basis_shift_between_bands"
    else:
        status = "cross_band_conflict"

    notes.append(
        f"{channel_label} slow decision is the official offset verdict: {slow_outcome} ({slow_action})."
    )
    notes.append(
        f"{channel_label} fast-band diagnostic: {fast_outcome} ({fast_action}), decision basis={fast_basis}."
    )

    if status == "band_consistent":
        notes.append(
            f"{channel_label} is consistent across slow and fast bands (basis slow={slow_basis}, fast={fast_basis})."
        )
    elif status == "basis_shift_between_bands":
        notes.append(
            f"{channel_label} keeps the same action across bands, but the internal basis changes (slow={slow_basis}, fast={fast_basis})."
        )
    else:
        notes.append(
            f"{channel_label} disagrees across bands (slow action={slow_action}, fast action={fast_action}). Treat this as a diagnostic inconsistency, not as an automatic override of the slow-band decision."
        )

    return status, notes


def print_cross_band_diagnostic(
    *,
    slow_stdout: str,
    fast_stdout: str,
) -> None:
    slow_rh_decision = parse_named_decision_block(slow_stdout, "ride height")
    slow_push_decision = parse_named_decision_block(slow_stdout, "push load")
    slow_final_decision = parse_named_decision_block(slow_stdout, "final")
    fast_rh_decision = parse_named_decision_block(fast_stdout, "ride height")
    fast_push_decision = parse_named_decision_block(fast_stdout, "push load")

    slow_rh_summary = parse_critical_channel_summary(slow_stdout, "ride height")
    slow_push_summary = parse_critical_channel_summary(slow_stdout, "push load")
    fast_rh_summary = parse_critical_channel_summary(fast_stdout, "ride height")
    fast_push_summary = parse_critical_channel_summary(fast_stdout, "push load")

    rh_status, rh_notes = describe_band_consistency(
        slow_rh_summary | slow_rh_decision,
        fast_rh_summary | fast_rh_decision,
        channel_label="Ride height",
    )
    push_status, push_notes = describe_band_consistency(
        slow_push_summary | slow_push_decision,
        fast_push_summary | fast_push_decision,
        channel_label="Push load",
    )

    statuses = {rh_status, push_status}
    if "cross_band_conflict" in statuses:
        diagnostic_label = "cross_band_conflict"
        diagnostic_reason = (
            "Slow remains the official offset-decision band, but at least one primary channel disagrees between slow and fast bands. "
            "Flag this case as a fault/setup inconsistency candidate for audit and ML labeling."
        )
    elif "basis_shift_between_bands" in statuses:
        diagnostic_label = "basis_shift_between_bands"
        diagnostic_reason = (
            "Slow remains the official offset-decision band. The final actions agree across bands, but at least one channel changes its internal decision basis between slow and fast bands."
        )
    else:
        diagnostic_label = "band_consistent"
        diagnostic_reason = (
            "Slow remains the official offset-decision band, and the fast band is directionally consistent with it."
        )

    print_section("Cross-Band Diagnostic")
    print("primary decision policy")
    print("  slow band is the official offset-decision band; fast band is used only as a diagnostic consistency/error check.")
    print(f"  official slow-band final outcome: {slow_final_decision.get('outcome', 'unavailable')}")
    print(f"  slow-band final action: {slow_final_decision.get('action', 'unavailable')}")
    print()

    print("ride height")
    print(f"  status: {rh_status}")
    for note in rh_notes:
        print(f"  - {note}")
    print()

    print("push load")
    print(f"  status: {push_status}")
    for note in push_notes:
        print(f"  - {note}")
    print()

    print("diagnostic interpretation")
    print(f"  label: {diagnostic_label}")
    print(f"  reason: {diagnostic_reason}")
    print("  suggested ML label use: keep the slow-band offset label as the primary target, and store this cross-band result as a separate fault/consistency target.")
    print()
    print("audit labels")
    print(f"  official offset decision: {slow_final_decision.get('outcome', 'unavailable')}")
    print(f"  diagnostic flag: {diagnostic_label}")


def main() -> None:
    args = parse_args()
    if args.segment_label == "pit" and args.pit_speed_band == "both":
        python = sys.executable
        base_command = [
            python,
            "-m",
            "src.data.pitlane_decision_tree",
            "--channel-set",
            args.channel_set,
            "--segment-label",
            args.segment_label,
            "--min-pit-band-samples",
            str(args.min_pit_band_samples),
            "--rh-threshold-mm",
            str(args.rh_threshold_mm),
            "--damper-threshold-mm",
            str(args.damper_threshold_mm),
            "--push-threshold-kg",
            str(args.push_threshold_kg),
            "--pitot-relative-threshold",
            str(args.pitot_relative_threshold),
            "--tyre-threshold-psi",
            str(args.tyre_threshold_psi),
            "--rh-prediction-threshold-mm",
            str(args.rh_prediction_threshold_mm),
            "--push-prediction-threshold-kg",
            str(args.push_prediction_threshold_kg),
            "--pitot-prediction-relative-threshold",
            str(args.pitot_prediction_relative_threshold),
            "--sync-grid-size",
            str(args.sync_grid_size),
            "--condition-speed-threshold-kmh",
            str(args.condition_speed_threshold_kmh),
            "--condition-accx-threshold",
            str(args.condition_accx_threshold),
            "--condition-accy-threshold",
            str(args.condition_accy_threshold),
        ]
        if args.plot:
            base_command.append("--plot")
        if args.output_dir != Path("data/processed/plots/segment_decision_tree"):
            base_command.extend(["--output-dir", str(args.output_dir)])

        band_outputs: dict[str, str] = {}
        for band in ("slow", "fast"):
            print_section(f"Pit Band: {band}")
            command = [*base_command, "--pit-speed-band", band]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            band_outputs[band] = result.stdout
            print(result.stdout.rstrip())
            if result.stderr.strip():
                print("\n[stderr]")
                print(result.stderr.rstrip())
            print()
        if "slow" in band_outputs and "fast" in band_outputs:
            print_cross_band_diagnostic(
                slow_stdout=band_outputs["slow"],
                fast_stdout=band_outputs["fast"],
            )
        return

    thresholds = ThresholdConfig(
        rh_mm=args.rh_threshold_mm,
        damper_mm=args.damper_threshold_mm,
        push_kg=args.push_threshold_kg,
        pitot_relative=args.pitot_relative_threshold,
        tyre_psi=args.tyre_threshold_psi,
        rh_prediction_mm=args.rh_prediction_threshold_mm,
        push_prediction_kg=args.push_prediction_threshold_kg,
        pitot_prediction_relative=args.pitot_prediction_relative_threshold,
    )
    channel_set = CHANNEL_SETS[args.channel_set]
    segment_label = args.segment_label

    sync_columns = sorted(
        {
            channel_set.ride_height,
            channel_set.push_load,
            channel_set.damper,
            channel_set.tyre_pressure,
            "carspeed_art",
            "avg_accx",
            "avg_accy",
            "pitot_c",
            "pair",
            *EXTRA_MAP_COLUMNS,
        }
    )
    sync_reference_columns = ["carspeed_art", "pair", "pitot_c", channel_set.damper]

    real_df = load_segmented_data("real")
    simu_df = load_segmented_data("simu")
    if segment_label == "pit":
        real_pit, simu_pit = select_matched_pit_segments_for_analysis(
            real_df=real_df,
            simu_df=simu_df,
            pit_speed_band=args.pit_speed_band,
            min_pit_band_samples=args.min_pit_band_samples,
            quality_column=channel_set.ride_height,
        )
    else:
        real_pit = get_first_segment(real_df, "real", segment_label)
        simu_pit = get_first_segment(simu_df, "simu", segment_label)
    raw_aligned_pit = build_raw_progress_aligned_df(
        real_df=real_pit,
        simu_df=simu_pit,
        columns=sync_columns,
        grid_size=args.sync_grid_size,
    )

    rh_track = get_optional_median(real_pit, channel_set.ride_height)
    rh_sim = get_optional_median(simu_pit, channel_set.ride_height)
    push_track = get_optional_median(real_pit, channel_set.push_load)
    push_sim = get_optional_median(simu_pit, channel_set.push_load)
    damper_track = get_optional_median(real_pit, channel_set.damper)
    damper_sim = get_optional_median(simu_pit, channel_set.damper)
    tyre_track = get_optional_median(real_pit, channel_set.tyre_pressure)
    tyre_sim = get_optional_median(simu_pit, channel_set.tyre_pressure)
    pitot_track = get_optional_median(real_pit, "pitot_c")
    pair_track = get_optional_median(real_pit, "pair")

    condition_checks = summarize_segment_comparability(
        [
            evaluate_condition_comparability(
                real_series=raw_aligned_pit["real__carspeed_art"],
                simu_series=raw_aligned_pit["simu__carspeed_art"],
                threshold=args.condition_speed_threshold_kmh,
                channel_name="carspeed_art",
            ),
            evaluate_condition_comparability(
                real_series=raw_aligned_pit["real__avg_accx"],
                simu_series=raw_aligned_pit["simu__avg_accx"],
                threshold=args.condition_accx_threshold,
                channel_name="avg_accx",
            ),
            evaluate_condition_comparability(
                real_series=raw_aligned_pit["real__avg_accy"],
                simu_series=raw_aligned_pit["simu__avg_accy"],
                threshold=args.condition_accy_threshold,
                channel_name="avg_accy",
            ),
        ]
    )

    rh_cmp = compare_track_vs_sim_higher_is_signal(
        real_series=raw_aligned_pit[f"real__{channel_set.ride_height}"],
        simu_series=raw_aligned_pit[f"simu__{channel_set.ride_height}"],
        threshold=thresholds.rh_mm,
        channel_name=channel_set.ride_height,
    )
    push_cmp = compare_track_vs_sim_lower_is_signal(
        real_series=raw_aligned_pit[f"real__{channel_set.push_load}"],
        simu_series=raw_aligned_pit[f"simu__{channel_set.push_load}"],
        threshold=thresholds.push_kg,
        channel_name=channel_set.push_load,
    )
    damper_cmp = compare_track_vs_sim_lower_is_signal(
        real_series=raw_aligned_pit[f"real__{channel_set.damper}"],
        simu_series=raw_aligned_pit[f"simu__{channel_set.damper}"],
        threshold=thresholds.damper_mm,
        channel_name=channel_set.damper,
    )
    real_tyre_column = f"real__{channel_set.tyre_pressure}"
    simu_tyre_column = f"simu__{channel_set.tyre_pressure}"
    if real_tyre_column in raw_aligned_pit.columns and simu_tyre_column in raw_aligned_pit.columns:
        tyre_cmp = compare_track_vs_sim_higher_is_signal(
            real_series=raw_aligned_pit[real_tyre_column],
            simu_series=raw_aligned_pit[simu_tyre_column],
            threshold=thresholds.tyre_psi,
            channel_name=channel_set.tyre_pressure,
        )
    else:
        missing_parts: list[str] = []
        if real_tyre_column not in raw_aligned_pit.columns:
            missing_parts.append("real channel missing")
        if simu_tyre_column not in raw_aligned_pit.columns:
            missing_parts.append("sim channel missing")
        tyre_cmp = unavailable_comparison_result(
            channel_name=channel_set.tyre_pressure,
            threshold=thresholds.tyre_psi,
            reason=", ".join(missing_parts),
        )
    pitot_cmp = compare_pitot_vs_pair(
        pitot_series=raw_aligned_pit["real__pitot_c"],
        pair_series=raw_aligned_pit["real__pair"],
        relative_threshold=thresholds.pitot_relative,
    )

    rh_from_damper_fit = fit_linear_relationship(
        x=pd.to_numeric(real_pit[channel_set.damper], errors="coerce"),
        y=pd.to_numeric(real_pit[channel_set.ride_height], errors="coerce"),
        input_median=damper_track,
        observed_median=rh_track,
        tolerance=thresholds.rh_prediction_mm,
        label="RH = f(Damper)",
    )
    rh_from_push_fit = fit_linear_relationship(
        x=pd.to_numeric(real_pit[channel_set.push_load], errors="coerce"),
        y=pd.to_numeric(real_pit[channel_set.ride_height], errors="coerce"),
        input_median=push_track,
        observed_median=rh_track,
        tolerance=thresholds.rh_prediction_mm,
        label="RH = f(Push load)",
    )
    push_from_damper_fit = fit_linear_relationship(
        x=pd.to_numeric(real_pit[channel_set.damper], errors="coerce"),
        y=pd.to_numeric(real_pit[channel_set.push_load], errors="coerce"),
        input_median=damper_track,
        observed_median=push_track,
        tolerance=thresholds.push_prediction_kg,
        label="Push load = f(Damper)",
    )
    pitot_from_pair_fit = fit_linear_relationship(
        x=pd.to_numeric(real_pit["pair"], errors="coerce"),
        y=pd.to_numeric(real_pit["pitot_c"], errors="coerce"),
        input_median=pair_track,
        observed_median=pitot_track,
        tolerance=(
            max(abs(pitot_track), 1e-9) * thresholds.pitot_prediction_relative
            if pitot_track is not None
            else thresholds.pitot_prediction_relative
        ),
        label="Pitot = f(pAir)",
    )

    rh_time_series = compute_time_series_agreement_evidence(
        real_series=raw_aligned_pit[f"real__{channel_set.ride_height}"],
        simu_series=raw_aligned_pit[f"simu__{channel_set.ride_height}"],
    )
    push_time_series = compute_time_series_agreement_evidence(
        real_series=raw_aligned_pit[f"real__{channel_set.push_load}"],
        simu_series=raw_aligned_pit[f"simu__{channel_set.push_load}"],
    )
    damper_time_series = compute_time_series_agreement_evidence(
        real_series=raw_aligned_pit[f"real__{channel_set.damper}"],
        simu_series=raw_aligned_pit[f"simu__{channel_set.damper}"],
    )
    pitot_time_series = compute_time_series_agreement_evidence(
        real_series=raw_aligned_pit["real__pitot_c"],
        simu_series=raw_aligned_pit["real__pair"],
    )
    plateau = detect_stable_plateau(
        raw_aligned_pit.rename(columns={"progress": "sync_progress"}),
        channel_set=channel_set,
    )
    plateau_df = slice_plateau_df(
        raw_aligned_pit.rename(columns={"progress": "sync_progress"}),
        plateau,
    )
    plateau_rh_cmp = None
    plateau_push_cmp = None
    plateau_damper_cmp = None
    plateau_pitot_cmp = None
    if plateau_df is not None and not plateau_df.empty:
        plateau_rh_cmp = compare_track_vs_sim_higher_is_signal(
            real_series=plateau_df[f"real__{channel_set.ride_height}"],
            simu_series=plateau_df[f"simu__{channel_set.ride_height}"],
            threshold=thresholds.rh_mm,
            channel_name=f"{channel_set.ride_height} plateau",
        )
        plateau_push_cmp = compare_track_vs_sim_lower_is_signal(
            real_series=plateau_df[f"real__{channel_set.push_load}"],
            simu_series=plateau_df[f"simu__{channel_set.push_load}"],
            threshold=thresholds.push_kg,
            channel_name=f"{channel_set.push_load} plateau",
        )
        plateau_damper_cmp = compare_track_vs_sim_lower_is_signal(
            real_series=plateau_df[f"real__{channel_set.damper}"],
            simu_series=plateau_df[f"simu__{channel_set.damper}"],
            threshold=thresholds.damper_mm,
            channel_name=f"{channel_set.damper} plateau",
        )
        plateau_pitot_cmp = compare_pitot_vs_pair(
            pitot_series=plateau_df["real__pitot_c"],
            pair_series=plateau_df["real__pair"],
            relative_threshold=thresholds.pitot_relative,
        )

    rh_helper_agreement = HelperAgreementSummary(0, 0, 0, 0, "not applicable")
    push_helper_agreement = HelperAgreementSummary(0, 0, 0, 0, "not applicable")
    rh_confidence = "low"
    push_confidence = "low"
    conflict_checks: list[str] = []
    blockers: list[str] = []

    full_run_sync = None
    sync_result = None
    synced_pit = None
    if condition_checks.comparable and segment_label == "pit":
        full_run_sync = synchronize_dataframes(
            real_df=real_df,
            simu_df=simu_df,
            columns=sync_columns,
            grid_size=args.sync_grid_size,
            reference_columns=sync_reference_columns,
        )
        sync_result = synchronize_dataframes(
            real_df=real_pit,
            simu_df=simu_pit,
            columns=sync_columns,
            grid_size=args.sync_grid_size,
            reference_columns=sync_reference_columns,
        )
        synced_pit = sync_result.synced_df
        rh_gate_cmp = select_gate_comparison(rh_cmp, plateau_rh_cmp)
        push_gate_cmp = select_gate_comparison(push_cmp, plateau_push_cmp)
        damper_gate_cmp = select_gate_comparison(damper_cmp, plateau_damper_cmp)
        pitot_gate_cmp = select_gate_comparison(pitot_cmp, plateau_pitot_cmp)
        rh_decision = decide_ride_height_offset(
            rh_cmp=rh_gate_cmp,
            damper_cmp=damper_gate_cmp,
            push_cmp=push_gate_cmp,
            pitot_cmp=pitot_gate_cmp,
            tyre_cmp=tyre_cmp,
            rh_from_damper_fit=rh_from_damper_fit,
            rh_from_push_fit=rh_from_push_fit,
        )
        push_decision = decide_push_load_offset(
            push_cmp=push_gate_cmp,
            damper_cmp=damper_gate_cmp,
            rh_cmp=rh_gate_cmp,
            pitot_cmp=pitot_gate_cmp,
            rh_from_push_fit=rh_from_push_fit,
            push_from_damper_fit=push_from_damper_fit,
        )
        rh_helper_agreement = build_helper_agreement_summary(
            target_channel="ride_height",
            rh_cmp=rh_gate_cmp,
            damper_cmp=damper_gate_cmp,
            push_cmp=push_gate_cmp,
            pitot_cmp=pitot_gate_cmp,
            tyre_cmp=tyre_cmp,
        )
        push_helper_agreement = build_helper_agreement_summary(
            target_channel="push_load",
            rh_cmp=rh_gate_cmp,
            damper_cmp=damper_gate_cmp,
            push_cmp=push_gate_cmp,
            pitot_cmp=pitot_gate_cmp,
        )
        rh_decision = resolve_unsettled_decision_with_evidence_layers(
            decision=rh_decision,
            channel="ride_height",
            comparison=rh_gate_cmp,
            plateau_comparison=plateau_rh_cmp,
            helper_agreement=rh_helper_agreement,
            fit_results=[rh_from_damper_fit, rh_from_push_fit, pitot_from_pair_fit],
            time_series=rh_time_series,
        )
        push_decision = resolve_unsettled_decision_with_evidence_layers(
            decision=push_decision,
            channel="push_load",
            comparison=push_gate_cmp,
            plateau_comparison=plateau_push_cmp,
            helper_agreement=push_helper_agreement,
            fit_results=[push_from_damper_fit, rh_from_push_fit, pitot_from_pair_fit],
            time_series=push_time_series,
        )
        blockers = build_offset_blockers(
            channel_set=channel_set,
            rh_cmp=rh_gate_cmp,
            damper_cmp=damper_gate_cmp,
            push_cmp=push_gate_cmp,
            pitot_cmp=pitot_gate_cmp,
            tyre_cmp=tyre_cmp,
        )
        final_decision = finalize_actions(
            rh_decision=rh_decision,
            push_decision=push_decision,
            blockers=blockers,
        )
        rh_confidence = decision_confidence_label(
            decision=rh_decision,
            comparison=rh_gate_cmp,
            plateau_comparison=plateau_rh_cmp,
            helper_agreement=rh_helper_agreement,
            fit_results=[rh_from_damper_fit, rh_from_push_fit, pitot_from_pair_fit],
            time_series=rh_time_series,
        )
        push_confidence = decision_confidence_label(
            decision=push_decision,
            comparison=push_gate_cmp,
            plateau_comparison=plateau_push_cmp,
            helper_agreement=push_helper_agreement,
            fit_results=[push_from_damper_fit, rh_from_push_fit, pitot_from_pair_fit],
            time_series=push_time_series,
        )
        conflict_checks = build_conflict_checks(
            rh_decision=rh_decision,
            push_decision=push_decision,
            rh_cmp=rh_gate_cmp,
            push_cmp=push_gate_cmp,
        )
    elif segment_label != "pit":
        full_run_sync = synchronize_dataframes(
            real_df=real_df,
            simu_df=simu_df,
            columns=sync_columns,
            grid_size=args.sync_grid_size,
            reference_columns=sync_reference_columns,
        ) if condition_checks.comparable else None
        sync_result = synchronize_dataframes(
            real_df=real_pit,
            simu_df=simu_pit,
            columns=sync_columns,
            grid_size=args.sync_grid_size,
            reference_columns=sync_reference_columns,
        ) if condition_checks.comparable else None
        synced_pit = sync_result.synced_df if sync_result is not None else None
        rh_decision, push_decision, final_decision = build_segment_placeholder_decisions(
            segment_label=segment_label,
            condition_checks=condition_checks,
        )
    else:
        rh_decision = DecisionResult(
            channel="ride_height",
            action="manual_review",
            summary="Condition mismatch",
            detail=condition_checks.detail,
        )
        push_decision = DecisionResult(
            channel="push_load",
            action="manual_review",
            summary="Condition mismatch",
            detail=condition_checks.detail,
        )
        final_decision = DecisionResult(
            channel="final",
            action="hold_offsets",
            summary="Condition pre-check failed",
            detail=condition_checks.detail,
        )

    if args.plot:
        generate_evidence_plots(
            output_dir=args.output_dir,
            segment_label=segment_label,
            channel_set_key=args.channel_set,
            channel_set=channel_set,
            real_df=real_df,
            simu_df=simu_df,
            raw_aligned_pit=raw_aligned_pit,
            synced_pit=(
                synced_pit
                if synced_pit is not None
                else raw_aligned_pit.rename(columns={"progress": "sync_progress"})
            ),
            real_pit=real_pit,
            simu_pit=simu_pit,
            thresholds=thresholds,
            rh_from_damper_fit=rh_from_damper_fit,
            rh_from_push_fit=rh_from_push_fit,
            push_from_damper_fit=push_from_damper_fit,
            pitot_from_pair_fit=pitot_from_pair_fit,
            condition_checks=condition_checks,
        )

    print_section("Segment Validation")
    print(f"Segment label: {segment_label}")
    print(f"Channel set: {args.channel_set} ({channel_set.name})")
    if segment_label == "pit":
        print("Selected pit groups:")
        print_selected_segment_info(real_pit)
        print_selected_segment_info(simu_pit)

    print_section("Synchronization")
    if full_run_sync is None or sync_result is None:
        print("Synchronization skipped because the raw pitlane comparability gate did not pass.")
    else:
        print(
            "Full run coarse sync: "
            f"method={full_run_sync.method}, "
            f"axis={full_run_sync.axis_method}, "
            f"refs={list(full_run_sync.reference_columns_used)}, "
            f"lag_steps={full_run_sync.lag_steps}, "
            f"lag_progress={full_run_sync.lag_progress:.4f}"
        )
        print(
            "Pit local fine sync: "
            f"method={sync_result.method}, "
            f"axis={sync_result.axis_method}, "
            f"refs={list(sync_result.reference_columns_used)}, "
            f"lag_steps={sync_result.lag_steps}, "
            f"lag_progress={sync_result.lag_progress:.4f}, "
            f"grid_size={sync_result.grid_size}, "
            f"real_points_used={sync_result.real_points_used}, "
            f"sim_points_used={sync_result.simu_points_used}"
        )

    print_section("Median Comparisons")
    for result in [rh_cmp, damper_cmp, push_cmp, pitot_cmp, tyre_cmp]:
        print(f"{result.state}: {result.detail}")

    print_section("Plateau Detection")
    print(plateau.detail)
    if plateau_rh_cmp is not None:
        print(f"ride_height plateau: {plateau_rh_cmp.state}: {plateau_rh_cmp.detail}")
    if plateau_push_cmp is not None:
        print(f"push_load plateau: {plateau_push_cmp.state}: {plateau_push_cmp.detail}")
    if plateau_damper_cmp is not None:
        print(f"damper plateau: {plateau_damper_cmp.state}: {plateau_damper_cmp.detail}")
    if plateau_pitot_cmp is not None:
        print(f"pitot plateau: {plateau_pitot_cmp.state}: {plateau_pitot_cmp.detail}")

    print_section("Condition Comparability")
    for result in condition_checks.channel_results:
        print(
            f"{result.channel_name} [{result.state}]: "
            f"median {format_metric_value(result.median_real)} - "
            f"{format_metric_value(result.median_simu)} = "
            f"{format_metric_value(result.median_delta)}, "
            f"mae={format_metric_value(result.mae)}, "
            f"within={format_metric_value(None if result.within_fraction is None else 100.0 * result.within_fraction, '%')}, "
            f"overlap={format_metric_value(None if result.overlap_ratio is None else 100.0 * result.overlap_ratio, '%')}, "
            f"pearson_r={format_metric_value(result.pearson_r)}"
        )
    print(condition_checks.detail)

    print_section("Raw Progress-Aligned Primary Metrics")
    print_signal_metrics(
        raw_aligned_pit,
        "ride_height",
        f"real__{channel_set.ride_height}",
        f"simu__{channel_set.ride_height}",
    )
    print_signal_metrics(
        raw_aligned_pit,
        "push_load",
        f"real__{channel_set.push_load}",
        f"simu__{channel_set.push_load}",
    )
    print_signal_metrics(
        raw_aligned_pit,
        "damper",
        f"real__{channel_set.damper}",
        f"simu__{channel_set.damper}",
    )
    print_signal_metrics(
        raw_aligned_pit,
        "pitot",
        "real__pitot_c",
        "real__pair",
    )
    print_signal_metrics(
        raw_aligned_pit,
        "pair",
        "real__pair",
        "simu__pair",
    )
    raw_tpms_real_column = f"real__{channel_set.tyre_pressure}"
    raw_tpms_simu_column = f"simu__{channel_set.tyre_pressure}"
    if raw_tpms_real_column in raw_aligned_pit.columns and raw_tpms_simu_column in raw_aligned_pit.columns:
        print_signal_metrics(
            raw_aligned_pit,
            "tpms",
            raw_tpms_real_column,
            raw_tpms_simu_column,
        )
    else:
        print("tpms: unavailable")
    print_section("Additional Map Signal Metrics")
    for column in EXTRA_MAP_COLUMNS:
        real_column = f"real__{column}"
        simu_column = f"simu__{column}"
        if real_column not in raw_aligned_pit.columns or simu_column not in raw_aligned_pit.columns:
            print(f"{column}: unavailable")
            continue
        print_signal_metrics(
            raw_aligned_pit,
            column,
            real_column,
            simu_column,
        )

    if synced_pit is not None:
        print_section("Synchronized Secondary Metrics")
        print_signal_metrics(
            synced_pit,
            "ride_height",
            f"real__{channel_set.ride_height}",
            f"simu__{channel_set.ride_height}",
        )
        print_signal_metrics(
            synced_pit,
            "push_load",
            f"real__{channel_set.push_load}",
            f"simu__{channel_set.push_load}",
        )
        print_signal_metrics(
            synced_pit,
            "damper",
            f"real__{channel_set.damper}",
            f"simu__{channel_set.damper}",
        )

    print_section("Tyre Pressure Detail")
    if tyre_cmp.difference is None:
        print(
            "TPMS delta unavailable: "
            f"track={format_optional_value(tyre_track)}, "
            f"sim={format_optional_value(tyre_sim)}"
        )
    else:
        print(
            f"track TPMS - sim TPMS = {tyre_cmp.difference:.4f} psi "
            f"(threshold {thresholds.tyre_psi:.4f} psi)"
        )
        if tyre_cmp.difference > thresholds.tyre_psi:
            print(
                "TPMS delta exceeds threshold: this points to a different issue "
                "rather than a ride-height offset candidate."
            )

    print_section("Evidence Fits")
    for fit in [
        rh_from_damper_fit,
        rh_from_push_fit,
        push_from_damper_fit,
        pitot_from_pair_fit,
    ]:
        print(f"{fit.status}: {fit.detail}")

    print_section("Fit Quality")
    for label, fit in [
        ("RH = f(Damper)", rh_from_damper_fit),
        ("RH = f(Push load)", rh_from_push_fit),
        ("Push load = f(Damper)", push_from_damper_fit),
        ("Pitot = f(pAir)", pitot_from_pair_fit),
    ]:
        quality_detail = fit.quality.detail if fit.quality is not None else "fit quality unavailable"
        print(f"{label}: {quality_detail}")

    print_section("Time-Series Agreement")
    for label, evidence in [
        ("ride_height", rh_time_series),
        ("push_load", push_time_series),
        ("damper", damper_time_series),
        ("pitot_vs_pair", pitot_time_series),
    ]:
        print(f"{label}: {evidence.detail}")

    print_decision_overview(
        rh_decision=rh_decision,
        push_decision=push_decision,
        final_decision=final_decision,
    )

    print_section("Helper Agreement")
    print(f"ride_height: {rh_helper_agreement.detail}")
    print(
        f"  supports={rh_helper_agreement.support_count}, "
        f"contradictions={rh_helper_agreement.contradiction_count}, "
        f"neutral={rh_helper_agreement.neutral_count}, "
        f"total={rh_helper_agreement.total_count}"
    )
    print(f"push_load: {push_helper_agreement.detail}")
    print(
        f"  supports={push_helper_agreement.support_count}, "
        f"contradictions={push_helper_agreement.contradiction_count}, "
        f"neutral={push_helper_agreement.neutral_count}, "
        f"total={push_helper_agreement.total_count}"
    )

    print_section("Decision Confidence")
    print(f"ride_height: {rh_confidence}")
    print(f"push_load: {push_confidence}")

    print_section("RH / Push Conflict Check")
    if conflict_checks:
        for item in conflict_checks:
            print(f"- {item}")
    else:
        print("No RH/push consistency conflicts detected.")

    print_critical_decision_summary(
        rh_cmp=rh_cmp,
        push_cmp=push_cmp,
        rh_decision=rh_decision,
        push_decision=push_decision,
        blockers=blockers,
        plateau_rh_cmp=plateau_rh_cmp,
        plateau_push_cmp=plateau_push_cmp,
        rh_helper_agreement=rh_helper_agreement,
        push_helper_agreement=push_helper_agreement,
        rh_confidence=rh_confidence,
        push_confidence=push_confidence,
    )


if __name__ == "__main__":
    main()
