import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.synchronization import synchronize_dataframes
from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file


REFERENCE_COLUMNS = ["carspeed_art", "pair", "pitot_c"]


@dataclass(frozen=True)
class SegmentDescriptor:
    segment_number: int
    sample_count: int
    distance_span: float | None
    speed_median: float | None
    pair_median: float | None
    pitot_median: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment-label",
        choices=["pit", "straight", "corner", "all"],
        default="all",
        help="Segment type to preview.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many candidate sim matches to show for each real segment.",
    )
    parser.add_argument(
        "--sync-grid-size",
        type=int,
        default=200,
        help="Grid size used for the preview correlation score.",
    )
    return parser.parse_args()


def load_segmented_data(source: str) -> pd.DataFrame:
    if source == "simu":
        return pd.read_csv(segmented_simudata_run_file(), low_memory=False)
    return pd.read_csv(segmented_rh_run_file(), low_memory=False)


def get_segments(df: pd.DataFrame, label: str) -> list[pd.DataFrame]:
    segment_df = df[df["segment_final"].astype("string") == label].copy()
    if segment_df.empty:
        return []

    segment_df["group"] = (segment_df.index.to_series().diff() != 1).cumsum()
    return [group_df.drop(columns=["group"]).copy() for _, group_df in segment_df.groupby("group")]


def optional_median(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    value = pd.to_numeric(df[column], errors="coerce").median(skipna=True)
    if pd.isna(value):
        return None
    return float(value)


def optional_distance_span(df: pd.DataFrame) -> float | None:
    for column in ["distancelap", "distance"]:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(values) < 2:
            continue
        return float(values.max() - values.min())
    return None


def describe_segment(segment_df: pd.DataFrame, segment_number: int) -> SegmentDescriptor:
    return SegmentDescriptor(
        segment_number=segment_number,
        sample_count=len(segment_df),
        distance_span=optional_distance_span(segment_df),
        speed_median=optional_median(segment_df, "carspeed_art"),
        pair_median=optional_median(segment_df, "pair"),
        pitot_median=optional_median(segment_df, "pitot_c"),
    )


def normalized_abs_delta(a: float | None, b: float | None, scale_floor: float = 1.0) -> float:
    if a is None or b is None:
        return 1.0
    scale = max(abs(a), abs(b), scale_floor)
    return abs(a - b) / scale


def correlation_preview_score(
    real_segment: pd.DataFrame,
    simu_segment: pd.DataFrame,
    grid_size: int,
) -> float:
    available_columns = [
        column
        for column in REFERENCE_COLUMNS
        if column in real_segment.columns and column in simu_segment.columns
    ]
    if not available_columns:
        return -1.0

    sync_result = synchronize_dataframes(
        real_df=real_segment,
        simu_df=simu_segment,
        columns=available_columns,
        grid_size=grid_size,
        reference_columns=available_columns,
    )

    correlations: list[float] = []
    for column in available_columns:
        aligned = pd.DataFrame(
            {
                "real": pd.to_numeric(sync_result.synced_df[f"real__{column}"], errors="coerce"),
                "simu": pd.to_numeric(sync_result.synced_df[f"simu__{column}"], errors="coerce"),
            }
        ).dropna()
        if len(aligned) < 3:
            continue
        real_values = aligned["real"].to_numpy(dtype="float64")
        simu_values = aligned["simu"].to_numpy(dtype="float64")
        if np.std(real_values) == 0 or np.std(simu_values) == 0:
            continue
        correlations.append(float(np.corrcoef(real_values, simu_values)[0, 1]))

    if not correlations:
        return -1.0
    return float(np.mean(correlations))


def candidate_score(
    real_desc: SegmentDescriptor,
    simu_desc: SegmentDescriptor,
    correlation_score: float,
) -> float:
    speed_penalty = normalized_abs_delta(real_desc.speed_median, simu_desc.speed_median, 5.0)
    span_penalty = normalized_abs_delta(real_desc.distance_span, simu_desc.distance_span, 5.0)
    pair_penalty = normalized_abs_delta(real_desc.pair_median, simu_desc.pair_median, 5.0)
    sample_penalty = normalized_abs_delta(
        float(real_desc.sample_count),
        float(simu_desc.sample_count),
        20.0,
    )
    correlation_penalty = 1.0 - max(min(correlation_score, 1.0), -1.0)
    return (
        0.35 * speed_penalty
        + 0.20 * span_penalty
        + 0.15 * pair_penalty
        + 0.10 * sample_penalty
        + 0.20 * correlation_penalty
    )


def preview_label(real_desc: SegmentDescriptor, simu_desc: SegmentDescriptor, corr: float, score: float) -> str:
    return (
        f"sim #{simu_desc.segment_number} | score={score:.4f} | corr={corr:.4f} | "
        f"speed {real_desc.speed_median:.2f}/{simu_desc.speed_median:.2f} | "
        f"span {real_desc.distance_span:.2f}/{simu_desc.distance_span:.2f} | "
        f"samples {real_desc.sample_count}/{simu_desc.sample_count}"
    )


def run_preview_for_label(
    label: str,
    real_df: pd.DataFrame,
    simu_df: pd.DataFrame,
    top_k: int,
    sync_grid_size: int,
) -> None:
    real_segments = get_segments(real_df, label)
    simu_segments = get_segments(simu_df, label)

    print(f"\n{label.upper()} matching preview")
    print(f"Real segments: {len(real_segments)}, Sim segments: {len(simu_segments)}")

    if not real_segments or not simu_segments:
        print("No segments available for preview.")
        return

    simu_descriptors = [
        describe_segment(segment_df, idx + 1)
        for idx, segment_df in enumerate(simu_segments)
    ]

    for real_idx, real_segment in enumerate(real_segments, start=1):
        real_desc = describe_segment(real_segment, real_idx)
        candidates = []

        for simu_idx, simu_segment in enumerate(simu_segments, start=1):
            simu_desc = simu_descriptors[simu_idx - 1]
            corr = correlation_preview_score(real_segment, simu_segment, sync_grid_size)
            score = candidate_score(real_desc, simu_desc, corr)
            candidates.append((score, corr, simu_desc))

        candidates.sort(key=lambda item: item[0])
        print(
            f"\nReal #{real_desc.segment_number} | "
            f"speed={real_desc.speed_median:.2f} | "
            f"span={real_desc.distance_span:.2f} | "
            f"samples={real_desc.sample_count}"
        )
        for score, corr, simu_desc in candidates[:top_k]:
            print("  " + preview_label(real_desc, simu_desc, corr, score))


def main() -> None:
    args = parse_args()
    real_df = load_segmented_data("real")
    simu_df = load_segmented_data("simu")

    labels = ["pit", "straight", "corner"] if args.segment_label == "all" else [args.segment_label]
    for label in labels:
        run_preview_for_label(
            label=label,
            real_df=real_df,
            simu_df=simu_df,
            top_k=args.top_k,
            sync_grid_size=args.sync_grid_size,
        )


if __name__ == "__main__":
    main()
