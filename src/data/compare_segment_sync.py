import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.comparison_metrics import (
    compute_signal_comparison_metrics,
    evaluate_absolute_threshold_comparison,
    evaluate_relative_threshold_comparison,
    format_metric_value,
)
from src.data.synchronization import synchronize_dataframes
from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file


@dataclass(frozen=True)
class ChannelSet:
    name: str
    ride_height: str
    push_load: str
    damper: str
    tyre_pressure: str


CHANNEL_SETS = {
    "fl": ChannelSet(
        name="front-left",
        ride_height="rh_f",
        push_load="pushavg_c",
        damper="damper_fl_art",
        tyre_pressure="tpms_p_fl",
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
        tyre_pressure="tpms_p_rr",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment-label",
        choices=["pit", "straight", "corner", "all"],
        required=True,
        help="Segment type to compare, or 'all' for pit/straight/corner separately.",
    )
    parser.add_argument(
        "--channel-set",
        choices=sorted(CHANNEL_SETS),
        default="fr",
        help="Corner / channel set to compare.",
    )
    parser.add_argument(
        "--real-segment-number",
        type=int,
        default=1,
        help="1-based segment number from the real dataset.",
    )
    parser.add_argument(
        "--simu-segment-number",
        type=int,
        default=1,
        help="1-based segment number from the simulation dataset.",
    )
    parser.add_argument(
        "--sync-grid-size",
        type=int,
        default=200,
        help="Number of synchronized comparison points.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show and save synchronized time-domain plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/plots/segment_sync"),
        help="Directory where comparison plots will be saved.",
    )
    parser.add_argument(
        "--loop-all-segments",
        action="store_true",
        help="Compare all matching segment numbers for the chosen segment label(s).",
    )
    parser.add_argument(
        "--save-summary-csv",
        action="store_true",
        help="Save a CSV summary of the batch segment comparisons.",
    )
    return parser.parse_args()


def load_segmented_data(source: str) -> pd.DataFrame:
    if source == "simu":
        return pd.read_csv(segmented_simudata_run_file(), low_memory=False)
    return pd.read_csv(segmented_rh_run_file(), low_memory=False)


def get_segments(df: pd.DataFrame, label: str) -> list[pd.DataFrame]:
    if "segment_final" not in df.columns:
        raise ValueError("Missing 'segment_final'. Run segmentation first.")

    segment_df = df[df["segment_final"].astype("string") == label].copy()
    if segment_df.empty:
        raise ValueError(f"No '{label}' rows found in the segmented dataset.")

    segment_df["group"] = (segment_df.index.to_series().diff() != 1).cumsum()
    segments: list[pd.DataFrame] = []

    for _, group_df in segment_df.groupby("group"):
        segments.append(group_df.drop(columns=["group"]).copy())

    return segments


def get_segment_by_number(segments: list[pd.DataFrame], number: int, source: str) -> pd.DataFrame:
    if number < 1 or number > len(segments):
        raise ValueError(
            f"{source} segment number {number} is out of range. "
            f"Available: 1..{len(segments)}"
        )
    return segments[number - 1].copy()


def get_synced_median(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None

    value = pd.to_numeric(df[column], errors="coerce").median(skipna=True)
    if pd.isna(value):
        return None
    return float(value)


def format_value(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.4f}"


def print_metric(label: str, real_value: float | None, simu_value: float | None) -> None:
    if real_value is None or simu_value is None:
        print(f"{label}: track={format_value(real_value)}, sim={format_value(simu_value)}")
        return

    delta = real_value - simu_value
    print(
        f"{label}: track={real_value:.4f}, sim={simu_value:.4f}, delta={delta:.4f}"
    )


def print_comparison_metrics(
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


def print_threshold_verdict(
    label: str,
    verdict_label: str,
    detail: str,
) -> None:
    print(f"{label}: {verdict_label} | {detail}")


def safe_metric_value(value: float | None) -> float:
    if value is None:
        return float("nan")
    return float(value)


def plot_synced_overlay(
    synced_df: pd.DataFrame,
    real_column: str,
    simu_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = pd.to_numeric(synced_df["sync_progress"], errors="coerce")
    real_y = pd.to_numeric(synced_df[real_column], errors="coerce")
    simu_y = pd.to_numeric(synced_df[simu_column], errors="coerce")

    plt.figure(figsize=(12, 5))
    if real_y.notna().any():
        plt.plot(x[real_y.notna()], real_y[real_y.notna()], label=f"track {real_column}")
    if simu_y.notna().any():
        plt.plot(x[simu_y.notna()], simu_y[simu_y.notna()], label=f"sim {simu_column}")

    plt.title(title)
    plt.xlabel("Synchronized segment progress")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.show()


def generate_plots(
    synced_df: pd.DataFrame,
    segment_label: str,
    channel_set_key: str,
    channel_set: ChannelSet,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_synced_overlay(
        synced_df=synced_df,
        real_column=f"real__{channel_set.ride_height}",
        simu_column=f"simu__{channel_set.ride_height}",
        title=f"{segment_label} {channel_set_key}: track vs sim ride height",
        ylabel=channel_set.ride_height,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_ride_height.png",
    )
    plot_synced_overlay(
        synced_df=synced_df,
        real_column=f"real__{channel_set.push_load}",
        simu_column=f"simu__{channel_set.push_load}",
        title=f"{segment_label} {channel_set_key}: track vs sim push load",
        ylabel=channel_set.push_load,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_push_load.png",
    )
    plot_synced_overlay(
        synced_df=synced_df,
        real_column=f"real__{channel_set.damper}",
        simu_column=f"simu__{channel_set.damper}",
        title=f"{segment_label} {channel_set_key}: track vs sim damper",
        ylabel=channel_set.damper,
        output_path=output_dir / f"{segment_label}_{channel_set_key}_damper.png",
    )
    plot_synced_overlay(
        synced_df=synced_df,
        real_column="real__pitot_c",
        simu_column="real__pair",
        title=f"{segment_label} {channel_set_key}: track pitot vs track pAir",
        ylabel="pressure",
        output_path=output_dir / f"{segment_label}_{channel_set_key}_pitot_vs_pair.png",
    )
    plot_synced_overlay(
        synced_df=synced_df,
        real_column="real__pair",
        simu_column="simu__pair",
        title=f"{segment_label} {channel_set_key}: track vs sim pair",
        ylabel="pair",
        output_path=output_dir / f"{segment_label}_{channel_set_key}_pair.png",
    )
    plot_synced_overlay(
        synced_df=synced_df,
        real_column="real__carspeed_art",
        simu_column="simu__carspeed_art",
        title=f"{segment_label} {channel_set_key}: track vs sim speed",
        ylabel="carspeed_art",
        output_path=output_dir / f"{segment_label}_{channel_set_key}_speed.png",
    )


def build_sync_columns(channel_set: ChannelSet) -> tuple[list[str], list[str]]:
    sync_columns = sorted(
        {
            "carspeed_art",
            "pair",
            "pitot_c",
            channel_set.ride_height,
            channel_set.push_load,
            channel_set.damper,
            channel_set.tyre_pressure,
        }
    )
    sync_reference_columns = ["carspeed_art", "pair", "pitot_c", channel_set.damper]
    return sync_columns, sync_reference_columns


def run_segment_comparison(
    *,
    segment_label: str,
    channel_set_key: str,
    channel_set: ChannelSet,
    real_segment_number: int,
    simu_segment_number: int,
    sync_grid_size: int,
    plot: bool,
    output_dir: Path,
    real_df: pd.DataFrame,
    simu_df: pd.DataFrame,
) -> dict:
    sync_columns, sync_reference_columns = build_sync_columns(channel_set)

    full_run_sync = synchronize_dataframes(
        real_df=real_df,
        simu_df=simu_df,
        columns=sync_columns,
        grid_size=sync_grid_size,
        reference_columns=sync_reference_columns,
    )

    real_segments = get_segments(real_df, segment_label)
    simu_segments = get_segments(simu_df, segment_label)
    real_segment = get_segment_by_number(
        real_segments, real_segment_number, "real"
    )
    simu_segment = get_segment_by_number(
        simu_segments, simu_segment_number, "simu"
    )

    local_sync = synchronize_dataframes(
        real_df=real_segment,
        simu_df=simu_segment,
        columns=sync_columns,
        grid_size=sync_grid_size,
        reference_columns=sync_reference_columns,
    )
    synced_df = local_sync.synced_df

    ride_height_metrics = compute_signal_comparison_metrics(
        synced_df[f"real__{channel_set.ride_height}"],
        synced_df[f"simu__{channel_set.ride_height}"],
    )
    push_load_metrics = compute_signal_comparison_metrics(
        synced_df[f"real__{channel_set.push_load}"],
        synced_df[f"simu__{channel_set.push_load}"],
    )
    damper_metrics = compute_signal_comparison_metrics(
        synced_df[f"real__{channel_set.damper}"],
        synced_df[f"simu__{channel_set.damper}"],
    )
    pitot_metrics = compute_signal_comparison_metrics(
        synced_df["real__pitot_c"],
        synced_df["real__pair"],
    )
    pair_metrics = compute_signal_comparison_metrics(
        synced_df["real__pair"],
        synced_df["simu__pair"],
    )
    tpms_metrics = compute_signal_comparison_metrics(
        synced_df[f"real__{channel_set.tyre_pressure}"],
        synced_df[f"simu__{channel_set.tyre_pressure}"],
    )

    print(f"\nSegment comparison: {segment_label}")
    print(f"Channel set: {channel_set_key} ({channel_set.name})")
    print(
        f"Real segment #{real_segment_number}: {len(real_segment)} samples, "
        f"Sim segment #{simu_segment_number}: {len(simu_segment)} samples"
    )
    print(
        "Full run coarse sync: "
        f"method={full_run_sync.method}, axis={full_run_sync.axis_method}, "
        f"refs={list(full_run_sync.reference_columns_used)}, "
        f"lag_steps={full_run_sync.lag_steps}, "
        f"lag_progress={full_run_sync.lag_progress:.4f}"
    )
    print(
        "Local segment fine sync: "
        f"method={local_sync.method}, axis={local_sync.axis_method}, "
        f"refs={list(local_sync.reference_columns_used)}, "
        f"lag_steps={local_sync.lag_steps}, "
        f"lag_progress={local_sync.lag_progress:.4f}"
    )

    print("\nSynchronized median comparisons")
    print_metric(
        "ride_height",
        get_synced_median(synced_df, f"real__{channel_set.ride_height}"),
        get_synced_median(synced_df, f"simu__{channel_set.ride_height}"),
    )
    print_metric(
        "push_load",
        get_synced_median(synced_df, f"real__{channel_set.push_load}"),
        get_synced_median(synced_df, f"simu__{channel_set.push_load}"),
    )
    print_metric(
        "damper",
        get_synced_median(synced_df, f"real__{channel_set.damper}"),
        get_synced_median(synced_df, f"simu__{channel_set.damper}"),
    )
    print_metric(
        "pitot",
        get_synced_median(synced_df, "real__pitot_c"),
        get_synced_median(synced_df, "real__pair"),
    )
    print_metric(
        "pair",
        get_synced_median(synced_df, "real__pair"),
        get_synced_median(synced_df, "simu__pair"),
    )
    print_metric(
        "tpms",
        get_synced_median(synced_df, f"real__{channel_set.tyre_pressure}"),
        get_synced_median(synced_df, f"simu__{channel_set.tyre_pressure}"),
    )

    print("\nSignal comparison metrics")
    print_comparison_metrics(synced_df, "ride_height", f"real__{channel_set.ride_height}", f"simu__{channel_set.ride_height}")
    print_comparison_metrics(synced_df, "push_load", f"real__{channel_set.push_load}", f"simu__{channel_set.push_load}")
    print_comparison_metrics(synced_df, "damper", f"real__{channel_set.damper}", f"simu__{channel_set.damper}")
    print_comparison_metrics(synced_df, "pitot", "real__pitot_c", "real__pair")
    print_comparison_metrics(synced_df, "pair", "real__pair", "simu__pair")
    print_comparison_metrics(synced_df, "tpms", f"real__{channel_set.tyre_pressure}", f"simu__{channel_set.tyre_pressure}")

    rh_verdict = evaluate_absolute_threshold_comparison(
        real_series=synced_df[f"real__{channel_set.ride_height}"],
        simu_series=synced_df[f"simu__{channel_set.ride_height}"],
        threshold=1.5,
        channel_name="ride_height",
        higher_state="higher_than_sim",
        lower_state="lower_than_sim",
    )
    push_verdict = evaluate_absolute_threshold_comparison(
        real_series=synced_df[f"real__{channel_set.push_load}"],
        simu_series=synced_df[f"simu__{channel_set.push_load}"],
        threshold=2.5,
        channel_name="push_load",
        higher_state="higher_than_sim",
        lower_state="lower_than_sim",
    )
    damper_verdict = evaluate_absolute_threshold_comparison(
        real_series=synced_df[f"real__{channel_set.damper}"],
        simu_series=synced_df[f"simu__{channel_set.damper}"],
        threshold=1.5,
        channel_name="damper",
        higher_state="higher_than_sim",
        lower_state="lower_than_sim",
    )
    tpms_verdict = evaluate_absolute_threshold_comparison(
        real_series=synced_df[f"real__{channel_set.tyre_pressure}"],
        simu_series=synced_df[f"simu__{channel_set.tyre_pressure}"],
        threshold=1.5,
        channel_name="tpms",
        higher_state="higher_than_sim",
        lower_state="lower_than_sim",
    )
    pitot_verdict = evaluate_relative_threshold_comparison(
        real_series=synced_df["real__pitot_c"],
        reference_series=synced_df["real__pair"],
        relative_threshold=0.03,
        channel_name="pitot vs pAir",
        lower_state="lower_than_pair",
        higher_state="higher_than_pair",
    )

    print("\nThreshold-aware verdicts")
    print_threshold_verdict("ride_height", rh_verdict.state, rh_verdict.detail)
    print_threshold_verdict("push_load", push_verdict.state, push_verdict.detail)
    print_threshold_verdict("damper", damper_verdict.state, damper_verdict.detail)
    print_threshold_verdict("tpms", tpms_verdict.state, tpms_verdict.detail)
    print_threshold_verdict("pitot_vs_pair", pitot_verdict.state, pitot_verdict.detail)

    if plot:
        generate_plots(
            synced_df=synced_df,
            segment_label=segment_label,
            channel_set_key=channel_set_key,
            channel_set=channel_set,
            output_dir=output_dir,
        )

    return {
        "segment_label": segment_label,
        "real_segment_number": real_segment_number,
        "simu_segment_number": simu_segment_number,
        "real_samples": len(real_segment),
        "simu_samples": len(simu_segment),
        "full_run_lag_steps": full_run_sync.lag_steps,
        "local_lag_steps": local_sync.lag_steps,
        "ride_height_state": rh_verdict.state,
        "push_load_state": push_verdict.state,
        "damper_state": damper_verdict.state,
        "tpms_state": tpms_verdict.state,
        "pitot_state": pitot_verdict.state,
        "ride_height_mean_error": safe_metric_value(ride_height_metrics.mean_error),
        "ride_height_nrmse_pct": safe_metric_value(None if ride_height_metrics.nrmse is None else 100.0 * ride_height_metrics.nrmse),
        "ride_height_r": safe_metric_value(ride_height_metrics.pearson_r),
        "push_load_mean_error": safe_metric_value(push_load_metrics.mean_error),
        "push_load_nrmse_pct": safe_metric_value(None if push_load_metrics.nrmse is None else 100.0 * push_load_metrics.nrmse),
        "push_load_r": safe_metric_value(push_load_metrics.pearson_r),
        "damper_mean_error": safe_metric_value(damper_metrics.mean_error),
        "damper_nrmse_pct": safe_metric_value(None if damper_metrics.nrmse is None else 100.0 * damper_metrics.nrmse),
        "damper_r": safe_metric_value(damper_metrics.pearson_r),
        "pitot_mean_error": safe_metric_value(pitot_metrics.mean_error),
        "pitot_nrmse_pct": safe_metric_value(None if pitot_metrics.nrmse is None else 100.0 * pitot_metrics.nrmse),
        "pitot_r": safe_metric_value(pitot_metrics.pearson_r),
        "pair_mean_error": safe_metric_value(pair_metrics.mean_error),
        "pair_nrmse_pct": safe_metric_value(None if pair_metrics.nrmse is None else 100.0 * pair_metrics.nrmse),
        "pair_r": safe_metric_value(pair_metrics.pearson_r),
        "tpms_mean_error": safe_metric_value(tpms_metrics.mean_error),
        "tpms_nrmse_pct": safe_metric_value(None if tpms_metrics.nrmse is None else 100.0 * tpms_metrics.nrmse),
        "tpms_r": safe_metric_value(tpms_metrics.pearson_r),
    }


def print_batch_summary(summary_rows: list[dict]) -> None:
    if not summary_rows:
        print("\nNo batch summary rows to print.")
        return

    summary_df = pd.DataFrame(summary_rows)
    columns = [
        "segment_label",
        "real_segment_number",
        "simu_segment_number",
        "real_samples",
        "simu_samples",
        "local_lag_steps",
        "ride_height_state",
        "push_load_state",
        "damper_state",
        "tpms_state",
        "pitot_state",
    ]
    print("\nBatch summary")
    print(summary_df[columns].to_string(index=False))


def build_summary_csv_path(output_dir: Path, segment_label: str, channel_set_key: str) -> Path:
    return output_dir / f"{segment_label}_{channel_set_key}_batch_summary.csv"


def main() -> None:
    args = parse_args()
    channel_set = CHANNEL_SETS[args.channel_set]
    real_df = load_segmented_data("real")
    simu_df = load_segmented_data("simu")

    segment_labels = (
        ["pit", "straight", "corner"]
        if args.segment_label == "all"
        else [args.segment_label]
    )

    batch_summary_rows: list[dict] = []

    for segment_label in segment_labels:
        real_segments = get_segments(real_df, segment_label)
        simu_segments = get_segments(simu_df, segment_label)

        if args.loop_all_segments:
            pair_count = min(len(real_segments), len(simu_segments))
            print(
                f"\nLooping all {segment_label} segments: "
                f"real={len(real_segments)}, simu={len(simu_segments)}, paired={pair_count}"
            )
            for segment_number in range(1, pair_count + 1):
                result = run_segment_comparison(
                    segment_label=segment_label,
                    channel_set_key=args.channel_set,
                    channel_set=channel_set,
                    real_segment_number=segment_number,
                    simu_segment_number=segment_number,
                    sync_grid_size=args.sync_grid_size,
                    plot=args.plot,
                    output_dir=args.output_dir,
                    real_df=real_df,
                    simu_df=simu_df,
                )
                batch_summary_rows.append(result)
        else:
            result = run_segment_comparison(
                segment_label=segment_label,
                channel_set_key=args.channel_set,
                channel_set=channel_set,
                real_segment_number=args.real_segment_number,
                simu_segment_number=args.simu_segment_number,
                sync_grid_size=args.sync_grid_size,
                plot=args.plot,
                output_dir=args.output_dir,
                real_df=real_df,
                simu_df=simu_df,
            )
            batch_summary_rows.append(result)

    if args.loop_all_segments:
        print_batch_summary(batch_summary_rows)
        if args.save_summary_csv:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = build_summary_csv_path(
                args.output_dir,
                args.segment_label,
                args.channel_set,
            )
            pd.DataFrame(batch_summary_rows).to_csv(summary_path, index=False)
            print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
