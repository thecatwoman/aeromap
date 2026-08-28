import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.apply_butterworth_filter import (
    DEFAULT_CUTOFF_HZ,
    DEFAULT_FILTER_ORDER,
    apply_butterworth_filter,
    infer_sampling_rate_hz,
)
from src.data.apply_median_filter import TARGET_COLUMNS
from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file


SEGMENT_LABELS = ["pit", "corner", "straight"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["real", "simu"],
        default="real",
        help="Dataset source to inspect when --input is not provided.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional explicit segmented CSV path.",
    )
    parser.add_argument(
        "--label",
        choices=["pit", "corner", "straight", "all"],
        default="all",
        help="Which segment label to plot.",
    )
    parser.add_argument(
        "--segment-number",
        type=int,
        default=1,
        help="1-based segment number within each label.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=DEFAULT_CUTOFF_HZ,
        help="Low-pass Butterworth cutoff frequency in Hz.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=DEFAULT_FILTER_ORDER,
        help="Butterworth filter order.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        help="Sampling rate in Hz. If omitted, try to infer it from the time column.",
    )
    parser.add_argument(
        "--x-mode",
        choices=["sample", "time"],
        default="sample",
        help="X-axis to use for the plots.",
    )
    parser.add_argument(
        "--plots-per-figure",
        type=int,
        default=3,
        help="Number of sensor subplots to show per figure.",
    )
    return parser.parse_args()


def get_input_path(source: str) -> Path:
    if source == "simu":
        return segmented_simudata_run_file()
    return segmented_rh_run_file()


def get_x_axis(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "time" and "time" in df.columns:
        return df["time"]
    return pd.Series(range(len(df)), index=df.index)


def get_segments(df: pd.DataFrame, label: str) -> list[pd.DataFrame]:
    if "segment_final" not in df.columns:
        raise ValueError("Missing 'segment_final'. Run segmentation first.")

    segment_df = df[df["segment_final"].astype("string") == label].copy()
    if segment_df.empty:
        return []

    segment_df["group"] = (segment_df.index.to_series().diff() != 1).cumsum()
    segments: list[pd.DataFrame] = []
    for _, group_df in segment_df.groupby("group"):
        segments.append(group_df.drop(columns=["group"]).copy())
    return segments


def chunk_columns(columns: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("--plots-per-figure must be greater than zero.")
    return [columns[i : i + size] for i in range(0, len(columns), size)]


def plot_segment(
    segment_df: pd.DataFrame,
    label: str,
    segment_number: int,
    x_mode: str,
    plots_per_figure: int,
    cutoff_hz: float,
) -> None:
    available_columns = [column for column in TARGET_COLUMNS if column in segment_df.columns]
    if not available_columns:
        print(f"[{label}] no target columns available")
        return

    x = get_x_axis(segment_df, x_mode)
    column_pages = chunk_columns(available_columns, plots_per_figure)

    for page_number, page_columns in enumerate(column_pages, start=1):
        fig, axes = plt.subplots(
            len(page_columns),
            1,
            figsize=(15, max(3.75 * len(page_columns), 6)),
            sharex=True,
        )

        if len(page_columns) == 1:
            axes = [axes]

        for ax, column in zip(axes, page_columns):
            raw = pd.to_numeric(segment_df[column], errors="coerce")
            butterworth = pd.to_numeric(
                segment_df[f"{column}_butterworth"],
                errors="coerce",
            )

            raw_mask = x.notna() & raw.notna()
            butterworth_mask = x.notna() & butterworth.notna()

            ax.plot(
                x[raw_mask],
                raw[raw_mask],
                label="raw",
                linewidth=0.9,
                alpha=0.55,
                color="0.45",
            )
            ax.plot(
                x[butterworth_mask],
                butterworth[butterworth_mask],
                label=f"butterworth {cutoff_hz:g} Hz",
                linewidth=1.8,
                color="tab:blue",
            )
            ax.set_title(column)
            ax.set_ylabel(column)
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel("Time" if x_mode == "time" else "Sample index")
        fig.suptitle(
            (
                f"{label} segment {segment_number} "
                f"(page {page_number}/{len(column_pages)})"
            ),
            fontsize=14,
        )
        plt.tight_layout()


def main() -> None:
    args = parse_args()

    input_path = args.input if args.input is not None else get_input_path(args.source)
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    if args.fs is not None:
        fs = args.fs
        fs_reason = "provided by --fs"
    else:
        fs, fs_reason = infer_sampling_rate_hz(df)

    filtered_df = apply_butterworth_filter(
        df=df,
        columns=TARGET_COLUMNS,
        cutoff_hz=args.cutoff_hz,
        fs=fs,
        order=args.order,
        replace_target_columns=False,
    )
    print(f"Sampling rate: {fs:.6f} Hz ({fs_reason})")

    labels = SEGMENT_LABELS if args.label == "all" else [args.label]

    for label in labels:
        segments = get_segments(filtered_df, label)
        if not segments:
            print(f"[{label}] no segments found")
            continue

        if args.segment_number < 1 or args.segment_number > len(segments):
            print(
                f"[{label}] segment-number {args.segment_number} is out of range "
                f"(available: 1..{len(segments)})"
            )
            continue

        segment_df = segments[args.segment_number - 1]
        print(
            f"[{label}] plotting segment {args.segment_number} "
            f"with {len(segment_df)} samples"
        )
        plot_segment(
            segment_df=segment_df,
            label=label,
            segment_number=args.segment_number,
            x_mode=args.x_mode,
            plots_per_figure=args.plots_per_figure,
            cutoff_hz=args.cutoff_hz,
        )

    plt.show()


if __name__ == "__main__":
    main()
