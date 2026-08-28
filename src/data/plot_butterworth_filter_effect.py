import argparse

import matplotlib.pyplot as plt
import pandas as pd

from macroway import apply_segmentation_macro_style
from src.data.apply_butterworth_filter import (
    CHANNEL_CUTOFFS_HZ,
    DEFAULT_CUTOFF_HZ,
    DEFAULT_FILTER_ORDER,
    RIDE_HEIGHT_BASE_CUTOFF_HZ,
    RIDE_HEIGHT_COLUMNS,
    RIDE_HEIGHT_CORNER_CUTOFF_HZ,
    apply_butterworth_filter,
    infer_sampling_rate_hz,
)
from src.data.apply_median_filter import TARGET_COLUMNS, get_input_path

SEGMENTATION_KWARGS = {
    "min_points": 150,
    "tol_speed": 2.5,
    "pit_tol_speed": 70.0,
    "pit_long_acc_max": 1.0,
    "pit_constant_window": 15,
    "pit_constant_speed_std_max": 0.10,
    "pit_constant_min_points": 90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["real", "simu"],
        default="real",
        help="Dataset source to inspect.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=None,
        help="Optional shared low-pass Butterworth cutoff frequency in Hz. If omitted, use channel-specific defaults where available.",
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
        "--start",
        type=int,
        default=0,
        help="Start row for the plotted slice.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1500,
        help="Number of rows to plot from the start index.",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Plot the full run. Overrides --start/--count.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=TARGET_COLUMNS,
        help="Columns to plot.",
    )
    return parser.parse_args()


def get_x_axis(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "time" and "time" in df.columns:
        return df["time"]
    return pd.Series(range(len(df)), index=df.index)


def slice_dataframe(df: pd.DataFrame, start: int, count: int) -> pd.DataFrame:
    if start < 0:
        raise ValueError("--start must be zero or positive.")
    if count <= 0:
        raise ValueError("--count must be greater than zero.")
    return df.iloc[start : start + count].copy()


def main() -> None:
    args = parse_args()

    input_path = get_input_path(args.source)
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    if not args.full_run:
        df = slice_dataframe(df, start=args.start, count=args.count)

    if args.fs is not None:
        fs = args.fs
        fs_reason = "provided by --fs"
    else:
        fs, fs_reason = infer_sampling_rate_hz(df)

    filter_df = df.copy()
    ride_height_requested = (
        args.cutoff_hz is None
        and any(column in RIDE_HEIGHT_COLUMNS for column in args.columns)
    )
    if ride_height_requested:
        filter_df = apply_segmentation_macro_style(filter_df, **SEGMENTATION_KWARGS)
        print(
            "Ride-height mode: segmented first, then Butterworth "
            f"({RIDE_HEIGHT_BASE_CUTOFF_HZ:g} Hz normally, "
            f"{RIDE_HEIGHT_CORNER_CUTOFF_HZ:g} Hz in corners)"
        )

    filtered_df = apply_butterworth_filter(
        df=filter_df,
        columns=args.columns,
        fs=fs,
        order=args.order,
        replace_target_columns=False,
        cutoff_by_column=(
            {column: args.cutoff_hz for column in args.columns}
            if args.cutoff_hz is not None
            else CHANNEL_CUTOFFS_HZ
        ),
    )

    available_columns = [column for column in args.columns if column in filtered_df.columns]
    if not available_columns:
        raise ValueError("None of the target columns are present in the selected dataset.")

    print(f"Sampling rate: {fs:.6f} Hz ({fs_reason})")

    x = get_x_axis(filtered_df, args.x_mode)
    fig, axes = plt.subplots(
        len(available_columns),
        1,
        figsize=(14, max(3.5 * len(available_columns), 6)),
        sharex=True,
    )

    if len(available_columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, available_columns):
        raw = pd.to_numeric(filtered_df[column], errors="coerce")
        butterworth = pd.to_numeric(
            filtered_df[f"{column}_butterworth"],
            errors="coerce",
        )

        raw_mask = x.notna() & raw.notna()
        filtered_mask = x.notna() & butterworth.notna()

        ax.plot(x[raw_mask], raw[raw_mask], label="raw", linewidth=1.0, alpha=0.7)
        ax.plot(
            x[filtered_mask],
            butterworth[filtered_mask],
            label=(
                f"butterworth {RIDE_HEIGHT_BASE_CUTOFF_HZ:g} Hz / "
                f"{RIDE_HEIGHT_CORNER_CUTOFF_HZ:g} Hz corners"
                if args.cutoff_hz is None and column in RIDE_HEIGHT_COLUMNS
                else
                f"butterworth {args.cutoff_hz:g} Hz"
                if args.cutoff_hz is not None
                else f"butterworth {CHANNEL_CUTOFFS_HZ.get(column, DEFAULT_CUTOFF_HZ):g} Hz"
            ),
            linewidth=1.8,
        )
        ax.set_title(column)
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time" if args.x_mode == "time" else "Sample index")
    fig.suptitle(
        (
            f"{args.source} Butterworth comparison "
            + (
                "full run"
                if args.full_run
                else f"rows {args.start} to {args.start + len(filtered_df) - 1}"
            )
        ),
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
