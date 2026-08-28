import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.data.apply_butterworth_filter import (
    DEFAULT_CUTOFF_HZ,
    DEFAULT_FILTER_ORDER,
    apply_butterworth_filter,
    infer_sampling_rate_hz,
)
from src.data.apply_median_filter import (
    DEFAULT_WINDOW,
    TARGET_COLUMNS,
    apply_median_filter,
    get_input_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["real", "simu"],
        default="real",
        help="Dataset source to inspect.",
    )
    parser.add_argument(
        "--median-window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Centered rolling median window size.",
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
        "--plots-per-figure",
        type=int,
        default=3,
        help="Number of sensor subplots to show per figure.",
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


def chunk_columns(columns: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("--plots-per-figure must be greater than zero.")
    return [columns[i : i + size] for i in range(0, len(columns), size)]


def main() -> None:
    args = parse_args()

    input_path = get_input_path(args.source)
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    df = slice_dataframe(df, start=args.start, count=args.count)

    if args.fs is not None:
        fs = args.fs
        fs_reason = "provided by --fs"
    else:
        fs, fs_reason = infer_sampling_rate_hz(df)

    median_df = apply_median_filter(
        df=df,
        columns=TARGET_COLUMNS,
        window=args.median_window,
        replace_target_columns=False,
    )
    butterworth_df = apply_butterworth_filter(
        df=df,
        columns=TARGET_COLUMNS,
        cutoff_hz=args.cutoff_hz,
        fs=fs,
        order=args.order,
        replace_target_columns=False,
    )

    available_columns = [column for column in TARGET_COLUMNS if column in df.columns]
    if not available_columns:
        raise ValueError("None of the target columns are present in the selected dataset.")

    print(f"Sampling rate: {fs:.6f} Hz ({fs_reason})")

    x = get_x_axis(df, args.x_mode)
    column_pages = chunk_columns(available_columns, args.plots_per_figure)

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
            raw = pd.to_numeric(df[column], errors="coerce")
            median = pd.to_numeric(median_df[f"{column}_median"], errors="coerce")
            butterworth = pd.to_numeric(
                butterworth_df[f"{column}_butterworth"],
                errors="coerce",
            )

            raw_mask = x.notna() & raw.notna()
            median_mask = x.notna() & median.notna()
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
                x[median_mask],
                median[median_mask],
                label=f"median w={args.median_window}",
                linewidth=1.6,
                color="tab:orange",
            )
            ax.plot(
                x[butterworth_mask],
                butterworth[butterworth_mask],
                label=f"butterworth {args.cutoff_hz:g} Hz",
                linewidth=1.8,
                color="tab:blue",
            )
            ax.set_title(column)
            ax.set_ylabel(column)
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel("Time" if args.x_mode == "time" else "Sample index")
        fig.suptitle(
            (
                f"{args.source} filter comparison rows {args.start} "
                f"to {args.start + len(df) - 1} "
                f"(page {page_number}/{len(column_pages)})"
            ),
            fontsize=14,
        )
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
