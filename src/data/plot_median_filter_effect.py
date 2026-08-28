import argparse

import matplotlib.pyplot as plt
import pandas as pd

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
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Centered rolling median window size.",
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
    df = slice_dataframe(df, start=args.start, count=args.count)

    filtered_df = apply_median_filter(
        df=df,
        columns=args.columns,
        window=args.window,
        replace_target_columns=False,
    )

    available_columns = [column for column in args.columns if column in filtered_df.columns]
    if not available_columns:
        raise ValueError("None of the target columns are present in the selected dataset.")

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
        median = pd.to_numeric(filtered_df[f"{column}_median"], errors="coerce")

        raw_mask = x.notna() & raw.notna()
        median_mask = x.notna() & median.notna()

        ax.plot(x[raw_mask], raw[raw_mask], label="raw", linewidth=1.0, alpha=0.7)
        ax.plot(
            x[median_mask],
            median[median_mask],
            label=f"median w={args.window}",
            linewidth=1.8,
        )
        ax.set_title(column)
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time" if args.x_mode == "time" else "Sample index")
    fig.suptitle(
        f"{args.source} median filter comparison rows {args.start} to {args.start + len(filtered_df) - 1}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
