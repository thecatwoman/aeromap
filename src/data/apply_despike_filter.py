import argparse
from pathlib import Path

import pandas as pd

from src.data.apply_median_filter import get_input_path


DEFAULT_WINDOW = 12
DEFAULT_MAD_K = 3.5
RIDE_HEIGHT_WINDOW = 12
RIDE_HEIGHT_MAD_K = 3.5

PUSH_CHANNEL_WINDOW = 20
PUSH_CHANNEL_MAD_K = 2.5

RIDE_HEIGHT_COLUMNS = [
    "rh_r",
    "rh_f",
]
PUSH_COLUMNS = [
    "pushard_c",
    "pusharg_c",
    "pushavd_c",
    "pushavg_c",
]
DESPIKE_COLUMNS = RIDE_HEIGHT_COLUMNS + PUSH_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["real", "simu"],
        default="real",
        help="Dataset source to filter when --input is not provided.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional explicit input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit output CSV path.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Centered rolling median window size for spike detection.",
    )
    parser.add_argument(
        "--mad-k",
        type=float,
        default=DEFAULT_MAD_K,
        help="MAD threshold multiplier used to flag spikes.",
    )
    parser.add_argument(
        "--replace-target-columns",
        action="store_true",
        help="Overwrite the target columns instead of writing *_despiked companions.",
    )
    return parser.parse_args()


def get_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_despiked{input_path.suffix}")


def compute_spike_mask(
    series: pd.Series,
    window: int,
    mad_k: float,
) -> tuple[pd.Series, pd.Series]:
    if window <= 1:
        raise ValueError("Despike window must be greater than 1.")
    if mad_k <= 0:
        raise ValueError("MAD threshold multiplier must be greater than 0.")

    clean = pd.to_numeric(series, errors="coerce")
    rolling_median = clean.rolling(
        window=window,
        center=True,
        min_periods=max(3, window // 2),
    ).median()
    residuals = clean - rolling_median
    median_residual = residuals.median(skipna=True)
    mad = (residuals - median_residual).abs().median(skipna=True)

    if pd.isna(mad) or mad == 0:
        return pd.Series(False, index=series.index), rolling_median

    threshold = mad_k * 1.4826 * mad
    mask = (residuals - median_residual).abs() > threshold
    return mask.fillna(False), rolling_median


def apply_despike_filter(
    df: pd.DataFrame,
    columns: list[str],
    window: int,
    mad_k: float,
    replace_target_columns: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    result = df.copy()
    spike_counts: dict[str, int] = {}

    for column in columns:
        if column not in result.columns:
            continue

        series = pd.to_numeric(result[column], errors="coerce")
        column_window = window
        column_mad_k = mad_k
        if column in RIDE_HEIGHT_COLUMNS:
            column_window = RIDE_HEIGHT_WINDOW
            column_mad_k = RIDE_HEIGHT_MAD_K
        elif column in PUSH_COLUMNS:
            column_window = PUSH_CHANNEL_WINDOW
            column_mad_k = PUSH_CHANNEL_MAD_K
        spike_mask, rolling_median = compute_spike_mask(
            series=series,
            window=column_window,
            mad_k=column_mad_k,
        )

        despiked = series.where(~spike_mask, rolling_median)
        output_column = column if replace_target_columns else f"{column}_despiked"
        result[output_column] = despiked
        spike_counts[column] = int(spike_mask.sum())

    return result, spike_counts


def main() -> None:
    args = parse_args()

    input_path = args.input if args.input is not None else get_input_path(args.source)
    output_path = args.output if args.output is not None else get_output_path(input_path)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    filtered_df, spike_counts = apply_despike_filter(
        df=df,
        columns=DESPIKE_COLUMNS,
        window=args.window,
        mad_k=args.mad_k,
        replace_target_columns=args.replace_target_columns,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)

    present_columns = [column for column in DESPIKE_COLUMNS if column in df.columns]
    missing_columns = [column for column in DESPIKE_COLUMNS if column not in df.columns]

    print(f"Window: {args.window}")
    print(f"MAD threshold multiplier: {args.mad_k}")
    print(f"Filtered columns present: {present_columns}")
    for column in present_columns:
        if column in RIDE_HEIGHT_COLUMNS:
            channel_window = RIDE_HEIGHT_WINDOW
            channel_mad_k = RIDE_HEIGHT_MAD_K
        elif column in PUSH_COLUMNS:
            channel_window = PUSH_CHANNEL_WINDOW
            channel_mad_k = PUSH_CHANNEL_MAD_K
        else:
            channel_window = args.window
            channel_mad_k = args.mad_k
        print(
            f"{column}: spikes replaced={spike_counts.get(column, 0)} | window={channel_window} | mad_k={channel_mad_k}"
        )
    if missing_columns:
        print(f"Missing columns skipped: {missing_columns}")
    print(
        "Mode: "
        + (
            "replaced target columns"
            if args.replace_target_columns
            else "added *_despiked columns"
        )
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
