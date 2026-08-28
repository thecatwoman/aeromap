import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from src.data.apply_despike_filter import apply_despike_filter, DESPIKE_COLUMNS
from src.data.apply_median_filter import get_input_path


DEFAULT_CUTOFF_HZ = 3
DEFAULT_FILTER_ORDER = 4
DEFAULT_SAMPLING_RATE_HZ = 100.0
DEFAULT_DESPIKE_WINDOW = 6
DEFAULT_DESPIKE_MAD_K = 3.5
RIDE_HEIGHT_BASE_CUTOFF_HZ = 2.5
RIDE_HEIGHT_CORNER_CUTOFF_HZ = 2.5

RIDE_HEIGHT_COLUMNS = ["rh_f", "rh_r"]
PUSH_COLUMNS = [
    "pushavd_c",
    "pushavg_c",
    "pushard_c",
    "pusharg_c",
]
FRONT_PUSH_COLUMNS = [
    "pushavd_c",
    "pushavg_c",
]
REAR_PUSH_COLUMNS = [
    "pushard_c",
    "pusharg_c",
]
PITOT_COLUMNS = ["pitot_c"]
DAMPER_COLUMNS = [
    "damper_fl_art",
    "damper_fr_art",
    "damper_rl_art",
    "damper_rr_art",
]
CHANNEL_CUTOFFS_HZ = {
    "pushavd_c": 2.5,
    "pushavg_c": 2.5,
    "pushard_c": 2.5,
    "pusharg_c": 2.5,
    "pitot_c": 5.0,
    "damper_fl_art": 3.0,
    "damper_fr_art": 3.0,
    "damper_rl_art": 3.0,
    "damper_rr_art": 3.0,
}
BUTTERWORTH_TARGET_COLUMNS = (
    RIDE_HEIGHT_COLUMNS + PUSH_COLUMNS + PITOT_COLUMNS + DAMPER_COLUMNS
)


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
        "--replace-target-columns",
        action="store_true",
        help="Overwrite the target columns instead of writing *_butterworth companions.",
    )
    parser.add_argument(
        "--despike-first",
        action="store_true",
        help="Replace extreme spikes with a rolling-median estimate before Butterworth filtering.",
    )
    parser.add_argument(
        "--despike-window",
        type=int,
        default=DEFAULT_DESPIKE_WINDOW,
        help="Centered rolling median window size used for spike detection.",
    )
    parser.add_argument(
        "--despike-mad-k",
        type=float,
        default=DEFAULT_DESPIKE_MAD_K,
        help="MAD threshold multiplier used to flag spikes before Butterworth filtering.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=BUTTERWORTH_TARGET_COLUMNS,
        help="Columns to Butterworth-filter.",
    )
    return parser.parse_args()


def get_output_path(input_path: Path) -> Path:
    return input_path.with_name(
        f"{input_path.stem}_butterworth_filtered{input_path.suffix}"
    )


def parse_time_value(value) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return None

    parts = text.split(":")
    if len(parts) != 2:
        return None

    major, minor = parts
    try:
        major_value = float(major)
    except ValueError:
        return None

    if not minor.isdigit():
        try:
            return float(text.replace(":", "."))
        except ValueError:
            return None

    scale = 10 ** len(minor)
    return major_value + (int(minor) / scale)


def infer_sampling_rate_hz(
    df: pd.DataFrame,
    default_fs: float = DEFAULT_SAMPLING_RATE_HZ,
) -> tuple[float, str]:
    if "time" not in df.columns:
        return default_fs, "time column missing, using default"

    parsed = df["time"].map(parse_time_value)
    diffs = parsed.diff().dropna()
    diffs = diffs[diffs > 0]

    if diffs.empty:
        return default_fs, "time column not usable, using default"

    dt = float(diffs.median())
    if dt <= 0:
        return default_fs, "non-positive median dt, using default"

    return 1.0 / dt, f"inferred from time median dt={dt:.6f}s"


def split_valid_segments(series: pd.Series) -> list[pd.Index]:
    valid_mask = series.notna()
    if not valid_mask.any():
        return []

    group_ids = (valid_mask != valid_mask.shift(fill_value=False)).cumsum()
    return [group.index for _, group in series[valid_mask].groupby(group_ids[valid_mask])]


def filter_valid_segment(
    values: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    if len(values) < 3:
        return values.copy()

    default_padlen = 3 * max(len(a), len(b))
    padlen = min(default_padlen, len(values) - 1)

    if padlen < 1:
        return values.copy()

    return filtfilt(b, a, values, padlen=padlen)


def get_segment_label_column(df: pd.DataFrame) -> str | None:
    for column in ["segment_final", "segment_macro_v3"]:
        if column in df.columns:
            return column
    return None


def get_filter_coefficients(
    cutoff_hz: float,
    fs: float,
    order: int,
    cache: dict[float, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if cutoff_hz <= 0:
        raise ValueError("Cutoff frequency must be greater than 0 Hz.")
    if fs <= 0:
        raise ValueError("Sampling rate must be greater than 0 Hz.")
    if cutoff_hz >= fs / 2:
        raise ValueError("Cutoff frequency must be lower than the Nyquist frequency.")
    if order < 1:
        raise ValueError("Filter order must be at least 1.")

    if cutoff_hz not in cache:
        cache[cutoff_hz] = cast(
            tuple[np.ndarray, np.ndarray],
            butter(order, cutoff_hz, btype="low", fs=fs, output="ba"),
        )
    return cache[cutoff_hz]


def contiguous_true_segments(mask: pd.Series) -> list[pd.Index]:
    if not mask.any():
        return []
    group_ids = (mask != mask.shift(fill_value=False)).cumsum()
    return [group.index for _, group in mask[mask].groupby(group_ids[mask])]


def apply_butterworth_filter(
    df: pd.DataFrame,
    columns: list[str],
    fs: float,
    order: int,
    replace_target_columns: bool = False,
    cutoff_by_column: dict[str, float] | None = None,
) -> pd.DataFrame:
    result = df.copy()
    coefficient_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    segment_label_column = get_segment_label_column(result)
    segment_labels = (
        result[segment_label_column].astype("string")
        if segment_label_column is not None
        else pd.Series(pd.NA, index=result.index, dtype="string")
    )

    for column in columns:
        if column not in result.columns:
            continue

        series = pd.to_numeric(result[column], errors="coerce")
        filtered = series.copy()
        base_cutoff_hz = (
            cutoff_by_column[column]
            if cutoff_by_column is not None and column in cutoff_by_column
            else DEFAULT_CUTOFF_HZ
        )

        if column in RIDE_HEIGHT_COLUMNS and segment_label_column is not None:
            valid_mask = series.notna()
            corner_mask = valid_mask & segment_labels.eq("corner").fillna(False)
            non_corner_mask = valid_mask & ~segment_labels.eq("corner").fillna(False)

            corner_b, corner_a = get_filter_coefficients(
                cutoff_hz=RIDE_HEIGHT_CORNER_CUTOFF_HZ,
                fs=fs,
                order=order,
                cache=coefficient_cache,
            )
            base_b, base_a = get_filter_coefficients(
                cutoff_hz=RIDE_HEIGHT_BASE_CUTOFF_HZ,
                fs=fs,
                order=order,
                cache=coefficient_cache,
            )

            for segment_index in contiguous_true_segments(corner_mask):
                segment_values = series.loc[segment_index].to_numpy(dtype="float64")
                filtered.loc[segment_index] = filter_valid_segment(
                    segment_values,
                    b=corner_b,
                    a=corner_a,
                )
            for segment_index in contiguous_true_segments(non_corner_mask):
                segment_values = series.loc[segment_index].to_numpy(dtype="float64")
                filtered.loc[segment_index] = filter_valid_segment(
                    segment_values,
                    b=base_b,
                    a=base_a,
                )
        else:
            b, a = get_filter_coefficients(
                cutoff_hz=base_cutoff_hz,
                fs=fs,
                order=order,
                cache=coefficient_cache,
            )
            for segment_index in split_valid_segments(series):
                segment_values = series.loc[segment_index].to_numpy(dtype="float64")
                filtered.loc[segment_index] = filter_valid_segment(segment_values, b=b, a=a)

        output_column = (
            column if replace_target_columns else f"{column}_butterworth"
        )
        result[output_column] = filtered

    return result


def main() -> None:
    args = parse_args()

    input_path = args.input if args.input is not None else get_input_path(args.source)
    output_path = args.output if args.output is not None else get_output_path(input_path)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    if args.fs is not None:
        fs = args.fs
        fs_reason = "provided by --fs"
    else:
        fs, fs_reason = infer_sampling_rate_hz(df)

    if args.despike_first:
        df, spike_counts = apply_despike_filter(
            df=df,
            columns=DESPIKE_COLUMNS,
            window=args.despike_window,
            mad_k=args.despike_mad_k,
            replace_target_columns=True,
        )
        total_spikes = sum(spike_counts.values())
        print(
            f"Despike first: window={args.despike_window}, "
            f"mad_k={args.despike_mad_k}, total spikes replaced={total_spikes}"
        )

    filtered_df = apply_butterworth_filter(
        df=df,
        columns=args.columns,
        fs=fs,
        order=args.order,
        replace_target_columns=args.replace_target_columns,
        cutoff_by_column=CHANNEL_CUTOFFS_HZ,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)

    present_columns = [column for column in args.columns if column in df.columns]
    missing_columns = [column for column in args.columns if column not in df.columns]

    print("Butterworth channel settings:")
    print(f"  rh_f: {RIDE_HEIGHT_BASE_CUTOFF_HZ} Hz normally, {RIDE_HEIGHT_CORNER_CUTOFF_HZ} Hz in corners")
    print(f"  rh_r: {RIDE_HEIGHT_BASE_CUTOFF_HZ} Hz normally, {RIDE_HEIGHT_CORNER_CUTOFF_HZ} Hz in corners")
    print("  front push rods: 2.5 Hz")
    print("  rear push rods: 2.5 Hz")
    for column, cutoff_hz in CHANNEL_CUTOFFS_HZ.items():
        if column not in PUSH_COLUMNS:
            print(f"  {column}: {cutoff_hz} Hz")
    print("Not Butterworth-filtered in the baseline default target list: pair")
    print(f"Order: {args.order}")
    print(f"Sampling rate: {fs:.6f} Hz ({fs_reason})")
    print(f"Filtered columns present: {present_columns}")
    if missing_columns:
        print(f"Missing columns skipped: {missing_columns}")
    print(
        "Mode: "
        + (
            "replaced target columns"
            if args.replace_target_columns
            else "added *_butterworth columns"
        )
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
