import argparse
from pathlib import Path

import pandas as pd

from src.run_paths import cleaned_merged_full_run_file, cleaned_simudata_full_run_file


TARGET_COLUMNS = [
    "pushavg_c",
    "pushavd_c",
    "pusharg_c",
    "pushard_c",
    "hdcav_c",
    "hdcar_c",
    "pitot_c",
    "pair",
]
RIDE_HEIGHT_COLUMNS = [
    "rh_f",
    "rh_r",
]
DEFAULT_WINDOW = 10


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
        help="Centered rolling median window size.",
    )
    parser.add_argument(
        "--replace-target-columns",
        action="store_true",
        help="Overwrite the target columns instead of writing *_median companions.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=TARGET_COLUMNS,
        help="Columns to median-filter.",
    )
    return parser.parse_args()

    
def get_input_path(source: str) -> Path:
    if source == "simu":
        return cleaned_simudata_full_run_file()
    return cleaned_merged_full_run_file()


def get_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_median_filtered{input_path.suffix}")


def apply_median_filter(
    df: pd.DataFrame,
    columns: list[str],
    window: int,
    replace_target_columns: bool = False,
) -> pd.DataFrame:
    if window <= 1:
        raise ValueError("Median filter window must be greater than 1.")

    result = df.copy()
    min_periods = max(1, window // 2)

    for column in columns:
        if column not in result.columns:
            continue

        filtered = pd.to_numeric(result[column], errors="coerce").rolling(
            window=window,
            center=True,
            min_periods=min_periods,
        ).median()

        output_column = column if replace_target_columns else f"{column}_median"
        result[output_column] = filtered

    return result


def main() -> None:
    args = parse_args()

    input_path = args.input if args.input is not None else get_input_path(args.source)
    output_path = args.output if args.output is not None else get_output_path(input_path)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    filtered_df = apply_median_filter(
        df=df,
        columns=args.columns,
        window=args.window,
        replace_target_columns=args.replace_target_columns,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)

    present_columns = [column for column in args.columns if column in df.columns]
    missing_columns = [column for column in args.columns if column not in df.columns]

    print(f"Window: {args.window}")
    print(f"Filtered columns present: {present_columns}")
    if missing_columns:
        print(f"Missing columns skipped: {missing_columns}")
    print(
        "Mode: "
        + (
            "replaced target columns"
            if args.replace_target_columns
            else "added *_median columns"
        )
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
