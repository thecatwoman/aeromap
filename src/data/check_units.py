import argparse

import pandas as pd

from src.run_paths import cleaned_merged_full_run_file, cleaned_simudata_full_run_file


DEFAULT_COLUMNS = [
    "distancelap",
    "carspeed_art",
    "avg_accx",
    "avg_accy",
    "pitot_c",
    "pair",
    "rh_f",
    "rh_r",
    "pushavg_c",
    "pushavd_c",
    "pusharg_c",
    "pushard_c",
    "damper_fl_art",
    "damper_fr_art",
    "damper_rl_art",
    "damper_rr_art",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--columns",
        nargs="*",
        default=DEFAULT_COLUMNS,
        help="Columns to compare between real and simulation.",
    )
    return parser.parse_args()


def summarize_column(df: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    if column not in df.columns:
        return {
            "count": 0,
            "missing": None,
            "min": None,
            "median": None,
            "max": None,
        }

    series = pd.to_numeric(df[column], errors="coerce")
    valid = series.dropna()

    if valid.empty:
        return {
            "count": 0,
            "missing": int(series.isna().sum()),
            "min": None,
            "median": None,
            "max": None,
        }

    return {
        "count": int(valid.count()),
        "missing": int(series.isna().sum()),
        "min": float(valid.min()),
        "median": float(valid.median()),
        "max": float(valid.max()),
    }


def format_value(value) -> str:
    if value is None:
        return "missing"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()

    real_df = pd.read_csv(cleaned_merged_full_run_file(), low_memory=False)
    simu_df = pd.read_csv(cleaned_simudata_full_run_file(), low_memory=False)

    print("Unit / scale sanity check\n")

    for column in args.columns:
        real_stats = summarize_column(real_df, column)
        simu_stats = summarize_column(simu_df, column)

        print(f"{column}")
        print(
            "  real : "
            f"count={format_value(real_stats['count'])}, "
            f"missing={format_value(real_stats['missing'])}, "
            f"min={format_value(real_stats['min'])}, "
            f"median={format_value(real_stats['median'])}, "
            f"max={format_value(real_stats['max'])}"
        )
        print(
            "  simu : "
            f"count={format_value(simu_stats['count'])}, "
            f"missing={format_value(simu_stats['missing'])}, "
            f"min={format_value(simu_stats['min'])}, "
            f"median={format_value(simu_stats['median'])}, "
            f"max={format_value(simu_stats['max'])}"
        )
        print()


if __name__ == "__main__":
    main()
