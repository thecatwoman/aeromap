import argparse

import pandas as pd

from macroway import apply_segmentation_macro_style
from src.run_paths import (
    cleaned_merged_full_run_file,
    cleaned_simudata_full_run_file,
    segmented_rh_run_file,
    segmented_simudata_run_file,
)


SEGMENTATION_KWARGS = {
    "min_points": 150,
    "tol_speed": 2.5,
    "pit_tol_speed": 70.0,
    "pit_long_acc_max": 1.0,
    "pit_constant_window": 15,
    "pit_constant_speed_std_max": 0.10,
    "pit_constant_min_points": 90,
}


def get_segmentation_paths(source: str):
    if source == "simu":
        return cleaned_simudata_full_run_file(), segmented_simudata_run_file()

    return cleaned_merged_full_run_file(), segmented_rh_run_file()


def run(source: str = "real") -> None:
    data_path, output_path = get_segmentation_paths(source)

    print(f"Loading {source} data...")
    df = pd.read_csv(data_path, low_memory=False)

    print("Running macroway segmentation...")
    df = apply_segmentation_macro_style(
        df,
        **SEGMENTATION_KWARGS,
    )

    print("Saving...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Done!")
    print(df["segment_final"].value_counts(dropna=False))
    print(f"Saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["real", "simu"],
        default="real",
        help="Dataset source to segment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run(source=args.source)


if __name__ == "__main__":
    main()
