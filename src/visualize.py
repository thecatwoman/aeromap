import argparse
from pathlib import Path

from macrovisu import visualize_segmented_data
from src.run_paths import (
    processed_simudata_dir,
    processed_run_dir,
    segmented_rh_run_file,
    segmented_simudata_run_file,
)


def threshold_real_segmented_file() -> Path:
    return processed_run_dir() / "Segmented" / (
        "barcelona_2026_merged_cleaned_macro_v3_run_46_segmented.csv"
    )


def threshold_simu_segmented_file() -> Path:
    return processed_simudata_dir() / (
        "barcelona_2026_simudata_segmented_macro_v3_run_46.csv"
    )


def test_real_segmented_file() -> Path:
    return processed_run_dir() / "Segmented" / (
        "barcelona_2026_merged_cleaned_test_run_46_segmented.csv"
    )


def test_simu_segmented_file() -> Path:
    return processed_simudata_dir() / (
        "barcelona_2026_simudata_segmented_test_run_46.csv"
    )


def get_visualization_path(source: str, segmenter: str):
    if source == "simu":
        if segmenter == "test":
            return test_simu_segmented_file(), "Simulator Segmentation - Test Segmenter"
        if segmenter == "threshold":
            return threshold_simu_segmented_file(), "Simulator Segmentation - Threshold"
        return segmented_simudata_run_file(), "Simulator Segmentation"

    if segmenter == "test":
        return test_real_segmented_file(), "Real Data Segmentation - Test Segmenter"
    if segmenter == "threshold":
        return threshold_real_segmented_file(), "Real Data Segmentation - Threshold"
    if segmenter == "macroway":
        return segmented_rh_run_file(), "Real Data Segmentation - Macroway"

    threshold_path = threshold_real_segmented_file()
    if threshold_path.exists():
        return threshold_path, "Real Data Segmentation - Threshold"

    return segmented_rh_run_file(), "Real Data Segmentation - Macroway"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["real", "simu"],
        default="real",
        help="Dataset source to visualize.",
    )
    parser.add_argument(
        "--segmenter",
        choices=["auto", "threshold", "macroway", "test"],
        default="auto",
        help="Which segmented output to visualize.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path, title = get_visualization_path(args.source, args.segmenter)
    visualize_segmented_data(data_path=data_path, title=title)


if __name__ == "__main__":
    main()
