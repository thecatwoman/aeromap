import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from macroway import apply_segmentation_macro_style
from src.data.apply_butterworth_filter import (
    BUTTERWORTH_TARGET_COLUMNS,
    CHANNEL_CUTOFFS_HZ,
    DEFAULT_CUTOFF_HZ,
    DEFAULT_DESPIKE_MAD_K,
    DEFAULT_DESPIKE_WINDOW,
    DEFAULT_FILTER_ORDER,
    RIDE_HEIGHT_BASE_CUTOFF_HZ,
    RIDE_HEIGHT_CORNER_CUTOFF_HZ,
    RIDE_HEIGHT_COLUMNS,
    apply_butterworth_filter,
    infer_sampling_rate_hz,
)
from src.data.apply_despike_filter import (
    DESPIKE_COLUMNS,
    apply_despike_filter,
)
from src.data.build_corner_map_table import (
    build_corner_summary,
    build_exact_two_way_table,
    build_two_way_table,
)
from src.run_paths import CURRENT_RUN, cleaned_merged_full_run_file, processed_run_dir


SEGMENTATION_KWARGS = {
    "min_points": 150,
    "tol_speed": 2.5,
    "pit_tol_speed": 70.0,
    "pit_long_acc_max": 1.0,
    "pit_constant_window": 15,
    "pit_constant_speed_std_max": 0.10,
    "pit_constant_min_points": 90,
}


@dataclass
class BranchConfig:
    name: str
    despike: bool
    butterworth: bool


BRANCHES = [
    BranchConfig(name="raw", despike=False, butterworth=False),
    BranchConfig(name="despike", despike=True, butterworth=False),
    BranchConfig(name="despike_butterworth", despike=True, butterworth=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled raw vs despike vs despike+Butterworth experiment on the real baseline dataset.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=cleaned_merged_full_run_file(),
        help="Baseline real dataset to branch from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=processed_run_dir() / "Filtering",
        help="Directory where branch datasets and reports will be saved.",
    )
    parser.add_argument(
        "--despike-window",
        type=int,
        default=DEFAULT_DESPIKE_WINDOW,
        help="Despike rolling median window size.",
    )
    parser.add_argument(
        "--despike-mad-k",
        type=float,
        default=DEFAULT_DESPIKE_MAD_K,
        help="Despike MAD threshold multiplier.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=DEFAULT_CUTOFF_HZ,
        help="Fallback Butterworth cutoff in Hz.",
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
        default=None,
        help="Optional explicit sampling rate. If omitted, infer from time.",
    )
    parser.add_argument(
        "--rh-f-bin",
        type=float,
        default=0.5,
        help="Bin size for rh_f map columns.",
    )
    parser.add_argument(
        "--rh-r-bin",
        type=float,
        default=0.5,
        help="Bin size for rh_r map rows.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimals to keep in exported tables.",
    )
    return parser.parse_args()


def apply_branch_filters(
    df: pd.DataFrame,
    config: BranchConfig,
    *,
    cutoff_hz: float,
    fs: float,
    order: int,
    despike_window: int,
    despike_mad_k: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    details: dict[str, Any] = {
        "despike": config.despike,
        "butterworth": config.butterworth,
        "despike_columns": [],
        "butterworth_columns_pre_segmentation": [],
        "butterworth_columns_post_segmentation": [],
        "spike_counts": {},
    }

    if config.despike:
        out, spike_counts = apply_despike_filter(
            df=out,
            columns=DESPIKE_COLUMNS,
            window=despike_window,
            mad_k=despike_mad_k,
            replace_target_columns=True,
        )
        details["despike_columns"] = [column for column in DESPIKE_COLUMNS if column in out.columns]
        details["spike_counts"] = spike_counts

    if config.butterworth:
        pre_segmentation_columns = [
            column for column in BUTTERWORTH_TARGET_COLUMNS if column not in RIDE_HEIGHT_COLUMNS
        ]
        out = apply_butterworth_filter(
            df=out,
            columns=pre_segmentation_columns,
            fs=fs,
            order=order,
            replace_target_columns=True,
            cutoff_by_column=CHANNEL_CUTOFFS_HZ,
        )
        details["butterworth_columns_pre_segmentation"] = [
            column for column in pre_segmentation_columns if column in out.columns
        ]

    return out, details


def segment_branch(df: pd.DataFrame) -> pd.DataFrame:
    return apply_segmentation_macro_style(df, **SEGMENTATION_KWARGS)


def apply_post_segmentation_ride_height_filter(
    df: pd.DataFrame,
    *,
    config: BranchConfig,
    fs: float,
    order: int,
    filter_details: dict[str, Any],
) -> pd.DataFrame:
    if not config.butterworth:
        return df

    out = apply_butterworth_filter(
        df=df,
        columns=RIDE_HEIGHT_COLUMNS,
        fs=fs,
        order=order,
        replace_target_columns=True,
        cutoff_by_column=CHANNEL_CUTOFFS_HZ,
    )
    filter_details["butterworth_columns_post_segmentation"] = [
        column for column in RIDE_HEIGHT_COLUMNS if column in out.columns
    ]
    return out


def export_corner_outputs(
    segmented_df: pd.DataFrame,
    *,
    branch_dir: Path,
    branch_name: str,
    decimals: int,
    rh_f_bin: float,
    rh_r_bin: float,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for value_column in ["scz_push_f_pitot", "scz_push_r_pitot"]:
        summary = build_corner_summary(
            df=segmented_df,
            segment_label="corner",
            value_column=value_column,
            input_path=branch_dir / f"{branch_name}_segmented.csv",
            segment_column="segment_final",
            segment_id_column="segment_id",
        )
        exact_table = build_exact_two_way_table(summary=summary, value_column=value_column, decimals=decimals)
        binned_table = build_two_way_table(
            summary=summary,
            value_column=value_column,
            rh_f_bin=rh_f_bin,
            rh_r_bin=rh_r_bin,
            decimals=decimals,
        )

        summary_path = branch_dir / f"{branch_name}_{value_column}_corner_summary.csv"
        exact_path = branch_dir / f"{branch_name}_{value_column}_two_way_table_exact.csv"
        binned_path = branch_dir / f"{branch_name}_{value_column}_two_way_table_binned.csv"
        summary.to_csv(summary_path, index=False)
        exact_table.to_csv(exact_path)
        binned_table.to_csv(binned_path)

        outputs[value_column] = {
            "summary_path": str(summary_path),
            "exact_table_path": str(exact_path),
            "binned_table_path": str(binned_path),
            "corner_segment_count": int(len(summary)),
            "rh_f_corner_median_mean": float(summary["rh_f_median"].mean()),
            "rh_f_corner_median_std": float(summary["rh_f_median"].std(ddof=0)),
            "rh_r_corner_median_mean": float(summary["rh_r_median"].mean()),
            "rh_r_corner_median_std": float(summary["rh_r_median"].std(ddof=0)),
            f"{value_column}_corner_median_mean": float(summary[f"{value_column}_median"].mean()),
            f"{value_column}_corner_median_std": float(summary[f"{value_column}_median"].std(ddof=0)),
        }
    return outputs


def segmentation_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["segment_final"].value_counts(dropna=False).to_dict()
    return {str(label): int(count) for label, count in counts.items()}


def summarize_branch(
    branch_name: str,
    filtered_df: pd.DataFrame,
    segmented_df: pd.DataFrame,
    filter_details: dict[str, Any],
    corner_outputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "branch": branch_name,
        "row_count": int(len(filtered_df)),
        "filter_details": filter_details,
        "segment_counts": segmentation_counts(segmented_df),
        "corner_outputs": corner_outputs,
    }


def build_comparison_rows(report_data: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_report = next((item for item in report_data if item["branch"] == "raw"), None)

    for item in report_data:
        row: dict[str, Any] = {
            "branch": item["branch"],
            "row_count": item["row_count"],
            "pit_samples": item["segment_counts"].get("pit", 0),
            "straight_samples": item["segment_counts"].get("straight", 0),
            "corner_samples": item["segment_counts"].get("corner", 0),
            "transition_samples": item["segment_counts"].get("transition", 0),
            "corner_segments_front": item["corner_outputs"]["scz_push_f_pitot"]["corner_segment_count"],
            "corner_segments_rear": item["corner_outputs"]["scz_push_r_pitot"]["corner_segment_count"],
            "rh_f_corner_std_front": item["corner_outputs"]["scz_push_f_pitot"]["rh_f_corner_median_std"],
            "rh_r_corner_std_front": item["corner_outputs"]["scz_push_f_pitot"]["rh_r_corner_median_std"],
            "scz_push_f_corner_std": item["corner_outputs"]["scz_push_f_pitot"]["scz_push_f_pitot_corner_median_std"],
            "scz_push_r_corner_std": item["corner_outputs"]["scz_push_r_pitot"]["scz_push_r_pitot_corner_median_std"],
        }
        if raw_report is not None:
            row["delta_corner_samples_vs_raw"] = row["corner_samples"] - raw_report["segment_counts"].get("corner", 0)
            row["delta_front_corner_segments_vs_raw"] = (
                row["corner_segments_front"]
                - raw_report["corner_outputs"]["scz_push_f_pitot"]["corner_segment_count"]
            )
            row["delta_rear_corner_segments_vs_raw"] = (
                row["corner_segments_rear"]
                - raw_report["corner_outputs"]["scz_push_r_pitot"]["corner_segment_count"]
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline dataset: {args.input}")
    base_df = pd.read_csv(args.input, low_memory=False)
    if args.fs is not None:
        fs = args.fs
        fs_reason = "provided by --fs"
    else:
        fs, fs_reason = infer_sampling_rate_hz(base_df)
    print(f"Sampling rate: {fs:.6f} Hz ({fs_reason})")
    print("Experiment policy: filtering is applied after dataset build and after Scz recalculation.")
    print("So this experiment tests segmentation/comparison stability, not recalculated-Scz formula changes.")
    print(f"Despike columns: {DESPIKE_COLUMNS}")
    print("Butterworth cutoffs:")
    print("  rh_f/rh_r are filtered after segmentation")
    print(f"  rh_f/rh_r: {RIDE_HEIGHT_BASE_CUTOFF_HZ} Hz")
    for column, cutoff_hz in CHANNEL_CUTOFFS_HZ.items():
        print(f"  {column}: {cutoff_hz} Hz")
    print("  no Butterworth: pair")

    report_data: list[dict[str, Any]] = []

    for branch in BRANCHES:
        branch_dir = args.output_dir / branch.name
        branch_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[filter-experiment] Processing branch: {branch.name}")
        filtered_df, filter_details = apply_branch_filters(
            df=base_df,
            config=branch,
            cutoff_hz=args.cutoff_hz,
            fs=fs,
            order=args.order,
            despike_window=args.despike_window,
            despike_mad_k=args.despike_mad_k,
        )
        filtered_path = branch_dir / f"{branch.name}_dataset.csv"
        segmented_df = segment_branch(filtered_df)
        segmented_df = apply_post_segmentation_ride_height_filter(
            segmented_df,
            config=branch,
            fs=fs,
            order=args.order,
            filter_details=filter_details,
        )
        for column in RIDE_HEIGHT_COLUMNS:
            if column in filtered_df.columns and column in segmented_df.columns:
                filtered_df[column] = segmented_df[column].to_numpy()
        filtered_df.to_csv(filtered_path, index=False)

        segmented_path = branch_dir / f"{branch.name}_segmented.csv"
        segmented_df.to_csv(segmented_path, index=False)

        corner_outputs = export_corner_outputs(
            segmented_df=segmented_df,
            branch_dir=branch_dir,
            branch_name=branch.name,
            decimals=args.decimals,
            rh_f_bin=args.rh_f_bin,
            rh_r_bin=args.rh_r_bin,
        )
        branch_report = summarize_branch(
            branch_name=branch.name,
            filtered_df=filtered_df,
            segmented_df=segmented_df,
            filter_details=filter_details,
            corner_outputs=corner_outputs,
        )
        branch_report["filtered_dataset_path"] = str(filtered_path)
        branch_report["segmented_dataset_path"] = str(segmented_path)
        report_data.append(branch_report)

    comparison_df = build_comparison_rows(report_data)
    comparison_csv = args.output_dir / "filter_experiment_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    report_payload = {
        "run_number": CURRENT_RUN,
        "input_dataset": str(args.input),
        "sampling_rate_hz": fs,
        "sampling_rate_reason": fs_reason,
        "segmentation_kwargs": SEGMENTATION_KWARGS,
        "despike_columns": DESPIKE_COLUMNS,
        "butterworth_columns": BUTTERWORTH_TARGET_COLUMNS,
        "despike_window": args.despike_window,
        "despike_mad_k": args.despike_mad_k,
        "butterworth_fallback_cutoff_hz": args.cutoff_hz,
        "ride_height_base_cutoff_hz": RIDE_HEIGHT_BASE_CUTOFF_HZ,
        "ride_height_corner_cutoff_hz": RIDE_HEIGHT_CORNER_CUTOFF_HZ,
        "butterworth_channel_cutoffs_hz": CHANNEL_CUTOFFS_HZ,
        "butterworth_policy": {
            "pre_segmentation": [column for column in BUTTERWORTH_TARGET_COLUMNS if column not in RIDE_HEIGHT_COLUMNS],
            "post_segmentation": RIDE_HEIGHT_COLUMNS,
        },
        "butterworth_order": args.order,
        "branches": report_data,
        "comparison_csv": str(comparison_csv),
    }
    report_json = args.output_dir / "filter_experiment_report.json"
    report_json.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nSaved comparison CSV: {comparison_csv}")
    print(f"Saved report JSON: {report_json}")


if __name__ == "__main__":
    main()
