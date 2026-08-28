import argparse
from pathlib import Path

import pandas as pd

from src.data.build_corner_map_table import (
    build_exact_two_way_table,
    build_segment_summary,
    build_two_way_table,
    segment_columns_for_segmenter,
)
from src.run_paths import CURRENT_RUN, processed_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build front and rear SCz sparse tables from selected segment types "
            "in an offset-applied or baseline segmented real dataset."
        ),
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=CURRENT_RUN,
        help="Run number to process.",
    )
    parser.add_argument(
        "--segmenter",
        choices=["threshold", "macroway"],
        default="macroway",
        help="Which segmentation family produced the segmented dataset.",
    )
    parser.add_argument(
        "--amount-source",
        choices=["frozen", "direct", "ml"],
        default="direct",
        help=(
            "Which dataset family to read when --input is not provided: "
            "frozen baseline, direct-offset export, or ML-offset export."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit segmented CSV path.",
    )
    parser.add_argument(
        "--segment-labels",
        nargs="+",
        choices=["corner", "straight", "pit"],
        default=["corner", "straight"],
        help="Segment labels to include in the extracted 3D overlay points.",
    )
    parser.add_argument(
        "--rh-f-bin",
        type=float,
        default=0.5,
        help="Bin size for rh_f table columns.",
    )
    parser.add_argument(
        "--rh-r-bin",
        type=float,
        default=0.5,
        help="Bin size for rh_r table rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/tables/segment_maps"),
        help="Directory where the summary and sparse-table CSVs will be saved.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimals to keep in exported values.",
    )
    return parser.parse_args()


def default_input_path(run_number: int, amount_source: str) -> Path:
    run_dir = processed_run_dir(run_number)
    if amount_source == "frozen":
        return run_dir / "Frozen_Baseline" / "datasets" / (
            f"barcelona_2026_merged_cleaned_RH_run_{run_number}_segmented.csv"
        )
    if amount_source == "ml":
        return run_dir / "Real_Offset_Applied_Baseline_v1_ML" / "datasets" / (
            f"barcelona_2026_merged_cleaned_RH_run_{run_number}_segmented_real_push_offset_applied_ml.csv"
        )
    return run_dir / "Real_Offset_Applied_Baseline_v1" / "datasets" / (
        f"barcelona_2026_merged_cleaned_RH_run_{run_number}_segmented_real_push_offset_applied.csv"
    )


def label_key(segment_labels: list[str]) -> str:
    return "_".join(segment_labels)


def dataset_key(amount_source: str) -> str:
    return {
        "frozen": "frozen_baseline",
        "direct": "direct_offset",
        "ml": "ml_offset",
    }[amount_source]


def export_one(
    df: pd.DataFrame,
    *,
    value_column: str,
    segment_labels: list[str],
    input_path: Path,
    segment_column: str,
    segment_id_column: str,
    rh_f_bin: float,
    rh_r_bin: float,
    decimals: int,
    output_dir: Path,
    segmenter: str,
    run_number: int,
    amount_source: str,
) -> None:
    summary = build_segment_summary(
        df=df,
        segment_labels=segment_labels,
        value_column=value_column,
        input_path=input_path,
        segment_column=segment_column,
        segment_id_column=segment_id_column,
    )
    exact_table = build_exact_two_way_table(summary=summary, value_column=value_column, decimals=decimals)
    table = build_two_way_table(
        summary=summary,
        value_column=value_column,
        rh_f_bin=rh_f_bin,
        rh_r_bin=rh_r_bin,
        decimals=decimals,
    )

    summary_export = summary.copy()
    for column in ["rh_f_median", "rh_r_median", f"{value_column}_median"]:
        summary_export[column] = summary_export[column].round(decimals)
    summary_export["segmenter"] = segmenter
    summary_export["run_number"] = run_number
    summary_export["dataset_key"] = dataset_key(amount_source)

    file_prefix = (
        f"run_{run_number}_{label_key(segment_labels)}_{value_column}_{segmenter}_{dataset_key(amount_source)}"
    )
    summary_path = output_dir / f"{file_prefix}_segment_summary.csv"
    exact_table_path = output_dir / f"{file_prefix}_two_way_table_exact.csv"
    table_path = output_dir / f"{file_prefix}_two_way_table_binned.csv"
    excel_path = output_dir / f"{file_prefix}_exports.xlsx"

    summary_export.to_csv(summary_path, index=False)
    exact_table.to_csv(exact_table_path)
    table.to_csv(table_path)

    try:
        with pd.ExcelWriter(excel_path) as writer:
            summary_export.to_excel(writer, sheet_name="segment_summary", index=False)
            exact_table.to_excel(writer, sheet_name="two_way_exact")
            table.to_excel(writer, sheet_name="two_way_binned")
        print(f"Saved Excel export: {excel_path}")
    except ModuleNotFoundError as exc:
        print(f"Excel export skipped: {exc}")

    print(f"Segments used for {value_column}: {len(summary)}")
    print(f"Saved per-segment summary: {summary_path}")
    print(f"Saved exact sparse two-way table: {exact_table_path}")
    print(f"Saved binned map-style two-way table: {table_path}")


def main() -> None:
    args = parse_args()
    input_path = args.input or default_input_path(args.run_number, args.amount_source)
    segment_column, segment_id_column = segment_columns_for_segmenter(args.segmenter)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    for value_column in ["scz_push_f_pitot", "scz_push_r_pitot"]:
        export_one(
            df=df,
            value_column=value_column,
            segment_labels=args.segment_labels,
            input_path=input_path,
            segment_column=segment_column,
            segment_id_column=segment_id_column,
            rh_f_bin=args.rh_f_bin,
            rh_r_bin=args.rh_r_bin,
            decimals=args.decimals,
            output_dir=args.output_dir,
            segmenter=args.segmenter,
            run_number=args.run_number,
            amount_source=args.amount_source,
        )


if __name__ == "__main__":
    main()
