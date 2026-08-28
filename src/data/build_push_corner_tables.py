import argparse
from pathlib import Path

import pandas as pd

from src.data.build_corner_map_table import (
    build_corner_summary,
    build_exact_two_way_table,
    build_two_way_table,
    default_input_for_segmenter,
    segment_columns_for_segmenter,
)
from src.run_paths import CURRENT_RUN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build front and rear push-based corner tables for a given run.",
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
        help="Which segmented dataset to use.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit segmented CSV path. If provided, used for both front and rear tables.",
    )
    parser.add_argument(
        "--segment-label",
        default="corner",
        help="Segment label to extract.",
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
        default=Path("data/processed/tables/corner_maps"),
        help="Directory where the corner summary and two-way table CSVs will be saved.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimals to keep in exported values.",
    )
    return parser.parse_args()


def export_one(
    df: pd.DataFrame,
    *,
    value_column: str,
    segment_label: str,
    input_path: Path,
    segment_column: str,
    segment_id_column: str,
    rh_f_bin: float,
    rh_r_bin: float,
    decimals: int,
    output_dir: Path,
    segmenter: str,
    run_number: int,
) -> None:
    summary = build_corner_summary(
        df=df,
        segment_label=segment_label,
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

    file_prefix = f"run_{run_number}_{segment_label}_{value_column}_{segmenter}"
    summary_path = output_dir / f"{file_prefix}_corner_summary.csv"
    exact_table_path = output_dir / f"{file_prefix}_two_way_table_exact.csv"
    table_path = output_dir / f"{file_prefix}_two_way_table_binned.csv"
    excel_path = output_dir / f"{file_prefix}_exports.xlsx"

    summary_export.to_csv(summary_path, index=False)
    exact_table.to_csv(exact_table_path)
    table.to_csv(table_path)

    try:
        with pd.ExcelWriter(excel_path) as writer:
            summary_export.to_excel(writer, sheet_name="corner_summary", index=False)
            exact_table.to_excel(writer, sheet_name="two_way_exact")
            table.to_excel(writer, sheet_name="two_way_binned")
        print(f"Saved Excel export: {excel_path}")
    except ModuleNotFoundError as exc:
        print(f"Excel export skipped: {exc}")

    print(f"Segments used for {value_column}: {len(summary)}")
    print(f"Saved per-corner summary: {summary_path}")
    print(f"Saved exact sparse two-way table: {exact_table_path}")
    print(f"Saved binned map-style two-way table: {table_path}")


def main() -> None:
    args = parse_args()
    input_path = args.input or default_input_for_segmenter(args.segmenter, args.run_number)
    segment_column, segment_id_column = segment_columns_for_segmenter(args.segmenter)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    for value_column in ["scz_push_f_pitot", "scz_push_r_pitot"]:
        export_one(
            df=df,
            value_column=value_column,
            segment_label=args.segment_label,
            input_path=input_path,
            segment_column=segment_column,
            segment_id_column=segment_id_column,
            rh_f_bin=args.rh_f_bin,
            rh_r_bin=args.rh_r_bin,
            decimals=args.decimals,
            output_dir=args.output_dir,
            segmenter=args.segmenter,
            run_number=args.run_number,
        )


if __name__ == "__main__":
    main()
