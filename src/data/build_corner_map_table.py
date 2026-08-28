import argparse
from pathlib import Path

import pandas as pd

from src.run_paths import CURRENT_RUN, processed_run_dir, segmented_rh_run_file


def threshold_segmented_run_file(run_number: int) -> Path:
    return processed_run_dir(run_number) / "Segmented" / (
        f"barcelona_2026_merged_cleaned_macro_v3_run_{run_number}_segmented.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a corner-based two-way table from a segmented real dataset.",
    )
    parser.add_argument(
        "--segmenter",
        choices=["threshold", "macroway"],
        default="threshold",
        help="Which segmentation output format to use.",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=CURRENT_RUN,
        help="Run number to read segmented data from and to include in output file names.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the segmented real CSV. Defaults depend on --segmenter.",
    )
    parser.add_argument(
        "--value-column",
        default="scz_f_map",
        help="Column to use as the table cell value (for example scz_f_map or scz_r_map).",
    )
    parser.add_argument(
        "--segment-label",
        default="corner",
        help="Segment label to extract. Defaults to corner.",
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
        help="Number of decimals to keep in exported medians and table headers.",
    )
    return parser.parse_args()


def default_input_for_segmenter(segmenter: str, run_number: int) -> Path:
    if segmenter == "macroway":
        return segmented_rh_run_file(run_number)
    return threshold_segmented_run_file(run_number)


def segment_columns_for_segmenter(segmenter: str) -> tuple[str, str]:
    if segmenter == "macroway":
        return "segment_final", "segment_id"
    return "segment_macro_v3", "segment_id_macro_v3"


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def round_to_step(values: pd.Series, step: float) -> pd.Series:
    if step <= 0:
        raise ValueError("Bin step must be greater than 0.")
    return (values / step).round() * step


def build_exact_two_way_table(summary: pd.DataFrame, value_column: str, decimals: int) -> pd.DataFrame:
    value_median_column = f"{value_column}_median"
    work = summary.copy()
    work["rh_f_header"] = work["rh_f_median"].round(decimals)
    work["rh_r_header"] = work["rh_r_median"].round(decimals)
    work[value_median_column] = work[value_median_column].round(decimals)

    table = work.pivot_table(
        index="rh_r_header",
        columns="rh_f_header",
        values=value_median_column,
        aggfunc="median",
    )
    table = table.sort_index().sort_index(axis=1)
    table.index.name = "rh_r"
    table.columns.name = "rh_f"
    return table


def build_segment_summary(
    df: pd.DataFrame,
    segment_labels: list[str],
    value_column: str,
    input_path: Path,
    segment_column: str,
    segment_id_column: str,
) -> pd.DataFrame:
    require_columns(
        df,
        [segment_column, segment_id_column, "rh_f", "rh_r", value_column],
    )

    if not segment_labels:
        raise ValueError("segment_labels must contain at least one label.")

    normalized_labels = [str(label) for label in segment_labels]
    segments = df[df[segment_column].astype("string").isin(normalized_labels)].copy()
    if segments.empty:
        labels_text = ", ".join(repr(label) for label in normalized_labels)
        raise ValueError(f"No rows found for segment labels [{labels_text}] in {input_path}.")

    rows: list[dict[str, float | int]] = []
    for segment_id, group in segments.groupby(segment_id_column, sort=True):
        rh_f = pd.to_numeric(group["rh_f"], errors="coerce")
        rh_r = pd.to_numeric(group["rh_r"], errors="coerce")
        value = pd.to_numeric(group[value_column], errors="coerce")
        segment_names = group[segment_column].astype("string").dropna().unique().tolist()

        if rh_f.dropna().empty or rh_r.dropna().empty or value.dropna().empty:
            continue

        rows.append(
            {
                segment_id_column: int(segment_id),
                segment_column: str(segment_names[0]) if segment_names else "",
                "sample_count": int(len(group)),
                "rh_f_median": float(rh_f.median()),
                "rh_r_median": float(rh_r.median()),
                f"{value_column}_median": float(value.median()),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No valid segments had usable rh_f, rh_r, and value-column medians.")
    return summary.sort_values(segment_id_column).reset_index(drop=True)


def build_corner_summary(
    df: pd.DataFrame,
    segment_label: str,
    value_column: str,
    input_path: Path,
    segment_column: str,
    segment_id_column: str,
) -> pd.DataFrame:
    return build_segment_summary(
        df=df,
        segment_labels=[segment_label],
        value_column=value_column,
        input_path=input_path,
        segment_column=segment_column,
        segment_id_column=segment_id_column,
    )


def build_two_way_table(
    summary: pd.DataFrame,
    value_column: str,
    rh_f_bin: float,
    rh_r_bin: float,
    decimals: int,
) -> pd.DataFrame:
    value_median_column = f"{value_column}_median"
    work = summary.copy()
    work["rh_f_bin"] = round_to_step(work["rh_f_median"], rh_f_bin)
    work["rh_r_bin"] = round_to_step(work["rh_r_median"], rh_r_bin)
    work[value_median_column] = work[value_median_column].round(decimals)

    table = work.pivot_table(
        index="rh_r_bin",
        columns="rh_f_bin",
        values=value_median_column,
        aggfunc="median",
    )
    table = table.sort_index().sort_index(axis=1)
    table.index = table.index.round(decimals)
    table.columns = table.columns.round(decimals)
    table.index.name = "rh_r"
    table.columns.name = "rh_f"
    return table


def main() -> None:
    args = parse_args()
    input_path = args.input or default_input_for_segmenter(args.segmenter, args.run_number)
    segment_column, segment_id_column = segment_columns_for_segmenter(args.segmenter)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    summary = build_corner_summary(
        df=df,
        segment_label=args.segment_label,
        value_column=args.value_column,
        input_path=input_path,
        segment_column=segment_column,
        segment_id_column=segment_id_column,
    )
    exact_table = build_exact_two_way_table(
        summary=summary,
        value_column=args.value_column,
        decimals=args.decimals,
    )
    table = build_two_way_table(
        summary=summary,
        value_column=args.value_column,
        rh_f_bin=args.rh_f_bin,
        rh_r_bin=args.rh_r_bin,
        decimals=args.decimals,
    )
    summary_export = summary.copy()
    for column in ["rh_f_median", "rh_r_median", f"{args.value_column}_median"]:
        summary_export[column] = summary_export[column].round(args.decimals)
    summary_export["segmenter"] = args.segmenter
    summary_export["run_number"] = args.run_number

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"run_{args.run_number}_{args.segment_label}_{args.value_column}_{args.segmenter}"
    summary_path = args.output_dir / f"{file_prefix}_corner_summary.csv"
    exact_table_path = args.output_dir / f"{file_prefix}_two_way_table_exact.csv"
    table_path = args.output_dir / f"{file_prefix}_two_way_table_binned.csv"
    excel_path = args.output_dir / f"{file_prefix}_exports.xlsx"

    summary_export.to_csv(summary_path, index=False)
    exact_table.to_csv(exact_table_path)
    table.to_csv(table_path)
    excel_saved = False
    try:
        with pd.ExcelWriter(excel_path) as writer:
            summary_export.to_excel(writer, sheet_name="corner_summary", index=False)
            exact_table.to_excel(writer, sheet_name="two_way_exact")
            table.to_excel(writer, sheet_name="two_way_binned")
        excel_saved = True
    except ModuleNotFoundError as exc:
        print(f"Excel export skipped: {exc}")

    print(f"Segments used: {len(summary)}")
    print(f"Saved per-corner summary: {summary_path}")
    print(f"Saved exact sparse two-way table: {exact_table_path}")
    print(f"Saved binned map-style two-way table: {table_path}")
    if excel_saved:
        print(f"Saved Excel export: {excel_path}")


if __name__ == "__main__":
    main()
