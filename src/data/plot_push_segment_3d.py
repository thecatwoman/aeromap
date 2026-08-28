import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.run_paths import CURRENT_RUN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SCz sparse tables built from selected segment types on the 3D reference surfaces.",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=CURRENT_RUN,
        help="Run number used in the exported sparse table names.",
    )
    parser.add_argument(
        "--side",
        choices=["front", "rear", "both"],
        default="both",
        help="Which SCz map to plot.",
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
        help="Which dataset family the sparse table came from.",
    )
    parser.add_argument(
        "--compare-amount-source",
        choices=["frozen", "direct", "ml"],
        default=None,
        help="Optional second dataset family to overlay for comparison on the same plot.",
    )
    parser.add_argument(
        "--segment-labels",
        nargs="+",
        choices=["corner", "straight", "pit"],
        default=["corner", "straight"],
        help="Segment labels included in the sparse table.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("data/processed/tables/segment_maps"),
        help="Directory containing the exported sparse tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/plots/reference_table"),
        help="Directory where the PNG and HTML files will be saved.",
    )
    parser.add_argument(
        "--surface-x-shift",
        type=float,
        default=-5.0,
        help="Shift applied to the reference surface X values.",
    )
    parser.add_argument(
        "--surface-y-shift",
        type=float,
        default=-5.0,
        help="Shift applied to the reference surface Y values.",
    )
    parser.add_argument(
        "--overlay-x-shift",
        type=float,
        default=-5.0,
        help="Shift applied to the overlay-table X values.",
    )
    parser.add_argument(
        "--overlay-y-shift",
        type=float,
        default=-5.0,
        help="Shift applied to the overlay-table Y values.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=28.0,
        help="3D camera elevation angle.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-135.0,
        help="3D camera azimuth angle.",
    )
    return parser.parse_args()


def label_key(segment_labels: list[str]) -> str:
    return "_".join(segment_labels)


def dataset_key(amount_source: str) -> str:
    return {
        "frozen": "frozen_baseline",
        "direct": "direct_offset",
        "ml": "ml_offset",
    }[amount_source]


def overlay_label(amount_source: str) -> str:
    if amount_source == "ml":
        return "straight-only from offseted + SCz-recalculated dataset"
    if amount_source == "direct":
        return "straight-only from offseted + SCz-recalculated dataset"
    return "straight-only from pre-offset dataset"


def overlay_table_path(run_number: int, side: str, segmenter: str, amount_source: str, segment_labels: list[str], table_dir: Path) -> Path:
    value_column = "scz_push_f_pitot" if side == "front" else "scz_push_r_pitot"
    filename = (
        f"run_{run_number}_{label_key(segment_labels)}_{value_column}_{segmenter}_{dataset_key(amount_source)}"
        "_two_way_table_exact.csv"
    )
    return table_dir / filename


def plot_module_name(side: str) -> str:
    return "src.data.plot_reference_table_3d" if side == "front" else "src.data.plot_reference_table_3d_scz_r"


def output_stem(run_number: int, side: str, segmenter: str, amount_source: str, segment_labels: list[str]) -> str:
    value_column = "scz_push_f_pitot" if side == "front" else "scz_push_r_pitot"
    return (
        f"run_{run_number}_{label_key(segment_labels)}_{value_column}_{segmenter}_{dataset_key(amount_source)}"
        "_overlay_shifted_axes"
    )


def run_one(side: str, args: argparse.Namespace) -> None:
    table_path = overlay_table_path(
        args.run_number,
        side,
        args.segmenter,
        args.amount_source,
        args.segment_labels,
        args.table_dir,
    )
    if not table_path.exists():
        raise FileNotFoundError(
            f"Missing overlay table for {side}: {table_path}\n"
            "Build the segment tables first with:\n"
            f"  .venv/bin/python -m src.data.build_push_segment_tables --run-number {args.run_number} "
            f"--amount-source {args.amount_source} --segment-labels {' '.join(args.segment_labels)}"
        )

    compare_table_path = None
    if args.compare_amount_source is not None:
        compare_table_path = overlay_table_path(
            args.run_number,
            side,
            args.segmenter,
            args.compare_amount_source,
            args.segment_labels,
            args.table_dir,
        )
        if not compare_table_path.exists():
            raise FileNotFoundError(
                f"Missing comparison overlay table for {side}: {compare_table_path}\n"
                "Build the comparison segment tables first with:\n"
                f"  .venv/bin/python -m src.data.build_push_segment_tables --run-number {args.run_number} "
                f"--amount-source {args.compare_amount_source} --segment-labels {' '.join(args.segment_labels)}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args.run_number, side, args.segmenter, args.amount_source, args.segment_labels)
    png_path = args.output_dir / f"{stem}.png"
    html_path = args.output_dir / f"{stem}.html"

    cmd = [
        sys.executable,
        "-m",
        plot_module_name(side),
        "--overlay",
        "--no-show",
        "--overlay-table",
        str(table_path),
        "--overlay-label",
        overlay_label(args.amount_source),
        "--output",
        str(png_path),
        "--html",
        str(html_path),
        "--surface-x-shift",
        str(args.surface_x_shift),
        "--surface-y-shift",
        str(args.surface_y_shift),
        "--overlay-x-shift",
        str(args.overlay_x_shift),
        "--overlay-y-shift",
        str(args.overlay_y_shift),
        "--elev",
        str(args.elev),
        "--azim",
        str(args.azim),
    ]
    if compare_table_path is not None and args.compare_amount_source is not None:
        cmd.extend(
            [
                "--compare-overlay-table",
                str(compare_table_path),
                "--compare-overlay-label",
                overlay_label(args.compare_amount_source),
            ]
        )

    print(f"Plotting {side} map from: {table_path}")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    sides = ["front", "rear"] if args.side == "both" else [args.side]
    for side in sides:
        run_one(side, args)


if __name__ == "__main__":
    main()
