import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.run_paths import CURRENT_RUN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot push-based corner tables on the reference 3D surfaces.",
    )
    parser.add_argument(
        "--side",
        choices=["front", "rear", "both"],
        default="both",
        help="Which push map to plot.",
    )
    parser.add_argument(
        "--segmenter",
        choices=["macroway", "threshold"],
        default="macroway",
        help="Which segmentation family produced the corner table.",
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
        help="Shift applied to the corner-table X values.",
    )
    parser.add_argument(
        "--overlay-y-shift",
        type=float,
        default=-5.0,
        help="Shift applied to the corner-table Y values.",
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


def overlay_table_path(side: str, segmenter: str) -> Path:
    value_column = "scz_push_f_pitot" if side == "front" else "scz_push_r_pitot"
    return Path("data/processed/tables/corner_maps") / (
        f"run_{CURRENT_RUN}_corner_{value_column}_{segmenter}_two_way_table_exact.csv"
    )


def plot_module_name(side: str) -> str:
    return "src.data.plot_reference_table_3d" if side == "front" else "src.data.plot_reference_table_3d_scz_r"


def output_stem(side: str, segmenter: str) -> str:
    value_column = "scz_push_f_pitot" if side == "front" else "scz_push_r_pitot"
    return f"run_{CURRENT_RUN}_{value_column}_{segmenter}_overlay_shifted_axes"


def run_one(side: str, args: argparse.Namespace) -> None:
    table_path = overlay_table_path(side, args.segmenter)
    if not table_path.exists():
        raise FileNotFoundError(
            f"Missing overlay table for {side}: {table_path}\n"
            "Build the corner tables first with:\n"
            "  .venv/bin/python -m src.data.build_push_corner_tables --segmenter macroway"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(side, args.segmenter)
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

    print(f"Plotting {side} push map from: {table_path}")
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
