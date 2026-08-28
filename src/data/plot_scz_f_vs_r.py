import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file


def get_data_path(source: str) -> Path:
    if source == "simu":
        return segmented_simudata_run_file()
    return segmented_rh_run_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["real", "simu"],
        default="real",
        help="Dataset source to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/plots/scz_maps"),
        help="Directory where the plot image will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading segmented data...")
    df = pd.read_csv(get_data_path(args.source), low_memory=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required = ["scz_f_map", "scz_r_map"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = pd.to_numeric(df["scz_f_map"], errors="coerce")
    y = pd.to_numeric(df["scz_r_map"], errors="coerce")
    mask = x.notna() & y.notna()
    if not mask.any():
        raise ValueError("No valid scz_f_map / scz_r_map points to plot.")

    plt.figure(figsize=(7, 6))
    plt.scatter(x[mask], y[mask], s=10, alpha=0.65)
    plt.title(f"scz_f_map vs scz_r_map ({args.source})")
    plt.xlabel("scz_f_map")
    plt.ylabel("scz_r_map")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = args.output_dir / f"{args.source}_scz_f_vs_scz_r.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
