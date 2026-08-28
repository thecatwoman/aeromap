import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.apply_despike_filter import DESPIKE_COLUMNS
from src.run_paths import cleaned_merged_full_run_file, cleaned_simudata_full_run_file


def get_base_path(source: str) -> Path:
    if source == "simu":
        return cleaned_simudata_full_run_file()
    return cleaned_merged_full_run_file()


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
        default=Path("data/processed/plots/raw_vs_despiked"),
        help="Directory where plot images will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_path = get_base_path(args.source)
    despiked_path = base_path.with_name(f"{base_path.stem}_despiked{base_path.suffix}")

    print(f"Loading raw file: {base_path}")
    raw_df = pd.read_csv(base_path, low_memory=False)
    print(f"Loading despiked file: {despiked_path}")
    despiked_df = pd.read_csv(despiked_path, low_memory=False)
    x = pd.Series(range(len(raw_df)), index=raw_df.index, dtype="int64")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for column in DESPIKE_COLUMNS:
        despiked_column = f"{column}_despiked"
        if column not in raw_df.columns or despiked_column not in despiked_df.columns:
            print(f"Skipping missing columns for: {column}")
            continue

        raw = pd.to_numeric(raw_df[column], errors="coerce")
        despiked = pd.to_numeric(despiked_df[despiked_column], errors="coerce")
        raw_mask = x.notna() & raw.notna()
        despiked_mask = x.notna() & despiked.notna()

        if not raw_mask.any() and not despiked_mask.any():
            print(f"Skipping empty columns for: {column}")
            continue

        raw_values = raw.to_numpy()
        despiked_values = despiked.to_numpy()
        changed = int(np.count_nonzero(raw_values != despiked_values))
        pct = 100.0 * changed / len(raw_values) if len(raw_values) else 0.0
        print(f"{column}: changed={changed}, pct={pct:.3f}%")

        plt.figure(figsize=(12, 5))
        if raw_mask.any():
            plt.plot(
                x[raw_mask],
                raw[raw_mask],
                linewidth=1.0,
                alpha=0.7,
                label="raw",
                color="0.45",
            )
        if despiked_mask.any():
            plt.plot(
                x[despiked_mask],
                despiked[despiked_mask],
                linewidth=1.8,
                label="despiked",
                color="tab:orange",
            )

        plt.title(f"{column}: raw vs despiked ({args.source})")
        plt.xlabel("Global sample index")
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.text(
            0.99,
            0.98,
            f"changed={changed}\npct={pct:.3f}%",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        plt.tight_layout()

        output_path = args.output_dir / f"{args.source}_{column}_raw_vs_despiked.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
