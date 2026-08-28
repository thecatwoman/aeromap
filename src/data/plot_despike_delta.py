import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
        default=Path("data/processed/plots/despike_delta"),
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for column in DESPIKE_COLUMNS:
        despiked_column = f"{column}_despiked"
        if column not in raw_df.columns or despiked_column not in despiked_df.columns:
            print(f"Skipping missing columns for: {column}")
            continue

        raw = pd.to_numeric(raw_df[column], errors="coerce")
        despiked = pd.to_numeric(despiked_df[despiked_column], errors="coerce")
        delta = raw - despiked
        changed_mask = raw.notna() & despiked.notna() & (delta != 0)

        if not changed_mask.any():
            print(f"No changed points for: {column}")
            continue

        x = pd.Series(range(len(raw_df)), index=raw_df.index, dtype="int64")

        plt.figure(figsize=(12, 5))
        plt.scatter(
            x[changed_mask],
            delta[changed_mask],
            s=18,
            alpha=0.75,
        )
        plt.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        plt.title(f"{column}: raw - despiked at changed points ({args.source})")
        plt.xlabel("Global sample index")
        plt.ylabel("raw - despiked")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = args.output_dir / f"{args.source}_{column}_despike_delta.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
