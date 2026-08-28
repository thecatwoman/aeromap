import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.apply_despike_filter import DESPIKE_COLUMNS
from src.run_paths import cleaned_merged_full_run_file, cleaned_simudata_full_run_file


def get_input_path(source: str) -> Path:
    if source == "simu":
        base_path = cleaned_simudata_full_run_file()
    else:
        base_path = cleaned_merged_full_run_file()
    return base_path.with_name(f"{base_path.stem}_despiked{base_path.suffix}")


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
        default=Path("data/processed/plots/full_run_despiked"),
        help="Directory where plot images will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = get_input_path(args.source)
    print(f"Loading despiked full-run data: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    x = pd.Series(range(len(df)), index=df.index, dtype="int64")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Full run samples: {len(df)}")

    for column in DESPIKE_COLUMNS:
        despiked_column = f"{column}_despiked"
        if despiked_column not in df.columns:
            print(f"Skipping missing column: {despiked_column}")
            continue

        y = pd.to_numeric(df[despiked_column], errors="coerce")
        mask = x.notna() & y.notna()

        if not mask.any():
            print(f"Skipping empty column: {despiked_column}")
            continue

        plt.figure(figsize=(12, 5))
        plt.plot(x[mask], y[mask], linewidth=1.8)
        plt.title(
            f"{despiked_column} vs global sample index over full run ({args.source})"
        )
        plt.xlabel("Global sample index")
        plt.ylabel(despiked_column)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = args.output_dir / f"{args.source}_{despiked_column}_full_run.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
