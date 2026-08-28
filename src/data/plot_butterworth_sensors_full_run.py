import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.apply_butterworth_filter import BUTTERWORTH_TARGET_COLUMNS
from src.data.apply_despike_filter import get_output_path as get_despiked_output_path
from src.data.apply_median_filter import get_input_path


def get_butterworth_input_path(source: str) -> Path:
    base_path = get_input_path(source)
    despiked_path = get_despiked_output_path(base_path)
    return despiked_path.with_name(
        f"{despiked_path.stem}_butterworth_filtered{despiked_path.suffix}"
    )


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
        default=Path("data/processed/plots/full_run_butterworth"),
        help="Directory where plot images will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = get_butterworth_input_path(args.source)
    print(f"Loading butterworth full-run data: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    x = pd.Series(range(len(df)), index=df.index, dtype="int64")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Full run samples: {len(df)}")

    for column in BUTTERWORTH_TARGET_COLUMNS:
        filtered_column = f"{column}_butterworth"
        if filtered_column not in df.columns:
            print(f"Skipping missing column: {filtered_column}")
            continue

        y = pd.to_numeric(df[filtered_column], errors="coerce")
        mask = x.notna() & y.notna()

        if not mask.any():
            print(f"Skipping empty column: {filtered_column}")
            continue

        plt.figure(figsize=(12, 5))
        plt.plot(x[mask], y[mask], linewidth=1.8)
        plt.title(
            f"{filtered_column} vs global sample index over full run ({args.source})"
        )
        plt.xlabel("Global sample index")
        plt.ylabel(filtered_column)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = args.output_dir / f"{args.source}_{filtered_column}_full_run.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
