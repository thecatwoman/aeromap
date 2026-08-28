import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.apply_butterworth_filter import BUTTERWORTH_TARGET_COLUMNS
from src.data.apply_despike_filter import DESPIKE_COLUMNS, get_output_path as get_despiked_output_path
from src.data.apply_median_filter import get_input_path


def get_paths(source: str) -> tuple[Path, Path, Path]:
    raw_path = get_input_path(source)
    despiked_path = get_despiked_output_path(raw_path)
    butterworth_path = despiked_path.with_name(
        f"{despiked_path.stem}_butterworth_filtered{despiked_path.suffix}"
    )
    return raw_path, despiked_path, butterworth_path


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
        default=Path("data/processed/plots/raw_despiked_butterworth"),
        help="Directory where plot images will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_path, despiked_path, butterworth_path = get_paths(args.source)
    print(f"Loading raw file: {raw_path}")
    raw_df = pd.read_csv(raw_path, low_memory=False)
    print(f"Loading despiked file: {despiked_path}")
    despiked_df = pd.read_csv(despiked_path, low_memory=False)
    print(f"Loading butterworth file: {butterworth_path}")
    butterworth_df = pd.read_csv(butterworth_path, low_memory=False)

    x = pd.Series(range(len(raw_df)), index=raw_df.index, dtype="int64")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for column in BUTTERWORTH_TARGET_COLUMNS:
        raw_column = column
        despiked_column = (
            f"{column}_despiked" if column in DESPIKE_COLUMNS else None
        )
        butterworth_column = f"{column}_butterworth"

        if raw_column not in raw_df.columns or butterworth_column not in butterworth_df.columns:
            print(f"Skipping missing columns for: {column}")
            continue

        raw = pd.to_numeric(raw_df[raw_column], errors="coerce")
        raw_mask = x.notna() & raw.notna()

        despiked = None
        despiked_mask = None
        if despiked_column is not None and despiked_column in despiked_df.columns:
            despiked = pd.to_numeric(despiked_df[despiked_column], errors="coerce")
            despiked_mask = x.notna() & despiked.notna()

        butterworth = pd.to_numeric(butterworth_df[butterworth_column], errors="coerce")
        butterworth_mask = x.notna() & butterworth.notna()

        if not raw_mask.any() and not butterworth_mask.any():
            print(f"Skipping empty columns for: {column}")
            continue

        plt.figure(figsize=(12, 5))
        if raw_mask.any():
            plt.plot(
                x[raw_mask],
                raw[raw_mask],
                linewidth=1.0,
                alpha=0.65,
                label="raw",
                color="0.45",
            )
        if despiked is not None and despiked_mask is not None and despiked_mask.any():
            plt.plot(
                x[despiked_mask],
                despiked[despiked_mask],
                linewidth=1.4,
                label="despiked",
                color="tab:orange",
            )
        if butterworth_mask.any():
            plt.plot(
                x[butterworth_mask],
                butterworth[butterworth_mask],
                linewidth=1.8,
                label="butterworth",
                color="tab:blue",
            )

        plt.title(f"{column}: raw vs despiked vs butterworth ({args.source})")
        plt.xlabel("Global sample index")
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        output_path = (
            args.output_dir / f"{args.source}_{column}_raw_despiked_butterworth.png"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
