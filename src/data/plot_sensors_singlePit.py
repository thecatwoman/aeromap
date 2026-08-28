import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file


SENSOR_COLUMNS = [
    "scz_f_map",
    "rh_f",
    "rh_r",
    "pushavg_c",
    "pushavd_c",
    "pusharg_c",
    "pushard_c",
    "tpms_p_fr",
    "tpms_p_rl",
    "damper_fl_art",
    "damper_fr_art",
    "damper_rl_art",
    "damper_rr_art",
    "pitot_c",
    "pair",
]


def get_data_path(source: str):
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
        default=Path("data/processed/plots/full_run_raw"),
        help="Directory where plot images will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading raw full-run data...")
    df = pd.read_csv(get_data_path(args.source), low_memory=False)
    x = pd.Series(range(len(df)), index=df.index, dtype="int64")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Full run samples: {len(df)}")

    for column in SENSOR_COLUMNS:
        if column not in df.columns:
            print(f"Skipping missing column: {column}")
            continue

        y = pd.to_numeric(df[column], errors="coerce")
        mask = x.notna() & y.notna()

        if not mask.any():
            print(f"Skipping empty column: {column}")
            continue

        plt.figure(figsize=(12, 5))
        plt.plot(x[mask], y[mask], linewidth=1.8)
        plt.title(f"{column} vs global sample index over full run ({args.source})")
        plt.xlabel("Global sample index")
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = args.output_dir / f"{args.source}_{column}_full_run.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
