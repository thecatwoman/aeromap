import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.run_paths import CURRENT_RUN, raw_run_dir


TPMS_COLUMNS = [
    "tpms_p_fl",
    "tpms_p_fr",
    "tpms_p_rl",
    "tpms_p_rr",
]


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def extract_lap_number(filename: str) -> int:
    match = re.search(r"lap(\d+)", filename.lower())
    if match is None:
        raise ValueError(f"Could not extract lap number from {filename}")
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-number",
        type=int,
        default=CURRENT_RUN,
        help="Run number to read from the raw data folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/plots/raw_tpms"),
        help="Directory where TPMS plot images will be saved.",
    )
    return parser.parse_args()


def load_raw_run(run_number: int) -> pd.DataFrame:
    run_dir = raw_run_dir(run_number)
    csv_paths = sorted(
        run_dir.glob("*.csv"),
        key=lambda path: extract_lap_number(path.name),
    )

    if not csv_paths:
        raise ValueError(f"No raw CSV files found in {run_dir}")

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [normalize_column_name(col) for col in df.columns]
        df["lap"] = extract_lap_number(path.name)
        df["source_file"] = path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()

    print(f"Loading raw TPMS data from Run_{args.run_number}...")
    df = load_raw_run(args.run_number)
    x = pd.Series(range(len(df)), index=df.index, dtype="int64")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Full raw samples: {len(df)}")
    print(f"Laps found: {sorted(df['lap'].dropna().unique().tolist())}")

    for column in TPMS_COLUMNS:
        if column not in df.columns:
            print(f"Skipping missing column: {column}")
            continue

        y = pd.to_numeric(df[column], errors="coerce")
        mask = x.notna() & y.notna()

        if not mask.any():
            print(f"Skipping empty column: {column}")
            continue

        plt.figure(figsize=(12, 5))
        plt.plot(x[mask], y[mask], linewidth=1.5)
        plt.title(f"{column} vs global sample index over raw run {args.run_number}")
        plt.xlabel("Global sample index")
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = args.output_dir / f"run_{args.run_number}_{column}_raw.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
