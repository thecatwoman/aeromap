from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.run_paths import segmented_rh_run_file


DATA_PATH = segmented_rh_run_file()

SENSOR_COLUMNS = [
    #"hdcav_c",
    #"hdcar_c",
    "rh_f",
    "rh_r",
    "pushavg_c",
    "pushavd_c",
    "pusharg_c",
    "pushard_c",
    #"fz_push_rl",
    #"fz_push_rr",
    #"fz_push_fl",
    #"fz_push_fr",
]


def load_segmented_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def get_segments(df: pd.DataFrame, label: str) -> list[pd.DataFrame]:
    if "segment_final" not in df.columns:
        raise ValueError("Missing 'segment_final'. Run segmentation first.")

    segment_df = df[df["segment_final"] == label].copy()
    if segment_df.empty:
        raise ValueError(f"No '{label}' rows found in the segmented dataset.")

    segment_df["group"] = (segment_df.index.to_series().diff() != 1).cumsum()

    segments: list[pd.DataFrame] = []
    for _, group_df in segment_df.groupby("group"):
        clean_group = group_df.drop(columns=["group"]).copy()
        segments.append(clean_group)

    return segments


def plot_segments(label: str) -> None:
    print("Loading segmented data...")
    df = load_segmented_data()
    segments = get_segments(df, label)

    print(f"Found {len(segments)} {label} segments")

    for segment_number, segment_df in [(3, segments[0])]:
        x = segment_df["time"]
        print(f"{label.capitalize()} segment {segment_number}: {len(segment_df)} samples")

        for column in SENSOR_COLUMNS:
            if column not in segment_df.columns:
                print(f"Skipping missing column: {column}")
                continue

            y = pd.to_numeric(segment_df[column], errors="coerce")
            mask = x.notna() & y.notna()

            if not mask.any():
                print(f"Skipping empty column: {column}")
                continue

            plt.figure(figsize=(12, 5))
            plt.plot(x[mask], y[mask], linewidth=1.8)
            plt.title(f"{column} vs time in {label} segment {segment_number}")
            plt.xlabel("Time")
            plt.ylabel(column)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
