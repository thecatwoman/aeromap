import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file


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
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(get_data_path(args.source))

    if "segment_final" not in df.columns:
        raise ValueError("Missing 'segment_final'. Run segmentation first.")

    pit_df = df[df["segment_final"] == "pit"].copy()
    if pit_df.empty:
        raise ValueError(
            f"No 'pit' rows found in the segmented {args.source} dataset."
        )

    pit_df["group"] = (pit_df.index.to_series().diff() != 1).cumsum()
    first_pit = pit_df[pit_df["group"] == pit_df["group"].iloc[0]].copy()

    first_pit["speed_bin"] = first_pit["carspeed_art"].round(1)
    binned = first_pit.groupby("speed_bin")["rh_f"].median().reset_index()

    plt.figure(figsize=(10, 6))
    plt.scatter(binned["speed_bin"], binned["rh_f"])
    plt.xlabel("Speed bin (km/h)")
    plt.ylabel("Median rh_f")
    plt.title(f"Binned rh_f - First Pit ({args.source})")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
