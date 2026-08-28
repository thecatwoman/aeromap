import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file

DATA_PATH = segmented_rh_run_file()
SIM_DATA_PATH = segmented_simudata_run_file()


def get_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing column. Tried: {candidates}")


def load_trace_columns(data_path: Path) -> tuple[pd.Series, pd.Series, str]:
    df = pd.read_csv(data_path, low_memory=False)
    speed_col = get_column(df, ["carspeed_art", "speed", "vehicle_speed"])
    distance_col = get_column(df, ["distancelap", "distance"])

    speed = pd.to_numeric(df[speed_col], errors="coerce")
    distance = pd.to_numeric(df[distance_col], errors="coerce")
    mask = speed.notna() & distance.notna()
    return distance[mask], speed[mask], distance_col


def visualize_segmented_data(data_path: Path = DATA_PATH, title: str = "Macro Segmentation - Continuous Trace"):
    print("Loading data...")
    df = pd.read_csv(data_path, low_memory=False)

    speed_col = get_column(df, ["carspeed_art", "speed", "vehicle_speed"])
    lap_col = get_column(df, ["laps_count", "lap", "lap_number"])
    if "segment_final" not in df.columns:
        raise ValueError("Missing 'segment_final'")

    laps = sorted(df[lap_col].dropna().unique())

    color_map = {
        "straight": "lime",
        "corner": "red",
        "pit": "magenta",
        "transition": "white",
    }

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 6))

    global_sample_offset = 0

    for lap in laps:
        lap_df = df[df[lap_col] == lap].copy()

        lap_df[speed_col] = pd.to_numeric(lap_df[speed_col], errors="coerce")
        lap_df = lap_df.dropna(subset=[speed_col, "segment_final"]).copy()

        if lap_df.empty:
            continue

        lap_df = lap_df.reset_index(drop=True)
        lap_df["global_sample"] = lap_df.index + global_sample_offset
        global_sample_offset = int(lap_df["global_sample"].iloc[-1]) + 1

        if "segment_id" in lap_df.columns:
            lap_df["run_id"] = lap_df["segment_id"]
        else:
            lap_df["run_id"] = (
                lap_df["segment_final"] != lap_df["segment_final"].shift()
            ).cumsum()

        for _, run_df in lap_df.groupby("run_id"):
            seg = run_df["segment_final"].iloc[0]
            color = color_map.get(seg, "white")

            ax.plot(
                run_df["global_sample"],
                run_df[speed_col],
                color=color,
                linewidth=2.0,
            )

    ax.set_title(title)
    ax.set_xlabel("global_sample")
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.25)

    # legend
    for seg, color in color_map.items():
        ax.plot([], [], color=color, linewidth=4, label=seg)

    ax.legend()
    plt.show()


def visualize_overlay(
    real_data_path: Path,
    sim_data_path: Path,
    title: str = "Real vs Simulation Speed Overlay",
) -> None:
    print("Loading real data...")
    real_distance, real_speed, real_distance_col = load_trace_columns(real_data_path)

    print("Loading simulation data...")
    sim_distance, sim_speed, sim_distance_col = load_trace_columns(sim_data_path)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(
        real_distance,
        real_speed,
        color="deepskyblue",
        linewidth=1.8,
        alpha=0.9,
        label="real",
    )
    ax.plot(
        sim_distance,
        sim_speed,
        color="orange",
        linewidth=1.8,
        alpha=0.85,
        linestyle="--",
        label="simu",
    )

    xlabel = real_distance_col if real_distance_col == sim_distance_col else (
        f"{real_distance_col} / {sim_distance_col}"
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to the segmented CSV file to visualize.",
    )
    parser.add_argument(
        "--title",
        default="Macro Segmentation - Continuous Trace",
        help="Plot title.",
    )
    parser.add_argument(
        "--overlay-sim",
        action="store_true",
        help="Overlay simulation speed-vs-distance trace on top of the real one.",
    )
    parser.add_argument(
        "--sim-data-path",
        type=Path,
        default=SIM_DATA_PATH,
        help="Path to the segmented simulation CSV file for overlay mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.overlay_sim:
        visualize_overlay(
            real_data_path=args.data_path,
            sim_data_path=args.sim_data_path,
            title=args.title,
        )
        return

    visualize_segmented_data(data_path=args.data_path, title=args.title)


if __name__ == "__main__":
    main()
