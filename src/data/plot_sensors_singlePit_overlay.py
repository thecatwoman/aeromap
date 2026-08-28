import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file


SENSOR_COLUMNS = [
    "hdcav_c",
    "hdcar_c",
    "rh_f",
    "rh_r",
    #"pushavg_c",
    #"pushavd_c",
    #"pusharg_c",
    #"pushard_c",
    #"fz_push_rl",
    #"fz_push_rr",
    #"fz_push_fl",
    #"fz_push_fr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x-mode",
        choices=["normalized", "sample"],
        default="normalized",
        help="How to align the two pit traces on the x-axis.",
    )
    return parser.parse_args()


def load_segmented_data(source: str) -> pd.DataFrame:
    if source == "simu":
        return pd.read_csv(segmented_simudata_run_file(), low_memory=False)
    return pd.read_csv(segmented_rh_run_file(), low_memory=False)


def get_first_pit_segment(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "segment_final" not in df.columns:
        raise ValueError(f"Missing 'segment_final' in {source} data.")

    pit_df = df[df["segment_final"] == "pit"].copy()
    if pit_df.empty:
        raise ValueError(f"No 'pit' rows found in segmented {source} data.")

    pit_df["group"] = (pit_df.index.to_series().diff() != 1).cumsum()
    first_group = pit_df["group"].iloc[0]
    first_pit = pit_df[pit_df["group"] == first_group].copy()
    return first_pit.drop(columns=["group"])


def print_selected_segment_info(df: pd.DataFrame, source: str) -> None:
    print(f"\n[{source}] segment counts:")
    print(df["segment_final"].value_counts(dropna=False).to_string())

    pit_df = df[df["segment_final"] == "pit"].copy()
    pit_df["group"] = (pit_df.index.to_series().diff() != 1).cumsum()
    first_group = pit_df["group"].iloc[0]
    first_pit = pit_df[pit_df["group"] == first_group].copy()

    print(f"[{source}] selected first pit group: {first_group}")
    print(
        f"[{source}] selected index range: "
        f"{first_pit.index.min()} -> {first_pit.index.max()}"
    )
    print(f"[{source}] selected samples: {len(first_pit)}")
    print(f"[{source}] labels in selected segment:")
    print(first_pit["segment_final"].value_counts(dropna=False).to_string())

def build_x_axis(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "sample":
        return pd.Series(range(len(df)), index=df.index, dtype="float64")

    if len(df) == 1:
        return pd.Series([0.0], index=df.index, dtype="float64")

    return pd.Series(
        [i / (len(df) - 1) for i in range(len(df))],
        index=df.index,
        dtype="float64",
    )


def main():
    args = parse_args()

    print("Loading real segmented data...")
    real_df = load_segmented_data("real")
    print("Loading simulator segmented data...")
    simu_df = load_segmented_data("simu")

    print_selected_segment_info(real_df, "real")
    print_selected_segment_info(simu_df, "simu")

    real_pit = get_first_pit_segment(real_df, "real")
    simu_pit = get_first_pit_segment(simu_df, "simu")

    print(f"Real first pit samples: {len(real_pit)}")
    print(f"Simu first pit samples: {len(simu_pit)}")

    real_x = build_x_axis(real_pit, args.x_mode)
    simu_x = build_x_axis(simu_pit, args.x_mode)

    for column in SENSOR_COLUMNS:
        if column not in real_pit.columns and column not in simu_pit.columns:
            print(f"Skipping missing column in both datasets: {column}")
            continue

        plt.figure(figsize=(12, 5))

        if column in real_pit.columns:
            real_y = pd.to_numeric(real_pit[column], errors="coerce")
            real_mask = real_x.notna() & real_y.notna()
            if real_mask.any():
                plt.plot(
                    real_x[real_mask],
                    real_y[real_mask],
                    linewidth=2.0,
                    label="real",
                )

        if column in simu_pit.columns:
            simu_y = pd.to_numeric(simu_pit[column], errors="coerce")
            simu_mask = simu_x.notna() & simu_y.notna()
            if simu_mask.any():
                plt.plot(
                    simu_x[simu_mask],
                    simu_y[simu_mask],
                    linewidth=2.0,
                    label="simu",
                )

        if not plt.gca().lines:
            plt.close()
            print(f"Skipping empty column: {column}")
            continue

        plt.title(f"{column} overlay in first pit segment")
        plt.xlabel(
            "Normalized pit progress" if args.x_mode == "normalized" else "Sample index"
        )
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
