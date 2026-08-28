import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.run_paths import cleaned_merged_full_run_file, cleaned_simudata_full_run_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["real", "simu", "both"],
        default="both",
        help="Dataset source to inspect.",
    )
    parser.add_argument(
        "--x-mode",
        choices=["sample", "time"],
        default="sample",
        help="X-axis to use for the plots.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["series", "xy"],
        default="series",
        help="Plot both signals over x, or plot pair vs pitot_c directly.",
    )
    return parser.parse_args()


def load_data(source: str) -> pd.DataFrame:
    if source == "simu":
        return pd.read_csv(cleaned_simudata_full_run_file(), low_memory=False)
    return pd.read_csv(cleaned_merged_full_run_file(), low_memory=False)


def get_x_axis(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "time" and "time" in df.columns:
        return df["time"]
    return pd.Series(range(len(df)), index=df.index)


def plot_source_series(ax, df: pd.DataFrame, source: str, x_mode: str) -> None:
    for column in ["pitot_c", "pair", "carspeed_art"]:
        if column not in df.columns:
            print(f"[{source}] missing column: {column}")
            continue

        x = get_x_axis(df, x_mode)
        y = pd.to_numeric(df[column], errors="coerce")
        mask = x.notna() & y.notna()

        if not mask.any():
            print(f"[{source}] empty column after cleaning: {column}")
            continue

        ax.plot(x[mask], y[mask], linewidth=1.6, label=column)

    ax.set_title(f"{source}: pitot_c vs pair vs carspeed_art")
    ax.set_xlabel("Time" if x_mode == "time" else "Sample index")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_source_xy(ax, df: pd.DataFrame, source: str) -> None:
    required = {"pitot_c", "pair"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        print(f"[{source}] missing columns: {', '.join(missing)}")
        return

    x = pd.to_numeric(df["pitot_c"], errors="coerce")
    y = pd.to_numeric(df["pair"], errors="coerce")
    mask = x.notna() & y.notna()

    if not mask.any():
        print(f"[{source}] no valid pitot_c/pair points after cleaning")
        return

    ax.scatter(x[mask], y[mask], s=10, alpha=0.6)
    ax.set_title(f"{source}: pair vs pitot_c")
    ax.set_xlabel("pitot_c")
    ax.set_ylabel("pair")
    ax.grid(True, alpha=0.3)


def main():
    args = parse_args()

    if args.source == "both":
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
        real_df = load_data("real")
        simu_df = load_data("simu")
        if args.plot_mode == "xy":
            plot_source_xy(axes[0], real_df, "real")
            plot_source_xy(axes[1], simu_df, "simu")
        else:
            plot_source_series(axes[0], real_df, "real", args.x_mode)
            plot_source_series(axes[1], simu_df, "simu", args.x_mode)
        plt.tight_layout()
        plt.show()
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    df = load_data(args.source)
    if args.plot_mode == "xy":
        plot_source_xy(ax, df, args.source)
    else:
        plot_source_series(ax, df, args.source, args.x_mode)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
