import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file


LAT_COL = "avg_accy"
LONG_COL = "avg_accx"
SPEED_COL = "carspeed_art"


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
        help="X-axis to use for series plots.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["series", "xy"],
        default="series",
        help="Plot pitot_c/pair over x, or plot pair vs pitot_c directly.",
    )
    return parser.parse_args()


def load_data(source: str) -> pd.DataFrame:
    if source == "simu":
        return pd.read_csv(segmented_simudata_run_file(), low_memory=False)
    return pd.read_csv(segmented_rh_run_file(), low_memory=False)


def get_condition_mask(df: pd.DataFrame) -> pd.Series:
    required = [LAT_COL, LONG_COL, SPEED_COL]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    working_df = df.copy()
    for col in [LAT_COL, LONG_COL, SPEED_COL, "pitot_c", "pair"]:
        if col in working_df.columns:
            working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    mask = (
        working_df[LAT_COL].notna()
        & working_df[LONG_COL].notna()
        & working_df[SPEED_COL].notna()
        & (working_df[LAT_COL].abs() > 1.0)
        & (working_df[LONG_COL].abs() < 0.5)
        & (working_df[SPEED_COL] > 70)
    )

    return mask


def filter_rows_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[get_condition_mask(df)].copy()


def get_x_axis(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "time" and "time" in df.columns:
        return df["time"]
    return pd.Series(range(len(df)), index=df.index)


def plot_source_series(ax, df: pd.DataFrame, source: str, x_mode: str) -> None:
    highlight_mask = get_condition_mask(df)
    color_map = {
        "pitot_c": "tab:green",
        "pair": "tab:orange",
    }
    full_run_label_added = False

    for column in ["pitot_c", "pair"]:
        if column not in df.columns:
            print(f"[{source}] missing column: {column}")
            continue

        x = get_x_axis(df, x_mode)
        y = pd.to_numeric(df[column], errors="coerce")
        base_mask = x.notna() & y.notna()
        matched_mask = base_mask & highlight_mask

        if not base_mask.any():
            print(f"[{source}] empty column: {column}")
            continue

        ax.plot(
            x[base_mask],
            y[base_mask],
            linewidth=1.2,
            color="tab:blue",
            alpha=0.9,
            label="full run" if not full_run_label_added else None,
        )
        full_run_label_added = True

        if matched_mask.any():
            run_id = (matched_mask != matched_mask.shift(fill_value=False)).cumsum()
            first_label = True
            for _, run_df in df[matched_mask].groupby(run_id[matched_mask]):
                run_x = x.loc[run_df.index]
                run_y = y.loc[run_df.index]
                run_valid = run_x.notna() & run_y.notna()
                if not run_valid.any():
                    continue
                ax.plot(
                    run_x[run_valid],
                    run_y[run_valid],
                    linewidth=2.8,
                    color=color_map[column],
                    label=f"{column} straight" if first_label else None,
                )
                first_label = False

    ax.set_title(f"{source}: pitot_c and pair with straight parts highlighted")
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
        print(f"[{source}] no valid pitot_c/pair points after filtering")
        return

    x_vals = x[mask].to_numpy(dtype="float64")
    y_vals = y[mask].to_numpy(dtype="float64")

    ax.scatter(x_vals, y_vals, s=12, alpha=0.6)

    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_fit = x_vals[finite_mask]
    y_fit = y_vals[finite_mask]

    if len(x_fit) >= 2 and np.unique(x_fit).size >= 2:
        try:
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="fit",
            )
            ax.text(
                0.62,
                0.94,
                f"y = {slope:.4f}x + {intercept:.4f}",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
        except np.linalg.LinAlgError:
            print(f"[{source}] fit line skipped: linear fit failed")

    ax.set_title(f"{source}: pair vs pitot_c on straights only")
    ax.set_xlabel("pitot_c")
    ax.set_ylabel("pair")
    ax.grid(True, alpha=0.3)
    ax.legend()

def main():
    args = parse_args()

    if args.source == "both":
        fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
        for ax, source in zip(axes, ["real", "simu"]):
            df = load_data(source)
            mask = get_condition_mask(df)
            filtered = filter_rows_by_condition(df)
            print(f"[{source}] total rows: {len(df)}")
            print(f"[{source}] matched rows: {int(mask.sum())}")
            if args.plot_mode == "xy":
                plot_source_xy(ax, filtered, source)
            else:
                plot_source_series(ax, df, source, args.x_mode)
        plt.tight_layout()
        plt.show()
        return

    df = load_data(args.source)
    mask = get_condition_mask(df)
    filtered = filter_rows_by_condition(df)
    print(f"[{args.source}] total rows: {len(df)}")
    print(f"[{args.source}] matched rows: {int(mask.sum())}")

    if args.plot_mode == "xy":
        fig, ax = plt.subplots(figsize=(14, 5))
        plot_source_xy(ax, filtered, args.source)
    else:
        fig, ax = plt.subplots(figsize=(14, 6), sharex=False)
        plot_source_series(ax, df, args.source, args.x_mode)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
