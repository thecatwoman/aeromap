import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.data.apply_median_filter import get_input_path


DEFAULT_COLUMNS = ["rh_f", "rh_r"]
SPEED_COLUMN = "carspeed_art"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw ride-height channels from the cleaned dataset with no extra filtering.",
    )
    parser.add_argument(
        "--source",
        choices=["real", "simu"],
        default="real",
        help="Dataset source to plot.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=DEFAULT_COLUMNS,
        help="Ride-height columns to plot.",
    )
    parser.add_argument(
        "--x-mode",
        choices=["sample", "time"],
        default="sample",
        help="X-axis to use.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row for the plotted slice.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1500,
        help="Number of rows to plot from the start index.",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Plot the full cleaned run. Overrides --start/--count.",
    )
    parser.add_argument(
        "--overlay-speed",
        action="store_true",
        help="Overlay carspeed_art on the same plot using a secondary y-axis.",
    )
    return parser.parse_args()


def get_x_axis(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "time" and "time" in df.columns:
        return pd.to_numeric(df["time"], errors="coerce")
    return pd.Series(range(len(df)), index=df.index, dtype="int64")


def slice_dataframe(df: pd.DataFrame, start: int, count: int) -> pd.DataFrame:
    if start < 0:
        raise ValueError("--start must be zero or positive.")
    if count <= 0:
        raise ValueError("--count must be greater than zero.")
    return df.iloc[start : start + count].copy()


def main() -> None:
    args = parse_args()

    input_path = get_input_path(args.source)
    print(f"Loading cleaned dataset: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    if not args.full_run:
        df = slice_dataframe(df, start=args.start, count=args.count)

    available_columns = [column for column in args.columns if column in df.columns]
    if not available_columns:
        raise ValueError("None of the requested columns are present in the selected dataset.")

    x = get_x_axis(df, args.x_mode)
    speed_requested = args.overlay_speed and SPEED_COLUMN in df.columns
    ride_height_columns = [column for column in available_columns if column != SPEED_COLUMN]

    if speed_requested and not ride_height_columns:
        ride_height_columns = [SPEED_COLUMN]
        speed_requested = False

    if speed_requested:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        speed_ax = ax.twinx()
        line_handles = []

        ride_height_colors = {
            "rh_f": "tab:blue",
            "rh_r": "tab:orange",
        }
        for column in ride_height_columns:
            y = pd.to_numeric(df[column], errors="coerce")
            mask = x.notna() & y.notna()
            (line,) = ax.plot(
                x[mask],
                y[mask],
                linewidth=1.3,
                color=ride_height_colors.get(column, "0.3"),
                label=column,
            )
            line_handles.append(line)

        speed = pd.to_numeric(df[SPEED_COLUMN], errors="coerce")
        speed_mask = x.notna() & speed.notna()
        (speed_line,) = speed_ax.plot(
            x[speed_mask],
            speed[speed_mask],
            linewidth=1.1,
            color="tab:red",
            alpha=0.8,
            label=SPEED_COLUMN,
        )
        line_handles.append(speed_line)

        ax.set_title("Ride height with carspeed overlay")
        ax.set_ylabel("Ride height")
        speed_ax.set_ylabel("carspeed_art")
        ax.grid(True, alpha=0.3)
        ax.legend(line_handles, [line.get_label() for line in line_handles], loc="upper right")
        ax.set_xlabel("Time" if args.x_mode == "time" else "Global sample index")
    else:
        fig, axes = plt.subplots(
            len(available_columns),
            1,
            figsize=(14, max(3.5 * len(available_columns), 6)),
            sharex=True,
        )

        if len(available_columns) == 1:
            axes = [axes]

        for ax, column in zip(axes, available_columns):
            y = pd.to_numeric(df[column], errors="coerce")
            mask = x.notna() & y.notna()
            ax.plot(x[mask], y[mask], linewidth=1.2, color="0.3")
            ax.set_title(column)
            ax.set_ylabel(column)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time" if args.x_mode == "time" else "Global sample index")
    fig.suptitle(
        (
            f"{args.source} raw cleaned ride height "
            + (
                "full run"
                if args.full_run
                else f"rows {args.start} to {args.start + len(df) - 1}"
            )
        ),
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
