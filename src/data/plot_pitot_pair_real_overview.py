import pandas as pd
import matplotlib.pyplot as plt

from src.data import plot_pitot_pair_condition_filtered as straight_plots
from src.data import plot_pitot_pair_pitlane_filtered as pit_plots
from src.data import plot_pitot_pair_sanity as sanity_plots
from src.run_paths import segmented_rh_run_file


def plot_pit_only_series(ax, df, source: str) -> None:
    pit_df = pit_plots.filter_rows_by_condition(df)

    if pit_df.empty:
        print(f"[{source}] no pit rows found")
        ax.set_title(f"{source}: pitot_c vs pair on pit-only rows")
        ax.set_xlabel("Filtered pit sample index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        return

    x = range(len(pit_df))
    color_map = {
        "pitot_c": "tab:blue",
        "pair": "tab:orange",
        "carspeed_art": "tab:red",
    }

    for column in ["pitot_c", "pair", "carspeed_art"]:
        if column not in pit_df.columns:
            print(f"[{source}] missing column: {column}")
            continue

        y = pit_df[column]
        ax.plot(x, y, linewidth=1.6, color=color_map.get(column), label=column)

    ax.set_title(f"{source}: pitot_c vs pair vs carspeed_art on pit-only rows")
    ax.set_xlabel("Filtered pit sample index")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    source = "real"
    df = pd.read_csv(segmented_rh_run_file(), low_memory=False)

    straight_mask = straight_plots.get_condition_mask(df)
    pit_mask = pit_plots.get_condition_mask(df)

    print(f"[{source}] total rows: {len(df)}")
    print(f"[{source}] straight matched rows: {int(straight_mask.sum())}")
    print(f"[{source}] pit matched rows: {int(pit_mask.sum())}")

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)

    sanity_plots.plot_source_series(axes[0], df, source, x_mode="sample")
    straight_plots.plot_source_xy(
        axes[1],
        straight_plots.filter_rows_by_condition(df),
        source,
    )
    plot_pit_only_series(axes[2], df, source)

    fig.suptitle("Real Data: pitot_c / pair overview", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


if __name__ == "__main__":
    main()
