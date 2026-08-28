import matplotlib.pyplot as plt

from src.data import plot_pitot_pair_condition_filtered as straight_plots
from src.data import plot_pitot_pair_pitlane_filtered as pit_plots
from src.data import plot_pitot_pair_sanity as sanity_plots


def main():
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=False)

    real_df = sanity_plots.load_data("real")
    simu_df = sanity_plots.load_data("simu")
    sanity_plots.plot_source_series(axes[0], real_df, "real", "sample")
    sanity_plots.plot_source_series(axes[1], simu_df, "simu", "sample")

    straight_real_df = straight_plots.load_data("real")
    straight_real_filtered = straight_plots.filter_rows_by_condition(straight_real_df)
    straight_plots.plot_source_xy(axes[2], straight_real_filtered, "real")

    pit_real_df = pit_plots.load_data("real")
    pit_plots.plot_source_series(axes[3], pit_real_df, "real", "sample")

    axes[0].set_title("Sanity Plot - Real")
    axes[1].set_title("Sanity Plot - Simu")
    axes[2].set_title("Straight Condition XY - Real")
    axes[3].set_title("Pit Condition Series - Real")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
