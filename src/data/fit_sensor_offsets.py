import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.run_paths import segmented_rh_run_file, segmented_simudata_run_file


SENSOR_CONFIG = {
    "rh_f": {"window": 31, "mad_k": 3.5},
    "rh_r": {"window": 31, "mad_k": 3.5},
    "pushavg_c": {"window": 31, "mad_k": 3.5},
    "pushavd_c": {"window": 31, "mad_k": 3.5},
    "pusharg_c": {"window": 31, "mad_k": 3.5},
    "pushard_c": {"window": 31, "mad_k": 3.5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show per-sensor diagnostic plots with outliers highlighted.",
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


def build_normalized_x(df: pd.DataFrame) -> pd.Series:
    if len(df) == 1:
        return pd.Series([0.0], index=df.index, dtype="float64")

    return pd.Series(
        np.linspace(0.0, 1.0, len(df)),
        index=df.index,
        dtype="float64",
    )


def compute_real_outlier_mask(series: pd.Series, window: int, mad_k: float) -> pd.Series:
    rolling_median = series.rolling(
        window=window,
        center=True,
        min_periods=max(3, window // 2),
    ).median()
    residuals = series - rolling_median
    median_residual = residuals.median(skipna=True)
    mad = (residuals - median_residual).abs().median(skipna=True)

    if pd.isna(mad) or mad == 0:
        return pd.Series(False, index=series.index)

    threshold = mad_k * 1.4826 * mad
    return (residuals - median_residual).abs() > threshold


def interpolate_to_grid(x: pd.Series, y: pd.Series, grid: np.ndarray) -> np.ndarray:
    clean = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    clean = clean.drop_duplicates(subset="x", keep="first")

    if len(clean) < 2:
        return np.full_like(grid, np.nan, dtype="float64")

    return np.interp(grid, clean["x"].to_numpy(), clean["y"].to_numpy())


def fit_sensor_offset(
    real_df: pd.DataFrame,
    simu_df: pd.DataFrame,
    sensor: str,
    window: int,
    mad_k: float,
) -> dict:
    real_x = build_normalized_x(real_df)
    simu_x = build_normalized_x(simu_df)

    real_y = pd.to_numeric(real_df[sensor], errors="coerce")
    simu_y = pd.to_numeric(simu_df[sensor], errors="coerce")

    real_outliers = compute_real_outlier_mask(real_y, window=window, mad_k=mad_k)
    real_inliers = real_y.notna() & (~real_outliers)
    simu_valid = simu_y.notna()

    grid = np.linspace(0.0, 1.0, 200)
    real_interp = interpolate_to_grid(real_x[real_inliers], real_y[real_inliers], grid)
    simu_interp = interpolate_to_grid(simu_x[simu_valid], simu_y[simu_valid], grid)

    diffs = real_interp - simu_interp
    diffs = diffs[~np.isnan(diffs)]

    if len(diffs) == 0:
        offset = np.nan
    else:
        offset = float(np.median(diffs))

    return {
        "sensor": sensor,
        "offset": offset,
        "real_samples": int(real_y.notna().sum()),
        "real_outliers": int(real_outliers.sum()),
        "real_inliers": int(real_inliers.sum()),
        "simu_samples": int(simu_valid.sum()),
        "grid_points_used": int(len(diffs)),
        "real_x": real_x,
        "real_y": real_y,
        "real_outliers_mask": real_outliers,
        "simu_x": simu_x,
        "simu_y": simu_y,
        "grid": grid,
        "real_interp": real_interp,
        "simu_interp": simu_interp,
    }


def plot_diagnostic(result: dict) -> None:
    sensor = result["sensor"]
    real_x = result["real_x"]
    real_y = result["real_y"]
    outliers = result["real_outliers_mask"]
    simu_x = result["simu_x"]
    simu_y = result["simu_y"]
    offset = result["offset"]

    plt.figure(figsize=(12, 5))
    plt.plot(real_x, real_y, label="real", linewidth=1.6, alpha=0.9)
    plt.plot(simu_x, simu_y, label="simu", linewidth=2.0, alpha=0.9)

    if not np.isnan(offset):
        plt.plot(
            real_x,
            real_y - offset,
            label=f"real - offset ({offset:.3f})",
            linewidth=2.0,
            linestyle="--",
        )

    if outliers.any():
        plt.scatter(
            real_x[outliers],
            real_y[outliers],
            color="red",
            s=22,
            label="real outliers",
            zorder=3,
        )

    plt.title(f"{sensor} offset fit diagnostic")
    plt.xlabel("Normalized pit progress")
    plt.ylabel(sensor)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    real_df = load_segmented_data("real")
    simu_df = load_segmented_data("simu")

    real_pit = get_first_pit_segment(real_df, "real")
    simu_pit = get_first_pit_segment(simu_df, "simu")

    print("Offset fit summary:\n")

    for sensor, config in SENSOR_CONFIG.items():
        if sensor not in real_pit.columns or sensor not in simu_pit.columns:
            print(f"{sensor}: skipped (missing in one of the datasets)")
            continue

        result = fit_sensor_offset(
            real_df=real_pit,
            simu_df=simu_pit,
            sensor=sensor,
            window=config["window"],
            mad_k=config["mad_k"],
        )

        offset = result["offset"]
        offset_text = "nan" if np.isnan(offset) else f"{offset:.6f}"
        print(
            f"{sensor}: offset={offset_text}, "
            f"real_outliers={result['real_outliers']}, "
            f"real_inliers={result['real_inliers']}, "
            f"simu_samples={result['simu_samples']}, "
            f"grid_points_used={result['grid_points_used']}"
        )

        if args.plot:
            plot_diagnostic(result)


if __name__ == "__main__":
    main()
