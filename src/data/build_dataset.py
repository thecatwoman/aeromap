import argparse
from pathlib import Path

from src.data.apply_despike_filter import (
    RIDE_HEIGHT_COLUMNS as DESPIKE_RIDE_HEIGHT_COLUMNS,
    apply_despike_filter,
)
from src.data.apply_butterworth_filter import (
    CHANNEL_CUTOFFS_HZ,
    DEFAULT_FILTER_ORDER,
    apply_butterworth_filter,
    infer_sampling_rate_hz,
)
from src.data.calculate_scz_push import calculate_scz_push_f, calculate_scz_push_r
from src.data.Load_full_dataset import build_run_dataset
from src.run_paths import (
    cleaned_merged_dir,
    cleaned_merged_full_run_file,
    cleaned_simudata_full_run_file,
    processed_simudata_dir,
    raw_run_dir,
    scz_dir,
    simudata_dir,
)


def get_dataset_paths(source: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    if source == "simu":
        run_dir = simudata_dir()
        rh_dir = run_dir / "Rideheight"
        sensors_dir = run_dir / "Sensors"
        scz_data_dir = run_dir / "Scz"
        processed_dir = processed_simudata_dir()
        output_file = cleaned_simudata_full_run_file()
        return run_dir, rh_dir, sensors_dir, scz_data_dir, processed_dir, output_file

    run_dir = raw_run_dir()
    rh_dir = run_dir / "Rideheight"
    sensors_dir = run_dir / "Sensors"
    scz_data_dir = scz_dir()
    processed_dir = cleaned_merged_dir()
    output_file = cleaned_merged_full_run_file()
    return run_dir, rh_dir, sensors_dir, scz_data_dir, processed_dir, output_file


def run(source: str = "real") -> None:
    run_dir, rh_dir, sensors_dir, scz_data_dir, processed_dir, output_file = get_dataset_paths(source)

    processed_dir.mkdir(parents=True, exist_ok=True)

    final_df = build_run_dataset(
        run_dir=run_dir,
        rh_dir=rh_dir,
        sensors_dir=sensors_dir,
        scz_dir=scz_data_dir,
    )
    final_df = calculate_scz_push_f(final_df)
    final_df = calculate_scz_push_r(final_df)
    ride_height_columns = [
        column for column in DESPIKE_RIDE_HEIGHT_COLUMNS if column in final_df.columns
    ]
    if ride_height_columns:
        final_df, ride_height_spike_counts = apply_despike_filter(
            df=final_df,
            columns=ride_height_columns,
            window=12,
            mad_k=3.5,
            replace_target_columns=True,
        )
    else:
        ride_height_spike_counts = {}
    fs, fs_reason = infer_sampling_rate_hz(final_df)
    final_df = apply_butterworth_filter(
        df=final_df,
        columns=[
            column
            for column in [
                *ride_height_columns,
                "pushavd_c",
                "pushavg_c",
                "pushard_c",
                "pusharg_c",
                "pitot_c",
                "damper_fl_art",
                "damper_fr_art",
                "damper_rl_art",
                "damper_rr_art",
            ]
            if column in final_df.columns
        ],
        fs=fs,
        order=DEFAULT_FILTER_ORDER,
        replace_target_columns=True,
        cutoff_by_column=CHANNEL_CUTOFFS_HZ,
    )
    final_df.to_csv(output_file, index=False)

    print(f"Source: {source}")
    print(f"Final shape: {final_df.shape}")
    if ride_height_columns:
        replaced = ", ".join(
            f"{column}={ride_height_spike_counts.get(column, 0)}"
            for column in ride_height_columns
        )
        print(f"Ride-height despike applied before Butterworth: yes ({replaced})")
    else:
        print("Ride-height despike applied before Butterworth: no ride-height columns present")
    print(f"Butterworth applied automatically: yes ({fs_reason})")
    print(f"Saved: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["real", "simu"],
        default="real",
        help="Dataset source to process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run(source=args.source)


if __name__ == "__main__":
    main()
