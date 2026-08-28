from pathlib import Path
import re
import pandas as pd

from src.data.clean_raw import clean_dataframe
from src.run_paths import cleaned_merged_full_run_file, raw_run_dir, scz_dir


RUN_DIR = raw_run_dir()
RH_DIR = RUN_DIR / "Rideheight"
SENSORS_DIR = RUN_DIR / "Sensors"
SCZ_DIR = scz_dir()
OUTPUT_PATH = cleaned_merged_full_run_file()


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.Index([normalize_column_name(str(col)) for col in df.columns])
    return df


def extract_lap_number(filename: str) -> int | None:
    match = re.search(r"lap(\d+)", filename.lower())
    if match:
        return int(match.group(1))
    return None


def get_main_files(run_dir: Path = RUN_DIR) -> dict[int, Path]:
    files_by_lap: dict[int, Path] = {}

    for path in run_dir.glob("*.csv"):
        if path.name.lower().startswith("sensors_"):
            continue
        lap = extract_lap_number(path.name)
        if lap is not None:
            files_by_lap[lap] = path

    return files_by_lap


def get_rh_files(rh_dir: Path = RH_DIR) -> dict[int, Path]:
    files_by_lap: dict[int, Path] = {}

    for path in rh_dir.glob("*.csv"):
        lap = extract_lap_number(path.name)
        if lap is not None:
            files_by_lap[lap] = path

    return files_by_lap


def get_sensor_files(sensors_dir: Path = SENSORS_DIR) -> dict[int, Path]:
    files_by_lap: dict[int, Path] = {}

    for path in sensors_dir.glob("*.csv"):
        lap = extract_lap_number(path.name)
        if lap is not None:
            files_by_lap[lap] = path

    return files_by_lap


def get_scz_files(scz_dir: Path = SCZ_DIR) -> dict[int, Path]:
    files_by_lap: dict[int, Path] = {}

    for path in scz_dir.glob("*.csv"):
        lap = extract_lap_number(path.name)
        if lap is not None:
            files_by_lap[lap] = path

    return files_by_lap


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = path.name
    return df


def load_one_lap(main_path: Path) -> pd.DataFrame:
    df = standardize_columns(load_csv(main_path))
    lap = extract_lap_number(main_path.name)
    df["lap"] = lap
    return clean_dataframe(df)


def merge_optional_dataset(base_df: pd.DataFrame, extra_df: pd.DataFrame, label: str) -> pd.DataFrame:
    preferred_keys = ["time"]
    merge_key: str | None = None

    for key in preferred_keys:
        if key in base_df.columns and key in extra_df.columns:
            merge_key = key
            break

    if merge_key is not None:
        extra_only_cols = [
            col for col in extra_df.columns if col == merge_key or col not in base_df.columns
        ]
        extra_df = extra_df[extra_only_cols]
        return pd.merge(base_df, extra_df, on=merge_key, how="inner")

    if len(base_df) != len(extra_df):
        raise ValueError(
            f"Cannot merge base lap data with {label}: "
            f"no common merge key found and row counts differ "
            f"({len(base_df)} vs {len(extra_df)})."
        )

    extra_only_cols = [col for col in extra_df.columns if col not in base_df.columns]
    return pd.concat(
        [base_df.reset_index(drop=True), extra_df[extra_only_cols].reset_index(drop=True)],
        axis=1,
    )


def merge_scz_dataset(base_df: pd.DataFrame, scz_df: pd.DataFrame, label: str) -> pd.DataFrame:
    preferred_keys = ["time"]
    merge_key: str | None = None

    for key in preferred_keys:
        if key in base_df.columns and key in scz_df.columns:
            merge_key = key
            break

    scz_only_cols = [
        col
        for col in scz_df.columns
        if col != "source_file" and (merge_key is None or col == merge_key or col not in base_df.columns)
    ]
    skipped_cols = [
        col
        for col in scz_df.columns
        if col not in scz_only_cols and col != "source_file"
    ]

    if merge_key is not None:
        merged = pd.merge(base_df, scz_df[scz_only_cols], on=merge_key, how="inner")
    else:
        if len(base_df) != len(scz_df):
            raise ValueError(
                f"Cannot merge base lap data with {label}: "
                f"no common merge key found and row counts differ "
                f"({len(base_df)} vs {len(scz_df)})."
            )
        merged = pd.concat(
            [base_df.reset_index(drop=True), scz_df[[col for col in scz_only_cols if col not in base_df.columns]].reset_index(drop=True)],
            axis=1,
        )

    print(
        f"  SCz merge: kept {max(0, len(scz_only_cols) - (1 if merge_key is not None else 0))} "
        f"new columns, skipped {len(skipped_cols)} duplicate columns from {label}"
    )
    return merged


def merge_one_lap(
    main_path: Path,
    rh_path: Path,
    sensor_path: Path | None = None,
    scz_path: Path | None = None,
) -> pd.DataFrame:
    main_df = standardize_columns(load_csv(main_path))
    rh_df = standardize_columns(load_csv(rh_path))

    merged = merge_optional_dataset(main_df, rh_df, rh_path.name)

    if sensor_path is not None:
        sensor_df = standardize_columns(load_csv(sensor_path))
        merged = merge_optional_dataset(merged, sensor_df, sensor_path.name)

    if scz_path is not None:
        scz_df = standardize_columns(load_csv(scz_path))
        merged = merge_scz_dataset(merged, scz_df, scz_path.name)

    lap = extract_lap_number(main_path.name)
    merged["lap"] = lap

    merged = clean_dataframe(merged)
    return merged


def merge_all_laps(
    run_dir: Path = RUN_DIR,
    rh_dir: Path = RH_DIR,
    sensors_dir: Path = SENSORS_DIR,
    scz_dir: Path = SCZ_DIR,
) -> pd.DataFrame:
    main_files = get_main_files(run_dir)
    rh_files = get_rh_files(rh_dir)
    sensor_files = get_sensor_files(sensors_dir)
    scz_files = get_scz_files(scz_dir)

    common_laps = sorted(set(main_files.keys()) & set(rh_files.keys()))

    if not common_laps:
        raise ValueError("No matching lap pairs found.")

    merged_laps: list[pd.DataFrame] = []

    for lap in common_laps:
        main_path = main_files[lap]
        rh_path = rh_files[lap]
        sensor_path = sensor_files.get(lap)
        scz_path = scz_files.get(lap)

        message = f"Merging Lap {lap}: {main_path.name} + {rh_path.name}"
        if sensor_path is not None:
            message += f" + {sensor_path.name}"
        if scz_path is not None:
            message += f" + {scz_path.name}"
        print(message)

        merged_df = merge_one_lap(
            main_path,
            rh_path,
            sensor_path=sensor_path,
            scz_path=scz_path,
        )
        print(f"  Result shape: {merged_df.shape}")

        merged_laps.append(merged_df)

    final_df = pd.concat(merged_laps, ignore_index=True)
    return final_df


def load_all_laps_direct(
    run_dir: Path = RUN_DIR,
    scz_dir: Path = SCZ_DIR,
) -> pd.DataFrame:
    main_files = get_main_files(run_dir)
    scz_files = get_scz_files(scz_dir)

    if not main_files:
        raise ValueError(f"No lap CSV files found in {run_dir}.")

    loaded_laps: list[pd.DataFrame] = []

    for lap in sorted(main_files):
        main_path = main_files[lap]
        scz_path = scz_files.get(lap)
        message = f"Loading Lap {lap}: {main_path.name}"
        if scz_path is not None:
            message += f" + {scz_path.name}"
        print(message)

        loaded_df = load_one_lap(main_path)
        if scz_path is not None:
            scz_df = standardize_columns(load_csv(scz_path))
            loaded_df = merge_scz_dataset(loaded_df, scz_df, scz_path.name)
        print(f"  Result shape: {loaded_df.shape}")
        loaded_laps.append(loaded_df)

    return pd.concat(loaded_laps, ignore_index=True)


def build_run_dataset(
    run_dir: Path = RUN_DIR,
    rh_dir: Path = RH_DIR,
    sensors_dir: Path = SENSORS_DIR,
    scz_dir: Path = SCZ_DIR,
) -> pd.DataFrame:
    has_rh_files = rh_dir.exists() and any(rh_dir.glob("*.csv"))
    has_sensor_files = sensors_dir.exists() and any(sensors_dir.glob("*.csv"))
    has_scz_files = scz_dir.exists() and any(scz_dir.glob("*.csv"))

    if has_rh_files:
        print("Detected separate Rideheight files. Using merge pipeline.")
        if has_sensor_files:
            print("Detected separate Sensors files. Including them in the merge.")
        else:
            print("No separate Sensors files found. Merging only base + Rideheight data.")
        if has_scz_files:
            print("Detected separate Scz files. Including Scz-only columns in the merge.")
        else:
            print("No separate Scz files found.")
        return merge_all_laps(run_dir=run_dir, rh_dir=rh_dir, sensors_dir=sensors_dir, scz_dir=scz_dir)

    print("No separate Rideheight folder detected. Loading already-combined lap files directly.")
    if has_scz_files:
        print("Detected separate Scz files. Including Scz-only columns in the direct-load merge.")
    else:
        print("No separate Scz files found.")
    return load_all_laps_direct(run_dir=run_dir, scz_dir=scz_dir)


def main() -> None:
    final_df = build_run_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Final shape: {final_df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
