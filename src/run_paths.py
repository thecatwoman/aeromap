import os
from pathlib import Path


TRACK = "Barcelona"
YEAR = 2026
DEFAULT_RUN = 46


def _current_run_from_env() -> int:
    raw_value = os.environ.get("AEROMAP_RUN")
    if raw_value is None or raw_value == "":
        return DEFAULT_RUN
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid AEROMAP_RUN value: {raw_value!r}. Expected an integer run number."
        ) from exc


CURRENT_RUN = _current_run_from_env()


def processed_run_dir(run_number: int = CURRENT_RUN) -> Path:
    return Path("data/processed") / f"Run_{run_number}"


def raw_run_dir(run_number: int = CURRENT_RUN) -> Path:
    return Path("data/raw") / f"{TRACK}_{YEAR}" / "Test" / f"Run_{run_number}"


def cleaned_merged_dir(run_number: int = CURRENT_RUN) -> Path:
    return processed_run_dir(run_number) / "Cleaned_Merged"


def segmented_dir(run_number: int = CURRENT_RUN) -> Path:
    return processed_run_dir(run_number) / "Segmented"


def results_rideheight_dir(run_number: 
    int = CURRENT_RUN) -> Path:
    return processed_run_dir(run_number) / "Results_Rideheight"


def simudata_dir(run_number: int = CURRENT_RUN) -> Path:
    run_dir = raw_run_dir(run_number)
    preferred_path = run_dir / "SimuData"
    if preferred_path.exists():
        return preferred_path

    legacy_path = run_dir / "Simudata"
    if legacy_path.exists():
        return legacy_path

    return preferred_path


def scz_dir(run_number: int = CURRENT_RUN) -> Path:
    return raw_run_dir(run_number) / f"Scz_RUN{run_number}"


def processed_simudata_dir(run_number: int = CURRENT_RUN) -> Path:
    return processed_run_dir(run_number) / "SimuData"


def cleaned_merged_full_run_file(run_number: int = CURRENT_RUN) -> Path:
    return cleaned_merged_dir(run_number) / (
        f"barcelona_{YEAR}_merged_cleaned_full_run_{run_number}.csv"
    )


def segmented_rh_run_file(run_number: int = CURRENT_RUN) -> Path:
    return segmented_dir(run_number) / (
        f"barcelona_{YEAR}_merged_cleaned_RH_run_{run_number}_segmented.csv"
    )


def merged_with_rh_file(run_number: int = CURRENT_RUN) -> Path:
    return results_rideheight_dir(run_number) / f"barcelona_{YEAR}_merged_with_rh.csv"


def merged_with_rh_segmented_file(run_number: int = CURRENT_RUN) -> Path:
    return results_rideheight_dir(run_number) / (
        f"barcelona_{YEAR}_merged_with_rh_segmented.csv"
    )


def cleaned_simudata_full_run_file(run_number: int = CURRENT_RUN) -> Path:
    return processed_simudata_dir(run_number) / (
        f"barcelona_{YEAR}_simudata_cleaned_full_run_{run_number}.csv"
    )


def segmented_simudata_run_file(run_number: int = CURRENT_RUN) -> Path:
    return processed_simudata_dir(run_number) / (
        f"barcelona_{YEAR}_simudata_segmented_run_{run_number}.csv"
    )
