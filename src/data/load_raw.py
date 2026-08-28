from pathlib import Path
import pandas as pd

#RAW_DIR = Path("data/raw/Barcelona_2026/Test/Run_46")
RAW_DIR = Path("data/raw/Barcelona_2026/Test/Run_46")

def find_csv_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted(raw_dir.rglob("*.csv"))


def load_csv_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = path.name
    return df


def load_all_csvs(raw_dir: Path = RAW_DIR) -> list[pd.DataFrame]:
    files = find_csv_files(raw_dir)

    datasets = []
    for file_path in files:
        try:
            df = load_csv_file(file_path)
            datasets.append(df)
            print(f"Loaded: {file_path.name} -> {df.shape}")
        except Exception as e:
            print(f"[WARN] Failed to load {file_path}: {e}")

    return datasets

def merge_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    if not datasets:
        return pd.DataFrame()

    merged_df = pd.concat(datasets, ignore_index=True)
    return merged_df