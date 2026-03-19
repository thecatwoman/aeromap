from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")


def find_csv_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted(raw_dir.rglob("*.csv"))


def load_csv_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = path.name
    return df


def load_all_csvs(raw_dir: Path = RAW_DIR):
    files = find_csv_files(raw_dir)

    datasets = []
    for file_path in files:
        try:
            df = load_csv_file(file_path)
            datasets.append((file_path, df))
        except Exception as e:
            print(f"[WARN] Failed to load {file_path}: {e}")

    return datasets