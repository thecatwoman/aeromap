import argparse
from pathlib import Path


def cmd_doctor() -> None:
    import pandas  # noqa
    import sklearn  # noqa
    import pyarrow  # noqa

    print("OK: pandas, sklearn, pyarrow imported")
    try:
        import xgboost  # noqa
        print("OK: xgboost imported")
    except Exception as e:
        print("WARN: xgboost not available:", e)

    print("Project root:", Path.cwd())


def cmd_build_dataset() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    print("Running dataset build...")
    print(f"Raw data directory: {raw_dir.resolve()}")
    print(f"Processed data directory: {processed_dir.resolve()}")

    if not raw_dir.exists():
        print("ERROR: data/raw directory does not exist.")
        return

    if not processed_dir.exists():
        print("ERROR: data/processed directory does not exist.")
        return

    csv_files = list(raw_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV file(s) in raw data.")

    if len(csv_files) == 0:
        print("No CSV files found yet. Waiting for Wintax4 exports.")
        return

    print("Placeholder build complete. Real dataset builder will be added next.")


def cmd_train_model() -> None:
    processed_dir = Path("data/processed")

    print("Running model training...")
    print(f"Processed data directory: {processed_dir.resolve()}")

    if not processed_dir.exists():
        print("ERROR: data/processed directory does not exist.")
        return

    parquet_files = list(processed_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} processed dataset file(s).")

    if len(parquet_files) == 0:
        print("No processed dataset found yet. Build the dataset first.")
        return

    print("Placeholder training complete. Real baseline model will be added later.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="aeromap")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Check environment and dependencies")
    sub.add_parser("build-dataset", help="Build processed dataset from raw CSV files")
    sub.add_parser("train-model", help="Train baseline model from processed dataset")

    args = parser.parse_args()

    if args.cmd == "doctor":
        cmd_doctor()
    elif args.cmd == "build-dataset":
        cmd_build_dataset()
    elif args.cmd == "train-model":
        cmd_train_model()


if __name__ == "__main__":
    main()