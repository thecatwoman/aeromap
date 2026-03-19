from pathlib import Path

from src.data.load_raw import load_all_csvs
from src.data.clean_raw import clean_dataframe

PROCESSED_DIR = Path("data/processed")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_csvs()

    if not datasets:
        print("No CSV files found in data/raw")
        return

    print(f"Found {len(datasets)} dataset(s)")

    for path, df in datasets:
        cleaned_df = clean_dataframe(df)

        output_name = f"{path.stem}_cleaned.csv"
        output_path = PROCESSED_DIR / output_name

        cleaned_df.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()