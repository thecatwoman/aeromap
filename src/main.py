from src.data.load_raw import load_all_csvs
from src.data.clean_raw import clean_dataframe


def main():
    datasets = load_all_csvs()

    print(f"Found {len(datasets)} dataset(s)")

    for path, df in datasets:
        cleaned_df = clean_dataframe(df)

        print(f"\nFile: {path.name}")
        print(f"Original shape: {df.shape}")
        print(f"Cleaned shape:  {cleaned_df.shape}")
        print("Columns:")
        print(list(cleaned_df.columns))


if __name__ == "__main__":
    main()