import pandas as pd


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
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all").copy()


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_keep = [col for col in df.columns if not col.startswith("unnamed:")]
    return df[cols_to_keep]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = drop_unnamed_columns(df)
    df = drop_empty_rows(df)
    df = drop_duplicate_rows(df)
    return df