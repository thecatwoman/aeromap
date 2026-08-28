import pandas as pd

from src.data.apply_despike_filter import (
    DESPIKE_COLUMNS,
    DEFAULT_MAD_K,
    DEFAULT_WINDOW,
    apply_despike_filter,
)


TPMS_FILL_COLUMNS = [
    "tpms_p_fr",
    "tpms_p_rl",
]
DROP_COLUMNS = [
    "tpms_p_fl",
    "tpms_p_rr",
]
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


def drop_zero_speed_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = pd.Series(True, index=df.index)

    if "carspeed_art" in df.columns:
        speed = pd.to_numeric(df["carspeed_art"], errors="coerce")
        mask &= speed > 0

    if "yaw0_c" in df.columns:
        mask &= df["yaw0_c"].notna()

    return df[mask]

def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_keep = [col for col in df.columns if not col.startswith("unnamed:")]
    return df[cols_to_keep]


def drop_known_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_keep = [col for col in df.columns if col not in DROP_COLUMNS]
    return df[cols_to_keep]

def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if col in ["source_file", "time"]:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fill_missing_tpms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TPMS_FILL_COLUMNS:
        if col not in df.columns:
            continue

        median_value = pd.to_numeric(df[col], errors="coerce").median(skipna=True)
        if pd.isna(median_value):
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median_value)

    return df


def apply_hampel_despike(df: pd.DataFrame) -> pd.DataFrame:
    despiked_df, _ = apply_despike_filter(
        df=df,
        columns=DESPIKE_COLUMNS,
        window=DEFAULT_WINDOW,
        mad_k=DEFAULT_MAD_K,
        replace_target_columns=True,
    )
    return despiked_df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = drop_unnamed_columns(df)
    df = drop_known_empty_columns(df)
    df = drop_empty_rows(df)
    df = drop_zero_speed_rows(df)
    df = drop_duplicate_rows(df)
    df = convert_numeric(df)
    df = fill_missing_tpms(df)
    df = apply_hampel_despike(df)

    return df
