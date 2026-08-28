import pandas as pd
from typing import Callable


# ============================================================
# CONFIG
# ============================================================
MIN_POINTS = 100
TOL_SPEED = 2.5

LAT_COL = "avg_accy"
LONG_COL = "avg_accx"
SPEED_COL = "carspeed_art"


# ============================================================
# ROW VALIDATORS
# ------------------------------------------------------------
# These mirror your VBA helper functions exactly.
# ============================================================
def is_row_valid_for_straight(row: pd.Series) -> bool:
    lat_acc = row[LAT_COL]
    long_acc = row[LONG_COL]
    speed = row[SPEED_COL]

    if pd.isna(lat_acc) or pd.isna(long_acc) or pd.isna(speed):
        return False

    return (
        abs(lat_acc) < 0.15
        and abs(long_acc) < 0.15
        and speed > 70
    )


def is_row_valid_for_corner(row: pd.Series) -> bool:
    lat_acc = row[LAT_COL]
    long_acc = row[LONG_COL]
    speed = row[SPEED_COL]

    if pd.isna(lat_acc) or pd.isna(long_acc) or pd.isna(speed):
        return False

    return (
        abs(lat_acc) > 1.0
        and abs(long_acc) < 0.5
        and speed > 70
    )


def is_row_valid_for_pitlane(
    row: pd.Series,
    lat_acc_max: float = 0.15,
    long_acc_max: float = 0.15,
    speed_min: float = 1.0,
    speed_max: float = 75.0,
) -> bool:
    lat_acc = row[LAT_COL]
    long_acc = row[LONG_COL]
    speed = row[SPEED_COL]

    if pd.isna(lat_acc) or pd.isna(long_acc) or pd.isna(speed):
        return False

    return (
        abs(lat_acc) < lat_acc_max
        and abs(long_acc) < long_acc_max
        and speed > speed_min
        and speed < speed_max
    )


def keep_constant_speed_subruns(
    df: pd.DataFrame,
    labels: pd.Series,
    label: str,
    window: int = 15,
    speed_std_max: float = 0.10,
    min_points: int = MIN_POINTS,
) -> pd.Series:
    kept = pd.Series(index=labels.index, data=pd.NA, dtype="object")
    label_mask = labels == label

    if not label_mask.any():
        return kept

    rolling_std = (
        df[SPEED_COL]
        .rolling(window=window, center=True, min_periods=max(3, window // 2))
        .std()
    )
    constant_mask = label_mask & (rolling_std <= speed_std_max)
    group_ids = (constant_mask != constant_mask.shift()).cumsum()

    for _, group_df in df[constant_mask].groupby(group_ids[constant_mask]):
        if len(group_df) >= min_points:
            kept.loc[group_df.index] = label

    return kept


# ============================================================
# CORE RUN DETECTOR
# ------------------------------------------------------------
# This is the direct Python equivalent of the VBA loops.
#
# Logic:
# - start at row i
# - if row i is valid, store ref_speed = speed[i]
# - continue while:
#     * row is valid
#     * abs(speed - ref_speed) <= tol_speed
# - if run length >= min_points, keep it
# - else discard it
# ============================================================
def detect_consecutive_runs(
    df: pd.DataFrame,
    validator: Callable[[pd.Series], bool],
    label: str,
    min_points: int = MIN_POINTS,
    tol_speed: float = TOL_SPEED,
) -> pd.Series:
    labels = pd.Series(index=df.index, data=pd.NA, dtype="object")
    i = 0
    n = len(df)

    while i < n:
        row = df.iloc[i]

        if validator(row):
            ref_speed = row[SPEED_COL]
            start_i = i
            j = i

            while j < n:
                current_row = df.iloc[j]

                if validator(current_row):
                    current_speed = current_row[SPEED_COL]
                    if abs(current_speed - ref_speed) <= tol_speed:
                        j += 1
                    else:
                        break
                else:
                    break

            run_length = j - start_i

            if run_length >= min_points:
                labels.iloc[start_i:j] = label

            i = j
        else:
            i += 1

    return labels


# ============================================================
# APPLY MACRO-STYLE SEGMENTATION
# ------------------------------------------------------------
# This reproduces the Excel macro logic in Python.
#
# Important:
# A row can theoretically qualify for more than one pass if rules
# overlap, so we apply priority when combining labels.
# The VBA wrote to separate columns; here we combine them into one.
# ============================================================
def apply_segmentation_macro_style(
    df: pd.DataFrame,
    min_points: int = MIN_POINTS,
    tol_speed: float = TOL_SPEED,
    pit_tol_speed: float | None = None,
    pit_lat_acc_max: float = 0.15,
    pit_long_acc_max: float = 0.15,
    pit_speed_min: float = 1.0,
    pit_speed_max: float = 75.0,
    pit_constant_window: int | None = None,
    pit_constant_speed_std_max: float | None = None,
    pit_constant_min_points: int | None = None,
) -> pd.DataFrame:
    df = df.copy()

    # Ensure numeric
    for col in [LAT_COL, LONG_COL, SPEED_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Detect runs exactly like VBA did in separate columns
    df["straight_candidate"] = detect_consecutive_runs(
        df=df,
        validator=is_row_valid_for_straight,
        label="straight",
        min_points=min_points,
        tol_speed=tol_speed,
    )

    df["corner_candidate"] = detect_consecutive_runs(
        df=df,
        validator=is_row_valid_for_corner,
        label="corner",
        min_points=min_points,
        tol_speed=tol_speed,
    )

    df["pit_candidate"] = detect_consecutive_runs(
        df=df,
        validator=lambda row: is_row_valid_for_pitlane(
            row,
            lat_acc_max=pit_lat_acc_max,
            long_acc_max=pit_long_acc_max,
            speed_min=pit_speed_min,
            speed_max=pit_speed_max,
        ),
        label="pit",
        min_points=min_points,
        tol_speed=pit_tol_speed if pit_tol_speed is not None else tol_speed,
    )

    if (
        pit_constant_window is not None
        and pit_constant_speed_std_max is not None
    ):
        df["pit_candidate"] = keep_constant_speed_subruns(
            df=df,
            labels=df["pit_candidate"],
            label="pit",
            window=pit_constant_window,
            speed_std_max=pit_constant_speed_std_max,
            min_points=(
                pit_constant_min_points
                if pit_constant_min_points is not None
                else min_points
            ),
        )

    # Combine into one final label column
    # Priority chosen to avoid pit getting overwritten by straight.
    df["segment_final"] = pd.NA
    df.loc[df["pit_candidate"].notna(), "segment_final"] = "pit"
    df.loc[df["corner_candidate"].notna(), "segment_final"] = "corner"
    df.loc[
        df["straight_candidate"].notna() & df["segment_final"].isna(),
        "segment_final"
    ] = "straight"

    # Everything else remains unlabeled / transition
    df["segment_final"] = df["segment_final"].fillna("transition")

    # Final continuous segment id
    df["segment_id"] = (
        df["segment_final"] != df["segment_final"].shift()
    ).cumsum()

    return df
