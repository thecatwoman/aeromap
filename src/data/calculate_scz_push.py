import numpy as np
import pandas as pd


SCZ_PUSH_F_REQUIRED_COLUMNS = [
    "pushavg_c",
    "pushavd_c",
    "st_unsprungmassf",
    "avg_accx",
    "st_carmass",
    "st_carwd",
    "st_fuelstartrunsheet",
    "st_fuelwd",
    "mfuelcons2",
    "st_unsprungmassr",
    "st_carwheelbase",
    "pbrakef",
    "st_fgeo_antidive",
    "hdcav_c",
    "hdcar_c",
    "rhposition_f_f3",
    "rhposition_r_f3",
    "st_cgh",
    "st_driverwd",
    "pitot_c",
    "st_airtemp_fia",
    "altitude",
]

SCZ_PUSH_R_REQUIRED_COLUMNS = [
    "pusharg_c",
    "pushard_c",
    "st_unsprungmassr",
    "avg_accx",
    "st_carmass",
    "st_carwd",
    "st_fuelstartrunsheet",
    "st_fuelwd",
    "mfuelcons2",
    "st_unsprungmassf",
    "st_carwheelbase",
    "pbrakef",
    "engload_map",
    "st_rgeo_antilift",
    "st_rgeo_antisquat",
    "hdcav_c",
    "hdcar_c",
    "rhposition_f_f3",
    "rhposition_r_f3",
    "st_cgh",
    "st_driverwd",
    "pitot_c",
    "st_airtemp_fia",
    "altitude",
]


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _sind_deg(values_deg: pd.Series) -> pd.Series:
    return pd.Series(np.sin(np.deg2rad(values_deg.to_numpy(dtype="float64"))), index=values_deg.index)


def _asind(values: pd.Series) -> pd.Series:
    clipped = np.clip(values.to_numpy(dtype="float64"), -1.0, 1.0)
    return pd.Series(np.degrees(np.arcsin(clipped)), index=values.index)


def can_calculate_scz_push_f(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in SCZ_PUSH_F_REQUIRED_COLUMNS)


def can_calculate_scz_push_r(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in SCZ_PUSH_R_REQUIRED_COLUMNS)


def calculate_scz_push_f(df: pd.DataFrame, smooth_window: int = 100) -> pd.DataFrame:
    if not can_calculate_scz_push_f(df):
        return df

    out = df.copy()

    if "scz_push_f_pitot" in out.columns and "scz_push_f_pitot_engineer" not in out.columns:
        out["scz_push_f_pitot_engineer"] = out["scz_push_f_pitot"]

    pushavg_c = _series(out, "pushavg_c")
    pushavd_c = _series(out, "pushavd_c")
    st_unsprungmassf = _series(out, "st_unsprungmassf")
    avg_accx = _series(out, "avg_accx")
    st_carmass = _series(out, "st_carmass")
    st_carwd = _series(out, "st_carwd")
    st_fuelstartrunsheet = _series(out, "st_fuelstartrunsheet")
    st_fuelwd = _series(out, "st_fuelwd")
    mfuelcons2 = _series(out, "mfuelcons2")
    st_unsprungmassr = _series(out, "st_unsprungmassr")
    st_carwheelbase = _series(out, "st_carwheelbase")
    pbrakef = _series(out, "pbrakef")
    st_fgeo_antidive = _series(out, "st_fgeo_antidive")
    hdcav_c = _series(out, "hdcav_c")
    hdcar_c = _series(out, "hdcar_c")
    rhposition_f_f3 = _series(out, "rhposition_f_f3")
    rhposition_r_f3 = _series(out, "rhposition_r_f3")
    st_cgh = _series(out, "st_cgh")
    pitot_c = _series(out, "pitot_c")
    st_airtemp_fia = _series(out, "st_airtemp_fia")
    altitude = _series(out, "altitude")

    st_driverwd = _series(out, "st_driverwd")

    mass_static_f_calc = st_carmass * st_carwd / 100.0 + st_fuelstartrunsheet * st_fuelwd / 100.0
    mass_static_r_calc = st_carmass * (1.0 - st_carwd / 100.0) + st_fuelstartrunsheet * (1.0 - st_fuelwd / 100.0)

    pitch_input = (hdcar_c - hdcav_c) / (st_carwheelbase + rhposition_f_f3 - rhposition_r_f3)
    pitch_rh_calc = _asind(pitch_input)
    rh_f_calc = hdcav_c + _sind_deg(pitch_rh_calc) * rhposition_f_f3
    cgh_f3_calc = st_driverwd / 100.0 * st_carwheelbase * _sind_deg(pitch_rh_calc) + rh_f_calc + st_cgh

    sprung_mass_total = (
        mass_static_f_calc
        - mfuelcons2
        - st_unsprungmassf
        + mass_static_r_calc
        - st_unsprungmassr
    )

    brake_gate = (pbrakef > 2.0).astype("float64")
    mass_transfer_anti_longi_f_calc = -(
        sprung_mass_total
        * avg_accx
        * 9.81
        * cgh_f3_calc
        / st_carwheelbase
        * brake_gate
        * st_fgeo_antidive
        / 100.0
    )
    mass_transfertot_longi_f_calc = -(
        sprung_mass_total
        * avg_accx
        * 9.81
        * cgh_f3_calc
        / st_carwheelbase
    )

    fz_push_f_calc = pushavg_c + pushavd_c + st_unsprungmassf * 9.81
    fz_push_wanti_f_calc = fz_push_f_calc + mass_transfer_anti_longi_f_calc

    pdensity_calc = 1013.25 * np.power((1.0 - ((0.0065 * altitude) / 288.15)), 5.255)
    airdensity_calc = (1.0 / (287.06 * (st_airtemp_fia + 273.15))) * (
        pdensity_calc * 100.0
        - (230.617 * 0.2 * np.exp((17.5043 * st_airtemp_fia) / (241.2 + st_airtemp_fia)))
    )

    pitot_speed_input = 2.0 * pitot_c / airdensity_calc
    pitot_speed_calc = pd.Series(np.nan, index=out.index, dtype="float64")
    valid_speed = pitot_speed_input > 0
    pitot_speed_calc.loc[valid_speed] = np.sqrt(pitot_speed_input.loc[valid_speed]) * 3.6
    vpitot_calc = pitot_speed_calc.copy()

    dynamic_pressure = 0.5 * airdensity_calc * vpitot_calc * vpitot_calc / (3.6 * 3.6)
    scz_push_f_raw = (
        fz_push_wanti_f_calc
        - (mass_static_f_calc - mfuelcons2 * st_fuelwd / 100.0) * 9.81
        - mass_transfertot_longi_f_calc
    ) / dynamic_pressure
    scz_push_f_pitot_calc = scz_push_f_raw.rolling(window=smooth_window, min_periods=1).mean()

    out["mass_static_f_calc"] = mass_static_f_calc
    out["mass_static_r_calc"] = mass_static_r_calc
    out["pitch_rh_calc"] = pitch_rh_calc
    out["rh_f_calc"] = rh_f_calc
    out["cgh_f3_calc"] = cgh_f3_calc
    out["mass_transfer_anti_longi_f_calc"] = mass_transfer_anti_longi_f_calc
    out["mass_transfertot_longi_f_calc"] = mass_transfertot_longi_f_calc
    out["fz_push_f_calc"] = fz_push_f_calc
    out["fz_push_wanti_f_calc"] = fz_push_wanti_f_calc
    out["pdensity_calc"] = pdensity_calc
    out["airdensity_calc"] = airdensity_calc
    out["pitot_speed_calc"] = pitot_speed_calc
    out["vpitot_calc"] = vpitot_calc
    out["scz_push_f_pitot_calc"] = scz_push_f_pitot_calc

    out["scz_push_f_pitot"] = scz_push_f_pitot_calc

    return out


def calculate_scz_push_r(df: pd.DataFrame, smooth_window: int = 100) -> pd.DataFrame:
    if not can_calculate_scz_push_r(df):
        return df

    out = df.copy()

    if "scz_push_r_pitot" in out.columns and "scz_push_r_pitot_engineer" not in out.columns:
        out["scz_push_r_pitot_engineer"] = out["scz_push_r_pitot"]

    pusharg_c = _series(out, "pusharg_c")
    pushard_c = _series(out, "pushard_c")
    st_unsprungmassr = _series(out, "st_unsprungmassr")
    avg_accx = _series(out, "avg_accx")
    st_carmass = _series(out, "st_carmass")
    st_carwd = _series(out, "st_carwd")
    st_fuelstartrunsheet = _series(out, "st_fuelstartrunsheet")
    st_fuelwd = _series(out, "st_fuelwd")
    mfuelcons2 = _series(out, "mfuelcons2")
    st_unsprungmassf = _series(out, "st_unsprungmassf")
    st_carwheelbase = _series(out, "st_carwheelbase")
    pbrakef = _series(out, "pbrakef")
    engload_map = _series(out, "engload_map")
    st_rgeo_antilift = _series(out, "st_rgeo_antilift")
    st_rgeo_antisquat = _series(out, "st_rgeo_antisquat")
    hdcav_c = _series(out, "hdcav_c")
    hdcar_c = _series(out, "hdcar_c")
    rhposition_f_f3 = _series(out, "rhposition_f_f3")
    rhposition_r_f3 = _series(out, "rhposition_r_f3")
    st_cgh = _series(out, "st_cgh")
    pitot_c = _series(out, "pitot_c")
    st_airtemp_fia = _series(out, "st_airtemp_fia")
    altitude = _series(out, "altitude")
    st_driverwd = _series(out, "st_driverwd")

    mass_static_f_calc = _series(out, "mass_static_f_calc")
    if mass_static_f_calc.isna().all():
        mass_static_f_calc = st_carmass * st_carwd / 100.0 + st_fuelstartrunsheet * st_fuelwd / 100.0

    mass_static_r_calc = _series(out, "mass_static_r_calc")
    if mass_static_r_calc.isna().all():
        mass_static_r_calc = st_carmass * (1.0 - st_carwd / 100.0) + st_fuelstartrunsheet * (1.0 - st_fuelwd / 100.0)

    pitch_rh_calc = _series(out, "pitch_rh_calc")
    rh_f_calc = _series(out, "rh_f_calc")
    cgh_f3_calc = _series(out, "cgh_f3_calc")
    if pitch_rh_calc.isna().all() or rh_f_calc.isna().all() or cgh_f3_calc.isna().all():
        pitch_input = (hdcar_c - hdcav_c) / (st_carwheelbase + rhposition_f_f3 - rhposition_r_f3)
        pitch_rh_calc = _asind(pitch_input)
        rh_f_calc = hdcav_c + _sind_deg(pitch_rh_calc) * rhposition_f_f3
        cgh_f3_calc = st_driverwd / 100.0 * st_carwheelbase * _sind_deg(pitch_rh_calc) + rh_f_calc + st_cgh

    sprung_mass_total = (
        mass_static_f_calc
        - mfuelcons2
        - st_unsprungmassf
        + mass_static_r_calc
        - st_unsprungmassr
    )

    brake_gate = (pbrakef > 2.0).astype("float64")
    engload_gate = (engload_map > 2.0).astype("float64")
    rear_geo_factor = (
        brake_gate * st_rgeo_antilift / 100.0
        + engload_gate * st_rgeo_antisquat / 100.0
    )
    mass_transfer_anti_longi_r_calc = -(
        sprung_mass_total
        * avg_accx
        * 9.81
        * cgh_f3_calc
        / st_carwheelbase
        * rear_geo_factor
    )
    mass_transfertot_longi_r_calc = -(
        sprung_mass_total
        * avg_accx
        * 9.81
        * cgh_f3_calc
        / st_carwheelbase
    )

    fz_push_r_calc = pusharg_c + pushard_c + st_unsprungmassr * 9.81
    fz_push_wanti_r_calc = fz_push_r_calc + mass_transfer_anti_longi_r_calc

    pdensity_calc = _series(out, "pdensity_calc")
    if pdensity_calc.isna().all():
        pdensity_calc = 1013.25 * np.power((1.0 - ((0.0065 * altitude) / 288.15)), 5.255)

    airdensity_calc = _series(out, "airdensity_calc")
    if airdensity_calc.isna().all():
        airdensity_calc = (1.0 / (287.06 * (st_airtemp_fia + 273.15))) * (
            pdensity_calc * 100.0
            - (230.617 * 0.2 * np.exp((17.5043 * st_airtemp_fia) / (241.2 + st_airtemp_fia)))
        )

    pitot_speed_calc = _series(out, "pitot_speed_calc")
    vpitot_calc = _series(out, "vpitot_calc")
    if pitot_speed_calc.isna().all() or vpitot_calc.isna().all():
        pitot_speed_input = 2.0 * pitot_c / airdensity_calc
        pitot_speed_calc = pd.Series(np.nan, index=out.index, dtype="float64")
        valid_speed = pitot_speed_input > 0
        pitot_speed_calc.loc[valid_speed] = np.sqrt(pitot_speed_input.loc[valid_speed]) * 3.6
        vpitot_calc = pitot_speed_calc.copy()

    dynamic_pressure = 0.5 * airdensity_calc * vpitot_calc * vpitot_calc / (3.6 * 3.6)
    scz_push_r_raw = (
        fz_push_wanti_r_calc
        - (mass_static_r_calc - mfuelcons2 * (1.0 - st_fuelwd / 100.0)) * 9.81
        - mass_transfertot_longi_r_calc
    ) / dynamic_pressure
    scz_push_r_pitot_calc = scz_push_r_raw.rolling(window=smooth_window, min_periods=1).mean()

    out["mass_transfer_anti_longi_r_calc"] = mass_transfer_anti_longi_r_calc
    out["mass_transfertot_longi_r_calc"] = mass_transfertot_longi_r_calc
    out["fz_push_r_calc"] = fz_push_r_calc
    out["fz_push_wanti_r_calc"] = fz_push_wanti_r_calc
    out["scz_push_r_pitot_calc"] = scz_push_r_pitot_calc

    out["scz_push_r_pitot"] = scz_push_r_pitot_calc

    return out
