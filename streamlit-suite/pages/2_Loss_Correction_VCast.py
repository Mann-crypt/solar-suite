# ============================================================
# STREAMLIT APP
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
#
# Clean version:
# - No duplicate read_input_file()
# - load_area_efficiency() is defined
# - st.cache_data returns only pickle-safe objects
# - Fixed Error % range/step is editable
# - Fixed optimization criterion is editable
# - Tracking parameters/bounds are editable
# - No ExcelFile object is cached/returned
# - Tracking no longer depends on hidden _file_bytes
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Forecast Correction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fa;
    }

    .main-header {
        font-size: 30px;
        font-weight: 700;
        color: #17202a;
        margin-bottom: 4px;
    }

    .sub-header {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 20px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        min-height: 112px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .metric-title {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
    }

    .metric-sub {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 5px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 650;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .info-box {
        background: #ffffff;
        border-left: 4px solid #4f46e5;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
    }

    .stDownloadButton button {
        width: 100%;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

N_CLUSTERS = len(CLUSTERS)


# ============================================================
# UI HELPERS
# ============================================================

def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_results():
    st.session_state.fixed_result = None
    st.session_state.tracking_result = None


# ============================================================
# WORKBOOK VALIDATION
# ============================================================

@st.cache_data(show_spinner=False)
def read_input_file(file_bytes):
    """
    Returns only pickle-serializable data.
    NEVER return pd.ExcelFile from a cached function.
    """

    required_sheets = [
        "Area & Efficiency",
        "Forecast Config",
        "Config Tilt Angle",
        "Result",
        "Fixed-C11",
        "Tracking",
        "Backend Cal C11",
        "Backend Cal C12",
        "Backend Cal C13",
        "Backend Cal C14",
        "Backend Cal C15",
    ]

    try:
        excel = pd.ExcelFile(
            io.BytesIO(file_bytes),
            engine="openpyxl",
        )

        sheet_names = list(excel.sheet_names)

    except Exception as exc:
        raise ValueError(
            f"Unable to open workbook: {exc}"
        ) from exc

    missing = [
        sheet for sheet in required_sheets
        if sheet not in sheet_names
    ]

    if missing:
        raise ValueError(
            "Missing required sheets: "
            + ", ".join(missing)
        )

    return sheet_names


# ============================================================
# LOAD AREA & EFFICIENCY
# ============================================================

@st.cache_data(show_spinner=False)
def load_area_efficiency(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        engine="openpyxl",
    )

    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    if "S.No." in df.columns:
        df = df[df["S.No."].notna()].copy()

    df.reset_index(drop=True, inplace=True)

    if "Standard PV Efficiency (%)" not in df.columns:
        raise ValueError(
            "Column 'Standard PV Efficiency (%)' was not found "
            "in Area & Efficiency."
        )

    standard_efficiency = (
        pd.to_numeric(
            df["Standard PV Efficiency (%)"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(standard_efficiency) < N_CLUSTERS:
        raise ValueError(
            "Less than 5 Standard PV Efficiency (%) values "
            "were found in Area & Efficiency."
        )

    standard_efficiency = standard_efficiency[:N_CLUSTERS]

    # Original workbook layout used by the calculation.
    area_raw = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=None,
        engine="openpyxl",
    )

    def read_weight_rows(start, end):
        values = pd.to_numeric(
            area_raw.iloc[start:end, 15],
            errors="coerce",
        ).fillna(0).to_numpy(dtype=float)

        if len(values) < N_CLUSTERS:
            values = np.pad(
                values,
                (0, N_CLUSTERS - len(values)),
                constant_values=0,
            )

        return values[:N_CLUSTERS]

    fixed_weights = read_weight_rows(2, 7)
    tracking_weights = read_weight_rows(28, 33)

    return (
        df,
        fixed_weights,
        tracking_weights,
        standard_efficiency,
    )


# ============================================================
# LOAD LATITUDE
# ============================================================

@st.cache_data(show_spinner=False)
def load_latitude(file_bytes):

    df_config = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
        engine="openpyxl",
    )

    df_config.columns = (
        pd.Index(df_config.columns)
        .astype(str)
        .str.strip()
    )

    if "Lat" not in df_config.columns:
        raise ValueError(
            "Column 'Lat' was not found in Forecast Config."
        )

    lat_series = pd.to_numeric(
        df_config["Lat"],
        errors="coerce",
    ).dropna()

    if lat_series.empty:
        raise ValueError(
            "No valid latitude was found in Forecast Config."
        )

    return float(lat_series.iloc[0])


# ============================================================
# LOAD TILT
# ============================================================

@st.cache_data(show_spinner=False)
def load_tilt(file_bytes):

    df_tilt = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle",
        header=7,
        engine="openpyxl",
    )

    df_tilt.columns = (
        pd.Index(df_tilt.columns)
        .astype(str)
        .str.strip()
    )

    # Support both named and Unnamed workbook columns.
    rename_map = {}

    if "Unnamed: 2" in df_tilt.columns:
        rename_map["Unnamed: 2"] = "Month_Num"

    if "Unnamed: 3" in df_tilt.columns:
        rename_map["Unnamed: 3"] = "Month"

    df_tilt = df_tilt.rename(columns=rename_map)

    if "Month_Num" not in df_tilt.columns:
        # Fall back to the third column.
        if len(df_tilt.columns) >= 3:
            df_tilt = df_tilt.rename(
                columns={df_tilt.columns[2]: "Month_Num"}
            )
        else:
            raise ValueError(
                "Could not identify Month number in Config Tilt Angle."
            )

    if "Fixed" not in df_tilt.columns:
        raise ValueError(
            "Column 'Fixed' was not found in Config Tilt Angle."
        )

    df_tilt["Month_Num"] = pd.to_numeric(
        df_tilt["Month_Num"],
        errors="coerce",
    )

    df_tilt["Fixed"] = pd.to_numeric(
        df_tilt["Fixed"],
        errors="coerce",
    )

    df_tilt = df_tilt.dropna(
        subset=["Month_Num", "Fixed"]
    )

    return (
        df_tilt
        .drop_duplicates("Month_Num")
        .set_index("Month_Num")["Fixed"]
        .to_dict()
    )


# ============================================================
# LOAD GHI
# ============================================================

@st.cache_data(show_spinner=False)
def load_ghi(file_bytes):

    df_ghi = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Result",
        usecols=range(6),
        engine="openpyxl",
    )

    if len(df_ghi.columns) < 6:
        raise ValueError(
            "Result sheet must contain at least 6 columns."
        )

    df_ghi = df_ghi.iloc[:, :6].copy()

    df_ghi.columns = [
        "Block",
        *GHI_COLS,
    ]

    block_numeric = pd.to_numeric(
        df_ghi["Block"],
        errors="coerce",
    )

    df_ghi = df_ghi[
        block_numeric.notna()
    ].copy()

    df_ghi["Block"] = block_numeric.loc[
        df_ghi.index
    ].to_numpy(dtype=float)

    for col in GHI_COLS:
        df_ghi[col] = pd.to_numeric(
            df_ghi[col],
            errors="coerce",
        ).fillna(0.0)

    blocks = df_ghi["Block"].to_numpy(dtype=float)

    ghi_matrix = df_ghi[
        GHI_COLS
    ].to_numpy(dtype=float)

    return (
        df_ghi.reset_index(drop=True),
        blocks,
        ghi_matrix,
    )


# ============================================================
# LOAD FIXED DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_fixed_data(file_bytes):

    df_fix = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Fixed-C11",
        header=1,
        engine="openpyxl",
    )

    df_fix.columns = (
        pd.Index(df_fix.columns)
        .astype(str)
        .str.strip()
    )

    if "Date" not in df_fix.columns:
        raise ValueError(
            "Column 'Date' was not found in Fixed-C11."
        )

    if "Actual" not in df_fix.columns:
        raise ValueError(
            "Column 'Actual' was not found in Fixed-C11."
        )

    dates = pd.to_datetime(
        df_fix["Date"],
        errors="coerce",
    )

    valid_dates = dates.notna()

    if not valid_dates.any():
        raise ValueError(
            "No valid Date rows found in Fixed-C11."
        )

    # Keep rows until the first blank date, matching the
    # original workbook behavior.
    invalid_positions = np.where(
        ~valid_dates.to_numpy()
    )[0]

    if len(invalid_positions) > 0:
        first_blank = invalid_positions[0]
        df_fix = df_fix.iloc[:first_blank].copy()
    else:
        df_fix = df_fix.loc[valid_dates].copy()

    df_fix.reset_index(drop=True, inplace=True)

    return df_fix


# ============================================================
# PREPARE SOLAR DATA
# ============================================================

def prepare_solar_data(
    df_fix,
    blocks_result,
    ghi_matrix,
    lat,
    month_number_to_tilt,
):

    n = min(
        len(df_fix),
        len(blocks_result),
        len(ghi_matrix),
    )

    if n <= 0:
        raise ValueError(
            "No valid forecast rows are available."
        )

    df_fix = df_fix.iloc[:n].copy()

    actual = (
        pd.to_numeric(
            df_fix["Actual"],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)[:n]
    )

    dates = pd.to_datetime(
        df_fix["Date"],
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError(
            "Invalid dates found in Fixed-C11."
        )

    blocks = np.asarray(
        blocks_result[:n],
        dtype=float,
    )

    ghi_matrix = np.asarray(
        ghi_matrix[:n],
        dtype=float,
    )

    # Solar declination.
    # Using day-of-year avoids dependence on a hard-coded year.
    day_of_year = (
        dates.dt.dayofyear.to_numpy(dtype=float)
    )

    declination = (
        23.45
        * np.sin(
            np.radians(
                360.0
                * (284.0 + day_of_year)
                / 365.0
            )
        )
    )

    elevation = (
        90.0
        - float(lat)
        + declination
    )

    months = dates.dt.month.to_numpy()

    tilt = np.array(
        [
            month_number_to_tilt.get(
                float(month),
                0.0,
            )
            for month in months
        ],
        dtype=float,
    )

    a_plus_b = elevation + tilt

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(a_plus_b)
    )

    sin_a_safe = np.where(
        np.abs(sin_a) < 1e-8,
        np.where(sin_a < 0, -1e-8, 1e-8),
        sin_a,
    )

    fixed_poa = (
        ghi_matrix
        * sin_ab[:, None]
        / sin_a_safe[:, None]
    )

    fixed_poa = np.nan_to_num(
        fixed_poa,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    valid_mask = (
        np.isfinite(actual)
        & (actual > 0)
    )

    if not valid_mask.any():
        raise ValueError(
            "Actual power contains no valid positive values."
        )

    actual_day = actual[valid_mask]

    actual_peak = float(
        np.max(actual_day)
    )

    actual_energy = float(
        np.sum(actual_day)
    )

    if actual_peak <= 0:
        raise ValueError(
            "Actual peak must be greater than zero."
        )

    if actual_energy <= 0:
        raise ValueError(
            "Actual energy must be greater than zero."
        )

    return {
        "df_fix": df_fix,
        "actual": actual,
        "dates": dates,
        "blocks": blocks,
        "ghi_matrix": ghi_matrix,
        "fixed_poa": fixed_poa,
        "valid_mask": valid_mask,
        "actual_day": actual_day,
        "actual_peak": actual_peak,
        "actual_energy": actual_energy,
        "n": n,
    }


# ============================================================
# SCORE FUNCTION
# ============================================================

def calculate_metrics(
    actual_day,
    predicted_day,
    actual_peak,
    actual_energy,
):
    predicted_day = np.asarray(
        predicted_day,
        dtype=float,
    )

    predicted_day = np.nan_to_num(
        predicted_day,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if len(predicted_day) != len(actual_day):
        raise ValueError(
            "Actual and predicted arrays have different lengths."
        )

    predicted_peak = float(
        np.max(predicted_day)
    )

    peak_error_abs = abs(
        actual_peak - predicted_peak
    )

    peak_error_pct = (
        peak_error_abs
        / actual_peak
        * 100.0
    )

    block_error = (
        np.mean(
            np.abs(
                actual_day - predicted_day
            )
        )
        / actual_peak
    )

    energy_error = (
        abs(
            actual_energy - np.sum(predicted_day)
        )
        / actual_energy
    )

    peak_error_relative = (
        peak_error_abs / actual_peak
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error_relative
        + 0.10 * energy_error
    )

    return {
        "predicted_peak": predicted_peak,
        "peak_error_abs": peak_error_abs,
        "peak_error_pct": peak_error_pct,
        "block_error": block_error,
        "energy_error": energy_error,
        "score": score,
    }


# ============================================================
# FIXED MODEL
# ============================================================

def run_fixed_model(
    solar,
    fixed_weights,
    standard_efficiency,
    error_min,
    error_max,
    error_step,
    optimize_by,
):
    fixed_weights = np.asarray(
        fixed_weights,
        dtype=float,
    )

    standard_efficiency = np.asarray(
        standard_efficiency,
        dtype=float,
    )

    if len(fixed_weights) != N_CLUSTERS:
        raise ValueError(
            "Fixed area/weight data must contain 5 clusters."
        )

    if len(standard_efficiency) != N_CLUSTERS:
        raise ValueError(
            "Efficiency data must contain 5 clusters."
        )

    if error_step <= 0:
        raise ValueError(
            "Error % step must be greater than zero."
        )

    if error_min < 0:
        raise ValueError(
            "Error % minimum cannot be negative."
        )

    if error_max <= error_min:
        raise ValueError(
            "Error % maximum must be greater than minimum."
        )

    # Error cannot reduce efficiency below zero.
    physical_max = float(
        np.min(standard_efficiency)
    )

    effective_max = min(
        float(error_max),
        physical_max,
    )

    if effective_max < error_min:
        raise ValueError(
            "The selected Error % range is above the available "
            "standard efficiency."
        )

    # Include the upper boundary reliably.
    loss_values = np.arange(
        float(error_min),
        effective_max + float(error_step) * 0.5,
        float(error_step),
    )

    loss_values = np.round(
        loss_values,
        10,
    )

    loss_values = loss_values[
        loss_values <= effective_max + 1e-9
    ]

    if len(loss_values) == 0:
        raise ValueError(
            "No valid Error % values were generated."
        )

    actual_day = solar["actual_day"]
    actual_peak = solar["actual_peak"]
    actual_energy = solar["actual_energy"]
    valid_mask = solar["valid_mask"]
    fixed_poa = solar["fixed_poa"]

    rows = []

    for error in loss_values:

        net_efficiency = np.maximum(
            standard_efficiency - error,
            0.0,
        )

        efficiency_factor = np.divide(
            net_efficiency,
            standard_efficiency,
            out=np.zeros_like(net_efficiency),
            where=standard_efficiency != 0,
        )

        adjusted_weights = (
            fixed_weights * efficiency_factor
        )

        power_matrix = (
            fixed_poa
            * adjusted_weights[None, :]
            / 1_000_000.0
        )

        predicted = power_matrix.sum(axis=1)

        predicted_day = predicted[valid_mask]

        metrics = calculate_metrics(
            actual_day,
            predicted_day,
            actual_peak,
            actual_energy,
        )

        rows.append(
            {
                "Error %": float(error),
                "Actual Peak": actual_peak,
                "Predicted Peak": metrics["predicted_peak"],
                "Peak Error": metrics["peak_error_abs"],
                "Peak Error (%)": metrics["peak_error_pct"],
                "Block Error": metrics["block_error"],
                "Energy Error": metrics["energy_error"],
                "Overall Score": metrics["score"],
            }
        )

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        raise ValueError(
            "Fixed optimization produced no results."
        )

    criterion_map = {
        "Peak Error": "Peak Error",
        "Overall Score": "Overall Score",
        "Block Error": "Block Error",
        "Energy Error": "Energy Error",
    }

    selected_column = criterion_map[optimize_by]

    best_idx = results_df[selected_column].idxmin()
    best_row = results_df.loc[best_idx]

    best_error = float(
        best_row["Error %"]
    )

    net_efficiency = np.maximum(
        standard_efficiency - best_error,
        0.0,
    )

    efficiency_factor = np.divide(
        net_efficiency,
        standard_efficiency,
        out=np.zeros_like(net_efficiency),
        where=standard_efficiency != 0,
    )

    final_weights = (
        fixed_weights * efficiency_factor
    )

    power_matrix = (
        fixed_poa
        * final_weights[None, :]
        / 1_000_000.0
    )

    forecast = power_matrix.sum(axis=1)

    fixed_day = forecast[valid_mask]

    final_metrics = calculate_metrics(
        actual_day,
        fixed_day,
        actual_peak,
        actual_energy,
    )

    df_fixed = solar["df_fix"].copy()

    for i, cluster in enumerate(CLUSTERS):
        df_fixed[
            f"{cluster}_Fixed Power=I*Ƞ*A"
        ] = power_matrix[:, i]

    df_fixed[
        "Total Power (CL1+CL2+…)"
    ] = forecast

    return {
        "best_error": best_error,
        "net_efficiency": net_efficiency,
        "final_weights": final_weights,
        "power_matrix": power_matrix,
        "forecast": forecast,
        "block_error": final_metrics["block_error"],
        "peak_error": final_metrics["peak_error_abs"]
        / actual_peak,
        "peak_error_pct": final_metrics["peak_error_pct"],
        "energy_error": final_metrics["energy_error"],
        "score": final_metrics["score"],
        "results_df": results_df,
        "df_fixed": df_fixed,
        "optimize_by": optimize_by,
    }


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
    blocks,
    ghi_matrix,
    tracking_weights,
):
    if not (
        start_block < max_block < end_block
    ):
        return None

    denominator_1 = (
        start_block - 1 - max_block
    )

    denominator_2 = (
        end_block + 1 - max_block
    )

    if denominator_1 == 0 or denominator_2 == 0:
        return None

    m1 = 90.0 / denominator_1
    m2 = 90.0 / denominator_2

    # Zenith should be positive during the day.
    # The original sign convention is retained.
    zenith = np.where(
        blocks <= max_block,
        np.minimum(
            89.0,
            m1 * (blocks - max_block),
        ),
        np.minimum(
            89.0,
            m2 * (blocks - max_block),
        ),
    )

    panel = np.where(
        blocks < max_block,
        np.where(
            zenith < abs(east_limit),
            zenith,
            abs(east_limit),
        ),
        np.where(
            (blocks > max_block)
            & (zenith > west_limit),
            west_limit,
            zenith,
        ),
    )

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None,
    )

    dhi = (
        ghi_matrix * float(DHI) / 100.0
    )

    dni = (
        ghi_matrix - dhi
    ) / cos_alpha[:, None]

    # Do not allow negative irradiance to create negative power.
    dni = np.maximum(
        dni,
        0.0,
    )

    tracking_power_matrix = (
        dni
        * tracking_weights[None, :]
        / 1_000_000.0
    )

    tracking_forecast = (
        tracking_power_matrix.sum(axis=1)
    )

    return (
        tracking_forecast,
        tracking_power_matrix,
        zenith,
        panel,
        dni,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def run_tracking_model(
    solar,
    tracking_weights,
    bounds,
    maxiter=40,
    popsize=15,
    optimize_by="Overall Score",
):
    actual_day = solar["actual_day"]
    actual_peak = solar["actual_peak"]
    actual_energy = solar["actual_energy"]
    valid_mask = solar["valid_mask"]
    blocks = solar["blocks"]
    ghi_matrix = solar["ghi_matrix"]

    def objective(x):

        DHI = int(round(x[0]))
        start_block = int(round(x[1]))
        end_block = int(round(x[2]))
        max_block = int(round(x[3]))
        east_limit = int(round(x[4]))
        west_limit = int(round(x[5]))

        if not (
            start_block < max_block < end_block
        ):
            return 1e9

        result = calculate_tracking(
            DHI,
            start_block,
            end_block,
            max_block,
            east_limit,
            west_limit,
            blocks,
            ghi_matrix,
            tracking_weights,
        )

        if result is None:
            return 1e9

        prediction = result[0]

        if not np.all(np.isfinite(prediction)):
            return 1e9

        prediction_day = prediction[valid_mask]

        if len(prediction_day) == 0:
            return 1e9

        metrics = calculate_metrics(
            actual_day,
            prediction_day,
            actual_peak,
            actual_energy,
        )

        if optimize_by == "Peak Error":
            return metrics["peak_error_abs"] / actual_peak

        if optimize_by == "Block Error":
            return metrics["block_error"]

        if optimize_by == "Energy Error":
            return metrics["energy_error"]

        return metrics["score"]

    scipy_bounds = [
        tuple(map(float, b))
        for b in bounds
    ]

    result = differential_evolution(
        objective,
        bounds=scipy_bounds,
        strategy="best1bin",
        maxiter=int(maxiter),
        popsize=int(popsize),
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    best = np.rint(
        result.x
    ).astype(int)

    DHI = int(best[0])
    start_block = int(best[1])
    end_block = int(best[2])
    max_block = int(best[3])
    east_limit = int(best[4])
    west_limit = int(best[5])

    final = calculate_tracking(
        DHI,
        start_block,
        end_block,
        max_block,
        east_limit,
        west_limit,
        blocks,
        ghi_matrix,
        tracking_weights,
    )

    if final is None:
        raise ValueError(
            "Tracking optimizer returned invalid parameters."
        )

    (
        forecast,
        power_matrix,
        zenith,
        panel,
        dni,
    ) = final

    tracking_day = forecast[valid_mask]

    metrics = calculate_metrics(
        actual_day,
        tracking_day,
        actual_peak,
        actual_energy,
    )

    df_tracking = solar["df_fix"].copy()

    df_tracking["Zenith Angle"] = zenith
    df_tracking["Panel Angle"] = panel

    for i, cluster in enumerate(CLUSTERS):
        df_tracking[
            f"{cluster}_Tracking Power=I*Ƞ*A"
        ] = power_matrix[:, i]

    df_tracking[
        "Tracking Power=I*Ƞ*A"
    ] = forecast

    return {
        "DHI": DHI,
        "start_block": start_block,
        "end_block": end_block,
        "max_block": max_block,
        "east_limit": east_limit,
        "west_limit": west_limit,
        "forecast": forecast,
        "power_matrix": power_matrix,
        "zenith": zenith,
        "panel": panel,
        "dni": dni,
        "block_error": metrics["block_error"],
        "peak_error": (
            metrics["peak_error_abs"]
            / actual_peak
        ),
        "peak_error_pct": metrics["peak_error_pct"],
        "energy_error": metrics["energy_error"],
        "score": metrics["score"],
        "optimizer_score": float(result.fun),
        "df_tracking": df_tracking,
        "optimize_by": optimize_by,
    }


# ============================================================
# EXCEL EXPORT
# ============================================================

def create_excel_download(
    mode,
    df_area,
    fixed_result=None,
    tracking_result=None,
    summary=None,
    optimized_parameters=None,
):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        df_area.to_excel(
            writer,
            sheet_name="Area & Efficiency",
            index=False,
        )

        if mode == "Fixed":

            fixed_result["df_fixed"].to_excel(
                writer,
                sheet_name="Fixed Results",
                index=False,
            )

            fixed_result["results_df"].to_excel(
                writer,
                sheet_name="Error Optimization",
                index=False,
            )

        else:

            tracking_result["df_tracking"].to_excel(
                writer,
                sheet_name="Tracking Results",
                index=False,
            )

        if summary is not None:
            summary.to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )

        if optimized_parameters is not None:
            optimized_parameters.to_excel(
                writer,
                sheet_name="Optimized Parameters",
                index=False,
            )

    output.seek(0)
    return output.getvalue()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-header">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-header">'
    'Fixed and Tracking solar forecast optimization'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "fixed_result" not in st.session_state:
    st.session_state.fixed_result = None

if "tracking_result" not in st.session_state:
    st.session_state.tracking_result = None

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### ⚙️ Input Configuration")

    uploaded_file = st.file_uploader(
        "Upload Excel Workbook",
        type=["xlsx", "xls"],
    )

    if uploaded_file is not None:

        if (
            st.session_state.last_file_name
            != uploaded_file.name
        ):
            st.session_state.last_file_name = uploaded_file.name
            clear_results()

    st.markdown("---")

    st.markdown(
        """
        **Required sheets**

        • Area & Efficiency  
        • Forecast Config  
        • Config Tilt Angle  
        • Result  
        • Fixed-C11  
        • Tracking  
        • Backend Cal C11-C15
        """
    )


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:
    st.info(
        "Upload the solar Excel workbook from the sidebar to begin."
    )
    st.stop()


# ============================================================
# FILE BYTES
# ============================================================

file_bytes = uploaded_file.getvalue()


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    with st.spinner("Reading workbook..."):

        sheet_names = read_input_file(
            file_bytes
        )

        (
            df_area,
            fixed_weights,
            tracking_weights,
            standard_efficiency,
        ) = load_area_efficiency(
            file_bytes
        )

        lat = load_latitude(
            file_bytes
        )

        month_number_to_tilt = load_tilt(
            file_bytes
        )

        (
            df_ghi,
            blocks_result,
            ghi_matrix,
        ) = load_ghi(
            file_bytes
        )

        df_fix = load_fixed_data(
            file_bytes
        )

        solar = prepare_solar_data(
            df_fix=df_fix,
            blocks_result=blocks_result,
            ghi_matrix=ghi_matrix,
            lat=lat,
            month_number_to_tilt=month_number_to_tilt,
        )

except Exception as exc:

    st.error(
        f"Unable to read workbook: {exc}"
    )
    st.stop()


# ============================================================
# WORKBOOK OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Workbook Overview</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "Clusters",
        str(N_CLUSTERS),
        "C11 to C15",
    )

with c2:
    metric_card(
        "Forecast Blocks",
        f"{solar['n']:,}",
        "15-minute blocks",
    )

with c3:
    metric_card(
        "Latitude",
        f"{lat:.4f}°",
        "Forecast configuration",
    )

with c4:
    metric_card(
        "Actual Peak",
        f"{solar['actual_peak']:.4f}",
        "MW",
    )


# ============================================================
# MODEL SELECTION
# ============================================================

st.markdown(
    '<div class="section-title">Model Selection</div>',
    unsafe_allow_html=True,
)

mode = st.segmented_control(
    "Select model",
    options=["Fixed", "Tracking"],
    default="Fixed",
    key="model_mode",
    label_visibility="collapsed",
)


# ============================================================
# FIXED MODEL
# ============================================================

if mode == "Fixed":

    st.markdown(
        '<div class="info-box">'
        '<b>Fixed Plant</b><br>'
        'Edit the Error % range and step below. The app tests every '
        'Error % value in the selected range and chooses the value '
        'that minimizes the selected error criterion.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # EDITABLE FIXED PARAMETERS
    # --------------------------------------------------------

    with st.expander(
        "⚙️ Fixed Optimization Parameters",
        expanded=True,
    ):

        fc1, fc2, fc3 = st.columns(3)

        physical_max = float(
            np.min(standard_efficiency)
        )

        with fc1:
            error_min = st.number_input(
                "Error % Minimum",
                min_value=0.0,
                max_value=max(0.0, physical_max),
                value=0.0,
                step=0.1,
                format="%.2f",
                help="Starting Error % tested by the optimizer.",
            )

        with fc2:
            error_max = st.number_input(
                "Error % Maximum",
                min_value=0.1,
                max_value=max(0.1, physical_max),
                value=float(
                    min(10.0, physical_max)
                ),
                step=0.1,
                format="%.2f",
                help=(
                    "Maximum Error % tested. It is automatically "
                    "limited by the lowest Standard PV Efficiency."
                ),
            )

        with fc3:
            error_step = st.number_input(
                "Error % Step",
                min_value=0.01,
                max_value=5.0,
                value=0.1,
                step=0.01,
                format="%.2f",
                help=(
                    "Smaller step = finer Error % search. "
                    "For example 0.1 tests 0.0, 0.1, 0.2..."
                ),
            )

        optimize_by = st.selectbox(
            "Optimize Error % By",
            [
                "Peak Error",
                "Overall Score",
                "Block Error",
                "Energy Error",
            ],
            index=0,
            help=(
                "Peak Error is recommended when the main objective "
                "is to match the actual peak."
            ),
        )

        st.caption(
            f"Available physical Error % range: "
            f"0.00% to {physical_max:.2f}%"
        )

    run_fixed = st.button(
        "▶ Run Fixed Optimization",
        type="primary",
        use_container_width=True,
    )

    if run_fixed:

        if error_max <= error_min:
            st.error(
                "Error % Maximum must be greater than Error % Minimum."
            )
            st.stop()

        if error_step <= 0:
            st.error(
                "Error % Step must be greater than zero."
            )
            st.stop()

        with st.spinner(
            "Running Fixed Error % optimization..."
        ):

            try:
                fixed_result = run_fixed_model(
                    solar=solar,
                    fixed_weights=fixed_weights,
                    standard_efficiency=standard_efficiency,
                    error_min=error_min,
                    error_max=error_max,
                    error_step=error_step,
                    optimize_by=optimize_by,
                )

                st.session_state.fixed_result = fixed_result

            except Exception as exc:
                st.error(
                    f"Fixed optimization failed: {exc}"
                )
                st.stop()

    fixed_result = st.session_state.fixed_result

    if fixed_result is None:

        st.warning(
            "Set the parameters and click "
            "**Run Fixed Optimization**."
        )

    else:

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Fixed Model Results</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            metric_card(
                "Optimized Error %",
                f"{fixed_result['best_error']:.2f}%",
                f"Optimized by {fixed_result['optimize_by']}",
            )

        with c2:
            metric_card(
                "Peak Power",
                f"{fixed_result['forecast'].max():.4f}",
                "MW",
            )

        with c3:
            metric_card(
                "Peak Error",
                f"{fixed_result['peak_error_pct']:.3f}%",
                "Relative error",
            )

        with c4:
            metric_card(
                "Block Error",
                f"{fixed_result['block_error']:.5f}",
                "Normalized MAE",
            )

        with c5:
            metric_card(
                "Overall Score",
                f"{fixed_result['score']:.5f}",
                "Lower is better",
            )

        # ----------------------------------------------------
        # EFFICIENCY
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Efficiency Results</div>',
            unsafe_allow_html=True,
        )

        efficiency_table = pd.DataFrame(
            {
                "Cluster": CLUSTERS,
                "Standard Efficiency (%)": standard_efficiency,
                "Error (%)": [
                    fixed_result["best_error"]
                ] * N_CLUSTERS,
                "Net Efficiency (%)": fixed_result[
                    "net_efficiency"
                ],
                "Original Fixed Area (m²)": fixed_weights,
                "Final Effective Area (m²)": fixed_result[
                    "final_weights"
                ],
            }
        )

        st.dataframe(
            efficiency_table,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Actual vs Fixed Forecast'
            '</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.arange(solar["n"]),
                y=solar["actual"],
                name="Actual",
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(solar["n"]),
                y=fixed_result["forecast"],
                name="Fixed Forecast",
                mode="lines",
            )
        )

        fig.update_layout(
            height=450,
            xaxis_title="Block",
            yaxis_title="Power (MW)",
            hovermode="x unified",
            template="plotly_white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # ERROR OPTIMIZATION TABLE
        # ----------------------------------------------------

        with st.expander(
            "📊 View Error % Optimization Results",
            expanded=False,
        ):

            st.dataframe(
                fixed_result["results_df"],
                use_container_width=True,
                height=400,
            )

        # ----------------------------------------------------
        # CLUSTER POWER
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Fixed Cluster Power</div>',
            unsafe_allow_html=True,
        )

        cluster_df = pd.DataFrame(
            fixed_result["power_matrix"],
            columns=[
                f"{cluster} Fixed Power"
                for cluster in CLUSTERS
            ],
        )

        cluster_df.insert(
            0,
            "Block",
            np.arange(solar["n"]),
        )

        st.dataframe(
            cluster_df,
            use_container_width=True,
            height=350,
        )

        # ----------------------------------------------------
        # FINAL DATA
        # ----------------------------------------------------

        with st.expander(
            "📄 View Final Fixed Dataset",
            expanded=False,
        ):

            st.dataframe(
                fixed_result["df_fixed"],
                use_container_width=True,
                height=400,
            )

        # ----------------------------------------------------
        # EXPORT
        # ----------------------------------------------------

        optimized_parameters = pd.DataFrame(
            {
                "Parameter": [
                    "Optimization Criterion",
                    "Error % Minimum",
                    "Error % Maximum",
                    "Error % Step",
                    "Optimized Error %",
                    "Actual Peak",
                    "Fixed Predicted Peak",
                    "Fixed Peak Error (%)",
                    "Fixed Block Error",
                    "Fixed Energy Error",
                    "Fixed Overall Score",
                ],
                "Value": [
                    fixed_result["optimize_by"],
                    error_min,
                    error_max,
                    error_step,
                    fixed_result["best_error"],
                    solar["actual_peak"],
                    fixed_result["forecast"].max(),
                    fixed_result["peak_error_pct"],
                    fixed_result["block_error"],
                    fixed_result["energy_error"],
                    fixed_result["score"],
                ],
            }
        )

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Optimization Criterion",
                    "Error %",
                    "Block Error",
                    "Peak Error (%)",
                    "Energy Error",
                    "Overall Score",
                    "Peak Power",
                ],
                "Fixed": [
                    fixed_result["optimize_by"],
                    fixed_result["best_error"],
                    fixed_result["block_error"],
                    fixed_result["peak_error_pct"],
                    fixed_result["energy_error"],
                    fixed_result["score"],
                    fixed_result["forecast"].max(),
                ],
            }
        )

        excel_output = create_excel_download(
            mode="Fixed",
            df_area=df_area,
            fixed_result=fixed_result,
            summary=summary,
            optimized_parameters=optimized_parameters,
        )

        st.download_button(
            "⬇ Download Fixed Results",
            data=excel_output,
            file_name="Solar_Fixed_Results.xlsx",
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True,
        )


# ============================================================
# TRACKING MODEL
# ============================================================

else:

    st.markdown(
        '<div class="info-box">'
        '<b>Tracking Plant</b><br>'
        'All main tracking optimization parameters can be edited '
        'before running Differential Evolution.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TRACKING PARAMETERS
    # --------------------------------------------------------

    with st.expander(
        "⚙️ Tracking Optimization Parameters",
        expanded=True,
    ):

        st.markdown("#### Optimization Target")

        optimize_by_tracking = st.selectbox(
            "Optimize Tracking By",
            [
                "Overall Score",
                "Peak Error",
                "Block Error",
                "Energy Error",
            ],
            index=0,
            key="tracking_optimize_by",
            help=(
                "Select Peak Error if matching the actual peak "
                "is the primary objective."
            ),
        )

        st.markdown("#### DHI")

        t1, t2, t3 = st.columns(3)

        with t1:
            dhi_min = st.number_input(
                "DHI Minimum (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
            )

        with t2:
            dhi_max = st.number_input(
                "DHI Maximum (%)",
                min_value=0,
                max_value=100,
                value=10,
                step=1,
            )

        with t3:
            st.number_input(
                "DHI Step",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                disabled=True,
                help="Differential Evolution evaluates integer DHI values.",
            )

        st.markdown("#### GHI Blocks")

        b1, b2, b3, b4 = st.columns(4)

        with b1:
            start_min = st.number_input(
                "GHI Start Min",
                min_value=0,
                max_value=95,
                value=10,
                step=1,
            )

        with b2:
            start_max = st.number_input(
                "GHI Start Max",
                min_value=1,
                max_value=95,
                value=30,
                step=1,
            )

        with b3:
            end_min = st.number_input(
                "GHI End Min",
                min_value=1,
                max_value=95,
                value=65,
                step=1,
            )

        with b4:
            end_max = st.number_input(
                "GHI End Max",
                min_value=2,
                max_value=96,
                value=80,
                step=1,
            )

        st.markdown("#### GHI Maximum Block")

        m1, m2 = st.columns(2)

        with m1:
            max_min = st.number_input(
                "GHI Max Min",
                min_value=1,
                max_value=95,
                value=47,
                step=1,
            )

        with m2:
            max_max = st.number_input(
                "GHI Max Max",
                min_value=1,
                max_value=95,
                value=53,
                step=1,
            )

        st.markdown("#### Tracking Angle Limits")

        a1, a2, a3, a4 = st.columns(4)

        with a1:
            east_min = st.number_input(
                "East Limit Min (°)",
                min_value=0,
                max_value=90,
                value=10,
                step=1,
            )

        with a2:
            east_max = st.number_input(
                "East Limit Max (°)",
                min_value=0,
                max_value=90,
                value=70,
                step=1,
            )

        with a3:
            west_min = st.number_input(
                "West Limit Min (°)",
                min_value=0,
                max_value=90,
                value=10,
                step=1,
            )

        with a4:
            west_max = st.number_input(
                "West Limit Max (°)",
                min_value=0,
                max_value=90,
                value=70,
                step=1,
            )

        st.markdown("#### Differential Evolution")

        d1, d2 = st.columns(2)

        with d1:
            maxiter = st.number_input(
                "Maximum Iterations",
                min_value=5,
                max_value=200,
                value=40,
                step=5,
            )

        with d2:
            popsize = st.number_input(
                "Population Size",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    tracking_parameter_error = None

    if dhi_max <= dhi_min:
        tracking_parameter_error = (
            "DHI Maximum must be greater than DHI Minimum."
        )

    elif start_max <= start_min:
        tracking_parameter_error = (
            "GHI Start Max must be greater than GHI Start Min."
        )

    elif end_max <= end_min:
        tracking_parameter_error = (
            "GHI End Max must be greater than GHI End Min."
        )

    elif max_max <= max_min:
        tracking_parameter_error = (
            "GHI Max Maximum must be greater than GHI Max Minimum."
        )

    elif east_max <= east_min:
        tracking_parameter_error = (
            "East Limit Maximum must be greater than East Limit Minimum."
        )

    elif west_max <= west_min:
        tracking_parameter_error = (
            "West Limit Maximum must be greater than West Limit Minimum."
        )

    if tracking_parameter_error:
        st.warning(tracking_parameter_error)

    run_tracking = st.button(
        "▶ Run Tracking Optimization",
        type="primary",
        use_container_width=True,
        disabled=tracking_parameter_error is not None,
    )

    if run_tracking:

        # Important: at least one valid relationship must be possible.
        if not (
            start_min < max_max
            and max_min < end_max
        ):
            st.error(
                "Tracking block ranges do not allow "
                "Start < Max < End. Please adjust the bounds."
            )
            st.stop()

        bounds = [
            (dhi_min, dhi_max),
            (start_min, start_max),
            (end_min, end_max),
            (max_min, max_max),
            (east_min, east_max),
            (west_min, west_max),
        ]

        with st.spinner(
            "Running Tracking Differential Evolution optimization..."
        ):

            try:

                tracking_result = run_tracking_model(
                    solar=solar,
                    tracking_weights=tracking_weights,
                    bounds=bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                    optimize_by=optimize_by_tracking,
                )

                st.session_state.tracking_result = (
                    tracking_result
                )

            except Exception as exc:

                st.error(
                    f"Tracking optimization failed: {exc}"
                )
                st.stop()

    tracking_result = (
        st.session_state.tracking_result
    )

    if tracking_result is None:

        st.warning(
            "Set the parameters and click "
            "**Run Tracking Optimization**."
        )

    else:

        # ----------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Optimized Tracking Parameters'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            metric_card(
                "DHI",
                f"{tracking_result['DHI']}%",
                "Optimized",
            )

        with c2:
            metric_card(
                "GHI Start",
                str(tracking_result["start_block"]),
                "Block",
            )

        with c3:
            metric_card(
                "GHI End",
                str(tracking_result["end_block"]),
                "Block",
            )

        with c4:
            metric_card(
                "GHI Max",
                str(tracking_result["max_block"]),
                "Block",
            )

        with c5:
            metric_card(
                "East Limit",
                f"{tracking_result['east_limit']}°",
                "Tracking limit",
            )

        with c6:
            metric_card(
                "West Limit",
                f"{tracking_result['west_limit']}°",
                "Tracking limit",
            )

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Tracking Model Performance'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Peak Power",
                f"{tracking_result['forecast'].max():.4f}",
                "MW",
            )

        with c2:
            metric_card(
                "Peak Error",
                f"{tracking_result['peak_error_pct']:.3f}%",
                "Relative error",
            )

        with c3:
            metric_card(
                "Energy Error",
                f"{tracking_result['energy_error'] * 100:.3f}%",
                "Relative error",
            )

        with c4:
            metric_card(
                "Overall Score",
                f"{tracking_result['score']:.5f}",
                "Lower is better",
            )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Actual vs Tracking Forecast'
            '</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.arange(solar["n"]),
                y=solar["actual"],
                name="Actual",
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(solar["n"]),
                y=tracking_result["forecast"],
                name="Tracking Forecast",
                mode="lines",
            )
        )

        fig.update_layout(
            height=450,
            xaxis_title="Block",
            yaxis_title="Power (MW)",
            hovermode="x unified",
            template="plotly_white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # ANGLES
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Tracking Angles'
            '</div>',
            unsafe_allow_html=True,
        )

        angle_df = pd.DataFrame(
            {
                "Block": np.arange(solar["n"]),
                "Zenith Angle": tracking_result["zenith"],
                "Panel Angle": tracking_result["panel"],
            }
        )

        st.dataframe(
            angle_df,
            use_container_width=True,
            height=300,
        )

        # ----------------------------------------------------
        # CLUSTER POWER
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Tracking Cluster Power'
            '</div>',
            unsafe_allow_html=True,
        )

        cluster_df = pd.DataFrame(
            tracking_result["power_matrix"],
            columns=[
                f"{c} Tracking Power"
                for c in CLUSTERS
            ],
        )

        cluster_df.insert(
            0,
            "Block",
            np.arange(solar["n"]),
        )

        st.dataframe(
            cluster_df,
            use_container_width=True,
            height=350,
        )

        # ----------------------------------------------------
        # FINAL DATASET
        # ----------------------------------------------------

        with st.expander(
            "📄 View Final Tracking Dataset",
            expanded=False,
        ):

            st.dataframe(
                tracking_result["df_tracking"],
                use_container_width=True,
                height=400,
            )

        # ----------------------------------------------------
        # EXPORT
        # ----------------------------------------------------

        optimized_parameters = pd.DataFrame(
            {
                "Parameter": [
                    "Optimization Criterion",
                    "DHI",
                    "GHI Starting Block",
                    "GHI Ending Block",
                    "GHI Max Block",
                    "East Tracking Limit",
                    "West Tracking Limit",
                    "Actual Peak",
                    "Tracking Predicted Peak",
                    "Tracking Peak Error (%)",
                    "Tracking Block Error",
                    "Tracking Energy Error",
                    "Tracking Overall Score",
                    "Optimizer Score",
                ],
                "Value": [
                    tracking_result["optimize_by"],
                    tracking_result["DHI"],
                    tracking_result["start_block"],
                    tracking_result["end_block"],
                    tracking_result["max_block"],
                    tracking_result["east_limit"],
                    tracking_result["west_limit"],
                    solar["actual_peak"],
                    tracking_result["forecast"].max(),
                    tracking_result["peak_error_pct"],
                    tracking_result["block_error"],
                    tracking_result["energy_error"],
                    tracking_result["score"],
                    tracking_result["optimizer_score"],
                ],
            }
        )

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Optimization Criterion",
                    "DHI (%)",
                    "GHI Starting Block",
                    "GHI Ending Block",
                    "GHI Max Block",
                    "East Tracking Limit",
                    "West Tracking Limit",
                    "Block Error",
                    "Peak Error (%)",
                    "Energy Error",
                    "Overall Score",
                    "Peak Power",
                ],
                "Tracking": [
                    tracking_result["optimize_by"],
                    tracking_result["DHI"],
                    tracking_result["start_block"],
                    tracking_result["end_block"],
                    tracking_result["max_block"],
                    tracking_result["east_limit"],
                    tracking_result["west_limit"],
                    tracking_result["block_error"],
                    tracking_result["peak_error_pct"],
                    tracking_result["energy_error"],
                    tracking_result["score"],
                    tracking_result["forecast"].max(),
                ],
            }
        )

        excel_output = create_excel_download(
            mode="Tracking",
            df_area=df_area,
            tracking_result=tracking_result,
            summary=summary,
            optimized_parameters=optimized_parameters,
        )

        st.download_button(
            "⬇ Download Tracking Results",
            data=excel_output,
            file_name="Solar_Tracking_Results.xlsx",
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Solar Forecast Correction | Fixed / Tracking Optimization"
)
