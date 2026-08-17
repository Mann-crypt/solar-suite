# ============================================================
# STREAMLIT APP
# SOLAR FORECAST OPTIMIZATION
# FIXED / TRACKING PLANT
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
    page_title="Solar Forecast Optimization",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f8fa;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .app-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .section-title {
        font-size: 20px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .small-note {
        color: #6b7280;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 12px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

CLUSTERS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15"
]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15"
]

N_CLUSTERS = len(CLUSTERS)

REQUIRED_SHEETS = [
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
    "Backend Cal C15"
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "data_loaded": False,
    "auto_results": None,
    "current_results": None,
    "workbook_name": None
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    return df


def safe_float(value, default=0.0):

    try:
        return float(value)
    except Exception:
        return default


def calculate_metrics(
    actual_day,
    prediction_day,
    actual_peak,
    actual_energy
):

    if len(prediction_day) == 0:

        return {
            "Block Error": np.nan,
            "Peak Error": np.nan,
            "Energy Error": np.nan,
            "Overall Score": np.nan,
            "Peak Power": np.nan
        }

    block_error = (
        np.mean(
            np.abs(
                actual_day
                - prediction_day
            )
        )
        / actual_peak
    )

    peak_error = (
        abs(
            actual_peak
            - prediction_day.max()
        )
        / actual_peak
    )

    energy_error = (
        abs(
            actual_energy
            - prediction_day.sum()
        )
        / actual_energy
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return {
        "Block Error": block_error,
        "Peak Error": peak_error,
        "Energy Error": energy_error,
        "Overall Score": score,
        "Peak Power": prediction_day.max()
    }


# ============================================================
# LOAD WORKBOOK
# ============================================================

def load_workbook(uploaded_file):

    excel = pd.ExcelFile(uploaded_file)

    available_sheets = excel.sheet_names

    missing_sheets = [
        sheet
        for sheet in REQUIRED_SHEETS
        if sheet not in available_sheets
    ]

    if missing_sheets:

        raise ValueError(
            "The following required sheets are missing:\n\n"
            + "\n".join(
                f"• {sheet}"
                for sheet in missing_sheets
            )
        )

    # ========================================================
    # 1. AREA & EFFICIENCY
    # ========================================================

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12)
    )

    df = clean_columns(df)

    df = df[
        df["S.No."].notna()
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    # ========================================================
    # 2. EFFECTIVE AREAS
    # ========================================================

    area_df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=None
    )

    fixed_weights = (
        pd.to_numeric(
            area_df.iloc[2:7, 15],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    tracking_weights = (
        pd.to_numeric(
            area_df.iloc[28:33, 15],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    if len(fixed_weights) != N_CLUSTERS:

        raise ValueError(
            "Could not read 5 fixed effective areas."
        )

    if len(tracking_weights) != N_CLUSTERS:

        raise ValueError(
            "Could not read 5 tracking effective areas."
        )

    # ========================================================
    # 3. STANDARD PV EFFICIENCY
    # ========================================================

    standard_efficiency = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
        errors="coerce"
    ).to_numpy(dtype=float)

    if len(standard_efficiency) < N_CLUSTERS:

        raise ValueError(
            "Less than 5 Standard PV Efficiency values found."
        )

    standard_efficiency = (
        standard_efficiency[:N_CLUSTERS]
    )

    # ========================================================
    # 4. FORECAST CONFIG
    # ========================================================

    df_config = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8
    )

    df_config.columns = (
        df_config.columns
        .astype(str)
        .str.strip()
    )

    lat = float(
        df_config.loc[0, "Lat"]
    )

    # ========================================================
    # 5. CONFIG TILT ANGLE
    # ========================================================

    df_tilt = pd.read_excel(
        uploaded_file,
        sheet_name="Config Tilt Angle",
        header=7
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month"
        }
    )

    df_tilt = df_tilt.dropna(
        subset=["Fixed"]
    ).copy()

    df_tilt["Month_Num"] = pd.to_numeric(
        df_tilt["Month_Num"],
        errors="coerce"
    )

    df_tilt["Fixed"] = pd.to_numeric(
        df_tilt["Fixed"],
        errors="coerce"
    )

    month_number_to_tilt = (
        df_tilt
        .dropna(
            subset=["Month_Num"]
        )
        .set_index("Month_Num")["Fixed"]
        .to_dict()
    )

    # ========================================================
    # 6. RESULT / GHI
    # ========================================================

    df_ghi = pd.read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=range(6)
    )

    df_ghi.columns = [
        "Block",
        *GHI_COLS
    ]

    df_ghi = df_ghi[
        pd.to_numeric(
            df_ghi["Block"],
            errors="coerce"
        ).notna()
    ].copy()

    for col in GHI_COLS:

        df_ghi[col] = pd.to_numeric(
            df_ghi[col],
            errors="coerce"
        ).fillna(0)

    blocks_result = pd.to_numeric(
        df_ghi["Block"],
        errors="coerce"
    ).to_numpy(dtype=float)

    ghi_matrix = np.column_stack([
        df_ghi[col].to_numpy(dtype=float)
        for col in GHI_COLS
    ])

    # ========================================================
    # 7. FIXED-C11
    # ========================================================

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-C11",
        header=1
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    date_valid = (
        df_fix["Date"].notna()
    )

    if not date_valid.any():

        raise ValueError(
            "No valid Date rows found in Fixed-C11."
        )

    first_blank = np.where(
        ~date_valid.to_numpy()
    )[0]

    if len(first_blank) > 0:

        df_fix = df_fix.iloc[
            :first_blank[0]
        ].copy()

    else:

        df_fix = df_fix.loc[
            date_valid
        ].copy()

    df_fix.reset_index(
        drop=True,
        inplace=True
    )

    # ========================================================
    # 8. ALIGN DATA
    # ========================================================

    actual_full = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce"
    ).fillna(0).to_numpy(dtype=float)

    n = min(
        len(df_fix),
        len(df_ghi)
    )

    if n == 0:

        raise ValueError(
            "No valid forecast rows available."
        )

    df_fix = df_fix.iloc[
        :n
    ].copy()

    df_ghi = df_ghi.iloc[
        :n
    ].copy()

    actual = actual_full[
        :n
    ]

    ghi_matrix = ghi_matrix[
        :n
    ]

    blocks = blocks_result[
        :n
    ]

    # ========================================================
    # 9. DATES
    # ========================================================

    dates = pd.to_datetime(
        df_fix["Date"],
        errors="coerce"
    )

    if dates.isna().any():

        raise ValueError(
            "Invalid dates found in Fixed-C11."
        )

    # ========================================================
    # 10. WORKBOOK REFERENCE DATE
    # ========================================================

    first_date = pd.Timestamp(
        year=2025,
        month=1,
        day=1
    )

    day_offset = (
        dates - first_date
    ).dt.days.to_numpy(
        dtype=float
    )

    # ========================================================
    # 11. DECLINATION
    # ========================================================

    declination = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_offset
                    + 1
                )
                / 365
            )
        )
    )

    # ========================================================
    # 12. ELEVATION
    # ========================================================

    elevation = (
        90
        - lat
        + declination
    )

    # ========================================================
    # 13. TILT
    # ========================================================

    months = (
        dates.dt.month.to_numpy()
    )

    tilt = np.array([
        month_number_to_tilt.get(
            float(month),
            0
        )
        for month in months
    ])

    # ========================================================
    # 14. SIN(a) / SIN(a+b)
    # ========================================================

    a_plus_b = (
        elevation
        + tilt
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(a_plus_b)
    )

    sin_a_safe = np.where(
        np.abs(sin_a) < 1e-8,
        1e-8,
        sin_a
    )

    # ========================================================
    # 15. FIXED POA
    # ========================================================

    fixed_poa = (
        ghi_matrix
        * sin_ab[:, None]
        / sin_a_safe[:, None]
    )

    # ========================================================
    # 16. VALID MASK
    # ========================================================

    valid_mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    if not valid_mask.any():

        raise ValueError(
            "Actual power contains no valid non-zero values."
        )

    actual_day = actual[
        valid_mask
    ]

    actual_peak = np.max(
        actual_day
    )

    actual_energy = np.sum(
        actual_day
    )

    if actual_peak <= 0:

        raise ValueError(
            "Actual peak must be greater than zero."
        )

    # ========================================================
    # 17. TRACKING DATA
    # ========================================================

    backend_list = []

    for cluster in CLUSTERS:

        backend_list.append(
            pd.read_excel(
                uploaded_file,
                sheet_name=f"Backend Cal {cluster}"
            )
        )

    # ========================================================
    # 18. TRACKING SHEET
    # ========================================================

    df_trac = pd.read_excel(
        uploaded_file,
        sheet_name="Tracking",
        header=1
    )

    df_trac = df_trac.iloc[
        :n
    ].copy()

    df_trac.reset_index(
        drop=True,
        inplace=True
    )

    return {
        "df": df,
        "df_fix": df_fix,
        "df_trac": df_trac,
        "area_df": area_df,
        "df_ghi": df_ghi,
        "backend_list": backend_list,
        "fixed_weights": fixed_weights,
        "tracking_weights": tracking_weights,
        "standard_efficiency": standard_efficiency,
        "lat": lat,
        "month_number_to_tilt": month_number_to_tilt,
        "dates": dates,
        "actual": actual,
        "actual_day": actual_day,
        "actual_peak": actual_peak,
        "actual_energy": actual_energy,
        "ghi_matrix": ghi_matrix,
        "blocks": blocks,
        "fixed_poa": fixed_poa,
        "valid_mask": valid_mask,
        "n": n
    }


# ============================================================
# FIXED LOSS OPTIMIZATION
# ============================================================

def optimize_fixed(data):

    fixed_weights = data["fixed_weights"]
    standard_efficiency = data["standard_efficiency"]
    fixed_poa = data["fixed_poa"]
    valid_mask = data["valid_mask"]
    actual_day = data["actual_day"]
    actual_peak = data["actual_peak"]
    actual_energy = data["actual_energy"]

    max_loss = np.min(
        standard_efficiency
    )

    results = []

    loss_values = np.arange(
        0,
        max_loss + 0.0001,
        0.1
    )

    for loss in loss_values:

        net_efficiency = (
            standard_efficiency
            - loss
        )

        net_efficiency = np.maximum(
            net_efficiency,
            0
        )

        efficiency_factor = np.divide(
            net_efficiency,
            standard_efficiency,
            out=np.zeros_like(
                net_efficiency
            ),
            where=(
                standard_efficiency != 0
            )
        )

        adjusted_fixed_weights = (
            fixed_weights
            * efficiency_factor
        )

        power_matrix = (
            fixed_poa
            * adjusted_fixed_weights[None, :]
            / 1_000_000
        )

        predicted = (
            power_matrix.sum(axis=1)
        )

        predicted_day = predicted[
            valid_mask
        ]

        if len(predicted_day) == 0:
            continue

        predicted_peak = (
            np.max(predicted_day)
        )

        peak_error = abs(
            actual_peak
            - predicted_peak
        )

        peak_error_percent = (
            peak_error
            / actual_peak
            * 100
        )

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    - predicted_day
                )
            )
            / actual_peak
        )

        predicted_energy = (
            np.sum(predicted_day)
        )

        energy_error = (
            abs(
                actual_energy
                - predicted_energy
            )
            / actual_energy
        )

        score = (
            0.80 * block_error
            + 0.10 * (
                peak_error
                / actual_peak
            )
            + 0.10 * energy_error
        )

        results.append({
            "Error %": loss,
            "Actual Peak": actual_peak,
            "Predicted Peak": predicted_peak,
            "Peak Error": peak_error,
            "Peak Error (%)": peak_error_percent,
            "Block Error": block_error,
            "Energy Error": energy_error,
            "Overall Score": score
        })

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise ValueError(
            "Fixed efficiency optimization "
            "did not produce any results."
        )

    best_row = results_df.loc[
        results_df[
            "Peak Error"
        ].idxmin()
    ]

    best_loss = float(
        best_row["Error %"]
    )

    return (
        results_df,
        best_loss
    )


# ============================================================
# FIXED MODEL
# ============================================================

def calculate_fixed(
    data,
    loss
):

    fixed_weights = data["fixed_weights"]
    standard_efficiency = data["standard_efficiency"]
    fixed_poa = data["fixed_poa"]
    valid_mask = data["valid_mask"]
    actual_day = data["actual_day"]
    actual_peak = data["actual_peak"]
    actual_energy = data["actual_energy"]

    net_efficiency = (
        standard_efficiency
        - loss
    )

    net_efficiency = np.maximum(
        net_efficiency,
        0
    )

    fixed_efficiency_factor = np.divide(
        net_efficiency,
        standard_efficiency,
        out=np.zeros_like(
            standard_efficiency
        ),
        where=(
            standard_efficiency != 0
        )
    )

    final_fixed_weights = (
        fixed_weights
        * fixed_efficiency_factor
    )

    final_fixed_power_matrix = (
        fixed_poa
        * final_fixed_weights[None, :]
        / 1_000_000
    )

    fixed_forecast = (
        final_fixed_power_matrix.sum(
            axis=1
        )
    )

    fixed_day = (
        fixed_forecast[
            valid_mask
        ]
    )

    metrics = calculate_metrics(
        actual_day,
        fixed_day,
        actual_peak,
        actual_energy
    )

    df_fix = data["df_fix"].copy()

    for i, cl in enumerate(CLUSTERS):

        df_fix[
            f"{cl}_Fixed Power=I*Ƞ*A"
        ] = final_fixed_power_matrix[
            :, i
        ]

    df_fix[
        "Total Power (CL1+CL2+…)"
    ] = fixed_forecast

    df_area = data["df"].copy()

    df_area["Total area (m2)"] = (
        pd.to_numeric(
            df_area["No of Module"],
            errors="coerce"
        )
        *
        pd.to_numeric(
            df_area["Area of 1 Module (m2)"],
            errors="coerce"
        )
    )

    df_area["Error %"] = loss

    df_area["Net Efficiency (%)"] = (
        pd.to_numeric(
            df_area[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce"
        )
        - loss
    )

    return {
        "df_fix": df_fix,
        "df": df_area,
        "forecast": fixed_forecast,
        "power_matrix": final_fixed_power_matrix,
        "final_weights": final_fixed_weights,
        "metrics": metrics
    }


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    data,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit
):

    blocks = data["blocks"]
    ghi_matrix = data["ghi_matrix"]
    tracking_weights = data["tracking_weights"]

    if not (
        start_block
        < max_block
        < end_block
    ):

        return None

    denominator_1 = (
        start_block
        - 1
        - max_block
    )

    denominator_2 = (
        end_block
        + 1
        - max_block
    )

    if (
        denominator_1 == 0
        or denominator_2 == 0
    ):

        return None

    m1 = (
        90
        / denominator_1
    )

    m2 = (
        90
        / denominator_2
    )

    zenith = np.where(
        blocks <= max_block,
        np.minimum(
            89,
            m1
            * (
                blocks
                - max_block
            )
        ),
        np.minimum(
            89,
            m2
            * (
                blocks
                - max_block
            )
        )
    )

    panel = np.where(
        blocks < max_block,

        np.where(
            zenith < abs(east_limit),
            zenith,
            abs(east_limit)
        ),

        np.where(
            (
                (blocks > max_block)
                &
                (zenith > west_limit)
            ),
            west_limit,
            zenith
        )
    )

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None
    )

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    tracking_power_matrix = (
        dni
        * tracking_weights[None, :]
        / 1_000_000
    )

    tracking_forecast = (
        tracking_power_matrix.sum(
            axis=1
        )
    )

    return (
        tracking_forecast,
        tracking_power_matrix,
        zenith,
        panel,
        dni
    )


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def tracking_objective(
    x,
    data
):

    DHI = int(
        round(x[0])
    )

    start_block = int(
        round(x[1])
    )

    end_block = int(
        round(x[2])
    )

    max_block = int(
        round(x[3])
    )

    east_limit = int(
        round(x[4])
    )

    west_limit = int(
        round(x[5])
    )

    if not (
        start_block
        < max_block
        < end_block
    ):

        return 1e9

    result = calculate_tracking(
        data,
        DHI,
        start_block,
        end_block,
        max_block,
        east_limit,
        west_limit
    )

    if result is None:

        return 1e9

    prediction = result[0]

    if not np.all(
        np.isfinite(prediction)
    ):

        return 1e9

    valid_mask = data["valid_mask"]

    prediction_day = (
        prediction[
            valid_mask
        ]
    )

    if len(prediction_day) == 0:

        return 1e9

    actual_day = data["actual_day"]
    actual_peak = data["actual_peak"]
    actual_energy = data["actual_energy"]

    block_error = (
        np.mean(
            np.abs(
                actual_day
                - prediction_day
            )
        )
        / actual_peak
    )

    peak_error = (
        abs(
            actual_peak
            - prediction_day.max()
        )
        / actual_peak
    )

    energy_error = (
        abs(
            actual_energy
            - prediction_day.sum()
        )
        / actual_energy
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return score


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(data):

    bounds = [

        (0, 10),       # DHI %

        (10, 30),      # GHI Starting Block

        (65, 80),      # GHI Ending Block

        (47, 53),      # GHI Max Block

        (10, 70),      # East Tracking Limit

        (10, 70)       # West Tracking Limit
    ]

    result = differential_evolution(
        lambda x: tracking_objective(
            x,
            data
        ),
        bounds=bounds,
        strategy="best1bin",
        maxiter=40,
        popsize=15,
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1
    )

    best = np.rint(
        result.x
    ).astype(int)

    DHI = best[0]
    start_block = best[1]
    end_block = best[2]
    max_block = best[3]
    east_limit = best[4]
    west_limit = best[5]

    rounded_score = tracking_objective(
        best,
        data
    )

    return {
        "continuous": result.x,
        "rounded": best,
        "optimizer_score": result.fun,
        "rounded_score": rounded_score,
        "DHI": DHI,
        "start_block": start_block,
        "end_block": end_block,
        "max_block": max_block,
        "east_limit": east_limit,
        "west_limit": west_limit
    }


# ============================================================
# TRACKING MODEL
# ============================================================

def calculate_tracking_model(
    data,
    params
):

    result = calculate_tracking(
        data,
        params["DHI"],
        params["start_block"],
        params["end_block"],
        params["max_block"],
        params["east_limit"],
        params["west_limit"]
    )

    if result is None:

        raise ValueError(
            "Invalid Tracking parameters. "
            "Ensure Start < Max < End."
        )

    (
        tracking_forecast,
        tracking_power_matrix,
        zenith,
        panel,
        dni
    ) = result

    valid_mask = data["valid_mask"]

    actual_day = data["actual_day"]
    actual_peak = data["actual_peak"]
    actual_energy = data["actual_energy"]

    tracking_day = (
        tracking_forecast[
            valid_mask
        ]
    )

    metrics = calculate_metrics(
        actual_day,
        tracking_day,
        actual_peak,
        actual_energy
    )

    df_trac = data["df_trac"].copy()

    df_trac[
        "Zenith Angle"
    ] = zenith

    df_trac[
        "Panel Angle"
    ] = panel

    for i, cl in enumerate(CLUSTERS):

        df_trac[
            f"{cl}_Tracking Power=I*Ƞ*A"
        ] = tracking_power_matrix[
            :, i
        ]

    df_trac[
        "Tracking Power=I*Ƞ*A"
    ] = tracking_forecast

    return {
        "df_trac": df_trac,
        "forecast": tracking_forecast,
        "power_matrix": tracking_power_matrix,
        "zenith": zenith,
        "panel": panel,
        "dni": dni,
        "metrics": metrics
    }


# ============================================================
# RUN AUTOMATIC CALCULATIONS
# ============================================================

def run_auto_calculation(data):

    # --------------------------------------------------------
    # Fixed
    # --------------------------------------------------------

    results_df, best_loss = optimize_fixed(
        data
    )

    fixed_result = calculate_fixed(
        data,
        best_loss
    )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    tracking_opt = optimize_tracking(
        data
    )

    tracking_params = {
        "DHI": tracking_opt["DHI"],
        "start_block": tracking_opt["start_block"],
        "end_block": tracking_opt["end_block"],
        "max_block": tracking_opt["max_block"],
        "east_limit": tracking_opt["east_limit"],
        "west_limit": tracking_opt["west_limit"]
    }

    tracking_result = calculate_tracking_model(
        data,
        tracking_params
    )

    return {
        "fixed": {
            "best_loss": best_loss,
            "results_df": results_df,
            "result": fixed_result
        },
        "tracking": {
            "optimization": tracking_opt,
            "params": tracking_params,
            "result": tracking_result
        }
    }


# ============================================================
# EXCEL EXPORT
# ============================================================

def create_excel(
    data,
    results
):

    output = io.BytesIO()

    fixed = results["fixed"]
    tracking = results["tracking"]

    fixed_result = fixed["result"]
    tracking_result = tracking["result"]

    best_loss = fixed["best_loss"]

    actual_peak = data["actual_peak"]

    fixed_metrics = fixed_result["metrics"]
    tracking_metrics = tracking_result["metrics"]

    summary = pd.DataFrame({

        "Metric": [

            "Efficiency Loss (%)",

            "DHI (%)",

            "GHI Starting Block",

            "GHI Ending Block",

            "GHI Max Block",

            "East Tracking Limit",

            "West Tracking Limit",

            "Block Error",

            "Peak Error",

            "Energy Error",

            "Overall Score",

            "Peak Power"

        ],

        "Fixed": [

            best_loss,

            np.nan,

            np.nan,

            np.nan,

            np.nan,

            np.nan,

            np.nan,

            fixed_metrics["Block Error"],

            fixed_metrics["Peak Error"],

            fixed_metrics["Energy Error"],

            fixed_metrics["Overall Score"],

            fixed_metrics["Peak Power"]

        ],

        "Tracking": [

            best_loss,

            tracking["params"]["DHI"],

            tracking["params"]["start_block"],

            tracking["params"]["end_block"],

            tracking["params"]["max_block"],

            tracking["params"]["east_limit"],

            tracking["params"]["west_limit"],

            tracking_metrics["Block Error"],

            tracking_metrics["Peak Error"],

            tracking_metrics["Energy Error"],

            tracking_metrics["Overall Score"],

            tracking_metrics["Peak Power"]

        ]
    })

    optimized_parameters = pd.DataFrame({

        "Parameter": [

            "Fixed Efficiency Loss (%)",

            "Fixed Actual Peak",

            "Fixed Predicted Peak",

            "Fixed Peak Error (%)",

            "Fixed Block Error",

            "Fixed Energy Error",

            "Fixed Overall Score",

            "Tracking DHI (%)",

            "Tracking GHI Starting Block",

            "Tracking GHI Ending Block",

            "Tracking GHI Max Block",

            "Tracking East Limit",

            "Tracking West Limit",

            "Tracking Actual Peak",

            "Tracking Predicted Peak",

            "Tracking Peak Error",

            "Tracking Block Error",

            "Tracking Energy Error",

            "Tracking Overall Score"

        ],

        "Value": [

            best_loss,

            actual_peak,

            fixed_metrics["Peak Power"],

            fixed_metrics["Peak Error"] * 100,

            fixed_metrics["Block Error"],

            fixed_metrics["Energy Error"],

            fixed_metrics["Overall Score"],

            tracking["params"]["DHI"],

            tracking["params"]["start_block"],

            tracking["params"]["end_block"],

            tracking["params"]["max_block"],

            tracking["params"]["east_limit"],

            tracking["params"]["west_limit"],

            actual_peak,

            tracking_metrics["Peak Power"],

            tracking_metrics["Peak Error"],

            tracking_metrics["Block Error"],

            tracking_metrics["Energy Error"],

            tracking_metrics["Overall Score"]

        ]
    })

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # -----------------------------------------------
        # Original calculated area & efficiency
        # -----------------------------------------------

        fixed_result["df"].to_excel(
            writer,
            sheet_name="Area & Efficiency",
            index=False
        )

        # -----------------------------------------------
        # Fixed output
        # -----------------------------------------------

        fixed_result["df_fix"].to_excel(
            writer,
            sheet_name="Fixed Results",
            index=False
        )

        # -----------------------------------------------
        # Tracking output
        # -----------------------------------------------

        tracking_result["df_trac"].to_excel(
            writer,
            sheet_name="Tracking Results",
            index=False
        )

        # -----------------------------------------------
        # Summary
        # -----------------------------------------------

        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        # -----------------------------------------------
        # Optimization results
        # -----------------------------------------------

        fixed["results_df"].to_excel(
            writer,
            sheet_name="Efficiency Loss Tests",
            index=False
        )

        # -----------------------------------------------
        # Optimized parameters
        # -----------------------------------------------

        optimized_parameters.to_excel(
            writer,
            sheet_name="Optimized Parameters",
            index=False
        )

        # -----------------------------------------------
        # GHI data
        # -----------------------------------------------

        data["df_ghi"].to_excel(
            writer,
            sheet_name="GHI Forecast",
            index=False
        )

    output.seek(0)

    return output.getvalue()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">☀️ Solar Forecast Optimization</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Automatic Fixed and Tracking plant optimization'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Workbook")

    uploaded_file = st.file_uploader(
        "Upload Excel Workbook",
        type=[
            "xlsx",
            "xls"
        ]
    )

    st.caption(
        "The workbook should contain the required "
        "solar calculation sheets."
    )

    if uploaded_file is not None:

        if (
            st.session_state["workbook_name"]
            != uploaded_file.name
        ):

            st.session_state["data_loaded"] = False
            st.session_state["auto_results"] = None
            st.session_state["current_results"] = None

            st.session_state["workbook_name"] = (
                uploaded_file.name
            )


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload the Excel workbook from the sidebar "
        "to start the calculation."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

if not st.session_state["data_loaded"]:

    with st.spinner(
        "Reading workbook and calculating inputs..."
    ):

        try:

            data = load_workbook(
                uploaded_file
            )

            st.session_state["data"] = data

        except Exception as e:

            st.error(
                f"Unable to read workbook:\n\n{e}"
            )

            st.stop()

    # ========================================================
    # AUTOMATIC OPTIMIZATION
    # ========================================================

    with st.spinner(
        "Running automatic optimization for Fixed and Tracking plants..."
    ):

        try:

            auto_results = run_auto_calculation(
                data
            )

            st.session_state[
                "auto_results"
            ] = auto_results

            st.session_state[
                "current_results"
            ] = auto_results.copy()

            st.session_state[
                "data_loaded"
            ] = True

        except Exception as e:

            st.error(
                f"Calculation failed:\n\n{e}"
            )

            st.stop()


# ============================================================
# GET DATA
# ============================================================

data = st.session_state["data"]

auto_results = (
    st.session_state["auto_results"]
)


# ============================================================
# WORKBOOK INFORMATION
# ============================================================

with st.expander(
    "Workbook Information",
    expanded=False
):

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Forecast Blocks",
        f"{data['n']:,}"
    )

    c2.metric(
        "Actual Peak",
        f"{data['actual_peak']:.4f}"
    )

    c3.metric(
        "Actual Energy",
        f"{data['actual_energy']:.4f}"
    )

    c4.metric(
        "Latitude",
        f"{data['lat']:.4f}°"
    )


# ============================================================
# PLANT SEGMENTED CONTROL
# ============================================================

plant = st.segmented_control(
    "Select Plant",
    options=[
        "Fixed Plant",
        "Tracking Plant"
    ],
    default="Fixed Plant",
    key="plant_selector"
)


if plant is None:

    plant = "Fixed Plant"


# ============================================================
# FIXED PLANT
# ============================================================

if plant == "Fixed Plant":

    st.markdown(
        '<div class="section-title">'
        'Fixed Plant'
        '</div>',
        unsafe_allow_html=True
    )

    fixed_auto = auto_results[
        "fixed"
    ]

    auto_loss = fixed_auto[
        "best_loss"
    ]

    # ========================================================
    # AUTOMATIC PARAMETERS
    # ========================================================

    st.subheader(
        "Automatically Calculated Parameters"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Maximum Loss Tested",
        f"{np.min(data['standard_efficiency']):.2f}%"
    )

    c2.metric(
        "Automatic Best Loss",
        f"{auto_loss:.2f}%"
    )

    c3.metric(
        "Actual Peak",
        f"{data['actual_peak']:.6f}"
    )

    c4.metric(
        "Forecast Blocks",
        f"{data['n']:,}"
    )

    # ========================================================
    # EFFECTIVE AREAS
    # ========================================================

    st.subheader(
        "Fixed Effective Areas"
    )

    area_display = pd.DataFrame({

        "Cluster": CLUSTERS,

        "Fixed Effective Area (m²)": (
            data["fixed_weights"]
        ),

        "Standard PV Efficiency (%)": (
            data["standard_efficiency"]
        )
    })

    st.dataframe(
        area_display,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # EDITABLE LOSS
    # ========================================================

    st.subheader(
        "Fixed Optimization Parameter"
    )

    st.caption(
        "The value below is automatically optimized first. "
        "You can edit it and recalculate the Fixed plant."
    )

    edited_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=float(
            np.min(
                data["standard_efficiency"]
            )
        ),
        value=float(auto_loss),
        step=0.1,
        format="%.2f",
        key="fixed_loss_editor"
    )

    if st.button(
        "Recalculate Fixed Plant",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Recalculating Fixed Plant..."
        ):

            fixed_result = calculate_fixed(
                data,
                edited_loss
            )

            st.session_state[
                "current_results"
            ]["fixed"] = {
                "best_loss": edited_loss,
                "results_df": fixed_auto[
                    "results_df"
                ],
                "result": fixed_result
            }

        st.success(
            f"Fixed Plant recalculated using "
            f"{edited_loss:.2f}% efficiency loss."
        )

    current_fixed = (
        st.session_state[
            "current_results"
        ]["fixed"]
    )

    fixed_result = current_fixed[
        "result"
    ]

    fixed_metrics = fixed_result[
        "metrics"
    ]

    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader(
        "Fixed Plant Results"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Actual Peak",
        f"{data['actual_peak']:.6f}"
    )

    c2.metric(
        "Fixed Peak",
        f"{fixed_metrics['Peak Power']:.6f}"
    )

    c3.metric(
        "Peak Error",
        f"{fixed_metrics['Peak Error'] * 100:.4f}%"
    )

    c4.metric(
        "Block Error",
        f"{fixed_metrics['Block Error']:.6f}"
    )

    c5.metric(
        "Overall Score",
        f"{fixed_metrics['Overall Score']:.6f}"
    )

    # ========================================================
    # FIXED CLUSTER OUTPUT
    # ========================================================

    st.subheader(
        "Fixed Plant Cluster Output"
    )

    st.dataframe(
        fixed_result["df_fix"],
        use_container_width=True,
        height=400
    )

    # ========================================================
    # EFFICIENCY TEST RESULTS
    # ========================================================

    with st.expander(
        "Efficiency Loss Test Results"
    ):

        st.dataframe(
            current_fixed[
                "results_df"
            ],
            use_container_width=True,
            height=400
        )


# ============================================================
# TRACKING PLANT
# ============================================================

else:

    st.markdown(
        '<div class="section-title">'
        'Tracking Plant'
        '</div>',
        unsafe_allow_html=True
    )

    tracking_auto = auto_results[
        "tracking"
    ]

    auto_params = tracking_auto[
        "params"
    ]

    # ========================================================
    # AUTOMATIC PARAMETERS
    # ========================================================

    st.subheader(
        "Automatically Optimized Parameters"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "DHI",
        f"{auto_params['DHI']}%"
    )

    c2.metric(
        "GHI Start Block",
        f"{auto_params['start_block']}"
    )

    c3.metric(
        "GHI End Block",
        f"{auto_params['end_block']}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "GHI Max Block",
        f"{auto_params['max_block']}"
    )

    c2.metric(
        "East Limit",
        f"{auto_params['east_limit']}°"
    )

    c3.metric(
        "West Limit",
        f"{auto_params['west_limit']}°"
    )

    # ========================================================
    # TRACKING EFFECTIVE AREAS
    # ========================================================

    st.subheader(
        "Tracking Effective Areas"
    )

    tracking_area_display = pd.DataFrame({

        "Cluster": CLUSTERS,

        "Tracking Effective Area (m²)": (
            data["tracking_weights"]
        ),

        "Standard PV Efficiency (%)": (
            data["standard_efficiency"]
        )
    })

    st.dataframe(
        tracking_area_display,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # EDITABLE PARAMETERS
    # ========================================================

    st.subheader(
        "Tracking Optimization Parameters"
    )

    st.caption(
        "These values were automatically optimized first. "
        "You can edit them and recalculate the Tracking plant."
    )

    c1, c2, c3 = st.columns(3)

    edited_dhi = c1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=10,
        value=int(
            auto_params["DHI"]
        ),
        step=1,
        key="tracking_dhi_editor"
    )

    edited_start = c2.number_input(
        "GHI Starting Block",
        min_value=10,
        max_value=30,
        value=int(
            auto_params["start_block"]
        ),
        step=1,
        key="tracking_start_editor"
    )

    edited_end = c3.number_input(
        "GHI Ending Block",
        min_value=65,
        max_value=80,
        value=int(
            auto_params["end_block"]
        ),
        step=1,
        key="tracking_end_editor"
    )

    c1, c2, c3 = st.columns(3)

    edited_max = c1.number_input(
        "GHI Max Block",
        min_value=47,
        max_value=53,
        value=int(
            auto_params["max_block"]
        ),
        step=1,
        key="tracking_max_editor"
    )

    edited_east = c2.number_input(
        "East Tracking Limit",
        min_value=10,
        max_value=70,
        value=int(
            auto_params["east_limit"]
        ),
        step=1,
        key="tracking_east_editor"
    )

    edited_west = c3.number_input(
        "West Tracking Limit",
        min_value=10,
        max_value=70,
        value=int(
            auto_params["west_limit"]
        ),
        step=1,
        key="tracking_west_editor"
    )

    # ========================================================
    # PARAMETER VALIDATION
    # ========================================================

    if not (
        edited_start
        < edited_max
        < edited_end
    ):

        st.error(
            "Invalid Tracking sequence. "
            "GHI Starting Block must be less than "
            "GHI Max Block, and GHI Max Block must be "
            "less than GHI Ending Block."
        )

    # ========================================================
    # RECALCULATE
    # ========================================================

    if st.button(
        "Recalculate Tracking Plant",
        type="primary",
        use_container_width=True,
        disabled=not (
            edited_start
            < edited_max
            < edited_end
        )
    ):

        with st.spinner(
            "Recalculating Tracking Plant..."
        ):

            tracking_params = {

                "DHI": int(
                    edited_dhi
                ),

                "start_block": int(
                    edited_start
                ),

                "end_block": int(
                    edited_end
                ),

                "max_block": int(
                    edited_max
                ),

                "east_limit": int(
                    edited_east
                ),

                "west_limit": int(
                    edited_west
                )
            }

            try:

                tracking_result = (
                    calculate_tracking_model(
                        data,
                        tracking_params
                    )
                )

                st.session_state[
                    "current_results"
                ]["tracking"] = {

                    "optimization": tracking_auto[
                        "optimization"
                    ],

                    "params": tracking_params,

                    "result": tracking_result
                }

                st.success(
                    "Tracking Plant recalculated successfully."
                )

            except Exception as e:

                st.error(
                    f"Tracking calculation failed: {e}"
                )

    current_tracking = (
        st.session_state[
            "current_results"
        ]["tracking"]
    )

    tracking_result = current_tracking[
        "result"
    ]

    tracking_metrics = tracking_result[
        "metrics"
    ]

    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader(
        "Tracking Plant Results"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Actual Peak",
        f"{data['actual_peak']:.6f}"
    )

    c2.metric(
        "Tracking Peak",
        f"{tracking_metrics['Peak Power']:.6f}"
    )

    c3.metric(
        "Peak Error",
        f"{tracking_metrics['Peak Error'] * 100:.4f}%"
    )

    c4.metric(
        "Block Error",
        f"{tracking_metrics['Block Error']:.6f}"
    )

    c5.metric(
        "Overall Score",
        f"{tracking_metrics['Overall Score']:.6f}"
    )

    # ========================================================
    # TRACKING OUTPUT
    # ========================================================

    st.subheader(
        "Tracking Plant Output"
    )

    st.dataframe(
        tracking_result["df_trac"],
        use_container_width=True,
        height=400
    )


# ============================================================
# COMMON FORECAST CHART
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Actual vs Forecast'
    '</div>',
    unsafe_allow_html=True
)

current_fixed = (
    st.session_state[
        "current_results"
    ]["fixed"]
)

current_tracking = (
    st.session_state[
        "current_results"
    ]["tracking"]
)

fixed_forecast = current_fixed[
    "result"
]["forecast"]

tracking_forecast = current_tracking[
    "result"
]["forecast"]

actual = data["actual"]

p = np.arange(
    data["n"]
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=p,
        y=actual,
        mode="lines",
        name="Actual",
        line=dict(
            width=2
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=p,
        y=fixed_forecast,
        mode="lines",
        name="Fixed Forecast",
        line=dict(
            width=2
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=p,
        y=tracking_forecast,
        mode="lines",
        name="Tracking Forecast",
        line=dict(
            width=2
        )
    )
)

fig.update_layout(
    height=500,
    xaxis_title="Block",
    yaxis_title="Power (MW)",
    title="Actual vs Fixed vs Tracking Forecast",
    hovermode="x unified",
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# FIXED VS TRACKING SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Fixed vs Tracking Summary'
    '</div>',
    unsafe_allow_html=True
)

fixed_metrics = current_fixed[
    "result"
]["metrics"]

tracking_metrics = current_tracking[
    "result"
]["metrics"]

current_loss = current_fixed[
    "best_loss"
]

current_tracking_params = (
    current_tracking[
        "params"
    ]
)

summary = pd.DataFrame({

    "Metric": [

        "Efficiency Loss (%)",

        "DHI (%)",

        "GHI Starting Block",

        "GHI Ending Block",

        "GHI Max Block",

        "East Tracking Limit",

        "West Tracking Limit",

        "Block Error",

        "Peak Error",

        "Energy Error",

        "Overall Score",

        "Peak Power"

    ],

    "Fixed": [

        current_loss,

        np.nan,

        np.nan,

        np.nan,

        np.nan,

        np.nan,

        np.nan,

        fixed_metrics["Block Error"],

        fixed_metrics["Peak Error"],

        fixed_metrics["Energy Error"],

        fixed_metrics["Overall Score"],

        fixed_metrics["Peak Power"]

    ],

    "Tracking": [

        current_loss,

        current_tracking_params["DHI"],

        current_tracking_params["start_block"],

        current_tracking_params["end_block"],

        current_tracking_params["max_block"],

        current_tracking_params["east_limit"],

        current_tracking_params["west_limit"],

        tracking_metrics["Block Error"],

        tracking_metrics["Peak Error"],

        tracking_metrics["Energy Error"],

        tracking_metrics["Overall Score"],

        tracking_metrics["Peak Power"]

    ]
})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FINAL OPTIMIZED PARAMETERS
# ============================================================

optimized_parameters = pd.DataFrame({

    "Parameter": [

        "Fixed Efficiency Loss (%)",

        "Fixed Actual Peak",

        "Fixed Predicted Peak",

        "Fixed Peak Error (%)",

        "Fixed Block Error",

        "Fixed Energy Error",

        "Fixed Overall Score",

        "Tracking DHI (%)",

        "Tracking GHI Starting Block",

        "Tracking GHI Ending Block",

        "Tracking GHI Max Block",

        "Tracking East Limit",

        "Tracking West Limit",

        "Tracking Actual Peak",

        "Tracking Predicted Peak",

        "Tracking Peak Error",

        "Tracking Block Error",

        "Tracking Energy Error",

        "Tracking Overall Score"

    ],

    "Value": [

        current_loss,

        data["actual_peak"],

        fixed_metrics["Peak Power"],

        fixed_metrics["Peak Error"] * 100,

        fixed_metrics["Block Error"],

        fixed_metrics["Energy Error"],

        fixed_metrics["Overall Score"],

        current_tracking_params["DHI"],

        current_tracking_params["start_block"],

        current_tracking_params["end_block"],

        current_tracking_params["max_block"],

        current_tracking_params["east_limit"],

        current_tracking_params["west_limit"],

        data["actual_peak"],

        tracking_metrics["Peak Power"],

        tracking_metrics["Peak Error"],

        tracking_metrics["Block Error"],

        tracking_metrics["Energy Error"],

        tracking_metrics["Overall Score"]

    ]
})


with st.expander(
    "Final Optimized Parameters",
    expanded=False
):

    st.dataframe(
        optimized_parameters,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Download Results'
    '</div>',
    unsafe_allow_html=True
)

try:

    excel_bytes = create_excel(
        data,
        st.session_state[
            "current_results"
        ]
    )

    st.download_button(
        label="📥 Download Complete Excel Report",
        data=excel_bytes,
        file_name=(
            "Solar_Forecast_Optimization_Report.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )

except Exception as e:

    st.error(
        f"Unable to generate Excel report: {e}"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>
    <div style="
        text-align:center;
        color:#9ca3af;
        font-size:12px;
    ">
        Solar Forecast Optimization
    </div>
    """,
    unsafe_allow_html=True
)
