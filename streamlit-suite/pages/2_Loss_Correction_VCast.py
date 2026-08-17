# ============================================================
# STREAMLIT APP
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
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
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7f8fa;
    }

    /* Header */
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

    /* Cards */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px;
        min-height: 120px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .metric-title {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
        color: #111827;
    }

    .metric-sub {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 5px;
    }

    /* Section */
    .section-title {
        font-size: 20px;
        font-weight: 650;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* Info */
    .info-box {
        background: #ffffff;
        border-left: 4px solid #4f46e5;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    /* Buttons */
    .stDownloadButton button {
        width: 100%;
        border-radius: 8px;
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


# ============================================================
# HELPER: KPI CARD
# ============================================================

def metric_card(
    title,
    value,
    subtitle=""
):

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-title">
                {title}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-sub">
                {subtitle}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# READ INPUT FILE
# ============================================================

@st.cache_data(show_spinner=False)
def read_input_file(file_bytes):

    excel = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

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
        "Backend Cal C15"
    ]

    missing = [
        s for s in required_sheets
        if s not in excel.sheet_names
    ]

    if missing:

        raise ValueError(
            "Missing required sheets: "
            + ", ".join(missing)
        )

    return excel


# ============================================================
# LOAD AREA & EFFICIENCY
# ============================================================

@st.cache_data(show_spinner=False)
def load_area_efficiency(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12)
    )

    df.columns = (
        df.columns
        .str.replace(
            "*",
            "",
            regex=False
        )
        .str.strip()
    )

    df = df[
        df["S.No."].notna()
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    area_df = pd.read_excel(
        io.BytesIO(file_bytes),
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

    standard_efficiency = (
        pd.to_numeric(
            df[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce"
        )
        .to_numpy(dtype=float)
    )

    if len(standard_efficiency) < N_CLUSTERS:

        raise ValueError(
            "Less than 5 Standard PV Efficiency "
            "values found."
        )

    standard_efficiency = (
        standard_efficiency[:N_CLUSTERS]
    )

    return (
        df,
        fixed_weights,
        tracking_weights,
        standard_efficiency
    )


# ============================================================
# LOAD FORECAST CONFIG
# ============================================================

@st.cache_data(show_spinner=False)
def load_latitude(file_bytes):

    df_config = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8
    )

    lat = float(
        df_config.loc[0, "Lat"]
    )

    return lat


# ============================================================
# LOAD TILT
# ============================================================

@st.cache_data(show_spinner=False)
def load_tilt(file_bytes):

    df_tilt = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle",
        header=7
    )

    df_tilt.columns = (
        df_tilt.columns
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

    return (
        df_tilt
        .dropna(
            subset=["Month_Num"]
        )
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

    blocks = pd.to_numeric(
        df_ghi["Block"],
        errors="coerce"
    ).to_numpy(dtype=float)

    ghi_matrix = np.column_stack([
        df_ghi[col].to_numpy(dtype=float)
        for col in GHI_COLS
    ])

    return (
        df_ghi,
        blocks,
        ghi_matrix
    )


# ============================================================
# LOAD FIXED DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_fixed_data(file_bytes):

    df_fix = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Fixed-C11",
        header=1
    )

    df_fix.columns = (
        df_fix.columns
        .str.strip()
    )

    date_valid = (
        df_fix["Date"].notna()
    )

    if not date_valid.any():

        raise ValueError(
            "No valid Date rows found "
            "in Fixed-C11."
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

    return df_fix


# ============================================================
# PREPARE SOLAR DATA
# ============================================================

def prepare_solar_data(
    df_fix,
    df_ghi,
    blocks_result,
    ghi_matrix,
    lat,
    month_number_to_tilt
):

    actual_full = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce"
    ).fillna(0).to_numpy(
        dtype=float
    )

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

    actual = actual_full[
        :n
    ]

    ghi_matrix = ghi_matrix[
        :n
    ]

    blocks = blocks_result[
        :n
    ]

    dates = pd.to_datetime(
        df_fix["Date"],
        errors="coerce"
    )

    if dates.isna().any():

        raise ValueError(
            "Invalid dates found "
            "in Fixed-C11."
        )

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

    elevation = (
        90
        - lat
        + declination
    )

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

    fixed_poa = (
        ghi_matrix
        * sin_ab[:, None]
        / sin_a_safe[:, None]
    )

    valid_mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    if not valid_mask.any():

        raise ValueError(
            "Actual power contains no "
            "valid non-zero values."
        )

    actual_day = (
        actual[
            valid_mask
        ]
    )

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
        "n": n
    }


# ============================================================
# FIXED OPTIMIZATION
# ============================================================

def run_fixed_model(
    solar,
    fixed_weights,
    standard_efficiency
):

    fixed_poa = solar["fixed_poa"]
    valid_mask = solar["valid_mask"]
    actual_day = solar["actual_day"]
    actual_peak = solar["actual_peak"]
    actual_energy = solar["actual_energy"]

    max_error = np.min(
        standard_efficiency
    )

    loss_values = np.arange(
        0,
        max_error + 0.0001,
        0.1
    )

    results = []

    for error in loss_values:

        net_efficiency = (
            standard_efficiency
            - error
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

        adjusted_weights = (
            fixed_weights
            * efficiency_factor
        )

        power_matrix = (
            fixed_poa
            * adjusted_weights[None, :]
            / 1_000_000
        )

        predicted = (
            power_matrix.sum(
                axis=1
            )
        )

        predicted_day = (
            predicted[
                valid_mask
            ]
        )

        if len(predicted_day) == 0:
            continue

        predicted_peak = (
            predicted_day.max()
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

        energy_error = (
            abs(
                actual_energy
                - predicted_day.sum()
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

            "Error %": error,

            "Actual Peak": actual_peak,

            "Predicted Peak": predicted_peak,

            "Peak Error": peak_error,

            "Peak Error (%)":
                peak_error_percent,

            "Block Error":
                block_error,

            "Energy Error":
                energy_error,

            "Overall Score":
                score
        })

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise ValueError(
            "Fixed optimization produced "
            "no results."
        )

    best_row = results_df.loc[
        results_df["Peak Error"].idxmin()
    ]

    best_error = float(
        best_row["Error %"]
    )

    net_efficiency = (
        standard_efficiency
        - best_error
    )

    net_efficiency = np.maximum(
        net_efficiency,
        0
    )

    efficiency_factor = np.divide(
        net_efficiency,
        standard_efficiency,

        out=np.zeros_like(
            standard_efficiency
        ),

        where=(
            standard_efficiency != 0
        )
    )

    final_weights = (
        fixed_weights
        * efficiency_factor
    )

    power_matrix = (
        fixed_poa
        * final_weights[None, :]
        / 1_000_000
    )

    forecast = (
        power_matrix.sum(
            axis=1
        )
    )

    fixed_day = (
        forecast[
            valid_mask
        ]
    )

    block_error = (
        np.mean(
            np.abs(
                actual_day
                - fixed_day
            )
        )
        / actual_peak
    )

    peak_error = (
        abs(
            actual_peak
            - fixed_day.max()
        )
        / actual_peak
    )

    energy_error = (
        abs(
            actual_energy
            - fixed_day.sum()
        )
        / actual_energy
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
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

        "net_efficiency":
            net_efficiency,

        "final_weights":
            final_weights,

        "power_matrix":
            power_matrix,

        "forecast":
            forecast,

        "block_error":
            block_error,

        "peak_error":
            peak_error,

        "energy_error":
            energy_error,

        "score":
            score,

        "results_df":
            results_df,

        "df_fixed":
            df_fixed
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
    tracking_weights
):

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

            zenith
            < abs(east_limit),

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
# TRACKING OPTIMIZATION
# ============================================================

def run_tracking_model(
    solar,
    tracking_weights,
    maxiter=40,
    popsize=15
):

    actual_day = solar["actual_day"]
    actual_peak = solar["actual_peak"]
    actual_energy = solar["actual_energy"]

    valid_mask = solar["valid_mask"]

    blocks = solar["blocks"]
    ghi_matrix = solar["ghi_matrix"]

    def objective(x):

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

            DHI,
            start_block,
            end_block,
            max_block,
            east_limit,
            west_limit,

            blocks,
            ghi_matrix,
            tracking_weights
        )

        if result is None:
            return 1e9

        prediction = result[0]

        if not np.all(
            np.isfinite(
                prediction
            )
        ):

            return 1e9

        prediction_day = (
            prediction[
                valid_mask
            ]
        )

        if len(prediction_day) == 0:
            return 1e9

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

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    bounds = [

        (0, 10),

        (10, 30),

        (65, 80),

        (47, 53),

        (10, 70),

        (10, 70)
    ]

    result = differential_evolution(

        objective,

        bounds=bounds,

        strategy="best1bin",

        maxiter=maxiter,

        popsize=popsize,

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

    final = calculate_tracking(

        DHI,
        start_block,
        end_block,
        max_block,
        east_limit,
        west_limit,

        blocks,
        ghi_matrix,
        tracking_weights
    )

    (
        forecast,
        power_matrix,
        zenith,
        panel,
        dni
    ) = final

    tracking_day = (
        forecast[
            valid_mask
        ]
    )

    block_error = (
        np.mean(
            np.abs(
                actual_day
                - tracking_day
            )
        )
        / actual_peak
    )

    peak_error = (
        abs(
            actual_peak
            - tracking_day.max()
        )
        / actual_peak
    )

    energy_error = (
        abs(
            actual_energy
            - tracking_day.sum()
        )
        / actual_energy
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    df_tracking = pd.read_excel(
        io.BytesIO(
            solar["_file_bytes"]
        ),
        sheet_name="Tracking",
        header=1
    )

    df_tracking = df_tracking.iloc[
        :solar["n"]
    ].copy()

    df_tracking.reset_index(
        drop=True,
        inplace=True
    )

    df_tracking[
        "Zenith Angle"
    ] = zenith

    df_tracking[
        "Panel Angle"
    ] = panel

    for i, cluster in enumerate(
        CLUSTERS
    ):

        df_tracking[
            f"{cluster}_Tracking Power=I*Ƞ*A"
        ] = power_matrix[:, i]

    df_tracking[
        "Tracking Power=I*Ƞ*A"
    ] = forecast

    return {

        "DHI": DHI,

        "start_block":
            start_block,

        "end_block":
            end_block,

        "max_block":
            max_block,

        "east_limit":
            east_limit,

        "west_limit":
            west_limit,

        "forecast":
            forecast,

        "power_matrix":
            power_matrix,

        "zenith":
            zenith,

        "panel":
            panel,

        "dni":
            dni,

        "block_error":
            block_error,

        "peak_error":
            peak_error,

        "energy_error":
            energy_error,

        "score":
            score,

        "optimizer_score":
            result.fun,

        "df_tracking":
            df_tracking
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
    optimized_parameters=None
):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df_area.to_excel(
            writer,
            sheet_name="Area & Efficiency",
            index=False
        )

        if mode == "Fixed":

            fixed_result["df_fixed"].to_excel(
                writer,
                sheet_name="Fixed Results",
                index=False
            )

            fixed_result["results_df"].to_excel(
                writer,
                sheet_name="Error Optimization",
                index=False
            )

        else:

            tracking_result["df_tracking"].to_excel(
                writer,
                sheet_name="Tracking Results",
                index=False
            )

        if summary is not None:

            summary.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )

        if optimized_parameters is not None:

            optimized_parameters.to_excel(
                writer,
                sheet_name="Optimized Parameters",
                index=False
            )

    output.seek(0)

    return output


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-header">'
    '☀️ Solar Forecast Correction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">'
    'Fixed and Tracking solar forecast optimization'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### ⚙️ Input Configuration")

    uploaded_file = st.file_uploader(
        "Upload Excel Workbook",
        type=[
            "xlsx",
            "xls"
        ]
    )

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
# LOAD FILE
# ============================================================

try:

    with st.spinner(
        "Reading workbook..."
    ):

        read_input_file(
            file_bytes
        )

        (
            df_area,
            fixed_weights,
            tracking_weights,
            standard_efficiency
        ) = load_area_efficiency(
            file_bytes
        )

        lat = load_latitude(
            file_bytes
        )

        month_number_to_tilt = (
            load_tilt(
                file_bytes
            )
        )

        (
            df_ghi,
            blocks_result,
            ghi_matrix
        ) = load_ghi(
            file_bytes
        )

        df_fix = load_fixed_data(
            file_bytes
        )

        solar = prepare_solar_data(

            df_fix,

            df_ghi,

            blocks_result,

            ghi_matrix,

            lat,

            month_number_to_tilt
        )

        # Store bytes for tracking export
        solar["_file_bytes"] = file_bytes

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


# ============================================================
# FILE SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Workbook Overview'
    '</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    metric_card(
        "Clusters",
        str(N_CLUSTERS),
        "C11 to C15"
    )

with c2:

    metric_card(
        "Forecast Blocks",
        f"{solar['n']:,}",
        "15-minute blocks"
    )

with c3:

    metric_card(
        "Latitude",
        f"{lat:.4f}°",
        "Forecast configuration"
    )

with c4:

    metric_card(
        "Actual Peak",
        f"{solar['actual_peak']:.4f}",
        "MW"
    )


# ============================================================
# RUN CALCULATIONS
# ============================================================

if "fixed_result" not in st.session_state:

    st.session_state.fixed_result = None

if "tracking_result" not in st.session_state:

    st.session_state.tracking_result = None


# ============================================================
# MODEL CONTROL
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Model Selection'
    '</div>',
    unsafe_allow_html=True
)


mode = st.segmented_control(
    "Select model",
    options=[
        "Fixed",
        "Tracking"
    ],
    default="Fixed",

    key="model_mode",

    label_visibility="collapsed"
)


# ============================================================
# FIXED MODEL
# ============================================================

if mode == "Fixed":

    st.markdown(
        '<div class="info-box">'
        '<b>Fixed Plant</b><br>'
        'Optimizes Error % in 0.1% increments and '
        'calculates the corrected fixed-tilt forecast.'
        '</div>',
        unsafe_allow_html=True
    )

    run_fixed = st.button(
        "▶ Run Fixed Optimization",
        type="primary",
        use_container_width=True
    )

    if run_fixed:

        with st.spinner(
            "Running Fixed Error % optimization..."
        ):

            fixed_result = run_fixed_model(

                solar,

                fixed_weights,

                standard_efficiency
            )

            st.session_state.fixed_result = (
                fixed_result
            )

    fixed_result = (
        st.session_state.fixed_result
    )

    if fixed_result is None:

        st.warning(
            "Click **Run Fixed Optimization** "
            "to calculate the Fixed model."
        )

    else:

        # ----------------------------------------------------
        # KPIs
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Fixed Model Results'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:

            metric_card(
                "Optimized Error %",
                f"{fixed_result['best_error']:.2f}%",
                "Minimum peak error"
            )

        with c2:

            metric_card(
                "Peak Power",
                f"{fixed_result['forecast'].max():.4f}",
                "MW"
            )

        with c3:

            metric_card(
                "Peak Error",
                f"{fixed_result['peak_error'] * 100:.3f}%",
                "Relative error"
            )

        with c4:

            metric_card(
                "Block Error",
                f"{fixed_result['block_error']:.5f}",
                "Normalized MAE"
            )

        with c5:

            metric_card(
                "Overall Score",
                f"{fixed_result['score']:.5f}",
                "Lower is better"
            )


        # ----------------------------------------------------
        # NET EFFICIENCY
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Efficiency Results'
            '</div>',
            unsafe_allow_html=True
        )

        efficiency_table = pd.DataFrame({

            "Cluster": CLUSTERS,

            "Standard Efficiency (%)":
                standard_efficiency,

            "Error %":
                [fixed_result["best_error"]] * N_CLUSTERS,

            "Net Efficiency (%)":
                fixed_result["net_efficiency"],

            "Original Fixed Area (m²)":
                fixed_weights,

            "Final Effective Area (m²)":
                fixed_result["final_weights"]
        })

        st.dataframe(
            efficiency_table,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # FORECAST CHART
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Actual vs Fixed Forecast'
            '</div>',
            unsafe_allow_html=True
        )

        p = np.arange(
            solar["n"]
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=p,
                y=solar["actual"],
                name="Actual",
                mode="lines"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=p,
                y=fixed_result["forecast"],
                name="Fixed Forecast",
                mode="lines"
            )
        )

        fig.update_layout(
            height=450,
            xaxis_title="Block",
            yaxis_title="Power (MW)",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # CLUSTER POWER
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Fixed Cluster Power'
            '</div>',
            unsafe_allow_html=True
        )

        cluster_df = pd.DataFrame(
            fixed_result["power_matrix"],
            columns=[
                f"{c} Fixed Power"
                for c in CLUSTERS
            ]
        )

        cluster_df.insert(
            0,
            "Block",
            np.arange(
                solar["n"]
            )
        )

        st.dataframe(
            cluster_df,
            use_container_width=True,
            height=350
        )


        # ----------------------------------------------------
        # ERROR OPTIMIZATION
        # ----------------------------------------------------

        with st.expander(
            "View Error % Optimization Results"
        ):

            st.dataframe(
                fixed_result["results_df"],
                use_container_width=True,
                height=400
            )


        # ----------------------------------------------------
        # FINAL DATA
        # ----------------------------------------------------

        with st.expander(
            "View Final Fixed Dataset"
        ):

            st.dataframe(
                fixed_result["df_fixed"],
                use_container_width=True,
                height=400
            )


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        optimized_parameters = pd.DataFrame({

            "Parameter": [

                "Fixed Error %",

                "Actual Peak",

                "Fixed Predicted Peak",

                "Fixed Peak Error (%)",

                "Fixed Block Error",

                "Fixed Energy Error",

                "Fixed Overall Score"
            ],

            "Value": [

                fixed_result["best_error"],

                solar["actual_peak"],

                fixed_result["forecast"].max(),

                fixed_result["peak_error"] * 100,

                fixed_result["block_error"],

                fixed_result["energy_error"],

                fixed_result["score"]
            ]
        })

        summary = pd.DataFrame({

            "Metric": [

                "Error %",

                "Block Error",

                "Peak Error",

                "Energy Error",

                "Overall Score",

                "Peak Power"
            ],

            "Fixed": [

                fixed_result["best_error"],

                fixed_result["block_error"],

                fixed_result["peak_error"],

                fixed_result["energy_error"],

                fixed_result["score"],

                fixed_result["forecast"].max()
            ]
        })

        excel_output = create_excel_download(

            "Fixed",

            df_area,

            fixed_result=fixed_result,

            summary=summary,

            optimized_parameters=
                optimized_parameters
        )

        st.download_button(

            "⬇ Download Fixed Results",

            data=excel_output,

            file_name=(
                "Solar_Fixed_Results.xlsx"
            ),

            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),

            use_container_width=True
        )


# ============================================================
# TRACKING MODEL
# ============================================================

else:

    st.markdown(
        '<div class="info-box">'
        '<b>Tracking Plant</b><br>'
        'Optimizes DHI, GHI block boundaries and '
        'east/west tracking limits using Differential Evolution.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TRACKING SETTINGS
    # --------------------------------------------------------

    with st.expander(
        "⚙️ Optimization Settings",
        expanded=False
    ):

        tc1, tc2 = st.columns(2)

        with tc1:

            maxiter = st.number_input(
                "Maximum Iterations",
                min_value=5,
                max_value=200,
                value=40,
                step=5
            )

        with tc2:

            popsize = st.number_input(
                "Population Size",
                min_value=5,
                max_value=50,
                value=15,
                step=5
            )


    run_tracking = st.button(
        "▶ Run Tracking Optimization",
        type="primary",
        use_container_width=True
    )


    if run_tracking:

        with st.spinner(
            "Running Tracking Differential Evolution optimization..."
        ):

            tracking_result = run_tracking_model(

                solar,

                tracking_weights,

                maxiter=maxiter,

                popsize=popsize
            )

            st.session_state.tracking_result = (
                tracking_result
            )


    tracking_result = (
        st.session_state.tracking_result
    )


    if tracking_result is None:

        st.warning(
            "Click **Run Tracking Optimization** "
            "to calculate the Tracking model."
        )

    else:

        # ----------------------------------------------------
        # TRACKING PARAMETERS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Optimized Tracking Parameters'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:

            metric_card(
                "DHI",
                f"{tracking_result['DHI']}%",
                "Optimized"
            )

        with c2:

            metric_card(
                "GHI Start",
                str(
                    tracking_result[
                        "start_block"
                    ]
                ),
                "Block"
            )

        with c3:

            metric_card(
                "GHI End",
                str(
                    tracking_result[
                        "end_block"
                    ]
                ),
                "Block"
            )

        with c4:

            metric_card(
                "GHI Max",
                str(
                    tracking_result[
                        "max_block"
                    ]
                ),
                "Block"
            )

        with c5:

            metric_card(
                "East Limit",
                f"{tracking_result['east_limit']}°",
                "Tracking limit"
            )

        with c6:

            metric_card(
                "West Limit",
                f"{tracking_result['west_limit']}°",
                "Tracking limit"
            )


        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Tracking Model Performance'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            metric_card(
                "Peak Power",
                f"{tracking_result['forecast'].max():.4f}",
                "MW"
            )

        with c2:

            metric_card(
                "Peak Error",
                f"{tracking_result['peak_error'] * 100:.3f}%",
                "Relative error"
            )

        with c3:

            metric_card(
                "Energy Error",
                f"{tracking_result['energy_error'] * 100:.3f}%",
                "Relative error"
            )

        with c4:

            metric_card(
                "Overall Score",
                f"{tracking_result['score']:.5f}",
                "Lower is better"
            )


        # ----------------------------------------------------
        # FORECAST CHART
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Actual vs Tracking Forecast'
            '</div>',
            unsafe_allow_html=True
        )

        p = np.arange(
            solar["n"]
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=p,
                y=solar["actual"],
                name="Actual",
                mode="lines"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=p,
                y=tracking_result["forecast"],
                name="Tracking Forecast",
                mode="lines"
            )
        )

        fig.update_layout(
            height=450,
            xaxis_title="Block",
            yaxis_title="Power (MW)",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # TRACKING ANGLES
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Tracking Angles'
            '</div>',
            unsafe_allow_html=True
        )

        angle_df = pd.DataFrame({

            "Block":
                np.arange(
                    solar["n"]
                ),

            "Zenith Angle":
                tracking_result["zenith"],

            "Panel Angle":
                tracking_result["panel"]
        })

        st.dataframe(
            angle_df,
            use_container_width=True,
            height=300
        )


        # ----------------------------------------------------
        # CLUSTER POWER
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Tracking Cluster Power'
            '</div>',
            unsafe_allow_html=True
        )

        cluster_df = pd.DataFrame(
            tracking_result[
                "power_matrix"
            ],
            columns=[
                f"{c} Tracking Power"
                for c in CLUSTERS
            ]
        )

        cluster_df.insert(
            0,
            "Block",
            np.arange(
                solar["n"]
            )
        )

        st.dataframe(
            cluster_df,
            use_container_width=True,
            height=350
        )


        # ----------------------------------------------------
        # FINAL DATASET
        # ----------------------------------------------------

        with st.expander(
            "View Final Tracking Dataset"
        ):

            st.dataframe(
                tracking_result[
                    "df_tracking"
                ],
                use_container_width=True,
                height=400
            )


        # ----------------------------------------------------
        # OPTIMIZED PARAMETERS TABLE
        # ----------------------------------------------------

        optimized_parameters = pd.DataFrame({

            "Parameter": [

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

                tracking_result["DHI"],

                tracking_result["start_block"],

                tracking_result["end_block"],

                tracking_result["max_block"],

                tracking_result["east_limit"],

                tracking_result["west_limit"],

                solar["actual_peak"],

                tracking_result["forecast"].max(),

                tracking_result["peak_error"],

                tracking_result["block_error"],

                tracking_result["energy_error"],

                tracking_result["score"]
            ]
        })


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary = pd.DataFrame({

            "Metric": [

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

            "Tracking": [

                tracking_result["DHI"],

                tracking_result["start_block"],

                tracking_result["end_block"],

                tracking_result["max_block"],

                tracking_result["east_limit"],

                tracking_result["west_limit"],

                tracking_result["block_error"],

                tracking_result["peak_error"],

                tracking_result["energy_error"],

                tracking_result["score"],

                tracking_result["forecast"].max()
            ]
        })


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        excel_output = create_excel_download(

            "Tracking",

            df_area,

            tracking_result=
                tracking_result,

            summary=summary,

            optimized_parameters=
                optimized_parameters
        )

        st.download_button(

            "⬇ Download Tracking Results",

            data=excel_output,

            file_name=(
                "Solar_Tracking_Results.xlsx"
            ),

            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),

            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Solar Forecast Correction | "
    "Fixed / Tracking Optimization"
)
