# ============================================================
# VCAST LOSS CORRECTION
# ============================================================

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loss Correction VCast — Solar Suite",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <h1 style='text-align:center;
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    font-size:40px;
    font-weight:800;'>
    ⚡ Solar Suite
    </h1>

    <p style='text-align:center;
    color:gray;
    font-size:14px;'>
    Forecast Correction Platform
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    div[data-testid="stDataEditor"] {
        border-radius: 10px;
    }

    div[data-testid="stNumberInput"] {
        margin-bottom: 4px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

CLUSTERS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

N_BLOCKS = 96


# ============================================================
# SESSION STATE
# ============================================================

if "vcast_input" not in st.session_state:

    st.session_state.vcast_input = pd.DataFrame(
        {
            "Actual": np.zeros(N_BLOCKS),
            "GHI C11": np.zeros(N_BLOCKS),
            "GHI C12": np.zeros(N_BLOCKS),
            "GHI C13": np.zeros(N_BLOCKS),
            "GHI C14": np.zeros(N_BLOCKS),
            "GHI C15": np.zeros(N_BLOCKS),
        }
    )


# ------------------------------------------------------------
# Fixed parameters
# ------------------------------------------------------------

if "vcast_fixed_loss" not in st.session_state:
    st.session_state.vcast_fixed_loss = 0.0


# ------------------------------------------------------------
# Tracking parameters
# ------------------------------------------------------------

if "vcast_tracking_loss" not in st.session_state:
    st.session_state.vcast_tracking_loss = 0.0

if "vcast_dhi" not in st.session_state:
    st.session_state.vcast_dhi = 5

if "vcast_start_block" not in st.session_state:
    st.session_state.vcast_start_block = 20

if "vcast_end_block" not in st.session_state:
    st.session_state.vcast_end_block = 72

if "vcast_max_block" not in st.session_state:
    st.session_state.vcast_max_block = 50

if "vcast_east_limit" not in st.session_state:
    st.session_state.vcast_east_limit = 45

if "vcast_west_limit" not in st.session_state:
    st.session_state.vcast_west_limit = 45


# ============================================================
# TITLE
# ============================================================

st.title(
    "Guruji ne kaha tha VCast Correct kardo bhyii🛐!!"
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload VCast Excel Workbook",
    type=["xlsx", "xls"],
)


if uploaded_file is None:

    st.info(
        "Please upload the VCast workbook to continue."
    )

    st.stop()


file_bytes = uploaded_file.getvalue()


# ============================================================
# WORKBOOK DETECTION
# ============================================================

@st.cache_data(show_spinner=False)
def detect_vcast_workbook(file_bytes):

    xls = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    sheets = xls.sheet_names

    if "Fixed-C11" in sheets:

        plant_type = "VCast"
        fixed_sheet = "Fixed-C11"

    elif "Fixed-CL1" in sheets:

        plant_type = "Cluster"
        fixed_sheet = "Fixed-CL1"

    elif "Fixed" in sheets:

        plant_type = "Fixed"
        fixed_sheet = "Fixed"

    else:

        raise ValueError(
            "Workbook does not contain Fixed, Fixed-CL1 or Fixed-C11."
        )

    return (
        plant_type,
        fixed_sheet,
        sheets,
    )


try:

    plant_type, fixed_sheet, sheet_names = (
        detect_vcast_workbook(file_bytes)
    )

except Exception as e:

    st.error(
        f"Unable to identify workbook: {e}"
    )

    st.stop()


# ============================================================
# VCAST VALIDATION
# ============================================================

if plant_type != "VCast":

    st.error(
        "This page is specifically for VCast workbooks. "
        "The uploaded workbook does not contain Fixed-C11."
    )

    st.stop()


# ============================================================
# READ FIXED-C11
# ============================================================

@st.cache_data(show_spinner=False)
def read_vcast_input(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Fixed-C11",
        header=1,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required = [
        "Date",
        "Actual",
    ]

    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"Required column '{col}' "
                "not found in Fixed-C11."
            )

    # --------------------------------------------------------
    # Stop at first blank Date
    # --------------------------------------------------------

    valid_date = df["Date"].notna()

    if not valid_date.any():

        raise ValueError(
            "No valid Date rows found in Fixed-C11."
        )

    first_blank = np.where(
        ~valid_date.to_numpy()
    )[0]

    if len(first_blank):

        df = df.iloc[
            :first_blank[0]
        ].copy()

    else:

        df = df.loc[
            valid_date
        ].copy()

    df = df.iloc[
        :N_BLOCKS
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True,
    )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    return df


try:

    source_df = read_vcast_input(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to read Fixed-C11: {e}"
    )

    st.stop()


# ============================================================
# READ RESULT / GHI
# ============================================================

@st.cache_data(show_spinner=False)
def read_result_ghi(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Result",
        usecols=range(6),
    )

    df.columns = [
        "Block",
        *GHI_COLS,
    ]

    df = df[
        pd.to_numeric(
            df["Block"],
            errors="coerce",
        ).notna()
    ].copy()

    for col in GHI_COLS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

    df["Block"] = pd.to_numeric(
        df["Block"],
        errors="coerce",
    )

    return df


try:

    result_ghi = read_result_ghi(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to read Result sheet: {e}"
    )

    st.stop()


# ============================================================
# ALIGN 96 BLOCKS
# ============================================================

n = min(
    len(source_df),
    len(result_ghi),
    N_BLOCKS,
)


source_df = source_df.iloc[
    :n
].copy()

result_ghi = result_ghi.iloc[
    :n
].copy()


# ============================================================
# DATES
# ============================================================

dates = pd.to_datetime(
    source_df["Date"],
    errors="coerce",
)


if dates.isna().any():

    st.error(
        "Invalid dates found in Fixed-C11."
    )

    st.stop()


# ============================================================
# AREA & EFFICIENCY
# ============================================================

@st.cache_data(show_spinner=False)
def read_area_efficiency(file_bytes):

    area = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    area.columns = (
        area.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False,
        )
        .str.strip()
    )

    return area


@st.cache_data(show_spinner=False)
def read_effective_areas(file_bytes):

    raw = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=None,
    )

    # Fixed effective areas: P3:P7
    fixed_weights = (
        pd.to_numeric(
            raw.iloc[
                2:7,
                15,
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    # Tracking effective areas: P29:P33
    tracking_weights = (
        pd.to_numeric(
            raw.iloc[
                28:33,
                15,
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    return (
        fixed_weights,
        tracking_weights,
    )


try:

    area_df = read_area_efficiency(
        file_bytes
    )

    (
        fixed_weights,
        tracking_weights,
    ) = read_effective_areas(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to read Area & Efficiency: {e}"
    )

    st.stop()


if len(fixed_weights) != 5:

    st.error(
        "Could not read five Fixed effective areas."
    )

    st.stop()


if len(tracking_weights) != 5:

    st.error(
        "Could not read five Tracking effective areas."
    )

    st.stop()


# ============================================================
# STANDARD PV EFFICIENCY
# ============================================================

standard_efficiency = pd.to_numeric(
    area_df[
        "Standard PV Efficiency (%)"
    ],
    errors="coerce",
).to_numpy(
    dtype=float
)


if len(standard_efficiency) < 5:

    st.error(
        "Less than five Standard PV Efficiency values found."
    )

    st.stop()


standard_efficiency = (
    standard_efficiency[:5]
)


# ============================================================
# FORECAST CONFIG
# ============================================================

@st.cache_data(show_spinner=False)
def read_latitude(file_bytes):

    config = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )

    return float(
        config.loc[
            0,
            "Lat",
        ]
    )


try:

    lat = read_latitude(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to read latitude: {e}"
    )

    st.stop()


# ============================================================
# CONFIG TILT ANGLE
# ============================================================

@st.cache_data(show_spinner=False)
def read_tilt(file_bytes):

    tilt_df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle",
        header=7,
    )

    tilt_df.columns = (
        tilt_df.columns
        .astype(str)
        .str.strip()
    )

    tilt_df = tilt_df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    tilt_df = tilt_df.dropna(
        subset=["Fixed"]
    ).copy()

    tilt_df["Month_Num"] = pd.to_numeric(
        tilt_df["Month_Num"],
        errors="coerce",
    )

    tilt_df["Fixed"] = pd.to_numeric(
        tilt_df["Fixed"],
        errors="coerce",
    )

    return (
        tilt_df
        .dropna(
            subset=["Month_Num"]
        )
        .set_index(
            "Month_Num"
        )["Fixed"]
        .to_dict()
    )


try:

    tilt_dict = read_tilt(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to read tilt configuration: {e}"
    )

    st.stop()


# ============================================================
# NUMPY INPUTS
# ============================================================

actual = source_df[
    "Actual"
].to_numpy(
    dtype=float
)

ghi_matrix = np.column_stack(
    [
        result_ghi[col].to_numpy(
            dtype=float
        )
        for col in GHI_COLS
    ]
)


blocks = np.arange(
    1,
    n + 1,
    dtype=float,
)


# ============================================================
# WORKBOOK REFERENCE DATE
# ============================================================

first_date = pd.Timestamp(
    year=2025,
    month=1,
    day=1,
)


day_offset = (
    dates
    - first_date
).dt.days.to_numpy(
    dtype=float
)


# ============================================================
# SOLAR GEOMETRY
# ============================================================

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


months = dates.dt.month.to_numpy()


tilt = np.array(
    [
        tilt_dict.get(
            float(month),
            0,
        )
        for month in months
    ],
    dtype=float,
)


a_plus_b = (
    elevation
    + tilt
)


sin_a = np.sin(
    np.radians(
        elevation
    )
)


sin_ab = np.sin(
    np.radians(
        a_plus_b
    )
)


sin_a_safe = np.where(
    np.abs(sin_a) < 1e-8,
    1e-8,
    sin_a,
)


# ============================================================
# FIXED POA
# ============================================================

fixed_poa = (
    ghi_matrix
    * sin_ab[:, None]
    / sin_a_safe[:, None]
)


# ============================================================
# ACTUAL MASK / METRICS BASE
# ============================================================

valid_mask = (
    np.isfinite(actual)
    &
    (actual != 0)
)


if not valid_mask.any():

    st.error(
        "Actual power contains no valid non-zero values."
    )

    st.stop()


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

    st.error(
        "Actual peak must be greater than zero."
    )

    st.stop()


# ============================================================
# HELPER: APPLY EFFICIENCY LOSS
# ============================================================

def efficiency_factor_from_loss(
    loss,
):

    net_efficiency = (
        standard_efficiency
        - loss
    )

    net_efficiency = np.maximum(
        net_efficiency,
        0,
    )

    return np.divide(
        net_efficiency,
        standard_efficiency,
        out=np.zeros_like(
            standard_efficiency
        ),
        where=(
            standard_efficiency != 0
        ),
    )


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    loss,
):

    factor = (
        efficiency_factor_from_loss(
            loss
        )
    )

    adjusted_area = (
        fixed_weights
        * factor
    )

    power_matrix = (
        fixed_poa
        * adjusted_area[None, :]
        / 1_000_000
    )

    forecast = (
        power_matrix.sum(
            axis=1
        )
    )

    return (
        forecast,
        power_matrix,
        adjusted_area,
    )


# ============================================================
# FIXED LOSS OPTIMIZATION
#
# IMPORTANT:
# Error % is selected by minimum PEAK ERROR.
# ============================================================

@st.cache_data(show_spinner=False)
def optimize_fixed_loss(
    actual_tuple,
    fixed_poa_tuple,
    fixed_weights_tuple,
    standard_efficiency_tuple,
):

    actual_arr = np.array(
        actual_tuple,
        dtype=float,
    )

    poa_arr = np.array(
        fixed_poa_tuple,
        dtype=float,
    )

    weights = np.array(
        fixed_weights_tuple,
        dtype=float,
    )

    efficiencies = np.array(
        standard_efficiency_tuple,
        dtype=float,
    )

    mask = (
        np.isfinite(actual_arr)
        &
        (actual_arr != 0)
    )

    act = actual_arr[
        mask
    ]

    peak = np.max(
        act
    )

    energy = np.sum(
        act
    )

    max_loss = np.min(
        efficiencies
    )

    best_loss = 0.0
    best_peak_error = np.inf
    best_forecast = None

    for loss in np.arange(
        0,
        max_loss + 0.0001,
        0.1,
    ):

        net_eff = np.maximum(
            efficiencies - loss,
            0,
        )

        factor = np.divide(
            net_eff,
            efficiencies,
            out=np.zeros_like(
                efficiencies
            ),
            where=(
                efficiencies != 0
            ),
        )

        adjusted = (
            weights
            * factor
        )

        forecast = (
            poa_arr
            * adjusted[None, :]
            / 1_000_000
        ).sum(
            axis=1
        )

        forecast_day = forecast[
            mask
        ]

        if len(forecast_day) == 0:
            continue

        predicted_peak = np.max(
            forecast_day
        )

        peak_error = abs(
            peak
            - predicted_peak
        )

        if peak_error < best_peak_error:

            best_peak_error = (
                peak_error
            )

            best_loss = (
                float(loss)
            )

            best_forecast = (
                forecast.copy()
            )

    return (
        best_loss,
        best_peak_error,
        best_forecast,
    )


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    loss,
    dhi_percent,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
):

    start_block = int(
        round(start_block)
    )

    end_block = int(
        round(end_block)
    )

    max_block = int(
        round(max_block)
    )

    dhi_percent = float(
        dhi_percent
    )

    east_limit = float(
        east_limit
    )

    west_limit = float(
        west_limit
    )

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
            ),
        ),

        np.minimum(
            89,
            m2
            * (
                blocks
                - max_block
            ),
        ),
    )

    panel = np.where(

        blocks < max_block,

        np.where(

            zenith
            < abs(east_limit),

            zenith,

            abs(east_limit),
        ),

        np.where(

            (
                (blocks > max_block)
                &
                (zenith > west_limit)
            ),

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

    # --------------------------------------------------------
    # DHI
    # --------------------------------------------------------

    dhi = (
        ghi_matrix
        * dhi_percent
        / 100
    )

    # --------------------------------------------------------
    # DNI
    # --------------------------------------------------------

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    # --------------------------------------------------------
    # Apply Tracking Efficiency Loss
    #
    # Tracking uses ORIGINAL tracking effective areas,
    # then applies the user-selected tracking loss.
    # --------------------------------------------------------

    factor = (
        efficiency_factor_from_loss(
            loss
        )
    )

    adjusted_tracking_weights = (
        tracking_weights
        * factor
    )

    tracking_power_matrix = (
        dni
        * adjusted_tracking_weights[None, :]
        / 1_000_000
    )

    tracking_forecast = (
        tracking_power_matrix.sum(
            axis=1
        )
    )

    return {
        "forecast": tracking_forecast,
        "power_matrix": tracking_power_matrix,
        "zenith": zenith,
        "panel": panel,
        "dni": dni,
    }


# ============================================================
# TRACKING LOSS OPTIMIZATION
#
# Error % is calculated first using the default tracking
# parameters. Then the remaining tracking parameters are
# optimized separately.
# ============================================================

@st.cache_data(show_spinner=False)
def optimize_tracking_loss(
    actual_tuple,
    ghi_tuple,
    tracking_weights_tuple,
    standard_efficiency_tuple,
    default_dhi,
    default_start,
    default_end,
    default_max,
    default_east,
    default_west,
):

    actual_arr = np.array(
        actual_tuple,
        dtype=float,
    )

    ghi_arr = np.array(
        ghi_tuple,
        dtype=float,
    )

    weights = np.array(
        tracking_weights_tuple,
        dtype=float,
    )

    efficiencies = np.array(
        standard_efficiency_tuple,
        dtype=float,
    )

    mask = (
        np.isfinite(actual_arr)
        &
        (actual_arr != 0)
    )

    act = actual_arr[
        mask
    ]

    actual_peak_local = np.max(
        act
    )

    max_loss = np.min(
        efficiencies
    )

    best_loss = 0.0
    best_peak_error = np.inf

    # Local calculation without modifying global state.
    local_blocks = np.arange(
        1,
        len(actual_arr) + 1,
        dtype=float,
    )

    for loss in np.arange(
        0,
        max_loss + 0.0001,
        0.1,
    ):

        start_block = int(
            default_start
        )

        end_block = int(
            default_end
        )

        max_block = int(
            default_max
        )

        if not (
            start_block
            < max_block
            < end_block
        ):

            continue

        d1 = (
            start_block
            - 1
            - max_block
        )

        d2 = (
            end_block
            + 1
            - max_block
        )

        if d1 == 0 or d2 == 0:
            continue

        m1 = 90 / d1
        m2 = 90 / d2

        zenith = np.where(

            local_blocks <= max_block,

            np.minimum(
                89,
                m1
                * (
                    local_blocks
                    - max_block
                ),
            ),

            np.minimum(
                89,
                m2
                * (
                    local_blocks
                    - max_block
                ),
            ),
        )

        panel = np.where(

            local_blocks < max_block,

            np.where(
                zenith
                < abs(default_east),
                zenith,
                abs(default_east),
            ),

            np.where(
                (
                    (local_blocks > max_block)
                    &
                    (zenith > default_west)
                ),
                default_west,
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
            ghi_arr
            * default_dhi
            / 100
        )

        dni = (
            ghi_arr
            - dhi
        ) / cos_alpha[:, None]

        net_eff = np.maximum(
            efficiencies - loss,
            0,
        )

        factor = np.divide(
            net_eff,
            efficiencies,
            out=np.zeros_like(
                efficiencies
            ),
            where=(
                efficiencies != 0
            ),
        )

        adjusted_weights = (
            weights
            * factor
        )

        forecast = (
            dni
            * adjusted_weights[None, :]
            / 1_000_000
        ).sum(
            axis=1
        )

        forecast_day = forecast[
            mask
        ]

        if len(forecast_day) == 0:
            continue

        predicted_peak = np.max(
            forecast_day
        )

        peak_error = abs(
            actual_peak_local
            - predicted_peak
        )

        if peak_error < best_peak_error:

            best_peak_error = (
                peak_error
            )

            best_loss = (
                float(loss)
            )

    return (
        best_loss,
        best_peak_error,
    )


# ============================================================
# TRACKING PARAMETER OPTIMIZATION
#
# Efficiency loss is kept as the automatically calculated
# tracking loss. The remaining Tracking parameters are then
# optimized.
# ============================================================

@st.cache_data(show_spinner=False)
def optimize_tracking_parameters(
    actual_tuple,
    ghi_tuple,
    tracking_weights_tuple,
    standard_efficiency_tuple,
    loss,
):

    actual_arr = np.array(
        actual_tuple,
        dtype=float,
    )

    ghi_arr = np.array(
        ghi_tuple,
        dtype=float,
    )

    weights = np.array(
        tracking_weights_tuple,
        dtype=float,
    )

    efficiencies = np.array(
        standard_efficiency_tuple,
        dtype=float,
    )

    local_blocks = np.arange(
        1,
        len(actual_arr) + 1,
        dtype=float,
    )

    mask = (
        np.isfinite(actual_arr)
        &
        (actual_arr != 0)
    )

    act = actual_arr[
        mask
    ]

    actual_peak_local = np.max(
        act
    )

    actual_energy_local = np.sum(
        act
    )

    factor = np.divide(
        np.maximum(
            efficiencies - loss,
            0,
        ),
        efficiencies,
        out=np.zeros_like(
            efficiencies
        ),
        where=(
            efficiencies != 0
        ),
    )

    adjusted_weights = (
        weights
        * factor
    )

    def objective_tracking(x):

        dhi = int(
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

        d1 = (
            start_block
            - 1
            - max_block
        )

        d2 = (
            end_block
            + 1
            - max_block
        )

        if d1 == 0 or d2 == 0:
            return 1e9

        m1 = 90 / d1
        m2 = 90 / d2

        zenith = np.where(

            local_blocks <= max_block,

            np.minimum(
                89,
                m1
                * (
                    local_blocks
                    - max_block
                ),
            ),

            np.minimum(
                89,
                m2
                * (
                    local_blocks
                    - max_block
                ),
            ),
        )

        panel = np.where(

            local_blocks < max_block,

            np.where(
                zenith < abs(east_limit),
                zenith,
                abs(east_limit),
            ),

            np.where(
                (
                    (local_blocks > max_block)
                    &
                    (zenith > west_limit)
                ),
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

        dhi_array = (
            ghi_arr
            * dhi
            / 100
        )

        dni = (
            ghi_arr
            - dhi_array
        ) / cos_alpha[:, None]

        prediction = (
            dni
            * adjusted_weights[None, :]
            / 1_000_000
        ).sum(
            axis=1
        )

        if not np.all(
            np.isfinite(prediction)
        ):

            return 1e9

        prediction_day = prediction[
            mask
        ]

        if len(prediction_day) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    act
                    - prediction_day
                )
            )
            / actual_peak_local
        )

        peak_error = (
            abs(
                actual_peak_local
                - prediction_day.max()
            )
            / actual_peak_local
        )

        energy_error = (
            abs(
                actual_energy_local
                - prediction_day.sum()
            )
            / actual_energy_local
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(

        objective_tracking,

        bounds=[
            (0, 10),
            (10, 30),
            (65, 80),
            (47, 53),
            (10, 70),
            (10, 70),
        ],

        strategy="best1bin",
        maxiter=40,
        popsize=15,
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

    return {
        "dhi": int(best[0]),
        "start": int(best[1]),
        "end": int(best[2]),
        "max": int(best[3]),
        "east": int(best[4]),
        "west": int(best[5]),
        "score": float(result.fun),
    }


# ============================================================
# AUTOMATIC OPTIMIZATION
# ============================================================

st.subheader(
    "⚙️ Automatic Optimization"
)

st.caption(
    "Error % is selected using minimum Peak Error. "
    "Tracking geometry parameters are optimized separately."
)


if st.button(
    "🚀 Optimize VCast",
    use_container_width=True,
    type="primary",
):

    with st.spinner(
        "Calculating VCast parameters..."
    ):

        # ----------------------------------------------------
        # FIXED LOSS
        # ----------------------------------------------------

        (
            fixed_loss_auto,
            fixed_peak_error_auto,
            _,
        ) = optimize_fixed_loss(

            tuple(actual),

            tuple(
                fixed_poa.ravel()
            ),

            tuple(
                fixed_weights
            ),

            tuple(
                standard_efficiency
            ),
        )

        # ----------------------------------------------------
        # TRACKING LOSS
        # ----------------------------------------------------

        (
            tracking_loss_auto,
            tracking_peak_error_auto,
        ) = optimize_tracking_loss(

            tuple(actual),

            tuple(
                ghi_matrix
            ),

            tuple(
                tracking_weights
            ),

            tuple(
                standard_efficiency
            ),

            st.session_state.vcast_dhi,

            st.session_state.vcast_start_block,

            st.session_state.vcast_end_block,

            st.session_state.vcast_max_block,

            st.session_state.vcast_east_limit,

            st.session_state.vcast_west_limit,
        )

        # ----------------------------------------------------
        # TRACKING PARAMETERS
        # ----------------------------------------------------

        tracking_params = (
            optimize_tracking_parameters(

                tuple(actual),

                tuple(
                    ghi_matrix
                ),

                tuple(
                    tracking_weights
                ),

                tuple(
                    standard_efficiency
                ),

                tracking_loss_auto,
            )
        )

    # --------------------------------------------------------
    # WRITE AUTOMATIC VALUES INTO WIDGET STATES
    #
    # This is the important part that fixes stale parameter UI.
    # --------------------------------------------------------

    st.session_state.vcast_fixed_loss = (
        float(fixed_loss_auto)
    )

    st.session_state.vcast_tracking_loss = (
        float(tracking_loss_auto)
    )

    st.session_state.vcast_dhi = (
        tracking_params["dhi"]
    )

    st.session_state.vcast_start_block = (
        tracking_params["start"]
    )

    st.session_state.vcast_end_block = (
        tracking_params["end"]
    )

    st.session_state.vcast_max_block = (
        tracking_params["max"]
    )

    st.session_state.vcast_east_limit = (
        tracking_params["east"]
    )

    st.session_state.vcast_west_limit = (
        tracking_params["west"]
    )

    st.rerun()


# ============================================================
# FIXED PLANT
# ============================================================

st.divider()

st.subheader(
    "☀️ Fixed Plant"
)


st.markdown(
    "### ⚙️ Efficiency Loss"
)


fixed_loss = st.number_input(

    "Error %",

    min_value=0.0,
    max_value=50.0,

    step=0.1,

    key="vcast_fixed_loss",

    help=(
        "Automatically calculated using minimum "
        "Peak Error. You can manually edit it."
    ),
)


# ============================================================
# FIXED FORECAST
# ============================================================

(
    fixed_forecast,
    fixed_power_matrix,
    final_fixed_weights,
) = calculate_fixed_forecast(
    fixed_loss
)


# ============================================================
# FIXED METRICS
# ============================================================

fixed_day = fixed_forecast[
    valid_mask
]


fixed_block_error = (
    np.mean(
        np.abs(
            actual_day
            - fixed_day
        )
    )
    / actual_peak
)


fixed_peak_error = (
    abs(
        actual_peak
        - fixed_day.max()
    )
    / actual_peak
)


fixed_energy_error = (
    abs(
        actual_energy
        - fixed_day.sum()
    )
    / actual_energy
)


fixed_score = (
    0.80 * fixed_block_error
    + 0.10 * fixed_peak_error
    + 0.10 * fixed_energy_error
)


# ============================================================
# FIXED GRAPH
# ============================================================

fixed_fig = go.Figure()


fixed_fig.add_trace(
    go.Scatter(
        x=blocks,
        y=actual,
        name="Actual",
        mode="lines",
        line=dict(
            color="#ef4444",
            width=3,
        ),
    )
)


fixed_fig.add_trace(
    go.Scatter(
        x=blocks,
        y=fixed_forecast,
        name="Fixed Forecast",
        mode="lines",
        line=dict(
            color="#00c6ff",
            width=3,
        ),
    )
)


fixed_fig.update_layout(
    height=500,
    template="streamlit",
    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),

    xaxis_title="Block",
    yaxis_title="Power (MW)",
)


st.plotly_chart(
    fixed_fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
    },
)


# ============================================================
# TRACKING PLANT
# ============================================================

st.divider()

st.subheader(
    "🔄 Tracking Plant"
)


st.markdown(
    "### ⚙️ Tracking Parameters"
)


# ============================================================
# TRACKING INPUTS
# ============================================================

track_col1, track_col2 = st.columns(2)


with track_col1:

    tracking_loss = st.number_input(

        "Error %",

        min_value=0.0,
        max_value=50.0,

        step=0.1,

        key="vcast_tracking_loss",

        help=(
            "Automatically calculated using minimum "
            "Peak Error. You can manually edit it."
        ),
    )


    dhi = st.number_input(

        "DHI (%)",

        min_value=0,
        max_value=100,

        step=1,

        key="vcast_dhi",
    )


    start_block = st.number_input(

        "GHI Starting Block",

        min_value=1,
        max_value=N_BLOCKS,

        step=1,

        key="vcast_start_block",
    )


with track_col2:

    end_block = st.number_input(

        "GHI Ending Block",

        min_value=1,
        max_value=N_BLOCKS,

        step=1,

        key="vcast_end_block",
    )


    max_block = st.number_input(

        "GHI Max Block",

        min_value=1,
        max_value=N_BLOCKS,

        step=1,

        key="vcast_max_block",
    )


    east_limit = st.number_input(

        "East Tracking Limit (°)",

        min_value=0,
        max_value=90,

        step=1,

        key="vcast_east_limit",
    )


west_limit = st.number_input(

    "West Tracking Limit (°)",

    min_value=0,
    max_value=90,

    step=1,

    key="vcast_west_limit",
)


# ============================================================
# TRACKING VALIDATION
# ============================================================

if not (
    start_block
    < max_block
    < end_block
):

    st.warning(
        "GHI Starting Block must be less than "
        "GHI Max Block, and GHI Max Block must "
        "be less than GHI Ending Block."
    )

    st.stop()


# ============================================================
# TRACKING FORECAST
# ============================================================

tracking_result = calculate_tracking(

    tracking_loss,

    dhi,

    start_block,

    end_block,

    max_block,

    east_limit,

    west_limit,
)


if tracking_result is None:

    st.error(
        "Unable to calculate Tracking forecast "
        "with the current parameters."
    )

    st.stop()


tracking_forecast = (
    tracking_result["forecast"]
)

tracking_power_matrix = (
    tracking_result["power_matrix"]
)

zenith = (
    tracking_result["zenith"]
)

panel = (
    tracking_result["panel"]
)

dni = (
    tracking_result["dni"]
)


# ============================================================
# TRACKING METRICS
# ============================================================

tracking_day = tracking_forecast[
    valid_mask
]


tracking_block_error = (
    np.mean(
        np.abs(
            actual_day
            - tracking_day
        )
    )
    / actual_peak
)


tracking_peak_error = (
    abs(
        actual_peak
        - tracking_day.max()
    )
    / actual_peak
)


tracking_energy_error = (
    abs(
        actual_energy
        - tracking_day.sum()
    )
    / actual_energy
)


tracking_score = (
    0.80 * tracking_block_error
    + 0.10 * tracking_peak_error
    + 0.10 * tracking_energy_error
)


# ============================================================
# IMPORTANT TIME BLOCKS
# ============================================================

time_start = datetime.strptime(
    "00:00",
    "%H:%M",
)


time_blocks = [
    (
        f"{(time_start + timedelta(minutes=15 * i)).strftime('%H:%M')}"
        f" - "
        f"{(time_start + timedelta(minutes=15 * (i + 1))).strftime('%H:%M')}"
    )
    for i in range(N_BLOCKS)
]


def get_time_block(
    block
):

    if (
        block is None
        or block < 1
        or block > N_BLOCKS
    ):

        return "—"

    return time_blocks[
        int(block) - 1
    ]


# ============================================================
# IMPORTANT TIME BLOCKS EXPANDER
# ============================================================

lookup_df = pd.DataFrame(
    {
        "Parameter": [
            "GHI Starting Block",
            "GHI Ending Block",
            "GHI Max Block",
            "Actual Generation Available Block (Lower Limit)",
            "Actual Generation Effective Block (Upper Limit)",
        ],

        "Block": [
            start_block,
            end_block,
            max_block,
            start_block + 3,
            end_block - 3,
        ],
    }
)


lookup_df["Time Block"] = (
    lookup_df["Block"]
    .apply(
        get_time_block
    )
)


with st.expander(
    "📅 Important Time Blocks"
):

    st.dataframe(
        lookup_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TRACKING GRAPH
# ============================================================

tracking_fig = go.Figure()


tracking_fig.add_trace(
    go.Scatter(
        x=blocks,
        y=actual,
        name="Actual",
        mode="lines",
        line=dict(
            color="#ef4444",
            width=3,
        ),
    )
)


tracking_fig.add_trace(
    go.Scatter(
        x=blocks,
        y=tracking_forecast,
        name="Tracking Forecast",
        mode="lines",
        line=dict(
            color="#22c55e",
            width=3,
        ),
    )
)


tracking_fig.update_layout(
    height=500,
    template="streamlit",
    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),

    xaxis_title="Block",
    yaxis_title="Power (MW)",
)


st.plotly_chart(
    tracking_fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
    },
)
