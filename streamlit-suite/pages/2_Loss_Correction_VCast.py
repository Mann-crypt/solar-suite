# ============================================================
# VCAST LOSS CORRECTION
# ============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datetime import datetime, timedelta
from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VCast Loss Correction — Solar Suite",
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

    <p style='text-align:center;color:gray;font-size:14px;'>
    Forecast Correction Platform
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

st.sidebar.info(
    "VCast Loss Correction\n\n"
    "Fixed + Tracking calculation"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    div[data-testid="stNumberInput"] {
        margin-bottom: 8px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
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

N_CLUSTERS = 5


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_FIXED_PARAMS = {
    "fixed_error": 0.0,
}

DEFAULT_TRACKING_PARAMS = {
    "tracking_error": 0.0,
    "dhi": 0.0,
    "ghi_start": 20,
    "ghi_end": 72,
    "ghi_max": 48,
    "east_limit": 30,
    "west_limit": 30,
}


if "vcast_file" not in st.session_state:
    st.session_state.vcast_file = None


if "vcast_loaded" not in st.session_state:
    st.session_state.vcast_loaded = False


if "vcast_fixed_params" not in st.session_state:
    st.session_state.vcast_fixed_params = (
        DEFAULT_FIXED_PARAMS.copy()
    )


if "vcast_tracking_params" not in st.session_state:
    st.session_state.vcast_tracking_params = (
        DEFAULT_TRACKING_PARAMS.copy()
    )


if "vcast_fixed_optimized" not in st.session_state:
    st.session_state.vcast_fixed_optimized = False


if "vcast_tracking_optimized" not in st.session_state:
    st.session_state.vcast_tracking_optimized = False


# ============================================================
# TITLE
# ============================================================

st.title(
    "Guruji ne kaha tha RT Correct kardo bhyii🛐!!"
)

st.caption(
    "VCast Loss Correction"
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
        "Upload a VCast workbook containing the "
        "`Fixed-C11` sheet to begin."
    )

    st.stop()


# ============================================================
# RESET WHEN NEW FILE IS UPLOADED
# ============================================================

file_signature = (
    uploaded_file.name,
    uploaded_file.size,
)


if (
    st.session_state.vcast_file
    != file_signature
):

    st.session_state.vcast_file = file_signature

    st.session_state.vcast_loaded = False

    st.session_state.vcast_fixed_params = (
        DEFAULT_FIXED_PARAMS.copy()
    )

    st.session_state.vcast_tracking_params = (
        DEFAULT_TRACKING_PARAMS.copy()
    )

    st.session_state.vcast_fixed_optimized = False

    st.session_state.vcast_tracking_optimized = False


# ============================================================
# READ WORKBOOK
# ============================================================

@st.cache_data(show_spinner=False)
def load_vcast_workbook(file_bytes):

    from io import BytesIO

    buffer = BytesIO(file_bytes)

    xls = pd.ExcelFile(buffer)

    return xls.sheet_names


sheet_names = load_vcast_workbook(
    uploaded_file.getvalue()
)


# ============================================================
# VCAST IDENTIFICATION
# ============================================================

if "Fixed-C11" not in sheet_names:

    st.error(
        "This workbook is not identified as VCast. "
        "The required `Fixed-C11` sheet was not found."
    )

    st.stop()


st.success(
    f"VCast workbook detected: `{uploaded_file.name}`"
)


# ============================================================
# LOAD ALL REQUIRED DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_vcast_data(file_bytes):

    from io import BytesIO

    buffer = BytesIO(file_bytes)

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    area_eff = pd.read_excel(
        buffer,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    area_eff.columns = (
        area_eff.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False,
        )
        .str.strip()
    )

    area_eff = area_eff[
        area_eff["S.No."].notna()
    ].copy()

    area_eff.reset_index(
        drop=True,
        inplace=True,
    )

    # --------------------------------------------------------
    # Raw Area & Efficiency
    # --------------------------------------------------------

    area_raw = pd.read_excel(
        buffer,
        sheet_name="Area & Efficiency",
        header=None,
    )

    # Fixed effective areas: P3:P7
    fixed_weights = (
        pd.to_numeric(
            area_raw.iloc[2:7, 15],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    # Tracking effective areas: P29:P33
    tracking_weights = (
        pd.to_numeric(
            area_raw.iloc[28:33, 15],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    if len(fixed_weights) != 5:
        raise ValueError(
            "Could not read Fixed effective areas."
        )

    if len(tracking_weights) != 5:
        raise ValueError(
            "Could not read Tracking effective areas."
        )

    # --------------------------------------------------------
    # Standard PV Efficiency
    # --------------------------------------------------------

    standard_efficiency = pd.to_numeric(
        area_eff[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    if len(standard_efficiency) < 5:
        raise ValueError(
            "Less than 5 Standard PV Efficiency "
            "values were found."
        )

    standard_efficiency = (
        standard_efficiency[:5]
    )

    # --------------------------------------------------------
    # Forecast Config
    # --------------------------------------------------------

    config = pd.read_excel(
        buffer,
        sheet_name="Forecast Config",
        header=8,
    )

    lat = float(
        config.loc[0, "Lat"]
    )

    # --------------------------------------------------------
    # Config Tilt Angle
    # --------------------------------------------------------

    tilt_df = pd.read_excel(
        buffer,
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

    month_to_tilt = (
        tilt_df
        .dropna(
            subset=["Month_Num"]
        )
        .set_index("Month_Num")["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # Result / GHI
    # --------------------------------------------------------

    ghi_df = pd.read_excel(
        buffer,
        sheet_name="Result",
        usecols=range(6),
    )

    ghi_df.columns = [
        "Block",
        *GHI_COLS,
    ]

    ghi_df = ghi_df[
        pd.to_numeric(
            ghi_df["Block"],
            errors="coerce",
        ).notna()
    ].copy()

    for col in GHI_COLS:

        ghi_df[col] = pd.to_numeric(
            ghi_df[col],
            errors="coerce",
        ).fillna(0)

    # --------------------------------------------------------
    # Fixed-C11
    # --------------------------------------------------------

    fixed_c11 = pd.read_excel(
        buffer,
        sheet_name="Fixed-C11",
        header=1,
    )

    fixed_c11.columns = (
        fixed_c11.columns
        .astype(str)
        .str.strip()
    )

    if "Date" not in fixed_c11.columns:
        raise ValueError(
            "`Date` column not found in Fixed-C11."
        )

    if "Actual" not in fixed_c11.columns:
        raise ValueError(
            "`Actual` column not found in Fixed-C11."
        )

    date_valid = (
        fixed_c11["Date"].notna()
    )

    if not date_valid.any():
        raise ValueError(
            "No valid Date rows found in Fixed-C11."
        )

    first_blank = np.where(
        ~date_valid.to_numpy()
    )[0]

    if len(first_blank) > 0:

        fixed_c11 = fixed_c11.iloc[
            :first_blank[0]
        ].copy()

    else:

        fixed_c11 = fixed_c11.loc[
            date_valid
        ].copy()

    fixed_c11.reset_index(
        drop=True,
        inplace=True,
    )

    # --------------------------------------------------------
    # Align
    # --------------------------------------------------------

    n = min(
        len(fixed_c11),
        len(ghi_df),
    )

    if n == 0:
        raise ValueError(
            "No valid VCast data available."
        )

    fixed_c11 = fixed_c11.iloc[
        :n
    ].copy()

    ghi_df = ghi_df.iloc[
        :n
    ].copy()

    dates = pd.to_datetime(
        fixed_c11["Date"],
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError(
            "Invalid dates found in Fixed-C11."
        )

    actual = pd.to_numeric(
        fixed_c11["Actual"],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    blocks = pd.to_numeric(
        ghi_df["Block"],
        errors="coerce",
    ).to_numpy(dtype=float)

    ghi_matrix = np.column_stack([
        ghi_df[col].to_numpy(
            dtype=float
        )
        for col in GHI_COLS
    ])

    # --------------------------------------------------------
    # Tracking Sheet
    # --------------------------------------------------------

    tracking_df = pd.read_excel(
        buffer,
        sheet_name="Tracking",
        header=1,
    )

    tracking_df = tracking_df.iloc[
        :n
    ].copy()

    tracking_df.reset_index(
        drop=True,
        inplace=True,
    )

    return {
        "area_eff": area_eff,
        "fixed_weights": fixed_weights,
        "tracking_weights": tracking_weights,
        "standard_efficiency": standard_efficiency,
        "lat": lat,
        "month_to_tilt": month_to_tilt,
        "dates": dates,
        "actual": actual,
        "blocks": blocks,
        "ghi_matrix": ghi_matrix,
        "tracking_df": tracking_df,
        "n": n,
    }


try:

    data = load_vcast_data(
        uploaded_file.getvalue()
    )

except Exception as e:

    st.error(
        f"Unable to read VCast workbook: {e}"
    )

    st.stop()


# ============================================================
# UNPACK
# ============================================================

area_eff = data["area_eff"]

fixed_weights = data["fixed_weights"]

tracking_weights = data["tracking_weights"]

standard_efficiency = (
    data["standard_efficiency"]
)

lat = data["lat"]

month_to_tilt = (
    data["month_to_tilt"]
)

dates = data["dates"]

actual = data["actual"]

blocks = data["blocks"]

ghi_matrix = data["ghi_matrix"]

tracking_df = data["tracking_df"]

n = data["n"]


# ============================================================
# SOLAR CALCULATIONS
# ============================================================

first_date = pd.Timestamp(
    year=2025,
    month=1,
    day=1,
)

day_offset = (
    dates - first_date
).dt.days.to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# Declination
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Elevation
# ------------------------------------------------------------

elevation = (
    90
    - lat
    + declination
)


# ------------------------------------------------------------
# Tilt
# ------------------------------------------------------------

months = (
    dates.dt.month.to_numpy()
)

tilt = np.array([
    month_to_tilt.get(
        float(month),
        0,
    )
    for month in months
])


# ------------------------------------------------------------
# Sine calculations
# ------------------------------------------------------------

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
    sin_a,
)


# ------------------------------------------------------------
# Fixed POA
# ------------------------------------------------------------

fixed_poa = (
    ghi_matrix
    * sin_ab[:, None]
    / sin_a_safe[:, None]
)


# ============================================================
# ACTUAL / VALID DATA
# ============================================================

valid_mask = (
    np.isfinite(actual)
    &
    (actual != 0)
)

if not valid_mask.any():

    st.error(
        "Actual power contains no valid "
        "non-zero values."
    )

    st.stop()


actual_day = (
    actual[
        valid_mask
    ]
)

actual_peak = (
    np.max(actual_day)
)

actual_energy = (
    np.sum(actual_day)
)


if actual_peak <= 0:

    st.error(
        "Actual peak must be greater than zero."
    )

    st.stop()


# ============================================================
# FIXED MODEL
# ============================================================

def calculate_fixed(
    loss_percent,
):

    net_efficiency = (
        standard_efficiency
        - loss_percent
    )

    net_efficiency = np.maximum(
        net_efficiency,
        0,
    )

    efficiency_factor = np.divide(
        net_efficiency,
        standard_efficiency,
        out=np.zeros_like(
            standard_efficiency
        ),
        where=(
            standard_efficiency != 0
        ),
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

    forecast = (
        power_matrix.sum(axis=1)
    )

    return (
        forecast,
        power_matrix,
    )


# ============================================================
# FIXED LOSS OPTIMIZATION
# Minimum PEAK Error
# ============================================================

@st.cache_data(show_spinner=False)
def optimize_fixed_loss(
    actual_tuple,
    fixed_weights_tuple,
    standard_efficiency_tuple,
    fixed_poa_tuple,
):

    actual_arr = np.array(
        actual_tuple,
        dtype=float,
    )

    fixed_weights_arr = np.array(
        fixed_weights_tuple,
        dtype=float,
    )

    standard_eff_arr = np.array(
        standard_efficiency_tuple,
        dtype=float,
    )

    fixed_poa_arr = np.array(
        fixed_poa_tuple,
        dtype=float,
    )

    mask = (
        np.isfinite(actual_arr)
        &
        (actual_arr != 0)
    )

    act = actual_arr[mask]

    actual_peak_local = (
        np.max(act)
    )

    max_loss_local = (
        np.min(
            standard_eff_arr
        )
    )

    best_loss_local = 0.0
    best_peak_error = np.inf

    for loss in np.arange(
        0,
        max_loss_local + 0.0001,
        0.1,
    ):

        net_eff = (
            standard_eff_arr
            - loss
        )

        net_eff = np.maximum(
            net_eff,
            0,
        )

        factor = np.divide(
            net_eff,
            standard_eff_arr,
            out=np.zeros_like(
                standard_eff_arr
            ),
            where=(
                standard_eff_arr != 0
            ),
        )

        weights = (
            fixed_weights_arr
            * factor
        )

        power = (
            fixed_poa_arr
            * weights[None, :]
            / 1_000_000
        )

        forecast = (
            power.sum(axis=1)
        )

        pred = (
            forecast[mask]
        )

        peak_error = abs(
            actual_peak_local
            - pred.max()
        )

        if peak_error < best_peak_error:

            best_peak_error = (
                peak_error
            )

            best_loss_local = (
                float(loss)
            )

    return best_loss_local


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    loss_percent,
    dhi,
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

    dhi_matrix = (
        ghi_matrix
        * dhi
        / 100
    )

    # --------------------------------------------------------
    # DNI
    # --------------------------------------------------------

    dni = (
        ghi_matrix
        - dhi_matrix
    ) / cos_alpha[:, None]

    # --------------------------------------------------------
    # Tracking efficiency loss
    # --------------------------------------------------------

    net_efficiency = (
        standard_efficiency
        - loss_percent
    )

    net_efficiency = np.maximum(
        net_efficiency,
        0,
    )

    efficiency_factor = np.divide(
        net_efficiency,
        standard_efficiency,
        out=np.zeros_like(
            standard_efficiency
        ),
        where=(
            standard_efficiency != 0
        ),
    )

    adjusted_tracking_weights = (
        tracking_weights
        * efficiency_factor
    )

    # --------------------------------------------------------
    # Tracking power
    # --------------------------------------------------------

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
# IMPORTANT:
# First optimize ONLY Error % using minimum Peak Error.
# Then optimize DHI / GHI / Tracking parameters.
# ============================================================

@st.cache_data(show_spinner=False)
def optimize_tracking_loss(
    actual_tuple,
    ghi_matrix_tuple,
    tracking_weights_tuple,
    standard_efficiency_tuple,
    blocks_tuple,
):

    actual_arr = np.array(
        actual_tuple,
        dtype=float,
    )

    ghi_arr = np.array(
        ghi_matrix_tuple,
        dtype=float,
    )

    tracking_weights_arr = np.array(
        tracking_weights_tuple,
        dtype=float,
    )

    standard_eff_arr = np.array(
        standard_efficiency_tuple,
        dtype=float,
    )

    blocks_arr = np.array(
        blocks_tuple,
        dtype=float,
    )

    mask = (
        np.isfinite(actual_arr)
        &
        (actual_arr != 0)
    )

    act = actual_arr[mask]

    actual_peak_local = (
        np.max(act)
    )

    max_loss_local = (
        np.min(
            standard_eff_arr
        )
    )

    # --------------------------------------------------------
    # Use the reference/default tracking parameters
    # for the first-stage loss calculation.
    # --------------------------------------------------------

    default_dhi = 0.0
    default_start = 20
    default_end = 72
    default_max = 48
    default_east = 30
    default_west = 30

    best_loss_local = 0.0
    best_peak_error = np.inf

    for loss in np.arange(
        0,
        max_loss_local + 0.0001,
        0.1,
    ):

        if not (
            default_start
            < default_max
            < default_end
        ):
            continue

        d1 = (
            default_start
            - 1
            - default_max
        )

        d2 = (
            default_end
            + 1
            - default_max
        )

        m1 = 90 / d1
        m2 = 90 / d2

        zenith = np.where(

            blocks_arr <= default_max,

            np.minimum(
                89,
                m1
                * (
                    blocks_arr
                    - default_max
                ),
            ),

            np.minimum(
                89,
                m2
                * (
                    blocks_arr
                    - default_max
                ),
            ),
        )

        panel = np.where(

            blocks_arr < default_max,

            np.where(
                zenith
                < abs(default_east),

                zenith,

                abs(default_east),
            ),

            np.where(

                (
                    (blocks_arr > default_max)
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

        dhi_matrix = (
            ghi_arr
            * default_dhi
            / 100
        )

        dni = (
            ghi_arr
            - dhi_matrix
        ) / cos_alpha[:, None]

        net_eff = (
            standard_eff_arr
            - loss
        )

        net_eff = np.maximum(
            net_eff,
            0,
        )

        factor = np.divide(
            net_eff,
            standard_eff_arr,
            out=np.zeros_like(
                standard_eff_arr
            ),
            where=(
                standard_eff_arr != 0
            ),
        )

        weights = (
            tracking_weights_arr
            * factor
        )

        power = (
            dni
            * weights[None, :]
            / 1_000_000
        )

        forecast = (
            power.sum(axis=1)
        )

        pred = (
            forecast[mask]
        )

        if len(pred) == 0:
            continue

        peak_error = abs(
            actual_peak_local
            - pred.max()
        )

        if peak_error < best_peak_error:

            best_peak_error = (
                peak_error
            )

            best_loss_local = (
                float(loss)
            )

    return best_loss_local


# ============================================================
# AUTOMATIC OPTIMIZATION
# ============================================================

st.subheader(
    "VCast Parameters"
)


if st.button(
    "🚀 Optimize VCast",
    type="primary",
    use_container_width=True,
):

    # --------------------------------------------------------
    # FIXED LOSS
    # --------------------------------------------------------

    with st.spinner(
        "Calculating Fixed Efficiency Loss..."
    ):

        fixed_loss = optimize_fixed_loss(
            tuple(actual),
            tuple(fixed_weights),
            tuple(standard_efficiency),
            tuple(fixed_poa.flatten()),
        )

        # Because cache input is flattened, reconstruct
        # fixed POA shape is handled below.
        fixed_loss = float(
            fixed_loss
        )

    # --------------------------------------------------------
    # TRACKING LOSS
    # --------------------------------------------------------

    with st.spinner(
        "Calculating Tracking Efficiency Loss..."
    ):

        tracking_loss = (
            optimize_tracking_loss(
                tuple(actual),
                tuple(ghi_matrix.flatten()),
                tuple(tracking_weights),
                tuple(standard_efficiency),
                tuple(blocks),
            )
        )

        tracking_loss = float(
            tracking_loss
        )

    # --------------------------------------------------------
    # TRACKING PARAMETER OPTIMIZATION
    #
    # Error % is now fixed.
    # Optimize all remaining Tracking parameters.
    # --------------------------------------------------------

    @st.cache_data(show_spinner=False)
    def optimize_tracking_parameters(
        actual_tuple,
        ghi_matrix_tuple,
        tracking_weights_tuple,
        standard_efficiency_tuple,
        blocks_tuple,
        valid_mask_tuple,
        tracking_loss_value,
    ):

        actual_arr = np.array(
            actual_tuple,
            dtype=float,
        )

        ghi_arr = np.array(
            ghi_matrix_tuple,
            dtype=float,
        )

        tracking_weights_arr = np.array(
            tracking_weights_tuple,
            dtype=float,
        )

        standard_eff_arr = np.array(
            standard_efficiency_tuple,
            dtype=float,
        )

        blocks_arr = np.array(
            blocks_tuple,
            dtype=float,
        )

        mask_arr = np.array(
            valid_mask_tuple,
            dtype=bool,
        )

        act = (
            actual_arr[mask_arr]
        )

        peak = (
            np.max(act)
        )

        def objective_tracking(x):

            dhi = int(
                round(x[0])
            )

            start = int(
                round(x[1])
            )

            end = int(
                round(x[2])
            )

            max_block = int(
                round(x[3])
            )

            east = int(
                round(x[4])
            )

            west = int(
                round(x[5])
            )

            if not (
                start
                < max_block
                < end
            ):
                return 1e9

            d1 = (
                start
                - 1
                - max_block
            )

            d2 = (
                end
                + 1
                - max_block
            )

            if d1 == 0 or d2 == 0:
                return 1e9

            m1 = 90 / d1
            m2 = 90 / d2

            zenith = np.where(

                blocks_arr <= max_block,

                np.minimum(
                    89,
                    m1
                    * (
                        blocks_arr
                        - max_block
                    ),
                ),

                np.minimum(
                    89,
                    m2
                    * (
                        blocks_arr
                        - max_block
                    ),
                ),
            )

            panel = np.where(

                blocks_arr < max_block,

                np.where(
                    zenith < abs(east),

                    zenith,

                    abs(east),
                ),

                np.where(

                    (
                        (blocks_arr > max_block)
                        &
                        (zenith > west)
                    ),

                    west,

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

            dhi_matrix = (
                ghi_arr
                * dhi
                / 100
            )

            dni = (
                ghi_arr
                - dhi_matrix
            ) / cos_alpha[:, None]

            net_eff = (
                standard_eff_arr
                - tracking_loss_value
            )

            net_eff = np.maximum(
                net_eff,
                0,
            )

            factor = np.divide(
                net_eff,
                standard_eff_arr,
                out=np.zeros_like(
                    standard_eff_arr
                ),
                where=(
                    standard_eff_arr != 0
                ),
            )

            weights = (
                tracking_weights_arr
                * factor
            )

            power = (
                dni
                * weights[None, :]
                / 1_000_000
            )

            prediction = (
                power.sum(axis=1)
            )

            pred = (
                prediction[mask_arr]
            )

            if len(pred) == 0:
                return 1e9

            if not np.all(
                np.isfinite(pred)
            ):
                return 1e9

            block_error = (
                np.mean(
                    np.abs(
                        act - pred
                    )
                )
                / peak
            )

            peak_error = (
                abs(
                    peak
                    - pred.max()
                )
                / peak
            )

            energy_error = (
                abs(
                    act.sum()
                    - pred.sum()
                )
                / act.sum()
            )

            return (
                0.80 * block_error
                +
                0.10 * peak_error
                +
                0.10 * energy_error
            )

        result = differential_evolution(

            objective_tracking,

            bounds=[
                (0, 10),      # DHI
                (10, 30),     # Start
                (65, 80),     # End
                (47, 53),     # Max
                (10, 70),     # East
                (10, 70),     # West
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
            "ghi_start": int(best[1]),
            "ghi_end": int(best[2]),
            "ghi_max": int(best[3]),
            "east_limit": int(best[4]),
            "west_limit": int(best[5]),
        }

    with st.spinner(
        "Optimizing Tracking parameters..."
    ):

        tracking_params = (
            optimize_tracking_parameters(
                tuple(actual),
                tuple(ghi_matrix.flatten()),
                tuple(tracking_weights),
                tuple(standard_efficiency),
                tuple(blocks),
                tuple(valid_mask),
                tracking_loss,
            )
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Update session state BEFORE rerun.
    # After rerun, number_input widgets receive
    # the new optimized values.
    # --------------------------------------------------------

    st.session_state.vcast_fixed_params = {
        "fixed_error": fixed_loss,
    }

    st.session_state.vcast_tracking_params = {
        "tracking_error": tracking_loss,
        **tracking_params,
    }

    st.session_state.vcast_fixed_optimized = True
    st.session_state.vcast_tracking_optimized = True

    st.success(
        "VCast optimization completed successfully."
    )

    st.rerun()


# ============================================================
# FIXED PARAMETERS
# ============================================================

st.markdown("### Fixed")

fixed_params = (
    st.session_state.vcast_fixed_params
)


fixed_error = st.number_input(
    "Efficiency Loss / Error (%)",
    min_value=0.0,
    max_value=float(
        max(standard_efficiency)
    ),
    value=float(
        fixed_params["fixed_error"]
    ),
    step=0.1,
    format="%.2f",
    key="vcast_fixed_error_input",
)


st.session_state.vcast_fixed_params[
    "fixed_error"
] = float(
    fixed_error
)


# ============================================================
# FIXED FORECAST
# ============================================================

fixed_forecast, fixed_power_matrix = (
    calculate_fixed(
        fixed_error
    )
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

st.markdown("### Tracking")

tracking_params = (
    st.session_state.vcast_tracking_params
)


tcol1, tcol2 = st.columns(2)


with tcol1:

    tracking_error = st.number_input(
        "Efficiency Loss / Error (%)",
        min_value=0.0,
        max_value=float(
            max(standard_efficiency)
        ),
        value=float(
            tracking_params[
                "tracking_error"
            ]
        ),
        step=0.1,
        format="%.2f",
        key="vcast_tracking_error_input",
    )

    dhi = st.number_input(
        "DHI (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(
            tracking_params["dhi"]
        ),
        step=1.0,
        format="%.0f",
        key="vcast_dhi_input",
    )

    ghi_start = st.number_input(
        "GHI Starting Block",
        min_value=1,
        max_value=96,
        value=int(
            tracking_params[
                "ghi_start"
            ]
        ),
        step=1,
        key="vcast_ghi_start_input",
    )

    ghi_end = st.number_input(
        "GHI Ending Block",
        min_value=1,
        max_value=96,
        value=int(
            tracking_params[
                "ghi_end"
            ]
        ),
        step=1,
        key="vcast_ghi_end_input",
    )


with tcol2:

    ghi_max = st.number_input(
        "GHI Max Block",
        min_value=1,
        max_value=96,
        value=int(
            tracking_params[
                "ghi_max"
            ]
        ),
        step=1,
        key="vcast_ghi_max_input",
    )

    east_limit = st.number_input(
        "East Tracking Limit",
        min_value=0,
        max_value=90,
        value=int(
            tracking_params[
                "east_limit"
            ]
        ),
        step=1,
        key="vcast_east_input",
    )

    west_limit = st.number_input(
        "West Tracking Limit",
        min_value=0,
        max_value=90,
        value=int(
            tracking_params[
                "west_limit"
            ]
        ),
        step=1,
        key="vcast_west_input",
    )


# ============================================================
# SAVE CURRENT TRACKING VALUES
# ============================================================

st.session_state.vcast_tracking_params = {

    "tracking_error": float(
        tracking_error
    ),

    "dhi": float(
        dhi
    ),

    "ghi_start": int(
        ghi_start
    ),

    "ghi_end": int(
        ghi_end
    ),

    "ghi_max": int(
        ghi_max
    ),

    "east_limit": int(
        east_limit
    ),

    "west_limit": int(
        west_limit
    ),
}


# ============================================================
# VALIDATE TRACKING BLOCKS
# ============================================================

if not (
    ghi_start
    < ghi_max
    < ghi_end
):

    st.warning(
        "GHI Starting Block must be less than "
        "GHI Max Block, and GHI Max Block must "
        "be less than GHI Ending Block."
    )

    st.stop()


# ============================================================
# FINAL TRACKING CALCULATION
# ============================================================

tracking_result = calculate_tracking(

    tracking_error,

    dhi,

    ghi_start,

    ghi_end,

    ghi_max,

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


# ============================================================
# IMPORTANT TIME BLOCKS
# ============================================================

def make_time_blocks():

    start = datetime.strptime(
        "00:00",
        "%H:%M",
    )

    return [
        (
            f"{(start + timedelta(minutes=15 * i)).strftime('%H:%M')}"
            f" - "
            f"{(start + timedelta(minutes=15 * (i + 1))).strftime('%H:%M')}"
        )
        for i in range(96)
    ]


time_blocks = make_time_blocks()


lookup_blocks = [
    ghi_start,
    ghi_end,
    ghi_start + 3,
    ghi_end - 3,
    ghi_max,
]


lookup_names = [
    "Parabolic Power Generation Starting Block",
    "Parabolic Power Generation Ending Block",
    "Actual Generation Available Block (Lower Limit)",
    "Actual Generation Effective Block (Upper Limit)",
    "GHI Max Block",
]


lookup_df = pd.DataFrame({
    "Parameter": lookup_names,
    "Block": lookup_blocks,
})


lookup_df["Time Block"] = lookup_df[
    "Block"
].apply(
    lambda x: (
        time_blocks[int(x) - 1]
        if 1 <= int(x) <= 96
        else "—"
    )
)


# ============================================================
# IMPORTANT TIME BLOCKS EXPANDER
# ============================================================

with st.expander(
    "📅 Important Time Blocks"
):

    st.dataframe(
        lookup_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# GRAPH
# ============================================================

st.subheader(
    "Actual vs Forecast"
)


fig = go.Figure()


fig.add_trace(
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


fig.add_trace(
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


fig.add_trace(
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


# ------------------------------------------------------------
# 95th percentile
# ------------------------------------------------------------

percentile_95 = np.percentile(
    actual[valid_mask],
    95,
)


fig.add_trace(
    go.Scatter(
        x=blocks,
        y=np.full(
            len(blocks),
            percentile_95,
        ),
        name="95th Percentile",
        mode="lines",
        line=dict(
            color="#a855f7",
            width=2,
            dash="dash",
        ),
    )
)


fig.update_layout(

    height=550,

    template="streamlit",

    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        itemclick="toggle",
        itemdoubleclick="toggleothers",
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


# ============================================================
# LEGEND HOVER HIGHLIGHT
# ============================================================

fig.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)


# ============================================================
# DISPLAY
# ============================================================

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
    },
)


# ============================================================
# OPTIONAL: DATA VIEW
# ============================================================

with st.expander(
    "📊 Forecast Data"
):

    output_df = pd.DataFrame({

        "Block": blocks,

        "Actual": actual,

        "Fixed Forecast": fixed_forecast,

        "Tracking Forecast": tracking_forecast,

        "Zenith Angle": zenith,

        "Panel Angle": panel,

    })

    st.dataframe(
        output_df,
        use_container_width=True,
        hide_index=True,
    )
