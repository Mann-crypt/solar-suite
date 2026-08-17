# ============================================================
# LOSS CORRECTION VCAST
# VCast workbook format:
#   Fixed-C11
#   Area & Efficiency
#   Forecast Config
#   Config Tilt Angle
#   Result
#   Tracking (optional)
#
# UI flow follows the Loss Correction page:
#   1. Upload workbook
#   2. Show VCast input data
#   3. Select Fixed / Tracking
#   4. Explicitly RUN LOSS CORRECTION
#   5. Show metrics, tables, graph and final parameters
#
# Heavy optimization runs ONLY after the Run button is clicked.
# Ordinary Streamlit reruns do not automatically start optimization.
# ============================================================

import io
import hashlib
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

    <p style='text-align:center;color:gray;font-size:14px;'>
    Forecast Correction Platform
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    div[data-testid="stDataEditor"] {
        border-radius: 10px;
    }

    div[data-testid="stMetric"] {
        border-radius: 10px;
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

REQUIRED_VCAST_SHEETS = [
    "Fixed-C11",
    "Area & Efficiency",
    "Forecast Config",
    "Config Tilt Angle",
    "Result",
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "vcast_file_id": None,
    "vcast_run": False,
    "vcast_fixed_result": None,
    "vcast_tracking_result": None,
    "vcast_editor_version": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TITLE
# ============================================================

st.title(
    "Guruji ne kaha tha VCast Loss Correction kardo bhyii 🛐!!"
)

st.caption(
    "VCast Loss Correction | Fixed-C11 workbook format"
)


# ============================================================
# HELPERS
# ============================================================

def file_id(uploaded_file):
    data = uploaded_file.getvalue()

    return (
        uploaded_file.name,
        len(data),
        hashlib.md5(data).hexdigest(),
    )


def numeric_array(series):
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )


def create_time_blocks(count=96):
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
        for i in range(count)
    ]


def calculate_metrics(actual, forecast):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    if not mask.any():
        raise ValueError(
            "Actual power contains no valid non-zero values."
        )

    actual_day = actual[mask]
    forecast_day = forecast[mask]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    if actual_peak <= 0:
        raise ValueError(
            "Actual peak must be greater than zero."
        )

    if actual_energy == 0:
        raise ValueError(
            "Actual energy must be greater than zero."
        )

    block_error = (
        np.mean(
            np.abs(
                actual_day
                -
                forecast_day
            )
        )
        /
        actual_peak
    )

    peak_error = (
        abs(
            actual_peak
            -
            forecast_day.max()
        )
        /
        actual_peak
    )

    energy_error = (
        abs(
            actual_energy
            -
            forecast_day.sum()
        )
        /
        actual_energy
    )

    score = (
        0.80 * block_error
        +
        0.10 * peak_error
        +
        0.10 * energy_error
    )

    return {
        "mask": mask,
        "actual_peak": actual_peak,
        "forecast_peak": forecast_day.max(),
        "block_error": block_error,
        "peak_error": peak_error,
        "energy_error": energy_error,
        "score": score,
    }


# ============================================================
# CACHED WORKBOOK READER
# ============================================================

@st.cache_data(
    show_spinner=False,
)
def read_vcast_workbook(file_bytes):
    """
    Read the VCast workbook once.

    This intentionally keeps the VCast sheet structure separate
    from the normal Fixed / Cluster workbook format.
    """

    excel = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    sheet_names = excel.sheet_names

    missing = [
        sheet
        for sheet in REQUIRED_VCAST_SHEETS
        if sheet not in sheet_names
    ]

    if missing:
        raise ValueError(
            "Missing VCast sheet(s): "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    area_raw = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=None,
    )

    area_df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    area_df.columns = (
        area_df.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False,
        )
        .str.strip()
    )

    if "Standard PV Efficiency (%)" not in area_df.columns:
        raise ValueError(
            "Column 'Standard PV Efficiency (%)' "
            "was not found in Area & Efficiency."
        )

    # --------------------------------------------------------
    # VCAST EFFECTIVE AREAS
    #
    # Fixed     : P3:P7
    # Tracking  : P29:P33
    # --------------------------------------------------------

    fixed_weights = (
        pd.to_numeric(
            area_raw.iloc[
                2:7,
                15,
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    tracking_weights = (
        pd.to_numeric(
            area_raw.iloc[
                28:33,
                15,
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    if len(fixed_weights) != 5:
        raise ValueError(
            "Could not read 5 Fixed effective-area values "
            "from Area & Efficiency column P."
        )

    if len(tracking_weights) != 5:
        raise ValueError(
            "Could not read 5 Tracking effective-area values "
            "from Area & Efficiency column P."
        )

    standard_efficiency = numeric_array(
        area_df[
            "Standard PV Efficiency (%)"
        ]
    )[:5]

    if len(standard_efficiency) != 5:
        raise ValueError(
            "Could not read 5 Standard PV Efficiency values."
        )

    # --------------------------------------------------------
    # FORECAST CONFIG
    # --------------------------------------------------------

    config = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )

    if "Lat" not in config.columns:
        raise ValueError(
            "Column 'Lat' was not found in Forecast Config."
        )

    lat = float(
        pd.to_numeric(
            config.loc[
                0,
                "Lat",
            ],
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # CONFIG TILT ANGLE
    # --------------------------------------------------------

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

    if "Fixed" not in tilt_df.columns:
        raise ValueError(
            "Column 'Fixed' was not found in Config Tilt Angle."
        )

    tilt_df["Month_Num"] = pd.to_numeric(
        tilt_df["Month_Num"],
        errors="coerce",
    )

    tilt_df["Fixed"] = pd.to_numeric(
        tilt_df["Fixed"],
        errors="coerce",
    )

    tilt_lookup = (
        tilt_df
        .dropna(
            subset=[
                "Month_Num",
                "Fixed",
            ]
        )
        .set_index(
            "Month_Num"
        )["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # RESULT / GHI
    # --------------------------------------------------------

    ghi_df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Result",
        usecols=range(6),
    )

    ghi_df.columns = [
        "Block",
        *GHI_COLS,
    ]

    ghi_df["Block"] = pd.to_numeric(
        ghi_df["Block"],
        errors="coerce",
    )

    ghi_df = ghi_df[
        ghi_df["Block"].notna()
    ].copy()

    for col in GHI_COLS:
        ghi_df[col] = pd.to_numeric(
            ghi_df[col],
            errors="coerce",
        ).fillna(0)

    # --------------------------------------------------------
    # FIXED-C11
    #
    # Stop at first blank Date.
    # --------------------------------------------------------

    fixed_df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Fixed-C11",
        header=1,
    )

    fixed_df.columns = (
        fixed_df.columns
        .astype(str)
        .str.strip()
    )

    if "Date" not in fixed_df.columns:
        raise ValueError(
            "Column 'Date' was not found in Fixed-C11."
        )

    if "Actual" not in fixed_df.columns:
        raise ValueError(
            "Column 'Actual' was not found in Fixed-C11."
        )

    date_valid = fixed_df["Date"].notna()

    if not date_valid.any():
        raise ValueError(
            "No valid Date rows found in Fixed-C11."
        )

    first_blank = np.where(
        ~date_valid.to_numpy()
    )[0]

    if len(first_blank) > 0:
        fixed_df = fixed_df.iloc[
            :first_blank[0]
        ].copy()
    else:
        fixed_df = fixed_df.loc[
            date_valid
        ].copy()

    fixed_df.reset_index(
        drop=True,
        inplace=True,
    )

    fixed_df["Date"] = pd.to_datetime(
        fixed_df["Date"],
        errors="coerce",
    )

    if fixed_df["Date"].isna().any():
        raise ValueError(
            "Invalid dates found in Fixed-C11."
        )

    # --------------------------------------------------------
    # ALIGN DATA
    # --------------------------------------------------------

    n = min(
        len(fixed_df),
        len(ghi_df),
    )

    if n == 0:
        raise ValueError(
            "No aligned VCast rows are available."
        )

    fixed_df = fixed_df.iloc[
        :n
    ].copy()

    ghi_df = ghi_df.iloc[
        :n
    ].copy()

    actual = numeric_array(
        fixed_df["Actual"]
    )[:n]

    blocks = ghi_df[
        "Block"
    ].to_numpy(
        dtype=float
    )[:n]

    ghi_matrix = ghi_df[
        GHI_COLS
    ].to_numpy(
        dtype=float
    )[:n]

    dates = fixed_df[
        "Date"
    ]

    # --------------------------------------------------------
    # SOLAR GEOMETRY
    # --------------------------------------------------------

    first_date = pd.Timestamp(
        "2025-01-01"
    )

    day_offset = (
        dates
        -
        first_date
    ).dt.days.to_numpy(
        dtype=float
    )

    declination = (
        23.45
        *
        np.sin(
            np.radians(
                360
                *
                (
                    284
                    +
                    day_offset
                    +
                    1
                )
                /
                365
            )
        )
    )

    elevation = (
        90
        -
        lat
        +
        declination
    )

    months = (
        dates
        .dt
        .month
        .to_numpy()
    )

    tilt = np.array(
        [
            tilt_lookup.get(
                float(month),
                0,
            )
            for month in months
        ]
    )

    sin_a = np.sin(
        np.radians(
            elevation
        )
    )

    sin_ab = np.sin(
        np.radians(
            elevation
            +
            tilt
        )
    )

    sin_a_safe = np.where(
        np.abs(sin_a) < 1e-8,
        1e-8,
        sin_a,
    )

    fixed_geometry_factor = (
        sin_ab
        /
        sin_a_safe
    )

    fixed_poa = (
        ghi_matrix
        *
        fixed_geometry_factor[:, None]
    )

    # --------------------------------------------------------
    # TRACKING SHEET
    # --------------------------------------------------------

    tracking_available = (
        "Tracking" in sheet_names
    )

    tracking_df = None

    if tracking_available:
        tracking_df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name="Tracking",
            header=1,
        )

        tracking_df.columns = (
            tracking_df.columns
            .astype(str)
            .str.strip()
        )

        tracking_df = tracking_df.iloc[
            :n
        ].copy()

        tracking_df.reset_index(
            drop=True,
            inplace=True,
        )

    return {
        "sheet_names": sheet_names,
        "area_df": area_df,
        "fixed_weights": fixed_weights,
        "tracking_weights": tracking_weights,
        "standard_efficiency": standard_efficiency,
        "lat": lat,
        "tilt_lookup": tilt_lookup,
        "fixed_df": fixed_df,
        "ghi_df": ghi_df,
        "actual": actual,
        "blocks": blocks,
        "ghi_matrix": ghi_matrix,
        "fixed_poa": fixed_poa,
        "tracking_available": tracking_available,
        "tracking_df": tracking_df,
    }


# ============================================================
# FIXED OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
)
def optimize_fixed(
    standard_efficiency_tuple,
    fixed_weights_tuple,
    fixed_poa_tuple,
    actual_tuple,
):
    std_eff = np.asarray(
        standard_efficiency_tuple,
        dtype=float,
    )

    fixed_weights = np.asarray(
        fixed_weights_tuple,
        dtype=float,
    )

    fixed_poa = np.asarray(
        fixed_poa_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    if not mask.any():
        raise ValueError(
            "No valid actual values available for Fixed optimization."
        )

    actual_day = actual[mask]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    results = []

    max_loss = float(
        np.min(
            std_eff
        )
    )

    for loss in np.arange(
        0,
        max_loss + 0.0001,
        0.1,
    ):
        net_efficiency = np.maximum(
            std_eff - loss,
            0,
        )

        efficiency_factor = np.divide(
            net_efficiency,
            std_eff,
            out=np.zeros_like(
                net_efficiency
            ),
            where=(
                std_eff != 0
            ),
        )

        final_weights = (
            fixed_weights
            *
            efficiency_factor
        )

        power_matrix = (
            fixed_poa
            *
            final_weights[None, :]
            /
            1_000_000
        )

        forecast = (
            power_matrix.sum(
                axis=1
            )
        )

        forecast_day = forecast[
            mask
        ]

        predicted_peak = (
            forecast_day.max()
        )

        peak_error = abs(
            actual_peak
            -
            predicted_peak
        )

        peak_error_percent = (
            peak_error
            /
            actual_peak
            *
            100
        )

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    -
                    forecast_day
                )
            )
            /
            actual_peak
        )

        predicted_energy = (
            forecast_day.sum()
        )

        energy_error = (
            abs(
                actual_energy
                -
                predicted_energy
            )
            /
            actual_energy
        )

        overall_score = (
            0.80 * block_error
            +
            0.10 * (
                peak_error
                /
                actual_peak
            )
            +
            0.10 * energy_error
        )

        results.append(
            {
                "Error %": loss,
                "Actual Peak": actual_peak,
                "Predicted Peak": predicted_peak,
                "Peak Error": peak_error,
                "Peak Error (%)": peak_error_percent,
                "Block Error": block_error,
                "Energy Error": energy_error,
                "Overall Score": overall_score,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        raise ValueError(
            "Fixed optimization produced no results."
        )

    # IMPORTANT:
    # VCast validated logic selects minimum Peak Error.
    best_row = results_df.loc[
        results_df[
            "Peak Error"
        ].idxmin()
    ]

    best_loss = float(
        best_row[
            "Error %"
        ]
    )

    final_net_efficiency = np.maximum(
        std_eff - best_loss,
        0,
    )

    final_efficiency_factor = np.divide(
        final_net_efficiency,
        std_eff,
        out=np.zeros_like(
            std_eff
        ),
        where=(
            std_eff != 0
        ),
    )

    final_weights = (
        fixed_weights
        *
        final_efficiency_factor
    )

    final_power_matrix = (
        fixed_poa
        *
        final_weights[None, :]
        /
        1_000_000
    )

    final_forecast = (
        final_power_matrix.sum(
            axis=1
        )
    )

    return (
        best_loss,
        final_forecast,
        final_power_matrix,
        final_net_efficiency,
        results_df,
    )


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    blocks,
    ghi_matrix,
    tracking_weights,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
):
    if not (
        start_block
        <
        max_block
        <
        end_block
    ):
        return None

    denominator_1 = (
        start_block
        -
        1
        -
        max_block
    )

    denominator_2 = (
        end_block
        +
        1
        -
        max_block
    )

    if (
        denominator_1 == 0
        or
        denominator_2 == 0
    ):
        return None

    m1 = (
        90
        /
        denominator_1
    )

    m2 = (
        90
        /
        denominator_2
    )

    zenith = np.where(
        blocks <= max_block,
        np.minimum(
            89,
            m1
            *
            (
                blocks
                -
                max_block
            ),
        ),
        np.minimum(
            89,
            m2
            *
            (
                blocks
                -
                max_block
            ),
        ),
    )

    panel = np.where(
        blocks < max_block,
        np.where(
            zenith
            <
            abs(
                east_limit
            ),
            zenith,
            abs(
                east_limit
            ),
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
        np.radians(
            panel
        )
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None,
    )

    dhi = (
        ghi_matrix
        *
        DHI
        /
        100
    )

    dni = (
        ghi_matrix
        -
        dhi
    ) / cos_alpha[:, None]

    tracking_power_matrix = (
        dni
        *
        tracking_weights[None, :]
        /
        1_000_000
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
        dni,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
)
def optimize_tracking(
    blocks_tuple,
    ghi_matrix_tuple,
    tracking_weights_tuple,
    actual_tuple,
):
    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    ghi_matrix = np.asarray(
        ghi_matrix_tuple,
        dtype=float,
    )

    tracking_weights = np.asarray(
        tracking_weights_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    if not mask.any():
        raise ValueError(
            "No valid actual values available for Tracking optimization."
        )

    actual_day = actual[
        mask
    ]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    def objective(x):
        DHI = int(
            round(
                x[0]
            )
        )

        start_block = int(
            round(
                x[1]
            )
        )

        end_block = int(
            round(
                x[2]
            )
        )

        max_block = int(
            round(
                x[3]
            )
        )

        east_limit = int(
            round(
                x[4]
            )
        )

        west_limit = int(
            round(
                x[5]
            )
        )

        result = calculate_tracking(
            blocks,
            ghi_matrix,
            tracking_weights,
            DHI,
            start_block,
            end_block,
            max_block,
            east_limit,
            west_limit,
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

        prediction_day = prediction[
            mask
        ]

        if len(prediction_day) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    -
                    prediction_day
                )
            )
            /
            actual_peak
        )

        peak_error = (
            abs(
                actual_peak
                -
                prediction_day.max()
            )
            /
            actual_peak
        )

        energy_error = (
            abs(
                actual_energy
                -
                prediction_day.sum()
            )
            /
            actual_energy
        )

        return (
            0.80 * block_error
            +
            0.10 * peak_error
            +
            0.10 * energy_error
        )

    bounds = [
        (0, 10),      # DHI
        (10, 30),     # GHI Start
        (65, 80),     # GHI End
        (47, 53),     # GHI Max
        (10, 70),     # East Limit
        (10, 70),     # West Limit
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
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

    rounded_score = objective(
        best
    )

    return (
        tuple(
            best.tolist()
        ),
        float(
            result.fun
        ),
        float(
            rounded_score
        ),
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload VCast Excel Workbook",
    type=["xlsx", "xls"],
    key="vcast_uploader",
)


if uploaded_file is None:
    st.info(
        "👆 Upload the VCast workbook to begin."
    )
    st.stop()


file_bytes = uploaded_file.getvalue()

current_file_id = file_id(
    uploaded_file
)


# ------------------------------------------------------------
# Reset only for a genuinely new file.
# No automatic st.rerun().
# ------------------------------------------------------------

if (
    st.session_state.vcast_file_id
    !=
    current_file_id
):
    st.session_state.vcast_file_id = (
        current_file_id
    )

    st.session_state.vcast_run = False

    st.session_state.vcast_fixed_result = None

    st.session_state.vcast_tracking_result = None

    st.session_state.vcast_editor_version += 1


# ============================================================
# READ WORKBOOK
# ============================================================

try:
    with st.spinner(
        "Reading VCast workbook..."
    ):
        workbook = read_vcast_workbook(
            file_bytes
        )

except Exception as e:
    st.error(
        f"Unable to read VCast workbook: {e}"
    )
    st.stop()


# ============================================================
# WORKBOOK INFORMATION
# ============================================================

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Workbook Type",
        "VCast",
    )

with info2:
    st.metric(
        "Fixed Sheet",
        "Fixed-C11",
    )

with info3:
    st.metric(
        "Tracking Available",
        "Yes"
        if workbook[
            "tracking_available"
        ]
        else "No",
    )


st.divider()


# ============================================================
# INPUT DATA
#
# Keep VCast sheet format:
# Date + Actual + GHI C11..C15
# ============================================================

st.subheader(
    "📥 Input Data"
)

input_df = pd.DataFrame(
    {
        "Date": workbook[
            "fixed_df"
        ]["Date"],
        "Actual": workbook[
            "actual"
        ],
    }
)

for col in GHI_COLS:
    input_df[col] = workbook[
        "ghi_df"
    ][col].to_numpy()


edited_df = st.data_editor(
    input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    height=380,
    key=(
        "vcast_input_editor_"
        f"{st.session_state.vcast_editor_version}"
    ),
    disabled=["Date"],
    column_config={
        "Date": st.column_config.DateColumn(
            "Date",
            format="DD-MM-YYYY",
        ),
        "Actual": st.column_config.NumberColumn(
            "Actual",
            format="%.4f",
        ),
        "GHI C11": st.column_config.NumberColumn(
            "GHI C11",
            format="%.4f",
        ),
        "GHI C12": st.column_config.NumberColumn(
            "GHI C12",
            format="%.4f",
        ),
        "GHI C13": st.column_config.NumberColumn(
            "GHI C13",
            format="%.4f",
        ),
        "GHI C14": st.column_config.NumberColumn(
            "GHI C14",
            format="%.4f",
        ),
        "GHI C15": st.column_config.NumberColumn(
            "GHI C15",
            format="%.4f",
        ),
    },
)


# ============================================================
# CLEAN USER INPUT
# ============================================================

edited_df["Actual"] = pd.to_numeric(
    edited_df["Actual"],
    errors="coerce",
).fillna(0)

for col in GHI_COLS:
    edited_df[col] = pd.to_numeric(
        edited_df[col],
        errors="coerce",
    ).fillna(0)


actual = edited_df[
    "Actual"
].to_numpy(
    dtype=float
)

ghi_matrix = edited_df[
    GHI_COLS
].to_numpy(
    dtype=float
)

blocks = workbook[
    "blocks"
][:len(edited_df)]


# ============================================================
# REBUILD FIXED POA USING EDITED GHI
#
# Solar geometry comes from the workbook dates.
# User-edited GHI is used for the actual calculation.
# ============================================================

original_ghi = workbook[
    "ghi_matrix"
][:len(edited_df)]

geometry_factor = np.divide(
    workbook[
        "fixed_poa"
    ][:len(edited_df)],
    original_ghi,
    out=np.zeros_like(
        original_ghi
    ),
    where=(
        np.abs(
            original_ghi
        )
        >
        1e-12
    ),
)

fixed_poa = (
    ghi_matrix
    *
    geometry_factor
)


# ============================================================
# CORRECTION TYPE
# ============================================================

st.subheader(
    "🌞 Correction Type"
)

plant_type = st.pills(
    "Select Correction Type",
    [
        "🏗️ Fixed",
        "🔄 Tracking",
    ],
    default="🏗️ Fixed",
    key="vcast_plant_type",
)


# ============================================================
# RUN BUTTON
# ============================================================

if st.button(
    "🚀 RUN LOSS CORRECTION",
    type="primary",
    use_container_width=True,
    key="vcast_run_loss_correction",
):
    st.session_state.vcast_run = True

    # Results are reset only when the user explicitly
    # starts a new calculation.
    st.session_state.vcast_fixed_result = None

    st.session_state.vcast_tracking_result = None


if not st.session_state.vcast_run:
    st.info(
        "Edit the VCast input data, select Fixed or Tracking, "
        "then click **RUN LOSS CORRECTION**."
    )
    st.stop()


# ============================================================
# FIXED MODEL
# ============================================================

if plant_type == "🏗️ Fixed":

    st.header(
        "🏗️ Fixed Loss Correction"
    )

    try:
        with st.spinner(
            "Optimizing Fixed efficiency loss..."
        ):
            fixed_result = optimize_fixed(
                tuple(
                    workbook[
                        "standard_efficiency"
                    ]
                ),
                tuple(
                    workbook[
                        "fixed_weights"
                    ]
                ),
                tuple(
                    fixed_poa.tolist()
                ),
                tuple(
                    actual.tolist()
                ),
            )

    except Exception as e:
        st.error(
            f"Fixed loss correction failed: {e}"
        )
        st.stop()


    (
        best_loss,
        fixed_forecast,
        fixed_power_matrix,
        final_net_efficiency,
        fixed_results_df,
    ) = fixed_result


    # Store result
    st.session_state.vcast_fixed_result = (
        best_loss,
        fixed_forecast,
        fixed_results_df,
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    fixed_metrics = calculate_metrics(
        actual,
        fixed_forecast,
    )

    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        st.metric(
            "Efficiency Loss",
            f"{best_loss:.2f}%",
        )

    with fc2:
        st.metric(
            "Actual Peak",
            f"{fixed_metrics['actual_peak']:.4f}",
        )

    with fc3:
        st.metric(
            "Fixed Peak",
            f"{fixed_metrics['forecast_peak']:.4f}",
        )

    with fc4:
        st.metric(
            "Peak Error",
            f"{fixed_metrics['peak_error'] * 100:.3f}%",
        )


    # --------------------------------------------------------
    # FIXED RESULTS
    # --------------------------------------------------------

    with st.expander(
        "📊 Fixed Optimization Results"
    ):
        st.dataframe(
            fixed_results_df,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # FINAL PARAMETERS
    # --------------------------------------------------------

    with st.expander(
        "📋 Final Optimized Parameters"
    ):
        final_parameters = pd.DataFrame(
            {
                "Parameter": [
                    "Workbook Type",
                    "Fixed Sheet",
                    "Fixed Efficiency Loss (%)",
                    "Actual Peak",
                    "Fixed Predicted Peak",
                    "Fixed Peak Error (%)",
                    "Fixed Block Error",
                    "Fixed Energy Error",
                    "Fixed Overall Score",
                ],
                "Value": [
                    "VCast",
                    "Fixed-C11",
                    best_loss,
                    fixed_metrics[
                        "actual_peak"
                    ],
                    fixed_metrics[
                        "forecast_peak"
                    ],
                    fixed_metrics[
                        "peak_error"
                    ] * 100,
                    fixed_metrics[
                        "block_error"
                    ],
                    fixed_metrics[
                        "energy_error"
                    ],
                    fixed_metrics[
                        "score"
                    ],
                ],
            }
        )

        st.dataframe(
            final_parameters,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    st.subheader(
        "Actual vs Fixed Forecast"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=blocks,
            y=actual,
            name="Actual",
            mode="lines",
            line=dict(
                width=3,
            ),
            hovertemplate=(
                "Block: %{x}"
                "<br>Actual: %{y:.4f}"
                "<extra></extra>"
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
                width=3,
            ),
            hovertemplate=(
                "Block: %{x}"
                "<br>Fixed: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=550,
        template="streamlit",
        hovermode="x unified",
        xaxis=dict(
            title="Block",
            dtick=4,
        ),
        yaxis=dict(
            title="Power (MW)",
        ),
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
            t=70,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


# ============================================================
# TRACKING MODEL
# ============================================================

else:

    if not workbook[
        "tracking_available"
    ]:
        st.error(
            "Tracking sheet was not found in this VCast workbook."
        )
        st.stop()


    st.header(
        "🔄 Tracking Loss Correction"
    )


    # --------------------------------------------------------
    # OPTIMIZE
    # --------------------------------------------------------

    try:
        with st.spinner(
            "Optimizing Tracking parameters..."
        ):
            tracking_result = optimize_tracking(
                tuple(
                    blocks.tolist()
                ),
                tuple(
                    ghi_matrix.tolist()
                ),
                tuple(
                    workbook[
                        "tracking_weights"
                    ].tolist()
                ),
                tuple(
                    actual.tolist()
                ),
            )

    except Exception as e:
        st.error(
            f"Tracking optimization failed: {e}"
        )
        st.stop()


    (
        best_tracking,
        optimizer_score,
        rounded_score,
    ) = tracking_result


    (
        DHI,
        GHI_Starting_Block,
        GHI_Ending_Block,
        GHI_Max_Block,
        Tracking_angle_lim_E,
        Tracking_angle_lim_W,
    ) = best_tracking


    # --------------------------------------------------------
    # FINAL TRACKING CALCULATION
    # --------------------------------------------------------

    tracking_output = calculate_tracking(
        blocks,
        ghi_matrix,
        workbook[
            "tracking_weights"
        ],
        DHI,
        GHI_Starting_Block,
        GHI_Ending_Block,
        GHI_Max_Block,
        Tracking_angle_lim_E,
        Tracking_angle_lim_W,
    )

    if tracking_output is None:
        st.error(
            "Unable to calculate final Tracking forecast."
        )
        st.stop()


    (
        tracking_forecast,
        tracking_power_matrix,
        zenith,
        panel,
        dni,
    ) = tracking_output


    tracking_metrics = calculate_metrics(
        actual,
        tracking_forecast,
    )


    st.session_state.vcast_tracking_result = (
        best_tracking,
        tracking_forecast,
        tracking_metrics,
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    tc1, tc2, tc3, tc4 = st.columns(4)

    with tc1:
        st.metric(
            "DHI",
            f"{DHI}%",
        )

    with tc2:
        st.metric(
            "GHI Max Block",
            int(
                GHI_Max_Block
            ),
        )

    with tc3:
        st.metric(
            "Tracking Peak Error",
            f"{tracking_metrics['peak_error'] * 100:.3f}%",
        )

    with tc4:
        st.metric(
            "Overall Score",
            f"{tracking_metrics['score']:.5f}",
        )


    # --------------------------------------------------------
    # OPTIMIZED PARAMETERS
    # --------------------------------------------------------

    with st.expander(
        "⚙️ Optimized Tracking Parameters",
        expanded=True,
    ):

        tracking_parameters = pd.DataFrame(
            {
                "Parameter": [
                    "DHI (%)",
                    "GHI Starting Block",
                    "GHI Ending Block",
                    "GHI Max Block",
                    "East Tracking Limit",
                    "West Tracking Limit",
                    "Optimizer Score",
                    "Rounded Score",
                ],
                "Value": [
                    DHI,
                    GHI_Starting_Block,
                    GHI_Ending_Block,
                    GHI_Max_Block,
                    Tracking_angle_lim_E,
                    Tracking_angle_lim_W,
                    optimizer_score,
                    rounded_score,
                ],
            }
        )

        st.dataframe(
            tracking_parameters,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # IMPORTANT TIME BLOCKS
    # --------------------------------------------------------

    time_blocks = create_time_blocks(
        96
    )

    tracking_lookup = pd.DataFrame(
        {
            "Parameter": [
                "GHI Starting Block",
                "GHI Ending Block",
                "GHI Maximum Block",
            ],
            "Block": [
                GHI_Starting_Block,
                GHI_Ending_Block,
                GHI_Max_Block,
            ],
        }
    )

    tracking_lookup[
        "Time Block"
    ] = tracking_lookup[
        "Block"
    ].apply(
        lambda x:
        time_blocks[
            int(x) - 1
        ]
        if 1 <= int(x) <= 96
        else "—"
    )

    with st.expander(
        "📅 Important Time Blocks"
    ):
        st.dataframe(
            tracking_lookup,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # TRACKING RESULTS
    # --------------------------------------------------------

    with st.expander(
        "📊 Tracking Results"
    ):

        tracking_summary = pd.DataFrame(
            {
                "Metric": [
                    "Actual Peak",
                    "Tracking Predicted Peak",
                    "Peak Error (%)",
                    "Block Error",
                    "Energy Error",
                    "Overall Score",
                ],
                "Value": [
                    tracking_metrics[
                        "actual_peak"
                    ],
                    tracking_metrics[
                        "forecast_peak"
                    ],
                    tracking_metrics[
                        "peak_error"
                    ] * 100,
                    tracking_metrics[
                        "block_error"
                    ],
                    tracking_metrics[
                        "energy_error"
                    ],
                    tracking_metrics[
                        "score"
                    ],
                ],
            }
        )

        st.dataframe(
            tracking_summary,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    st.subheader(
        "Actual vs Tracking Forecast"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=blocks,
            y=actual,
            name="Actual",
            mode="lines",
            line=dict(
                width=3,
            ),
            hovertemplate=(
                "Block: %{x}"
                "<br>Actual: %{y:.4f}"
                "<extra></extra>"
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
                width=3,
            ),
            hovertemplate=(
                "Block: %{x}"
                "<br>Tracking: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=550,
        template="streamlit",
        hovermode="x unified",
        xaxis=dict(
            title="Block",
            dtick=4,
        ),
        yaxis=dict(
            title="Power (MW)",
        ),
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
            t=70,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


    # --------------------------------------------------------
    # TRACKING OUTPUT DATA
    # --------------------------------------------------------

    with st.expander(
        "📄 Tracking Calculation Output"
    ):

        tracking_output_df = workbook[
            "tracking_df"
        ].copy()

        tracking_output_df[
            "Zenith Angle"
        ] = zenith

        tracking_output_df[
            "Panel Angle"
        ] = panel

        for i, cluster in enumerate(
            CLUSTERS
        ):
            tracking_output_df[
                f"{cluster}_Tracking Power=I*Ƞ*A"
            ] = tracking_power_matrix[
                :,
                i,
            ]

        tracking_output_df[
            "Tracking Power=I*Ƞ*A"
        ] = tracking_forecast

        st.dataframe(
            tracking_output_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# NOTE
# ============================================================
#
# This page intentionally keeps VCast calculation logic separate:
#
# Fixed:
#   - Fixed-C11 Actual
#   - Result GHI C11..C15
#   - Fixed effective areas P3:P7
#   - Efficiency-loss sweep
#   - Best loss = minimum Peak Error
#
# Tracking:
#   - Result GHI C11..C15
#   - Tracking effective areas P29:P33
#   - DHI
#   - GHI Start / End / Max blocks
#   - East / West tracking limits
#   - Differential Evolution
#   - Block / Peak / Energy errors
#
# Heavy optimization is cached and only called after RUN.
# ============================================================
