# ============================================================
# VCAST LOSS CORRECTION - STREAMLIT PAGE
# ============================================================
#
# VCAST workbook identification:
#
#   Fixed-C11 sheet  -> VCast
#
# Plant modes:
#
#   Fixed
#       -> Error % optimized using minimum Peak Error
#       -> Net Efficiency
#       -> Effective Area
#       -> Cluster Fixed Power
#
#   Tracking
#       -> Error % optimized first using minimum Peak Error
#       -> DHI
#       -> GHI Starting Block
#       -> GHI Ending Block
#       -> GHI Max Block
#       -> East Tracking Limit
#       -> West Tracking Limit
#
# All automatically calculated parameters are editable.
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VCast Loss Correction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------ */
    /* GLOBAL */
    /* ------------------------------------------------------ */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        letter-spacing: -0.3px;
    }

    /* ------------------------------------------------------ */
    /* HEADER */
    /* ------------------------------------------------------ */

    .page-header {
        padding: 1.1rem 1.3rem;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            #f7fbff,
            #eef6ff
        );
        border: 1px solid #d9e8f7;
        margin-bottom: 1.2rem;
    }

    .page-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .page-subtitle {
        color: #667085;
        font-size: 0.95rem;
    }

    /* ------------------------------------------------------ */
    /* CARDS */
    /* ------------------------------------------------------ */

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        min-height: 105px;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #667085;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #111827;
    }

    .metric-sub {
        font-size: 0.75rem;
        color: #667085;
        margin-top: 0.2rem;
    }

    /* ------------------------------------------------------ */
    /* SECTION */
    /* ------------------------------------------------------ */

    .section-title {
        font-size: 1.15rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    /* ------------------------------------------------------ */
    /* STATUS */
    /* ------------------------------------------------------ */

    .status-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 1px solid #d9e8f7;
        background: #f7fbff;
        color: #344054;
        margin: 0.6rem 0 1rem 0;
    }

    /* ------------------------------------------------------ */
    /* INFO */
    /* ------------------------------------------------------ */

    .info-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e4e7ec;
        color: #475467;
        font-size: 0.88rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="page-header">

        <div class="page-title">
            ☀️ VCast Loss Correction
        </div>

        <div class="page-subtitle">
            Fixed and Tracking plant power correction using
            efficiency loss, GHI and tracking parameter optimization.
        </div>

    </div>
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


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {

    "file_bytes": None,

    "file_name": None,

    "workbook_type": None,

    "auto_fixed_error": 0.0,

    "fixed_error": 0.0,

    "tracking_error": 0.0,

    "tracking_dhi": 5,

    "tracking_start": 20,

    "tracking_end": 72,

    "tracking_max": 48,

    "tracking_east": 70,

    "tracking_west": 70,

    "tracking_auto_done": False,

    "fixed_auto_done": False,

}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Workbook</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload VCast Excel workbook",
    type=["xlsx", "xls"],
    help=(
        "VCast workbook must contain the Fixed-C11 sheet."
    )
)


if uploaded_file is not None:

    st.session_state["file_bytes"] = (
        uploaded_file.getvalue()
    )

    st.session_state["file_name"] = (
        uploaded_file.name
    )


if st.session_state["file_bytes"] is None:

    st.info(
        "Upload the VCast workbook to start the calculation."
    )

    st.stop()


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    workbook = pd.ExcelFile(
        io.BytesIO(
            st.session_state["file_bytes"]
        )
    )

    sheet_names = workbook.sheet_names

except Exception as exc:

    st.error(
        f"Unable to read workbook: {exc}"
    )

    st.stop()


# ============================================================
# VCAST IDENTIFICATION
# ============================================================

if "Fixed-C11" in sheet_names:

    workbook_type = "VCast"

else:

    workbook_type = "Unknown"


st.session_state["workbook_type"] = workbook_type


if workbook_type != "VCast":

    st.error(
        "This page is designed for VCast workbooks. "
        "The required 'Fixed-C11' sheet was not found."
    )

    st.write("Available sheets:")

    st.write(sheet_names)

    st.stop()


st.markdown(
    f"""
    <div class="status-box">
        <b>Workbook:</b> {st.session_state["file_name"]}
        &nbsp;&nbsp; | &nbsp;&nbsp;
        <b>Detected type:</b> VCast
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">Plant Type</div>',
    unsafe_allow_html=True
)

plant_type = st.segmented_control(
    "Plant Type",
    options=[
        "Fixed",
        "Tracking"
    ],
    default="Fixed",
    key="plant_type_selector"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(dataframe):

    dataframe = dataframe.copy()

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False
        )
        .str.strip()
    )

    return dataframe


def first_blank_position(
    dataframe,
    column
):

    valid = dataframe[column].notna()

    if not valid.any():

        return len(dataframe)

    blank_indices = np.where(
        ~valid.to_numpy()
    )[0]

    if len(blank_indices) == 0:

        return len(dataframe)

    return int(
        blank_indices[0]
    )


def safe_numeric(
    series,
    default=0.0
):

    result = pd.to_numeric(
        series,
        errors="coerce"
    )

    return result.fillna(
        default
    )


def metric_card(
    label,
    value,
    sub=""
):

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-sub">
                {sub}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def calculate_metrics(
    actual,
    forecast
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    forecast = np.asarray(
        forecast,
        dtype=float
    )

    valid = (
        np.isfinite(actual)
        &
        np.isfinite(forecast)
        &
        (actual != 0)
    )

    if not valid.any():

        return {
            "actual_peak": np.nan,
            "forecast_peak": np.nan,
            "peak_error": np.nan,
            "peak_error_pct": np.nan,
            "block_error": np.nan,
            "energy_error": np.nan,
            "score": np.nan
        }

    actual_day = actual[valid]

    forecast_day = forecast[valid]

    actual_peak = np.max(
        actual_day
    )

    forecast_peak = np.max(
        forecast_day
    )

    actual_energy = np.sum(
        actual_day
    )

    forecast_energy = np.sum(
        forecast_day
    )

    peak_error = abs(
        actual_peak
        -
        forecast_peak
    )

    peak_error_pct = (
        peak_error
        /
        actual_peak
        *
        100
        if actual_peak != 0
        else np.nan
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

    energy_error = (
        abs(
            actual_energy
            -
            forecast_energy
        )
        /
        actual_energy
        if actual_energy != 0
        else np.nan
    )

    score = (
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

    return {

        "actual_peak": actual_peak,

        "forecast_peak": forecast_peak,

        "peak_error": peak_error,

        "peak_error_pct": peak_error_pct,

        "block_error": block_error,

        "energy_error": energy_error,

        "score": score
    }


# ============================================================
# LOAD COMMON DATA
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_common_data(
    file_bytes
):

    buffer = io.BytesIO(
        file_bytes
    )

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    df = pd.read_excel(
        buffer,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12)
    )

    df = clean_columns(
        df
    )

    if "S.No." not in df.columns:

        raise ValueError(
            "S.No. column not found in Area & Efficiency."
        )

    pos = first_blank_position(
        df,
        "S.No."
    )

    df = df.iloc[
        :pos
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
        "Error %"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # TOTAL AREA
    # --------------------------------------------------------

    df["Total area (m2)"] = (
        safe_numeric(
            df["No of Module"]
        )
        *
        safe_numeric(
            df["Area of 1 Module (m2)"
               ]
        )
    )

    # --------------------------------------------------------
    # AREA / CLUSTER TABLE
    # --------------------------------------------------------

    buffer.seek(0)

    df_w = pd.read_excel(
        buffer,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15]
    )

    df_w = clean_columns(
        df_w
    )

    if "Clusters" not in df_w.columns:

        # Some workbook versions may have
        # slightly different spelling.

        cluster_col = (
            df_w.columns[0]
        )

        df_w = df_w.rename(
            columns={
                cluster_col: "Clusters"
            }
        )

    pos = first_blank_position(
        df_w,
        "Clusters"
    )

    df_w = df_w.iloc[
        :pos
    ].copy()

    df_w.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # LATITUDE
    # --------------------------------------------------------

    buffer.seek(0)

    df_config = pd.read_excel(
        buffer,
        sheet_name="Forecast Config",
        header=8
    )

    if "Lat" not in df_config.columns:

        raise ValueError(
            "Lat column not found in Forecast Config."
        )

    lat = float(
        pd.to_numeric(
            df_config.loc[
                0,
                "Lat"
            ],
            errors="coerce"
        )
    )

    # --------------------------------------------------------
    # TILT
    # --------------------------------------------------------

    buffer.seek(0)

    df_tilt = pd.read_excel(
        buffer,
        sheet_name="Config Tilt Angle",
        header=7
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" not in df_tilt.columns:

        raise ValueError(
            "Fixed tilt column not found."
        )

    pos = first_blank_position(
        df_tilt,
        "Fixed"
    )

    df_tilt = df_tilt.iloc[
        :pos
    ].copy()

    df_tilt = df_tilt.dropna(
        how="all",
        axis=1
    )

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month"
        }
    )

    if (
        "Month" not in df_tilt.columns
        or
        "Month_Num" not in df_tilt.columns
    ):

        raise ValueError(
            "Month information not found "
            "in Config Tilt Angle."
        )

    df_tilt["Fixed"] = pd.to_numeric(
        df_tilt["Fixed"],
        errors="coerce"
    )

    month_lookup = (
        df_tilt
        .dropna(
            subset=[
                "Month",
                "Fixed"
            ]
        )
        .set_index(
            "Month"
        )["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # RESULT / GHI
    # --------------------------------------------------------

    buffer.seek(0)

    df_ghi = pd.read_excel(
        buffer,
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

    # --------------------------------------------------------
    # VCAST ACTUAL
    # --------------------------------------------------------

    buffer.seek(0)

    df_vcast = pd.read_excel(
        buffer,
        sheet_name="Fixed-C11",
        header=1
    )

    df_vcast = clean_columns(
        df_vcast
    )

    if "Date" not in df_vcast.columns:

        raise ValueError(
            "Date column not found in Fixed-C11."
        )

    pos = first_blank_position(
        df_vcast,
        "Date"
    )

    df_vcast = df_vcast.iloc[
        :pos
    ].copy()

    df_vcast.reset_index(
        drop=True,
        inplace=True
    )

    if "Actual" not in df_vcast.columns:

        raise ValueError(
            "Actual column not found in Fixed-C11."
        )

    df_vcast["Actual"] = pd.to_numeric(
        df_vcast["Actual"],
        errors="coerce"
    ).fillna(0)

    df_vcast["Date"] = pd.to_datetime(
        df_vcast["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # ALIGN
    # --------------------------------------------------------

    n = min(
        len(df_vcast),
        len(df_ghi)
    )

    if n == 0:

        raise ValueError(
            "No aligned VCast rows found."
        )

    df_vcast = df_vcast.iloc[
        :n
    ].copy()

    df_ghi = df_ghi.iloc[
        :n
    ].copy()

    actual = (
        df_vcast["Actual"]
        .to_numpy(
            dtype=float
        )
    )

    ghi_matrix = np.column_stack(
        [
            df_ghi[col].to_numpy(
                dtype=float
            )
            for col in GHI_COLS
        ]
    )

    blocks = pd.to_numeric(
        df_ghi["Block"],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    dates = pd.to_datetime(
        df_vcast["Date"],
        errors="coerce"
    )

    # Original workbook dates are retained.
    #
    # If invalid dates exist, use today's date only
    # for those invalid records.

    fallback_date = (
        pd.Timestamp.today()
        .normalize()
    )

    dates = dates.fillna(
        fallback_date
    )

    # --------------------------------------------------------
    # SOLAR GEOMETRY
    # --------------------------------------------------------

    first_date = pd.Timestamp(
        year=2025,
        month=1,
        day=1
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

    # Month names are used because that is how
    # the workbook tilt lookup is structured.

    month_names = (
        dates
        .dt
        .strftime("%B")
    )

    tilt = np.array(
        [
            month_lookup.get(
                month_name,
                0
            )
            for month_name in month_names
        ],
        dtype=float
    )

    a_plus_b = (
        elevation
        +
        tilt
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
        sin_a
    )

    # --------------------------------------------------------
    # FIXED POA FOR ALL FIVE CLUSTERS
    # --------------------------------------------------------

    fixed_poa = (
        ghi_matrix
        *
        sin_ab[:, None]
        /
        sin_a_safe[:, None]
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "df": df,

        "df_w": df_w,

        "df_ghi": df_ghi,

        "df_vcast": df_vcast,

        "actual": actual,

        "ghi_matrix": ghi_matrix,

        "blocks": blocks,

        "dates": dates,

        "lat": lat,

        "tilt": tilt,

        "declination": declination,

        "elevation": elevation,

        "sin_a": sin_a,

        "sin_ab": sin_ab,

        "fixed_poa": fixed_poa,

        "n": n
    }


# ============================================================
# LOAD DATA
# ============================================================

try:

    data = load_common_data(
        st.session_state["file_bytes"]
    )

except Exception as exc:

    st.error(
        f"Data loading error: {exc}"
    )

    st.stop()


df = data["df"]

df_w = data["df_w"]

df_vcast = data["df_vcast"]

actual = data["actual"]

ghi_matrix = data["ghi_matrix"]

blocks = data["blocks"]

fixed_poa = data["fixed_poa"]

n = data["n"]


# ============================================================
# ACTUAL VALIDATION
# ============================================================

valid_mask = (
    np.isfinite(actual)
    &
    (actual != 0)
)

if not valid_mask.any():

    st.error(
        "No valid non-zero Actual Power values found."
    )

    st.stop()


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

    st.error(
        "Actual Peak must be greater than zero."
    )

    st.stop()


# ============================================================
# EFFECTIVE AREA CALCULATION
# ============================================================

def calculate_effective_areas(
    error_pct
):

    temp_df = df.copy()

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    temp_df["Error %"] = (
        error_pct
    )

    # --------------------------------------------------------
    # Net Efficiency
    # --------------------------------------------------------

    temp_df["Net Efficiency (%)"] = (
        safe_numeric(
            temp_df[
                "Standard PV Efficiency (%)"
            ]
        )
        -
        error_pct
    )

    temp_df[
        "Net Efficiency (%)"
    ] = np.maximum(
        temp_df[
            "Net Efficiency (%)"
        ],
        0
    )

    # --------------------------------------------------------
    # Total Area
    # --------------------------------------------------------

    temp_df["Total area (m2)"] = (
        safe_numeric(
            temp_df[
                "No of Module"
            ]
        )
        *
        safe_numeric(
            temp_df[
                "Area of 1 Module (m2)"
            ]
        )
    )

    # --------------------------------------------------------
    # Effective Area
    # --------------------------------------------------------

    temp_df["Eff Area"] = (
        temp_df[
            "Net Efficiency (%)"
        ]
        *
        temp_df[
            "Total area (m2)"
        ]
        /
        100
    )

    # --------------------------------------------------------
    # Cluster sums
    # --------------------------------------------------------

    cluster_sums = (
        temp_df
        .groupby(
            "Clusters"
        )["Eff Area"]
        .sum()
    )

    effective_areas = (
        df_w[
            "Clusters"
        ]
        .map(
            cluster_sums
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    return (
        temp_df,
        effective_areas
    )


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    error_pct
):

    temp_df, effective_areas = (
        calculate_effective_areas(
            error_pct
        )
    )

    power_matrix = (
        fixed_poa
        *
        effective_areas[None, :]
        /
        1_000_000
    )

    forecast = (
        power_matrix.sum(
            axis=1
        )
    )

    return (
        forecast,
        power_matrix,
        effective_areas,
        temp_df
    )


# ============================================================
# FIXED ERROR OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False
)
def optimize_fixed_error(
    file_bytes
):

    loaded = load_common_data(
        file_bytes
    )

    local_df = loaded["df"]

    local_df_w = loaded["df_w"]

    local_actual = loaded["actual"]

    local_fixed_poa = loaded["fixed_poa"]

    local_valid = (
        np.isfinite(
            local_actual
        )
        &
        (
            local_actual
            != 0
        )
    )

    local_actual_day = (
        local_actual[
            local_valid
        ]
    )

    local_actual_peak = (
        np.max(
            local_actual_day
        )
    )

    rows = []

    # --------------------------------------------------------
    # Error scan
    #
    # 0 to 10% in 0.1% steps
    # --------------------------------------------------------

    for error in np.arange(
        0,
        10.0001,
        0.1
    ):

        temp = local_df.copy()

        temp["Error %"] = (
            error
        )

        temp[
            "Net Efficiency (%)"
        ] = (
            pd.to_numeric(
                temp[
                    "Standard PV Efficiency (%)"
                ],
                errors="coerce"
            )
            -
            error
        )

        temp[
            "Net Efficiency (%)"
        ] = np.maximum(
            temp[
                "Net Efficiency (%)"
            ],
            0
        )

        temp[
            "Total area (m2)"
        ] = (
            pd.to_numeric(
                temp[
                    "No of Module"
                ],
                errors="coerce"
            ).fillna(0)
            *
            pd.to_numeric(
                temp[
                    "Area of 1 Module (m2)"
                ],
                errors="coerce"
            ).fillna(0)
        )

        temp[
            "Eff Area"
        ] = (
            temp[
                "Net Efficiency (%)"
            ]
            *
            temp[
                "Total area (m2)"
            ]
            /
            100
        )

        cluster_sums = (
            temp
            .groupby(
                "Clusters"
            )["Eff Area"]
            .sum()
        )

        weights = (
            local_df_w[
                "Clusters"
            ]
            .map(
                cluster_sums
            )
            .fillna(0)
            .to_numpy(
                dtype=float
            )
        )

        prediction = (
            local_fixed_poa
            *
            weights[None, :]
            /
            1_000_000
        ).sum(
            axis=1
        )

        prediction_day = (
            prediction[
                local_valid
            ]
        )

        calculated_peak = (
            np.max(
                prediction_day
            )
        )

        peak_error = abs(
            calculated_peak
            -
            local_actual_peak
        )

        peak_error_pct = (
            peak_error
            /
            local_actual_peak
            *
            100
        )

        rows.append({

            "Error %": error,

            "Calculated Peak":
                calculated_peak,

            "Actual Peak":
                local_actual_peak,

            "Peak Error":
                peak_error,

            "Peak Error %":
                peak_error_pct
        })

    result_df = pd.DataFrame(
        rows
    )

    best_index = (
        result_df[
            "Peak Error"
        ].idxmin()
    )

    best_error = float(
        result_df.loc[
            best_index,
            "Error %"
        ]
    )

    return (
        best_error,
        result_df
    )


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking_forecast(
    error_pct,
    dhi,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit
):

    # --------------------------------------------------------
    # Validate block sequence
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Zenith
    # --------------------------------------------------------

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
            )
        ),

        np.minimum(
            89,
            m2
            *
            (
                blocks
                -
                max_block
            )
        )
    )

    # --------------------------------------------------------
    # Panel
    # --------------------------------------------------------

    panel = np.where(

        blocks < max_block,

        np.minimum(
            zenith,
            abs(
                east_limit
            )
        ),

        np.where(

            (
                (blocks > max_block)
                &
                (
                    zenith
                    >
                    west_limit
                )
            ),

            west_limit,

            zenith
        )
    )

    # --------------------------------------------------------
    # Cosine
    # --------------------------------------------------------

    cos_alpha = np.cos(
        np.radians(
            panel
        )
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None
    )

    # --------------------------------------------------------
    # Effective Areas
    #
    # Error is applied here.
    # --------------------------------------------------------

    _, tracking_weights = (
        calculate_effective_areas(
            error_pct
        )
    )

    # --------------------------------------------------------
    # DHI
    # --------------------------------------------------------

    dhi_matrix = (
        ghi_matrix
        *
        dhi
        /
        100
    )

    # --------------------------------------------------------
    # DNI
    # --------------------------------------------------------

    dni = (
        ghi_matrix
        -
        dhi_matrix
    ) / cos_alpha[:, None]

    # --------------------------------------------------------
    # Tracking Power
    # --------------------------------------------------------

    power_matrix = (
        dni
        *
        tracking_weights[None, :]
        /
        1_000_000
    )

    forecast = (
        power_matrix.sum(
            axis=1
        )
    )

    if not np.all(
        np.isfinite(
            forecast
        )
    ):

        return None

    return {

        "forecast": forecast,

        "power_matrix": power_matrix,

        "zenith": zenith,

        "panel": panel,

        "dni": dni,

        "weights": tracking_weights
    }


# ============================================================
# TRACKING ERROR OPTIMIZATION
# ============================================================
#
# IMPORTANT:
#
# User requirement:
#
#   1. Error % first
#   2. Then DHI / GHI blocks / angle limits
#
# Therefore Error % is calculated first using the current
# initial tracking parameter values.
#
# Once Error % is selected, the six tracking parameters
# are optimized using that Error %.
#
# ============================================================

@st.cache_data(
    show_spinner=False
)
def optimize_tracking_error(
    file_bytes,
    initial_dhi,
    initial_start,
    initial_end,
    initial_max,
    initial_east,
    initial_west
):

    loaded = load_common_data(
        file_bytes
    )

    local_actual = loaded[
        "actual"
    ]

    local_valid = (
        np.isfinite(
            local_actual
        )
        &
        (
            local_actual
            != 0
        )
    )

    local_actual_day = (
        local_actual[
            local_valid
        ]
    )

    local_actual_peak = (
        np.max(
            local_actual_day
        )
    )

    rows = []

    for error in np.arange(
        0,
        10.0001,
        0.1
    ):

        result = calculate_tracking_forecast_static(
            loaded,
            error,
            initial_dhi,
            initial_start,
            initial_end,
            initial_max,
            initial_east,
            initial_west
        )

        if result is None:

            continue

        prediction = (
            result["forecast"]
        )

        prediction_day = (
            prediction[
                local_valid
            ]
        )

        calculated_peak = (
            np.max(
                prediction_day
            )
        )

        peak_error = abs(
            calculated_peak
            -
            local_actual_peak
        )

        peak_error_pct = (
            peak_error
            /
            local_actual_peak
            *
            100
        )

        rows.append({

            "Error %": error,

            "Calculated Peak":
                calculated_peak,

            "Actual Peak":
                local_actual_peak,

            "Peak Error":
                peak_error,

            "Peak Error %":
                peak_error_pct
        })

    result_df = pd.DataFrame(
        rows
    )

    if result_df.empty:

        raise ValueError(
            "Tracking Error % optimization "
            "could not produce valid results."
        )

    best_index = (
        result_df[
            "Peak Error"
        ].idxmin()
    )

    best_error = float(
        result_df.loc[
            best_index,
            "Error %"
        ]
    )

    return (
        best_error,
        result_df
    )


def calculate_tracking_forecast_static(
    loaded,
    error_pct,
    dhi,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit
):

    local_blocks = loaded[
        "blocks"
    ]

    local_ghi = loaded[
        "ghi_matrix"
    ]

    # --------------------------------------------------------
    # Effective areas
    # --------------------------------------------------------

    local_df = loaded[
        "df"
    ]

    local_df_w = loaded[
        "df_w"
    ]

    temp = local_df.copy()

    temp["Error %"] = (
        error_pct
    )

    temp[
        "Net Efficiency (%)"
    ] = (
        pd.to_numeric(
            temp[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce"
        )
        -
        error_pct
    )

    temp[
        "Net Efficiency (%)"
    ] = np.maximum(
        temp[
            "Net Efficiency (%)"
        ],
        0
    )

    temp[
        "Total area (m2)"
    ] = (
        pd.to_numeric(
            temp[
                "No of Module"
            ],
            errors="coerce"
        ).fillna(0)
        *
        pd.to_numeric(
            temp[
                "Area of 1 Module (m2)"
            ],
            errors="coerce"
        ).fillna(0)
    )

    temp[
        "Eff Area"
    ] = (
        temp[
            "Net Efficiency (%)"
        ]
        *
        temp[
            "Total area (m2)"
        ]
        /
        100
    )

    cluster_sums = (
        temp
        .groupby(
            "Clusters"
        )["Eff Area"]
        .sum()
    )

    weights = (
        local_df_w[
            "Clusters"
        ]
        .map(
            cluster_sums
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not (
        start_block
        <
        max_block
        <
        end_block
    ):

        return None

    den1 = (
        start_block
        -
        1
        -
        max_block
    )

    den2 = (
        end_block
        +
        1
        -
        max_block
    )

    if den1 == 0 or den2 == 0:

        return None

    m1 = (
        90
        /
        den1
    )

    m2 = (
        90
        /
        den2
    )

    zenith = np.where(

        local_blocks <= max_block,

        np.minimum(
            89,
            m1
            *
            (
                local_blocks
                -
                max_block
            )
        ),

        np.minimum(
            89,
            m2
            *
            (
                local_blocks
                -
                max_block
            )
        )
    )

    panel = np.where(

        local_blocks < max_block,

        np.minimum(
            zenith,
            abs(
                east_limit
            )
        ),

        np.where(

            (
                (local_blocks > max_block)
                &
                (
                    zenith
                    >
                    west_limit
                )
            ),

            west_limit,

            zenith
        )
    )

    cos_alpha = np.cos(
        np.radians(
            panel
        )
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None
    )

    dhi_matrix = (
        local_ghi
        *
        dhi
        /
        100
    )

    dni = (
        local_ghi
        -
        dhi_matrix
    ) / cos_alpha[:, None]

    forecast = (
        dni
        @
        weights
    ) / 1_000_000

    return {

        "forecast": forecast,

        "weights": weights,

        "zenith": zenith,

        "panel": panel,

        "dni": dni
    }


# ============================================================
# RUN AUTOMATIC OPTIMIZATION
# ============================================================

if plant_type == "Fixed":

    if not st.session_state[
        "fixed_auto_done"
    ]:

        with st.spinner(
            "Calculating Fixed Error %..."
        ):

            (
                auto_error,
                fixed_error_results
            ) = optimize_fixed_error(
                st.session_state[
                    "file_bytes"
                ]
            )

        st.session_state[
            "auto_fixed_error"
        ] = auto_error

        st.session_state[
            "fixed_error"
        ] = auto_error

        st.session_state[
            "fixed_auto_done"
        ] = True

    else:

        (
            auto_error,
            fixed_error_results
        ) = optimize_fixed_error(
            st.session_state[
                "file_bytes"
            ]
        )


elif plant_type == "Tracking":

    # --------------------------------------------------------
    # First calculate Tracking Error %
    # --------------------------------------------------------

    if not st.session_state[
        "tracking_auto_done"
    ]:

        with st.spinner(
            "Step 1/2: Calculating Tracking Error %..."
        ):

            (
                tracking_auto_error,
                tracking_error_results
            ) = optimize_tracking_error(

                st.session_state[
                    "file_bytes"
                ],

                st.session_state[
                    "tracking_dhi"
                ],

                st.session_state[
                    "tracking_start"
                ],

                st.session_state[
                    "tracking_end"
                ],

                st.session_state[
                    "tracking_max"
                ],

                st.session_state[
                    "tracking_east"
                ],

                st.session_state[
                    "tracking_west"
                ]
            )

        st.session_state[
            "tracking_error"
        ] = tracking_auto_error

        st.session_state[
            "tracking_auto_done"
        ] = True

    else:

        (
            tracking_auto_error,
            tracking_error_results
        ) = optimize_tracking_error(

            st.session_state[
                "file_bytes"
            ],

            st.session_state[
                "tracking_dhi"
            ],

            st.session_state[
                "tracking_start"
            ],

            st.session_state[
                "tracking_end"
            ],

            st.session_state[
                "tracking_max"
            ],

            st.session_state[
                "tracking_east"
            ],

            st.session_state[
                "tracking_west"
            ]
        )


# ============================================================
# PARAMETER SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Calculated Parameters</div>',
    unsafe_allow_html=True
)


if plant_type == "Fixed":

    # --------------------------------------------------------
    # Fixed Error
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        fixed_error = st.number_input(
            "Error / Efficiency Loss (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(
                st.session_state[
                    "fixed_error"
                ]
            ),
            step=0.1,
            format="%.2f",
            key="fixed_error_input"
        )

    with col2:

        st.metric(
            "Auto Calculated Error %",
            f"{auto_error:.2f}%"
        )

    with col3:

        st.metric(
            "Optimization Basis",
            "Minimum Peak Error"
        )

    st.session_state[
        "fixed_error"
    ] = fixed_error

    active_error = fixed_error


else:

    # --------------------------------------------------------
    # Tracking parameters
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        tracking_error = st.number_input(
            "Error / Efficiency Loss (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(
                st.session_state[
                    "tracking_error"
                ]
            ),
            step=0.1,
            format="%.2f",
            key="tracking_error_input"
        )

    with col2:

        st.metric(
            "Auto Calculated Error %",
            f"{tracking_auto_error:.2f}%"
        )

    with col3:

        st.metric(
            "Optimization Basis",
            "Minimum Peak Error"
        )

    st.session_state[
        "tracking_error"
    ] = tracking_error

    active_error = tracking_error

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        dhi = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(
                st.session_state[
                    "tracking_dhi"
                ]
            ),
            step=1,
            key="tracking_dhi_input"
        )

    with col2:

        start_block = st.number_input(
            "GHI Starting Block",
            min_value=1,
            max_value=95,
            value=int(
                st.session_state[
                    "tracking_start"
                ]
            ),
            step=1,
            key="tracking_start_input"
        )

    with col3:

        end_block = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                st.session_state[
                    "tracking_end"
                ]
            ),
            step=1,
            key="tracking_end_input"
        )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        max_block = st.number_input(
            "GHI Max Block",
            min_value=1,
            max_value=96,
            value=int(
                st.session_state[
                    "tracking_max"
                ]
            ),
            step=1,
            key="tracking_max_input"
        )

    with col2:

        east_limit = st.number_input(
            "East Tracking Limit (°)",
            min_value=0,
            max_value=90,
            value=int(
                st.session_state[
                    "tracking_east"
                ]
            ),
            step=1,
            key="tracking_east_input"
        )

    with col3:

        west_limit = st.number_input(
            "West Tracking Limit (°)",
            min_value=0,
            max_value=90,
            value=int(
                st.session_state[
                    "tracking_west"
                ]
            ),
            step=1,
            key="tracking_west_input"
        )

    st.session_state[
        "tracking_dhi"
    ] = dhi

    st.session_state[
        "tracking_start"
    ] = start_block

    st.session_state[
        "tracking_end"
    ] = end_block

    st.session_state[
        "tracking_max"
    ] = max_block

    st.session_state[
        "tracking_east"
    ] = east_limit

    st.session_state[
        "tracking_west"
    ] = west_limit


# ============================================================
# CALCULATE CURRENT MODEL
# ============================================================

if plant_type == "Fixed":

    (
        forecast,
        power_matrix,
        effective_areas,
        final_area_df
    ) = calculate_fixed_forecast(
        active_error
    )

    final_metrics = calculate_metrics(
        actual,
        forecast
    )

    parameter_status = (
        f"Fixed model using Error % = "
        f"{active_error:.2f}%"
    )

else:

    if not (
        start_block
        <
        max_block
        <
        end_block
    ):

        st.error(
            "Invalid tracking blocks. "
            "Required: Start Block < Max Block < End Block."
        )

        st.stop()

    tracking_result = (
        calculate_tracking_forecast(
            active_error,
            dhi,
            start_block,
            end_block,
            max_block,
            east_limit,
            west_limit
        )
    )

    if tracking_result is None:

        st.error(
            "Tracking calculation failed. "
            "Check the tracking parameters."
        )

        st.stop()

    forecast = (
        tracking_result[
            "forecast"
        ]
    )

    power_matrix = (
        tracking_result[
            "power_matrix"
        ]
    )

    effective_areas = (
        tracking_result[
            "weights"
        ]
    )

    zenith = (
        tracking_result[
            "zenith"
        ]
    )

    panel = (
        tracking_result[
            "panel"
        ]
    )

    dni = (
        tracking_result[
            "dni"
        ]
    )

    final_area_df, _ = (
        calculate_effective_areas(
            active_error
        )
    )

    final_metrics = calculate_metrics(
        actual,
        forecast
    )

    parameter_status = (
        f"Tracking model using Error % = "
        f"{active_error:.2f}%"
    )


# ============================================================
# METRICS
# ============================================================

st.markdown(
    '<div class="section-title">Model Performance</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4, m5 = st.columns(
    5
)

with m1:

    metric_card(
        "Actual Peak",
        f"{final_metrics['actual_peak']:.4f} MW",
        "Reference"
    )

with m2:

    metric_card(
        "Forecast Peak",
        f"{final_metrics['forecast_peak']:.4f} MW",
        "Current model"
    )

with m3:

    metric_card(
        "Peak Error",
        f"{final_metrics['peak_error_pct']:.3f}%",
        "Lower is better"
    )

with m4:

    metric_card(
        "Block Error",
        f"{final_metrics['block_error']:.5f}",
        "Normalized"
    )

with m5:

    metric_card(
        "Energy Error",
        f"{final_metrics['energy_error'] * 100:.3f}%",
        "Lower is better"
    )


st.markdown(
    f"""
    <div class="status-box">
        <b>Active Model:</b> {parameter_status}
        &nbsp;&nbsp; | &nbsp;&nbsp;
        <b>Overall Score:</b>
        {final_metrics["score"]:.6f}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TRACKING PARAMETER SUMMARY
# ============================================================

if plant_type == "Tracking":

    st.markdown(
        '<div class="section-title">Tracking Parameters</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4, p5, p6 = st.columns(
        6
    )

    with p1:
        metric_card(
            "DHI",
            f"{dhi}%",
            "Editable"
        )

    with p2:
        metric_card(
            "Start Block",
            str(start_block),
            "Editable"
        )

    with p3:
        metric_card(
            "End Block",
            str(end_block),
            "Editable"
        )

    with p4:
        metric_card(
            "Max Block",
            str(max_block),
            "Editable"
        )

    with p5:
        metric_card(
            "East Limit",
            f"{east_limit}°",
            "Editable"
        )

    with p6:
        metric_card(
            "West Limit",
            f"{west_limit}°",
            "Editable"
        )


# ============================================================
# CHART
# ============================================================

st.markdown(
    '<div class="section-title">Actual vs Forecast</div>',
    unsafe_allow_html=True
)

x = np.arange(
    len(actual)
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=x,
        y=actual,
        mode="lines",
        name="Actual",
        line=dict(
            width=2.5
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=x,
        y=forecast,
        mode="lines",
        name=(
            "Fixed Forecast"
            if plant_type == "Fixed"
            else "Tracking Forecast"
        ),
        line=dict(
            width=2.5
        )
    )
)

fig.update_layout(

    height=480,

    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    ),

    xaxis_title="Block",

    yaxis_title="Power (MW)",

    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),

    template="plotly_white"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# TRACKING ANGLES
# ============================================================

if plant_type == "Tracking":

    st.markdown(
        '<div class="section-title">Tracking Angle Profile</div>',
        unsafe_allow_html=True
    )

    angle_fig = go.Figure()

    angle_fig.add_trace(
        go.Scatter(
            x=x,
            y=zenith,
            mode="lines",
            name="Zenith Angle"
        )
    )

    angle_fig.add_trace(
        go.Scatter(
            x=x,
            y=panel,
            mode="lines",
            name="Panel Angle"
        )
    )

    angle_fig.update_layout(

        height=400,

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),

        xaxis_title="Block",

        yaxis_title="Angle (°)",

        hovermode="x unified",

        template="plotly_white"
    )

    st.plotly_chart(
        angle_fig,
        use_container_width=True
    )


# ============================================================
# CLUSTER RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">Cluster Results</div>',
    unsafe_allow_html=True
)

cluster_result = pd.DataFrame({

    "Cluster": CLUSTERS,

    "Effective Area (m²)":
        effective_areas,

    "Peak Forecast (MW)":
        [
            np.max(
                power_matrix[
                    valid_mask,
                    i
                ]
            )
            for i in range(
                min(
                    5,
                    power_matrix.shape[1]
                )
            )
        ]
})

st.dataframe(
    cluster_result,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# EFFICIENCY TABLE
# ============================================================

st.markdown(
    '<div class="section-title">Area & Efficiency</div>',
    unsafe_allow_html=True
)

display_area_df = (
    final_area_df
    .copy()
)

st.dataframe(
    display_area_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ERROR OPTIMIZATION TABLE
# ============================================================

st.markdown(
    '<div class="section-title">Error % Optimization</div>',
    unsafe_allow_html=True
)

if plant_type == "Fixed":

    error_table = (
        fixed_error_results
        .copy()
    )

else:

    error_table = (
        tracking_error_results
        .copy()
    )


best_error_row = (
    error_table
    .loc[
        error_table[
            "Peak Error"
        ].idxmin()
    ]
)


st.dataframe(
    error_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TRACKING PARAMETER OPTIMIZATION
# ============================================================

if plant_type == "Tracking":

    st.markdown(
        '<div class="section-title">'
        'Tracking Parameter Optimization'
        '</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "⚙️ Auto Optimize Tracking Parameters",
        type="primary",
        use_container_width=False
    ):

        with st.spinner(
            "Optimizing DHI, GHI blocks and tracking limits..."
        ):

            local_file = (
                st.session_state[
                    "file_bytes"
                ]
            )

            loaded = load_common_data(
                local_file
            )

            local_actual = (
                loaded[
                    "actual"
                ]
            )

            local_valid = (
                np.isfinite(
                    local_actual
                )
                &
                (
                    local_actual
                    != 0
                )
            )

            local_actual_day = (
                local_actual[
                    local_valid
                ]
            )

            local_actual_peak = (
                np.max(
                    local_actual_day
                )
            )

            local_actual_energy = (
                np.sum(
                    local_actual_day
                )
            )

            def tracking_objective(
                values
            ):

                local_dhi = int(
                    round(
                        values[0]
                    )
                )

                local_start = int(
                    round(
                        values[1]
                    )
                )

                local_end = int(
                    round(
                        values[2]
                    )
                )

                local_max = int(
                    round(
                        values[3]
                    )
                )

                local_east = int(
                    round(
                        values[4]
                    )
                )

                local_west = int(
                    round(
                        values[5]
                    )
                )

                if not (
                    local_start
                    <
                    local_max
                    <
                    local_end
                ):

                    return 1e9

                calc = (
                    calculate_tracking_forecast_static(

                        loaded,

                        active_error,

                        local_dhi,

                        local_start,

                        local_end,

                        local_max,

                        local_east,

                        local_west
                    )
                )

                if calc is None:

                    return 1e9

                pred = (
                    calc[
                        "forecast"
                    ]
                )

                if not np.all(
                    np.isfinite(
                        pred
                    )
                ):

                    return 1e9

                pred_day = (
                    pred[
                        local_valid
                    ]
                )

                if len(
                    pred_day
                ) == 0:

                    return 1e9

                block_error = (
                    np.mean(
                        np.abs(
                            local_actual_day
                            -
                            pred_day
                        )
                    )
                    /
                    local_actual_peak
                )

                peak_error = (
                    abs(
                        local_actual_peak
                        -
                        pred_day.max()
                    )
                    /
                    local_actual_peak
                )

                energy_error = (
                    abs(
                        local_actual_energy
                        -
                        pred_day.sum()
                    )
                    /
                    local_actual_energy
                )

                return (
                    0.80 * block_error
                    +
                    0.10 * peak_error
                    +
                    0.10 * energy_error
                )

            optimization_result = (
                differential_evolution(

                    tracking_objective,

                    bounds=[
                        (0, 10),
                        (10, 30),
                        (65, 80),
                        (47, 53),
                        (10, 70),
                        (10, 70)
                    ],

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
            )

            optimized = (
                np.rint(
                    optimization_result.x
                ).astype(int)
            )

            st.session_state[
                "tracking_dhi"
            ] = int(
                optimized[0]
            )

            st.session_state[
                "tracking_start"
            ] = int(
                optimized[1]
            )

            st.session_state[
                "tracking_end"
            ] = int(
                optimized[2]
            )

            st.session_state[
                "tracking_max"
            ] = int(
                optimized[3]
            )

            st.session_state[
                "tracking_east"
            ] = int(
                optimized[4]
            )

            st.session_state[
                "tracking_west"
            ] = int(
                optimized[5]
            )

        st.success(
            "Tracking parameters optimized successfully. "
            "The editable controls above now contain the optimized values."
        )

        st.rerun()


# ============================================================
# POWER DATA
# ============================================================

st.markdown(
    '<div class="section-title">Forecast Data</div>',
    unsafe_allow_html=True
)

output_df = df_vcast.copy()

if plant_type == "Fixed":

    for i, cluster in enumerate(
        CLUSTERS
    ):

        if i < power_matrix.shape[1]:

            output_df[
                f"{cluster}_Fixed Power=I*Ƞ*A"
            ] = (
                power_matrix[
                    :len(output_df),
                    i
                ]
            )

    output_df[
        "Total Power (CL1+CL2+…)"
    ] = forecast

else:

    output_df[
        "Zenith Angle"
    ] = zenith

    output_df[
        "Panel Angle"
    ] = panel

    output_df[
        "DHI (%)"
    ] = dhi

    output_df[
        "DNI"
    ] = (
        dni.sum(
            axis=1
        )
    )

    for i, cluster in enumerate(
        CLUSTERS
    ):

        if i < power_matrix.shape[1]:

            output_df[
                f"{cluster}_Tracking Power=I*Ƞ*A"
            ] = (
                power_matrix[
                    :len(output_df),
                    i
                ]
            )

    output_df[
        "Tracking Power=I*Ƞ*A"
    ] = forecast


st.dataframe(
    output_df,
    use_container_width=True,
    height=500
)


# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.markdown(
    '<div class="section-title">Download</div>',
    unsafe_allow_html=True
)


def create_excel_report():

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary_df = pd.DataFrame({

            "Metric": [

                "Plant Type",

                "Error / Efficiency Loss (%)",

                "Actual Peak",

                "Forecast Peak",

                "Peak Error",

                "Peak Error (%)",

                "Block Error",

                "Energy Error",

                "Overall Score"
            ],

            "Value": [

                plant_type,

                active_error,

                final_metrics[
                    "actual_peak"
                ],

                final_metrics[
                    "forecast_peak"
                ],

                final_metrics[
                    "peak_error"
                ],

                final_metrics[
                    "peak_error_pct"
                ],

                final_metrics[
                    "block_error"
                ],

                final_metrics[
                    "energy_error"
                ],

                final_metrics[
                    "score"
                ]
            ]
        })

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        if plant_type == "Tracking":

            parameters_df = pd.DataFrame({

                "Parameter": [

                    "Error %",

                    "DHI (%)",

                    "GHI Starting Block",

                    "GHI Ending Block",

                    "GHI Max Block",

                    "East Tracking Limit",

                    "West Tracking Limit"
                ],

                "Value": [

                    active_error,

                    dhi,

                    start_block,

                    end_block,

                    max_block,

                    east_limit,

                    west_limit
                ]
            })

        else:

            parameters_df = pd.DataFrame({

                "Parameter": [

                    "Error %"
                ],

                "Value": [

                    active_error
                ]
            })

        parameters_df.to_excel(
            writer,
            sheet_name="Parameters",
            index=False
        )

        # ----------------------------------------------------
        # Area & Efficiency
        # ----------------------------------------------------

        final_area_df.to_excel(
            writer,
            sheet_name="Area & Efficiency",
            index=False
        )

        # ----------------------------------------------------
        # Forecast
        # ----------------------------------------------------

        output_df.to_excel(
            writer,
            sheet_name="Forecast",
            index=False
        )

        # ----------------------------------------------------
        # Error Optimization
        # ----------------------------------------------------

        error_table.to_excel(
            writer,
            sheet_name="Error Optimization",
            index=False
        )

        # ----------------------------------------------------
        # Cluster Results
        # ----------------------------------------------------

        cluster_result.to_excel(
            writer,
            sheet_name="Cluster Results",
            index=False
        )

    output.seek(0)

    return output


report_bytes = (
    create_excel_report()
)

st.download_button(

    label="⬇️ Download VCast Loss Correction Report",

    data=report_bytes,

    file_name=(
        "VCast_Loss_Correction_Report.xlsx"
    ),

    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),

    type="primary",

    use_container_width=True
)


# ============================================================
# FINAL STATUS
# ============================================================

st.markdown(
    """
    <div class="info-box">

    <b>Calculation logic:</b><br><br>

    <b>Fixed:</b>
    Error % is automatically scanned from 0% to 10% in
    0.1% steps and the value producing the minimum Peak Error
    is selected. Net Efficiency and Effective Area are then
    recalculated using that Error %.<br><br>

    <b>Tracking:</b>
    Error % is calculated first using minimum Peak Error.
    The selected Error % is then used while optimizing DHI,
    GHI Starting Block, GHI Ending Block, GHI Max Block,
    East Tracking Limit and West Tracking Limit.<br><br>

    All automatically calculated values remain editable
    after calculation.

    </div>
    """,
    unsafe_allow_html=True
)
