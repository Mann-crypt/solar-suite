# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
#
# STREAMLIT VERSION
#
# FLOW:
#   1. Upload Excel workbook
#   2. Edit GHI / Actual data
#   3. Select Fixed / Tracking
#   4. Run Automatic Calculation
#   5. Automatically optimize Error %
#   6. If Tracking, automatically optimize tracking parameters
#   7. Show editable parameters
#   8. Changing parameters immediately updates forecast
#   9. Changing parameters NEVER reruns optimization
#
# IMPORTANT:
#   - Differential Evolution runs ONLY from Run button.
#   - Error % is editable after calculation.
#   - Tracking parameters are editable after calculation.
#   - No Apply/Recalculate button is required.
#   - Graph and results remain visible while parameters change.
#   - Uploading a new workbook resets the calculation state.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Forecast Correction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

PLANT_OPTIONS = [
    "Fixed",
    "Tracking",
]

GHI_COLUMNS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

CLUSTERS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
]

FIXED_POA_COLUMNS = [
    "POA fixed",
    "POA Fixed-C12",
    "POA Fixed-C13",
    "POA Fixed-C14",
    "POA Fixed-C15",
]

POWER_COLUMNS = [
    f"CL{i}_Fixed Power=I*Ƞ*A"
    for i in range(1, 6)
]

TOTAL_POWER_COLUMN = "Total Power (CL1+CL2+…)"


# ============================================================
# TRACKING OPTIMIZATION BOUNDS
# ============================================================

TRACKING_BOUNDS = [
    (0, 10),     # DHI
    (10, 30),    # GHI Starting Block
    (65, 80),    # GHI Ending Block
    (47, 53),    # GHI Max Block
    (10, 70),    # Tracking East Limit
    (10, 70),    # Tracking West Limit
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }

    .app-title {
        font-size: 30px;
        font-weight: 750;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 92px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.035);
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 23px;
        font-weight: 750;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 7px;
    }

    div[data-testid="stSegmentedControl"] {
        width: 100%;
    }

    div[data-testid="stSegmentedControl"] > div {
        width: 100%;
    }

    div[data-testid="stSegmentedControl"] button {
        flex: 1;
        font-weight: 650;
    }

    .run-note {
        color: #6b7280;
        font-size: 12px;
        margin-top: 5px;
    }

    .parameter-note {
        color: #6b7280;
        font-size: 12px;
        margin-top: -5px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "calculated": False,
    "calculation_data": None,

    "plant_type": "Fixed",
    "calculated_plant_type": None,

    "input_df": None,

    "last_file_hash": None,

    # Editable parameter state
    "error_value": None,

    "tracking_params": None,

    # Indicates whether current editable parameters
    # have valid forecast data behind them.
    "forecast_ready": False,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Automatic parameter optimization with editable final parameters"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def numeric_array(value):
    """
    Safely convert input into numeric numpy array.
    """

    if isinstance(value, pd.Series):

        return (
            pd.to_numeric(
                value,
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(dtype=float)
        )

    return (
        pd.to_numeric(
            pd.Series(value),
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )


def numeric_series(value):
    """
    Safely convert input into numeric Series.
    """

    return pd.to_numeric(
        value,
        errors="coerce",
    ).fillna(0)


def safe_float(value, default=0.0):
    """
    Safely convert value to float.
    """

    try:

        result = float(
            pd.to_numeric(
                value,
                errors="coerce",
            )
        )

        if not np.isfinite(result):
            return float(default)

        return result

    except Exception:

        return float(default)


def file_hash(uploaded_file):
    """
    Stable SHA-256 hash for uploaded workbook.
    """

    return hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


def read_excel_bytes(file_bytes, **kwargs):
    """
    Read Excel directly from memory.
    """

    return pd.read_excel(
        io.BytesIO(file_bytes),
        **kwargs,
    )


def reset_calculation_state():
    """
    Reset calculation-related state.

    IMPORTANT:
    This is NOT called when an editable parameter changes.
    """

    st.session_state.calculated = False
    st.session_state.calculation_data = None
    st.session_state.calculated_plant_type = None

    st.session_state.error_value = None
    st.session_state.tracking_params = None
    st.session_state.forecast_ready = False


def reset_file_state():
    """
    Completely reset state when a new workbook is uploaded.
    """

    reset_calculation_state()

    st.session_state.input_df = None
    st.session_state.plant_type = "Fixed"

    # Remove widget state associated with previous workbook.
    keys_to_remove = [
        "plant_type_selector",
        "solar_input_editor",
    ]

    for key in list(st.session_state.keys()):

        if (
            key in keys_to_remove
            or key.startswith("error_widget_")
            or key.startswith("tracking_")
        ):
            try:
                del st.session_state[key]
            except Exception:
                pass


# ============================================================
# CACHED WORKBOOK LOAD
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=3,
)
def load_workbook(file_bytes):

    area = read_excel_bytes(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=[1],
        usecols=range(12),
    )

    cluster = read_excel_bytes(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    ghi = read_excel_bytes(
        file_bytes,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    forecast_config = read_excel_bytes(
        file_bytes,
        sheet_name="Forecast Config",
        header=[8],
    )

    tilt = read_excel_bytes(
        file_bytes,
        sheet_name="Config Tilt Angle",
        header=[7],
    )

    fixed = read_excel_bytes(
        file_bytes,
        sheet_name="Fixed-C11",
        header=[1],
    )

    tracking = read_excel_bytes(
        file_bytes,
        sheet_name="Tracking",
        header=1,
    )

    backend = {}

    for cluster_name in CLUSTERS:

        backend[cluster_name] = read_excel_bytes(
            file_bytes,
            sheet_name=f"Backend Cal {cluster_name}",
        )

    return {
        "area": area,
        "cluster": cluster,
        "ghi": ghi,
        "forecast_config": forecast_config,
        "tilt": tilt,
        "fixed": fixed,
        "tracking": tracking,
        "backend": backend,
    }


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def prepare_area_efficiency(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False,
        )
        .str.strip()
    )

    if "S.No." in df.columns:

        mask = df["S.No."].isna()

        if mask.any():

            first = np.flatnonzero(
                mask.to_numpy()
            )[0]

            df = df.iloc[:first].copy()

    required = [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]

    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"Missing column in Area & Efficiency: {col}"
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df.reset_index(drop=True)


# ============================================================
# CLUSTER TABLE
# ============================================================

def prepare_cluster_table(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "Clusters" in df.columns:

        mask = df["Clusters"].isna()

        if mask.any():

            first = np.flatnonzero(
                mask.to_numpy()
            )[0]

            df = df.iloc[:first].copy()

    return df.reset_index(drop=True)


# ============================================================
# GHI
# ============================================================

def prepare_ghi(df):

    df = df.copy()

    df = df.fillna(0)

    for col in GHI_COLUMNS:

        if col not in df.columns:

            raise ValueError(
                f"Missing GHI column: {col}"
            )

        df[col] = numeric_series(
            df[col]
        )

    return df.reset_index(drop=True)


# ============================================================
# LATITUDE
# ============================================================

def prepare_latitude(df):

    if "Lat" not in df.columns:

        raise ValueError(
            "Latitude column 'Lat' not found."
        )

    lat = pd.to_numeric(
        df.loc[0, "Lat"],
        errors="coerce",
    )

    if pd.isna(lat):

        raise ValueError(
            "Latitude value is invalid."
        )

    return float(lat)


# ============================================================
# TILT
# ============================================================

def prepare_tilt(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" not in df.columns:

        raise ValueError(
            "Fixed tilt column not found."
        )

    if df["Fixed"].isna().any():

        mask = df["Fixed"].isna()

        first = np.flatnonzero(
            mask.to_numpy()
        )[0]

        df = df.iloc[:first].copy()

    df = df.dropna(
        how="all",
        axis=1,
    )

    df = df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    if "Month" not in df.columns:

        raise ValueError(
            "Month column not found in Config Tilt Angle."
        )

    return (
        df
        .set_index("Month")["Fixed"]
        .to_dict()
    )


# ============================================================
# FIXED DATA
# ============================================================

def prepare_fixed(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "Actual" not in df.columns:

        raise ValueError(
            "Actual column not found in Fixed-C11."
        )

    if "Date" in df.columns:

        mask = df["Date"].isna()

        if mask.any():

            first = np.flatnonzero(
                mask.to_numpy()
            )[0]

            df = df.iloc[:first].copy()

    df["Actual"] = numeric_series(
        df["Actual"]
    )

    return df.reset_index(drop=True)


# ============================================================
# BUILD INPUT DATAFRAME
# ============================================================

def build_input_dataframe(
    df_ghi,
    df_fixed,
):

    n = min(
        len(df_ghi),
        len(df_fixed),
    )

    if n <= 0:

        raise ValueError(
            "No matching GHI / Actual data found."
        )

    result = pd.DataFrame()

    for col in GHI_COLUMNS:

        result[col] = (
            numeric_series(
                df_ghi[col]
            )
            .iloc[:n]
            .to_numpy()
        )

    result["Actual"] = (
        numeric_series(
            df_fixed["Actual"]
        )
        .iloc[:n]
        .to_numpy()
    )

    return result.reset_index(drop=True)


# ============================================================
# APPLY EDITED INPUT DATA
# ============================================================

def apply_input_dataframe(
    input_df,
    df_ghi,
    df_fixed,
):

    if input_df is None:

        raise ValueError(
            "Input dataframe is missing."
        )

    n = len(input_df)

    if n == 0:

        raise ValueError(
            "Edited input dataframe is empty."
        )

    required = GHI_COLUMNS + ["Actual"]

    missing = [
        col
        for col in required
        if col not in input_df.columns
    ]

    if missing:

        raise ValueError(
            "Missing input columns: "
            + ", ".join(missing)
        )

    if n > len(df_ghi):

        raise ValueError(
            "Edited input contains more rows "
            "than the original GHI dataset."
        )

    if n > len(df_fixed):

        raise ValueError(
            "Edited input contains more rows "
            "than the original Actual dataset."
        )

    df_ghi = df_ghi.iloc[:n].copy()

    df_fixed = df_fixed.iloc[:n].copy()

    for col in GHI_COLUMNS:

        df_ghi[col] = (
            numeric_series(
                input_df[col]
            )
            .to_numpy()
        )

    df_fixed["Actual"] = (
        numeric_series(
            input_df["Actual"]
        )
        .to_numpy()
    )

    return (
        df_ghi.reset_index(drop=True),
        df_fixed.reset_index(drop=True),
    )


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def prepare_fixed_geometry(
    df_fix,
    df_ghi,
    lat,
    month_lookup,
):

    df_fix = df_fix.copy()

    n = min(
        len(df_fix),
        len(df_ghi),
    )

    if n <= 0:

        raise ValueError(
            "No data available for solar geometry."
        )

    df_fix = df_fix.iloc[:n].copy()

    df_ghi = df_ghi.iloc[:n].copy()

    # Preserve existing calculation logic.
    today = pd.Timestamp.today()

    df_fix["Date"] = today

    first_date = (
        today
        .replace(
            month=1,
            day=1,
        )
        .normalize()
    )

    day_number = (
        (
            df_fix["Date"]
            - first_date
        ).dt.days
        + 1
    )

    df_fix["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_number
                )
                / 365
            )
        )
    )

    df_fix["Elevation angle a"] = (
        90
        - lat
        + df_fix["Declination Angle ∆"]
    )

    df_fix["Tilt Angle b"] = (
        df_fix["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

    df_fix["Tilt Angle b"] = (
        pd.to_numeric(
            df_fix["Tilt Angle b"],
            errors="coerce",
        )
        .fillna(0)
    )

    df_fix["a+b"] = (
        df_fix["Elevation angle a"]
        + df_fix["Tilt Angle b"]
    )

    df_fix["SIN(a+b)"] = np.sin(
        np.radians(
            df_fix["a+b"]
        )
    )

    df_fix["Sin(a)"] = np.sin(
        np.radians(
            df_fix["Elevation angle a"]
        )
    )

    sin_a = (
        df_fix["Sin(a)"]
        .replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # C11
    # --------------------------------------------------------

    df_fix["GHI*sin(a)"] = (
        df_ghi["GHI C11"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)"] = (
        df_ghi["GHI C11"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA fixed"] = (
        df_fix["GHI*sin(a+b)"]
        / sin_a
    )

    # --------------------------------------------------------
    # C12
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL2"] = (
        df_ghi["GHI C12"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)-CL2"] = (
        df_ghi["GHI C12"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA Fixed-C12"] = (
        df_fix["GHI*sin(a+b)-CL2"]
        / sin_a
    )

    # --------------------------------------------------------
    # C13
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL3"] = (
        df_ghi["GHI C13"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)-CL3"] = (
        df_ghi["GHI C13"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA Fixed-C13"] = (
        df_fix["GHI*sin(a+b)-CL3"]
        / sin_a
    )

    # --------------------------------------------------------
    # C14
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL4"] = (
        df_ghi["GHI C14"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)-CL4"] = (
        df_ghi["GHI C14"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA Fixed-C14"] = (
        df_fix["GHI*sin(a+b)-CL4"]
        / sin_a
    )

    # --------------------------------------------------------
    # C15
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL5"] = (
        df_ghi["GHI C15"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)-CL5"] = (
        df_ghi["GHI C15"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA Fixed-C15"] = (
        df_fix["GHI*sin(a+b)-CL5"]
        / sin_a
    )

    return df_fix.reset_index(drop=True)


# ============================================================
# EFFECTIVE AREA
# ============================================================

def calculate_effective_area(
    df_original,
    df_w_original,
    error,
):

    df = df_original.copy()

    df_w = df_w_original.copy()

    error = safe_float(
        error,
        default=0,
    )

    df["Error %"] = error

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - error
    )

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

    cluster_sums = (
        df.groupby(
            "Clusters"
        )["Eff Area"]
        .sum()
    )

    if "Clusters" not in df_w.columns:

        raise ValueError(
            "Clusters column not found in cluster table."
        )

    df_w["Eff Area(m2)"] = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0)
    )

    return (
        df,
        df_w,
    )


# ============================================================
# FIXED POWER
# ============================================================

def calculate_fixed_power(
    df_fix,
    df_w,
):

    result = df_fix.copy()

    if len(df_w) < 5:

        raise ValueError(
            "Cluster table must contain at least 5 rows."
        )

    for i, poa_col in enumerate(
        FIXED_POA_COLUMNS
    ):

        if poa_col not in result.columns:

            raise ValueError(
                f"Missing POA column: {poa_col}"
            )

        area_value = safe_float(
            df_w.iloc[i]["Eff Area(m2)"],
            default=0,
        )

        result[
            POWER_COLUMNS[i]
        ] = (
            numeric_series(
                result[poa_col]
            )
            * area_value
            / 1_000_000
        )

    result[TOTAL_POWER_COLUMN] = (
        result[POWER_COLUMNS]
        .sum(axis=1)
    )

    return result


# ============================================================
# AUTOMATIC ERROR OPTIMIZATION
#
# EXPENSIVE OPERATION
#
# Runs ONLY from Run Automatic Calculation.
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_error_cached(
    df_original,
    df_w_original,
    df_fix,
):

    actual = numeric_array(
        df_fix["Actual"]
    )

    if len(actual) == 0:

        raise ValueError(
            "Actual dataset is empty."
        )

    actual_peak = actual.max()

    if actual_peak <= 0:

        raise ValueError(
            "No non-zero Actual values found."
        )

    results = []

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        _, df_w = (
            calculate_effective_area(
                df_original,
                df_w_original,
                error,
            )
        )

        calculated = (
            calculate_fixed_power(
                df_fix,
                df_w,
            )
        )

        forecast = numeric_array(
            calculated[
                TOTAL_POWER_COLUMN
            ]
        )

        if len(forecast) == 0:
            continue

        calculated_peak = forecast.max()

        peak_error = abs(
            calculated_peak
            - actual_peak
        )

        peak_error_pct = (
            peak_error
            / actual_peak
            * 100
        )

        results.append(
            {
                "Error %": round(
                    error,
                    1,
                ),
                "Calculated Peak":
                    calculated_peak,
                "Actual Peak":
                    actual_peak,
                "Peak Error":
                    peak_error,
                "Peak Error %":
                    peak_error_pct,
            }
        )

    if not results:

        raise ValueError(
            "Error optimization produced no valid results."
        )

    result_df = pd.DataFrame(
        results
    )

    best_row = result_df.loc[
        result_df[
            "Peak Error"
        ].idxmin()
    ]

    return (
        float(
            best_row["Error %"]
        ),
        result_df,
    )


# ============================================================
# TRACKING DATA
# ============================================================

def prepare_tracking_data(
    backend,
):

    if "C11" not in backend:

        raise ValueError(
            "Backend Cal C11 sheet not found."
        )

    df_backend = (
        backend["C11"].copy()
    )

    if "Block No." not in df_backend.columns:

        raise ValueError(
            "Block No. column not found in Backend Cal C11."
        )

    blocks = numeric_array(
        df_backend["Block No."]
    )

    if len(blocks) == 0:

        raise ValueError(
            "No tracking block data found."
        )

    return blocks


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

    if denominator_1 == 0:
        return None

    if denominator_2 == 0:
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

        np.minimum(
            zenith,
            abs(east_limit),
        ),

        np.where(
            (
                (blocks > max_block)
                &
                (
                    zenith
                    > west_limit
                )
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
        tracking_power_matrix
        .sum(axis=1)
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
#
# EXPENSIVE OPERATION
#
# Runs ONLY from Run button.
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def optimize_tracking_cached(
    blocks_tuple,
    ghi_matrix_tuple,
    actual_tuple,
    tracking_weights_tuple,
):

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    ghi_matrix = np.asarray(
        ghi_matrix_tuple,
        dtype=float,
    )

    actual_full = np.asarray(
        actual_tuple,
        dtype=float,
    )

    tracking_weights = np.asarray(
        tracking_weights_tuple,
        dtype=float,
    )

    n = min(
        len(blocks),
        len(ghi_matrix),
        len(actual_full),
    )

    if n <= 0:

        raise ValueError(
            "No tracking data available."
        )

    blocks = blocks[:n]

    ghi_matrix = ghi_matrix[:n]

    actual_full = actual_full[:n]

    valid_mask = (
        actual_full != 0
    )

    if not valid_mask.any():

        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual_day = (
        actual_full[
            valid_mask
        ]
    )

    actual_peak = actual_day.max()

    actual_energy = actual_day.sum()

    if actual_peak <= 0:

        raise ValueError(
            "Actual peak is zero for Tracking."
        )

    if actual_energy == 0:

        raise ValueError(
            "Actual energy is zero for Tracking."
        )

    def objective(x):

        DHI = int(round(x[0]))

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

        if not np.all(
            np.isfinite(prediction)
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

    result = differential_evolution(
        objective,
        bounds=TRACKING_BOUNDS,
        strategy="best1bin",
        maxiter=40,
        popsize=15,
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
        updating="immediate",
    )

    best = np.rint(
        result.x
    ).astype(int)

    parameters = {
        "DHI":
            int(best[0]),

        "GHI Starting Block":
            int(best[1]),

        "GHI Ending Block":
            int(best[2]),

        "GHI Max Block":
            int(best[3]),

        "Tracking East Limit":
            int(best[4]),

        "Tracking West Limit":
            int(best[5]),
    }

    return parameters


# ============================================================
# TRACKING WEIGHTS
# ============================================================

def get_tracking_weights(
    df_w,
):

    if "Eff Area(m2)" in df_w.columns:

        values = (
            df_w[
                "Eff Area(m2)"
            ]
            .iloc[:5]
        )

    elif len(df_w.columns) >= 2:

        values = (
            df_w
            .iloc[:5, 1]
        )

    else:

        raise ValueError(
            "Unable to determine tracking weights."
        )

    return (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )


# ============================================================
# PREPARE TRACKING ARRAYS
# ============================================================

def prepare_tracking_arrays(
    backend,
    df_ghi,
    df_fix,
    df_w,
):

    blocks = (
        prepare_tracking_data(
            backend
        )
    )

    n = min(
        len(blocks),
        len(df_ghi),
        len(df_fix),
    )

    if n <= 0:

        raise ValueError(
            "No matching Tracking data found."
        )

    blocks = blocks[:n]

    ghi_matrix = np.column_stack(
        [
            numeric_array(
                df_ghi[col]
            )[:n]
            for col in GHI_COLUMNS
        ]
    )

    actual = (
        numeric_array(
            df_fix["Actual"]
        )[:n]
    )

    tracking_weights = (
        get_tracking_weights(
            df_w
        )
    )

    if len(tracking_weights) < 5:

        raise ValueError(
            "Tracking requires five cluster weights."
        )

    tracking_weights = (
        tracking_weights[:5]
    )

    return (
        blocks,
        ghi_matrix,
        actual,
        tracking_weights,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    forecast,
):

    actual = numeric_array(
        actual
    )

    forecast = numeric_array(
        forecast
    )

    n = min(
        len(actual),
        len(forecast),
    )

    if n == 0:

        return {
            "Actual Peak": 0.0,
            "Forecast Peak": 0.0,
            "Peak Error": 0.0,
            "Peak Error %": 0.0,
        }

    actual = actual[:n]

    forecast = forecast[:n]

    actual_peak = float(
        actual.max()
    )

    forecast_peak = float(
        forecast.max()
    )

    peak_error = abs(
        forecast_peak
        - actual_peak
    )

    peak_error_pct = (
        peak_error
        / actual_peak
        * 100
        if actual_peak > 0
        else 0
    )

    return {
        "Actual Peak":
            actual_peak,

        "Forecast Peak":
            forecast_peak,

        "Peak Error":
            peak_error,

        "Peak Error %":
            peak_error_pct,
    }


# ============================================================
# GRAPH
# ============================================================

def build_graph(
    actual,
    forecast,
    title,
):

    actual = numeric_array(
        actual
    )

    forecast = numeric_array(
        forecast
    )

    n = min(
        len(actual),
        len(forecast),
    )

    actual = actual[:n]

    forecast = forecast[:n]

    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(
                width=2.2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                width=2.2,
            ),
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.01,
        },
        height=430,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(
            l=30,
            r=30,
            t=55,
            b=30,
        ),
        xaxis=dict(
            title="Block",
            dtick=5,
        ),
        yaxis=dict(
            title="Power",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
    )

    return fig


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📂 Input Data</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Solar Excel File",
    type=[
        "xlsx",
        "xls",
    ],
    label_visibility="collapsed",
)


if uploaded_file is None:

    st.info(
        "Upload the Solar Excel file to start."
    )

    st.stop()


# ============================================================
# FILE CHANGE DETECTION
# ============================================================

current_hash = file_hash(
    uploaded_file
)

previous_hash = (
    st.session_state.get(
        "last_file_hash"
    )
)


if previous_hash != current_hash:

    reset_file_state()

    st.session_state.last_file_hash = (
        current_hash
    )


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    file_bytes = (
        uploaded_file.getvalue()
    )

    workbook = load_workbook(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


# ============================================================
# PREPARE RAW DATA
# ============================================================

try:

    df_original = (
        prepare_area_efficiency(
            workbook["area"]
        )
    )

    df_w_original = (
        prepare_cluster_table(
            workbook["cluster"]
        )
    )

    df_ghi_raw = (
        prepare_ghi(
            workbook["ghi"]
        )
    )

    lat = (
        prepare_latitude(
            workbook["forecast_config"]
        )
    )

    month_lookup = (
        prepare_tilt(
            workbook["tilt"]
        )
    )

    df_fix_raw = (
        prepare_fixed(
            workbook["fixed"]
        )
    )

except Exception as e:

    st.error(
        f"Input preparation failed: {e}"
    )

    st.stop()


# ============================================================
# INITIALIZE INPUT DATA
# ============================================================

if st.session_state.input_df is None:

    try:

        st.session_state.input_df = (
            build_input_dataframe(
                df_ghi_raw,
                df_fix_raw,
            )
        )

    except Exception as e:

        st.error(
            f"Unable to build input data: {e}"
        )

        st.stop()


# ============================================================
# INPUT DATA EDITOR
# ============================================================

st.markdown(
    '<div class="section-title">📝 GHI / Actual Input</div>',
    unsafe_allow_html=True,
)

input_df = st.data_editor(
    st.session_state.input_df,
    width="stretch",
    height=270,
    num_rows="fixed",
    hide_index=True,
    key="solar_input_editor",
    column_config={
        col: st.column_config.NumberColumn(
            col,
            format="%.3f",
        )
        for col in GHI_COLUMNS
    }
    | {
        "Actual": st.column_config.NumberColumn(
            "Actual",
            format="%.3f",
        )
    },
)


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">🌱 Plant Type</div>',
    unsafe_allow_html=True,
)


if (
    st.session_state.get(
        "plant_type"
    )
    not in PLANT_OPTIONS
):

    st.session_state.plant_type = "Fixed"


plant_type = st.segmented_control(
    "Plant Type",
    options=PLANT_OPTIONS,
    default=st.session_state.plant_type,
    selection_mode="single",
    width="stretch",
    label_visibility="collapsed",
    key="plant_type_selector",
)


if plant_type is None:

    plant_type = (
        st.session_state.plant_type
    )


if plant_type not in PLANT_OPTIONS:

    plant_type = "Fixed"


# ============================================================
# HANDLE PLANT TYPE CHANGE
#
# IMPORTANT:
# Changing plant type invalidates calculation because the
# calculation belongs to a different model.
#
# Parameter changes do NOT come here.
# ============================================================

if plant_type != st.session_state.plant_type:

    st.session_state.plant_type = (
        plant_type
    )

    reset_calculation_state()

    # Stop this run so the next rerun starts cleanly.
    st.rerun()


# ============================================================
# RUN AUTOMATIC CALCULATION BUTTON
# ============================================================

st.markdown("")

run_clicked = st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    width="stretch",
)

st.markdown(
    '<div class="run-note">'
    "Automatic optimization runs only when this button is clicked. "
    "Changing parameters afterwards does not rerun optimization."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

if run_clicked:

    try:

        with st.spinner(
            "Running automatic optimization..."
        ):

            # ------------------------------------------------
            # Save current edited input
            # ------------------------------------------------

            st.session_state.input_df = (
                input_df.copy()
            )

            # ------------------------------------------------
            # Apply GHI / Actual edits
            # ------------------------------------------------

            (
                df_ghi,
                df_fix_raw_user,
            ) = apply_input_dataframe(
                input_df,
                df_ghi_raw,
                df_fix_raw,
            )

            # ------------------------------------------------
            # Solar geometry
            # ------------------------------------------------

            df_fix = (
                prepare_fixed_geometry(
                    df_fix_raw_user,
                    df_ghi,
                    lat,
                    month_lookup,
                )
            )

            # ------------------------------------------------
            # Optimize Error %
            # ------------------------------------------------

            (
                best_error,
                error_results,
            ) = optimize_error_cached(
                df_original,
                df_w_original,
                df_fix,
            )

            # ------------------------------------------------
            # Apply optimized Error %
            # ------------------------------------------------

            (
                df_final,
                df_w_final,
            ) = calculate_effective_area(
                df_original,
                df_w_original,
                best_error,
            )

            # ------------------------------------------------
            # Fixed forecast
            # ------------------------------------------------

            fixed_final = (
                calculate_fixed_power(
                    df_fix,
                    df_w_final,
                )
            )

            # ------------------------------------------------
            # Tracking
            # ------------------------------------------------

            tracking_parameters = None
            tracking_forecast = None

            if plant_type == "Tracking":

                (
                    blocks,
                    ghi_matrix,
                    actual_tracking,
                    tracking_weights,
                ) = prepare_tracking_arrays(
                    workbook["backend"],
                    df_ghi,
                    df_fix,
                    df_w_final,
                )

                # --------------------------------------------
                # Optimize tracking parameters
                # --------------------------------------------

                tracking_parameters = (
                    optimize_tracking_cached(
                        tuple(
                            blocks.tolist()
                        ),
                        tuple(
                            map(
                                tuple,
                                ghi_matrix.tolist(),
                            )
                        ),
                        tuple(
                            actual_tracking.tolist()
                        ),
                        tuple(
                            tracking_weights.tolist()
                        ),
                    )
                )

                # --------------------------------------------
                # Validate optimized parameters
                # --------------------------------------------

                if not (
                    tracking_parameters[
                        "GHI Starting Block"
                    ]
                    <
                    tracking_parameters[
                        "GHI Max Block"
                    ]
                    <
                    tracking_parameters[
                        "GHI Ending Block"
                    ]
                ):

                    raise ValueError(
                        "Automatic tracking optimization "
                        "returned invalid block parameters."
                    )

                # --------------------------------------------
                # Calculate optimized tracking forecast
                # --------------------------------------------

                tracking_result = (
                    calculate_tracking(
                        tracking_parameters[
                            "DHI"
                        ],
                        tracking_parameters[
                            "GHI Starting Block"
                        ],
                        tracking_parameters[
                            "GHI Ending Block"
                        ],
                        tracking_parameters[
                            "GHI Max Block"
                        ],
                        tracking_parameters[
                            "Tracking East Limit"
                        ],
                        tracking_parameters[
                            "Tracking West Limit"
                        ],
                        blocks,
                        ghi_matrix,
                        tracking_weights,
                    )
                )

                if tracking_result is None:

                    raise ValueError(
                        "Automatic tracking parameters "
                        "produced an invalid result."
                    )

                tracking_forecast = (
                    tracking_result[0]
                )

            # ------------------------------------------------
            # Store expensive calculation results
            # ------------------------------------------------

            st.session_state.calculation_data = {

                "df_original":
                    df_original,

                "df_w_original":
                    df_w_original,

                "df_final":
                    df_final,

                "df_w_final":
                    df_w_final,

                "df_ghi":
                    df_ghi,

                "df_fix":
                    df_fix,

                "fixed_final":
                    fixed_final,

                "best_error":
                    best_error,

                "error_results":
                    error_results,

                "tracking_parameters":
                    tracking_parameters,

                "tracking_forecast":
                    tracking_forecast,
            }

            # ------------------------------------------------
            # Initialize editable parameter state
            # ------------------------------------------------

            st.session_state.error_value = (
                float(best_error)
            )

            if tracking_parameters is not None:

                st.session_state.tracking_params = {
                    key: int(value)
                    for key, value
                    in tracking_parameters.items()
                }

            else:

                st.session_state.tracking_params = None

            # ------------------------------------------------
            # Mark calculation as complete
            # ------------------------------------------------

            st.session_state.calculated = True

            st.session_state.calculated_plant_type = (
                plant_type
            )

            st.session_state.forecast_ready = True

        st.success(
            "Automatic calculation completed successfully."
        )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )

        st.stop()


# ============================================================
# WAIT FOR AUTOMATIC CALCULATION
# ============================================================

if not st.session_state.calculated:

    st.info(
        "Edit GHI / Actual values, select the plant type, "
        "then click Run Automatic Calculation."
    )

    st.stop()


# ============================================================
# PROTECT AGAINST PLANT-TYPE MISMATCH
# ============================================================

if (
    st.session_state.get(
        "calculated_plant_type"
    )
    != plant_type
):

    reset_calculation_state()

    st.warning(
        "Plant type changed. "
        "Run Automatic Calculation again."
    )

    st.stop()


# ============================================================
# RESULTS DATA
# ============================================================

data = (
    st.session_state.get(
        "calculation_data"
    )
)


if data is None:

    reset_calculation_state()

    st.warning(
        "Calculation results are unavailable. "
        "Please run Automatic Calculation again."
    )

    st.stop()


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="parameter-note">'
    "You can change these parameters directly. "
    "The forecast and graph update automatically."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# ERROR %
#
# IMPORTANT:
# This widget updates only the current forecast.
# It NEVER calls differential_evolution.
# ============================================================

error_key = (
    f"error_widget_{current_hash}"
)


if (
    st.session_state.error_value
    is None
):

    st.session_state.error_value = (
        float(data["best_error"])
    )


error_value = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=20.0,
    step=0.1,
    format="%.1f",
    key=error_key,
)


# Keep session state synchronized with widget.
st.session_state.error_value = (
    float(error_value)
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

if plant_type == "Tracking":

    params = (
        st.session_state.get(
            "tracking_params"
        )
    )

    if params is None:

        params = {
            key: int(value)
            for key, value
            in data[
                "tracking_parameters"
            ].items()
        }

        st.session_state.tracking_params = (
            params
        )

    st.markdown(
        "#### Tracking Parameters"
    )

    c1, c2, c3 = st.columns(3)

    # --------------------------------------------------------
    # DHI
    # --------------------------------------------------------

    with c1:

        dhi_key = (
            f"tracking_dhi_{current_hash}"
        )

        if dhi_key not in st.session_state:

            st.session_state[dhi_key] = (
                int(
                    params["DHI"]
                )
            )

        dhi_value = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            step=1,
            key=dhi_key,
        )

        # ----------------------------------------------------
        # START BLOCK
        # ----------------------------------------------------

        start_key = (
            f"tracking_start_{current_hash}"
        )

        if start_key not in st.session_state:

            st.session_state[start_key] = (
                int(
                    params[
                        "GHI Starting Block"
                    ]
                )
            )

        start_value = st.number_input(
            "GHI Starting Block",
            min_value=0,
            max_value=95,
            step=1,
            key=start_key,
        )

    # --------------------------------------------------------
    # END / MAX
    # --------------------------------------------------------

    with c2:

        end_key = (
            f"tracking_end_{current_hash}"
        )

        if end_key not in st.session_state:

            st.session_state[end_key] = (
                int(
                    params[
                        "GHI Ending Block"
                    ]
                )
            )

        end_value = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            step=1,
            key=end_key,
        )

        max_key = (
            f"tracking_max_{current_hash}"
        )

        if max_key not in st.session_state:

            st.session_state[max_key] = (
                int(
                    params[
                        "GHI Max Block"
                    ]
                )
            )

        max_value = st.number_input(
            "GHI Max Block",
            min_value=0,
            max_value=95,
            step=1,
            key=max_key,
        )

    # --------------------------------------------------------
    # EAST / WEST
    # --------------------------------------------------------

    with c3:

        east_key = (
            f"tracking_east_{current_hash}"
        )

        if east_key not in st.session_state:

            st.session_state[east_key] = (
                int(
                    params[
                        "Tracking East Limit"
                    ]
                )
            )

        east_value = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            step=1,
            key=east_key,
        )

        west_key = (
            f"tracking_west_{current_hash}"
        )

        if west_key not in st.session_state:

            st.session_state[west_key] = (
                int(
                    params[
                        "Tracking West Limit"
                    ]
                )
            )

        west_value = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            step=1,
            key=west_key,
        )

    # --------------------------------------------------------
    # Synchronize editable tracking state
    # --------------------------------------------------------

    st.session_state.tracking_params = {

        "DHI":
            int(dhi_value),

        "GHI Starting Block":
            int(start_value),

        "GHI Ending Block":
            int(end_value),

        "GHI Max Block":
            int(max_value),

        "Tracking East Limit":
            int(east_value),

        "Tracking West Limit":
            int(west_value),
    }


# ============================================================
# CHEAP FINAL FORECAST CALCULATION
#
# THIS SECTION RUNS ON EVERY STREAMLIT RERUN.
#
# IT DOES NOT RUN DIFFERENTIAL EVOLUTION.
# ============================================================

try:

    # --------------------------------------------------------
    # APPLY CURRENT ERROR %
    # --------------------------------------------------------

    (
        df_final,
        df_w_final,
    ) = calculate_effective_area(
        data["df_original"],
        data["df_w_original"],
        error_value,
    )

    # --------------------------------------------------------
    # FIXED FORECAST
    # --------------------------------------------------------

    fixed_final = (
        calculate_fixed_power(
            data["df_fix"],
            df_w_final,
        )
    )

    # --------------------------------------------------------
    # TRACKING FORECAST
    # --------------------------------------------------------

    if plant_type == "Tracking":

        (
            blocks,
            ghi_matrix,
            actual_tracking,
            tracking_weights,
        ) = prepare_tracking_arrays(
            workbook["backend"],
            data["df_ghi"],
            data["df_fix"],
            df_w_final,
        )

        # ----------------------------------------------------
        # Validate tracking parameters
        # ----------------------------------------------------

        if not (
            start_value
            < max_value
            < end_value
        ):

            st.error(
                "Tracking parameters must satisfy: "
                "GHI Starting Block < "
                "GHI Max Block < "
                "GHI Ending Block."
            )

            st.stop()

        # ----------------------------------------------------
        # Cheap tracking calculation
        # ----------------------------------------------------

        tracking_result = (
            calculate_tracking(
                int(dhi_value),
                int(start_value),
                int(end_value),
                int(max_value),
                int(east_value),
                int(west_value),
                blocks,
                ghi_matrix,
                tracking_weights,
            )
        )

        if tracking_result is None:

            st.error(
                "Invalid tracking parameters."
            )

            st.stop()

        tracking_forecast = (
            tracking_result[0]
        )

        actual = (
            actual_tracking
        )

        forecast = (
            tracking_forecast
        )

        graph_title = (
            "Tracking Plant | Actual vs Forecast"
        )

    # --------------------------------------------------------
    # FIXED FORECAST
    # --------------------------------------------------------

    else:

        actual = numeric_array(
            data["df_fix"]["Actual"]
        )

        forecast = numeric_array(
            fixed_final[
                TOTAL_POWER_COLUMN
            ]
        )

        graph_title = (
            "Fixed Plant | Actual vs Forecast"
        )


except Exception as e:

    st.error(
        f"Forecast calculation failed: {e}"
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

metrics = calculate_metrics(
    actual,
    forecast,
)


st.markdown(
    '<div class="section-title">📊 Results</div>',
    unsafe_allow_html=True,
)


m1, m2, m3, m4 = st.columns(4)


# ============================================================
# ACTUAL PEAK
# ============================================================

with m1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Actual Peak
            </div>
            <div class="metric-value">
                {metrics["Actual Peak"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FORECAST PEAK
# ============================================================

with m2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Forecast Peak
            </div>
            <div class="metric-value">
                {metrics["Forecast Peak"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PEAK ERROR
# ============================================================

with m3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Peak Error
            </div>
            <div class="metric-value">
                {metrics["Peak Error"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PEAK ERROR %
# ============================================================

with m4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Peak Error %
            </div>
            <div class="metric-value">
                {metrics["Peak Error %"]:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FORECAST GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)


fig = build_graph(
    actual,
    forecast,
    graph_title,
)


st.plotly_chart(
    fig,
    width="stretch",
    config={
        "displayModeBar": False,
        "responsive": True,
    },
)
