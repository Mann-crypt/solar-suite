# ============================================================
# LOSS CORRECTION VCAST
# FIXED / TRACKING
# CLEAN + OPTIMIZED STREAMLIT VERSION
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
    page_title="Solar Loss Correction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .main-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .sub-title {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .metric-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .section-title {
        font-size: 20px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

GHI_COLUMNS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

POWER_COLUMNS = [
    "CL1_Fixed Power=I*Ƞ*A",
    "CL2_Fixed Power=I*Ƞ*A",
    "CL3_Fixed Power=I*Ƞ*A",
    "CL4_Fixed Power=I*Ƞ*A",
    "CL5_Fixed Power=I*Ƞ*A",
]

DEFAULT_ERROR_MIN = 0.0
DEFAULT_ERROR_MAX = 10.0
DEFAULT_ERROR_STEP = 0.1


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "plant_type": "Fixed",
    "error_min": 0.0,
    "error_max": 10.0,
    "error_step": 0.1,

    "best_error": None,

    "DHI": 1,
    "GHI_Starting_Block": 30,
    "GHI_Ending_Block": 79,
    "GHI_Max_Block": 53,
    "Tracking_angle_lim_E": 11,
    "Tracking_angle_lim_W": 23,

    "calculated": False,
    "optimization_done": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">☀️ Solar Loss Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    "Fixed / Tracking plant calculation and optimization"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def clean_columns(df):
    """Clean Excel column names."""

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    return df


def trim_at_first_null(df, column):
    """Trim dataframe at first null in a specified column."""

    df = df.copy()

    if column not in df.columns:
        return df

    null_idx = df[df[column].isna()].index

    if len(null_idx) > 0:
        first_pos = df.index.get_loc(null_idx[0])
        df = df.iloc[:first_pos]

    return df.reset_index(drop=True)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


# ============================================================
# EXCEL LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_excel(file_bytes):

    excel = pd.ExcelFile(io.BytesIO(file_bytes))

    sheets = excel.sheet_names

    return sheets


@st.cache_data(show_spinner=False)
def read_sheet(file_bytes, sheet_name, **kwargs):

    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
        **kwargs,
    )


# ============================================================
# AREA / EFFICIENCY
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_area_data(file_bytes):

    df = read_sheet(
        file_bytes,
        "Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df = clean_columns(df)

    df = trim_at_first_null(df, "S.No.")

    # --------------------------------------------------------
    # Ensure required columns are numeric
    # --------------------------------------------------------

    numeric_columns = [
        "Standard PV Efficiency (%)",
        "Error %",
        "Total area (m2)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Total area
    # --------------------------------------------------------

    if (
        "No of Module" in df.columns
        and "Area of 1 Module (m2)" in df.columns
    ):

        df["Total area (m2)"] = (
            df["No of Module"]
            * df["Area of 1 Module (m2)"]
        )

    return df


@st.cache_data(show_spinner=False)
def prepare_cluster_table(file_bytes):

    df_w = read_sheet(
        file_bytes,
        "Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df_w = clean_columns(df_w)

    df_w = trim_at_first_null(
        df_w,
        "Clusters",
    )

    return df_w.reset_index(drop=True)


# ============================================================
# EFFECTIVE AREA
# ============================================================

def calculate_effective_area(
    df_area,
    df_cluster,
    error_percent,
):
    """
    IMPORTANT:

    Error % is applied exactly ONCE here.

    Standard Efficiency
            ↓
    Standard - Error
            ↓
    Net Efficiency
            ↓
    Effective Area
    """

    df = df_area.copy()
    df_w = df_cluster.copy()

    error_percent = float(error_percent)

    # --------------------------------------------------------
    # ERROR APPLIED ONCE
    # --------------------------------------------------------

    df["Error %"] = error_percent

    df["Net Efficiency (%)"] = (
        pd.to_numeric(
            df["Standard PV Efficiency (%)"],
            errors="coerce",
        )
        - error_percent
    )

    # --------------------------------------------------------
    # TOTAL AREA
    # --------------------------------------------------------

    df["Total area (m2)"] = (
        pd.to_numeric(
            df["No of Module"],
            errors="coerce",
        ).fillna(0)
        *
        pd.to_numeric(
            df["Area of 1 Module (m2)"],
            errors="coerce",
        ).fillna(0)
    )

    # --------------------------------------------------------
    # EFFECTIVE AREA
    # --------------------------------------------------------

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100.0
    )

    # --------------------------------------------------------
    # CLUSTER SUM
    # --------------------------------------------------------

    cluster_sums = (
        df.groupby(
            "Clusters",
            dropna=False,
        )["Eff Area"]
        .sum()
    )

    df_w["Eff Area(m2)"] = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0.0)
    )

    return df, df_w


# ============================================================
# GHI INPUT
# ============================================================

def prepare_ghi_input(df_ghi):

    df = df_ghi.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    missing = [
        col
        for col in GHI_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing GHI columns: "
            + ", ".join(missing)
        )

    for col in GHI_COLUMNS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0.0)

    return df


# ============================================================
# ACTUAL INPUT
# ============================================================

def prepare_actual(df_actual):

    df = df_actual.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "Actual" not in df.columns:

        # Allow a one-column dataframe
        if len(df.columns) == 1:

            df = df.rename(
                columns={
                    df.columns[0]: "Actual"
                }
            )

        else:

            raise ValueError(
                "Actual input must contain "
                "'Actual' column."
            )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0.0)

    return df


# ============================================================
# FIXED ANGLE CALCULATION
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_fixed_geometry(
    ghi_df,
    lat,
    month_lookup,
):

    df_ghi = ghi_df.copy()

    n = len(df_ghi)

    dates = pd.Series(
        pd.Timestamp.today().normalize(),
        index=np.arange(n),
    )

    first_date = (
        pd.Timestamp.today()
        .replace(
            month=1,
            day=1,
        )
        .normalize()
    )

    day_number = (
        dates - first_date
    ).dt.days + 1

    declination = (
        23.45
        *
        np.sin(
            np.radians(
                360
                * (284 + day_number)
                / 365
            )
        )
    )

    elevation = (
        90
        - float(lat)
        + declination
    )

    month_name = dates.dt.strftime("%B")

    tilt = month_name.map(
        month_lookup
    )

    tilt = pd.to_numeric(
        tilt,
        errors="coerce",
    ).fillna(0.0)

    a_plus_b = (
        elevation + tilt
    )

    sin_a_plus_b = np.sin(
        np.radians(a_plus_b)
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    # Avoid divide-by-zero
    safe_sin_a = np.where(
        np.abs(sin_a) < 1e-10,
        np.nan,
        sin_a,
    )

    geometry = pd.DataFrame(
        {
            "Declination Angle ∆":
                declination.to_numpy(),

            "Elevation angle a":
                elevation.to_numpy(),

            "Tilt Angle b":
                tilt.to_numpy(),

            "a+b":
                a_plus_b.to_numpy(),

            "SIN(a+b)":
                sin_a_plus_b,

            "Sin(a)":
                sin_a,
        }
    )

    for i, ghi_col in enumerate(
        GHI_COLUMNS
    ):

        ghi = df_ghi[
            ghi_col
        ].to_numpy(
            dtype=float
        )

        suffix = (
            ""
            if i == 0
            else f"-CL{i + 1}"
        )

        geometry[
            f"GHI*sin(a){suffix}"
        ] = (
            ghi * sin_a
        )

        geometry[
            f"GHI*sin(a+b){suffix}"
        ] = (
            ghi * sin_a_plus_b
        )

        geometry[
            (
                "POA fixed"
                if i == 0
                else f"POA Fixed-C12"
                if i == 1
                else f"POA Fixed-C13"
                if i == 2
                else f"POA Fixed-C14"
                if i == 3
                else f"POA Fixed-C15"
            )
        ] = (
            ghi
            * sin_a_plus_b
            / safe_sin_a
        )

    return geometry


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    geometry,
    cluster_df,
):

    areas = (
        pd.to_numeric(
            cluster_df["Eff Area(m2)"],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=float
        )
    )

    forecast = np.zeros(
        len(geometry),
        dtype=float,
    )

    for i in range(5):

        if i == 0:
            poa_col = "POA fixed"
        elif i == 1:
            poa_col = "POA Fixed-C12"
        elif i == 2:
            poa_col = "POA Fixed-C13"
        elif i == 3:
            poa_col = "POA Fixed-C14"
        else:
            poa_col = "POA Fixed-C15"

        power = (
            pd.to_numeric(
                geometry[poa_col],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(
                dtype=float
            )
            * areas[i]
            / 1_000_000
        )

        forecast += power

    return forecast


# ============================================================
# TRACKING GEOMETRY
# ============================================================

def tracking_geometry(
    blocks,
    DHI,
    GHI_Starting_Block,
    GHI_Ending_Block,
    GHI_Max_Block,
    Tracking_angle_lim_E,
    Tracking_angle_lim_W,
):

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not (
        GHI_Starting_Block
        < GHI_Max_Block
        < GHI_Ending_Block
    ):
        return None, None

    denominator_1 = (
        GHI_Starting_Block
        - 1
        - GHI_Max_Block
    )

    denominator_2 = (
        GHI_Ending_Block
        + 1
        - GHI_Max_Block
    )

    if denominator_1 == 0:
        return None, None

    if denominator_2 == 0:
        return None, None

    # --------------------------------------------------------
    # Same equations as Jupyter
    # --------------------------------------------------------

    m1 = (
        90
        / denominator_1
    )

    m2 = (
        90
        / denominator_2
    )

    zenith = np.where(
        blocks <= GHI_Max_Block,

        np.minimum(
            89,
            m1
            * (
                blocks
                - GHI_Max_Block
            ),
        ),

        np.minimum(
            89,
            m2
            * (
                blocks
                - GHI_Max_Block
            ),
        ),
    )

    panel = np.where(
        blocks < GHI_Max_Block,

        np.minimum(
            zenith,
            abs(
                Tracking_angle_lim_E
            ),
        ),

        np.where(
            (
                (blocks > GHI_Max_Block)
                &
                (
                    zenith
                    > Tracking_angle_lim_W
                )
            ),

            Tracking_angle_lim_W,

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

    return zenith, panel, cos_alpha


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    ghi_matrix,
    cluster_areas,
    blocks,
    DHI,
    GHI_Starting_Block,
    GHI_Ending_Block,
    GHI_Max_Block,
    Tracking_angle_lim_E,
    Tracking_angle_lim_W,
):

    (
        zenith,
        panel,
        cos_alpha,
    ) = tracking_geometry(
        blocks,
        DHI,
        GHI_Starting_Block,
        GHI_Ending_Block,
        GHI_Max_Block,
        Tracking_angle_lim_E,
        Tracking_angle_lim_W,
    )

    if zenith is None:
        return None, None, None

    # --------------------------------------------------------
    # DHI
    # --------------------------------------------------------

    dhi = (
        ghi_matrix
        * float(DHI)
        / 100.0
    )

    # --------------------------------------------------------
    # DNI
    # --------------------------------------------------------

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    forecast = (
        dni
        @ cluster_areas
    ) / 1_000_000

    return (
        forecast,
        zenith,
        panel,
    )


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def make_tracking_objective(
    ghi_matrix,
    cluster_areas,
    blocks,
    actual,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    # --------------------------------------------------------
    # EXACTLY THE SAME MASK LOGIC
    # --------------------------------------------------------

    mask = actual != 0

    if not np.any(mask):
        raise ValueError(
            "No non-zero Actual values found."
        )

    actual_valid = actual[mask]

    actual_max = np.max(
        actual_valid
    )

    actual_sum = np.sum(
        actual_valid
    )

    if actual_max == 0:
        raise ValueError(
            "Actual peak is zero."
        )

    if actual_sum == 0:
        raise ValueError(
            "Actual energy is zero."
        )

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

        result = (
            calculate_tracking_forecast(
                ghi_matrix,
                cluster_areas,
                blocks,
                DHI,
                start_block,
                end_block,
                max_block,
                east_limit,
                west_limit,
            )
        )

        if result[0] is None:
            return 1e9

        prediction_full = result[0]

        if (
            np.isnan(
                prediction_full
            ).any()
            or
            np.isinf(
                prediction_full
            ).any()
        ):
            return 1e9

        prediction = (
            prediction_full[mask]
        )

        if len(prediction) == 0:
            return 1e9

        prediction_max = np.max(
            prediction
        )

        prediction_sum = np.sum(
            prediction
        )

        block_error = (
            np.mean(
                np.abs(
                    actual_valid
                    - prediction
                )
            )
            / actual_max
        )

        peak_error = (
            abs(
                actual_max
                - prediction_max
            )
            / actual_max
        )

        energy_error = (
            abs(
                actual_sum
                - prediction_sum
            )
            / actual_sum
        )

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return float(score)

    return objective


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
)
def optimize_tracking_cached(
    ghi_matrix,
    cluster_areas,
    blocks,
    actual,
):

    objective = (
        make_tracking_objective(
            ghi_matrix,
            cluster_areas,
            blocks,
            actual,
        )
    )

    bounds = [
        (0, 10),
        (10, 30),
        (65, 80),
        (47, 53),
        (10, 70),
        (10, 70),
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=40,
        popsize=15,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
        updating="immediate",
    )

    best = np.round(
        result.x
    ).astype(int)

    return (
        best,
        float(result.fun),
    )


# ============================================================
# FIXED ERROR OPTIMIZATION
# ============================================================

def optimize_fixed_error(
    df_area,
    df_cluster,
    geometry,
    actual,
    error_min,
    error_max,
    error_step,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    actual_peak = np.max(
        actual
    )

    if actual_peak == 0:
        raise ValueError(
            "Actual peak is zero."
        )

    errors = np.arange(
        error_min,
        error_max + error_step * 0.5,
        error_step,
    )

    best_error = None
    best_error_value = np.inf
    best_forecast = None
    results = []

    for error in errors:

        _, cluster_df = (
            calculate_effective_area(
                df_area,
                df_cluster,
                error,
            )
        )

        forecast = (
            calculate_fixed_forecast(
                geometry,
                cluster_df,
            )
        )

        calculated_peak = np.max(
            forecast
        )

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
                "Error %": error,
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

        if peak_error < best_error_value:

            best_error_value = (
                peak_error
            )

            best_error = (
                float(error)
            )

            best_forecast = (
                forecast.copy()
            )

    return (
        best_error,
        best_forecast,
        pd.DataFrame(results),
    )


# ============================================================
# MONTH / TILT LOADER
# ============================================================

@st.cache_data(show_spinner=False)
def load_tilt_data(file_bytes):

    df_tilt = read_sheet(
        file_bytes,
        "Config Tilt Angle",
        header=7,
    )

    df_tilt = clean_columns(
        df_tilt
    )

    df_tilt = trim_at_first_null(
        df_tilt,
        "Fixed",
    )

    df_tilt = df_tilt.dropna(
        how="all",
        axis=1,
    )

    rename_map = {}

    if "Unnamed: 2" in df_tilt.columns:
        rename_map["Unnamed: 2"] = "Month_Num"

    if "Unnamed: 3" in df_tilt.columns:
        rename_map["Unnamed: 3"] = "Month"

    df_tilt = df_tilt.rename(
        columns=rename_map
    )

    if (
        "Month" not in df_tilt.columns
        or "Fixed" not in df_tilt.columns
    ):
        raise ValueError(
            "Could not read Month/Fixed "
            "values from Config Tilt Angle."
        )

    df_tilt["Fixed"] = pd.to_numeric(
        df_tilt["Fixed"],
        errors="coerce",
    )

    return (
        df_tilt
        .dropna(
            subset=["Month"]
        )
        .set_index("Month")["Fixed"]
        .to_dict()
    )


# ============================================================
# LATITUDE
# ============================================================

@st.cache_data(show_spinner=False)
def load_latitude(file_bytes):

    df = read_sheet(
        file_bytes,
        "Forecast Config",
        header=8,
    )

    if "Lat" not in df.columns:
        raise ValueError(
            "'Lat' column not found "
            "in Forecast Config."
        )

    lat = pd.to_numeric(
        df.loc[0, "Lat"],
        errors="coerce",
    )

    if pd.isna(lat):
        raise ValueError(
            "Invalid latitude."
        )

    return float(lat)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">1. Input Data</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Excel file",
    type=["xlsx", "xls"],
    help=(
        "Upload the Excel workbook containing "
        "Area & Efficiency and configuration sheets."
    ),
)


if uploaded_file is None:

    st.info(
        "Upload your Excel file to start."
    )

    st.stop()


file_bytes = uploaded_file.getvalue()


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">2. Plant Type</div>',
    unsafe_allow_html=True,
)

plant_type = st.segmented_control(
    "Select plant type",
    options=[
        "Fixed",
        "Tracking",
    ],
    default=st.session_state.plant_type,
)

if plant_type is None:
    plant_type = "Fixed"

st.session_state.plant_type = (
    plant_type
)


# ============================================================
# USER GHI INPUT
# ============================================================

st.markdown(
    '<div class="section-title">'
    "3. GHI Forecast Input"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Enter GHI forecast values for C11-C15."
)

default_ghi = pd.DataFrame(
    columns=GHI_COLUMNS
)

edited_ghi = st.data_editor(
    default_ghi,
    num_rows="dynamic",
    width="stretch",
    key="ghi_editor",
)

# ------------------------------------------------------------
# Load GHI from Result sheet if editor is empty
# ------------------------------------------------------------

if edited_ghi.empty:

    try:

        auto_ghi = read_sheet(
            file_bytes,
            "Result",
            usecols=[0, 1, 2, 3, 4, 5],
        )

        auto_ghi = auto_ghi.fillna(0)

        auto_ghi.columns = (
            auto_ghi.columns
            .astype(str)
            .str.strip()
        )

        available = [
            col
            for col in GHI_COLUMNS
            if col in auto_ghi.columns
        ]

        if len(available) == 5:

            st.caption(
                "Using GHI values from Result sheet."
            )

            ghi_input = (
                auto_ghi[GHI_COLUMNS]
                .copy()
            )

        else:

            st.warning(
                "Enter GHI data above."
            )

            st.stop()

    except Exception as exc:

        st.error(
            f"Could not load GHI: {exc}"
        )

        st.stop()

else:

    try:
        ghi_input = prepare_ghi_input(
            edited_ghi
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()


# ============================================================
# USER ACTUAL INPUT
# ============================================================

st.markdown(
    '<div class="section-title">'
    "4. Actual Power Input"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Enter the Actual power column."
)

default_actual = pd.DataFrame(
    columns=["Actual"]
)

edited_actual = st.data_editor(
    default_actual,
    num_rows="dynamic",
    width="stretch",
    key="actual_editor",
)

if edited_actual.empty:

    try:

        auto_actual = read_sheet(
            file_bytes,
            "Fixed-C11",
            header=1,
        )

        auto_actual = clean_columns(
            auto_actual
        )

        auto_actual = trim_at_first_null(
            auto_actual,
            "Date",
        )

        if "Actual" not in auto_actual.columns:

            raise ValueError(
                "'Actual' column not found."
            )

        actual_input = (
            auto_actual[["Actual"]]
            .copy()
        )

        st.caption(
            "Using Actual values from Fixed-C11 sheet."
        )

    except Exception as exc:

        st.error(
            f"Could not load Actual: {exc}"
        )

        st.stop()

else:

    try:

        actual_input = (
            prepare_actual(
                edited_actual
            )
        )

    except Exception as exc:

        st.error(str(exc))

        st.stop()


# ============================================================
# LENGTH VALIDATION
# ============================================================

n_ghi = len(
    ghi_input
)

n_actual = len(
    actual_input
)

if n_ghi != n_actual:

    st.error(
        f"GHI rows ({n_ghi}) and "
        f"Actual rows ({n_actual}) "
        "must have the same length."
    )

    st.stop()


if n_ghi == 0:

    st.warning(
        "Enter GHI and Actual data."
    )

    st.stop()


# ============================================================
# LOAD EXCEL CONFIG
# ============================================================

try:

    df_area = (
        prepare_area_data(
            file_bytes
        )
    )

    df_cluster = (
        prepare_cluster_table(
            file_bytes
        )
    )

    lat = load_latitude(
        file_bytes
    )

    month_lookup = (
        load_tilt_data(
            file_bytes
        )
    )

except Exception as exc:

    st.error(
        f"Excel configuration error: {exc}"
    )

    st.stop()


# ============================================================
# ERROR RANGE
# ============================================================

st.markdown(
    '<div class="section-title">'
    "5. Optimization Settings"
    "</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:

    error_min = st.number_input(
        "Error % minimum",
        min_value=-100.0,
        max_value=100.0,
        value=float(
            st.session_state.error_min
        ),
        step=0.1,
        key="error_min_input",
    )

with c2:

    error_max = st.number_input(
        "Error % maximum",
        min_value=-100.0,
        max_value=100.0,
        value=float(
            st.session_state.error_max
        ),
        step=0.1,
        key="error_max_input",
    )

with c3:

    error_step = st.number_input(
        "Error % step",
        min_value=0.01,
        max_value=10.0,
        value=float(
            st.session_state.error_step
        ),
        step=0.1,
        key="error_step_input",
    )


if error_min > error_max:

    st.error(
        "Error minimum cannot be greater "
        "than Error maximum."
    )

    st.stop()


# ============================================================
# GHI MATRIX
# ============================================================

ghi_matrix = (
    ghi_input[GHI_COLUMNS]
    .to_numpy(
        dtype=float
    )
)

actual_array = (
    actual_input["Actual"]
    .to_numpy(
        dtype=float
    )
)

blocks = np.arange(
    len(ghi_input),
    dtype=float,
)


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    "6. Automatic Calculation"
    "</div>",
    unsafe_allow_html=True,
)


run_auto = st.button(
    "⚡ Run Automatic Optimization",
    type="primary",
    use_container_width=True,
)


if run_auto:

    # ========================================================
    # FIXED
    # ========================================================

    if plant_type == "Fixed":

        with st.spinner(
            "Optimizing Fixed plant..."
        ):

            try:

                geometry = (
                    prepare_fixed_geometry(
                        ghi_input,
                        lat,
                        month_lookup,
                    )
                )

                (
                    best_error,
                    best_forecast,
                    optimization_table,
                ) = optimize_fixed_error(
                    df_area,
                    df_cluster,
                    geometry,
                    actual_array,
                    error_min,
                    error_max,
                    error_step,
                )

                st.session_state.best_error = (
                    best_error
                )

                st.session_state.calculated = (
                    True
                )

                st.session_state.optimization_done = (
                    True
                )

                st.session_state.final_forecast = (
                    best_forecast
                )

                st.session_state.optimization_table = (
                    optimization_table
                )

            except Exception as exc:

                st.error(
                    f"Fixed optimization failed: {exc}"
                )

    # ========================================================
    # TRACKING
    # ========================================================

    else:

        with st.spinner(
            "Optimizing Tracking plant..."
        ):

            try:

                # ------------------------------------------------
                # FIRST optimize Error %
                # ------------------------------------------------
                #
                # Error is applied ONLY here.
                # The resulting cluster areas are then passed
                # into tracking optimization.
                # ------------------------------------------------

                best_error = None
                best_error_value = np.inf
                best_cluster_df = None

                errors = np.arange(
                    error_min,
                    error_max
                    + error_step * 0.5,
                    error_step,
                )

                # ------------------------------------------------
                # Calculate tracking objective for each Error %
                # ------------------------------------------------

                actual_valid_mask = (
                    actual_array != 0
                )

                if not np.any(
                    actual_valid_mask
                ):
                    raise ValueError(
                        "No non-zero Actual values found "
                        "for Tracking."
                    )

                actual_valid = (
                    actual_array[
                        actual_valid_mask
                    ]
                )

                actual_max = (
                    np.max(
                        actual_valid
                    )
                )

                actual_sum = (
                    np.sum(
                        actual_valid
                    )
                )

                for error in errors:

                    _, temp_cluster = (
                        calculate_effective_area(
                            df_area,
                            df_cluster,
                            error,
                        )
                    )

                    cluster_areas = (
                        pd.to_numeric(
                            temp_cluster[
                                "Eff Area(m2)"
                            ],
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .to_numpy(
                            dtype=float
                        )
                    )

                    objective = (
                        make_tracking_objective(
                            ghi_matrix,
                            cluster_areas,
                            blocks,
                            actual_array,
                        )
                    )

                    # Optimize tracking with this Error %
                    result = differential_evolution(
                        objective,
                        bounds=[
                            (0, 10),
                            (10, 30),
                            (65, 80),
                            (47, 53),
                            (10, 70),
                            (10, 70),
                        ],
                        strategy="best1bin",
                        maxiter=20,
                        popsize=8,
                        tol=0.002,
                        mutation=(0.5, 1),
                        recombination=0.7,
                        seed=42,
                        polish=True,
                        workers=1,
                    )

                    if (
                        result.fun
                        < best_error_value
                    ):

                        best_error_value = (
                            result.fun
                        )

                        best_error = (
                            float(error)
                        )

                        best_cluster_df = (
                            temp_cluster.copy()
                        )

                        best_tracking_result = (
                            result
                        )

                # ------------------------------------------------
                # Save automatic result
                # ------------------------------------------------

                best = np.round(
                    best_tracking_result.x
                ).astype(int)

                (
                    st.session_state.DHI,
                    st.session_state.GHI_Starting_Block,
                    st.session_state.GHI_Ending_Block,
                    st.session_state.GHI_Max_Block,
                    st.session_state.Tracking_angle_lim_E,
                    st.session_state.Tracking_angle_lim_W,
                ) = best.tolist()

                st.session_state.best_error = (
                    best_error
                )

                st.session_state.calculated = (
                    True
                )

                st.session_state.optimization_done = (
                    True
                )

                st.session_state.tracking_score = (
                    best_error_value
                )

                st.session_state.cluster_df = (
                    best_cluster_df
                )

            except Exception as exc:

                st.error(
                    f"Tracking optimization failed: {exc}"
                )


# ============================================================
# PARAMETERS AFTER AUTOMATIC CALCULATION
# ============================================================

if st.session_state.optimization_done:

    st.markdown(
        '<div class="section-title">'
        "7. Optimized Parameters"
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # ERROR %
    # ========================================================

    c1, c2 = st.columns(2)

    with c1:

        best_error = st.number_input(
            "Error %",
            min_value=-100.0,
            max_value=100.0,
            value=float(
                st.session_state.best_error
            ),
            step=0.1,
            key="best_error_input",
        )

    with c2:

        st.metric(
            "Plant Type",
            plant_type,
        )

    # ========================================================
    # TRACKING PARAMETERS
    # ========================================================

    if plant_type == "Tracking":

        c1, c2, c3 = st.columns(3)

        with c1:

            DHI = st.number_input(
                "DHI (%)",
                min_value=0,
                max_value=100,
                value=int(
                    st.session_state.DHI
                ),
                step=1,
            )

        with c2:

            GHI_Starting_Block = st.number_input(
                "GHI Starting Block",
                min_value=1,
                max_value=95,
                value=int(
                    st.session_state.GHI_Starting_Block
                ),
                step=1,
            )

        with c3:

            GHI_Ending_Block = st.number_input(
                "GHI Ending Block",
                min_value=1,
                max_value=95,
                value=int(
                    st.session_state.GHI_Ending_Block
                ),
                step=1,
            )

        c1, c2, c3 = st.columns(3)

        with c1:

            GHI_Max_Block = st.number_input(
                "GHI Max Block",
                min_value=1,
                max_value=95,
                value=int(
                    st.session_state.GHI_Max_Block
                ),
                step=1,
            )

        with c2:

            Tracking_angle_lim_E = st.number_input(
                "Tracking East Limit",
                min_value=0,
                max_value=90,
                value=int(
                    st.session_state.Tracking_angle_lim_E
                ),
                step=1,
            )

        with c3:

            Tracking_angle_lim_W = st.number_input(
                "Tracking West Limit",
                min_value=0,
                max_value=90,
                value=int(
                    st.session_state.Tracking_angle_lim_W
                ),
                step=1,
            )

        if not (
            GHI_Starting_Block
            < GHI_Max_Block
            < GHI_Ending_Block
        ):

            st.error(
                "Required condition: "
                "Starting Block < Max Block < Ending Block"
            )

            st.stop()

    # ========================================================
    # RECALCULATE BUTTON
    # ========================================================

    recalculate = st.button(
        "🔄 Recalculate Forecast",
        type="primary",
        use_container_width=True,
    )

    if recalculate:

        # ====================================================
        # FIXED
        # ====================================================

        if plant_type == "Fixed":

            try:

                geometry = (
                    prepare_fixed_geometry(
                        ghi_input,
                        lat,
                        month_lookup,
                    )
                )

                (
                    final_area,
                    final_cluster,
                ) = calculate_effective_area(
                    df_area,
                    df_cluster,
                    best_error,
                )

                forecast = (
                    calculate_fixed_forecast(
                        geometry,
                        final_cluster,
                    )
                )

                st.session_state.final_forecast = (
                    forecast
                )

                st.session_state.final_cluster = (
                    final_cluster
                )

                st.session_state.final_area = (
                    final_area
                )

                st.session_state.calculated = (
                    True
                )

            except Exception as exc:

                st.error(
                    f"Fixed calculation failed: {exc}"
                )

        # ====================================================
        # TRACKING
        # ====================================================

        else:

            try:

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Error % is applied exactly ONCE.
                #
                # We DO NOT modify the GHI or Actual data
                # with Error %.
                #
                # We DO NOT subtract Error % again inside
                # tracking.
                # ------------------------------------------------

                (
                    final_area,
                    final_cluster,
                ) = calculate_effective_area(
                    df_area,
                    df_cluster,
                    best_error,
                )

                cluster_areas = (
                    pd.to_numeric(
                        final_cluster[
                            "Eff Area(m2)"
                        ],
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .to_numpy(
                        dtype=float
                    )
                )

                (
                    forecast,
                    zenith,
                    panel,
                ) = calculate_tracking_forecast(
                    ghi_matrix,
                    cluster_areas,
                    blocks,
                    DHI,
                    GHI_Starting_Block,
                    GHI_Ending_Block,
                    GHI_Max_Block,
                    Tracking_angle_lim_E,
                    Tracking_angle_lim_W,
                )

                if forecast is None:

                    raise ValueError(
                        "Invalid Tracking block configuration."
                    )

                st.session_state.final_forecast = (
                    forecast
                )

                st.session_state.final_cluster = (
                    final_cluster
                )

                st.session_state.final_area = (
                    final_area
                )

                st.session_state.final_zenith = (
                    zenith
                )

                st.session_state.final_panel = (
                    panel
                )

                st.session_state.calculated = (
                    True
                )

                # Save edited parameters
                st.session_state.DHI = DHI
                st.session_state.GHI_Starting_Block = (
                    GHI_Starting_Block
                )
                st.session_state.GHI_Ending_Block = (
                    GHI_Ending_Block
                )
                st.session_state.GHI_Max_Block = (
                    GHI_Max_Block
                )
                st.session_state.Tracking_angle_lim_E = (
                    Tracking_angle_lim_E
                )
                st.session_state.Tracking_angle_lim_W = (
                    Tracking_angle_lim_W
                )

            except Exception as exc:

                st.error(
                    f"Tracking calculation failed: {exc}"
                )


# ============================================================
# FINAL RESULT
# ============================================================

if st.session_state.calculated:

    forecast = np.asarray(
        st.session_state.final_forecast,
        dtype=float,
    )

    actual = actual_array

    # ========================================================
    # METRICS
    # ========================================================

    calculated_peak = np.max(
        forecast
    )

    actual_peak = np.max(
        actual
    )

    peak_error = abs(
        calculated_peak
        - actual_peak
    )

    peak_error_pct = (
        peak_error
        / actual_peak
        * 100
        if actual_peak != 0
        else np.nan
    )

    calculated_energy = np.sum(
        forecast
    )

    actual_energy = np.sum(
        actual
    )

    energy_error_pct = (
        abs(
            calculated_energy
            - actual_energy
        )
        / actual_energy
        * 100
        if actual_energy != 0
        else np.nan
    )

    # ========================================================
    # RESULT METRICS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        "8. Final Result"
        "</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Error %",
        f"{best_error:.2f}%",
    )

    m2.metric(
        "Forecast Peak",
        f"{calculated_peak:.3f}",
    )

    m3.metric(
        "Actual Peak",
        f"{actual_peak:.3f}",
    )

    m4.metric(
        "Peak Error",
        f"{peak_error_pct:.2f}%",
    )

    # ========================================================
    # GRAPH
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        "Forecast vs Actual"
        "</div>",
        unsafe_allow_html=True,
    )

    x = np.arange(
        len(forecast)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                width=2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(
                width=2,
            ),
        )
    )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        xaxis_title="Block",
        yaxis_title="Power",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # TRACKING ANGLES
    # ========================================================

    if (
        plant_type == "Tracking"
        and "final_zenith"
        in st.session_state
    ):

        st.markdown(
            '<div class="section-title">'
            "Tracking Angles"
            "</div>",
            unsafe_allow_html=True,
        )

        angle_df = pd.DataFrame(
            {
                "Block":
                    blocks.astype(int),

                "Zenith Angle":
                    st.session_state.final_zenith,

                "Panel Angle":
                    st.session_state.final_panel,
            }
        )

        angle_fig = go.Figure()

        angle_fig.add_trace(
            go.Scatter(
                x=angle_df["Block"],
                y=angle_df["Zenith Angle"],
                mode="lines",
                name="Zenith Angle",
            )
        )

        angle_fig.add_trace(
            go.Scatter(
                x=angle_df["Block"],
                y=angle_df["Panel Angle"],
                mode="lines",
                name="Panel Angle",
            )
        )

        angle_fig.update_layout(
            height=400,
            hovermode="x unified",
            xaxis_title="Block",
            yaxis_title="Angle (°)",
        )

        st.plotly_chart(
            angle_fig,
            use_container_width=True,
        )

    # ========================================================
    # CLUSTER EFFECTIVE AREAS
    # ========================================================

    if "final_cluster" in st.session_state:

        st.markdown(
            '<div class="section-title">'
            "Cluster Effective Areas"
            "</div>",
            unsafe_allow_html=True,
        )

        display_cluster = (
            st.session_state.final_cluster
            .copy()
        )

        st.dataframe(
            display_cluster,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # DOWNLOAD RESULT
    # ========================================================

    result_df = pd.DataFrame(
        {
            "Block":
                np.arange(
                    len(forecast)
                ),

            "Forecast":
                forecast,

            "Actual":
                actual,

            "Error":
                forecast - actual,

            "Absolute Error":
                np.abs(
                    forecast - actual
                ),
        }
    )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        result_df.to_excel(
            writer,
            sheet_name="Forecast Result",
            index=False,
        )

        if "final_cluster" in st.session_state:

            st.session_state.final_cluster.to_excel(
                writer,
                sheet_name="Cluster Areas",
                index=False,
            )

        if (
            plant_type == "Tracking"
            and "final_zenith"
            in st.session_state
        ):

            angle_df.to_excel(
                writer,
                sheet_name="Tracking Angles",
                index=False,
            )

    st.download_button(
        "⬇️ Download Result",
        data=output.getvalue(),
        file_name=(
            "Solar_Loss_Correction_Result.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        use_container_width=True,
    )
