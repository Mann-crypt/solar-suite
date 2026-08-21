# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# CLEAN + NON-FREEZING STREAMLIT PAGE
#
# Calculation logic preserved:
#   1. Automatic Error % optimization
#   2. Error % applied once to effective area
#   3. Fixed forecast calculation
#   4. Tracking parameter optimization when Tracking selected
#   5. Editable parameters after automatic calculation
#
# Performance:
#   - Excel loading cached
#   - Error optimization cached
#   - Tracking optimization cached
#   - Tracking optimization ONLY runs when Run Calculation clicked
#   - Fixed plant NEVER runs tracking optimization
#   - Editing parameters only performs cheap final calculation
# ============================================================

import io
import hashlib
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
    initial_sidebar_state="collapsed",
)


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
    "input_df": None,
    "last_file_hash": None,
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


def numeric_array(value):
    """
    Safely convert Series / DataFrame column / numpy array / list
    to a numeric numpy array.

    Fixes the previous:
        numpy.ndarray has no attribute fillna
    problem.
    """
    if isinstance(value, pd.Series):
        return (
            pd.to_numeric(value, errors="coerce")
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
    return pd.to_numeric(
        value,
        errors="coerce",
    ).fillna(0)


def file_hash(uploaded_file):
    uploaded_file.seek(0)
    content = uploaded_file.getvalue()
    uploaded_file.seek(0)
    return hashlib.md5(content).hexdigest()


def read_excel_bytes(file_bytes, **kwargs):
    return pd.read_excel(
        io.BytesIO(file_bytes),
        **kwargs,
    )


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
# CLEAN AREA & EFFICIENCY
# ============================================================

def prepare_area_efficiency(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    if "S.No." in df.columns:

        mask = df["S.No."].isna()

        if mask.any():
            first = np.flatnonzero(mask.to_numpy())[0]
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
            first = np.flatnonzero(mask.to_numpy())[0]
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

        first = np.flatnonzero(mask.to_numpy())[0]

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

            first = np.flatnonzero(mask.to_numpy())[0]

            df = df.iloc[:first].copy()

    df["Actual"] = numeric_series(
        df["Actual"]
    )

    return df.reset_index(drop=True)


# ============================================================
# INPUT DATAFRAME
# ============================================================

def build_input_dataframe(
    df_ghi,
    df_fixed,
):

    n = min(
        len(df_ghi),
        len(df_fixed),
    )

    result = pd.DataFrame()

    for col in GHI_COLUMNS:
        result[col] = numeric_series(
            df_ghi[col]
        ).iloc[:n].to_numpy()

    result["Actual"] = numeric_series(
        df_fixed["Actual"]
    ).iloc[:n].to_numpy()

    return result.reset_index(drop=True)


# ============================================================
# APPLY USER INPUT DATA
# ============================================================

def apply_input_dataframe(
    input_df,
    df_ghi,
    df_fixed,
):

    n = len(input_df)

    if n == 0:
        raise ValueError(
            "Input dataframe is empty."
        )

    df_ghi = df_ghi.iloc[:n].copy()
    df_fixed = df_fixed.iloc[:n].copy()

    for col in GHI_COLUMNS:

        df_ghi[col] = numeric_series(
            input_df[col]
        ).to_numpy()

    df_fixed["Actual"] = numeric_series(
        input_df["Actual"]
    ).to_numpy()

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

    df_fix = df_fix.iloc[:n].copy()
    df_ghi = df_ghi.iloc[:n].copy()

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

    df_fix["Tilt Angle b"] = pd.to_numeric(
        df_fix["Tilt Angle b"],
        errors="coerce",
    ).fillna(0)

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

    sin_a = df_fix["Sin(a)"].replace(
        0,
        np.nan,
    )

    # C11
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

    # C12
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

    # C13
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

    # C14
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

    # C15
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

    error = float(error)

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
        df.groupby("Clusters")["Eff Area"]
        .sum()
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

    for i, poa_col in enumerate(
        FIXED_POA_COLUMNS
    ):

        result[
            POWER_COLUMNS[i]
        ] = (
            result[poa_col]
            * float(
                pd.to_numeric(
                    df_w.iloc[i]["Eff Area(m2)"],
                    errors="coerce",
                )
                or 0
            )
        ) / 1_000_000

    result[TOTAL_POWER_COLUMN] = (
        result[POWER_COLUMNS]
        .sum(axis=1)
    )

    return result


# ============================================================
# AUTOMATIC ERROR OPTIMIZATION
#
# Cached.
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

    actual_peak = (
        actual.max()
        if len(actual)
        else 0
    )

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

        _, df_w = calculate_effective_area(
            df_original,
            df_w_original,
            error,
        )

        calculated = calculate_fixed_power(
            df_fix,
            df_w,
        )

        forecast = numeric_array(
            calculated[
                TOTAL_POWER_COLUMN
            ]
        )

        calculated_peak = (
            forecast.max()
            if len(forecast)
            else 0
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

    result_df = pd.DataFrame(
        results
    )

    best_row = result_df.loc[
        result_df[
            "Peak Error"
        ].idxmin()
    ]

    return (
        float(best_row["Error %"]),
        result_df,
    )


# ============================================================
# TRACKING PREPARATION
# ============================================================

def prepare_tracking_data(
    backend,
):

    if "C11" not in backend:
        raise ValueError(
            "Backend Cal C11 sheet not found."
        )

    df_backend = backend["C11"].copy()

    if "Block No." not in df_backend.columns:
        raise ValueError(
            "Block No. column not found in Backend Cal C11."
        )

    blocks = numeric_array(
        df_backend["Block No."]
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

    m1 = 90 / denominator_1
    m2 = 90 / denominator_2

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
# IMPORTANT:
# This function is ONLY called when Tracking is selected
# and Run Calculation is clicked.
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

    blocks = blocks[:n]
    ghi_matrix = ghi_matrix[:n]
    actual_full = actual_full[:n]

    valid_mask = actual_full != 0

    if not valid_mask.any():
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual_day = actual_full[
        valid_mask
    ]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    def objective(x):

        DHI = int(round(x[0]))
        start_block = int(round(x[1]))
        end_block = int(round(x[2]))
        max_block = int(round(x[3]))
        east_limit = int(round(x[4]))
        west_limit = int(round(x[5]))

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

        prediction_day = prediction[
            valid_mask
        ]

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
        (10, 70),
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
        updating="immediate",
    )

    best = np.rint(
        result.x
    ).astype(int)

    parameters = {
        "DHI": int(best[0]),
        "GHI Starting Block": int(best[1]),
        "GHI Ending Block": int(best[2]),
        "GHI Max Block": int(best[3]),
        "Tracking East Limit": int(best[4]),
        "Tracking West Limit": int(best[5]),
    }

    return parameters


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    forecast,
):

    actual = numeric_array(actual)
    forecast = numeric_array(forecast)

    n = min(
        len(actual),
        len(forecast),
    )

    actual = actual[:n]
    forecast = forecast[:n]

    if len(actual) == 0:
        return {
            "Actual Peak": 0,
            "Forecast Peak": 0,
        }

    return {
        "Actual Peak":
            float(actual.max()),

        "Forecast Peak":
            float(forecast.max()),
    }


# ============================================================
# GRAPH
# ============================================================

def build_graph(
    actual,
    forecast,
    title,
):

    actual = numeric_array(actual)
    forecast = numeric_array(forecast)

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
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)


if uploaded_file is None:

    st.info(
        "Upload the Solar Excel file to start."
    )

    st.stop()


# ============================================================
# FILE CHANGE RESET
# ============================================================

current_hash = file_hash(
    uploaded_file
)

if (
    st.session_state.last_file_hash
    != current_hash
):

    st.session_state.calculated = False
    st.session_state.calculation_data = None
    st.session_state.input_df = None
    st.session_state.last_file_hash = current_hash


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    file_bytes = uploaded_file.getvalue()

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

    df_original = prepare_area_efficiency(
        workbook["area"]
    )

    df_w_original = prepare_cluster_table(
        workbook["cluster"]
    )

    df_ghi_raw = prepare_ghi(
        workbook["ghi"]
    )

    lat = prepare_latitude(
        workbook["forecast_config"]
    )

    month_lookup = prepare_tilt(
        workbook["tilt"]
    )

    df_fix_raw = prepare_fixed(
        workbook["fixed"]
    )

except Exception as e:

    st.error(
        f"Input preparation failed: {e}"
    )

    st.stop()


# ============================================================
# INPUT DATA EDITOR
#
# Horizontal dataframe.
# No calculation occurs when the user edits it.
# ============================================================

if st.session_state.input_df is None:

    st.session_state.input_df = (
        build_input_dataframe(
            df_ghi_raw,
            df_fix_raw,
        )
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

plant_type = st.segmented_control(
    "Plant Type",
    options=["Fixed", "Tracking"],
    default=st.session_state.plant_type,
    selection_mode="single",
    width="stretch",
    label_visibility="collapsed",
)

if plant_type is None:
    plant_type = st.session_state.plant_type

st.session_state.plant_type = plant_type


# ============================================================
# RUN CALCULATION
#
# THIS IS THE ONLY PLACE WHERE EXPENSIVE OPTIMIZATION RUNS.
# ============================================================

st.markdown("")

run_clicked = st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    width="stretch",
)


if run_clicked:

    try:

        with st.spinner(
            "Calculating automatic parameters..."
        ):

            # ------------------------------------------------
            # SAVE USER INPUT
            # ------------------------------------------------

            st.session_state.input_df = (
                input_df.copy()
            )

            # ------------------------------------------------
            # APPLY EDITED GHI + ACTUAL
            # ------------------------------------------------

            df_ghi, df_fix_raw_user = (
                apply_input_dataframe(
                    input_df,
                    df_ghi_raw,
                    df_fix_raw,
                )
            )

            # ------------------------------------------------
            # SOLAR GEOMETRY
            # ------------------------------------------------

            df_fix = prepare_fixed_geometry(
                df_fix_raw_user,
                df_ghi,
                lat,
                month_lookup,
            )

            # ------------------------------------------------
            # AUTOMATIC ERROR %
            #
            # Runs once.
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
            # APPLY ERROR %
            #
            # Error applied exactly once.
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
            # FIXED FORECAST
            #
            # Always calculate because it is cheap.
            # ------------------------------------------------

            fixed_final = calculate_fixed_power(
                df_fix,
                df_w_final,
            )

            # ------------------------------------------------
            # TRACKING
            #
            # ONLY RUN FOR TRACKING.
            # ------------------------------------------------

            tracking_parameters = None
            tracking_forecast = None

            if plant_type == "Tracking":

                blocks = prepare_tracking_data(
                    workbook["backend"]
                )

                # Match dimensions safely.
                n = min(
                    len(blocks),
                    len(df_ghi),
                    len(df_fix),
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

                actual_tracking = (
                    numeric_array(
                        df_fix["Actual"]
                    )[:n]
                )

                tracking_weights = (
                    pd.to_numeric(
                        df_w_final.iloc[:5, 1],
                        errors="coerce",
                    )
                    .fillna(0)
                    .to_numpy(dtype=float)
                )

                # ------------------------------------------------
                # AUTOMATIC TRACKING PARAMETERS
                #
                # THIS IS EXPENSIVE.
                # BUT IT RUNS ONLY HERE.
                # ------------------------------------------------

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

                # ------------------------------------------------
                # FINAL TRACKING FORECAST
                # ------------------------------------------------

                tracking_result = (
                    calculate_tracking(
                        tracking_parameters["DHI"],
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
                        "Automatic tracking parameters produced an invalid result."
                    )

                tracking_forecast = (
                    tracking_result[0]
                )

            # ------------------------------------------------
            # SAVE RESULTS
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

            st.session_state.calculated = True

        st.success(
            "Automatic calculation completed successfully."
        )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )


# ============================================================
# WAIT FOR CALCULATION
# ============================================================

if not st.session_state.calculated:

    st.caption(
        "Edit GHI / Actual values above, select the plant type, "
        "then click Run Automatic Calculation."
    )

    st.stop()


# ============================================================
# RESULTS DATA
# ============================================================

data = (
    st.session_state.calculation_data
)


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)


# ============================================================
# ERROR PARAMETER
# ============================================================

error_value = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=20.0,
    value=float(
        data["best_error"]
    ),
    step=0.1,
    format="%.1f",
    key="editable_error",
)


# ============================================================
# TRACKING PARAMETERS
#
# These are cheap to change.
# They do NOT run Differential Evolution.
# ============================================================

if plant_type == "Tracking":

    params = data[
        "tracking_parameters"
    ]

    if params is None:

        st.warning(
            "Tracking parameters are not available. "
            "Run the automatic calculation again."
        )

        st.stop()

    st.markdown(
        "#### Tracking Parameters"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi_value = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(
                params["DHI"]
            ),
            step=1,
            key="tracking_dhi",
        )

        start_value = st.number_input(
            "GHI Starting Block",
            min_value=0,
            max_value=95,
            value=int(
                params[
                    "GHI Starting Block"
                ]
            ),
            step=1,
            key="tracking_start",
        )

    with c2:

        end_value = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                params[
                    "GHI Ending Block"
                ]
            ),
            step=1,
            key="tracking_end",
        )

        max_value = st.number_input(
            "GHI Max Block",
            min_value=0,
            max_value=95,
            value=int(
                params[
                    "GHI Max Block"
                ]
            ),
            step=1,
            key="tracking_max",
        )

    with c3:

        east_value = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            value=int(
                params[
                    "Tracking East Limit"
                ]
            ),
            step=1,
            key="tracking_east",
        )

        west_value = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            value=int(
                params[
                    "Tracking West Limit"
                ]
            ),
            step=1,
            key="tracking_west",
        )


# ============================================================
# RECALCULATE CHEAP FINAL FORECAST
#
# IMPORTANT:
# NO OPTIMIZER HERE.
# ============================================================

try:

    (
        df_final,
        df_w_final,
    ) = calculate_effective_area(
        data["df_original"],
        data["df_w_original"],
        error_value,
    )

    fixed_final = calculate_fixed_power(
        data["df_fix"],
        df_w_final,
    )

    # --------------------------------------------------------
    # TRACKING FINAL FORECAST
    # --------------------------------------------------------

    if plant_type == "Tracking":

        blocks = prepare_tracking_data(
            workbook["backend"]
        )

        df_ghi = data["df_ghi"]
        df_fix = data["df_fix"]

        n = min(
            len(blocks),
            len(df_ghi),
            len(df_fix),
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

        actual_tracking = (
            numeric_array(
                df_fix["Actual"]
            )[:n]
        )

        tracking_weights = (
            pd.to_numeric(
                df_w_final.iloc[:5, 1],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(dtype=float)
        )

        # Validate edited tracking parameters.
        if not (
            start_value
            < max_value
            < end_value
        ):

            st.error(
                "Tracking parameters must satisfy: "
                "GHI Starting Block < GHI Max Block < GHI Ending Block."
            )

            st.stop()

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

        actual = actual_tracking
        forecast = tracking_forecast

        graph_title = (
            "Tracking Plant | Actual vs Forecast"
        )

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


m1, m2 = st.columns(2)


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
