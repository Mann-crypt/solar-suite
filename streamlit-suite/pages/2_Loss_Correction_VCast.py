# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# Clean + Fast + Bug-Fixed Streamlit App
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
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1 {
        margin-bottom: 0.1rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }

    .stButton > button {
        width: 100%;
    }

    .result-card {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,.25);
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]


# ============================================================
# BASIC HELPERS
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


def trim_at_blank(df, column):
    if column not in df.columns:
        return df.copy()

    valid = df[column].notna()

    if not valid.any():
        return df.iloc[0:0].copy()

    invalid = np.flatnonzero(
        ~valid.to_numpy()
    )

    if len(invalid):
        return df.iloc[:invalid[0]].copy()

    return df.copy()


def numeric_array(series):
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )


def numeric_df_array(df, columns):
    """
    Safely convert multiple columns to a NumPy matrix.

    This avoids the pd.to_numeric(DataFrame) bug.
    """
    return (
        df.loc[:, columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )


def numeric_df_sum(df, columns):
    """
    Safe numeric sum for multiple columns.
    """
    arr = numeric_df_array(
        df,
        columns,
    )

    return float(
        np.sum(arr)
    )


def numeric_series_sum(df, column):
    return float(
        numeric_array(
            df[column]
        ).sum()
    )


def dataframe_signature(df, columns):
    """
    Fast and stable signature for edited input data.
    """
    arr = numeric_df_array(
        df,
        columns,
    )

    return hashlib.md5(
        arr.tobytes()
    ).hexdigest()


def series_signature(df, column):
    arr = numeric_array(
        df[column]
    )

    return hashlib.md5(
        arr.tobytes()
    ).hexdigest()


# ============================================================
# LOAD WORKBOOK
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_workbook(file_bytes):

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    bio = io.BytesIO(file_bytes)

    area = pd.read_excel(
        bio,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    area = clean_columns(area)

    area = trim_at_blank(
        area,
        "S.No.",
    )

    # --------------------------------------------------------
    # Cluster mapping
    # --------------------------------------------------------

    bio.seek(0)

    cluster = pd.read_excel(
        bio,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    cluster = clean_columns(cluster)

    cluster = trim_at_blank(
        cluster,
        "Clusters",
    )

    # --------------------------------------------------------
    # Forecast Config
    # --------------------------------------------------------

    bio.seek(0)

    config = pd.read_excel(
        bio,
        sheet_name="Forecast Config",
        header=8,
    )

    if "Lat" not in config.columns:
        raise ValueError(
            "Column 'Lat' not found in Forecast Config."
        )

    lat = float(
        pd.to_numeric(
            config.loc[0, "Lat"],
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    bio.seek(0)

    tilt = pd.read_excel(
        bio,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    tilt = clean_columns(
        tilt
    )

    tilt = trim_at_blank(
        tilt,
        "Fixed",
    )

    tilt = tilt.dropna(
        how="all",
        axis=1,
    )

    tilt = tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    if (
        "Month" not in tilt.columns
        or "Fixed" not in tilt.columns
    ):
        raise ValueError(
            "Unable to read Fixed tilt information."
        )

    month_lookup = (
        pd.to_numeric(
            tilt["Fixed"],
            errors="coerce",
        )
        .groupby(
            tilt["Month"]
        )
        .first()
        .to_dict()
    )

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    bio.seek(0)

    ghi = pd.read_excel(
        bio,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    ghi = clean_columns(
        ghi
    )

    missing = [
        c for c in GHI_COLS
        if c not in ghi.columns
    ]

    if missing:
        raise ValueError(
            "Missing GHI columns: "
            + ", ".join(missing)
        )

    ghi[GHI_COLS] = (
        ghi[GHI_COLS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Actual
    # --------------------------------------------------------

    bio.seek(0)

    fixed = pd.read_excel(
        bio,
        sheet_name="Fixed-C11",
        header=1,
    )

    fixed = clean_columns(
        fixed
    )

    fixed = trim_at_blank(
        fixed,
        "Date",
    )

    if "Actual" not in fixed.columns:
        raise ValueError(
            "Column 'Actual' not found in Fixed-C11."
        )

    actual = (
        pd.to_numeric(
            fixed["Actual"],
            errors="coerce",
        )
        .fillna(0)
        .to_frame("Actual")
    )

    # --------------------------------------------------------
    # Backend tracking blocks
    # --------------------------------------------------------

    backend_blocks = []

    for i in range(1, 6):

        bio.seek(0)

        try:
            backend = pd.read_excel(
                bio,
                sheet_name=f"Backend Cal C{i}",
            )

            backend = clean_columns(
                backend
            )

            if "Block No." in backend.columns:

                backend_blocks.append(
                    numeric_array(
                        backend["Block No."]
                    )
                )

            else:
                backend_blocks.append(
                    None
                )

        except Exception:
            backend_blocks.append(
                None
            )

    # --------------------------------------------------------
    # Tracking sheet
    # --------------------------------------------------------

    bio.seek(0)

    try:
        tracking = pd.read_excel(
            bio,
            sheet_name="Tracking",
            header=1,
        )

        tracking = clean_columns(
            tracking
        )

    except Exception:
        tracking = pd.DataFrame()

    return {
        "area": area,
        "cluster": cluster,
        "lat": lat,
        "month_lookup": month_lookup,
        "ghi": ghi,
        "actual": actual,
        "backend_blocks": backend_blocks,
        "tracking": tracking,
    }


# ============================================================
# PLANT DATA
# ============================================================

@st.cache_data(
    show_spinner=False
)
def prepare_plant_data(data):

    df = data["area"].copy()
    df_w = data["cluster"].copy()

    if "Error %" not in df.columns:
        df["Error %"] = 0.0

    if (
        "Total area (m2)" not in df.columns
        and "No of Module" in df.columns
        and "Area of 1 Module (m2)" in df.columns
    ):

        df["Total area (m2)"] = (
            numeric_array(
                df["No of Module"]
            )
            *
            numeric_array(
                df["Area of 1 Module (m2)"]
            )
        )

    required = [
        "Standard PV Efficiency (%)",
        "Total area (m2)",
        "Clusters",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing Area & Efficiency columns: "
            + ", ".join(missing)
        )

    if "Clusters" not in df_w.columns:
        raise ValueError(
            "Clusters column missing from cluster mapping."
        )

    return df, df_w


# ============================================================
# EFFECTIVE AREA
# ============================================================

def calculate_cluster_area(
    df,
    df_w,
    error_percent,
):
    """
    IMPORTANT:
    Error % is applied exactly ONCE here.

    Net Efficiency =
        Standard Efficiency - Error %

    Effective Area =
        Net Efficiency × Total Area / 100
    """

    standard_efficiency = numeric_array(
        df["Standard PV Efficiency (%)"]
    )

    total_area = numeric_array(
        df["Total area (m2)"]
    )

    clusters = (
        df["Clusters"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # ERROR APPLIED ONCE
    # --------------------------------------------------------

    net_efficiency = (
        standard_efficiency
        - float(error_percent)
    )

    # Prevent negative effective efficiency
    net_efficiency = np.maximum(
        net_efficiency,
        0,
    )

    effective_area = (
        net_efficiency
        * total_area
        / 100.0
    )

    cluster_sum = (
        pd.DataFrame(
            {
                "Clusters": clusters,
                "Eff Area": effective_area,
            }
        )
        .groupby(
            "Clusters"
        )["Eff Area"]
        .sum()
    )

    cluster_keys = (
        df_w["Clusters"]
        .astype(str)
        .str.strip()
    )

    cluster_area = (
        cluster_keys
        .map(cluster_sum)
        .fillna(0)
        .to_numpy(dtype=float)
    )

    return (
        net_efficiency,
        effective_area,
        cluster_area,
    )


# ============================================================
# FIXED BASE
# ============================================================

@st.cache_data(
    show_spinner=False
)
def prepare_fixed_base(
    ghi_values,
    actual_values,
    lat,
    month_lookup,
):

    ghi = np.asarray(
        ghi_values,
        dtype=float,
    )

    actual = np.asarray(
        actual_values,
        dtype=float,
    )

    n = min(
        len(ghi),
        len(actual),
    )

    ghi = ghi[:n]
    actual = actual[:n]

    # --------------------------------------------------------
    # Date logic
    # --------------------------------------------------------

    # Keep original Jupyter-style date calculation.
    today = pd.Timestamp.today().normalize()

    dates = pd.date_range(
        start=today,
        periods=n,
        freq="15min",
    )

    first_date = pd.Timestamp(
        year=today.year,
        month=1,
        day=1,
    )

    day_number = (
        dates - first_date
    ).days.to_numpy()

    declination = (
        23.45
        *
        np.sin(
            np.radians(
                360
                *
                (
                    284
                    + day_number
                    + 1
                )
                / 365
            )
        )
    )

    elevation = (
        90
        - float(lat)
        + declination
    )

    month_names = dates.strftime(
        "%B"
    )

    tilt = np.array(
        [
            float(
                month_lookup.get(
                    month,
                    0,
                )
                or 0
            )
            for month in month_names
        ],
        dtype=float,
    )

    a_plus_b = (
        elevation
        + tilt
    )

    sin_ab = np.sin(
        np.radians(
            a_plus_b
        )
    )

    sin_a = np.sin(
        np.radians(
            elevation
        )
    )

    safe_sin_a = np.where(
        np.abs(sin_a) < 1e-10,
        np.nan,
        sin_a,
    )

    poa = (
        ghi
        * sin_ab[:, None]
        / safe_sin_a[:, None]
    )

    return {
        "ghi": ghi,
        "actual": actual,
        "poa": poa,
        "declination": declination,
        "elevation": elevation,
        "tilt": tilt,
    }


# ============================================================
# FIXED FORECAST
# ============================================================

def fixed_forecast(
    base,
    cluster_area,
):

    forecast = (
        np.nan_to_num(
            base["poa"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        @ cluster_area
    ) / 1_000_000.0

    return forecast


# ============================================================
# FIXED OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False
)
def optimize_fixed(
    plant_df,
    cluster_df,
    base,
    error_min,
    error_max,
    error_step,
):

    errors = np.arange(
        float(error_min),
        float(error_max)
        + float(error_step) * 0.5,
        float(error_step),
    )

    actual = base["actual"]

    if len(actual) == 0:
        raise ValueError(
            "No Actual data available."
        )

    actual_peak = float(
        np.max(actual)
    )

    best_error = float(
        error_min
    )

    best_peak_error = np.inf
    best_forecast = None

    rows = []

    for error in errors:

        _, _, cluster_area = (
            calculate_cluster_area(
                plant_df,
                cluster_df,
                error,
            )
        )

        forecast = fixed_forecast(
            base,
            cluster_area,
        )

        calculated_peak = float(
            np.max(forecast)
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

        rows.append(
            {
                "Error %": float(error),
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

        if peak_error < best_peak_error:

            best_peak_error = peak_error

            best_error = float(
                error
            )

            best_forecast = (
                forecast.copy()
            )

    return (
        best_error,
        best_forecast,
        pd.DataFrame(rows),
    )


# ============================================================
# TRACKING BASE
# ============================================================

@st.cache_data(
    show_spinner=False
)
def prepare_tracking_base(
    ghi_values,
    actual_values,
    backend_blocks,
):

    ghi = np.asarray(
        ghi_values,
        dtype=float,
    )

    actual = np.asarray(
        actual_values,
        dtype=float,
    )

    n = min(
        len(ghi),
        len(actual),
    )

    ghi = ghi[:n]
    actual = actual[:n]

    blocks = None

    if backend_blocks:

        first = backend_blocks[0]

        if (
            first is not None
            and len(first) >= n
        ):
            blocks = (
                np.asarray(
                    first[:n],
                    dtype=float,
                )
            )

    if blocks is None:
        blocks = np.arange(
            n,
            dtype=float,
        )

    return {
        "ghi": ghi,
        "actual": actual,
        "blocks": blocks,
    }


# ============================================================
# TRACKING FORECAST
# ============================================================

def tracking_forecast(
    base,
    cluster_area,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
):

    blocks = base["blocks"]
    ghi = base["ghi"]

    start_block = int(
        start_block
    )

    end_block = int(
        end_block
    )

    max_block = int(
        max_block
    )

    east_limit = float(
        east_limit
    )

    west_limit = float(
        west_limit
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not (
        start_block
        < max_block
        < end_block
    ):
        return None

    den1 = (
        start_block
        - 1
        - max_block
    )

    den2 = (
        end_block
        + 1
        - max_block
    )

    if den1 == 0 or den2 == 0:
        return None

    # --------------------------------------------------------
    # Zenith
    # --------------------------------------------------------

    m1 = 90.0 / den1
    m2 = 90.0 / den2

    zenith = np.where(
        blocks <= max_block,

        np.minimum(
            89.0,
            m1
            * (
                blocks
                - max_block
            ),
        ),

        np.minimum(
            89.0,
            m2
            * (
                blocks
                - max_block
            ),
        ),
    )

    # --------------------------------------------------------
    # Panel angle
    # --------------------------------------------------------

    panel = np.where(
        blocks < max_block,

        np.maximum(
            zenith,
            -abs(east_limit),
        ),

        np.where(
            (
                (blocks > max_block)
                & (
                    zenith
                    > west_limit
                )
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

    # --------------------------------------------------------
    # DHI / DNI
    # --------------------------------------------------------

    dhi = (
        ghi
        * float(DHI)
        / 100.0
    )

    dni = (
        ghi
        - dhi
    ) / cos_alpha[:, None]

    dni = np.nan_to_num(
        dni,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # POWER
    # --------------------------------------------------------

    forecast = (
        dni
        @ cluster_area
    ) / 1_000_000.0

    forecast = np.nan_to_num(
        forecast,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return {
        "forecast": forecast,
        "zenith": zenith,
        "panel": panel,
    }


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False
)
def optimize_tracking(
    base,
    cluster_area,
    bounds,
    maxiter,
    popsize,
    seed,
):

    actual_full = base["actual"]

    mask = (
        np.isfinite(actual_full)
        & (
            actual_full
            != 0
        )
    )

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values found."
        )

    actual = actual_full[
        mask
    ]

    actual_max = float(
        np.max(actual)
    )

    actual_sum = float(
        np.sum(actual)
    )

    if (
        actual_max <= 0
        or actual_sum <= 0
    ):
        raise ValueError(
            "Actual data contains no usable values."
        )

    def objective(x):

        DHI = int(
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

        result = tracking_forecast(
            base,
            cluster_area,
            DHI,
            start,
            end,
            max_block,
            east,
            west,
        )

        if result is None:
            return 1e9

        prediction = (
            result["forecast"][mask]
        )

        if (
            len(prediction) == 0
            or not np.all(
                np.isfinite(
                    prediction
                )
            )
        ):
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual
                    - prediction
                )
            )
            / actual_max
        )

        peak_error = (
            abs(
                actual_max
                - np.max(
                    prediction
                )
            )
            / actual_max
        )

        energy_error = (
            abs(
                actual_sum
                - np.sum(
                    prediction
                )
            )
            / actual_sum
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=int(maxiter),
        popsize=int(popsize),
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=int(seed),
        polish=True,
        workers=1,
    )

    best = np.round(
        result.x
    ).astype(int)

    params = {
        "DHI": int(best[0]),
        "GHI Starting Block": int(best[1]),
        "GHI Ending Block": int(best[2]),
        "GHI Max Block": int(best[3]),
        "Tracking East Limit": int(best[4]),
        "Tracking West Limit": int(best[5]),
    }

    return (
        params,
        float(result.fun),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    forecast,
    actual,
):

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    n = min(
        len(forecast),
        len(actual),
    )

    forecast = forecast[:n]
    actual = actual[:n]

    if n == 0:
        return {
            "Forecast Peak": 0,
            "Actual Peak": 0,
            "Peak Error": 0,
            "Peak Error %": np.nan,
            "Energy Error %": np.nan,
        }

    forecast_peak = float(
        np.max(forecast)
    )

    actual_peak = float(
        np.max(actual)
    )

    peak_error = abs(
        forecast_peak
        - actual_peak
    )

    peak_error_pct = (
        peak_error
        / actual_peak
        * 100
        if actual_peak != 0
        else np.nan
    )

    actual_energy = float(
        np.sum(actual)
    )

    energy_error_pct = (
        abs(
            np.sum(forecast)
            - actual_energy
        )
        / actual_energy
        * 100
        if actual_energy != 0
        else np.nan
    )

    return {
        "Forecast Peak":
            forecast_peak,
        "Actual Peak":
            actual_peak,
        "Peak Error":
            peak_error,
        "Peak Error %":
            peak_error_pct,
        "Energy Error %":
            energy_error_pct,
    }


# ============================================================
# FORECAST GRAPH
# ============================================================

def plot_forecast(
    forecast,
    actual,
    title,
):

    n = min(
        len(forecast),
        len(actual),
    )

    x = np.arange(
        n
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast[:n],
            mode="lines",
            name="Forecast",
            line=dict(
                width=2
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual[:n],
            mode="lines",
            name="Actual",
            line=dict(
                width=2
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
        hovermode="x unified",
        xaxis_title="Block",
        yaxis_title="Power (MW)",
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
# HEADER
# ============================================================

st.title(
    "☀️ Solar Forecast Correction"
)

st.caption(
    "Fixed / Tracking plant optimization"
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel Workbook",
    type=[
        "xlsx",
        "xls",
    ],
)


if uploaded_file is None:

    st.info(
        "Upload the Excel workbook to load GHI Forecast and Actual Power."
    )

    st.stop()


# ============================================================
# FILE ID
# ============================================================

file_bytes = (
    uploaded_file.getvalue()
)

file_id = hashlib.md5(
    file_bytes
).hexdigest()


# ============================================================
# LOAD
# ============================================================

try:

    data = load_workbook(
        file_bytes
    )

    plant_df, cluster_df = (
        prepare_plant_data(
            data
        )
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


# ============================================================
# INPUT DATA
# ============================================================

st.subheader(
    "Input Data"
)

col1, col2 = st.columns(
    2
)

with col1:

    st.markdown(
        "#### GHI Forecast"
    )

    ghi_input = st.data_editor(
        data["ghi"].copy(),
        use_container_width=True,
        num_rows="fixed",
        key=f"ghi_input_{file_id}",
    )


with col2:

    st.markdown(
        "#### Actual Power"
    )

    actual_input = st.data_editor(
        data["actual"].copy(),
        use_container_width=True,
        num_rows="fixed",
        key=f"actual_input_{file_id}",
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

missing_ghi = [
    col
    for col in GHI_COLS
    if col not in ghi_input.columns
]

if missing_ghi:

    st.error(
        "Missing GHI columns: "
        + ", ".join(
            missing_ghi
        )
    )

    st.stop()


if "Actual" not in actual_input.columns:

    st.error(
        "Actual column is required."
    )

    st.stop()


# ============================================================
# CONVERT INPUT ONCE
# ============================================================

ghi_values = numeric_df_array(
    ghi_input,
    GHI_COLS,
)

actual_values = numeric_array(
    actual_input["Actual"]
)


# ============================================================
# PLANT TYPE
# ============================================================

st.subheader(
    "Plant Type"
)

plant_type = st.segmented_control(
    "Select Plant Type",
    [
        "Fixed",
        "Tracking",
    ],
    default="Fixed",
)


# ============================================================
# FIXED
# ============================================================

if plant_type == "Fixed":

    st.subheader(
        "Optimization Parameters"
    )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:

        error_min = st.number_input(
            "Error Min (%)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.1,
        )

    with c2:

        error_max = st.number_input(
            "Error Max (%)",
            min_value=0.1,
            max_value=50.0,
            value=10.0,
            step=0.1,
        )

    with c3:

        error_step = st.number_input(
            "Error Step (%)",
            min_value=0.01,
            max_value=5.0,
            value=0.1,
            step=0.1,
        )

    if error_max <= error_min:

        st.error(
            "Error Max must be greater than Error Min."
        )

        st.stop()

    # --------------------------------------------------------
    # Base
    # --------------------------------------------------------

    base = prepare_fixed_base(
        ghi_values,
        actual_values,
        data["lat"],
        data["month_lookup"],
    )

    # --------------------------------------------------------
    # Optimization signature
    # --------------------------------------------------------

    fixed_signature = (
        file_id,
        dataframe_signature(
            ghi_input,
            GHI_COLS,
        ),
        series_signature(
            actual_input,
            "Actual",
        ),
        float(
            data["lat"]
        ),
        float(error_min),
        float(error_max),
        float(error_step),
    )

    if (
        st.session_state.get(
            "fixed_signature"
        )
        != fixed_signature
    ):

        with st.spinner(
            "Calculating best Error %..."
        ):

            (
                best_error,
                _,
                opt_table,
            ) = optimize_fixed(
                plant_df,
                cluster_df,
                base,
                error_min,
                error_max,
                error_step,
            )

        st.session_state.fixed_signature = (
            fixed_signature
        )

        st.session_state.fixed_best_error = (
            best_error
        )

        st.session_state.fixed_opt_table = (
            opt_table
        )

    best_error = (
        st.session_state.fixed_best_error
    )

    # --------------------------------------------------------
    # Final Error
    # --------------------------------------------------------

    selected_error = st.number_input(
        "Final Error (%)",
        min_value=float(
            error_min
        ),
        max_value=float(
            error_max
        ),
        value=float(
            np.clip(
                best_error,
                error_min,
                error_max,
            )
        ),
        step=float(
            error_step
        ),
        key="fixed_final_error",
    )

    # --------------------------------------------------------
    # ERROR APPLIED ONCE
    # --------------------------------------------------------

    (
        net_efficiency,
        effective_area,
        cluster_area,
    ) = calculate_cluster_area(
        plant_df,
        cluster_df,
        selected_error,
    )

    forecast = fixed_forecast(
        base,
        cluster_area,
    )

    actual = base["actual"]

    result_metrics = calculate_metrics(
        forecast,
        actual,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.subheader(
        "Results"
    )

    r1, r2, r3, r4 = st.columns(
        4
    )

    r1.metric(
        "Error %",
        f"{selected_error:.2f}%",
    )

    r2.metric(
        "Forecast Peak",
        f"{result_metrics['Forecast Peak']:.3f}",
    )

    r3.metric(
        "Actual Peak",
        f"{result_metrics['Actual Peak']:.3f}",
    )

    r4.metric(
        "Peak Error",
        f"{result_metrics['Peak Error %']:.2f}%",
    )

    st.plotly_chart(
        plot_forecast(
            forecast,
            actual,
            "Fixed Plant: Forecast vs Actual",
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Additional metrics
    # --------------------------------------------------------

    e1, e2 = st.columns(
        2
    )

    e1.metric(
        "Energy Error",
        f"{result_metrics['Energy Error %']:.2f}%",
    )

    e2.metric(
        "Effective Area",
        f"{np.sum(effective_area):,.2f} m²",
    )

    # --------------------------------------------------------
    # Optimization details
    # --------------------------------------------------------

    with st.expander(
        "Optimization Details"
    ):

        st.dataframe(
            st.session_state.fixed_opt_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TRACKING
# ============================================================

else:

    st.subheader(
        "Tracking Parameters"
    )

    # --------------------------------------------------------
    # Error / DHI
    # --------------------------------------------------------

    c1, c2 = st.columns(
        2
    )

    with c1:

        tracking_error = st.number_input(
            "Error (%)",
            min_value=0.0,
            max_value=50.0,
            value=4.9,
            step=0.1,
        )

    with c2:

        dhi_min = st.number_input(
            "DHI Min (%)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )

    # --------------------------------------------------------
    # Tracking base
    # --------------------------------------------------------

    tracking_base = (
        prepare_tracking_base(
            ghi_values,
            actual_values,
            data["backend_blocks"],
        )
    )

    # --------------------------------------------------------
    # ERROR APPLIED ONCE
    # --------------------------------------------------------

    (
        _,
        _,
        cluster_area,
    ) = calculate_cluster_area(
        plant_df,
        cluster_df,
        tracking_error,
    )

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    st.markdown(
        "#### Tracking Optimization Bounds"
    )

    b1, b2, b3 = st.columns(
        3
    )

    with b1:

        start_min = st.number_input(
            "Starting Block Min",
            min_value=0,
            max_value=95,
            value=10,
            step=1,
        )

        end_min = st.number_input(
            "Ending Block Min",
            min_value=1,
            max_value=96,
            value=65,
            step=1,
        )

    with b2:

        start_max = st.number_input(
            "Starting Block Max",
            min_value=1,
            max_value=95,
            value=30,
            step=1,
        )

        end_max = st.number_input(
            "Ending Block Max",
            min_value=1,
            max_value=96,
            value=80,
            step=1,
        )

    with b3:

        max_min = st.number_input(
            "Max Block Min",
            min_value=1,
            max_value=95,
            value=47,
            step=1,
        )

        max_max = st.number_input(
            "Max Block Max",
            min_value=1,
            max_value=95,
            value=53,
            step=1,
        )

    a1, a2 = st.columns(
        2
    )

    with a1:

        east_min = st.number_input(
            "East Limit Min",
            min_value=0,
            max_value=90,
            value=10,
            step=1,
        )

        east_max = st.number_input(
            "East Limit Max",
            min_value=1,
            max_value=90,
            value=70,
            step=1,
        )

    with a2:

        west_min = st.number_input(
            "West Limit Min",
            min_value=0,
            max_value=90,
            value=10,
            step=1,
        )

        west_max = st.number_input(
            "West Limit Max",
            min_value=1,
            max_value=90,
            value=70,
            step=1,
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    st.markdown(
        "#### Optimizer Settings"
    )

    o1, o2, o3 = st.columns(
        3
    )

    with o1:

        maxiter = st.number_input(
            "Optimization Iterations",
            min_value=1,
            max_value=200,
            value=40,
            step=1,
        )

    with o2:

        popsize = st.number_input(
            "Population Size",
            min_value=3,
            max_value=50,
            value=15,
            step=1,
        )

    with o3:

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    valid_bounds = (
        start_min < start_max
        and end_min < end_max
        and max_min < max_max
        and east_min < east_max
        and west_min < west_max
    )

    if not valid_bounds:

        st.error(
            "Invalid optimization bounds."
        )

        st.stop()

    # --------------------------------------------------------
    # Optimization signature
    # --------------------------------------------------------

    tracking_signature = (
        file_id,
        dataframe_signature(
            ghi_input,
            GHI_COLS,
        ),
        series_signature(
            actual_input,
            "Actual",
        ),
        tracking_error,
        dhi_min,
        start_min,
        start_max,
        end_min,
        end_max,
        max_min,
        max_max,
        east_min,
        east_max,
        west_min,
        west_max,
        maxiter,
        popsize,
        seed,
    )

    # --------------------------------------------------------
    # Run optimizer only when inputs change
    # --------------------------------------------------------

    if (
        st.session_state.get(
            "tracking_signature"
        )
        != tracking_signature
    ):

        bounds = [
            (
                dhi_min,
                10,
            ),
            (
                start_min,
                start_max,
            ),
            (
                end_min,
                end_max,
            ),
            (
                max_min,
                max_max,
            ),
            (
                east_min,
                east_max,
            ),
            (
                west_min,
                west_max,
            ),
        ]

        with st.spinner(
            "Optimizing tracking parameters..."
        ):

            try:

                (
                    best_params,
                    best_score,
                ) = optimize_tracking(
                    tracking_base,
                    cluster_area,
                    bounds,
                    maxiter,
                    popsize,
                    seed,
                )

            except Exception as e:

                st.error(
                    f"Tracking optimization failed: {e}"
                )

                st.stop()

        st.session_state.tracking_signature = (
            tracking_signature
        )

        st.session_state.tracking_best = (
            best_params
        )

        st.session_state.tracking_score = (
            best_score
        )

    best = (
        st.session_state.tracking_best
    )

    # --------------------------------------------------------
    # Final parameters
    # --------------------------------------------------------

    st.markdown(
        "#### Final Tracking Parameters"
    )

    t1, t2, t3 = st.columns(
        3
    )

    with t1:

        DHI = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            value=int(
                np.clip(
                    best["DHI"],
                    dhi_min,
                    10,
                )
            ),
            step=1,
            key="tracking_dhi",
        )

        start_block = st.number_input(
            "GHI Starting Block",
            min_value=start_min,
            max_value=start_max,
            value=int(
                np.clip(
                    best[
                        "GHI Starting Block"
                    ],
                    start_min,
                    start_max,
                )
            ),
            step=1,
            key="tracking_start",
        )

    with t2:

        end_block = st.number_input(
            "GHI Ending Block",
            min_value=end_min,
            max_value=end_max,
            value=int(
                np.clip(
                    best[
                        "GHI Ending Block"
                    ],
                    end_min,
                    end_max,
                )
            ),
            step=1,
            key="tracking_end",
        )

        max_block = st.number_input(
            "GHI Max Block",
            min_value=max_min,
            max_value=max_max,
            value=int(
                np.clip(
                    best[
                        "GHI Max Block"
                    ],
                    max_min,
                    max_max,
                )
            ),
            step=1,
            key="tracking_max",
        )

    with t3:

        east_limit = st.number_input(
            "Tracking East Limit",
            min_value=east_min,
            max_value=east_max,
            value=int(
                np.clip(
                    best[
                        "Tracking East Limit"
                    ],
                    east_min,
                    east_max,
                )
            ),
            step=1,
            key="tracking_east",
        )

        west_limit = st.number_input(
            "Tracking West Limit",
            min_value=west_min,
            max_value=west_max,
            value=int(
                np.clip(
                    best[
                        "Tracking West Limit"
                    ],
                    west_min,
                    west_max,
                )
            ),
            step=1,
            key="tracking_west",
        )

    # --------------------------------------------------------
    # Parameter validation
    # --------------------------------------------------------

    if not (
        start_block
        < max_block
        < end_block
    ):

        st.error(
            "Required condition: "
            "GHI Starting Block < "
            "GHI Max Block < "
            "GHI Ending Block"
        )

        st.stop()

    # --------------------------------------------------------
    # Final calculation
    # --------------------------------------------------------

    final = tracking_forecast(
        tracking_base,
        cluster_area,
        DHI,
        start_block,
        end_block,
        max_block,
        east_limit,
        west_limit,
    )

    if final is None:

        st.error(
            "Unable to calculate Tracking forecast."
        )

        st.stop()

    forecast = final[
        "forecast"
    ]

    actual = tracking_base[
        "actual"
    ]

    result_metrics = calculate_metrics(
        forecast,
        actual,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.subheader(
        "Results"
    )

    r1, r2, r3, r4 = st.columns(
        4
    )

    r1.metric(
        "Error %",
        f"{tracking_error:.2f}%",
    )

    r2.metric(
        "Forecast Peak",
        f"{result_metrics['Forecast Peak']:.3f}",
    )

    r3.metric(
        "Actual Peak",
        f"{result_metrics['Actual Peak']:.3f}",
    )

    r4.metric(
        "Peak Error",
        f"{result_metrics['Peak Error %']:.2f}%",
    )

    st.plotly_chart(
        plot_forecast(
            forecast,
            actual,
            "Tracking Plant: Forecast vs Actual",
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Extra metrics
    # --------------------------------------------------------

    e1, e2 = st.columns(
        2
    )

    e1.metric(
        "Energy Error",
        f"{result_metrics['Energy Error %']:.2f}%",
    )

    e2.metric(
        "Optimization Score",
        f"{st.session_state.tracking_score:.6f}",
    )

    # --------------------------------------------------------
    # Tracking angle graph
    # --------------------------------------------------------

    with st.expander(
        "Tracking Angles"
    ):

        x = np.arange(
            len(
                final["zenith"]
            )
        )

        fig_angle = go.Figure()

        fig_angle.add_trace(
            go.Scatter(
                x=x,
                y=final["zenith"],
                mode="lines",
                name="Zenith Angle",
            )
        )

        fig_angle.add_trace(
            go.Scatter(
                x=x,
                y=final["panel"],
                mode="lines",
                name="Panel Angle",
            )
        )

        fig_angle.update_layout(
            height=350,
            xaxis_title="Block",
            yaxis_title="Angle (°)",
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_angle,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Optimization result
    # --------------------------------------------------------

    with st.expander(
        "Automatic Optimization Result"
    ):

        result_df = pd.DataFrame(
            [
                {
                    "Parameter": key,
                    "Value": value,
                }
                for key, value
                in best.items()
            ]
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "The uploaded workbook is the default input source. "
    "Edited GHI and Actual values are used immediately."
)
