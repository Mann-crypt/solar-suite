# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# ============================================================

import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.optimize import differential_evolution


# ============================================================
# PAGE
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

PLANTS = ["Fixed", "Tracking"]

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

GHI_COLS = [f"GHI {c}" for c in CLUSTERS]

POA_COLS = [
    "POA fixed",
    "POA Fixed-C12",
    "POA Fixed-C13",
    "POA Fixed-C14",
    "POA Fixed-C15",
]

POWER_COLS = [
    f"CL{i}_Fixed Power=I*Ƞ*A"
    for i in range(1, 6)
]

TOTAL_POWER = "Total Power (CL1+CL2+…)"


# DHI, Start, End, Max, East, West
TRACKING_BOUNDS = [
    (0, 10),
    (10, 30),
    (65, 80),
    (47, 53),
    (10, 70),
    (10, 70),
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .app-title {
        font-size: 30px;
        font-weight: 750;
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
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 90px;
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
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 7px;
    }

    div[data-testid="stSegmentedControl"] {
        width: 100%;
    }

    div[data-testid="stSegmentedControl"] button {
        flex: 1;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "calculated": False,
    "calculation_data": None,
    "plant_type": "Fixed",
    "calculated_plant_type": None,
    "input_df": None,
    "last_file_hash": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_results():
    st.session_state.calculated = False
    st.session_state.calculation_data = None
    st.session_state.calculated_plant_type = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Automatic loss and tracking parameter optimization"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def num_series(x):
    return pd.to_numeric(x, errors="coerce").fillna(0)


def num_array(x):
    return num_series(pd.Series(x)).to_numpy(dtype=float)


def safe_float(x, default=0.0):
    try:
        x = float(pd.to_numeric(x, errors="coerce"))
        return x if np.isfinite(x) else default
    except Exception:
        return default


def read_sheet(file_bytes, **kwargs):
    """Read one Excel sheet and guarantee a DataFrame."""
    df = pd.read_excel(io.BytesIO(file_bytes), **kwargs)

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"Excel sheet did not return a table. Received {type(df).__name__}."
        )

    return df.copy()


def file_hash(file):
    return hashlib.sha256(file.getvalue()).hexdigest()


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )
    return df


def require_columns(df, columns, sheet_name):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{sheet_name} is missing: {', '.join(missing)}"
        )


# ============================================================
# WORKBOOK
# ============================================================

@st.cache_data(show_spinner=False, max_entries=3)
def load_workbook(file_bytes):

    sheets = {}

    sheets["area"] = read_sheet(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    sheets["cluster"] = read_sheet(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    sheets["ghi"] = read_sheet(
        file_bytes,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    sheets["forecast_config"] = read_sheet(
        file_bytes,
        sheet_name="Forecast Config",
        header=8,
    )

    sheets["tilt"] = read_sheet(
        file_bytes,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    sheets["fixed"] = read_sheet(
        file_bytes,
        sheet_name="Fixed-C11",
        header=1,
    )

    sheets["tracking"] = read_sheet(
        file_bytes,
        sheet_name="Tracking",
        header=1,
    )

    sheets["backend"] = {}

    for cluster in CLUSTERS:
        sheets["backend"][cluster] = read_sheet(
            file_bytes,
            sheet_name=f"Backend Cal {cluster}",
        )

    return sheets


# ============================================================
# PREPARE AREA
# ============================================================

def prepare_area(df):

    df = clean_columns(df)

    required = [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]

    require_columns(
        df,
        required,
        "Area & Efficiency",
    )

    if "S.No." in df.columns:
        mask = df["S.No."].isna()

        if mask.any():
            df = df.iloc[:np.flatnonzero(mask)[0]]

    for col in required:
        df[col] = num_series(df[col])

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df.reset_index(drop=True)


# ============================================================
# CLUSTER TABLE
# ============================================================

def prepare_cluster(df):

    df = clean_columns(df)

    require_columns(
        df,
        ["Clusters"],
        "Cluster table",
    )

    mask = df["Clusters"].isna()

    if mask.any():
        df = df.iloc[:np.flatnonzero(mask)[0]]

    return df.reset_index(drop=True)


# ============================================================
# GHI
# ============================================================

def prepare_ghi(df):

    df = clean_columns(df)

    require_columns(
        df,
        GHI_COLS,
        "Result",
    )

    for col in GHI_COLS:
        df[col] = num_series(df[col])

    return df.reset_index(drop=True)


# ============================================================
# LATITUDE
# ============================================================

def prepare_latitude(df):

    df = clean_columns(df)

    require_columns(
        df,
        ["Lat"],
        "Forecast Config",
    )

    lat = safe_float(
        df.loc[0, "Lat"],
        np.nan,
    )

    if not np.isfinite(lat):
        raise ValueError("Invalid latitude in Forecast Config.")

    return lat


# ============================================================
# TILT
# ============================================================

def prepare_tilt(df):

    df = clean_columns(df)

    require_columns(
        df,
        ["Fixed"],
        "Config Tilt Angle",
    )

    # Find month column robustly.
    month_col = None

    for col in df.columns:
        if str(col).strip().lower() == "month":
            month_col = col
            break

    if month_col is None:

        # Original workbook sometimes stores Month
        # under unnamed columns.
        for col in df.columns:
            values = df[col].astype(str).str.strip().str.lower()

            if values.isin(
                [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ]
            ).any():
                month_col = col
                break

    if month_col is None:
        raise ValueError(
            "Month column not found in Config Tilt Angle."
        )

    result = {}

    for _, row in df.iterrows():

        month = str(row[month_col]).strip()

        tilt = safe_float(
            row["Fixed"],
            np.nan,
        )

        if month.lower() != "nan" and np.isfinite(tilt):
            result[month] = tilt

    if not result:
        raise ValueError(
            "No valid monthly tilt values found."
        )

    return result


# ============================================================
# FIXED ACTUAL
# ============================================================

def prepare_fixed(df):

    df = clean_columns(df)

    require_columns(
        df,
        ["Actual"],
        "Fixed-C11",
    )

    if "Date" in df.columns:

        mask = df["Date"].isna()

        if mask.any():
            df = df.iloc[:np.flatnonzero(mask)[0]]

    df["Actual"] = num_series(df["Actual"])

    return df.reset_index(drop=True)


# ============================================================
# INPUT DATA
# ============================================================

def build_input(ghi, fixed):

    n = min(
        len(ghi),
        len(fixed),
    )

    if n <= 0:
        raise ValueError("No GHI / Actual data found.")

    result = ghi[GHI_COLS].iloc[:n].copy()

    result["Actual"] = fixed["Actual"].iloc[:n].to_numpy()

    return result.reset_index(drop=True)


def apply_input(input_df, ghi, fixed):

    required = GHI_COLS + ["Actual"]

    require_columns(
        input_df,
        required,
        "Input data",
    )

    n = len(input_df)

    if n == 0:
        raise ValueError("Input data is empty.")

    if n > min(len(ghi), len(fixed)):
        raise ValueError(
            "Edited input contains more rows than the original data."
        )

    ghi = ghi.iloc[:n].copy()
    fixed = fixed.iloc[:n].copy()

    for col in GHI_COLS:
        ghi[col] = num_series(input_df[col]).to_numpy()

    fixed["Actual"] = num_series(
        input_df["Actual"]
    ).to_numpy()

    return ghi.reset_index(drop=True), fixed.reset_index(drop=True)


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def prepare_geometry(
    fixed,
    ghi,
    lat,
    tilt_lookup,
):

    n = min(
        len(fixed),
        len(ghi),
    )

    if n <= 0:
        raise ValueError("No data available for geometry.")

    df = fixed.iloc[:n].copy()
    ghi = ghi.iloc[:n].copy()

    # Preserve existing calculation approach.
    today = pd.Timestamp.today().normalize()

    df["Date"] = today

    day_number = today.dayofyear

    declination = (
        23.45
        * np.sin(
            np.radians(
                360 * (284 + day_number) / 365
            )
        )
    )

    df["Declination Angle ∆"] = declination

    df["Elevation angle a"] = (
        90
        - lat
        + declination
    )

    month_name = today.strftime("%B")

    tilt = tilt_lookup.get(
        month_name,
        tilt_lookup.get(
            month_name.strip(),
            0,
        ),
    )

    df["Tilt Angle b"] = safe_float(
        tilt,
        0,
    )

    df["a+b"] = (
        df["Elevation angle a"]
        + df["Tilt Angle b"]
    )

    df["SIN(a+b)"] = np.sin(
        np.radians(df["a+b"])
    )

    df["Sin(a)"] = np.sin(
        np.radians(df["Elevation angle a"])
    )

    sin_a = df["Sin(a)"].replace(
        0,
        np.nan,
    )

    # --------------------------------------------------------
    # Generate ALL cluster calculations dynamically.
    # This fixes the C15 / CL5 error.
    # --------------------------------------------------------

    for cluster, ghi_col, poa_col in zip(
        CLUSTERS,
        GHI_COLS,
        POA_COLS,
    ):

        ghi_values = num_series(
            ghi[ghi_col]
        )

        suffix = "" if cluster == "C11" else f"-{cluster.replace('C', 'CL')}"

        ghi_sina_col = (
            "GHI*sin(a)"
            if cluster == "C11"
            else f"GHI*sin(a)-{suffix[1:]}"
        )

        ghi_sinab_col = (
            "GHI*sin(a+b)"
            if cluster == "C11"
            else f"GHI*sin(a+b)-{suffix[1:]}"
        )

        df[ghi_sina_col] = (
            ghi_values
            * df["Sin(a)"]
        )

        df[ghi_sinab_col] = (
            ghi_values
            * df["SIN(a+b)"]
        )

        df[poa_col] = (
            df[ghi_sinab_col]
            / sin_a
        )

    # Make sure every required POA column exists.
    for col in POA_COLS:
        if col not in df.columns:
            raise ValueError(
                f"Unable to create required column: {col}"
            )

    return df.reset_index(drop=True)


# ============================================================
# EFFECTIVE AREA
# ============================================================

def effective_area(
    original,
    cluster_original,
    error,
):

    df = original.copy()
    weights = cluster_original.copy()

    error = safe_float(error)

    df["Error %"] = error

    df["Net Efficiency (%)"] = (
        num_series(
            df["Standard PV Efficiency (%)"]
        )
        - error
    )

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

    cluster_sum = (
        df.groupby("Clusters")["Eff Area"]
        .sum()
    )

    weights["Eff Area(m2)"] = (
        weights["Clusters"]
        .map(cluster_sum)
        .fillna(0)
    )

    return df, weights


# ============================================================
# FIXED POWER
# ============================================================

def fixed_power(df, weights):

    result = df.copy()

    if len(weights) < 5:
        raise ValueError(
            "At least five cluster weights are required."
        )

    for i, poa_col in enumerate(POA_COLS):

        if poa_col not in result.columns:
            raise ValueError(
                f"Missing POA column: {poa_col}"
            )

        area = safe_float(
            weights.iloc[i]["Eff Area(m2)"]
        )

        result[POWER_COLS[i]] = (
            num_series(result[poa_col])
            * area
            / 1_000_000
        )

    result[TOTAL_POWER] = result[POWER_COLS].sum(axis=1)

    return result


# ============================================================
# ERROR OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_error(
    original,
    cluster_original,
    df_fix,
):

    actual = num_array(df_fix["Actual"])

    if len(actual) == 0:
        raise ValueError("Actual data is empty.")

    actual_peak = actual.max()

    if actual_peak <= 0:
        raise ValueError(
            "Actual peak must be greater than zero."
        )

    results = []

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        _, weights = effective_area(
            original,
            cluster_original,
            error,
        )

        calculated = fixed_power(
            df_fix,
            weights,
        )

        forecast = num_array(
            calculated[TOTAL_POWER]
        )

        if len(forecast) == 0:
            continue

        calculated_peak = forecast.max()

        peak_error = abs(
            calculated_peak - actual_peak
        )

        peak_error_pct = (
            peak_error
            / actual_peak
            * 100
        )

        results.append(
            [
                round(error, 1),
                calculated_peak,
                actual_peak,
                peak_error,
                peak_error_pct,
            ]
        )

    if not results:
        raise ValueError(
            "Error optimization returned no valid result."
        )

    result_df = pd.DataFrame(
        results,
        columns=[
            "Error %",
            "Calculated Peak",
            "Actual Peak",
            "Peak Error",
            "Peak Error %",
        ],
    )

    best = result_df.loc[
        result_df["Peak Error"].idxmin()
    ]

    return (
        float(best["Error %"]),
        result_df,
    )


# ============================================================
# TRACKING DATA
# ============================================================

def tracking_blocks(backend):

    df = backend["C11"].copy()

    require_columns(
        df,
        ["Block No."],
        "Backend Cal C11",
    )

    blocks = num_array(
        df["Block No."]
    )

    if len(blocks) == 0:
        raise ValueError(
            "No tracking blocks found."
        )

    return blocks


def tracking_weights(weights):

    require_columns(
        weights,
        ["Eff Area(m2)"],
        "Cluster table",
    )

    values = (
        pd.to_numeric(
            weights["Eff Area(m2)"].iloc[:5],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    if len(values) < 5:
        raise ValueError(
            "Tracking requires five cluster weights."
        )

    return values


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    dhi,
    start,
    end,
    max_block,
    east,
    west,
    blocks,
    ghi,
    weights,
):

    if not (
        start < max_block < end
    ):
        return None

    d1 = start - 1 - max_block
    d2 = end + 1 - max_block

    if d1 == 0 or d2 == 0:
        return None

    m1 = 90 / d1
    m2 = 90 / d2

    zenith = np.where(
        blocks <= max_block,
        np.minimum(
            89,
            m1 * (blocks - max_block),
        ),
        np.minimum(
            89,
            m2 * (blocks - max_block),
        ),
    )

    panel = np.where(
        blocks < max_block,
        np.minimum(
            zenith,
            abs(east),
        ),
        np.where(
            (
                (blocks > max_block)
                & (zenith > west)
            ),
            west,
            zenith,
        ),
    )

    cos_panel = np.clip(
        np.cos(np.radians(panel)),
        1e-6,
        None,
    )

    dhi_component = (
        ghi
        * dhi
        / 100
    )

    dni = (
        ghi
        - dhi_component
    ) / cos_panel[:, None]

    power_matrix = (
        dni
        * weights[None, :]
        / 1_000_000
    )

    forecast = power_matrix.sum(axis=1)

    return (
        forecast,
        power_matrix,
        zenith,
        panel,
        dni,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def optimize_tracking(
    blocks_tuple,
    ghi_tuple,
    actual_tuple,
    weights_tuple,
):

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    ghi = np.asarray(
        ghi_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    weights = np.asarray(
        weights_tuple,
        dtype=float,
    )

    n = min(
        len(blocks),
        len(ghi),
        len(actual),
    )

    blocks = blocks[:n]
    ghi = ghi[:n]
    actual = actual[:n]

    mask = actual != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values for Tracking."
        )

    actual_day = actual[mask]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    def objective(x):

        values = np.rint(x).astype(int)

        result = calculate_tracking(
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            blocks,
            ghi,
            weights,
        )

        if result is None:
            return 1e9

        prediction = result[0]

        if not np.all(np.isfinite(prediction)):
            return 1e9

        prediction_day = prediction[mask]

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

    return {
        "DHI": int(best[0]),
        "GHI Starting Block": int(best[1]),
        "GHI Ending Block": int(best[2]),
        "GHI Max Block": int(best[3]),
        "Tracking East Limit": int(best[4]),
        "Tracking West Limit": int(best[5]),
    }


# ============================================================
# METRICS
# ============================================================

def metrics(actual, forecast):

    actual = num_array(actual)
    forecast = num_array(forecast)

    n = min(
        len(actual),
        len(forecast),
    )

    if n == 0:
        return 0, 0

    return (
        float(actual[:n].max()),
        float(forecast[:n].max()),
    )


# ============================================================
# GRAPH
# ============================================================

def graph(actual, forecast, title):

    actual = num_array(actual)
    forecast = num_array(forecast)

    n = min(
        len(actual),
        len(forecast),
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual[:n],
            mode="lines",
            name="Actual",
            line=dict(width=2.2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=forecast[:n],
            mode="lines",
            name="Forecast",
            line=dict(width=2.2),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.01,
        ),
        height=430,
        template="plotly_white",
        hovermode="x unified",
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
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📂 Input Data</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload Solar Excel File",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)

if uploaded is None:

    st.info(
        "Upload the Solar Excel file to start."
    )

    st.stop()


# ============================================================
# FILE CHANGE
# ============================================================

current_hash = file_hash(uploaded)

if (
    st.session_state.last_file_hash
    != current_hash
):

    st.session_state.last_file_hash = current_hash

    st.session_state.input_df = None

    st.session_state.plant_type = "Fixed"

    reset_results()

    for key in [
        "plant_type_selector",
        "solar_input_editor",
    ]:

        if key in st.session_state:
            del st.session_state[key]


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    workbook = load_workbook(
        uploaded.getvalue()
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


# ============================================================
# PREPARE INPUTS
# ============================================================

try:

    area = prepare_area(
        workbook["area"]
    )

    cluster = prepare_cluster(
        workbook["cluster"]
    )

    ghi_raw = prepare_ghi(
        workbook["ghi"]
    )

    fixed_raw = prepare_fixed(
        workbook["fixed"]
    )

    lat = prepare_latitude(
        workbook["forecast_config"]
    )

    tilt = prepare_tilt(
        workbook["tilt"]
    )

except Exception as e:

    st.error(
        f"Input preparation failed: {e}"
    )

    st.stop()


# ============================================================
# INPUT EDITOR
# ============================================================

if st.session_state.input_df is None:

    try:

        st.session_state.input_df = build_input(
            ghi_raw,
            fixed_raw,
        )

    except Exception as e:

        st.error(
            f"Unable to build input data: {e}"
        )

        st.stop()


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
        for col in GHI_COLS + ["Actual"]
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
    PLANTS,
    default=st.session_state.plant_type,
    selection_mode="single",
    width="stretch",
    label_visibility="collapsed",
    key="plant_type_selector",
)

if plant_type not in PLANTS:
    plant_type = "Fixed"

if plant_type != st.session_state.plant_type:

    st.session_state.plant_type = plant_type

    reset_results()

    st.rerun()


# ============================================================
# RUN
# ============================================================

st.markdown("")

run = st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    width="stretch",
)


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

if run:

    try:

        with st.spinner(
            "Running automatic calculation..."
        ):

            # Save edited input.
            st.session_state.input_df = input_df.copy()

            # Apply user-edited data.
            ghi, fixed_raw_user = apply_input(
                input_df,
                ghi_raw,
                fixed_raw,
            )

            # Solar geometry.
            fixed = prepare_geometry(
                fixed_raw_user,
                ghi,
                lat,
                tilt,
            )

            # ------------------------------------------------
            # AUTOMATIC ERROR OPTIMIZATION
            # ------------------------------------------------

            best_error, error_table = optimize_error(
                area,
                cluster,
                fixed,
            )

            # Apply optimized error.
            area_final, weights_final = effective_area(
                area,
                cluster,
                best_error,
            )

            # Fixed forecast.
            fixed_result = fixed_power(
                fixed,
                weights_final,
            )

            # ------------------------------------------------
            # TRACKING OPTIMIZATION
            # ------------------------------------------------

            tracking_params = None

            if plant_type == "Tracking":

                blocks = tracking_blocks(
                    workbook["backend"]
                )

                n = min(
                    len(blocks),
                    len(ghi),
                    len(fixed),
                )

                blocks = blocks[:n]

                ghi_matrix = np.column_stack(
                    [
                        num_array(ghi[c])[:n]
                        for c in GHI_COLS
                    ]
                )

                actual_tracking = num_array(
                    fixed["Actual"]
                )[:n]

                weights_tracking = tracking_weights(
                    weights_final
                )

                tracking_params = optimize_tracking(
                    tuple(blocks.tolist()),
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
                        weights_tracking.tolist()
                    ),
                )

            # Save all calculation data.
            st.session_state.calculation_data = {
                "area": area,
                "cluster": cluster,
                "area_final": area_final,
                "weights_final": weights_final,
                "ghi": ghi,
                "fixed": fixed,
                "fixed_result": fixed_result,
                "best_error": best_error,
                "error_table": error_table,
                "tracking_params": tracking_params,
            }

            st.session_state.calculated = True

            st.session_state.calculated_plant_type = (
                plant_type
            )

        st.success(
            "Automatic calculation completed successfully."
        )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )

        st.stop()


# ============================================================
# WAIT
# ============================================================

if not st.session_state.calculated:

    st.caption(
        "Edit the GHI / Actual values, select the plant type, "
        "then click Run Automatic Calculation."
    )

    st.stop()


# ============================================================
# DATA
# ============================================================

data = st.session_state.calculation_data

if data is None:
    reset_results()
    st.stop()


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)

error_key = f"error_parameter_{current_hash}"

error_value = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=20.0,
    value=float(data["best_error"]),
    step=0.1,
    format="%.1f",
    key=error_key,
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

if plant_type == "Tracking":

    params = data["tracking_params"]

    if params is None:
        st.error(
            "Tracking parameters are unavailable."
        )
        st.stop()

    st.markdown(
        "#### Tracking Parameters"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi = st.number_input(
            "DHI (%)",
            0,
            100,
            int(params["DHI"]),
            1,
            key=f"dhi_{current_hash}",
        )

        start = st.number_input(
            "GHI Starting Block",
            0,
            95,
            int(params["GHI Starting Block"]),
            1,
            key=f"start_{current_hash}",
        )

    with c2:

        end = st.number_input(
            "GHI Ending Block",
            1,
            96,
            int(params["GHI Ending Block"]),
            1,
            key=f"end_{current_hash}",
        )

        max_block = st.number_input(
            "GHI Max Block",
            0,
            95,
            int(params["GHI Max Block"]),
            1,
            key=f"max_{current_hash}",
        )

    with c3:

        east = st.number_input(
            "Tracking East Limit",
            0,
            90,
            int(params["Tracking East Limit"]),
            1,
            key=f"east_{current_hash}",
        )

        west = st.number_input(
            "Tracking West Limit",
            0,
            90,
            int(params["Tracking West Limit"]),
            1,
            key=f"west_{current_hash}",
        )


# ============================================================
# FINAL FORECAST
#
# IMPORTANT:
# No Differential Evolution here.
# Parameter edits are cheap recalculations only.
# ============================================================

try:

    _, weights = effective_area(
        data["area"],
        data["cluster"],
        error_value,
    )

    fixed_result = fixed_power(
        data["fixed"],
        weights,
    )

    if plant_type == "Fixed":

        actual = num_array(
            data["fixed"]["Actual"]
        )

        forecast = num_array(
            fixed_result[TOTAL_POWER]
        )

        title = (
            "Fixed Plant | Actual vs Forecast"
        )

    else:

        if not (
            start < max_block < end
        ):

            st.error(
                "Tracking parameters must satisfy: "
                "GHI Starting Block < GHI Max Block < GHI Ending Block."
            )

            st.stop()

        blocks = tracking_blocks(
            workbook["backend"]
        )

        n = min(
            len(blocks),
            len(data["ghi"]),
            len(data["fixed"]),
        )

        blocks = blocks[:n]

        ghi_matrix = np.column_stack(
            [
                num_array(
                    data["ghi"][c]
                )[:n]
                for c in GHI_COLS
            ]
        )

        actual = num_array(
            data["fixed"]["Actual"]
        )[:n]

        weights = tracking_weights(
            weights
        )

        tracking_result = calculate_tracking(
            int(dhi),
            int(start),
            int(end),
            int(max_block),
            int(east),
            int(west),
            blocks,
            ghi_matrix,
            weights,
        )

        if tracking_result is None:
            st.error(
                "Invalid tracking parameters."
            )
            st.stop()

        forecast = tracking_result[0]

        title = (
            "Tracking Plant | Actual vs Forecast"
        )

except Exception as e:

    st.error(
        f"Forecast calculation failed: {e}"
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

actual_peak, forecast_peak = metrics(
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
            <div class="metric-label">Actual Peak</div>
            <div class="metric-value">
                {actual_peak:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Forecast Peak</div>
            <div class="metric-value">
                {forecast_peak:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)

st.plotly_chart(
    graph(
        actual,
        forecast,
        title,
    ),
    width="stretch",
    config={
        "displayModeBar": False,
        "responsive": True,
    },
)
