# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# Compact + Cloud Safe
# ============================================================

import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

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


TRACKING_BOUNDS = [
    (0, 10),     # DHI
    (10, 30),    # GHI starting block
    (65, 80),    # GHI ending block
    (47, 53),    # GHI max block
    (10, 70),    # East limit
    (10, 70),    # West limit
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

    .title {
        font-size: 30px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 15px;
    }

    .section {
        font-size: 19px;
        font-weight: 700;
        margin: 17px 0 9px 0;
    }

    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 13px 16px;
        min-height: 85px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
    }

    .metric-value {
        font-size: 23px;
        font-weight: 750;
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

defaults = {
    "calculated": False,
    "calculation_data": None,
    "plant_type": "Fixed",
    "calculated_plant_type": None,
    "input_df": None,
    "file_hash": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_results():
    st.session_state.calculated = False
    st.session_state.calculation_data = None
    st.session_state.calculated_plant_type = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Automatic optimization with editable final parameters'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def num_series(x):
    return pd.to_numeric(x, errors="coerce").fillna(0)


def num_array(x):
    return num_series(pd.Series(x)).to_numpy(dtype=float)


def safe_float(x, default=0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def read_excel_bytes(data, **kwargs):
    return pd.read_excel(io.BytesIO(data), **kwargs)


def ensure_df(obj, name):
    """
    Prevent "'str' object has no attribute copy".
    Every workbook object used by the calculation must be a DataFrame.
    """
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"Workbook sheet '{name}' was not loaded as a DataFrame "
            f"(received {type(obj).__name__})."
        )

    return obj.copy()


def hash_file(uploaded):
    return hashlib.sha256(
        uploaded.getvalue()
    ).hexdigest()


# ============================================================
# LOAD WORKBOOK
# ============================================================

@st.cache_data(show_spinner=False, max_entries=3)
def load_workbook(data):

    result = {}

    sheets = {
        "area": (
            "Area & Efficiency",
            {"header": [1], "usecols": range(12)},
        ),
        "cluster": (
            "Area & Efficiency",
            {"header": 1, "usecols": [14, 15]},
        ),
        "ghi": (
            "Result",
            {"usecols": [0, 1, 2, 3, 4, 5]},
        ),
        "forecast_config": (
            "Forecast Config",
            {"header": [8]},
        ),
        "tilt": (
            "Config Tilt Angle",
            {"header": [7]},
        ),
        "fixed": (
            "Fixed-C11",
            {"header": [1]},
        ),
        "tracking": (
            "Tracking",
            {"header": 1},
        ),
    }

    for key, (sheet, kwargs) in sheets.items():

        try:
            result[key] = read_excel_bytes(
                data,
                sheet_name=sheet,
                **kwargs,
            )

        except Exception as e:
            raise ValueError(
                f"Unable to read sheet '{sheet}': {e}"
            )

    result["backend"] = {}

    for cluster in CLUSTERS:

        sheet = f"Backend Cal {cluster}"

        try:
            result["backend"][cluster] = read_excel_bytes(
                data,
                sheet_name=sheet,
            )

        except Exception as e:
            raise ValueError(
                f"Unable to read sheet '{sheet}': {e}"
            )

    return result


# ============================================================
# PREPARE AREA
# ============================================================

def prepare_area(df):

    df = ensure_df(df, "Area & Efficiency")

    df.columns = (
        df.columns.astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    if "S.No." in df.columns:

        m = df["S.No."].isna()

        if m.any():
            df = df.iloc[:np.flatnonzero(m)[0]].copy()

    required = [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]

    for col in required:

        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}' in Area & Efficiency."
            )

        df[col] = num_series(df[col])

    if "Clusters" not in df.columns:
        raise ValueError(
            "Missing 'Clusters' column in Area & Efficiency."
        )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df.reset_index(drop=True)


# ============================================================
# CLUSTER TABLE
# ============================================================

def prepare_cluster(df):

    df = ensure_df(df, "Area & Efficiency cluster table")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Clusters" not in df.columns:
        raise ValueError(
            "Missing 'Clusters' column in cluster table."
        )

    return df.dropna(
        subset=["Clusters"]
    ).reset_index(drop=True)


# ============================================================
# GHI
# ============================================================

def prepare_ghi(df):

    df = ensure_df(df, "Result")

    for col in GHI_COLS:

        if col not in df.columns:
            raise ValueError(
                f"Missing GHI column '{col}' in Result."
            )

        df[col] = num_series(df[col])

    return df.reset_index(drop=True)


# ============================================================
# LATITUDE
# ============================================================

def prepare_latitude(df):

    df = ensure_df(df, "Forecast Config")

    if "Lat" not in df.columns:
        raise ValueError(
            "Latitude column 'Lat' not found."
        )

    lat = pd.to_numeric(
        df["Lat"].iloc[0],
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

    df = ensure_df(df, "Config Tilt Angle")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Fixed" not in df.columns:
        raise ValueError(
            "Column 'Fixed' not found in Config Tilt Angle."
        )

    # Find Month column robustly
    month_col = None

    for col in df.columns:
        if str(col).strip().lower() == "month":
            month_col = col
            break

    if month_col is None:

        for col in df.columns:
            values = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

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
            ).sum() >= 2:

                month_col = col
                break

    if month_col is None:
        raise ValueError(
            "Month column not found in Config Tilt Angle."
        )

    out = {}

    for _, row in df.iterrows():

        month = str(
            row[month_col]
        ).strip()

        if month.lower() in {
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
        }:

            value = safe_float(
                row["Fixed"],
                0,
            )

            out[month] = value

    if not out:
        raise ValueError(
            "No valid monthly Fixed tilt values found."
        )

    return out


# ============================================================
# FIXED ACTUAL
# ============================================================

def prepare_fixed(df):

    df = ensure_df(df, "Fixed-C11")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Actual" not in df.columns:
        raise ValueError(
            "Actual column not found in Fixed-C11."
        )

    df["Actual"] = num_series(
        df["Actual"]
    )

    return df.reset_index(drop=True)


# ============================================================
# BUILD INPUT
# ============================================================

def build_input(ghi, fixed):

    n = min(
        len(ghi),
        len(fixed),
    )

    if n <= 0:
        raise ValueError(
            "No GHI / Actual data available."
        )

    out = pd.DataFrame()

    for col in GHI_COLS:
        out[col] = (
            num_series(
                ghi[col]
            )
            .iloc[:n]
            .to_numpy()
        )

    out["Actual"] = (
        num_series(
            fixed["Actual"]
        )
        .iloc[:n]
        .to_numpy()
    )

    return out


def apply_input(inp, ghi, fixed):

    required = GHI_COLS + ["Actual"]

    missing = [
        c for c in required
        if c not in inp.columns
    ]

    if missing:
        raise ValueError(
            "Missing input columns: "
            + ", ".join(missing)
        )

    n = len(inp)

    if n == 0:
        raise ValueError(
            "Input table is empty."
        )

    if n > len(ghi) or n > len(fixed):
        raise ValueError(
            "Edited input contains more rows than the original data."
        )

    ghi = ghi.iloc[:n].copy()
    fixed = fixed.iloc[:n].copy()

    for col in GHI_COLS:
        ghi[col] = num_series(
            inp[col]
        ).to_numpy()

    fixed["Actual"] = num_series(
        inp["Actual"]
    ).to_numpy()

    return ghi.reset_index(drop=True), fixed.reset_index(drop=True)


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def prepare_geometry(fixed, ghi, lat, tilt_lookup):

    fixed = ensure_df(
        fixed,
        "Prepared Fixed data",
    )

    ghi = ensure_df(
        ghi,
        "Prepared GHI data",
    )

    n = min(
        len(fixed),
        len(ghi),
    )

    if n <= 0:
        raise ValueError(
            "No data available for solar geometry."
        )

    fixed = fixed.iloc[:n].copy()
    ghi = ghi.iloc[:n].copy()

    # Preserve existing calculation logic
    today = pd.Timestamp.today().normalize()

    fixed["Date"] = today

    day_number = (
        today.dayofyear
    )

    declination = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + day_number)
                / 365
            )
        )
    )

    fixed["Declination Angle ∆"] = declination

    fixed["Elevation angle a"] = (
        90
        - lat
        + declination
    )

    month_name = today.strftime("%B")

    tilt = safe_float(
        tilt_lookup.get(
            month_name,
            0,
        ),
        0,
    )

    fixed["Tilt Angle b"] = tilt

    fixed["a+b"] = (
        fixed["Elevation angle a"]
        + fixed["Tilt Angle b"]
    )

    fixed["SIN(a+b)"] = np.sin(
        np.radians(
            fixed["a+b"]
        )
    )

    fixed["Sin(a)"] = np.sin(
        np.radians(
            fixed["Elevation angle a"]
        )
    )

    sin_a = (
        fixed["Sin(a)"]
        .replace(0, np.nan)
    )

    # --------------------------------------------------------
    # ALL FIVE CLUSTERS
    # --------------------------------------------------------

    for i, cluster in enumerate(CLUSTERS):

        ghi_col = GHI_COLS[i]

        if i == 0:
            suffix = ""
            poa_col = "POA fixed"
        else:
            suffix = f"-CL{i + 1}"
            poa_col = f"POA Fixed-C{11 + (i - 1) * 1 + 1}"

            # Explicit names used by workbook
            poa_col = POA_COLS[i]

        fixed[
            f"GHI*sin(a){suffix}"
        ] = (
            ghi[ghi_col]
            * fixed["Sin(a)"]
        )

        fixed[
            f"GHI*sin(a+b){suffix}"
        ] = (
            ghi[ghi_col]
            * fixed["SIN(a+b)"]
        )

        fixed[poa_col] = (
            fixed[
                f"GHI*sin(a+b){suffix}"
            ]
            / sin_a
        )

    # Ensure exact expected names
    fixed["POA fixed"] = (
        fixed["GHI*sin(a+b)"]
        / sin_a
    )

    for i in range(1, 5):

        cluster_no = i + 1

        fixed[
            f"POA Fixed-C{cluster_no}"
        ] = (
            fixed[
                f"GHI*sin(a+b)-CL{cluster_no}"
            ]
            / sin_a
        )

    return fixed.reset_index(drop=True)


# ============================================================
# EFFECTIVE AREA
# ============================================================

def effective_area(
    area_original,
    cluster_original,
    error,
):

    area = area_original.copy()
    cluster = cluster_original.copy()

    error = safe_float(error)

    area["Error %"] = error

    area["Net Efficiency (%)"] = (
        area["Standard PV Efficiency (%)"]
        - error
    )

    area["Eff Area"] = (
        area["Net Efficiency (%)"]
        * area["Total area (m2)"]
        / 100
    )

    sums = (
        area.groupby("Clusters")["Eff Area"]
        .sum()
    )

    cluster["Eff Area(m2)"] = (
        cluster["Clusters"]
        .map(sums)
        .fillna(0)
    )

    return area, cluster


# ============================================================
# FIXED POWER
# ============================================================

def fixed_power(df, cluster):

    result = df.copy()

    if len(cluster) < 5:
        raise ValueError(
            "At least five cluster rows are required."
        )

    for i, poa in enumerate(POA_COLS):

        if poa not in result.columns:
            raise ValueError(
                f"Missing POA column '{poa}'."
            )

        area = safe_float(
            cluster.iloc[i]["Eff Area(m2)"]
        )

        result[POWER_COLS[i]] = (
            num_series(result[poa])
            * area
            / 1_000_000
        )

    result[TOTAL_POWER] = (
        result[POWER_COLS]
        .sum(axis=1)
    )

    return result


# ============================================================
# AUTOMATIC ERROR OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_error(
    area,
    cluster,
    fixed,
):

    actual = num_array(
        fixed["Actual"]
    )

    if len(actual) == 0 or actual.max() <= 0:
        raise ValueError(
            "Actual data contains no positive values."
        )

    actual_peak = actual.max()

    rows = []

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        _, c = effective_area(
            area,
            cluster,
            error,
        )

        result = fixed_power(
            fixed,
            c,
        )

        forecast = num_array(
            result[TOTAL_POWER]
        )

        peak = forecast.max()

        error_abs = abs(
            peak - actual_peak
        )

        rows.append(
            {
                "Error %": round(error, 1),
                "Calculated Peak": peak,
                "Actual Peak": actual_peak,
                "Peak Error": error_abs,
                "Peak Error %":
                    error_abs
                    / actual_peak
                    * 100,
            }
        )

    table = pd.DataFrame(rows)

    best = table.loc[
        table["Peak Error"].idxmin()
    ]

    return (
        float(best["Error %"]),
        table,
    )


# ============================================================
# TRACKING DATA
# ============================================================

def tracking_arrays(
    backend,
    ghi,
    fixed,
    cluster,
):

    backend_df = ensure_df(
        backend["C11"],
        "Backend Cal C11",
    )

    if "Block No." not in backend_df.columns:
        raise ValueError(
            "Block No. not found in Backend Cal C11."
        )

    blocks = num_array(
        backend_df["Block No."]
    )

    n = min(
        len(blocks),
        len(ghi),
        len(fixed),
    )

    if n <= 0:
        raise ValueError(
            "No Tracking data available."
        )

    matrix = np.column_stack(
        [
            num_array(ghi[c])[:n]
            for c in GHI_COLS
        ]
    )

    actual = num_array(
        fixed["Actual"]
    )[:n]

    weights = (
        pd.to_numeric(
            cluster["Eff Area(m2)"].iloc[:5],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    if len(weights) < 5:
        raise ValueError(
            "Five tracking cluster weights are required."
        )

    return (
        blocks[:n],
        matrix,
        actual,
        weights,
    )


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

    if not start < max_block < end:
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

    dhi_part = (
        ghi
        * dhi
        / 100
    )

    dni = (
        ghi - dhi_part
    ) / cos_panel[:, None]

    power = (
        dni
        * weights[None, :]
        / 1_000_000
    )

    forecast = power.sum(axis=1)

    return forecast, power, zenith, panel, dni


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

    peak = actual_day.max()
    energy = actual_day.sum()

    def objective(x):

        p = np.rint(x).astype(int)

        result = calculate_tracking(
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5],
            blocks,
            ghi,
            weights,
        )

        if result is None:
            return 1e9

        prediction = result[0]

        if not np.all(
            np.isfinite(prediction)
        ):
            return 1e9

        pred_day = prediction[mask]

        block_error = (
            np.mean(
                np.abs(
                    actual_day - pred_day
                )
            )
            / peak
        )

        peak_error = (
            abs(
                peak - pred_day.max()
            )
            / peak
        )

        energy_error = (
            abs(
                energy - pred_day.sum()
            )
            / energy
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

    x = np.rint(
        result.x
    ).astype(int)

    return {
        "DHI": int(x[0]),
        "GHI Starting Block": int(x[1]),
        "GHI Ending Block": int(x[2]),
        "GHI Max Block": int(x[3]),
        "Tracking East Limit": int(x[4]),
        "Tracking West Limit": int(x[5]),
    }


# ============================================================
# GRAPH
# ============================================================

def make_graph(
    actual,
    forecast,
    title,
):

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
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=forecast[:n],
            mode="lines",
            name="Forecast",
        )
    )

    fig.update_layout(
        title=title,
        height=430,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(
            l=30,
            r=30,
            t=55,
            b=30,
        ),
        xaxis_title="Block",
        yaxis_title="Power",
    )

    return fig


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section">📂 Input Data</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Solar Excel File",
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

current_hash = hash_file(uploaded)

if st.session_state.file_hash != current_hash:

    st.session_state.file_hash = current_hash
    st.session_state.input_df = None
    st.session_state.plant_type = "Fixed"

    reset_results()

    for key in [
        "solar_input_editor",
        "plant_selector",
    ]:

        if key in st.session_state:
            del st.session_state[key]


# ============================================================
# LOAD
# ============================================================

try:

    workbook = load_workbook(
        uploaded.getvalue()
    )

except Exception as e:

    st.error(
        f"Workbook loading failed: {e}"
    )

    st.stop()


# ============================================================
# PREPARE
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

    latitude = prepare_latitude(
        workbook["forecast_config"]
    )

    tilt_lookup = prepare_tilt(
        workbook["tilt"]
    )

    fixed_raw = prepare_fixed(
        workbook["fixed"]
    )

except Exception as e:

    st.error(
        f"Input preparation failed: {e}"
    )

    st.stop()


# ============================================================
# INPUT TABLE
# ============================================================

if st.session_state.input_df is None:

    try:

        st.session_state.input_df = build_input(
            ghi_raw,
            fixed_raw,
        )

    except Exception as e:

        st.error(
            f"Input data creation failed: {e}"
        )

        st.stop()


st.markdown(
    '<div class="section">✏️ GHI / Actual Input</div>',
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
        c: st.column_config.NumberColumn(
            c,
            format="%.3f",
        )
        for c in GHI_COLS + ["Actual"]
    },
)


# ============================================================
# PLANT
# ============================================================

st.markdown(
    '<div class="section">🌱 Plant Type</div>',
    unsafe_allow_html=True,
)

plant = st.segmented_control(
    "Plant Type",
    options=PLANTS,
    default=st.session_state.plant_type,
    selection_mode="single",
    width="stretch",
    label_visibility="collapsed",
    key="plant_selector",
)

if plant is None:
    plant = "Fixed"

if plant != st.session_state.plant_type:

    st.session_state.plant_type = plant

    if (
        st.session_state.calculated_plant_type
        != plant
    ):
        reset_results()

else:

    st.session_state.plant_type = plant


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

            st.session_state.input_df = (
                input_df.copy()
            )

            ghi, fixed_raw_user = apply_input(
                input_df,
                ghi_raw,
                fixed_raw,
            )

            fixed_geometry = prepare_geometry(
                fixed_raw_user,
                ghi,
                latitude,
                tilt_lookup,
            )

            # ----------------------------------------------
            # AUTOMATIC ERROR %
            # ----------------------------------------------

            best_error, error_table = (
                optimize_error(
                    area,
                    cluster,
                    fixed_geometry,
                )
            )

            # ----------------------------------------------
            # APPLY ERROR
            # ----------------------------------------------

            final_area, final_cluster = (
                effective_area(
                    area,
                    cluster,
                    best_error,
                )
            )

            # ----------------------------------------------
            # FIXED
            # ----------------------------------------------

            fixed_result = fixed_power(
                fixed_geometry,
                final_cluster,
            )

            # ----------------------------------------------
            # TRACKING
            # ----------------------------------------------

            tracking_params = None
            tracking_forecast = None

            if plant == "Tracking":

                blocks, ghi_matrix, actual, weights = (
                    tracking_arrays(
                        workbook["backend"],
                        ghi,
                        fixed_geometry,
                        final_cluster,
                    )
                )

                tracking_params = optimize_tracking(
                    tuple(blocks.tolist()),
                    tuple(
                        tuple(row)
                        for row in ghi_matrix
                    ),
                    tuple(actual.tolist()),
                    tuple(weights.tolist()),
                )

                tracking_result = calculate_tracking(
                    tracking_params["DHI"],
                    tracking_params[
                        "GHI Starting Block"
                    ],
                    tracking_params[
                        "GHI Ending Block"
                    ],
                    tracking_params[
                        "GHI Max Block"
                    ],
                    tracking_params[
                        "Tracking East Limit"
                    ],
                    tracking_params[
                        "Tracking West Limit"
                    ],
                    blocks,
                    ghi_matrix,
                    weights,
                )

                if tracking_result is None:
                    raise ValueError(
                        "Automatic Tracking optimization "
                        "returned invalid parameters."
                    )

                tracking_forecast = (
                    tracking_result[0]
                )

            # ----------------------------------------------
            # SAVE
            # ----------------------------------------------

            st.session_state.calculation_data = {
                "area": area,
                "cluster": cluster,
                "final_area": final_area,
                "final_cluster": final_cluster,
                "ghi": ghi,
                "fixed": fixed_geometry,
                "fixed_result": fixed_result,
                "best_error": best_error,
                "error_table": error_table,
                "tracking_params": tracking_params,
                "tracking_forecast": tracking_forecast,
            }

            st.session_state.calculated = True
            st.session_state.calculated_plant_type = plant

        st.success(
            "Automatic calculation completed successfully."
        )

    except Exception as e:

        reset_results()

        st.error(
            f"Calculation failed: {e}"
        )


# ============================================================
# WAIT
# ============================================================

if not st.session_state.calculated:

    st.caption(
        "Edit GHI / Actual values, select Fixed or Tracking, "
        "then run the automatic calculation."
    )

    st.stop()


# ============================================================
# PLANT VALIDATION
# ============================================================

if (
    st.session_state.calculated_plant_type
    != plant
):

    reset_results()

    st.warning(
        "Plant type changed. Run Automatic Calculation again."
    )

    st.stop()


data = st.session_state.calculation_data

if data is None:

    reset_results()

    st.warning(
        "Calculation results are unavailable."
    )

    st.stop()


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)

error = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=20.0,
    value=float(
        data["best_error"]
    ),
    step=0.1,
    format="%.1f",
    key=f"error_{current_hash}",
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

if plant == "Tracking":

    p = data["tracking_params"]

    if p is None:
        st.error(
            "Tracking parameters are unavailable."
        )
        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi = st.number_input(
            "DHI (%)",
            0,
            100,
            int(p["DHI"]),
            1,
            key=f"dhi_{current_hash}",
        )

        start = st.number_input(
            "GHI Starting Block",
            0,
            95,
            int(p["GHI Starting Block"]),
            1,
            key=f"start_{current_hash}",
        )

    with c2:

        end = st.number_input(
            "GHI Ending Block",
            1,
            96,
            int(p["GHI Ending Block"]),
            1,
            key=f"end_{current_hash}",
        )

        maximum = st.number_input(
            "GHI Max Block",
            0,
            95,
            int(p["GHI Max Block"]),
            1,
            key=f"max_{current_hash}",
        )

    with c3:

        east = st.number_input(
            "Tracking East Limit",
            0,
            90,
            int(p["Tracking East Limit"]),
            1,
            key=f"east_{current_hash}",
        )

        west = st.number_input(
            "Tracking West Limit",
            0,
            90,
            int(p["Tracking West Limit"]),
            1,
            key=f"west_{current_hash}",
        )


# ============================================================
# FINAL CHEAP CALCULATION
# IMPORTANT:
# NO DIFFERENTIAL EVOLUTION HERE
# ============================================================

try:

    final_area, final_cluster = (
        effective_area(
            data["area"],
            data["cluster"],
            error,
        )
    )

    fixed_result = fixed_power(
        data["fixed"],
        final_cluster,
    )

    if plant == "Tracking":

        if not start < maximum < end:

            st.error(
                "Tracking parameters must satisfy: "
                "GHI Starting Block < "
                "GHI Max Block < "
                "GHI Ending Block."
            )

            st.stop()

        blocks, ghi_matrix, actual, weights = (
            tracking_arrays(
                workbook["backend"],
                data["ghi"],
                data["fixed"],
                final_cluster,
            )
        )

        result = calculate_tracking(
            int(dhi),
            int(start),
            int(end),
            int(maximum),
            int(east),
            int(west),
            blocks,
            ghi_matrix,
            weights,
        )

        if result is None:
            raise ValueError(
                "Invalid Tracking parameters."
            )

        forecast = result[0]
        actual_values = actual

        title = (
            "Tracking Plant | Actual vs Forecast"
        )

    else:

        actual_values = num_array(
            data["fixed"]["Actual"]
        )

        forecast = num_array(
            fixed_result[TOTAL_POWER]
        )

        title = (
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

actual_peak = (
    float(np.max(actual_values))
    if len(actual_values)
    else 0
)

forecast_peak = (
    float(np.max(forecast))
    if len(forecast)
    else 0
)

st.markdown(
    '<div class="section">📊 Results</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:

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

with c2:

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
    '<div class="section">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)

fig = make_graph(
    actual_values,
    forecast,
    title,
)

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "displayModeBar": False,
        "responsive": True,
    },
)
