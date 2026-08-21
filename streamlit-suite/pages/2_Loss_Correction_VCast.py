# ============================================================
# SOLAR FORECAST CORRECTION
# ROBUST + COMPACT + CLOUD SAFE
# FIXED / TRACKING
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
)


# ============================================================
# CONSTANTS
# ============================================================

PLANTS = ["Fixed", "Tracking"]

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

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
    (0, 10),      # DHI
    (10, 30),     # Start
    (65, 80),     # End
    (47, 53),     # Max
    (10, 70),     # East
    (10, 70),     # West
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
    }

    .subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 15px;
    }

    .section {
        font-size: 19px;
        font-weight: 700;
        margin: 18px 0 9px;
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "plant" not in st.session_state:
    st.session_state.plant = "Fixed"

if "input_df" not in st.session_state:
    st.session_state.input_df = None


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_name(x):
    return (
        str(x)
        .replace("\n", " ")
        .replace("*", "")
        .strip()
        .lower()
    )


def clean_columns(df):
    df = df.copy()
    df.columns = [
        str(c).replace("\n", " ").replace("*", "").strip()
        for c in df.columns
    ]
    return df


def numeric(x):
    """
    Safely convert Series / array / scalar to numeric.
    Never calls .fillna() on a scalar.
    """
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").fillna(0)

    if isinstance(x, pd.DataFrame):
        return x.apply(pd.to_numeric, errors="coerce").fillna(0)

    if np.isscalar(x):
        try:
            v = float(x)
            return 0.0 if not np.isfinite(v) else v
        except Exception:
            return 0.0

    arr = pd.to_numeric(
        pd.Series(x),
        errors="coerce",
    ).fillna(0)

    return arr


def arr(x):
    if isinstance(x, pd.Series):
        return pd.to_numeric(
            x,
            errors="coerce",
        ).fillna(0).to_numpy(dtype=float)

    if np.isscalar(x):
        return np.array([float(x)])

    return pd.to_numeric(
        pd.Series(x),
        errors="coerce",
    ).fillna(0).to_numpy(dtype=float)


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def df_copy(x, name="data"):
    if not isinstance(x, pd.DataFrame):
        raise TypeError(
            f"{name} is not a DataFrame. "
            f"Received {type(x).__name__}."
        )
    return x.copy()


def file_hash(data):
    return hashlib.sha256(data).hexdigest()


# ============================================================
# ROBUST EXCEL READER
# ============================================================

@st.cache_data(show_spinner=False, max_entries=3)
def read_all_sheets(data):

    excel = pd.ExcelFile(
        io.BytesIO(data)
    )

    sheets = {}

    for sheet in excel.sheet_names:

        # Read raw sheet first.
        raw = pd.read_excel(
            io.BytesIO(data),
            sheet_name=sheet,
            header=None,
        )

        sheets[sheet] = raw

    return sheets


def detect_header(raw, keywords):
    """
    Find row containing the largest number of
    requested keywords.
    """

    if not isinstance(raw, pd.DataFrame):
        return 0

    best_row = 0
    best_score = -1

    keys = [
        clean_name(k)
        for k in keywords
    ]

    scan = min(
        len(raw),
        30,
    )

    for r in range(scan):

        values = [
            clean_name(v)
            for v in raw.iloc[r].tolist()
        ]

        score = 0

        for key in keys:
            if any(
                key == v or key in v
                for v in values
            ):
                score += 1

        if score > best_score:
            best_score = score
            best_row = r

    return best_row


def sheet_df(
    sheets,
    sheet_name,
    keywords=None,
):

    if sheet_name not in sheets:
        raise ValueError(
            f"Sheet '{sheet_name}' not found."
        )

    raw = sheets[sheet_name]

    if keywords:
        header = detect_header(
            raw,
            keywords,
        )
    else:
        header = 0

    df = raw.iloc[
        header + 1:
    ].copy()

    df.columns = raw.iloc[
        header
    ].tolist()

    return clean_columns(
        df.reset_index(drop=True)
    )


# ============================================================
# FIND COLUMN
# ============================================================

def find_col(df, names, required=True):

    df = df_copy(df)

    lookup = {
        clean_name(c): c
        for c in df.columns
    }

    # Exact
    for name in names:

        key = clean_name(name)

        if key in lookup:
            return lookup[key]

    # Partial
    for name in names:

        key = clean_name(name)

        for normalized, original in lookup.items():

            if key in normalized:
                return original

    if required:
        raise ValueError(
            "Required column not found.\n\n"
            f"Looking for: {names}\n\n"
            f"Available columns: "
            f"{list(df.columns)}"
        )

    return None


# ============================================================
# FIND SHEET
# ============================================================

def find_sheet(sheets, names):

    lookup = {
        clean_name(k): k
        for k in sheets
    }

    for name in names:

        key = clean_name(name)

        if key in lookup:
            return lookup[key]

    for name in names:

        key = clean_name(name)

        for normalized, original in lookup.items():

            if key in normalized:
                return original

    return None


# ============================================================
# AREA
# ============================================================

def prepare_area(sheets):

    sheet = find_sheet(
        sheets,
        ["Area & Efficiency"],
    )

    if sheet is None:
        raise ValueError(
            "Area & Efficiency sheet not found."
        )

    df = sheet_df(
        sheets,
        sheet,
        [
            "Clusters",
            "Standard PV Efficiency",
            "No of Module",
            "Area of 1 Module",
        ],
    )

    cluster_col = find_col(
        df,
        ["Clusters", "Cluster"],
    )

    efficiency_col = find_col(
        df,
        [
            "Standard PV Efficiency (%)",
            "Standard PV Efficiency",
            "PV Efficiency",
            "Efficiency",
        ],
    )

    module_col = find_col(
        df,
        [
            "No of Module",
            "No. of Module",
            "Number of Module",
            "Modules",
        ],
    )

    area_col = find_col(
        df,
        [
            "Area of 1 Module (m2)",
            "Area of 1 Module",
            "Module Area",
        ],
    )

    df = df[
        [
            cluster_col,
            efficiency_col,
            module_col,
            area_col,
        ]
    ].copy()

    df.columns = [
        "Clusters",
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]

    for c in df.columns[1:]:

        df[c] = numeric(
            df[c]
        )

    df["Clusters"] = (
        df["Clusters"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["Clusters"].isin(
            CLUSTERS
        )
    ].copy()

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    if df.empty:
        raise ValueError(
            "No C11-C15 rows found in Area & Efficiency."
        )

    return df.reset_index(drop=True)


# ============================================================
# CLUSTER AREA
# ============================================================

def prepare_cluster(sheets):

    sheet = find_sheet(
        sheets,
        ["Area & Efficiency"],
    )

    df = sheet_df(
        sheets,
        sheet,
        ["Clusters"],
    )

    cluster_col = find_col(
        df,
        ["Clusters", "Cluster"],
    )

    out = pd.DataFrame()

    out["Clusters"] = (
        df[cluster_col]
        .astype(str)
        .str.strip()
    )

    return out[
        out["Clusters"].isin(CLUSTERS)
    ].drop_duplicates(
        "Clusters"
    ).reset_index(drop=True)


# ============================================================
# GHI
# ============================================================

def prepare_ghi(sheets):

    sheet = find_sheet(
        sheets,
        ["Result"],
    )

    if sheet is None:
        raise ValueError(
            "Result sheet not found."
        )

    df = sheet_df(
        sheets,
        sheet,
        GHI_COLS,
    )

    result = pd.DataFrame()

    for col in GHI_COLS:

        actual_col = find_col(
            df,
            [col],
        )

        result[col] = numeric(
            df[actual_col]
        )

    return result.reset_index(
        drop=True
    )


# ============================================================
# ACTUAL
# ============================================================

def prepare_actual(sheets):

    sheet = find_sheet(
        sheets,
        ["Fixed-C11"],
    )

    if sheet is None:
        raise ValueError(
            "Fixed-C11 sheet not found."
        )

    df = sheet_df(
        sheets,
        sheet,
        [
            "Actual",
            "Actual Power",
            "Power",
        ],
    )

    actual_col = find_col(
        df,
        [
            "Actual",
            "Actual Power",
            "Actual Power MW",
            "Actual MW",
        ],
    )

    actual = numeric(
        df[actual_col]
    )

    return pd.DataFrame(
        {
            "Actual": actual
        }
    )


# ============================================================
# LATITUDE
# ============================================================

def prepare_latitude(sheets):

    sheet = find_sheet(
        sheets,
        ["Forecast Config"],
    )

    if sheet is None:
        raise ValueError(
            "Forecast Config sheet not found."
        )

    df = sheet_df(
        sheets,
        sheet,
        ["Lat", "Latitude"],
    )

    col = find_col(
        df,
        ["Lat", "Latitude"],
    )

    values = numeric(
        df[col]
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise ValueError(
            "Latitude value not found."
        )

    return float(
        values[0]
    )


# ============================================================
# TILT
# ============================================================

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def prepare_tilt(sheets):

    sheet = find_sheet(
        sheets,
        ["Config Tilt Angle"],
    )

    if sheet is None:
        raise ValueError(
            "Config Tilt Angle sheet not found."
        )

    df = sheet_df(
        sheets,
        sheet,
        [
            "Month",
            "Fixed",
        ],
    )

    # Find Month intelligently.
    month_col = find_col(
        df,
        ["Month"],
        required=False,
    )

    if month_col is None:

        for col in df.columns:

            vals = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            count = vals.isin(
                [m.lower() for m in MONTHS]
            ).sum()

            if count >= 2:
                month_col = col
                break

    if month_col is None:
        raise ValueError(
            "Month column could not be detected "
            "in Config Tilt Angle."
        )

    fixed_col = find_col(
        df,
        ["Fixed", "Fixed Tilt"],
    )

    result = {}

    for _, row in df.iterrows():

        month = str(
            row[month_col]
        ).strip()

        for m in MONTHS:

            if month.lower() == m.lower():

                result[m] = safe_float(
                    row[fixed_col]
                )

    if not result:

        raise ValueError(
            "No monthly Fixed tilt values found."
        )

    return result


# ============================================================
# BUILD INPUT
# ============================================================

def build_input(ghi, actual):

    n = min(
        len(ghi),
        len(actual),
    )

    if n <= 0:
        raise ValueError(
            "No GHI / Actual records found."
        )

    out = pd.DataFrame()

    for c in GHI_COLS:
        out[c] = numeric(
            ghi[c]
        ).iloc[:n].to_numpy()

    out["Actual"] = numeric(
        actual["Actual"]
    ).iloc[:n].to_numpy()

    return out


# ============================================================
# GEOMETRY
# ============================================================

def geometry(
    ghi,
    actual,
    latitude,
    tilt_lookup,
):

    n = min(
        len(ghi),
        len(actual),
    )

    if n <= 0:
        raise ValueError(
            "No data available."
        )

    ghi = ghi.iloc[:n].copy()

    fixed = pd.DataFrame()

    fixed["Actual"] = numeric(
        actual["Actual"]
    ).iloc[:n].to_numpy()

    today = pd.Timestamp.today()

    day = today.dayofyear

    declination = (
        23.45
        * np.sin(
            np.radians(
                360 * (284 + day) / 365
            )
        )
    )

    elevation = (
        90
        - latitude
        + declination
    )

    tilt = safe_float(
        tilt_lookup.get(
            today.strftime("%B"),
            0,
        )
    )

    fixed["Declination Angle ∆"] = declination
    fixed["Elevation angle a"] = elevation
    fixed["Tilt Angle b"] = tilt
    fixed["a+b"] = elevation + tilt

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(elevation + tilt)
    )

    # Avoid division by zero.
    safe_sin_a = (
        sin_a
        if abs(sin_a) > 1e-8
        else 1e-8
    )

    fixed["Sin(a)"] = sin_a
    fixed["SIN(a+b)"] = sin_ab

    for i, cluster in enumerate(CLUSTERS):

        g = numeric(
            ghi[GHI_COLS[i]]
        ).iloc[:n].to_numpy()

        suffix = (
            ""
            if i == 0
            else f"-CL{i + 1}"
        )

        fixed[
            f"GHI*sin(a){suffix}"
        ] = g * sin_a

        fixed[
            f"GHI*sin(a+b){suffix}"
        ] = g * sin_ab

        fixed[
            POA_COLS[i]
        ] = (
            g * sin_ab
            / safe_sin_a
        )

    return fixed


# ============================================================
# EFFICIENCY
# ============================================================

def apply_efficiency(
    area,
    cluster,
    error,
):

    a = area.copy()
    c = cluster.copy()

    error = safe_float(error)

    a["Error %"] = error

    a["Net Efficiency (%)"] = (
        a["Standard PV Efficiency (%)"]
        - error
    ).clip(lower=0)

    a["Eff Area"] = (
        a["Net Efficiency (%)"]
        * a["Total area (m2)"]
        / 100
    )

    mapping = (
        a.groupby("Clusters")["Eff Area"]
        .sum()
    )

    c["Eff Area(m2)"] = (
        c["Clusters"]
        .map(mapping)
        .fillna(0)
    )

    # Guarantee C11-C15 order.
    c = (
        pd.DataFrame(
            {"Clusters": CLUSTERS}
        )
        .merge(
            c,
            on="Clusters",
            how="left",
        )
    )

    c["Eff Area(m2)"] = numeric(
        c["Eff Area(m2)"]
    )

    return a, c


# ============================================================
# FIXED POWER
# ============================================================

def fixed_calculation(
    geometry_df,
    cluster,
):

    result = geometry_df.copy()

    for i, poa in enumerate(POA_COLS):

        area = safe_float(
            cluster.iloc[i]["Eff Area(m2)"]
        )

        result[POWER_COLS[i]] = (
            numeric(result[poa])
            * area
            / 1_000_000
        )

    result[TOTAL_POWER] = (
        result[POWER_COLS]
        .sum(axis=1)
    )

    return result


# ============================================================
# AUTOMATIC ERROR %
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_error(
    area,
    cluster,
    geometry_df,
):

    actual = arr(
        geometry_df["Actual"]
    )

    if len(actual) == 0:
        raise ValueError(
            "Actual data is empty."
        )

    peak = float(
        np.max(actual)
    )

    if peak <= 0:
        raise ValueError(
            "Actual peak must be greater than zero."
        )

    best_error = 0
    best_score = np.inf
    rows = []

    # Keep this cheap.
    for error in np.arange(
        0,
        20.01,
        0.1,
    ):

        _, c = apply_efficiency(
            area,
            cluster,
            error,
        )

        result = fixed_calculation(
            geometry_df,
            c,
        )

        forecast = arr(
            result[TOTAL_POWER]
        )

        n = min(
            len(actual),
            len(forecast),
        )

        a = actual[:n]
        f = forecast[:n]

        score = abs(
            np.max(f)
            - peak
        )

        peak_error_pct = (
            score / peak * 100
        )

        rows.append(
            {
                "Error %": round(
                    error,
                    1,
                ),
                "Forecast Peak": np.max(f),
                "Actual Peak": peak,
                "Peak Error": score,
                "Peak Error %": peak_error_pct,
            }
        )

        if score < best_score:

            best_score = score
            best_error = error

    return (
        round(
            best_error,
            1,
        ),
        pd.DataFrame(rows),
    )


# ============================================================
# TRACKING BACKEND
# ============================================================

def prepare_tracking(
    sheets,
    ghi,
    actual,
    cluster,
):

    backend_sheet = find_sheet(
        sheets,
        ["Backend Cal C11"],
    )

    if backend_sheet is None:
        raise ValueError(
            "Backend Cal C11 sheet not found."
        )

    backend = sheet_df(
        sheets,
        backend_sheet,
        [
            "Block No.",
            "Block",
        ],
    )

    block_col = find_col(
        backend,
        [
            "Block No.",
            "Block No",
            "Block",
        ],
    )

    blocks = arr(
        backend[block_col]
    )

    n = min(
        len(blocks),
        len(ghi),
        len(actual),
    )

    matrix = np.column_stack(
        [
            arr(
                ghi[c]
            )[:n]
            for c in GHI_COLS
        ]
    )

    actual_array = arr(
        actual["Actual"]
    )[:n]

    weights = numeric(
        cluster["Eff Area(m2)"]
        .iloc[:5]
    ).to_numpy(
        dtype=float
    )

    return (
        blocks[:n],
        matrix,
        actual_array,
        weights,
    )


# ============================================================
# TRACKING MODEL
# ============================================================

def tracking_model(
    dhi,
    start,
    end,
    maximum,
    east,
    west,
    blocks,
    ghi,
    weights,
):

    if not (
        start < maximum < end
    ):
        return None

    d1 = start - maximum
    d2 = end - maximum

    if d1 == 0 or d2 == 0:
        return None

    m1 = 90 / d1
    m2 = 90 / d2

    zenith = np.where(
        blocks <= maximum,
        np.minimum(
            89,
            abs(
                m1
                * (
                    blocks
                    - maximum
                )
            ),
        ),
        np.minimum(
            89,
            abs(
                m2
                * (
                    blocks
                    - maximum
                )
            ),
        ),
    )

    panel = np.where(
        blocks < maximum,
        np.minimum(
            zenith,
            east,
        ),
        np.minimum(
            zenith,
            west,
        ),
    )

    cos_panel = np.clip(
        np.cos(
            np.radians(panel)
        ),
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

    dni = np.maximum(
        dni,
        0,
    )

    power = (
        dni
        * weights[None, :]
        / 1_000_000
    )

    forecast = power.sum(
        axis=1
    )

    return (
        forecast,
        power,
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

    mask = actual > 0

    if not mask.any():
        raise ValueError(
            "Actual data has no positive values."
        )

    a = actual[mask]

    peak = a.max()
    energy = a.sum()

    def objective(x):

        x = np.rint(x).astype(int)

        output = tracking_model(
            *x,
            blocks,
            ghi,
            weights,
        )

        if output is None:
            return 1e9

        forecast = output[0]

        if not np.all(
            np.isfinite(forecast)
        ):
            return 1e9

        f = forecast[mask]

        block = (
            np.mean(
                np.abs(
                    a - f
                )
            )
            / peak
        )

        peak_error = (
            abs(
                peak - f.max()
            )
            / peak
        )

        energy_error = (
            abs(
                energy - f.sum()
            )
            / energy
        )

        return (
            0.80 * block
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(
        objective,
        TRACKING_BOUNDS,
        maxiter=30,
        popsize=10,
        seed=42,
        polish=True,
        workers=1,
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
# COMPLETE CALCULATION PIPELINE
# ============================================================

def calculate_all(
    sheets,
    input_df,
    plant,
):

    # -----------------------------
    # Read input
    # -----------------------------

    area = prepare_area(
        sheets
    )

    cluster = prepare_cluster(
        sheets
    )

    ghi = pd.DataFrame()

    for c in GHI_COLS:
        ghi[c] = numeric(
            input_df[c]
        )

    actual = pd.DataFrame(
        {
            "Actual": numeric(
                input_df["Actual"]
            )
        }
    )

    latitude = prepare_latitude(
        sheets
    )

    tilt = prepare_tilt(
        sheets
    )

    # -----------------------------
    # Geometry
    # -----------------------------

    geo = geometry(
        ghi,
        actual,
        latitude,
        tilt,
    )

    # -----------------------------
    # Error optimization
    # -----------------------------

    best_error, error_table = (
        optimize_error(
            area,
            cluster,
            geo,
        )
    )

    final_area, final_cluster = (
        apply_efficiency(
            area,
            cluster,
            best_error,
        )
    )

    # -----------------------------
    # Fixed
    # -----------------------------

    fixed_result = fixed_calculation(
        geo,
        final_cluster,
    )

    # -----------------------------
    # Tracking
    # -----------------------------

    tracking_params = None
    tracking_forecast = None

    if plant == "Tracking":

        blocks, ghi_matrix, actual_array, weights = (
            prepare_tracking(
                sheets,
                ghi,
                actual,
                final_cluster,
            )
        )

        tracking_params = (
            optimize_tracking(
                tuple(
                    blocks.tolist()
                ),
                tuple(
                    tuple(x)
                    for x in ghi_matrix
                ),
                tuple(
                    actual_array.tolist()
                ),
                tuple(
                    weights.tolist()
                ),
            )
        )

        result = tracking_model(
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

        if result is None:
            raise ValueError(
                "Invalid optimized Tracking parameters."
            )

        tracking_forecast = result[0]

    return {
        "area": area,
        "cluster": cluster,
        "final_area": final_area,
        "final_cluster": final_cluster,
        "geometry": geo,
        "fixed_result": fixed_result,
        "best_error": best_error,
        "error_table": error_table,
        "tracking_params": tracking_params,
        "tracking_forecast": tracking_forecast,
    }


# ============================================================
# FINAL FORECAST AFTER EDITING PARAMETERS
# ============================================================

def calculate_final_forecast(
    data,
    error,
    plant,
    tracking_params=None,
):

    _, cluster = apply_efficiency(
        data["area"],
        data["cluster"],
        error,
    )

    fixed_result = fixed_calculation(
        data["geometry"],
        cluster,
    )

    if plant == "Fixed":

        return (
            arr(
                data["geometry"]["Actual"]
            ),
            arr(
                fixed_result[TOTAL_POWER]
            ),
            fixed_result,
        )

    p = tracking_params

    blocks, ghi_matrix, actual, weights = (
        prepare_tracking(
            data["_sheets"],
            data["_ghi"],
            pd.DataFrame(
                {
                    "Actual":
                    data["geometry"]["Actual"]
                }
            ),
            cluster,
        )
    )

    result = tracking_model(
        int(p["DHI"]),
        int(p["GHI Starting Block"]),
        int(p["GHI Ending Block"]),
        int(p["GHI Max Block"]),
        int(p["Tracking East Limit"]),
        int(p["Tracking West Limit"]),
        blocks,
        ghi_matrix,
        weights,
    )

    if result is None:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    return (
        actual,
        result[0],
        fixed_result,
    )


# ============================================================
# GRAPH
# ============================================================

def graph(actual, forecast, title):

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
        xaxis_title="Block",
        yaxis_title="Power",
        margin=dict(
            l=30,
            r=30,
            t=55,
            b=30,
        ),
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Automatic optimization with editable parameters'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section">📂 Input Data</div>',
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

raw_file = uploaded.getvalue()
current_hash = file_hash(
    raw_file
)

if (
    st.session_state.file_hash
    != current_hash
):

    st.session_state.file_hash = (
        current_hash
    )

    st.session_state.results = None
    st.session_state.input_df = None


# ============================================================
# LOAD
# ============================================================

try:

    sheets = read_all_sheets(
        raw_file
    )

except Exception as e:

    st.error(
        f"Workbook loading failed: {e}"
    )

    st.stop()


# ============================================================
# BUILD INITIAL INPUT
# ============================================================

if st.session_state.input_df is None:

    try:

        ghi = prepare_ghi(
            sheets
        )

        actual = prepare_actual(
            sheets
        )

        st.session_state.input_df = (
            build_input(
                ghi,
                actual,
            )
        )

    except Exception as e:

        st.error(
            f"Input preparation failed: {e}"
        )

        st.stop()


# ============================================================
# INPUT EDITOR
# ============================================================

st.markdown(
    '<div class="section">✏️ GHI / Actual Input</div>',
    unsafe_allow_html=True,
)

input_df = st.data_editor(
    st.session_state.input_df,
    width="stretch",
    height=300,
    num_rows="fixed",
    hide_index=True,
    key="input_editor",
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
    "Plant",
    PLANTS,
    default=st.session_state.plant,
    key="plant_control",
    width="stretch",
    label_visibility="collapsed",
)

if plant is None:
    plant = st.session_state.plant

if plant != st.session_state.plant:

    st.session_state.plant = plant
    st.session_state.results = None


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
            "Calculating..."
        ):

            clean_input = input_df.copy()

            for c in (
                GHI_COLS
                + ["Actual"]
            ):
                clean_input[c] = numeric(
                    clean_input[c]
                )

            result = calculate_all(
                sheets,
                clean_input,
                plant,
            )

            # Keep required data for final
            # parameter recalculation.
            result["_sheets"] = sheets
            result["_ghi"] = pd.DataFrame(
                {
                    c:
                    clean_input[c]
                    for c in GHI_COLS
                }
            )

            st.session_state.input_df = (
                clean_input
            )

            st.session_state.results = (
                result
            )

        st.success(
            "Automatic calculation completed."
        )

    except Exception as e:

        st.session_state.results = None

        st.error(
            f"Calculation failed: {e}"
        )


# ============================================================
# RESULTS NOT READY
# ============================================================

if st.session_state.results is None:

    st.caption(
        "Edit the input data, select Fixed or Tracking, "
        "then click Run Automatic Calculation."
    )

    st.stop()


# ============================================================
# DATA
# ============================================================

data = st.session_state.results


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
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

tracking_params = (
    data["tracking_params"]
    if plant == "Tracking"
    else None
)

if plant == "Tracking":

    if tracking_params is None:

        st.error(
            "Tracking parameters unavailable."
        )

        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi = st.number_input(
            "DHI (%)",
            0,
            100,
            int(
                tracking_params["DHI"]
            ),
            1,
        )

        start = st.number_input(
            "GHI Starting Block",
            0,
            95,
            int(
                tracking_params[
                    "GHI Starting Block"
                ]
            ),
            1,
        )

    with c2:

        end = st.number_input(
            "GHI Ending Block",
            1,
            96,
            int(
                tracking_params[
                    "GHI Ending Block"
                ]
            ),
            1,
        )

        maximum = st.number_input(
            "GHI Max Block",
            0,
            95,
            int(
                tracking_params[
                    "GHI Max Block"
                ]
            ),
            1,
        )

    with c3:

        east = st.number_input(
            "Tracking East Limit",
            0,
            90,
            int(
                tracking_params[
                    "Tracking East Limit"
                ]
            ),
            1,
        )

        west = st.number_input(
            "Tracking West Limit",
            0,
            90,
            int(
                tracking_params[
                    "Tracking West Limit"
                ]
            ),
            1,
        )

    tracking_params = {
        "DHI": dhi,
        "GHI Starting Block": start,
        "GHI Ending Block": end,
        "GHI Max Block": maximum,
        "Tracking East Limit": east,
        "Tracking West Limit": west,
    }


# ============================================================
# FINAL CALCULATION
# ============================================================

try:

    final_area, final_cluster = (
        apply_efficiency(
            data["area"],
            data["cluster"],
            error,
        )
    )

    fixed_result = fixed_calculation(
        data["geometry"],
        final_cluster,
    )

    if plant == "Fixed":

        actual = arr(
            data["geometry"]["Actual"]
        )

        forecast = arr(
            fixed_result[
                TOTAL_POWER
            ]
        )

        title = (
            "Fixed Plant | Actual vs Forecast"
        )

    else:

        if not (
            start
            < maximum
            < end
        ):

            st.error(
                "Tracking parameters must satisfy: "
                "Starting < Max < Ending."
            )

            st.stop()

        blocks, ghi_matrix, actual, weights = (
            prepare_tracking(
                sheets,
                data["_ghi"],
                pd.DataFrame(
                    {
                        "Actual":
                        data["geometry"]["Actual"]
                    }
                ),
                final_cluster,
            )
        )

        tracking_result = tracking_model(
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

        if tracking_result is None:

            raise ValueError(
                "Invalid Tracking parameters."
            )

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
# METRICS
# ============================================================

actual_peak = (
    float(np.max(actual))
    if len(actual)
    else 0
)

forecast_peak = (
    float(np.max(forecast))
    if len(forecast)
    else 0
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


st.markdown(
    '<div class="section">📊 Results</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Actual Peak
            </div>
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
            <div class="metric-label">
                Forecast Peak
            </div>
            <div class="metric-value">
                {forecast_peak:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Peak Error
            </div>
            <div class="metric-value">
                {peak_error_pct:.2f}%
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


# ============================================================
# AUTOMATIC ERROR TABLE
# ============================================================

with st.expander(
    "🔎 Error % Optimization Details"
):

    table = data[
        "error_table"
    ].copy()

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# EFFECTIVE AREA
# ============================================================

with st.expander(
    "📐 Effective Area / Efficiency"
):

    display_area = (
        final_area[
            [
                "Clusters",
                "Standard PV Efficiency (%)",
                "Error %",
                "Net Efficiency (%)",
                "Total area (m2)",
                "Eff Area",
            ]
        ]
        .copy()
    )

    st.dataframe(
        display_area,
        width="stretch",
        hide_index=True,
    )
