# ============================================================
# SOLAR FORECAST CORRECTION
# ROBUST / COMPACT / FIXED + TRACKING
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
    "CL1 Power",
    "CL2 Power",
    "CL3 Power",
    "CL4 Power",
    "CL5 Power",
]

TOTAL_POWER = "Total Power"

TRACKING_BOUNDS = [
    (0, 10),     # DHI
    (10, 30),    # GHI start
    (65, 80),    # GHI end
    (47, 53),    # GHI max
    (10, 70),    # East
    (10, 70),    # West
]


# ============================================================
# CSS
# ============================================================

st.markdown("""
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
    padding: 14px 16px;
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
""", unsafe_allow_html=True)


# ============================================================
# SESSION
# ============================================================

DEFAULTS = {
    "file_hash": None,
    "input_df": None,
    "results": None,
    "plant": "Fixed",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_results():
    st.session_state.results = None


# ============================================================
# BASIC HELPERS
# ============================================================

def norm(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("\n", " ")
        .replace("  ", " ")
    )


def numeric(x):
    return pd.to_numeric(x, errors="coerce").fillna(0)


def arr(x):
    return numeric(pd.Series(x)).to_numpy(float)


def safe_float(x, default=0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ============================================================
# EXCEL READER
# ============================================================

@st.cache_data(show_spinner=False, max_entries=3)
def read_all_sheets(data):
    return pd.read_excel(
        io.BytesIO(data),
        sheet_name=None,
        header=None,
    )


def sheet_name(sheets, wanted):
    wanted = norm(wanted)

    for name in sheets:
        if norm(name) == wanted:
            return name

    for name in sheets:
        if wanted in norm(name):
            return name

    return None


def get_sheet(sheets, wanted):
    name = sheet_name(sheets, wanted)

    if name is None:
        raise ValueError(
            f"Sheet not found: {wanted}"
        )

    return sheets[name].copy()


# ============================================================
# HEADER DETECTION
# ============================================================

def detect_header(raw, keywords, max_rows=30):
    """
    Finds the row that most closely resembles a real header.
    """

    best_row = None
    best_score = -1

    limit = min(max_rows, len(raw))

    for r in range(limit):

        values = [
            norm(v)
            for v in raw.iloc[r].tolist()
            if pd.notna(v)
        ]

        score = 0

        for keyword in keywords:
            k = norm(keyword)

            if any(
                k == v or k in v
                for v in values
            ):
                score += 1

        if score > best_score:
            best_score = score
            best_row = r

    if best_row is None or best_score <= 0:
        raise ValueError(
            f"Could not detect header row. "
            f"Looking for: {keywords}"
        )

    df = raw.iloc[best_row + 1:].copy()
    df.columns = raw.iloc[best_row].astype(str).str.strip()

    return df.reset_index(drop=True)


# ============================================================
# COLUMN FINDER
# ============================================================

def find_col(df, aliases, required=True):
    aliases = [norm(x) for x in aliases]

    columns = list(df.columns)

    # Exact
    for c in columns:
        if norm(c) in aliases:
            return c

    # Contains
    for alias in aliases:
        for c in columns:
            if alias in norm(c):
                return c

    if required:
        raise ValueError(
            "Required column not found.\n\n"
            f"Looking for: {aliases}\n\n"
            f"Available columns: {columns}"
        )

    return None


# ============================================================
# ACTUAL POWER
# ============================================================

def find_actual(df):

    aliases = [
        "actual",
        "actual power",
        "actual power mw",
        "actual mw",
        "actual generation",
        "actual generation mw",
        "db value",
        "db_value",
    ]

    col = find_col(
        df,
        aliases,
        required=False,
    )

    if col is not None:
        return col

    # Numeric fallback
    candidates = []

    for c in df.columns:

        name = norm(c)

        if any(
            x in name
            for x in [
                "date",
                "time",
                "timestamp",
                "step",
                "block",
                "unnamed",
            ]
        ):
            continue

        v = pd.to_numeric(
            df[c],
            errors="coerce",
        )

        valid = v.notna().sum()

        if valid:
            candidates.append(
                (c, valid)
            )

    if candidates:
        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )
        return candidates[0][0]

    raise ValueError(
        "Could not identify Actual Power column."
    )


# ============================================================
# RESULT / GHI
# ============================================================

def prepare_result(raw):

    keywords = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
    ]

    try:
        df = detect_header(
            raw,
            keywords,
        )
    except Exception:
        # Sometimes GHI columns are the first row
        df = raw.copy()
        df.columns = df.iloc[0].astype(str)
        df = df.iloc[1:].reset_index(drop=True)

    # Locate GHI columns flexibly
    output = pd.DataFrame()

    for cluster in CLUSTERS:

        aliases = [
            f"GHI {cluster}",
            f"GHI_{cluster}",
            f"{cluster} GHI",
            cluster,
        ]

        col = find_col(
            df,
            aliases,
            required=False,
        )

        if col is None:

            # Try columns containing both GHI and cluster
            for c in df.columns:

                n = norm(c)

                if (
                    "ghi" in n
                    and cluster.lower() in n
                ):
                    col = c
                    break

        if col is None:
            raise ValueError(
                f"GHI column for {cluster} not found.\n"
                f"Available columns: {list(df.columns)}"
            )

        output[f"GHI {cluster}"] = numeric(
            df[col]
        )

    return output.reset_index(drop=True)


# ============================================================
# FIXED ACTUAL
# ============================================================

def prepare_fixed(raw):

    try:
        df = detect_header(
            raw,
            [
                "Actual",
                "Actual Power",
                "DB Value",
            ],
        )
    except Exception:
        df = raw.copy()

    actual_col = find_actual(df)

    out = pd.DataFrame()
    out["Actual"] = numeric(
        df[actual_col]
    )

    # Date detection
    date_col = find_col(
        df,
        [
            "date",
            "datetime",
            "timestamp",
            "time",
        ],
        required=False,
    )

    if date_col is not None:
        out["DateTime"] = pd.to_datetime(
            df[date_col],
            errors="coerce",
        )

    return out.reset_index(drop=True)


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def prepare_area(raw):

    keywords = [
        "Clusters",
        "Standard PV Efficiency",
        "No of Module",
        "Area of 1 Module",
    ]

    df = detect_header(
        raw,
        keywords,
    )

    cluster_col = find_col(
        df,
        [
            "Clusters",
            "Cluster",
        ],
    )

    efficiency_col = find_col(
        df,
        [
            "Standard PV Efficiency (%)",
            "Standard PV Efficiency",
            "PV Efficiency",
            "Efficiency (%)",
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
            "No Module",
        ],
    )

    module_area_col = find_col(
        df,
        [
            "Area of 1 Module (m2)",
            "Area of 1 Module",
            "Module Area",
            "Area Module",
        ],
    )

    out = pd.DataFrame()

    out["Clusters"] = (
        df[cluster_col]
        .astype(str)
        .str.strip()
    )

    out["Standard PV Efficiency (%)"] = numeric(
        df[efficiency_col]
    )

    out["No of Module"] = numeric(
        df[module_col]
    )

    out["Area of 1 Module (m2)"] = numeric(
        df[module_area_col]
    )

    out["Total area (m2)"] = (
        out["No of Module"]
        * out["Area of 1 Module (m2)"]
    )

    out = out[
        out["Clusters"].str.upper().isin(CLUSTERS)
    ]

    if out.empty:
        raise ValueError(
            "No C11-C15 rows found in Area & Efficiency."
        )

    return out.reset_index(drop=True)


# ============================================================
# CLUSTER WEIGHTS
# ============================================================

def prepare_cluster_weights(area):

    result = (
        area.groupby("Clusters")[
            "Total area (m2)"
        ]
        .sum()
        .reindex(CLUSTERS)
        .fillna(0)
        .reset_index()
    )

    result.rename(
        columns={
            "Total area (m2)": "Eff Area(m2)"
        },
        inplace=True,
    )

    return result


# ============================================================
# LATITUDE
# ============================================================

def prepare_latitude(raw):

    df = detect_header(
        raw,
        ["Lat", "Latitude"],
    )

    col = find_col(
        df,
        ["Lat", "Latitude"],
    )

    values = numeric(df[col])

    values = values[values != 0]

    if values.empty:
        raise ValueError(
            "Latitude value could not be found."
        )

    return float(values.iloc[0])


# ============================================================
# TILT
# ============================================================

MONTHS = [
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


def prepare_tilt(raw):

    # First attempt: find Fixed column
    try:
        df = detect_header(
            raw,
            [
                "Month",
                "Fixed",
            ],
        )
    except Exception:
        df = raw.copy()

    fixed_col = find_col(
        df,
        [
            "Fixed",
            "Fixed Tilt",
            "Tilt",
        ],
        required=False,
    )

    # If Fixed was not found, search every row
    if fixed_col is None:

        for r in range(min(30, len(raw))):

            row = raw.iloc[r].astype(str)

            if any(
                "fixed" in norm(x)
                for x in row
            ):

                df = raw.iloc[r + 1:].copy()
                df.columns = (
                    raw.iloc[r]
                    .astype(str)
                    .str.strip()
                )

                fixed_col = find_col(
                    df,
                    [
                        "Fixed",
                        "Fixed Tilt",
                        "Tilt",
                    ],
                    required=False,
                )

                if fixed_col is not None:
                    break

    if fixed_col is None:
        raise ValueError(
            "Fixed tilt column not found in Config Tilt Angle."
        )

    month_col = find_col(
        df,
        ["Month", "Months"],
        required=False,
    )

    result = {}

    if month_col is not None:

        for _, row in df.iterrows():

            month = norm(
                row[month_col]
            )

            if month in MONTHS:
                result[month] = safe_float(
                    row[fixed_col]
                )

    # Fallback: search month names anywhere
    if not result:

        for _, row in df.iterrows():

            values = [
                norm(x)
                for x in row.tolist()
            ]

            for month in MONTHS:

                if month in values:

                    idx = values.index(month)

                    if idx + 1 < len(row):

                        result[month] = safe_float(
                            row.iloc[idx + 1]
                        )

    if not result:

        raise ValueError(
            "Could not identify monthly Fixed tilt values."
        )

    return result


# ============================================================
# BACKEND BLOCKS
# ============================================================

def prepare_backend(raw):

    try:
        df = detect_header(
            raw,
            [
                "Block No.",
                "Block",
            ],
        )
    except Exception:
        df = raw.copy()

    block_col = find_col(
        df,
        [
            "Block No.",
            "Block No",
            "Block",
            "Block Number",
        ],
    )

    return pd.DataFrame({
        "Block": numeric(
            df[block_col]
        )
    })


# ============================================================
# COMPLETE WORKBOOK PREPARATION
# ============================================================

@st.cache_data(show_spinner=False, max_entries=3)
def prepare_workbook(data):

    sheets = read_all_sheets(data)

    # -----------------------------
    # Required sheets
    # -----------------------------

    area_raw = get_sheet(
        sheets,
        "Area & Efficiency",
    )

    result_raw = get_sheet(
        sheets,
        "Result",
    )

    fixed_raw = get_sheet(
        sheets,
        "Fixed-C11",
    )

    forecast_raw = get_sheet(
        sheets,
        "Forecast Config",
    )

    tilt_raw = get_sheet(
        sheets,
        "Config Tilt Angle",
    )

    # -----------------------------
    # Prepare
    # -----------------------------

    area = prepare_area(
        area_raw
    )

    cluster = prepare_cluster_weights(
        area
    )

    ghi = prepare_result(
        result_raw
    )

    fixed = prepare_fixed(
        fixed_raw
    )

    latitude = prepare_latitude(
        forecast_raw
    )

    tilt = prepare_tilt(
        tilt_raw
    )

    # -----------------------------
    # Tracking backend
    # -----------------------------

    backend = {}

    for cluster_name in CLUSTERS:

        target = f"Backend Cal {cluster_name}"

        name = sheet_name(
            sheets,
            target,
        )

        if name is not None:

            try:
                backend[cluster_name] = (
                    prepare_backend(
                        sheets[name]
                    )
                )
            except Exception:
                pass

    return {
        "area": area,
        "cluster": cluster,
        "ghi": ghi,
        "fixed": fixed,
        "latitude": latitude,
        "tilt": tilt,
        "backend": backend,
    }


# ============================================================
# INPUT TABLE
# ============================================================

def build_input(ghi, fixed):

    n = min(
        len(ghi),
        len(fixed),
    )

    if n == 0:
        raise ValueError(
            "No common GHI / Actual rows found."
        )

    out = ghi.iloc[:n].copy()

    out["Actual"] = (
        fixed["Actual"]
        .iloc[:n]
        .to_numpy()
    )

    return out


def apply_input(user_df):

    user_df = user_df.copy()

    required = GHI_COLS + ["Actual"]

    for col in required:

        if col not in user_df.columns:
            raise ValueError(
                f"Missing input column: {col}"
            )

        user_df[col] = numeric(
            user_df[col]
        )

    ghi = user_df[GHI_COLS].copy()

    fixed = pd.DataFrame({
        "Actual": user_df["Actual"]
    })

    return ghi.reset_index(drop=True), fixed.reset_index(drop=True)


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def geometry(ghi, fixed, latitude, tilt):

    n = min(
        len(ghi),
        len(fixed),
    )

    ghi = ghi.iloc[:n].copy()
    fixed = fixed.iloc[:n].copy()

    # Use actual dates when available.
    if "DateTime" in fixed:

        dates = pd.to_datetime(
            fixed["DateTime"],
            errors="coerce",
        )

        dates = dates.ffill().bfill()

    else:

        dates = pd.Series(
            pd.Timestamp.today(),
            index=fixed.index,
        )

    day = dates.dt.dayofyear.to_numpy()

    month = dates.dt.month_name().fillna(
        pd.Timestamp.today().strftime("%B")
    )

    declination = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + day)
                / 365
            )
        )
    )

    elevation = (
        90
        - latitude
        + declination
    )

    tilt_values = np.array([
        tilt.get(
            str(m).lower(),
            0,
        )
        for m in month
    ])

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(
            elevation + tilt_values
        )
    )

    result = fixed.copy()

    for i, cluster in enumerate(CLUSTERS):

        result[
            f"POA {cluster}"
        ] = (
            ghi[f"GHI {cluster}"]
            * sin_ab
            / np.where(
                np.abs(sin_a) < 1e-6,
                np.nan,
                sin_a,
            )
        )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)


# ============================================================
# EFFICIENCY
# ============================================================

def calculate_efficiency(area, error):

    result = area.copy()

    result["Error %"] = error

    result["Net Efficiency (%)"] = np.maximum(
        result["Standard PV Efficiency (%)"]
        - error,
        0,
    )

    result["Eff Area"] = (
        result["Net Efficiency (%)"]
        * result["Total area (m2)"]
        / 100
    )

    weights = (
        result.groupby("Clusters")[
            "Eff Area"
        ]
        .sum()
        .reindex(CLUSTERS)
        .fillna(0)
    )

    cluster = pd.DataFrame({
        "Clusters": CLUSTERS,
        "Eff Area(m2)": weights.values,
    })

    return result, cluster


# ============================================================
# FIXED FORECAST
# ============================================================

def fixed_forecast(geo, cluster):

    out = geo.copy()

    for i, c in enumerate(CLUSTERS):

        area = safe_float(
            cluster.loc[
                cluster["Clusters"] == c,
                "Eff Area(m2)"
            ].iloc[0]
            if not cluster.loc[
                cluster["Clusters"] == c
            ].empty
            else 0
        )

        out[
            POWER_COLS[i]
        ] = (
            numeric(
                out[f"POA {c}"]
            )
            * area
            / 1_000_000
        )

    out[TOTAL_POWER] = (
        out[POWER_COLS]
        .sum(axis=1)
    )

    return out


# ============================================================
# AUTOMATIC ERROR OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_error(area, geo, actual_tuple):

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    peak = actual.max()

    if peak <= 0:
        raise ValueError(
            "Actual Power has no positive values."
        )

    best_error = 0
    best_score = np.inf
    table = []

    for error in np.arange(
        0,
        20.01,
        0.1,
    ):

        _, cluster = calculate_efficiency(
            area,
            error,
        )

        forecast_df = fixed_forecast(
            geo,
            cluster,
        )

        forecast = arr(
            forecast_df[TOTAL_POWER]
        )

        n = min(
            len(actual),
            len(forecast),
        )

        a = actual[:n]
        f = forecast[:n]

        peak_error = abs(
            a.max() - f.max()
        )

        energy_error = abs(
            a.sum() - f.sum()
        )

        block_error = np.mean(
            np.abs(a - f)
        )

        score = (
            0.80
            * block_error
            / max(peak, 1e-6)
            + 0.10
            * peak_error
            / max(peak, 1e-6)
            + 0.10
            * energy_error
            / max(a.sum(), 1e-6)
        )

        table.append({
            "Error %": round(error, 1),
            "Forecast Peak": f.max(),
            "Actual Peak": a.max(),
            "Peak Error": peak_error,
            "Peak Error %": (
                peak_error
                / peak
                * 100
            ),
            "Score": score,
        })

        if score < best_score:

            best_score = score
            best_error = error

    return (
        round(float(best_error), 1),
        pd.DataFrame(table),
    )


# ============================================================
# TRACKING
# ============================================================

def tracking_data(workbook, ghi, geo, cluster):

    backend = workbook["backend"]

    if "C11" not in backend:
        raise ValueError(
            "Backend Cal C11 sheet is required for Tracking."
        )

    blocks = arr(
        backend["C11"]["Block"]
    )

    n = min(
        len(blocks),
        len(ghi),
        len(geo),
    )

    matrix = np.column_stack([
        arr(
            ghi[f"GHI {c}"]
        )[:n]
        for c in CLUSTERS
    ])

    actual = arr(
        geo["Actual"]
    )[:n]

    weights = (
        cluster["Eff Area(m2)"]
        .to_numpy(float)
    )

    return (
        blocks[:n],
        matrix,
        actual,
        weights,
    )


def tracking_calc(
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

    d1 = start - 1 - maximum
    d2 = end + 1 - maximum

    if d1 == 0 or d2 == 0:
        return None

    m1 = 90 / d1
    m2 = 90 / d2

    zenith = np.where(
        blocks <= maximum,
        m1 * (
            blocks - maximum
        ),
        m2 * (
            blocks - maximum
        ),
    )

    zenith = np.clip(
        zenith,
        -89,
        89,
    )

    panel = np.where(
        blocks < maximum,
        np.minimum(
            zenith,
            abs(east),
        ),
        np.minimum(
            zenith,
            abs(west),
        ),
    )

    cos_panel = np.clip(
        np.cos(
            np.radians(panel)
        ),
        1e-4,
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

    forecast = power.sum(axis=1)

    return forecast


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

    mask = actual > 0

    if not mask.any():
        raise ValueError(
            "Tracking Actual Power contains no positive values."
        )

    a = actual[mask]

    peak = max(a.max(), 1e-6)
    energy = max(a.sum(), 1e-6)

    def objective(x):

        p = np.rint(x).astype(int)

        forecast = tracking_calc(
            *p,
            blocks,
            ghi,
            weights,
        )

        if forecast is None:
            return 1e9

        f = forecast[mask]

        if not np.all(
            np.isfinite(f)
        ):
            return 1e9

        block = np.mean(
            np.abs(a - f)
        ) / peak

        peak_error = abs(
            a.max() - f.max()
        ) / peak

        energy_error = abs(
            a.sum() - f.sum()
        ) / energy

        return (
            0.80 * block
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(
        objective,
        TRACKING_BOUNDS,
        maxiter=35,
        popsize=10,
        seed=42,
        tol=0.001,
        polish=True,
        workers=1,
    )

    p = np.rint(
        result.x
    ).astype(int)

    return {
        "DHI": int(p[0]),
        "GHI Starting Block": int(p[1]),
        "GHI Ending Block": int(p[2]),
        "GHI Max Block": int(p[3]),
        "Tracking East Limit": int(p[4]),
        "Tracking West Limit": int(p[5]),
    }


# ============================================================
# FINAL CALCULATION
# ============================================================

def calculate_all(
    workbook,
    input_df,
    plant,
    error=None,
    tracking_params=None,
    optimize=True,
):

    # ------------------------------------------
    # Input
    # ------------------------------------------

    ghi, fixed = apply_input(
        input_df
    )

    # ------------------------------------------
    # Geometry
    # ------------------------------------------

    geo = geometry(
        ghi,
        fixed,
        workbook["latitude"],
        workbook["tilt"],
    )

    # Actual must remain available
    geo["Actual"] = fixed["Actual"].to_numpy()

    # ------------------------------------------
    # Error optimization
    # ------------------------------------------

    if optimize or error is None:

        error, error_table = optimize_error(
            workbook["area"],
            geo,
            tuple(
                geo["Actual"].to_numpy()
            ),
        )

    else:

        error_table = pd.DataFrame()

    # ------------------------------------------
    # Efficiency
    # ------------------------------------------

    final_area, cluster = (
        calculate_efficiency(
            workbook["area"],
            error,
        )
    )

    # ------------------------------------------
    # Fixed
    # ------------------------------------------

    fixed_result = fixed_forecast(
        geo,
        cluster,
    )

    result = {
        "error": error,
        "error_table": error_table,
        "area": final_area,
        "cluster": cluster,
        "geo": geo,
        "fixed": fixed_result,
        "forecast": arr(
            fixed_result[TOTAL_POWER]
        ),
        "actual": arr(
            geo["Actual"]
        ),
        "tracking_params": None,
    }

    # ------------------------------------------
    # Tracking
    # ------------------------------------------

    if plant == "Tracking":

        blocks, ghi_matrix, actual, weights = (
            tracking_data(
                workbook,
                ghi,
                geo,
                cluster,
            )
        )

        if tracking_params is None:

            tracking_params = optimize_tracking(
                tuple(blocks.tolist()),
                tuple(
                    tuple(x)
                    for x in ghi_matrix
                ),
                tuple(actual.tolist()),
                tuple(weights.tolist()),
            )

        p = tracking_params

        forecast = tracking_calc(
            p["DHI"],
            p["GHI Starting Block"],
            p["GHI Ending Block"],
            p["GHI Max Block"],
            p["Tracking East Limit"],
            p["Tracking West Limit"],
            blocks,
            ghi_matrix,
            weights,
        )

        if forecast is None:
            raise ValueError(
                "Invalid Tracking parameters."
            )

        result["forecast"] = forecast
        result["actual"] = actual
        result["tracking_params"] = p

    return result


# ============================================================
# GRAPH
# ============================================================

def graph(actual, forecast, plant):

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
        title=f"{plant} Plant | Actual vs Forecast",
        height=450,
        template="plotly_white",
        hovermode="x unified",
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
# HEADER
# ============================================================

st.markdown(
    '<div class="title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Automatic calculation with editable parameters'
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
        "Upload your Solar Excel workbook to begin."
    )

    st.stop()


# ============================================================
# FILE CHANGE
# ============================================================

file_hash = sha(
    uploaded.getvalue()
)

if (
    st.session_state.file_hash
    != file_hash
):

    st.session_state.file_hash = file_hash
    st.session_state.input_df = None
    st.session_state.results = None
    st.session_state.plant = "Fixed"

    for key in [
        "plant_selector",
        "error_input",
        "dhi_input",
        "start_input",
        "end_input",
        "max_input",
        "east_input",
        "west_input",
    ]:

        st.session_state.pop(
            key,
            None,
        )


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    workbook = prepare_workbook(
        uploaded.getvalue()
    )

except Exception as e:

    st.error(
        f"Input preparation failed:\n\n{e}"
    )

    st.stop()


# ============================================================
# CREATE INPUT
# ============================================================

if st.session_state.input_df is None:

    try:

        st.session_state.input_df = (
            build_input(
                workbook["ghi"],
                workbook["fixed"],
            )
        )

    except Exception as e:

        st.error(
            f"Input creation failed:\n\n{e}"
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
    height=280,
    hide_index=True,
    num_rows="fixed",
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
    "Plant",
    ["Fixed", "Tracking"],
    default=st.session_state.plant,
    selection_mode="single",
    width="stretch",
    label_visibility="collapsed",
    key="plant_selector",
)

if plant is None:
    plant = "Fixed"

if plant != st.session_state.plant:

    st.session_state.plant = plant
    reset_results()


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
            "Calculating... Please wait."
        ):

            st.session_state.input_df = (
                input_df.copy()
            )

            result = calculate_all(
                workbook,
                input_df,
                plant,
                optimize=True,
            )

            st.session_state.results = result

        st.success(
            "Automatic calculation completed."
        )

    except Exception as e:

        st.session_state.results = None

        st.error(
            f"Calculation failed:\n\n{e}"
        )


# ============================================================
# RESULTS CHECK
# ============================================================

if st.session_state.results is None:

    st.caption(
        "Edit the input if required, select Fixed or Tracking, "
        "then click Run Automatic Calculation."
    )

    st.stop()


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
        data["error"]
    ),
    step=0.1,
    format="%.1f",
    key="error_input",
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

tracking_params = None

if plant == "Tracking":

    p = data["tracking_params"]

    if p is None:
        st.error(
            "Tracking parameters unavailable."
        )
        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(p["DHI"]),
            step=1,
            key="dhi_input",
        )

        start = st.number_input(
            "GHI Starting Block",
            min_value=0,
            max_value=95,
            value=int(
                p["GHI Starting Block"]
            ),
            step=1,
            key="start_input",
        )

    with c2:

        end = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                p["GHI Ending Block"]
            ),
            step=1,
            key="end_input",
        )

        maximum = st.number_input(
            "GHI Max Block",
            min_value=0,
            max_value=95,
            value=int(
                p["GHI Max Block"]
            ),
            step=1,
            key="max_input",
        )

    with c3:

        east = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            value=int(
                p["Tracking East Limit"]
            ),
            step=1,
            key="east_input",
        )

        west = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            value=int(
                p["Tracking West Limit"]
            ),
            step=1,
            key="west_input",
        )

    tracking_params = {
        "DHI": int(dhi),
        "GHI Starting Block": int(start),
        "GHI Ending Block": int(end),
        "GHI Max Block": int(maximum),
        "Tracking East Limit": int(east),
        "Tracking West Limit": int(west),
    }


# ============================================================
# LIVE FINAL CALCULATION
# NO OPTIMIZATION HERE
# ============================================================

try:

    final = calculate_all(
        workbook,
        input_df,
        plant,
        error=error,
        tracking_params=tracking_params,
        optimize=False,
    )

except Exception as e:

    st.error(
        f"Forecast calculation failed:\n\n{e}"
    )

    st.stop()


# ============================================================
# METRICS
# ============================================================

actual = final["actual"]
forecast = final["forecast"]

actual_peak = (
    float(actual.max())
    if len(actual)
    else 0
)

forecast_peak = (
    float(forecast.max())
    if len(forecast)
    else 0
)

peak_error = abs(
    actual_peak
    - forecast_peak
)

peak_error_pct = (
    peak_error
    / actual_peak
    * 100
    if actual_peak
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

with c3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Peak Error</div>
            <div class="metric-value">
                {peak_error_pct:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FORECAST GRAPH
# ============================================================

st.markdown(
    '<div class="section">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)

st.plotly_chart(
    graph(
        actual,
        forecast,
        plant,
    ),
    width="stretch",
    config={
        "displayModeBar": False,
        "responsive": True,
    },
)


# ============================================================
# OPTIMIZATION DETAILS
# ============================================================

with st.expander(
    "🔎 Calculation Details"
):

    st.write(
        f"**Automatic Error %:** "
        f"{data['error']:.1f}%"
    )

    st.write(
        f"**Latitude:** "
        f"{workbook['latitude']:.4f}"
    )

    st.write(
        f"**Clusters:** "
        f"{', '.join(CLUSTERS)}"
    )

    st.dataframe(
        final["area"],
        width="stretch",
        hide_index=True,
    )

    if plant == "Tracking":

        st.write(
            "**Tracking Parameters**"
        )

        st.dataframe(
            pd.DataFrame(
                [
                    final["tracking_params"]
                ]
            ),
            width="stretch",
            hide_index=True,
        )
