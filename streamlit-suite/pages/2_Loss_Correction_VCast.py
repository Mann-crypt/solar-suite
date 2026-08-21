# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# Compact + Robust Workbook Detection
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

st.markdown("""
<style>
.block-container {
    max-width: 1500px;
    padding-top: 1rem;
}
.section {
    font-size: 19px;
    font-weight: 700;
    margin: 18px 0 8px;
}
.metric-card {
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 14px;
}
.metric-label {
    color: #777;
    font-size: 12px;
}
.metric-value {
    font-size: 24px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


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
# BASIC HELPERS
# ============================================================

def clean_name(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace("*", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "")
        .replace(".", "")
    )


def numeric(x):
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").fillna(0)

    if isinstance(x, pd.DataFrame):
        return pd.to_numeric(
            x.iloc[:, 0],
            errors="coerce"
        ).fillna(0)

    return pd.Series(
        pd.to_numeric(
            np.asarray(x).reshape(-1),
            errors="coerce"
        )
    ).fillna(0)


def arr(x):
    return numeric(x).to_numpy(dtype=float)


def scalar(x, default=0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except:
        return default


def df_copy(x):
    if not isinstance(x, pd.DataFrame):
        raise TypeError(
            f"Expected DataFrame, got {type(x).__name__}"
        )
    return x.copy()


def find_column(df, names):
    lookup = {
        clean_name(c): c
        for c in df.columns
    }

    for name in names:
        key = clean_name(name)

        if key in lookup:
            return lookup[key]

    return None


def first_numeric_column(df, exclude=None):
    exclude = exclude or []

    for c in df.columns:

        if c in exclude:
            continue

        values = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        if values.notna().sum() >= 2:
            return c

    return None


def hash_file(uploaded):
    return hashlib.sha256(
        uploaded.getvalue()
    ).hexdigest()


# ============================================================
# READ SHEET
# ============================================================

def read_sheet(data, sheet, header=None, usecols=None):

    try:
        return pd.read_excel(
            io.BytesIO(data),
            sheet_name=sheet,
            header=header,
            usecols=usecols,
        )

    except Exception:
        # Retry without usecols if workbook layout differs
        return pd.read_excel(
            io.BytesIO(data),
            sheet_name=sheet,
            header=header,
        )


# ============================================================
# WORKBOOK
# ============================================================

@st.cache_data(show_spinner=False)
def load_book(data):

    xls = pd.ExcelFile(
        io.BytesIO(data)
    )

    sheets = xls.sheet_names

    def get(name, header=None, usecols=None):

        if name not in sheets:
            return pd.DataFrame()

        return read_sheet(
            data,
            name,
            header,
            usecols,
        )

    book = {
        "area": get(
            "Area & Efficiency",
            header=1
        ),

        "result": get(
            "Result"
        ),

        "forecast": get(
            "Forecast Config",
            header=8
        ),

        "tilt": get(
            "Config Tilt Angle",
            header=7
        ),

        "fixed": get(
            "Fixed-C11",
            header=1
        ),

        "tracking": get(
            "Tracking",
            header=1
        ),

        "backend": {},
    }

    for c in CLUSTERS:

        name = f"Backend Cal {c}"

        if name in sheets:
            book["backend"][c] = get(name)

    return book


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def prepare_area(df):

    df = df_copy(df)

    if df.empty:
        raise ValueError(
            "Area & Efficiency sheet is empty."
        )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    cluster_col = find_column(
        df,
        ["Clusters", "Cluster"]
    )

    eff_col = find_column(
        df,
        [
            "Standard PV Efficiency (%)",
            "PV Efficiency (%)",
            "Efficiency (%)",
            "Efficiency",
        ]
    )

    module_col = find_column(
        df,
        [
            "No of Module",
            "No. of Module",
            "Number of Module",
            "Modules",
            "Module",
        ]
    )

    area_col = find_column(
        df,
        [
            "Area of 1 Module (m2)",
            "Area of 1 Module",
            "Module Area",
            "Area",
        ]
    )

    # --------------------------------------------------------
    # If exact columns don't exist, try positional detection
    # --------------------------------------------------------

    if cluster_col is None:

        for c in df.columns:

            vals = (
                df[c]
                .astype(str)
                .str.upper()
            )

            if vals.str.contains("C11").any():
                cluster_col = c
                break

    if eff_col is None:

        candidates = []

        for c in df.columns:

            v = pd.to_numeric(
                df[c],
                errors="coerce"
            )

            if v.notna().sum() >= 2:
                med = v.dropna().median()

                if 0 < med <= 100:
                    candidates.append(c)

        if candidates:
            eff_col = candidates[0]

    if module_col is None:

        candidates = []

        for c in df.columns:

            v = pd.to_numeric(
                df[c],
                errors="coerce"
            )

            if v.notna().sum() >= 2:

                med = v.dropna().median()

                if med > 1:
                    candidates.append(c)

        if candidates:
            module_col = candidates[0]

    if area_col is None:

        candidates = []

        for c in df.columns:

            v = pd.to_numeric(
                df[c],
                errors="coerce"
            )

            if v.notna().sum() >= 2:

                med = v.dropna().median()

                if 0 < med < 100:
                    candidates.append(c)

        if candidates:
            area_col = candidates[-1]

    missing = []

    if cluster_col is None:
        missing.append("Clusters")

    if eff_col is None:
        missing.append("Efficiency")

    if module_col is None:
        missing.append("No of Module")

    if area_col is None:
        missing.append("Area of 1 Module")

    if missing:
        raise ValueError(
            "Could not identify Area & Efficiency columns: "
            + ", ".join(missing)
            + "\n\nAvailable columns: "
            + str(list(df.columns))
        )

    out = pd.DataFrame()

    out["Clusters"] = (
        df[cluster_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out["Standard PV Efficiency (%)"] = numeric(
        df[eff_col]
    )

    out["No of Module"] = numeric(
        df[module_col]
    )

    out["Area of 1 Module (m2)"] = numeric(
        df[area_col]
    )

    out["Total area (m2)"] = (
        out["No of Module"]
        * out["Area of 1 Module (m2)"]
    )

    out = out[
        out["Clusters"].isin(CLUSTERS)
    ]

    return out.reset_index(drop=True)


# ============================================================
# CLUSTER AREAS
# ============================================================

def prepare_cluster(area):

    result = pd.DataFrame({
        "Clusters": CLUSTERS
    })

    areas = (
        area.groupby("Clusters")[
            "Total area (m2)"
        ]
        .sum()
    )

    result["Total area (m2)"] = (
        result["Clusters"]
        .map(areas)
        .fillna(0)
    )

    return result


# ============================================================
# GHI
# ============================================================

def prepare_ghi(df):

    df = df_copy(df)

    if df.empty:
        raise ValueError(
            "Result sheet is empty."
        )

    out = pd.DataFrame()

    # First try exact names
    for i, wanted in enumerate(GHI_COLS):

        col = find_column(
            df,
            [
                wanted,
                f"GHI C{i+11}",
                f"GHI_{i+11}",
                f"C{i+11} GHI",
            ]
        )

        if col is not None:
            out[wanted] = numeric(
                df[col]
            )

    # If names were not found, use numeric columns
    if len(out.columns) < 5:

        numeric_cols = []

        for c in df.columns:

            v = pd.to_numeric(
                df[c],
                errors="coerce"
            )

            if v.notna().sum() > 2:
                numeric_cols.append(c)

        # Remove columns already used
        used = set(
            [
                find_column(
                    df,
                    [c]
                )
                for c in out.columns
            ]
        )

        numeric_cols = [
            c for c in numeric_cols
            if c not in used
        ]

        for c in numeric_cols:

            if len(out.columns) >= 5:
                break

            name = GHI_COLS[
                len(out.columns)
            ]

            out[name] = numeric(
                df[c]
            )

    if len(out.columns) != 5:

        raise ValueError(
            "Could not identify 5 GHI columns.\n\n"
            f"Available columns: {list(df.columns)}"
        )

    return out.reset_index(drop=True)


# ============================================================
# ACTUAL POWER
# ============================================================

def prepare_actual(df):

    df = df_copy(df)

    if df.empty:
        raise ValueError(
            "Fixed-C11 sheet is empty."
        )

    col = find_column(
        df,
        [
            "Actual",
            "Actual Power",
            "Actual Power MW",
            "Actual(MW)",
            "Actual Power (MW)",
            "Power Actual",
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Some workbooks have headers incorrectly read as row 1.
    # Search ALL columns instead of assuming header position.
    # --------------------------------------------------------

    if col is None:

        # Check whether one of the column names itself
        # looks like a date or value
        for c in df.columns:

            name = str(c).strip().lower()

            if (
                "actual" in name
                or "power" in name
            ):
                col = c
                break

    if col is None:

        # Fallback:
        # choose the last useful numeric column
        candidates = []

        for c in df.columns:

            v = pd.to_numeric(
                df[c],
                errors="coerce"
            )

            if v.notna().sum() > 2:
                candidates.append(c)

        if candidates:
            col = candidates[-1]

    if col is None:

        raise ValueError(
            "Actual Power column could not be identified.\n\n"
            f"Available columns: {list(df.columns)}"
        )

    actual = numeric(
        df[col]
    ).reset_index(drop=True)

    return actual


# ============================================================
# LATITUDE
# ============================================================

def prepare_latitude(df):

    df = df_copy(df)

    if df.empty:
        return 28.6

    col = find_column(
        df,
        [
            "Lat",
            "Latitude"
        ]
    )

    if col is None:

        for c in df.columns:

            if "lat" in clean_name(c):
                col = c
                break

    if col is None:
        return 28.6

    values = numeric(
        df[col]
    )

    values = values[
        values != 0
    ]

    if values.empty:
        return 28.6

    return float(
        values.iloc[0]
    )


# ============================================================
# TILT
# ============================================================

def prepare_tilt(df):

    # Default is intentionally safe
    default = {
        "January": 0,
        "February": 0,
        "March": 0,
        "April": 0,
        "May": 0,
        "June": 0,
        "July": 0,
        "August": 0,
        "September": 0,
        "October": 0,
        "November": 0,
        "December": 0,
    }

    if not isinstance(
        df,
        pd.DataFrame
    ) or df.empty:
        return default

    df = df.copy()

    month_col = None
    fixed_col = None

    for c in df.columns:

        key = clean_name(c)

        if key == "month":
            month_col = c

        if key == "fixed":
            fixed_col = c

    # If month header is not present,
    # find column containing month names.
    if month_col is None:

        months = {
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
        }

        for c in df.columns:

            vals = (
                df[c]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            count = vals.isin(months).sum()

            if count >= 2:
                month_col = c
                break

    if fixed_col is None:

        for c in df.columns:

            if clean_name(c) == "fixed":
                fixed_col = c
                break

    if month_col is None or fixed_col is None:

        # Do NOT fail calculation.
        # Return zero tilt.
        return default

    for _, row in df.iterrows():

        month = str(
            row[month_col]
        ).strip()

        if month in default:

            value = scalar(
                row[fixed_col],
                0
            )

            default[month] = value

    return default


# ============================================================
# BUILD INPUT
# ============================================================

def build_input(ghi, actual):

    n = min(
        len(ghi),
        len(actual)
    )

    if n <= 0:
        raise ValueError(
            "No overlapping GHI and Actual data."
        )

    out = ghi.iloc[:n].copy()

    out["Actual"] = (
        actual.iloc[:n]
        .to_numpy()
    )

    return out.reset_index(drop=True)


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def geometry(
    ghi,
    actual,
    latitude,
    tilt_lookup
):

    n = min(
        len(ghi),
        len(actual)
    )

    ghi = ghi.iloc[:n].copy()

    actual = (
        actual.iloc[:n]
        .reset_index(drop=True)
    )

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

    tilt = scalar(
        tilt_lookup.get(
            today.strftime("%B"),
            0
        )
    )

    a = np.radians(elevation)
    ab = np.radians(
        elevation + tilt
    )

    sin_a = np.sin(a)

    if abs(sin_a) < 1e-8:
        sin_a = 1e-8

    out = pd.DataFrame(index=range(n))

    for i in range(5):

        g = arr(
            ghi[GHI_COLS[i]]
        )[:n]

        out[
            f"GHI*sin(a)-CL{i+1}"
        ] = (
            g * sin_a
        )

        out[
            f"GHI*sin(a+b)-CL{i+1}"
        ] = (
            g * np.sin(ab)
        )

        out[
            POA_COLS[i]
        ] = (
            g
            * np.sin(ab)
            / sin_a
        )

    out["Actual"] = arr(actual)

    return out


# ============================================================
# EFFECTIVE AREA
# ============================================================

def calculate_area(
    area,
    cluster,
    error
):

    a = area.copy()
    c = cluster.copy()

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

    sums = (
        a.groupby("Clusters")[
            "Eff Area"
        ]
        .sum()
    )

    c["Eff Area(m2)"] = (
        c["Clusters"]
        .map(sums)
        .fillna(0)
    )

    return a, c


# ============================================================
# FIXED POWER
# ============================================================

def fixed_forecast(
    geometry_df,
    cluster
):

    result = geometry_df.copy()

    weights = (
        pd.to_numeric(
            cluster["Eff Area(m2)"],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    # Always exactly 5 weights
    weights = np.resize(
        weights,
        5
    )

    for i in range(5):

        result[
            POWER_COLS[i]
        ] = (
            numeric(
                result[
                    POA_COLS[i]
                ]
            )
            * weights[i]
            / 1_000_000
        )

    result[TOTAL_POWER] = (
        result[POWER_COLS]
        .sum(axis=1)
    )

    return result


# ============================================================
# AUTOMATIC ERROR
# ============================================================

@st.cache_data(show_spinner=False)
def optimize_error(
    area,
    cluster,
    geometry_df
):

    actual = arr(
        geometry_df["Actual"]
    )

    peak = (
        np.max(actual)
        if len(actual)
        else 0
    )

    if peak <= 0:
        raise ValueError(
            "Actual Power contains no positive values."
        )

    rows = []

    for error in np.arange(
        0,
        10.01,
        0.1
    ):

        _, c = calculate_area(
            area,
            cluster,
            error
        )

        f = fixed_forecast(
            geometry_df,
            c
        )

        forecast = arr(
            f[TOTAL_POWER]
        )

        fp = (
            np.max(forecast)
            if len(forecast)
            else 0
        )

        pe = abs(
            fp - peak
        )

        rows.append([
            round(error, 1),
            fp,
            pe,
            pe / peak * 100
        ])

    table = pd.DataFrame(
        rows,
        columns=[
            "Error %",
            "Forecast Peak",
            "Peak Error",
            "Peak Error %",
        ]
    )

    best = table.loc[
        table["Peak Error"].idxmin()
    ]

    return (
        float(best["Error %"]),
        table
    )


# ============================================================
# TRACKING
# ============================================================

def tracking_data(
    book,
    ghi,
    geometry_df,
    cluster
):

    backend = book.get(
        "backend",
        {}
    )

    if "C11" not in backend:
        raise ValueError(
            "Backend Cal C11 sheet not found."
        )

    b = backend["C11"]

    block_col = find_column(
        b,
        [
            "Block No.",
            "Block",
            "Block No"
        ]
    )

    if block_col is None:
        block_col = b.columns[0]

    blocks = arr(
        b[block_col]
    )

    n = min(
        len(blocks),
        len(ghi),
        len(geometry_df)
    )

    matrix = np.column_stack([
        arr(
            ghi[c]
        )[:n]
        for c in GHI_COLS
    ])

    actual = arr(
        geometry_df["Actual"]
    )[:n]

    weights = (
        pd.to_numeric(
            cluster["Eff Area(m2)"],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    weights = np.resize(
        weights,
        5
    )

    return (
        blocks[:n],
        matrix,
        actual,
        weights
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
    weights
):

    if not (
        start < maximum < end
    ):
        return None

    d1 = start - maximum
    d2 = end - maximum

    if d1 == 0 or d2 == 0:
        return None

    morning = (
        90 / abs(d1)
    )

    evening = (
        90 / abs(d2)
    )

    angle = np.where(
        blocks <= maximum,
        (blocks - maximum) * morning,
        (blocks - maximum) * evening
    )

    angle = np.clip(
        angle,
        -89,
        89
    )

    angle = np.where(
        blocks < maximum,
        np.clip(
            angle,
            -abs(east),
            abs(east)
        ),
        np.clip(
            angle,
            -abs(west),
            abs(west)
        )
    )

    cos_a = np.clip(
        np.cos(
            np.radians(angle)
        ),
        0.01,
        None
    )

    dhi_part = (
        ghi
        * dhi
        / 100
    )

    dni = (
        ghi - dhi_part
    ) / cos_a[:, None]

    power = (
        dni
        * weights
        / 1_000_000
    )

    forecast = power.sum(
        axis=1
    )

    return forecast


@st.cache_data(show_spinner=False)
def optimize_tracking(
    blocks,
    ghi,
    actual,
    weights
):

    blocks = np.asarray(
        blocks,
        dtype=float
    )

    ghi = np.asarray(
        ghi,
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float
    )

    weights = np.asarray(
        weights,
        dtype=float
    )

    mask = actual > 0

    if not mask.any():
        raise ValueError(
            "No positive Actual Power."
        )

    a = actual[mask]

    def objective(x):

        p = np.rint(x).astype(int)

        forecast = tracking_calc(
            *p,
            blocks,
            ghi,
            weights
        )

        if forecast is None:
            return 1e9

        f = forecast[mask]

        if not np.all(
            np.isfinite(f)
        ):
            return 1e9

        rmse = np.mean(
            np.abs(a - f)
        ) / max(
            np.max(a),
            1e-9
        )

        peak = abs(
            np.max(a)
            - np.max(f)
        ) / max(
            np.max(a),
            1e-9
        )

        energy = abs(
            np.sum(a)
            - np.sum(f)
        ) / max(
            np.sum(a),
            1e-9
        )

        return (
            0.8 * rmse
            + 0.1 * peak
            + 0.1 * energy
        )

    result = differential_evolution(
        objective,
        TRACKING_BOUNDS,
        maxiter=25,
        popsize=8,
        seed=42,
        polish=False
    )

    x = np.rint(
        result.x
    ).astype(int)

    return {
        "DHI": x[0],
        "GHI Starting Block": x[1],
        "GHI Ending Block": x[2],
        "GHI Max Block": x[3],
        "Tracking East Limit": x[4],
        "Tracking West Limit": x[5],
    }


# ============================================================
# COMPLETE CALCULATION
# ============================================================

def calculate_all(
    book,
    input_df,
    plant
):

    # --------------------------------------------------------
    # PREPARE INPUT
    # --------------------------------------------------------

    area = prepare_area(
        book["area"]
    )

    cluster = prepare_cluster(
        area
    )

    ghi = input_df[
        GHI_COLS
    ].copy()

    actual = numeric(
        input_df["Actual"]
    )

    latitude = prepare_latitude(
        book["forecast"]
    )

    tilt = prepare_tilt(
        book["tilt"]
    )

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    geo = geometry(
        ghi,
        actual,
        latitude,
        tilt
    )

    # --------------------------------------------------------
    # AUTOMATIC ERROR
    # --------------------------------------------------------

    best_error, error_table = (
        optimize_error(
            area,
            cluster,
            geo
        )
    )

    final_area, final_cluster = (
        calculate_area(
            area,
            cluster,
            best_error
        )
    )

    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    fixed = fixed_forecast(
        geo,
        final_cluster
    )

    result = {
        "area": area,
        "cluster": cluster,
        "final_area": final_area,
        "final_cluster": final_cluster,
        "geometry": geo,
        "fixed": fixed,
        "best_error": best_error,
        "error_table": error_table,
        "tracking": None,
    }

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    if plant == "Tracking":

        blocks, matrix, act, weights = (
            tracking_data(
                book,
                ghi,
                geo,
                final_cluster
            )
        )

        params = optimize_tracking(
            tuple(blocks),
            tuple(
                tuple(x)
                for x in matrix
            ),
            tuple(act),
            tuple(weights)
        )

        result["tracking"] = {
            "params": params,
            "blocks": blocks,
            "ghi": matrix,
            "actual": act,
            "weights": weights,
        }

    return result


# ============================================================
# GRAPH
# ============================================================

def graph(
    actual,
    forecast,
    title
):

    n = min(
        len(actual),
        len(forecast)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual[:n],
            name="Actual",
            mode="lines"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=forecast[:n],
            name="Forecast",
            mode="lines"
        )
    )

    fig.update_layout(
        title=title,
        height=430,
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


# ============================================================
# UI
# ============================================================

st.title(
    "☀️ Solar Forecast Correction"
)

st.caption(
    "Automatic calculation with editable parameters"
)


uploaded = st.file_uploader(
    "Upload Solar Excel File",
    type=["xlsx", "xls"]
)

if uploaded is None:

    st.info(
        "Upload the Excel workbook to begin."
    )

    st.stop()


# ============================================================
# FILE RESET
# ============================================================

file_hash = hash_file(
    uploaded
)

if st.session_state.get(
    "file_hash"
) != file_hash:

    st.session_state.file_hash = (
        file_hash
    )

    st.session_state.result = None
    st.session_state.input_df = None


# ============================================================
# LOAD
# ============================================================

try:

    book = load_book(
        uploaded.getvalue()
    )

except Exception as e:

    st.error(
        f"Workbook loading failed: {e}"
    )

    st.stop()


# ============================================================
# INPUT PREPARATION
# ============================================================

try:

    ghi = prepare_ghi(
        book["result"]
    )

    actual = prepare_actual(
        book["fixed"]
    )

    if st.session_state.input_df is None:

        st.session_state.input_df = (
            build_input(
                ghi,
                actual
            )
        )

except Exception as e:

    st.error(
        f"Input preparation failed:\n\n{e}"
    )

    st.stop()


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section">📂 GHI / Actual Input</div>',
    unsafe_allow_html=True
)

input_df = st.data_editor(
    st.session_state.input_df,
    width="stretch",
    height=280,
    num_rows="fixed",
    hide_index=True,
    column_config={
        c: st.column_config.NumberColumn(
            c,
            format="%.3f"
        )
        for c in GHI_COLS + ["Actual"]
    }
)


# ============================================================
# PLANT
# ============================================================

st.markdown(
    '<div class="section">🌱 Plant Type</div>',
    unsafe_allow_html=True
)

plant = st.segmented_control(
    "Plant",
    ["Fixed", "Tracking"],
    default="Fixed",
    width="stretch",
    label_visibility="collapsed"
)


# ============================================================
# RUN
# ============================================================

run = st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    width="stretch"
)


if run:

    try:

        with st.spinner(
            "Calculating..."
        ):

            st.session_state.input_df = (
                input_df.copy()
            )

            result = calculate_all(
                book,
                input_df,
                plant
            )

            st.session_state.result = (
                result
            )

        st.success(
            "Calculation completed."
        )

    except Exception as e:

        st.session_state.result = None

        st.error(
            f"Calculation failed:\n\n{e}"
        )


# ============================================================
# RESULT CHECK
# ============================================================

result = st.session_state.get(
    "result"
)

if result is None:

    st.caption(
        "Edit the input if required, select the plant type, "
        "then click Run Automatic Calculation."
    )

    st.stop()


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section">⚙️ Parameters</div>',
    unsafe_allow_html=True
)

error = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=20.0,
    value=float(
        result["best_error"]
    ),
    step=0.1
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

tracking_values = None

if plant == "Tracking":

    tracking_values = result[
        "tracking"
    ]["params"]

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi = st.number_input(
            "DHI (%)",
            0,
            100,
            int(
                tracking_values["DHI"]
            )
        )

        start = st.number_input(
            "GHI Starting Block",
            0,
            95,
            int(
                tracking_values[
                    "GHI Starting Block"
                ]
            )
        )

    with c2:

        end = st.number_input(
            "GHI Ending Block",
            1,
            96,
            int(
                tracking_values[
                    "GHI Ending Block"
                ]
            )
        )

        maximum = st.number_input(
            "GHI Max Block",
            0,
            95,
            int(
                tracking_values[
                    "GHI Max Block"
                ]
            )
        )

    with c3:

        east = st.number_input(
            "Tracking East Limit",
            0,
            90,
            int(
                tracking_values[
                    "Tracking East Limit"
                ]
            )
        )

        west = st.number_input(
            "Tracking West Limit",
            0,
            90,
            int(
                tracking_values[
                    "Tracking West Limit"
                ]
            )
        )


# ============================================================
# FINAL FORECAST
# ============================================================

try:

    _, final_cluster = (
        calculate_area(
            result["area"],
            result["cluster"],
            error
        )
    )

    fixed = fixed_forecast(
        result["geometry"],
        final_cluster
    )

    actual = arr(
        result["geometry"]["Actual"]
    )

    if plant == "Fixed":

        forecast = arr(
            fixed[TOTAL_POWER]
        )

        title = (
            "Fixed Plant | Actual vs Forecast"
        )

    else:

        t = result["tracking"]

        if not (
            start < maximum < end
        ):
            raise ValueError(
                "GHI Starting Block < "
                "GHI Max Block < "
                "GHI Ending Block"
            )

        forecast = tracking_calc(
            int(dhi),
            int(start),
            int(end),
            int(maximum),
            int(east),
            int(west),
            t["blocks"],
            t["ghi"],
            np.asarray(
                final_cluster[
                    "Eff Area(m2)"
                ]
            )
        )

        title = (
            "Tracking Plant | Actual vs Forecast"
        )

except Exception as e:

    st.error(
        f"Forecast calculation failed:\n\n{e}"
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

actual_peak = (
    np.max(actual)
    if len(actual)
    else 0
)

forecast_peak = (
    np.max(forecast)
    if len(forecast)
    else 0
)

st.markdown(
    '<div class="section">📊 Results</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

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
        unsafe_allow_html=True
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
        unsafe_allow_html=True
    )


# ============================================================
# GRAPH
# ============================================================

st.markdown(
    '<div class="section">📈 Forecast Comparison</div>',
    unsafe_allow_html=True
)

st.plotly_chart(
    graph(
        actual,
        forecast,
        title
    ),
    width="stretch",
    config={
        "displayModeBar": False
    }
)
