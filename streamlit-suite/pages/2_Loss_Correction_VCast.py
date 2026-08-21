# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# Compact + Robust + Cloud Safe
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

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

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

DEFAULTS = {
    "file_hash": None,
    "input_df": None,
    "plant": "Fixed",
    "result": None,
    "calculated": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_results():
    st.session_state.result = None
    st.session_state.calculated = False


# ============================================================
# BASIC HELPERS
# ============================================================

def numeric_series(x):
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").fillna(0)

    return pd.to_numeric(
        pd.Series(x),
        errors="coerce",
    ).fillna(0)


def numeric_array(x):
    return numeric_series(x).to_numpy(dtype=float)


def safe_float(value, default=0.0):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def clean_columns(df):
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


def require_df(obj, name):
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"{name} was not loaded as a DataFrame. "
            f"Received {type(obj).__name__}."
        )

    return obj.copy()


def file_hash(uploaded):
    return hashlib.sha256(
        uploaded.getvalue()
    ).hexdigest()


# ============================================================
# EXCEL LOADING
# ============================================================

@st.cache_data(show_spinner=False, max_entries=3)
def load_excel(data):

    xls = pd.ExcelFile(
        io.BytesIO(data)
    )

    sheets = set(xls.sheet_names)

    def read(sheet, **kwargs):

        if sheet not in sheets:
            raise ValueError(
                f"Required sheet '{sheet}' not found. "
                f"Available sheets: {xls.sheet_names}"
            )

        return pd.read_excel(
            io.BytesIO(data),
            sheet_name=sheet,
            **kwargs,
        )

    workbook = {}

    # Main sheets
    workbook["area"] = read(
        "Area & Efficiency",
        header=1,
    )

    workbook["result"] = read(
        "Result",
    )

    workbook["forecast_config"] = read(
        "Forecast Config",
        header=8,
    )

    workbook["tilt"] = read(
        "Config Tilt Angle",
        header=7,
    )

    workbook["fixed"] = read(
        "Fixed-C11",
        header=1,
    )

    workbook["tracking"] = read(
        "Tracking",
        header=1,
    )

    # Backend sheets
    workbook["backend"] = {}

    for cluster in CLUSTERS:

        sheet = f"Backend Cal {cluster}"

        if sheet in sheets:
            workbook["backend"][cluster] = read(sheet)
        else:
            workbook["backend"][cluster] = pd.DataFrame()

    return workbook


# ============================================================
# AREA + EFFICIENCY
# ============================================================

def prepare_area(df):

    df = require_df(
        df,
        "Area & Efficiency",
    )

    df = clean_columns(df)

    required = [
        "Clusters",
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in Area & Efficiency: "
            + ", ".join(missing)
        )

    df = df[
        df["Clusters"].notna()
    ].copy()

    df["Clusters"] = (
        df["Clusters"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[
        df["Clusters"].isin(CLUSTERS)
    ].copy()

    for col in required[1:]:

        df[col] = numeric_series(
            df[col]
        )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df.reset_index(drop=True)


def calculate_cluster_area(area, error):

    error = safe_float(error)

    temp = area.copy()

    temp["Error %"] = error

    temp["Net Efficiency (%)"] = (
        temp["Standard PV Efficiency (%)"]
        - error
    )

    temp["Eff Area"] = (
        temp["Net Efficiency (%)"]
        * temp["Total area (m2)"]
        / 100
    )

    cluster = (
        temp.groupby("Clusters", as_index=False)
        ["Eff Area"]
        .sum()
        .rename(
            columns={
                "Eff Area": "Eff Area(m2)"
            }
        )
    )

    # Always return all five clusters
    cluster = (
        pd.DataFrame(
            {"Clusters": CLUSTERS}
        )
        .merge(
            cluster,
            on="Clusters",
            how="left",
        )
        .fillna(0)
    )

    return temp, cluster


# ============================================================
# GHI
# ============================================================

def prepare_ghi(df):

    df = require_df(
        df,
        "Result",
    )

    df = clean_columns(df)

    missing = [
        c for c in GHI_COLS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing GHI columns in Result: "
            + ", ".join(missing)
        )

    for col in GHI_COLS:
        df[col] = numeric_series(
            df[col]
        )

    return df.reset_index(drop=True)


# ============================================================
# ACTUAL
# ============================================================

def prepare_actual(df):

    df = require_df(
        df,
        "Fixed-C11",
    )

    df = clean_columns(df)

    if "Actual" not in df.columns:
        raise ValueError(
            "Column 'Actual' not found in Fixed-C11."
        )

    df["Actual"] = numeric_series(
        df["Actual"]
    )

    return df.reset_index(drop=True)


# ============================================================
# LATITUDE
# ============================================================

def prepare_latitude(df):

    df = require_df(
        df,
        "Forecast Config",
    )

    df = clean_columns(df)

    lat_col = next(
        (
            c for c in df.columns
            if str(c).strip().lower() == "lat"
        ),
        None,
    )

    if lat_col is None:

        lat_col = next(
            (
                c for c in df.columns
                if "lat" in str(c).lower()
            ),
            None,
        )

    if lat_col is None:
        raise ValueError(
            "Latitude column could not be found "
            "in Forecast Config."
        )

    values = numeric_series(
        df[lat_col]
    )

    values = values[
        values.notna()
    ]

    if values.empty:
        raise ValueError(
            "Latitude value is invalid."
        )

    return float(values.iloc[0])


# ============================================================
# MONTH / TILT
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


def find_month_column(df):

    # Exact names first
    for col in df.columns:

        name = (
            str(col)
            .strip()
            .lower()
            .replace("_", " ")
        )

        if name in {
            "month",
            "months",
            "month name",
            "monthname",
        }:
            return col

    # Detect from actual values
    month_set = {
        x.lower()
        for x in MONTHS
    }

    for col in df.columns:

        values = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if values.isin(
            month_set
        ).sum() >= 2:
            return col

    return None


def find_fixed_tilt_column(df):

    for col in df.columns:

        name = (
            str(col)
            .strip()
            .lower()
        )

        if name == "fixed":
            return col

    for col in df.columns:

        if "fixed" in str(col).lower():
            return col

    return None


def prepare_tilt(df):

    df = require_df(
        df,
        "Config Tilt Angle",
    )

    df = clean_columns(df)

    month_col = find_month_column(df)

    fixed_col = find_fixed_tilt_column(df)

    if month_col is None:

        # Last-resort detection:
        # sometimes Excel has month names in first column
        if len(df.columns) > 0:

            candidate = df.columns[0]

            values = (
                df[candidate]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            month_set = {
                x.lower()
                for x in MONTHS
            }

            if values.isin(
                month_set
            ).sum() >= 2:
                month_col = candidate

    if month_col is None:

        raise ValueError(
            "Could not identify the Month column "
            "in Config Tilt Angle. "
            f"Available columns: {list(df.columns)}"
        )

    if fixed_col is None:

        raise ValueError(
            "Could not identify the Fixed tilt column "
            "in Config Tilt Angle. "
            f"Available columns: {list(df.columns)}"
        )

    result = {}

    for _, row in df.iterrows():

        month = (
            str(row[month_col])
            .strip()
            .lower()
        )

        for valid_month in MONTHS:

            if month == valid_month.lower():

                value = safe_float(
                    row[fixed_col],
                    np.nan,
                )

                if np.isfinite(value):
                    result[valid_month] = value

                break

    if not result:

        raise ValueError(
            "No valid monthly Fixed tilt values "
            "were found in Config Tilt Angle."
        )

    return result


# ============================================================
# INPUT TABLE
# ============================================================

def create_input(ghi, actual):

    n = min(
        len(ghi),
        len(actual),
    )

    if n <= 0:
        raise ValueError(
            "No GHI / Actual data available."
        )

    result = pd.DataFrame(
        {
            col: numeric_series(
                ghi[col]
            ).iloc[:n].to_numpy()
            for col in GHI_COLS
        }
    )

    result["Actual"] = (
        numeric_series(
            actual["Actual"]
        )
        .iloc[:n]
        .to_numpy()
    )

    return result


def apply_input(user_df, ghi, actual):

    user_df = require_df(
        user_df,
        "Edited Input",
    )

    required = GHI_COLS + ["Actual"]

    missing = [
        c for c in required
        if c not in user_df.columns
    ]

    if missing:
        raise ValueError(
            "Input is missing: "
            + ", ".join(missing)
        )

    n = len(user_df)

    if n == 0:
        raise ValueError(
            "Input table is empty."
        )

    if n > len(ghi) or n > len(actual):
        raise ValueError(
            "Edited input has more rows than "
            "the original workbook."
        )

    new_ghi = ghi.iloc[:n].copy()
    new_actual = actual.iloc[:n].copy()

    for col in GHI_COLS:

        new_ghi[col] = numeric_series(
            user_df[col]
        ).to_numpy()

    new_actual["Actual"] = numeric_series(
        user_df["Actual"]
    ).to_numpy()

    return (
        new_ghi.reset_index(drop=True),
        new_actual.reset_index(drop=True),
    )


# ============================================================
# DATE / GEOMETRY
# ============================================================

def detect_date(df):

    df = require_df(
        df,
        "Fixed data",
    )

    possible = [
        c for c in df.columns
        if any(
            word in str(c).lower()
            for word in [
                "date",
                "datetime",
                "time stamp",
                "timestamp",
            ]
        )
    ]

    for col in possible:

        values = pd.to_datetime(
            df[col],
            errors="coerce",
        )

        values = values.dropna()

        if not values.empty:
            return values.iloc[0].normalize()

    # Safe fallback
    return pd.Timestamp.today().normalize()


def prepare_geometry(
    actual_df,
    ghi_df,
    latitude,
    tilt_lookup,
):

    n = min(
        len(actual_df),
        len(ghi_df),
    )

    if n <= 0:
        raise ValueError(
            "No data available for geometry."
        )

    date = detect_date(actual_df)

    day = date.dayofyear

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

    month = date.strftime("%B")

    tilt = safe_float(
        tilt_lookup.get(
            month,
            0,
        )
    )

    result = actual_df.iloc[:n].copy()

    result["Date"] = date

    result["Declination Angle ∆"] = declination

    result["Elevation angle a"] = elevation

    result["Tilt Angle b"] = tilt

    result["a+b"] = (
        elevation + tilt
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(
            elevation + tilt
        )
    )

    # Avoid divide-by-zero
    if abs(sin_a) < 1e-8:
        sin_a = 1e-8

    result["Sin(a)"] = sin_a
    result["SIN(a+b)"] = sin_ab

    return result.reset_index(drop=True), {
        "sin_a": sin_a,
        "sin_ab": sin_ab,
        "date": date,
        "tilt": tilt,
    }


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed(
    ghi,
    actual,
    cluster,
    geometry,
):

    n = min(
        len(ghi),
        len(actual),
    )

    sin_a = geometry["sin_a"]
    sin_ab = geometry["sin_ab"]

    # GHI -> POA
    poa = np.column_stack(
        [
            numeric_array(
                ghi[col]
            )[:n]
            * sin_ab
            / sin_a
            for col in GHI_COLS
        ]
    )

    weights = (
        numeric_series(
            cluster["Eff Area(m2)"]
        )
        .iloc[:5]
        .to_numpy(
            dtype=float
        )
    )

    if len(weights) < 5:
        raise ValueError(
            "Five cluster effective areas are required."
        )

    power = (
        poa
        * weights[None, :]
        / 1_000_000
    )

    forecast = power.sum(axis=1)

    return {
        "forecast": forecast,
        "poa": poa,
        "power": power,
        "actual": numeric_array(
            actual["Actual"]
        )[:n],
    }


# ============================================================
# AUTOMATIC ERROR OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_error(
    area,
    ghi,
    actual,
    geometry,
):

    actual_values = numeric_array(
        actual["Actual"]
    )

    if len(actual_values) == 0:
        raise ValueError(
            "Actual data is empty."
        )

    actual_peak = np.max(
        actual_values
    )

    if actual_peak <= 0:
        raise ValueError(
            "Actual data contains no positive peak."
        )

    rows = []

    for error in np.arange(
        0,
        20.01,
        0.1,
    ):

        _, cluster = calculate_cluster_area(
            area,
            error,
        )

        result = calculate_fixed(
            ghi,
            actual,
            cluster,
            geometry,
        )

        forecast_peak = np.max(
            result["forecast"]
        )

        peak_error = abs(
            forecast_peak
            - actual_peak
        )

        rows.append(
            {
                "Error %": round(
                    error,
                    1,
                ),
                "Actual Peak": actual_peak,
                "Calculated Peak": forecast_peak,
                "Peak Error": peak_error,
                "Peak Error %":
                    peak_error
                    / actual_peak
                    * 100,
            }
        )

    table = pd.DataFrame(rows)

    best_idx = table[
        "Peak Error"
    ].idxmin()

    best_error = float(
        table.loc[
            best_idx,
            "Error %",
        ]
    )

    return best_error, table


# ============================================================
# TRACKING INPUT
# ============================================================

def prepare_tracking(
    backend,
    ghi,
    actual,
    cluster,
):

    backend_df = backend.get(
        "C11",
        pd.DataFrame(),
    )

    if backend_df.empty:
        raise ValueError(
            "Backend Cal C11 is empty."
        )

    backend_df = clean_columns(
        backend_df
    )

    block_col = next(
        (
            c for c in backend_df.columns
            if str(c).strip().lower()
            in {
                "block no.",
                "block no",
                "block",
            }
        ),
        None,
    )

    if block_col is None:
        raise ValueError(
            "Block No. column not found "
            "in Backend Cal C11."
        )

    blocks = numeric_array(
        backend_df[block_col]
    )

    n = min(
        len(blocks),
        len(ghi),
        len(actual),
    )

    matrix = np.column_stack(
        [
            numeric_array(
                ghi[col]
            )[:n]
            for col in GHI_COLS
        ]
    )

    actual_values = numeric_array(
        actual["Actual"]
    )[:n]

    weights = (
        numeric_series(
            cluster["Eff Area(m2)"]
        )
        .iloc[:5]
        .to_numpy(
            dtype=float
        )
    )

    if len(weights) < 5:
        raise ValueError(
            "Five cluster weights are required "
            "for Tracking."
        )

    return (
        blocks[:n],
        matrix,
        actual_values,
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
        start
        < maximum
        < end
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
        np.minimum(
            89,
            m1 * (
                blocks - maximum
            ),
        ),
        np.minimum(
            89,
            m2 * (
                blocks - maximum
            ),
        ),
    )

    panel = np.where(
        blocks < maximum,
        np.minimum(
            zenith,
            abs(east),
        ),
        np.where(
            (
                blocks > maximum
            )
            & (
                zenith > west
            ),
            west,
            zenith,
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

    power = (
        dni
        * weights[None, :]
        / 1_000_000
    )

    forecast = power.sum(
        axis=1
    )

    return {
        "forecast": forecast,
        "power": power,
        "zenith": zenith,
        "panel": panel,
        "dni": dni,
    }


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def optimize_tracking(
    blocks,
    ghi,
    actual,
    weights,
):

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    ghi = np.asarray(
        ghi,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    weights = np.asarray(
        weights,
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

    mask = (
        actual > 0
    ) & np.isfinite(actual)

    if not mask.any():
        raise ValueError(
            "No positive Actual values "
            "for Tracking optimization."
        )

    actual_day = actual[mask]

    peak = actual_day.max()

    energy = actual_day.sum()

    def objective(x):

        p = np.rint(x).astype(int)

        result = tracking_model(
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

        prediction = result["forecast"]

        if not np.all(
            np.isfinite(prediction)
        ):
            return 1e9

        pred_day = prediction[mask]

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    - pred_day
                )
            )
            / peak
        )

        peak_error = (
            abs(
                peak
                - pred_day.max()
            )
            / peak
        )

        energy_error = (
            abs(
                energy
                - pred_day.sum()
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
        popsize=10,
        tol=0.001,
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
# COMPLETE AUTOMATIC CALCULATION
# ============================================================

def run_calculation(
    workbook,
    user_input,
    plant,
):

    # --------------------------------------------
    # Prepare workbook inputs
    # --------------------------------------------

    area = prepare_area(
        workbook["area"]
    )

    ghi_raw = prepare_ghi(
        workbook["result"]
    )

    actual_raw = prepare_actual(
        workbook["fixed"]
    )

    latitude = prepare_latitude(
        workbook["forecast_config"]
    )

    tilt_lookup = prepare_tilt(
        workbook["tilt"]
    )

    # --------------------------------------------
    # Apply edited input
    # --------------------------------------------

    ghi, actual = apply_input(
        user_input,
        ghi_raw,
        actual_raw,
    )

    # --------------------------------------------
    # Geometry
    # --------------------------------------------

    geometry_df, geometry = (
        prepare_geometry(
            actual,
            ghi,
            latitude,
            tilt_lookup,
        )
    )

    # --------------------------------------------
    # Automatic Error %
    # --------------------------------------------

    best_error, error_table = (
        optimize_error(
            area,
            ghi,
            actual,
            geometry,
        )
    )

    # --------------------------------------------
    # Final effective area
    # --------------------------------------------

    final_area, cluster = (
        calculate_cluster_area(
            area,
            best_error,
        )
    )

    # --------------------------------------------
    # Fixed
    # --------------------------------------------

    fixed = calculate_fixed(
        ghi,
        actual,
        cluster,
        geometry,
    )

    # --------------------------------------------
    # Tracking
    # --------------------------------------------

    tracking_params = None

    if plant == "Tracking":

        blocks, ghi_matrix, actual_values, weights = (
            prepare_tracking(
                workbook["backend"],
                ghi,
                actual,
                cluster,
            )
        )

        tracking_params = optimize_tracking(
            tuple(blocks.tolist()),
            tuple(
                tuple(row)
                for row in ghi_matrix
            ),
            tuple(actual_values.tolist()),
            tuple(weights.tolist()),
        )

    return {
        "area": area,
        "final_area": final_area,
        "cluster": cluster,
        "ghi": ghi,
        "actual": actual,
        "geometry": geometry_df,
        "geometry_meta": geometry,
        "best_error": best_error,
        "error_table": error_table,
        "fixed": fixed,
        "tracking_params": tracking_params,
        "latitude": latitude,
    }


# ============================================================
# FINAL FORECAST
# ============================================================

def calculate_final_forecast(
    data,
    error,
    plant,
    tracking_params=None,
):

    _, cluster = calculate_cluster_area(
        data["area"],
        error,
    )

    fixed = calculate_fixed(
        data["ghi"],
        data["actual"],
        cluster,
        data["geometry_meta"],
    )

    if plant == "Fixed":

        return {
            "forecast": fixed["forecast"],
            "actual": fixed["actual"],
            "cluster": cluster,
            "fixed": fixed,
        }

    # Tracking
    blocks, ghi_matrix, actual, weights = (
        prepare_tracking(
            data["_backend"],
            data["ghi"],
            data["actual"],
            cluster,
        )
    )

    if tracking_params is None:
        raise ValueError(
            "Tracking parameters are unavailable."
        )

    result = tracking_model(
        int(tracking_params["DHI"]),
        int(tracking_params["GHI Starting Block"]),
        int(tracking_params["GHI Ending Block"]),
        int(tracking_params["GHI Max Block"]),
        int(tracking_params["Tracking East Limit"]),
        int(tracking_params["Tracking West Limit"]),
        blocks,
        ghi_matrix,
        weights,
    )

    if result is None:
        raise ValueError(
            "Invalid Tracking parameters. "
            "Ensure Starting < Max < Ending."
        )

    return {
        "forecast": result["forecast"],
        "actual": actual,
        "cluster": cluster,
        "tracking": result,
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
# FILE
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

current_hash = file_hash(
    uploaded
)

if (
    st.session_state.file_hash
    != current_hash
):

    st.session_state.file_hash = (
        current_hash
    )

    st.session_state.input_df = None
    st.session_state.plant = "Fixed"

    reset_results()


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    workbook = load_excel(
        uploaded.getvalue()
    )

except Exception as e:

    st.error(
        f"Workbook loading failed: {e}"
    )

    st.stop()


# ============================================================
# CREATE INPUT
# ============================================================

if st.session_state.input_df is None:

    try:

        ghi_temp = prepare_ghi(
            workbook["result"]
        )

        actual_temp = prepare_actual(
            workbook["fixed"]
        )

        st.session_state.input_df = (
            create_input(
                ghi_temp,
                actual_temp,
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
    height=270,
    num_rows="fixed",
    hide_index=True,
    key=f"input_editor_{current_hash}",
    column_config={
        col: st.column_config.NumberColumn(
            col,
            format="%.3f",
        )
        for col in GHI_COLS + ["Actual"]
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
    default=st.session_state.plant,
    selection_mode="single",
    width="stretch",
    label_visibility="collapsed",
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
# RUN AUTOMATIC CALCULATION
# ============================================================

if run:

    try:

        with st.spinner(
            "Running automatic calculation..."
        ):

            user_input = input_df.copy()

            result = run_calculation(
                workbook,
                user_input,
                plant,
            )

            # Backend is needed later for
            # lightweight Tracking recalculation
            result["_backend"] = (
                workbook["backend"]
            )

            st.session_state.result = result
            st.session_state.input_df = (
                user_input
            )
            st.session_state.calculated = True

        st.success(
            "Automatic calculation completed."
        )

    except Exception as e:

        reset_results()

        st.error(
            f"Calculation failed: {e}"
        )


# ============================================================
# STOP UNTIL CALCULATED
# ============================================================

if not st.session_state.calculated:

    st.caption(
        "Edit GHI / Actual values, select Fixed "
        "or Tracking, then run the calculation."
    )

    st.stop()


data = st.session_state.result


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

tracking_params = None

if plant == "Tracking":

    p = data["tracking_params"]

    if p is None:

        st.error(
            "Tracking parameters are unavailable."
        )

        st.stop()

    st.markdown(
        '<div class="section">🎯 Tracking Parameters</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(
                p["DHI"]
            ),
            step=1,
        )

        start = st.number_input(
            "GHI Starting Block",
            min_value=0,
            max_value=95,
            value=int(
                p["GHI Starting Block"]
            ),
            step=1,
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
        )

        maximum = st.number_input(
            "GHI Max Block",
            min_value=0,
            max_value=95,
            value=int(
                p["GHI Max Block"]
            ),
            step=1,
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
        )

        west = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            value=int(
                p["Tracking West Limit"]
            ),
            step=1,
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
# FINAL LIGHTWEIGHT CALCULATION
# ============================================================

try:

    final = calculate_final_forecast(
        data,
        error,
        plant,
        tracking_params,
    )

    actual = final["actual"]
    forecast = final["forecast"]

except Exception as e:

    st.error(
        f"Forecast calculation failed: {e}"
    )

    st.stop()


# ============================================================
# RESULTS
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

title = (
    f"{plant} Plant | Actual vs Forecast"
)

fig = make_graph(
    actual,
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


# ============================================================
# OPTIMIZATION DETAILS
# ============================================================

with st.expander(
    "🔍 Automatic Optimization Details"
):

    st.write(
        f"**Automatically selected Error:** "
        f"{data['best_error']:.1f}%"
    )

    st.write(
        f"**Latitude:** "
        f"{data['latitude']:.4f}°"
    )

    st.write(
        f"**Tilt:** "
        f"{data['geometry_meta']['tilt']:.2f}°"
    )

    st.write(
        f"**Calculation date:** "
        f"{data['geometry_meta']['date'].date()}"
    )

    st.dataframe(
        data["cluster"],
        width="stretch",
        hide_index=True,
    )


# ============================================================
# ERROR OPTIMIZATION TABLE
# ============================================================

with st.expander(
    "📋 Error Optimization Table"
):

    st.dataframe(
        data["error_table"],
        width="stretch",
        hide_index=True,
    )
