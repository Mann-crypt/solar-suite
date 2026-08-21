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


TRACKING_BOUNDS = [
    (0, 10),     # DHI
    (10, 30),    # GHI Starting Block
    (65, 80),    # GHI Ending Block
    (47, 53),    # GHI Max Block
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

for key, value in {
    "plant": "Fixed",
    "file_hash": None,
    "input_df": None,
    "result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_results():
    st.session_state.result = None


# ============================================================
# BASIC HELPERS
# ============================================================

def norm(x):
    """Normalize Excel text for robust matching."""
    return (
        str(x)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("*", "")
        .replace("_", " ")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("%", " percent ")
        .replace("  ", " ")
        .strip()
    )


def numeric(x):
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").fillna(0)

    return pd.to_numeric(
        pd.Series(x),
        errors="coerce",
    ).fillna(0)


def arr(x):
    return numeric(x).to_numpy(dtype=float)


def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def clean_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]
    return df


def file_hash(uploaded):
    return hashlib.sha256(
        uploaded.getvalue()
    ).hexdigest()


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    aliases,
    required=True,
):
    """
    Find an Excel column using normalized aliases.
    First tries exact match, then partial match.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected DataFrame, got {type(df).__name__}"
        )

    columns = list(df.columns)

    normalized = {
        norm(c): c
        for c in columns
    }

    aliases = [
        norm(a)
        for a in aliases
    ]

    # Exact
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]

    # Partial
    for col in columns:
        nc = norm(col)

        for alias in aliases:
            if alias in nc or nc in alias:
                return col

    if required:
        raise ValueError(
            "Required column not found.\n\n"
            f"Looking for: {aliases}\n\n"
            f"Available columns:\n"
            f"{list(columns)}"
        )

    return None


# ============================================================
# FIND HEADER ROW
# ============================================================

def find_header_row(
    data,
    aliases,
    max_rows=30,
):
    """
    Automatically locate the Excel header row.
    """

    preview = pd.read_excel(
        io.BytesIO(data),
        header=None,
        nrows=max_rows,
    )

    aliases = [norm(a) for a in aliases]

    best_row = None
    best_score = -1

    for r in range(len(preview)):

        values = [
            norm(v)
            for v in preview.iloc[r].tolist()
            if pd.notna(v)
        ]

        score = 0

        for alias in aliases:
            if any(
                alias == v
                or alias in v
                or v in alias
                for v in values
            ):
                score += 1

        if score > best_score:
            best_score = score
            best_row = r

    if best_score <= 0:
        return None

    return best_row


def read_sheet_auto(
    data,
    sheet,
    aliases=None,
):
    """
    Read sheet using automatically detected header.
    """

    if aliases is None:
        aliases = []

    header = find_header_row(
        data,
        aliases,
    )

    if header is None:
        # fallback
        return clean_columns(
            pd.read_excel(
                io.BytesIO(data),
                sheet_name=sheet,
            )
        )

    return clean_columns(
        pd.read_excel(
            io.BytesIO(data),
            sheet_name=sheet,
            header=header,
        )
    )


# ============================================================
# LOAD WORKBOOK
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=3,
)
def load_workbook(data):

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    area = read_sheet_auto(
        data,
        "Area & Efficiency",
        [
            "Clusters",
            "Standard PV Efficiency (%)",
            "No of Module",
            "Area of 1 Module (m2)",
        ],
    )

    # --------------------------------------------------------
    # Result / GHI
    # --------------------------------------------------------

    ghi = read_sheet_auto(
        data,
        "Result",
        GHI_COLS,
    )

    # --------------------------------------------------------
    # Forecast Config
    # --------------------------------------------------------

    forecast_config = read_sheet_auto(
        data,
        "Forecast Config",
        ["Lat"],
    )

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    tilt = read_sheet_auto(
        data,
        "Config Tilt Angle",
        ["Month", "Fixed"],
    )

    # --------------------------------------------------------
    # Fixed
    # --------------------------------------------------------

    fixed = read_sheet_auto(
        data,
        "Fixed-C11",
        ["Actual"],
    )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    tracking = read_sheet_auto(
        data,
        "Tracking",
        [
            "Block No.",
        ],
    )

    # --------------------------------------------------------
    # Backend
    # --------------------------------------------------------

    backend = {}

    for cluster in CLUSTERS:

        sheet = f"Backend Cal {cluster}"

        try:
            backend[cluster] = read_sheet_auto(
                data,
                sheet,
                ["Block No."],
            )
        except Exception:
            backend[cluster] = pd.DataFrame()

    return {
        "area": area,
        "ghi": ghi,
        "forecast_config": forecast_config,
        "tilt": tilt,
        "fixed": fixed,
        "tracking": tracking,
        "backend": backend,
    }


# ============================================================
# AREA PREPARATION
# ============================================================

def prepare_area(df):

    df = clean_columns(df)

    cluster_col = find_column(
        df,
        [
            "Clusters",
            "Cluster",
        ],
    )

    efficiency_col = find_column(
        df,
        [
            "Standard PV Efficiency (%)",
            "Standard PV Efficiency",
            "PV Efficiency (%)",
            "Efficiency (%)",
            "Efficiency",
        ],
    )

    module_col = find_column(
        df,
        [
            "No of Module",
            "No. of Module",
            "Number of Module",
            "No Modules",
            "Modules",
        ],
    )

    area_col = find_column(
        df,
        [
            "Area of 1 Module (m2)",
            "Area of 1 Module",
            "Module Area (m2)",
            "Module Area",
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
        df[area_col]
    )

    # Remove empty / junk rows
    out = out[
        out["Clusters"].isin(CLUSTERS)
    ].copy()

    if out.empty:
        raise ValueError(
            "No C11-C15 cluster rows found in "
            "Area & Efficiency."
        )

    out["Total area (m2)"] = (
        out["No of Module"]
        * out["Area of 1 Module (m2)"]
    )

    return out.reset_index(drop=True)


# ============================================================
# GHI
# ============================================================

def prepare_ghi(df):

    df = clean_columns(df)

    out = pd.DataFrame()

    for cluster in CLUSTERS:

        expected = f"GHI {cluster}"

        col = find_column(
            df,
            [
                expected,
                f"GHI_{cluster}",
                f"GHI-{cluster}",
                f"{cluster} GHI",
            ],
        )

        out[expected] = numeric(
            df[col]
        )

    return out.reset_index(drop=True)


# ============================================================
# ACTUAL
# ============================================================

def prepare_actual(df):

    df = clean_columns(df)

    col = find_column(
        df,
        [
            "Actual",
            "Actual Power",
            "Actual Power (MW)",
            "Actual Power MW",
        ],
    )

    return pd.DataFrame(
        {
            "Actual": numeric(
                df[col]
            )
        }
    ).reset_index(drop=True)


# ============================================================
# LATITUDE
# ============================================================

def prepare_latitude(df):

    df = clean_columns(df)

    col = find_column(
        df,
        [
            "Lat",
            "Latitude",
        ],
    )

    values = numeric(
        df[col]
    )

    values = values[
        np.isfinite(values)
    ]

    if values.empty:
        raise ValueError(
            "Valid latitude value not found."
        )

    return float(values.iloc[0])


# ============================================================
# TILT
# ============================================================

def prepare_tilt(df):

    df = clean_columns(df)

    month_col = find_column(
        df,
        [
            "Month",
            "Months",
        ],
        required=False,
    )

    fixed_col = find_column(
        df,
        [
            "Fixed",
            "Fixed Tilt",
            "Fixed Tilt Angle",
        ],
    )

    # --------------------------------------------------------
    # If Month column exists
    # --------------------------------------------------------

    if month_col is not None:

        months = {}

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

                months[
                    month.lower()
                ] = safe_float(
                    row[fixed_col]
                )

        if months:
            return months

    # --------------------------------------------------------
    # Month column may have been read incorrectly.
    # Try first column containing month names.
    # --------------------------------------------------------

    month_names = {
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

    for col in df.columns:

        vals = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if vals.isin(month_names).sum() >= 2:

            months = {}

            for i in df.index:

                month = str(
                    df.loc[i, col]
                ).strip().lower()

                if month in month_names:

                    months[month] = safe_float(
                        df.loc[i, fixed_col]
                    )

            if months:
                return months

    raise ValueError(
        "Month column missing in Config Tilt Angle "
        "and month names could not be detected."
    )


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
            "No GHI / Actual data found."
        )

    out = ghi.iloc[:n].copy()

    out["Actual"] = (
        actual["Actual"]
        .iloc[:n]
        .to_numpy()
    )

    return out.reset_index(drop=True)


def apply_input(user_df):

    user_df = user_df.copy()

    for col in GHI_COLS + ["Actual"]:

        if col not in user_df.columns:
            raise ValueError(
                f"Missing input column: {col}"
            )

        user_df[col] = numeric(
            user_df[col]
        )

    return (
        user_df[GHI_COLS]
        .copy()
        .reset_index(drop=True),
        pd.DataFrame(
            {
                "Actual":
                    user_df["Actual"]
                    .to_numpy()
            }
        ),
    )


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def solar_geometry(
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

    g = ghi.iloc[:n].copy()
    out = actual.iloc[:n].copy()

    today = pd.Timestamp.today()

    day_number = today.dayofyear

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

    elevation = (
        90
        - latitude
        + declination
    )

    month = today.strftime(
        "%B"
    ).lower()

    tilt = safe_float(
        tilt_lookup.get(
            month,
            0,
        )
    )

    out["Declination Angle ∆"] = declination
    out["Elevation angle a"] = elevation
    out["Tilt Angle b"] = tilt

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(
            elevation + tilt
        )
    )

    out["Sin(a)"] = sin_a
    out["SIN(a+b)"] = sin_ab

    denominator = (
        sin_a
        if abs(sin_a) > 1e-9
        else 1e-9
    )

    # --------------------------------------------------------
    # C11-C15
    # --------------------------------------------------------

    for i, cluster in enumerate(CLUSTERS):

        poa = POA_COLS[i]

        out[
            f"GHI*sin(a)-{cluster}"
        ] = (
            g[f"GHI {cluster}"]
            * sin_a
        )

        out[
            f"GHI*sin(a+b)-{cluster}"
        ] = (
            g[f"GHI {cluster}"]
            * sin_ab
        )

        out[poa] = (
            out[
                f"GHI*sin(a+b)-{cluster}"
            ]
            / denominator
        )

    return out.reset_index(drop=True)


# ============================================================
# EFFECTIVE AREA
# ============================================================

def calculate_effective_area(
    area,
    error,
):

    a = area.copy()

    a["Error %"] = float(error)

    a["Net Efficiency (%)"] = (
        a["Standard PV Efficiency (%)"]
        - float(error)
    )

    a["Eff Area"] = (
        a["Net Efficiency (%)"]
        * a["Total area (m2)"]
        / 100
    )

    cluster = (
        a.groupby("Clusters")["Eff Area"]
        .sum()
        .reindex(CLUSTERS)
        .fillna(0)
        .reset_index()
    )

    cluster.columns = [
        "Clusters",
        "Eff Area(m2)",
    ]

    return a, cluster


# ============================================================
# FIXED POWER
# ============================================================

def calculate_fixed_power(
    geometry,
    cluster,
):

    result = geometry.copy()

    weights = (
        cluster.set_index(
            "Clusters"
        )["Eff Area(m2)"]
        .reindex(CLUSTERS)
        .fillna(0)
    )

    for i, cluster_name in enumerate(
        CLUSTERS
    ):

        result[
            POWER_COLS[i]
        ] = (
            numeric(
                result[POA_COLS[i]]
            )
            * safe_float(
                weights.loc[
                    cluster_name
                ]
            )
            / 1_000_000
        )

    result[TOTAL_POWER] = (
        result[POWER_COLS]
        .sum(axis=1)
    )

    return result


# ============================================================
# FIXED ERROR OPTIMIZATION
# ============================================================

def optimize_fixed(
    area,
    geometry,
):

    actual = arr(
        geometry["Actual"]
    )

    if len(actual) == 0:
        raise ValueError(
            "Actual data is empty."
        )

    peak = np.max(actual)

    if peak <= 0:
        raise ValueError(
            "Actual data contains no positive peak."
        )

    records = []

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        _, cluster = (
            calculate_effective_area(
                area,
                error,
            )
        )

        result = calculate_fixed_power(
            geometry,
            cluster,
        )

        forecast = arr(
            result[TOTAL_POWER]
        )

        forecast_peak = (
            np.max(forecast)
            if len(forecast)
            else 0
        )

        peak_error = abs(
            forecast_peak - peak
        )

        records.append(
            [
                round(error, 1),
                forecast_peak,
                peak,
                peak_error,
                peak_error / peak * 100,
            ]
        )

    table = pd.DataFrame(
        records,
        columns=[
            "Error %",
            "Calculated Peak",
            "Actual Peak",
            "Peak Error",
            "Peak Error %",
        ],
    )

    best = table.loc[
        table["Peak Error"].idxmin()
    ]

    return (
        float(best["Error %"]),
        table,
    )


# ============================================================
# TRACKING
# ============================================================

def get_tracking_data(
    workbook,
    ghi,
    geometry,
    cluster,
):

    backend = workbook[
        "backend"
    ]["C11"]

    if backend.empty:
        raise ValueError(
            "Backend Cal C11 is empty."
        )

    block_col = find_column(
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
        len(geometry),
    )

    weights = (
        cluster.set_index(
            "Clusters"
        )["Eff Area(m2)"]
        .reindex(CLUSTERS)
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    matrix = np.column_stack(
        [
            arr(
                ghi[
                    f"GHI {c}"
                ]
            )[:n]
            for c in CLUSTERS
        ]
    )

    actual = arr(
        geometry["Actual"]
    )[:n]

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
            m1
            * (
                blocks
                - maximum
            ),
        ),
        np.minimum(
            89,
            m2
            * (
                blocks
                - maximum
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
                (blocks > maximum)
                & (zenith > west)
            ),
            west,
            zenith,
        ),
    )

    cos_panel = np.clip(
        np.cos(
            np.radians(
                panel
            )
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
        ghi
        - dhi_part
    ) / cos_panel[:, None]

    power = (
        dni
        * weights[None, :]
        / 1_000_000
    )

    forecast = (
        power.sum(axis=1)
    )

    forecast[
        ~np.isfinite(forecast)
    ] = 0

    return (
        forecast,
        power,
        zenith,
        panel,
        dni,
    )


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def optimize_tracking_cached(
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

    mask = (
        np.isfinite(actual)
        & (actual != 0)
    )

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values."
        )

    actual_day = actual[mask]

    peak = np.max(
        actual_day
    )

    energy = np.sum(
        actual_day
    )

    def objective(x):

        p = np.rint(x).astype(int)

        result = tracking_calc(
            *p,
            blocks,
            ghi,
            weights,
        )

        if result is None:
            return 1e9

        prediction = result[0]

        pred_day = prediction[
            mask
        ]

        if not np.all(
            np.isfinite(pred_day)
        ):
            return 1e9

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
                - np.max(
                    pred_day
                )
            )
            / peak
        )

        energy_error = (
            abs(
                energy
                - np.sum(
                    pred_day
                )
            )
            / max(
                energy,
                1e-9,
            )
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(
        objective,
        TRACKING_BOUNDS,
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
# COMPLETE AUTOMATIC CALCULATION
# ============================================================

def run_calculation(
    workbook,
    user_input,
    plant,
):

    # --------------------------------------------------------
    # Prepare workbook data
    # --------------------------------------------------------

    area = prepare_area(
        workbook["area"]
    )

    ghi_master = prepare_ghi(
        workbook["ghi"]
    )

    actual_master = prepare_actual(
        workbook["fixed"]
    )

    latitude = prepare_latitude(
        workbook["forecast_config"]
    )

    tilt = prepare_tilt(
        workbook["tilt"]
    )

    # --------------------------------------------------------
    # User input
    # --------------------------------------------------------

    ghi, actual = apply_input(
        user_input
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    geometry = solar_geometry(
        ghi,
        actual,
        latitude,
        tilt,
    )

    # --------------------------------------------------------
    # Automatic Error %
    # --------------------------------------------------------

    best_error, error_table = (
        optimize_fixed(
            area,
            geometry,
        )
    )

    # --------------------------------------------------------
    # Final efficiency
    # --------------------------------------------------------

    final_area, final_cluster = (
        calculate_effective_area(
            area,
            best_error,
        )
    )

    # --------------------------------------------------------
    # Fixed
    # --------------------------------------------------------

    fixed_result = (
        calculate_fixed_power(
            geometry,
            final_cluster,
        )
    )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    tracking_params = None

    if plant == "Tracking":

        (
            blocks,
            ghi_matrix,
            actual_array,
            weights,
        ) = get_tracking_data(
            workbook,
            ghi,
            geometry,
            final_cluster,
        )

        tracking_params = (
            optimize_tracking_cached(
                tuple(
                    blocks.tolist()
                ),
                tuple(
                    tuple(row)
                    for row in ghi_matrix
                ),
                tuple(
                    actual_array.tolist()
                ),
                tuple(
                    weights.tolist()
                ),
            )
        )

    return {
        "area": area,
        "geometry": geometry,
        "ghi": ghi,
        "final_area": final_area,
        "final_cluster": final_cluster,
        "fixed_result": fixed_result,
        "best_error": best_error,
        "error_table": error_table,
        "tracking_params": tracking_params,
    }


# ============================================================
# FINAL FORECAST
# ============================================================

def calculate_final_forecast(
    workbook,
    data,
    plant,
    error,
    tracking_params=None,
):

    _, cluster = (
        calculate_effective_area(
            data["area"],
            error,
        )
    )

    fixed_result = (
        calculate_fixed_power(
            data["geometry"],
            cluster,
        )
    )

    if plant == "Fixed":

        return (
            arr(
                data["geometry"]["Actual"]
            ),
            arr(
                fixed_result[
                    TOTAL_POWER
                ]
            ),
        )

    blocks, ghi_matrix, actual, weights = (
        get_tracking_data(
            workbook,
            data["ghi"],
            data["geometry"],
            cluster,
        )
    )

    p = tracking_params

    if p is None:
        raise ValueError(
            "Tracking parameters unavailable."
        )

    result = tracking_calc(
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

    if result is None:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    return (
        actual,
        result[0],
    )


# ============================================================
# GRAPH
# ============================================================

def graph(
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
    st.session_state.result = None
    st.session_state.plant = "Fixed"

    for key in [
        "solar_input_editor",
        "plant_selector",
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
        f"Workbook loading failed: {e}"
    )

    st.stop()


# ============================================================
# PREPARE BASIC INPUT
# ============================================================

try:

    ghi = prepare_ghi(
        workbook["ghi"]
    )

    actual = prepare_actual(
        workbook["fixed"]
    )

    if st.session_state.input_df is None:

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
# INPUT TABLE
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
    key="solar_input_editor",
    column_config={
        c: st.column_config.NumberColumn(
            c,
            format="%.3f",
        )
        for c in (
            GHI_COLS
            + ["Actual"]
        )
    },
)


# ============================================================
# PLANT TYPE
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
    key="plant_selector",
)

plant = plant or "Fixed"

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
# HEAVY CALCULATION ONLY ON RUN
# ============================================================

if run:

    try:

        with st.spinner(
            "Running automatic calculation..."
        ):

            edited = input_df.copy()

            result = run_calculation(
                workbook,
                edited,
                plant,
            )

            st.session_state.input_df = (
                edited
            )

            st.session_state.result = (
                result
            )

        st.success(
            "Automatic calculation completed successfully."
        )

    except Exception as e:

        st.session_state.result = None

        st.error(
            f"Calculation failed: {e}"
        )


# ============================================================
# WAIT FOR RUN
# ============================================================

if st.session_state.result is None:

    st.caption(
        "Edit GHI / Actual values, select Fixed or "
        "Tracking, then run the automatic calculation."
    )

    st.stop()


# ============================================================
# RESULT DATA
# ============================================================

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
    key=f"error_{current_hash}",
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

tracking_values = None

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
            int(
                p[
                    "GHI Starting Block"
                ]
            ),
            1,
            key=f"start_{current_hash}",
        )

    with c2:

        end = st.number_input(
            "GHI Ending Block",
            1,
            96,
            int(
                p[
                    "GHI Ending Block"
                ]
            ),
            1,
            key=f"end_{current_hash}",
        )

        maximum = st.number_input(
            "GHI Max Block",
            0,
            95,
            int(
                p[
                    "GHI Max Block"
                ]
            ),
            1,
            key=f"max_{current_hash}",
        )

    with c3:

        east = st.number_input(
            "Tracking East Limit",
            0,
            90,
            int(
                p[
                    "Tracking East Limit"
                ]
            ),
            1,
            key=f"east_{current_hash}",
        )

        west = st.number_input(
            "Tracking West Limit",
            0,
            90,
            int(
                p[
                    "Tracking West Limit"
                ]
            ),
            1,
            key=f"west_{current_hash}",
        )

    tracking_values = {
        "DHI": int(dhi),
        "GHI Starting Block": int(start),
        "GHI Ending Block": int(end),
        "GHI Max Block": int(maximum),
        "Tracking East Limit": int(east),
        "Tracking West Limit": int(west),
    }


# ============================================================
# FINAL LIGHTWEIGHT CALCULATION
# ============================================================

try:

    if plant == "Tracking":

        if not (
            tracking_values[
                "GHI Starting Block"
            ]
            <
            tracking_values[
                "GHI Max Block"
            ]
            <
            tracking_values[
                "GHI Ending Block"
            ]
        ):

            st.error(
                "Tracking parameters must satisfy: "
                "Starting Block < Max Block < Ending Block."
            )

            st.stop()

    actual_values, forecast = (
        calculate_final_forecast(
            workbook,
            data,
            plant,
            error,
            tracking_values
            if plant == "Tracking"
            else None,
        )
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
    float(
        np.max(actual_values)
    )
    if len(actual_values)
    else 0
)

forecast_peak = (
    float(
        np.max(forecast)
    )
    if len(forecast)
    else 0
)

peak_error = (
    abs(
        forecast_peak
        - actual_peak
    )
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
                {peak_error:.2f}%
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

fig = graph(
    actual_values,
    forecast,
    (
        "Tracking Plant | Actual vs Forecast"
        if plant == "Tracking"
        else
        "Fixed Plant | Actual vs Forecast"
    ),
)

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "displayModeBar": False,
        "responsive": True,
    },
)
