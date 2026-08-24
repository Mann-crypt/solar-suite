# ============================================================
# LOSS CORRECTION MODEL
# Fixed / Tracking
#
# Stable + optimized Streamlit architecture
#
# Calculation formulas preserved.
# Heavy work is cached.
# Tracking parameter changes update automatically.
# No recalculation button is required for parameter changes.
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
    page_title="Loss Correction Model",
    page_icon="☀️",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_OPT_ITER = 40
OPT_POPSIZE = 10

PARAM_BOUNDS = [
    (0, 10),      # DHI
    (0, 30),      # Starting Block
    (65, 80),     # Ending Block
    (44, 60),     # Max Block
    (0, 70),      # East Limit
    (0, 70),      # West Limit
]

FIXED_PLANT = "🏗️ Fixed"
TRACKING_PLANT = "🔄 Tracking"

CLUSTER_GHI_COLUMNS = [
    "CL1-GHI",
    "CL2-GHI",
    "CL3-GHI",
    "CL4-GHI",
    "CL5-GHI",
]

CLUSTER_WEIGHT_COLUMNS = [
    "CL-1",
    "CL-2",
    "CL-3",
    "CL-4",
    "CL-5",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 2rem;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #8b949e;
        margin-bottom: 22px;
    }

    .section {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 18px 0 8px;
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
        min-height: 44px;
    }

    div.stButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "plant_type": FIXED_PLANT,
    "uploaded_signature": None,

    "input_df": None,
    "input_editor_version": 0,

    "model_started": False,

    "tracking_params": None,

    "fixed_loss": None,
    "tracking_loss": None,

    "cluster_fixed_loss": None,
    "cluster_tracking_loss": None,

    "last_plant_type": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GENERIC HELPERS
# ============================================================

def numeric(values):

    if isinstance(values, pd.Series):

        return pd.to_numeric(
            values,
            errors="coerce",
        ).fillna(0)

    return pd.Series(
        pd.to_numeric(
            np.asarray(values),
            errors="coerce",
        )
    ).fillna(0)


def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return float(default)


def safe_array(values):

    if isinstance(values, pd.Series):

        values = pd.to_numeric(
            values,
            errors="coerce",
        ).fillna(0)

    else:

        values = pd.Series(
            pd.to_numeric(
                values,
                errors="coerce",
            )
        ).fillna(0)

    return np.asarray(
        values,
        dtype=float,
    )


def validate_columns(
    df,
    required,
    name="Data",
):

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} is missing required column(s): "
            f"{', '.join(missing)}"
        )


def clean_data_rows(df):

    result = df.copy()

    if "Date" in result.columns:

        null_rows = result[
            result["Date"].isna()
        ].index

        if len(null_rows):

            position = result.index.get_loc(
                null_rows[0]
            )

            result = result.iloc[:position]

    return result.reset_index(drop=True)


def workbook_signature(file_bytes):

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ============================================================
# RESET
# ============================================================

def reset_calculation_state():

    st.session_state.tracking_params = None

    st.session_state.fixed_loss = None
    st.session_state.tracking_loss = None

    st.session_state.cluster_fixed_loss = None
    st.session_state.cluster_tracking_loss = None


def reset_workbook():

    st.session_state.input_df = None

    st.session_state.model_started = False

    st.session_state.tracking_params = None

    st.session_state.fixed_loss = None
    st.session_state.tracking_loss = None

    st.session_state.cluster_fixed_loss = None
    st.session_state.cluster_tracking_loss = None

    st.session_state.input_editor_version += 1


# ============================================================
# WORKBOOK
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def get_sheet_names(file_bytes):

    excel = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    return tuple(
        excel.sheet_names
    )


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_area_efficiency(
    file_bytes,
    cluster,
):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(8) if cluster else None,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        [
            "Module Type",
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        "Area & Efficiency",
    )

    null_rows = df[
        df["Module Type"].isna()
    ].index

    if len(null_rows):

        position = df.index.get_loc(
            null_rows[0]
        )

        df = df.iloc[:position]

    df = df.dropna(
        subset=[
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        how="all",
    )

    df[
        "Standard PV Efficiency (%)"
    ] = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
        errors="coerce",
    ).fillna(0)

    df["Total area(m2)"] = pd.to_numeric(
        df["Total area(m2)"],
        errors="coerce",
    ).fillna(0)

    return df.reset_index(
        drop=True
    )


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_cluster_weights(
    file_bytes,
):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[
            12,
            13,
            14,
            15,
            16,
        ],
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        CLUSTER_WEIGHT_COLUMNS,
        "Cluster Weights",
    )

    weights = {}

    for column in CLUSTER_WEIGHT_COLUMNS:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        weights[column] = (
            float(values.iloc[0])
            if len(values)
            else 0.0
        )

    return weights


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_latitude(
    file_bytes,
):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        ["Lat"],
        "Forecast Config",
    )

    values = pd.to_numeric(
        df["Lat"],
        errors="coerce",
    ).dropna()

    if values.empty:

        raise ValueError(
            "No valid latitude found in Forecast Config."
        )

    return float(
        values.iloc[0]
    )


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_tilt_lookup(
    file_bytes,
):

    try:

        df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name="Config Tilt Angle",
            header=7,
        )

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        if "Fixed" not in df.columns:
            return {}

        null_rows = df[
            df["Fixed"].isna()
        ].index

        if len(null_rows):

            position = df.index.get_loc(
                null_rows[0]
            )

            df = df.iloc[:position]

        df = df.dropna(
            axis=1,
            how="all",
        )

        df = df.rename(
            columns={
                "Unnamed: 2": "Month_Num",
                "Unnamed: 3": "Month",
            }
        )

        if "Month" not in df.columns:
            return {}

        df["Fixed"] = pd.to_numeric(
            df["Fixed"],
            errors="coerce",
        )

        df["Month"] = (
            df["Month"]
            .astype(str)
            .str.strip()
        )

        return (
            df.dropna(
                subset=["Month"]
            )
            .set_index("Month")["Fixed"]
            .dropna()
            .to_dict()
        )

    except Exception:

        return {}


# ============================================================
# INPUT DATA
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def load_input_data(
    file_bytes,
    cluster,
):

    sheet = (
        "Fixed-CL1"
        if cluster
        else "Fixed"
    )

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet,
        header=1,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    df = clean_data_rows(df)

    validate_columns(
        df,
        ["Actual"],
        "Forecast Sheet",
    )

    if cluster:

        try:

            result = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name="Result",
                usecols=range(6),
            )

            result = result.fillna(0)

            for i, column in enumerate(
                CLUSTER_GHI_COLUMNS
            ):

                if column in df.columns:
                    continue

                if i >= len(result.columns):
                    continue

                values = pd.to_numeric(
                    result.iloc[
                        :len(df),
                        i,
                    ],
                    errors="coerce",
                ).fillna(0).to_numpy()

                if len(values) < len(df):

                    values = np.pad(
                        values,
                        (
                            0,
                            len(df) - len(values),
                        ),
                    )

                df[column] = values

        except Exception:
            pass

        validate_columns(
            df,
            CLUSTER_GHI_COLUMNS,
            "Cluster Forecast",
        )

        for column in CLUSTER_GHI_COLUMNS:

            df[column] = numeric(
                df[column]
            ).to_numpy()

    else:

        validate_columns(
            df,
            ["GHI_Forecast"],
            "Fixed Forecast",
        )

        df["GHI_Forecast"] = numeric(
            df["GHI_Forecast"]
        ).to_numpy()

    df["Actual"] = numeric(
        df["Actual"]
    ).to_numpy()

    return df


# ============================================================
# SOLAR ANGLES
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=8,
)
def prepare_solar_angles_cached(
    input_df,
    lat,
    tilt_items,
    tracking,
):

    result = input_df.copy()

    if "Date" in result.columns:

        dates = pd.to_datetime(
            result["Date"],
            errors="coerce",
        )

        if dates.notna().any():

            fallback = dates.dropna().iloc[0]

            dates = dates.fillna(
                fallback
            )

        else:

            dates = pd.Series(
                pd.Timestamp.today(),
                index=result.index,
            )

    else:

        dates = pd.Series(
            pd.Timestamp.today(),
            index=result.index,
        )

    dates = pd.to_datetime(
        dates
    ).dt.normalize()

    result["Date"] = dates

    first_date = pd.to_datetime(
        dates.dt.year.astype(str)
        + "-01-01"
    )

    day_number = (
        dates - first_date
    ).dt.days + 1

    result["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + day_number)
                / 365
            )
        )
    )

    result["Elevation angle a"] = (
        90
        - float(lat)
        + result[
            "Declination Angle ∆"
        ]
    )

    tilt_lookup = dict(
        tilt_items
    )

    if tracking:

        result["Tilt Angle b"] = 0.0

    elif tilt_lookup:

        result["Tilt Angle b"] = (
            result["Date"]
            .dt.strftime("%B")
            .map(tilt_lookup)
            .fillna(0)
        )

    else:

        result["Tilt Angle b"] = 0.0

    result["a+b"] = (
        result["Elevation angle a"]
        + result["Tilt Angle b"]
    )

    result["SIN(a+b)"] = np.sin(
        np.radians(
            result["a+b"]
        )
    )

    result["Sin(a)"] = (
        np.sin(
            np.radians(
                result["Elevation angle a"]
            )
        )
        .clip(
            lower=1e-6
        )
    )

    return result


# ============================================================
# EFFICIENCY
# ============================================================

def calculate_efficiency_loss(
    df,
    poa,
    actual,
):

    standard = pd.to_numeric(
        df[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    area = pd.to_numeric(
        df["Total area(m2)"],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    actual = safe_array(actual)
    poa = safe_array(poa)

    n = min(
        len(actual),
        len(poa),
    )

    if n == 0:
        return 0.0

    actual = actual[:n]
    poa = poa[:n]

    valid = (
        np.isfinite(actual)
        & np.isfinite(poa)
    )

    actual = actual[valid]
    poa = poa[valid]

    if len(actual) == 0:
        return 0.0

    actual_peak = np.nanmax(
        actual
    )

    poa_peak = np.nanmax(
        poa
    )

    if (
        not np.isfinite(actual_peak)
        or not np.isfinite(poa_peak)
        or actual_peak <= 0
        or poa_peak <= 0
    ):
        return 0.0

    base_area = np.sum(
        area
        * standard
        / 100
    )

    loss_coeff = np.sum(
        area / 100
    )

    if (
        not np.isfinite(loss_coeff)
        or loss_coeff <= 0
    ):
        return 0.0

    target_area = (
        actual_peak
        * 1_000_000
        / poa_peak
    )

    loss = (
        base_area
        - target_area
    ) / loss_coeff

    max_loss = (
        np.nanmin(standard)
        if len(standard)
        else 0.0
    )

    return float(
        np.clip(
            loss,
            0,
            max_loss,
        )
    )


def apply_efficiency_loss(
    df,
    loss,
):

    result = df.copy()

    standard = pd.to_numeric(
        result[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).fillna(0)

    area = pd.to_numeric(
        result["Total area(m2)"],
        errors="coerce",
    ).fillna(0)

    result[
        "Efficiency Losses(%)"
    ] = float(loss)

    result[
        "Net Efficiency (%)"
    ] = (
        standard - float(loss)
    ).clip(
        lower=0
    )

    result["Eff Area"] = (
        area
        * result[
            "Net Efficiency (%)"
        ]
        / 100
    )

    return result


# ============================================================
# BACKEND BLOCKS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_backend_blocks(
    file_bytes,
    cluster,
):

    sheet = (
        "Backend Cal CL1"
        if cluster
        else "Backend Cal"
    )

    backend = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet,
    )

    backend.columns = (
        backend.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        backend,
        ["Block No."],
        sheet,
    )

    return numeric(
        backend["Block No."]
    ).to_numpy(
        dtype=float
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=8,
)
def optimize_tracking_cached(
    blocks_tuple,
    weighted_ghi_tuple,
    actual_tuple,
):

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    weighted_ghi = np.asarray(
        weighted_ghi_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    n = min(
        len(blocks),
        len(weighted_ghi),
        len(actual),
    )

    blocks = blocks[:n]
    weighted_ghi = weighted_ghi[:n]
    actual = actual[:n]

    mask = (
        np.isfinite(blocks)
        & np.isfinite(weighted_ghi)
        & np.isfinite(actual)
        & (actual != 0)
    )

    blocks = blocks[mask]
    weighted_ghi = weighted_ghi[mask]
    actual = actual[mask]

    if len(actual) == 0:

        raise ValueError(
            "No valid Actual power values found."
        )

    actual_peak = np.max(
        actual
    )

    actual_energy = np.sum(
        actual
    )

    if actual_peak <= 0:

        raise ValueError(
            "Actual peak power is invalid."
        )

    if actual_energy <= 0:

        raise ValueError(
            "Actual energy is invalid."
        )

    def objective(x):

        (
            dhi,
            start,
            end,
            max_block,
            east,
            west,
        ) = np.rint(x).astype(int)

        if not (
            start
            < max_block
            < end
        ):
            return 1e9

        d1 = (
            start
            - 1
            - max_block
        )

        d2 = (
            end
            + 1
            - max_block
        )

        if d1 == 0 or d2 == 0:
            return 1e9

        m1 = 90 / d1
        m2 = 90 / d2

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
                abs(east),
            ),
            np.where(
                (
                    blocks > max_block
                )
                & (
                    zenith > west
                ),
                west,
                zenith,
            ),
        )

        cos_alpha = np.clip(
            np.cos(
                np.radians(panel)
            ),
            1e-6,
            None,
        )

        prediction = (
            weighted_ghi
            * (
                1
                - dhi / 100
            )
            / cos_alpha
            / 1_000_000
        )

        if not np.all(
            np.isfinite(prediction)
        ):
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual
                    - prediction
                )
            )
            / actual_peak
        )

        peak_error = (
            abs(
                actual_peak
                - np.max(prediction)
            )
            / actual_peak
        )

        energy_error = (
            abs(
                actual_energy
                - np.sum(prediction)
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
        bounds=PARAM_BOUNDS,
        strategy="best1bin",
        maxiter=MAX_OPT_ITER,
        popsize=OPT_POPSIZE,
        tol=0.005,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=False,
        workers=1,
        integrality=[
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    )

    best = np.rint(
        result.x
    ).astype(int)

    return {
        "DHI": int(best[0]),
        "start": int(best[1]),
        "end": int(best[2]),
        "max": int(best[3]),
        "east": int(best[4]),
        "west": int(best[5]),
    }


# ============================================================
# TRACKING FORECAST
# ============================================================

def tracking_forecast(
    blocks,
    weighted_ghi,
    params,
):

    dhi = int(params["DHI"])
    start = int(params["start"])
    end = int(params["end"])
    max_block = int(params["max"])
    east = int(params["east"])
    west = int(params["west"])

    if not (
        start
        < max_block
        < end
    ):

        raise ValueError(
            "Starting Block < Max Block < Ending Block is required."
        )

    d1 = (
        start
        - 1
        - max_block
    )

    d2 = (
        end
        + 1
        - max_block
    )

    if d1 == 0 or d2 == 0:

        raise ValueError(
            "Invalid tracking block configuration."
        )

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    weighted_ghi = np.asarray(
        weighted_ghi,
        dtype=float,
    )

    n = min(
        len(blocks),
        len(weighted_ghi),
    )

    blocks = blocks[:n]
    weighted_ghi = weighted_ghi[:n]

    m1 = 90 / d1
    m2 = 90 / d2

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
            abs(east),
        ),
        np.where(
            (
                blocks > max_block
            )
            & (
                zenith > west
            ),
            west,
            zenith,
        ),
    )

    cos_alpha = np.clip(
        np.cos(
            np.radians(panel)
        ),
        1e-6,
        None,
    )

    return (
        weighted_ghi
        * (
            1
            - dhi / 100
        )
        / cos_alpha
        / 1_000_000
    )


# ============================================================
# CACHED FIXED FORECAST
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=8,
)
def fixed_forecast_cached(
    input_df,
    base_df,
    lat,
    tilt_items,
    cluster,
    weights_items,
    loss,
):

    weights = dict(
        weights_items
    )

    solar = prepare_solar_angles_cached(
        input_df,
        lat,
        tilt_items,
        False,
    )

    actual = safe_array(
        input_df["Actual"]
    )

    efficiency = apply_efficiency_loss(
        base_df,
        loss,
    )

    forecast = np.zeros(
        len(input_df),
        dtype=float,
    )

    if cluster:

        for ghi_col, weight_col in zip(
            CLUSTER_GHI_COLUMNS,
            CLUSTER_WEIGHT_COLUMNS,
        ):

            weight = safe_float(
                weights.get(
                    weight_col,
                    0,
                )
            )

            eff_area = (
                efficiency[
                    "Total area(m2)"
                ]
                * efficiency[
                    "Net Efficiency (%)"
                ]
                / 100
                * weight
            ).sum()

            poa = (
                solar[ghi_col]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            forecast += (
                poa.to_numpy(
                    dtype=float
                )
                * float(eff_area)
                / 1_000_000
            )

        title = (
            "🏗️ Fixed Cluster Forecast vs Actual"
        )

    else:

        poa = (
            solar["GHI_Forecast"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

        forecast = (
            poa.to_numpy(
                dtype=float
            )
            * float(
                efficiency["Eff Area"].sum()
            )
            / 1_000_000
        )

        title = (
            "🏗️ Fixed Forecast vs Actual"
        )

    return (
        forecast,
        actual,
        efficiency,
        title,
    )


# ============================================================
# CACHED TRACKING FORECAST
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=12,
)
def tracking_forecast_cached(
    input_df,
    base_df,
    lat,
    cluster,
    weights_items,
    blocks,
    loss,
    params_items,
):

    weights = dict(
        weights_items
    )

    params = dict(
        params_items
    )

    solar = prepare_solar_angles_cached(
        input_df,
        lat,
        tuple(),
        True,
    )

    actual = safe_array(
        input_df["Actual"]
    )

    efficiency = apply_efficiency_loss(
        base_df,
        loss,
    )

    if cluster:

        weighted_ghi = np.zeros(
            len(input_df),
            dtype=float,
        )

        for ghi_col, weight_col in zip(
            CLUSTER_GHI_COLUMNS,
            CLUSTER_WEIGHT_COLUMNS,
        ):

            weight = safe_float(
                weights.get(
                    weight_col,
                    0,
                )
            )

            eff_area = (
                efficiency[
                    "Total area(m2)"
                ]
                * efficiency[
                    "Net Efficiency (%)"
                ]
                / 100
                * weight
            ).sum()

            weighted_ghi += (
                input_df[ghi_col]
                .to_numpy(
                    dtype=float
                )
                * float(eff_area)
            )

        title = (
            "🔄 Tracking Cluster Forecast vs Actual"
        )

    else:

        weighted_ghi = (
            input_df[
                "GHI_Forecast"
            ]
            .to_numpy(
                dtype=float
            )
            * float(
                efficiency["Eff Area"].sum()
            )
        )

        title = (
            "🔄 Tracking Forecast vs Actual"
        )

    n = min(
        len(blocks),
        len(weighted_ghi),
        len(actual),
    )

    forecast = tracking_forecast(
        np.asarray(blocks[:n]),
        weighted_ghi[:n],
        params,
    )

    return (
        forecast,
        actual[:n],
        efficiency,
        title,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    forecast,
    actual,
):

    forecast = safe_array(
        forecast
    )

    actual = safe_array(
        actual
    )

    n = min(
        len(forecast),
        len(actual),
    )

    if n == 0:

        return {
            "MAPE": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Peak Error": np.nan,
            "Energy Error": np.nan,
        }

    forecast = forecast[:n]
    actual = actual[:n]

    valid = (
        np.isfinite(forecast)
        & np.isfinite(actual)
    )

    forecast = forecast[valid]
    actual = actual[valid]

    if len(actual) == 0:

        return {
            "MAPE": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Peak Error": np.nan,
            "Energy Error": np.nan,
        }

    error = (
        forecast
        - actual
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    nonzero = (
        np.abs(actual) > 1e-9
    )

    if np.any(nonzero):

        mape = (
            np.mean(
                np.abs(
                    error[nonzero]
                    / actual[nonzero]
                )
            )
            * 100
        )

    else:

        mape = np.nan

    actual_peak = np.max(
        np.abs(actual)
    )

    if actual_peak > 0:

        peak_error = (
            abs(
                np.max(forecast)
                - np.max(actual)
            )
            / actual_peak
            * 100
        )

        energy_error = (
            abs(
                np.sum(forecast)
                - np.sum(actual)
            )
            / np.sum(
                np.abs(actual)
            )
            * 100
        )

    else:

        peak_error = np.nan
        energy_error = np.nan

    return {
        "MAPE": float(mape),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Peak Error": float(peak_error),
        "Energy Error": float(energy_error),
    }


def show_metrics(
    forecast,
    actual,
):

    metrics = calculate_metrics(
        forecast,
        actual,
    )

    columns = st.columns(5)

    values = [
        (
            "MAPE",
            metrics["MAPE"],
            "%",
        ),
        (
            "MAE",
            metrics["MAE"],
            " MW",
        ),
        (
            "RMSE",
            metrics["RMSE"],
            " MW",
        ),
        (
            "Peak Error",
            metrics["Peak Error"],
            "%",
        ),
        (
            "Energy Error",
            metrics["Energy Error"],
            "%",
        ),
    ]

    for column, (
        label,
        value,
        suffix,
    ) in zip(
        columns,
        values,
    ):

        if np.isfinite(value):

            column.metric(
                label,
                f"{value:.2f}{suffix}",
            )

        else:

            column.metric(
                label,
                "N/A",
            )


# ============================================================
# CHART
# ============================================================

def show_forecast_chart(
    forecast,
    actual,
    title,
):

    forecast = safe_array(
        forecast
    )

    actual = safe_array(
        actual
    )

    n = min(
        len(forecast),
        len(actual),
    )

    if n == 0:

        st.warning(
            "No forecast data available."
        )

        return

    forecast = forecast[:n]
    actual = actual[:n]

    # Keep calculations at full resolution.
    # Reduce chart payload for very large datasets.
    max_points = 3000

    if n > max_points:

        step = int(
            np.ceil(
                n / max_points
            )
        )

        x = np.arange(
            1,
            n + 1,
            step,
        )

        chart_forecast = forecast[::step]
        chart_actual = actual[::step]

    else:

        x = np.arange(
            1,
            n + 1,
        )

        chart_forecast = forecast
        chart_actual = actual

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=chart_forecast,
            mode="lines",
            name="Forecast",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=chart_actual,
            mode="lines",
            name="Actual",
        )
    )

    fig.update_layout(
        title=title,
        height=430,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
        ),
        xaxis_title="15 Minute Block",
        yaxis_title="Power (MW)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
        },
    )


# ============================================================
# EFFICIENCY CONTROL
# ============================================================

def efficiency_control(
    df,
    default_loss,
    key,
):

    values = pd.to_numeric(
        df[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).dropna()

    max_loss = (
        float(values.min())
        if not values.empty
        else 0.0
    )

    default_loss = float(
        np.clip(
            safe_float(
                default_loss
            ),
            0,
            max_loss,
        )
    )

    return st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=max_loss,
        value=default_loss,
        step=0.1,
        format="%.2f",
        key=key,
    )


# ============================================================
# TRACKING PARAMETERS
# ============================================================

def tracking_parameter_controls(
    params,
    prefix,
):

    st.markdown(
        '<div class="section">⚙️ Tracking Parameters</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Parameters update automatically."
    )

    c1, c2, c3 = st.columns(3)

    dhi = c1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=10,
        value=int(params["DHI"]),
        step=1,
        key=f"{prefix}_dhi",
    )

    start = c2.number_input(
        "Starting Block",
        min_value=0,
        max_value=30,
        value=int(params["start"]),
        step=1,
        key=f"{prefix}_start",
    )

    end = c3.number_input(
        "Ending Block",
        min_value=65,
        max_value=80,
        value=int(params["end"]),
        step=1,
        key=f"{prefix}_end",
    )

    c1, c2, c3 = st.columns(3)

    max_block = c1.number_input(
        "Max Block",
        min_value=44,
        max_value=60,
        value=int(params["max"]),
        step=1,
        key=f"{prefix}_max",
    )

    east = c2.number_input(
        "East Limit",
        min_value=0,
        max_value=70,
        value=int(params["east"]),
        step=1,
        key=f"{prefix}_east",
    )

    west = c3.number_input(
        "West Limit",
        min_value=0,
        max_value=70,
        value=int(params["west"]),
        step=1,
        key=f"{prefix}_west",
    )

    return {
        "DHI": int(dhi),
        "start": int(start),
        "end": int(end),
        "max": int(max_block),
        "east": int(east),
        "west": int(west),
    }


# ============================================================
# INPUT EDITOR
# ============================================================

def input_data_editor(
    df,
    cluster,
):

    st.markdown(
        '<div class="section">📊 Input GHI & Actual Power</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Edit GHI forecast and Actual values, then click "
        "**Update Input Data**."
    )

    if cluster:

        columns = [
            "Actual",
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

    else:

        columns = [
            "GHI_Forecast",
            "Actual",
        ]

    columns = [
        c
        for c in columns
        if c in df.columns
    ]

    display = df[
        columns
    ].copy()

    for column in columns:

        display[column] = numeric(
            display[column]
        ).to_numpy()

    with st.form(
        "loss_input_form",
        clear_on_submit=False,
        border=False,
    ):

        edited = st.data_editor(
            display,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            key=(
                f"loss_input_editor_"
                f"{st.session_state.input_editor_version}"
            ),
            column_config={
                column:
                st.column_config.NumberColumn(
                    column,
                    step=0.01,
                    format="%.2f",
                )
                for column in columns
            },
        )

        submitted = st.form_submit_button(
            "✓ Update Input Data",
            type="secondary",
        )

    if submitted:

        result = df.copy()

        for column in columns:

            result[column] = numeric(
                edited[column]
            ).to_numpy()

        st.session_state.input_df = result

        st.session_state.model_started = False

        reset_calculation_state()

        st.session_state.input_editor_version += 1

        st.rerun()

    return (
        st.session_state.input_df
        if st.session_state.input_df is not None
        else df
    )


# ============================================================
# RESULTS FRAGMENT
#
# This is the major freezing fix.
#
# Tracking parameter changes rerun only this fragment instead
# of rebuilding the whole page.
# ============================================================

@st.fragment
def calculation_area(
    plant_type,
    input_df,
    base_df,
    lat,
    tilt_lookup,
    cluster,
    weights,
    file_bytes,
):

    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    if plant_type == FIXED_PLANT:

        solar = prepare_solar_angles_cached(
            input_df,
            lat,
            tuple(
                tilt_lookup.items()
            ),
            False,
        )

        if cluster:

            poa = (
                solar["CL1-GHI"]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            loss_key = (
                "cluster_fixed_loss"
            )

            widget_key = (
                "cluster_fixed_loss_input"
            )

        else:

            poa = (
                solar["GHI_Forecast"]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            loss_key = "fixed_loss"

            widget_key = (
                "fixed_loss_input"
            )

        if (
            st.session_state[loss_key]
            is None
        ):

            st.session_state[loss_key] = (
                calculate_efficiency_loss(
                    base_df,
                    poa,
                    input_df["Actual"],
                )
            )

        loss = efficiency_control(
            base_df,
            st.session_state[loss_key],
            widget_key,
        )

        st.session_state[loss_key] = (
            float(loss)
        )

        (
            forecast,
            actual,
            efficiency,
            title,
        ) = fixed_forecast_cached(
            input_df,
            base_df,
            lat,
            tuple(
                tilt_lookup.items()
            ),
            cluster,
            tuple(
                weights.items()
            ),
            float(loss),
        )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    else:

        solar = prepare_solar_angles_cached(
            input_df,
            lat,
            tuple(),
            True,
        )

        if cluster:

            poa = (
                solar["CL1-GHI"]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            loss_key = (
                "cluster_tracking_loss"
            )

            widget_key = (
                "cluster_tracking_loss_input"
            )

        else:

            poa = (
                solar["GHI_Forecast"]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            loss_key = "tracking_loss"

            widget_key = (
                "tracking_loss_input"
            )

        if (
            st.session_state[loss_key]
            is None
        ):

            st.session_state[loss_key] = (
                calculate_efficiency_loss(
                    base_df,
                    poa,
                    input_df["Actual"],
                )
            )

        loss = efficiency_control(
            base_df,
            st.session_state[loss_key],
            widget_key,
        )

        st.session_state[loss_key] = (
            float(loss)
        )

        blocks = read_backend_blocks(
            file_bytes,
            cluster,
        )

        # ----------------------------------------------------
        # OPTIMIZATION ONLY WHEN NEEDED
        # ----------------------------------------------------

        if (
            st.session_state.tracking_params
            is None
        ):

            efficiency = apply_efficiency_loss(
                base_df,
                loss,
            )

            if cluster:

                weighted_ghi = np.zeros(
                    len(input_df),
                    dtype=float,
                )

                for (
                    ghi_col,
                    weight_col,
                ) in zip(
                    CLUSTER_GHI_COLUMNS,
                    CLUSTER_WEIGHT_COLUMNS,
                ):

                    weight = safe_float(
                        weights.get(
                            weight_col,
                            0.0,
                        )
                    )

                    eff_area = (
                        efficiency[
                            "Total area(m2)"
                        ]
                        * efficiency[
                            "Net Efficiency (%)"
                        ]
                        / 100
                        * weight
                    ).sum()

                    weighted_ghi += (
                        input_df[
                            ghi_col
                        ].to_numpy(
                            dtype=float
                        )
                        * float(eff_area)
                    )

            else:

                weighted_ghi = (
                    input_df[
                        "GHI_Forecast"
                    ].to_numpy(
                        dtype=float
                    )
                    * float(
                        efficiency[
                            "Eff Area"
                        ].sum()
                    )
                )

            n = min(
                len(blocks),
                len(weighted_ghi),
                len(input_df),
            )

            with st.spinner(
                "🔄 Optimizing tracking parameters..."
            ):

                params = (
                    optimize_tracking_cached(
                        tuple(
                            blocks[:n].tolist()
                        ),
                        tuple(
                            weighted_ghi[
                                :n
                            ].tolist()
                        ),
                        tuple(
                            input_df[
                                "Actual"
                            ]
                            .to_numpy(
                                dtype=float
                            )[:n]
                            .tolist()
                        ),
                    )
                )

            st.session_state.tracking_params = (
                params
            )

        params = (
            tracking_parameter_controls(
                st.session_state.tracking_params,
                (
                    "cluster_tracking"
                    if cluster
                    else "tracking"
                ),
            )
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Parameter changes update immediately.
        # No second button.
        #
        # Optimization does NOT run again.
        # ----------------------------------------------------

        st.session_state.tracking_params = (
            params
        )

        (
            forecast,
            actual,
            efficiency,
            title,
        ) = tracking_forecast_cached(
            input_df,
            base_df,
            lat,
            cluster,
            tuple(
                weights.items()
            ),
            tuple(
                blocks.tolist()
            ),
            float(loss),
            tuple(
                params.items()
            ),
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">📊 Results</div>',
        unsafe_allow_html=True,
    )

    with st.expander(
        "🔍 View Efficiency Calculations"
    ):

        st.dataframe(
            efficiency.round(2),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        '<div class="section">📈 Forecast Performance</div>',
        unsafe_allow_html=True,
    )

    show_metrics(
        forecast,
        actual,
    )

    show_forecast_chart(
        forecast,
        actual,
        title,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="title">☀️ Loss Correction Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Upload your workbook, edit GHI/Actual data, "
        "select Fixed or Tracking, and run the correction."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # FILE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">📁 Input Excel</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel Workbook",
        type=["xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload the Excel workbook to begin."
        )

        return

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:

        st.error(
            "The uploaded workbook is empty."
        )

        return

    # --------------------------------------------------------
    # WORKBOOK CHANGE
    # --------------------------------------------------------

    signature = workbook_signature(
        file_bytes
    )

    if (
        st.session_state.uploaded_signature
        != signature
    ):

        st.session_state.uploaded_signature = (
            signature
        )

        reset_workbook()

    # --------------------------------------------------------
    # READ WORKBOOK
    # --------------------------------------------------------

    try:

        sheets = get_sheet_names(
            file_bytes
        )

        cluster = (
            "Fixed" not in sheets
        )

        base_df = read_area_efficiency(
            file_bytes,
            cluster,
        )

        lat = read_latitude(
            file_bytes
        )

        tilt_lookup = read_tilt_lookup(
            file_bytes
        )

        weights = (
            read_cluster_weights(
                file_bytes
            )
            if cluster
            else {}
        )

        original_input = load_input_data(
            file_bytes,
            cluster,
        )

    except Exception as exc:

        st.error(
            "Unable to load workbook."
        )

        st.exception(exc)

        return

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    if (
        st.session_state.input_df
        is None
    ):

        st.session_state.input_df = (
            original_input.copy()
        )

    input_df = input_data_editor(
        st.session_state.input_df,
        cluster,
    )

    # --------------------------------------------------------
    # PLANT TYPE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    plant_type = st.segmented_control(
        "Plant Type",
        [
            FIXED_PLANT,
            TRACKING_PLANT,
        ],
        default=st.session_state.plant_type,
        selection_mode="single",
        key="plant_type",
        label_visibility="collapsed",
        width="stretch",
    )

    plant_type = (
        plant_type
        or FIXED_PLANT
    )

    # --------------------------------------------------------
    # PLANT CHANGE
    # --------------------------------------------------------

    previous = (
        st.session_state.last_plant_type
    )

    if (
        previous is not None
        and previous != plant_type
    ):

        st.session_state.tracking_params = None

        if plant_type == FIXED_PLANT:

            st.session_state.fixed_loss = None
            st.session_state.cluster_fixed_loss = None

        else:

            st.session_state.tracking_loss = None
            st.session_state.cluster_tracking_loss = None

    st.session_state.last_plant_type = (
        plant_type
    )

    # --------------------------------------------------------
    # RUN BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚀 RUN LOSS CORRECTION",
        type="primary",
        width="stretch",
        key="run_loss_correction",
    ):

        st.session_state.model_started = True

        if plant_type == TRACKING_PLANT:

            # Explicit RUN means a fresh optimization.
            st.session_state.tracking_params = None

            st.session_state.tracking_loss = None
            st.session_state.cluster_tracking_loss = None

        else:

            st.session_state.fixed_loss = None
            st.session_state.cluster_fixed_loss = None

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    if not st.session_state.model_started:

        st.info(
            "Select the plant type and click "
            "**RUN LOSS CORRECTION**."
        )

        return

    # --------------------------------------------------------
    # CALCULATION
    #
    # IMPORTANT:
    # This is a fragment.
    #
    # Tracking parameter changes therefore do not force the
    # entire page to rebuild.
    # --------------------------------------------------------

    calculation_area(
        plant_type,
        input_df,
        base_df,
        lat,
        tilt_lookup,
        cluster,
        weights,
        file_bytes,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
