# ============================================================
# LOSS CORRECTION MODEL
# Fixed / Tracking
#
# Freeze-resistant Streamlit architecture
#
# Calculation formulas preserved.
#
# Heavy operations:
#   - Excel reading
#   - Solar-angle preparation
#   - Automatic efficiency calculation
#   - Tracking optimization
#
# are performed only when required.
#
# Parameter changes DO NOT require another RUN button.
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

MAX_CHART_POINTS = 6000


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

    "model_started": False,

    # Heavy/prepared model data.
    "prepared_model": None,

    # Final lightweight result.
    "result": None,

    # Tracking optimization result.
    "tracking_params": None,

    # Efficiency states.
    "fixed_loss": None,
    "tracking_loss": None,
    "cluster_fixed_loss": None,
    "cluster_tracking_loss": None,

    "last_plant_type": None,

    "input_editor_version": 0,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# BASIC HELPERS
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


def safe_float(
    value,
    default=0.0,
):

    try:

        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return float(default)


def safe_array(values):

    return (
        pd.to_numeric(
            pd.Series(values),
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )


def validate_columns(
    df,
    required,
    name="Data",
):

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} is missing required column(s): "
            f"{', '.join(missing)}"
        )


def clean_data_rows(
    df,
    date_column="Date",
):

    result = df.copy()

    if date_column in result.columns:

        null_rows = result[
            result[date_column].isna()
        ].index

        if len(null_rows):

            first_null = null_rows[0]

            position = result.index.get_loc(
                first_null
            )

            result = result.iloc[:position]

    return result.reset_index(drop=True)


def workbook_signature(
    file_bytes,
):

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ============================================================
# STATE RESET
# ============================================================

def reset_model(
    clear_tracking=True,
):

    st.session_state.model_started = False
    st.session_state.prepared_model = None
    st.session_state.result = None

    if clear_tracking:
        st.session_state.tracking_params = None

    for key in [
        "fixed_loss",
        "tracking_loss",
        "cluster_fixed_loss",
        "cluster_tracking_loss",
    ]:
        st.session_state[key] = None


def reset_workbook():

    st.session_state.input_df = None
    st.session_state.last_plant_type = None

    st.session_state.input_editor_version += 1

    reset_model(
        clear_tracking=True
    )


# ============================================================
# CACHED EXCEL FUNCTIONS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def workbook_sheets(
    file_bytes,
):

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

        first_null = null_rows[0]

        position = df.index.get_loc(
            first_null
        )

        df = df.iloc[:position]

    df = df.dropna(
        subset=[
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        how="all",
    ).copy()

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

            first_null = null_rows[0]

            position = df.index.get_loc(
                first_null
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

    df = clean_data_rows(
        df
    )

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
            ).fillna(0)

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

    else:

        validate_columns(
            df,
            ["GHI_Forecast"],
            "Fixed Forecast",
        )

    df["Actual"] = numeric(
        df["Actual"]
    ).to_numpy()

    if cluster:

        for column in CLUSTER_GHI_COLUMNS:

            df[column] = numeric(
                df[column]
            ).to_numpy()

    else:

        df["GHI_Forecast"] = numeric(
            df["GHI_Forecast"]
        ).to_numpy()

    return df


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
# SOLAR ANGLES
# ============================================================

def prepare_solar_angles(
    df,
    lat,
    tilt_lookup=None,
    tracking=False,
):

    result = df.copy()

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

    validate_columns(
        df,
        [
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        "Efficiency Data",
    )

    standard = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
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

    if not np.isfinite(max_loss):
        max_loss = 0.0

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

    loss = safe_float(
        loss
    )

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
    ] = loss

    result[
        "Net Efficiency (%)"
    ] = (
        standard - loss
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


def maximum_efficiency_loss(
    df,
):

    values = pd.to_numeric(
        df[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    minimum = float(
        values.min()
    )

    if not np.isfinite(minimum):
        return 0.0

    return max(
        0.0,
        minimum,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=8,
)
def optimize_tracking(
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
# PREPARE HEAVY MODEL
# ============================================================

def prepare_model(
    base_df,
    input_df,
    lat,
    tilt_lookup,
    weights,
    file_bytes,
    cluster,
):

    actual = safe_array(
        input_df["Actual"]
    )

    fixed_solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=False,
    )

    tracking_solar = prepare_solar_angles(
        input_df,
        lat,
        None,
        tracking=True,
    )

    if cluster:

        fixed_poa = (
            fixed_solar["CL1-GHI"]
            * fixed_solar["SIN(a+b)"]
            / fixed_solar["Sin(a)"]
        )

        tracking_poa = (
            tracking_solar["CL1-GHI"]
            * tracking_solar["SIN(a+b)"]
            / tracking_solar["Sin(a)"]
        )

        fixed_auto = calculate_efficiency_loss(
            base_df,
            fixed_poa,
            actual,
        )

        tracking_auto = calculate_efficiency_loss(
            base_df,
            tracking_poa,
            actual,
        )

        blocks = read_backend_blocks(
            file_bytes,
            True,
        )

    else:

        fixed_poa = (
            fixed_solar["GHI_Forecast"]
            * fixed_solar["SIN(a+b)"]
            / fixed_solar["Sin(a)"]
        )

        tracking_poa = (
            tracking_solar["GHI_Forecast"]
            * tracking_solar["SIN(a+b)"]
            / tracking_solar["Sin(a)"]
        )

        fixed_auto = calculate_efficiency_loss(
            base_df,
            fixed_poa,
            actual,
        )

        tracking_auto = calculate_efficiency_loss(
            base_df,
            tracking_poa,
            actual,
        )

        blocks = read_backend_blocks(
            file_bytes,
            False,
        )

    return {
        "base_df": base_df.copy(),
        "input_df": input_df.copy(),
        "actual": actual,
        "fixed_solar": fixed_solar,
        "tracking_solar": tracking_solar,
        "fixed_auto": fixed_auto,
        "tracking_auto": tracking_auto,
        "weights": weights,
        "blocks": blocks,
        "cluster": cluster,
    }


# ============================================================
# WEIGHTED GHI
# ============================================================

def calculate_weighted_ghi(
    model,
    efficiency_df,
):

    input_df = model["input_df"]
    cluster = model["cluster"]

    if cluster:

        weights = model["weights"]

        weighted = np.zeros(
            len(input_df),
            dtype=float,
        )

        for ghi_col, weight_col in zip(
            CLUSTER_GHI_COLUMNS,
            CLUSTER_WEIGHT_COLUMNS,
        ):

            cluster_weight = safe_float(
                weights.get(
                    weight_col,
                    0.0,
                )
            )

            eff_area = (
                efficiency_df[
                    "Total area(m2)"
                ]
                * efficiency_df[
                    "Net Efficiency (%)"
                ]
                / 100
                * cluster_weight
            ).sum()

            weighted += (
                input_df[
                    ghi_col
                ].to_numpy(
                    dtype=float
                )
                * float(eff_area)
            )

        return weighted

    return (
        input_df[
            "GHI_Forecast"
        ].to_numpy(
            dtype=float
        )
        * float(
            efficiency_df[
                "Eff Area"
            ].sum()
        )
    )


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    model,
    loss,
):

    base_df = model["base_df"]
    input_df = model["input_df"]
    solar = model["fixed_solar"]

    efficiency_df = apply_efficiency_loss(
        base_df,
        loss,
    )

    if model["cluster"]:

        forecast = np.zeros(
            len(input_df),
            dtype=float,
        )

        for ghi_col, weight_col in zip(
            CLUSTER_GHI_COLUMNS,
            CLUSTER_WEIGHT_COLUMNS,
        ):

            cluster_weight = safe_float(
                model["weights"].get(
                    weight_col,
                    0.0,
                )
            )

            eff_area = (
                efficiency_df[
                    "Total area(m2)"
                ]
                * efficiency_df[
                    "Net Efficiency (%)"
                ]
                / 100
                * cluster_weight
            ).sum()

            poa = (
                solar[ghi_col].to_numpy(
                    dtype=float
                )
                * solar[
                    "SIN(a+b)"
                ].to_numpy(
                    dtype=float
                )
                / solar[
                    "Sin(a)"
                ].to_numpy(
                    dtype=float
                )
            )

            forecast += (
                poa
                * float(eff_area)
                / 1_000_000
            )

        title = (
            "🏗️ Fixed Cluster Forecast vs Actual"
        )

    else:

        poa = (
            solar[
                "GHI_Forecast"
            ].to_numpy(
                dtype=float
            )
            * solar[
                "SIN(a+b)"
            ].to_numpy(
                dtype=float
            )
            / solar[
                "Sin(a)"
            ].to_numpy(
                dtype=float
            )
        )

        forecast = (
            poa
            * float(
                efficiency_df[
                    "Eff Area"
                ].sum()
            )
            / 1_000_000
        )

        title = (
            "🏗️ Fixed Forecast vs Actual"
        )

    return {
        "forecast": forecast,
        "actual": model["actual"],
        "efficiency": efficiency_df,
        "title": title,
    }


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    model,
    loss,
    params,
):

    efficiency_df = apply_efficiency_loss(
        model["base_df"],
        loss,
    )

    weighted = calculate_weighted_ghi(
        model,
        efficiency_df,
    )

    blocks = model["blocks"]
    actual = model["actual"]

    n = min(
        len(blocks),
        len(weighted),
        len(actual),
    )

    if n == 0:

        raise ValueError(
            "No matching tracking data found."
        )

    forecast = tracking_forecast(
        blocks[:n],
        weighted[:n],
        params,
    )

    title = (
        "🔄 Tracking Cluster Forecast vs Actual"
        if model["cluster"]
        else "🔄 Tracking Forecast vs Actual"
    )

    return {
        "forecast": forecast,
        "actual": actual[:n],
        "efficiency": efficiency_df,
        "tracking_params": params,
        "title": title,
    }


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
        "Parameters are editable. "
        "Forecast and metrics update automatically."
    )

    columns = st.columns(3)

    settings = [
        (
            "DHI",
            "DHI (%)",
            0,
            10,
        ),
        (
            "start",
            "Starting Block",
            0,
            30,
        ),
        (
            "end",
            "Ending Block",
            65,
            80,
        ),
        (
            "max",
            "Max Block",
            44,
            60,
        ),
        (
            "east",
            "East Limit",
            0,
            70,
        ),
        (
            "west",
            "West Limit",
            0,
            70,
        ),
    ]

    values = {}

    for index, (
        key,
        label,
        minimum,
        maximum,
    ) in enumerate(settings):

        widget_key = (
            f"{prefix}_{key}"
        )

        if widget_key not in st.session_state:

            st.session_state[
                widget_key
            ] = int(
                params[key]
            )

        values[key] = columns[
            index % 3
        ].number_input(
            label,
            min_value=minimum,
            max_value=maximum,
            step=1,
            key=widget_key,
        )

    return {
        key: int(value)
        for key, value in values.items()
    }


# ============================================================
# EFFICIENCY CONTROL
# ============================================================

def efficiency_control(
    auto_loss,
    df,
    key,
):

    maximum = maximum_efficiency_loss(
        df
    )

    default = float(
        np.clip(
            safe_float(auto_loss),
            0,
            maximum,
        )
    )

    if key not in st.session_state:

        st.session_state[key] = default

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=maximum,
        step=0.1,
        format="%.2f",
        key=key,
    )

    return float(loss)


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

    items = [
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
    ) in zip(columns, items):

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
# EFFICIENCY TABLE
# ============================================================

def show_efficiency_table(
    df,
):

    columns = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)",
        "Eff Area",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    display = df[
        columns
    ].copy()

    numeric_columns = (
        display.select_dtypes(
            include=np.number
        ).columns
    )

    if len(numeric_columns):

        display[
            numeric_columns
        ] = (
            display[
                numeric_columns
            ].round(2)
        )

    with st.expander(
        "🔍 View Efficiency Calculations"
    ):

        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# FORECAST CHART
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

    x = np.arange(
        1,
        n + 1,
    )

    # --------------------------------------------------------
    # Display-only downsampling.
    #
    # IMPORTANT:
    # Calculations and metrics still use full data.
    # Only Plotly rendering is reduced.
    # --------------------------------------------------------

    if n > MAX_CHART_POINTS:

        indices = np.linspace(
            0,
            n - 1,
            MAX_CHART_POINTS,
            dtype=int,
        )

        x = x[indices]
        forecast = forecast[indices]
        actual = actual[indices]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                color="#2563EB",
                width=2.5,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(
                color="#EF4444",
                width=2.5,
            ),
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
# PLANT SELECTOR
# ============================================================

def plant_selector():

    st.markdown(
        '<div class="section">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    return (
        st.segmented_control(
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
        or FIXED_PLANT
    )


# ============================================================
# INPUT EDITOR
# ============================================================

def input_data_editor(
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
        column
        for column in columns
        if column in st.session_state.input_df.columns
    ]

    display = (
        st.session_state.input_df[
            columns
        ]
        .copy()
    )

    for column in columns:

        display[column] = numeric(
            display[column]
        ).to_numpy()

    editor_key = (
        "loss_input_editor_"
        f"{st.session_state.input_editor_version}"
    )

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
            key=editor_key,
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

        result = (
            st.session_state.input_df.copy()
        )

        for column in columns:

            result[column] = numeric(
                edited[column]
            ).to_numpy()

        st.session_state.input_df = result

        reset_model(
            clear_tracking=False
        )

        st.session_state.input_editor_version += 1

        st.success(
            "Input data updated. Click RUN LOSS CORRECTION."
        )

        st.rerun()

    return st.session_state.input_df


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="title">☀️ Loss Correction Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Upload your workbook, edit GHI/Actual data if required, "
        "select Fixed or Tracking, and run the correction."
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # UPLOAD
    # ========================================================

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

    # ========================================================
    # WORKBOOK SIGNATURE
    # ========================================================

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

    # ========================================================
    # READ WORKBOOK
    # ========================================================

    try:

        sheets = workbook_sheets(
            file_bytes
        )

        cluster = (
            "Fixed" not in sheets
        )

        base_df = read_area_efficiency(
            file_bytes,
            cluster,
        )

        latitude = read_latitude(
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
            else None
        )

        original_input = load_input_data(
            file_bytes,
            cluster,
        )

    except Exception as exc:

        st.error(
            "Unable to load workbook parameters."
        )

        st.exception(
            exc
        )

        return

    # ========================================================
    # INITIAL INPUT
    # ========================================================

    if (
        st.session_state.input_df
        is None
    ):

        st.session_state.input_df = (
            original_input.copy()
        )

    input_data_editor(
        cluster
    )

    # ========================================================
    # PLANT
    # ========================================================

    plant_type = plant_selector()

    previous_plant = (
        st.session_state.last_plant_type
    )

    if previous_plant is None:

        st.session_state.last_plant_type = (
            plant_type
        )

    elif previous_plant != plant_type:

        reset_model(
            clear_tracking=True
        )

        st.session_state.last_plant_type = (
            plant_type
        )

        # Clear old tracking widget values.
        prefixes = [
            "tracking",
            "cluster_tracking",
        ]

        for prefix in prefixes:

            for key in [
                "DHI",
                "start",
                "end",
                "max",
                "east",
                "west",
            ]:

                st.session_state.pop(
                    f"{prefix}_{key}",
                    None,
                )

    # ========================================================
    # RUN
    #
    # This is the ONLY button required to start/rebuild
    # the heavy model.
    #
    # Parameter changes do NOT require this button.
    # ========================================================

    st.markdown("")

    run_clicked = st.button(
        "🚀 RUN LOSS CORRECTION",
        type="primary",
        width="stretch",
        key="run_loss_correction",
    )

    if run_clicked:

        reset_model(
            clear_tracking=True
        )

        st.session_state.model_started = True

    # ========================================================
    # WAIT
    # ========================================================

    if not st.session_state.model_started:

        st.info(
            "Select the plant type and click "
            "**RUN LOSS CORRECTION**."
        )

        return

    # ========================================================
    # PREPARE HEAVY MODEL ONLY ONCE
    # ========================================================

    try:

        if (
            st.session_state.prepared_model
            is None
        ):

            with st.spinner(
                "Preparing solar model..."
            ):

                st.session_state.prepared_model = (
                    prepare_model(
                        base_df,
                        st.session_state.input_df,
                        latitude,
                        tilt_lookup,
                        weights,
                        file_bytes,
                        cluster,
                    )
                )

        model = (
            st.session_state.prepared_model
        )

        # ----------------------------------------------------
        # Safety check.
        #
        # If input data somehow changed without using the
        # Update button, do not calculate using stale data.
        # ----------------------------------------------------

        if not model["input_df"].equals(
            st.session_state.input_df
        ):

            st.session_state.model_started = False
            st.session_state.prepared_model = None
            st.session_state.result = None

            st.warning(
                "Input data changed. "
                "Click RUN LOSS CORRECTION again."
            )

            return

        # ====================================================
        # FIXED
        # ====================================================

        if plant_type == FIXED_PLANT:

            if cluster:

                efficiency_key = (
                    "cluster_fixed_loss_input"
                )

                state_key = (
                    "cluster_fixed_loss"
                )

            else:

                efficiency_key = (
                    "fixed_loss_input"
                )

                state_key = (
                    "fixed_loss"
                )

            loss = efficiency_control(
                model["fixed_auto"],
                model["base_df"],
                efficiency_key,
            )

            st.session_state[
                state_key
            ] = loss

            result = calculate_fixed_forecast(
                model,
                loss,
            )

        # ====================================================
        # TRACKING
        # ====================================================

        else:

            if cluster:

                efficiency_key = (
                    "cluster_tracking_loss_input"
                )

                state_key = (
                    "cluster_tracking_loss"
                )

                prefix = (
                    "cluster_tracking"
                )

            else:

                efficiency_key = (
                    "tracking_loss_input"
                )

                state_key = (
                    "tracking_loss"
                )

                prefix = "tracking"

            loss = efficiency_control(
                model["tracking_auto"],
                model["base_df"],
                efficiency_key,
            )

            st.session_state[
                state_key
            ] = loss

            # ----------------------------------------------
            # Optimize ONLY if we do not already have
            # optimization parameters.
            # ----------------------------------------------

            if (
                st.session_state.tracking_params
                is None
            ):

                efficiency_df = (
                    apply_efficiency_loss(
                        model["base_df"],
                        loss,
                    )
                )

                weighted = (
                    calculate_weighted_ghi(
                        model,
                        efficiency_df,
                    )
                )

                blocks = model["blocks"]
                actual = model["actual"]

                n = min(
                    len(blocks),
                    len(weighted),
                    len(actual),
                )

                if n == 0:

                    raise ValueError(
                        "No matching tracking data found."
                    )

                with st.spinner(
                    "🔄 Optimizing tracking parameters..."
                ):

                    st.session_state.tracking_params = (
                        optimize_tracking(
                            tuple(
                                blocks[:n].tolist()
                            ),
                            tuple(
                                weighted[:n].tolist()
                            ),
                            tuple(
                                actual[:n].tolist()
                            ),
                        )
                    )

                # Initialize widget state once.
                for key, value in (
                    st.session_state.tracking_params
                    .items()
                ):

                    widget_key = (
                        f"{prefix}_{key}"
                    )

                    if widget_key not in st.session_state:

                        st.session_state[
                            widget_key
                        ] = int(value)

            # ----------------------------------------------
            # Parameters are now lightweight UI state.
            #
            # Changing these causes only the forecast
            # calculation below to run.
            # ----------------------------------------------

            params = tracking_parameter_controls(
                st.session_state.tracking_params,
                prefix,
            )

            st.session_state.tracking_params = (
                params
            )

            result = calculate_tracking_forecast(
                model,
                loss,
                params,
            )

        st.session_state.result = result

    except Exception as exc:

        st.error(
            "❌ Loss correction failed."
        )

        st.exception(
            exc
        )

        return

    # ========================================================
    # RESULTS
    # ========================================================

    result = (
        st.session_state.result
    )

    st.markdown(
        '<div class="section">📊 Results</div>',
        unsafe_allow_html=True,
    )

    show_efficiency_table(
        result["efficiency"]
    )

    st.markdown(
        '<div class="section">📈 Forecast Performance</div>',
        unsafe_allow_html=True,
    )

    show_metrics(
        result["forecast"],
        result["actual"],
    )

    show_forecast_chart(
        result["forecast"],
        result["actual"],
        result["title"],
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
