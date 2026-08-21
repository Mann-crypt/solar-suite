# ============================================================
# LOSS CORRECTION MODEL
# Fixed / Tracking
# Optimized Streamlit Page
#
# Calculation logic preserved
# UI / execution flow optimized to reduce freezing
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

    # Input
    "input_df": None,
    "input_editor_version": 0,

    # Model
    "model_result": None,
    "run_requested": False,

    # Tracking
    "tracking_params": None,

    # Efficiency losses
    "fixed_loss": None,
    "tracking_loss": None,
    "cluster_fixed_loss": None,
    "cluster_tracking_loss": None,

    # Internal signatures
    "last_plant_type": None,
    "last_input_signature": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GENERIC HELPERS
# ============================================================

def numeric(values):
    """
    Safely convert values to numeric.
    """

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


def validate_columns(df, required, name="Data"):
    """
    Validate required columns.
    """

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


def clean_data_rows(df, date_column="Date"):
    """
    Remove rows after the first blank Date.
    """

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


def workbook_signature(file_bytes):
    """
    Stable workbook signature.
    """

    return hashlib.md5(
        file_bytes
    ).hexdigest()


def dataframe_signature(df):
    """
    Signature for editable input data.

    Used only to detect actual input changes.
    """

    if df is None:
        return None

    try:
        data = pd.util.hash_pandas_object(
            df,
            index=True,
        ).values

        return hashlib.md5(
            data.tobytes()
        ).hexdigest()

    except Exception:
        return (
            len(df),
            tuple(df.columns),
        )


def safe_float(value, default=0.0):
    """
    Convert a value to float safely.
    """

    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return float(default)


def safe_array(values):
    """
    Convert values to clean float numpy array.
    """

    if isinstance(values, pd.Series):

        return pd.to_numeric(
            values,
            errors="coerce",
        ).fillna(0).to_numpy(
            dtype=float
        )

    return pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )


# ============================================================
# RESET HELPERS
# ============================================================

def reset_model_state():
    """
    Reset only model-dependent state.
    """

    st.session_state.model_result = None
    st.session_state.run_requested = False

    st.session_state.tracking_params = None

    st.session_state.fixed_loss = None
    st.session_state.tracking_loss = None

    st.session_state.cluster_fixed_loss = None
    st.session_state.cluster_tracking_loss = None


def reset_for_new_workbook(signature):
    """
    Reset everything dependent on a new workbook.
    """

    st.session_state.uploaded_signature = signature

    st.session_state.input_df = None
    st.session_state.input_editor_version += 1

    st.session_state.last_input_signature = None

    reset_model_state()


# ============================================================
# CACHED EXCEL HELPERS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def workbook_sheets(file_bytes):
    """
    Return workbook sheet names.
    """

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
def read_area_efficiency_cached(
    file_bytes,
    cluster,
):
    """
    Read Area & Efficiency sheet.
    """

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
    )

    df[
        "Standard PV Efficiency (%)"
    ] = pd.to_numeric(
        df[
            "Standard PV Efficiency (%)"
        ],
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
def read_cluster_weights_cached(
    file_bytes,
):
    """
    Read cluster area weights.
    """

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[12, 13, 14, 15, 16],
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

    for col in CLUSTER_WEIGHT_COLUMNS:

        series = pd.to_numeric(
            df[col],
            errors="coerce",
        ).dropna()

        weights[col] = (
            float(series.iloc[0])
            if len(series)
            else 0.0
        )

    return weights


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_latitude_cached(
    file_bytes,
):
    """
    Read latitude.
    """

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
def read_tilt_lookup_cached(
    file_bytes,
):
    """
    Read monthly fixed tilt values.
    """

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
def load_input_data_cached(
    file_bytes,
    cluster,
):
    """
    Load forecast input.

    Fixed:
        Fixed

    Cluster:
        Fixed-CL1
    """

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

            for i, col in enumerate(
                CLUSTER_GHI_COLUMNS
            ):

                if col in df.columns:
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

                df[col] = values

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

        for col in CLUSTER_GHI_COLUMNS:

            df[col] = numeric(
                df[col]
            ).to_numpy()

    else:

        df["GHI_Forecast"] = numeric(
            df["GHI_Forecast"]
        ).to_numpy()

    return df


# ============================================================
# SOLAR ANGLES
# ============================================================

def prepare_solar_angles(
    df,
    lat,
    tilt_lookup=None,
    tracking=False,
):
    """
    Calculate solar-angle components.

    Existing calculation preserved.
    """

    result = df.copy()

    if "Date" in result.columns:

        dates = pd.to_datetime(
            result["Date"],
            errors="coerce",
        )

        if dates.notna().any():

            valid_dates = dates.dropna()

            fallback_date = (
                valid_dates.iloc[0]
                if len(valid_dates)
                else pd.Timestamp.today()
            )

            dates = dates.fillna(
                fallback_date
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
        ).clip(
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
    """
    Automatically calculate efficiency loss.
    """

    validate_columns(
        df,
        [
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        "Efficiency Data",
    )

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
    """
    Apply efficiency loss.
    """

    result = df.copy()

    loss = safe_float(
        loss,
        default=0.0,
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


# ============================================================
# TRACKING OPTIMIZER
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
    """
    Optimize tracking parameters.

    Calculation and objective are unchanged.
    """

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

    if n == 0:
        raise ValueError(
            "No tracking data available."
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

        if (
            d1 == 0
            or d2 == 0
        ):
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
                - np.max(
                    prediction
                )
            )
            / actual_peak
        )

        energy_error = (
            abs(
                actual_energy
                - np.sum(
                    prediction
                )
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
    """
    Calculate tracking forecast.
    """

    dhi = int(
        params["DHI"]
    )

    start = int(
        params["start"]
    )

    end = int(
        params["end"]
    )

    max_block = int(
        params["max"]
    )

    east = int(
        params["east"]
    )

    west = int(
        params["west"]
    )

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

    if (
        d1 == 0
        or d2 == 0
    ):
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

    forecast = (
        weighted_ghi
        * (
            1
            - dhi / 100
        )
        / cos_alpha
        / 1_000_000
    )

    return forecast


# ============================================================
# INPUT EDITOR
# ============================================================

def input_data_editor(
    df,
    cluster,
):
    """
    Editable input data.

    There is NO Apply/Recalculate button.

    The data editor itself is only used for editing.
    The model runs only after RUN LOSS CORRECTION.
    """

    st.markdown(
        '<div class="section">📊 Input GHI & Actual Power</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Edit the values directly if required. "
        "Changes are used the next time you click "
        "**RUN LOSS CORRECTION**."
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
        col
        for col in columns
        if col in df.columns
    ]

    display = df[
        columns
    ].copy()

    for col in columns:

        display[col] = numeric(
            display[col]
        ).to_numpy()

    editor_key = (
        f"loss_input_editor_"
        f"{st.session_state.input_editor_version}"
    )

    edited = st.data_editor(
        display,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key=editor_key,
        column_config={
            col: st.column_config.NumberColumn(
                col,
                step=0.01,
                format="%.2f",
            )
            for col in columns
        },
    )

    result = df.copy()

    for col in columns:

        result[col] = numeric(
            edited[col]
        ).to_numpy()

    return result


# ============================================================
# PLANT SELECTOR
# ============================================================

def plant_selector():

    st.markdown(
        '<div class="section">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    selected = st.segmented_control(
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

    return selected or FIXED_PLANT


# ============================================================
# TRACKING PARAMETERS
# ============================================================

def tracking_parameter_controls(
    params,
    prefix,
):
    """
    Editable tracking parameters.
    """

    st.markdown(
        '<div class="section">⚙️ Tracking Parameters</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Parameters are automatically optimized on the first run. "
        "After that, you can edit them directly."
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
# EFFICIENCY CONTROL
# ============================================================

def efficiency_control(
    df,
    auto_loss,
    key,
):
    """
    Editable efficiency loss.
    """

    efficiency = pd.to_numeric(
        df[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).dropna()

    if efficiency.empty:

        max_loss = 0.0

    else:

        max_loss = float(
            efficiency.min()
        )

        if not np.isfinite(
            max_loss
        ):
            max_loss = 0.0

    default_loss = safe_float(
        auto_loss,
        default=0.0,
    )

    default_loss = float(
        np.clip(
            default_loss,
            0,
            max_loss,
        )
    )

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=max_loss,
        value=default_loss,
        step=0.1,
        format="%.2f",
        key=key,
    )

    return (
        apply_efficiency_loss(
            df,
            loss,
        ),
        float(loss),
    )


# ============================================================
# EFFICIENCY TABLE
# ============================================================

def show_efficiency_table(df):

    cols = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)",
        "Eff Area",
    ]

    cols = [
        col
        for col in cols
        if col in df.columns
    ]

    display = df[
        cols
    ].copy()

    numeric_cols = (
        display.select_dtypes(
            include=np.number
        ).columns
    )

    if len(numeric_cols):

        display[numeric_cols] = (
            display[numeric_cols]
            .round(2)
        )

    with st.expander(
        "🔍 View Efficiency Calculations"
    ):

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    forecast,
    actual,
):
    """
    Calculate forecast metrics.
    """

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

    c1, c2, c3, c4, c5 = st.columns(5)

    values = [
        (
            c1,
            "MAPE",
            metrics["MAPE"],
            "%",
        ),
        (
            c2,
            "MAE",
            metrics["MAE"],
            " MW",
        ),
        (
            c3,
            "RMSE",
            metrics["RMSE"],
            " MW",
        ),
        (
            c4,
            "Peak Error",
            metrics["Peak Error"],
            "%",
        ),
        (
            c5,
            "Energy Error",
            metrics["Energy Error"],
            "%",
        ),
    ]

    for column, label, value, suffix in values:

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
# FORECAST CHART
# ============================================================

def show_forecast_chart(
    forecast,
    actual,
    title,
):
    """
    Forecast vs Actual chart.
    """

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
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


# ============================================================
# BACKEND BLOCKS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_backend_blocks_cached(
    file_bytes,
    cluster,
):
    """
    Read tracking backend block numbers.
    """

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

    blocks = numeric(
        backend["Block No."]
    ).to_numpy(
        dtype=float
    )

    return blocks


# ============================================================
# FIXED FORECAST
# ============================================================

def run_fixed(
    df,
    input_df,
    lat,
    tilt_lookup,
    cluster,
    weights=None,
):
    """
    Run Fixed plant forecast.

    Calculation logic preserved.
    """

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=False,
    )

    actual = safe_array(
        input_df["Actual"]
    )

    if cluster:

        poa_for_loss = (
            solar["CL1-GHI"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

        auto_loss = calculate_efficiency_loss(
            df,
            poa_for_loss,
            actual,
        )

        existing_loss = (
            st.session_state.cluster_fixed_loss
        )

        if existing_loss is None:
            existing_loss = auto_loss

        df, loss = efficiency_control(
            df,
            existing_loss,
            "cluster_fixed_loss_input",
        )

        st.session_state.cluster_fixed_loss = loss

        forecast = np.zeros(
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
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * cluster_weight
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

        auto_loss = calculate_efficiency_loss(
            df,
            poa,
            actual,
        )

        existing_loss = (
            st.session_state.fixed_loss
        )

        if existing_loss is None:
            existing_loss = auto_loss

        df, loss = efficiency_control(
            df,
            existing_loss,
            "fixed_loss_input",
        )

        st.session_state.fixed_loss = loss

        eff_area_total = (
            df["Eff Area"].sum()
        )

        forecast = (
            poa.to_numpy(
                dtype=float
            )
            * float(eff_area_total)
            / 1_000_000
        )

        title = (
            "🏗️ Fixed Forecast vs Actual"
        )

    show_efficiency_table(
        df
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

    return {
        "forecast": forecast,
        "actual": actual,
        "efficiency": df,
    }


# ============================================================
# TRACKING FORECAST
# ============================================================

def run_tracking(
    df,
    input_df,
    lat,
    tilt_lookup,
    file_bytes,
    cluster,
):
    """
    Run Tracking plant forecast.
    """

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=True,
    )

    actual = safe_array(
        input_df["Actual"]
    )

    if cluster:

        weights = read_cluster_weights_cached(
            file_bytes
        )

        poa_for_loss = (
            solar["CL1-GHI"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

        auto_loss = calculate_efficiency_loss(
            df,
            poa_for_loss,
            actual,
        )

        existing_loss = (
            st.session_state.cluster_tracking_loss
        )

        if existing_loss is None:
            existing_loss = auto_loss

        df, loss = efficiency_control(
            df,
            existing_loss,
            "cluster_tracking_loss_input",
        )

        st.session_state.cluster_tracking_loss = loss

        weighted_ghi = np.zeros(
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
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * cluster_weight
            ).sum()

            weighted_ghi += (
                input_df[ghi_col]
                .to_numpy(
                    dtype=float
                )
                * float(eff_area)
            )

        prefix = "cluster_tracking"

        title = (
            "🔄 Tracking Cluster Forecast vs Actual"
        )

    else:

        poa_for_loss = (
            solar["GHI_Forecast"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

        auto_loss = calculate_efficiency_loss(
            df,
            poa_for_loss,
            actual,
        )

        existing_loss = (
            st.session_state.tracking_loss
        )

        if existing_loss is None:
            existing_loss = auto_loss

        df, loss = efficiency_control(
            df,
            existing_loss,
            "tracking_loss_input",
        )

        st.session_state.tracking_loss = loss

        weighted_ghi = (
            input_df["GHI_Forecast"]
            .to_numpy(
                dtype=float
            )
            * df["Eff Area"].sum()
        )

        prefix = "tracking"

        title = (
            "🔄 Tracking Forecast vs Actual"
        )

    # --------------------------------------------------------
    # BACKEND BLOCKS
    # --------------------------------------------------------

    blocks = read_backend_blocks_cached(
        file_bytes,
        cluster,
    )

    n = min(
        len(blocks),
        len(weighted_ghi),
        len(actual),
    )

    if n == 0:
        raise ValueError(
            "No matching tracking data found."
        )

    blocks = blocks[:n]
    weighted_ghi = weighted_ghi[:n]
    actual = actual[:n]

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    if st.session_state.tracking_params is None:

        with st.spinner(
            "🔄 Optimizing tracking parameters..."
        ):

            params = optimize_tracking_cached(
                tuple(blocks.tolist()),
                tuple(weighted_ghi.tolist()),
                tuple(actual.tolist()),
            )

        st.session_state.tracking_params = params

    else:

        params = (
            st.session_state.tracking_params
        )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    params = tracking_parameter_controls(
        params,
        prefix,
    )

    st.session_state.tracking_params = params

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast = tracking_forecast(
        blocks,
        weighted_ghi,
        params,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    show_efficiency_table(
        df
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

    return {
        "forecast": forecast,
        "actual": actual,
        "efficiency": df,
        "tracking_params": params,
    }


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
    # FILE
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

    current_signature = workbook_signature(
        file_bytes
    )

    if (
        st.session_state.uploaded_signature
        != current_signature
    ):

        reset_for_new_workbook(
            current_signature
        )

    # ========================================================
    # WORKBOOK SHEETS
    # ========================================================

    try:

        sheets = workbook_sheets(
            file_bytes
        )

    except Exception as exc:

        st.error(
            "Unable to read the uploaded workbook."
        )

        st.exception(exc)

        return

    # ========================================================
    # WORKBOOK TYPE
    # ========================================================

    cluster = (
        "Fixed" not in sheets
    )

    # ========================================================
    # LOAD WORKBOOK PARAMETERS
    # ========================================================

    try:

        base_df = read_area_efficiency_cached(
            file_bytes,
            cluster,
        )

        lat = read_latitude_cached(
            file_bytes
        )

        tilt_lookup = read_tilt_lookup_cached(
            file_bytes
        )

        weights = (
            read_cluster_weights_cached(
                file_bytes
            )
            if cluster
            else None
        )

        original_input = load_input_data_cached(
            file_bytes,
            cluster,
        )

    except Exception as exc:

        st.error(
            "Unable to load workbook parameters."
        )

        st.exception(exc)

        return

    # ========================================================
    # INPUT DATA
    # ========================================================

    if st.session_state.input_df is None:

        st.session_state.input_df = (
            original_input.copy()
        )

    input_df = input_data_editor(
        st.session_state.input_df,
        cluster,
    )

    # ========================================================
    # DO NOT CONTINUOUSLY WRITE THE DATAFRAME TO STATE
    # UNLESS IT ACTUALLY CHANGED
    # ========================================================

    new_input_signature = dataframe_signature(
        input_df
    )

    if (
        st.session_state.last_input_signature
        is None
    ):

        st.session_state.last_input_signature = (
            new_input_signature
        )

    elif (
        new_input_signature
        != st.session_state.last_input_signature
    ):

        st.session_state.input_df = (
            input_df.copy()
        )

        st.session_state.last_input_signature = (
            new_input_signature
        )

        # Do not immediately run the model.
        # The user controls execution through RUN.
        reset_model_state()

    else:

        st.session_state.input_df = (
            input_df
        )

    # ========================================================
    # PLANT TYPE
    # ========================================================

    plant_type = plant_selector()

    # ========================================================
    # PLANT CHANGE
    # ========================================================

    previous_plant = (
        st.session_state.last_plant_type
    )

    if (
        previous_plant is not None
        and previous_plant != plant_type
    ):

        # Keep input data.
        # Clear only model-specific results.
        reset_model_state()

    st.session_state.last_plant_type = (
        plant_type
    )

    # ========================================================
    # RUN BUTTON
    # ========================================================

    st.markdown("")

    run_clicked = st.button(
        "🚀 RUN LOSS CORRECTION",
        type="primary",
        use_container_width=True,
        key="run_loss_correction",
    )

    if run_clicked:

        reset_model_state()

        st.session_state.run_requested = True

    # ========================================================
    # SHOW LAST RESULT
    # ========================================================

    if (
        not st.session_state.run_requested
        and st.session_state.model_result
        is not None
    ):

        return

    # ========================================================
    # NOTHING TO RUN
    # ========================================================

    if not st.session_state.run_requested:

        st.info(
            "Select the plant type and click "
            "**RUN LOSS CORRECTION**."
        )

        return

    # ========================================================
    # MODEL EXECUTION
    # ========================================================

    try:

        # ----------------------------------------------------
        # FIXED
        # ----------------------------------------------------

        if plant_type == FIXED_PLANT:

            result = run_fixed(
                base_df,
                input_df,
                lat,
                tilt_lookup,
                cluster,
                weights,
            )

        # ----------------------------------------------------
        # TRACKING
        # ----------------------------------------------------

        else:

            result = run_tracking(
                base_df,
                input_df,
                lat,
                tilt_lookup,
                file_bytes,
                cluster,
            )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        st.session_state.model_result = result

        st.session_state.run_requested = False

    except Exception as exc:

        st.session_state.model_result = None
        st.session_state.run_requested = False

        st.error(
            "❌ Loss correction failed."
        )

        st.exception(exc)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
