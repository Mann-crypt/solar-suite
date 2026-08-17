# ============================================================
# STREAMLIT APP
# LOSS CORRECTION MODEL
# FIXED / TRACKING
# ============================================================

import io
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Loss Correction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_OPT_ITER = 40
OPT_POPSIZE = 10

PARAM_BOUNDS = [
    (0, 10),    # DHI %
    (0, 30),    # Starting block
    (65, 80),   # Ending block
    (44, 60),   # Max block
    (0, 70),    # East limit
    (0, 70),    # West limit
]

TRACKING_PARAM_NAMES = [
    "DHI",
    "Starting Block",
    "Ending Block",
    "Max Block",
    "East Limit",
    "West Limit",
]

CLUSTER_GHI_COLS = [
    "CL1-GHI",
    "CL2-GHI",
    "CL3-GHI",
    "CL4-GHI",
    "CL5-GHI",
]

CLUSTER_WEIGHT_COLS = [
    "CL-1",
    "CL-2",
    "CL-3",
    "CL-4",
    "CL-5",
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        padding: 22px 26px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.20);
        background: linear-gradient(
            135deg,
            rgba(59,130,246,.12),
            rgba(16,185,129,.07)
        );
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 750;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #8b949e;
        font-size: .98rem;
    }

    .section-title {
        font-size: 1.22rem;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 10px;
    }

    .card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 16px;
        padding: 16px;
        background: rgba(128,128,128,.035);
    }

    .metric-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 15px;
        padding: 14px 16px;
        min-height: 92px;
        background: rgba(128,128,128,.035);
    }

    .metric-label {
        font-size: .78rem;
        color: #8b949e;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 720;
    }

    .success-box {
        border: 1px solid rgba(34,197,94,.35);
        background: rgba(34,197,94,.08);
        border-radius: 14px;
        padding: 13px 16px;
        margin: 12px 0;
    }

    .warning-box {
        border: 1px solid rgba(245,158,11,.35);
        background: rgba(245,158,11,.08);
        border-radius: 14px;
        padding: 13px 16px;
        margin: 12px 0;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    div.stButton > button {
        border-radius: 11px;
        min-height: 46px;
        font-weight: 650;
    }

    div[data-testid="stDataEditor"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .small-note {
        color: #8b949e;
        font-size: .84rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "plant_type": "🏗️ Fixed",
    "input_df": None,
    "input_file_name": None,
    "model_result": None,
    "tracking_params": None,
    "last_run_signature": None,
    "input_editor_version": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GENERIC HELPERS
# ============================================================

def reset_model():
    st.session_state.model_result = None
    st.session_state.tracking_params = None
    st.session_state.last_run_signature = None


def validate_columns(
    df: pd.DataFrame,
    required,
    name: str = "Data",
):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required column(s): "
            f"{', '.join(missing)}"
        )


def clean_data_rows(
    df: pd.DataFrame,
    date_column: str = "Date",
) -> pd.DataFrame:
    df = df.copy()

    if date_column in df.columns:
        null_idx = df[df[date_column].isna()].index

        if len(null_idx):
            first_idx = null_idx[0]
            pos = df.index.get_loc(first_idx)
            df = df.iloc[:pos]

    return df.reset_index(drop=True)


def numeric_series(
    series,
    fill_value: float = 0.0,
) -> np.ndarray:
    return pd.to_numeric(
        pd.Series(series),
        errors="coerce",
    ).fillna(fill_value).to_numpy(float)


def safe_error(actual_peak, forecast_peak):
    if actual_peak == 0:
        return 0.0
    return abs(actual_peak - forecast_peak) / abs(actual_peak) * 100


def calculate_metrics(
    forecast,
    actual,
) -> Dict[str, float]:

    n = min(len(forecast), len(actual))

    forecast = np.asarray(forecast[:n], dtype=float)
    actual = np.asarray(actual[:n], dtype=float)

    valid = (
        np.isfinite(forecast)
        & np.isfinite(actual)
    )

    forecast = forecast[valid]
    actual = actual[valid]

    if len(actual) == 0:
        return {
            "actual_peak": 0.0,
            "forecast_peak": 0.0,
            "peak_error": 0.0,
            "actual_energy": 0.0,
            "forecast_energy": 0.0,
            "energy_error": 0.0,
            "mae": 0.0,
        }

    actual_peak = float(np.max(actual))
    forecast_peak = float(np.max(forecast))

    actual_energy = float(np.sum(actual))
    forecast_energy = float(np.sum(forecast))

    peak_error = safe_error(
        actual_peak,
        forecast_peak,
    )

    energy_error = (
        abs(actual_energy - forecast_energy)
        / abs(actual_energy)
        * 100
        if actual_energy != 0
        else 0.0
    )

    mae = float(np.mean(np.abs(actual - forecast)))

    return {
        "actual_peak": actual_peak,
        "forecast_peak": forecast_peak,
        "peak_error": peak_error,
        "actual_energy": actual_energy,
        "forecast_energy": forecast_energy,
        "energy_error": energy_error,
        "mae": mae,
    }


# ============================================================
# WORKBOOK HELPERS
# ============================================================

def get_sheet_names(uploaded_file):
    uploaded_file.seek(0)
    return pd.ExcelFile(uploaded_file).sheet_names


def is_cluster_workbook(uploaded_file) -> bool:
    sheets = get_sheet_names(uploaded_file)

    # Non-cluster workbook has the standard "Fixed" sheet.
    # Cluster workbook has "Fixed-CL1" instead.
    return "Fixed" not in sheets


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def read_area_efficiency(
    uploaded_file,
    cluster: bool,
) -> pd.DataFrame:

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(8) if cluster else None,
    )

    df.columns = (
        df.columns.astype(str)
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

    if "Module Type" in df.columns:
        null_idx = df[df["Module Type"].isna()].index

        if len(null_idx):
            first_idx = null_idx[0]
            pos = df.index.get_loc(first_idx)
            df = df.iloc[:pos]

    df = df.dropna(
        subset=[
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        how="all",
    ).reset_index(drop=True)

    df["Standard PV Efficiency (%)"] = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
        errors="coerce",
    )

    df["Total area(m2)"] = pd.to_numeric(
        df["Total area(m2)"],
        errors="coerce",
    )

    return df


# ============================================================
# CLUSTER WEIGHTS
# ============================================================

def read_cluster_weights(uploaded_file) -> Dict[str, float]:

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[12, 13, 14, 15, 16],
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        CLUSTER_WEIGHT_COLS,
        "Cluster Weights",
    )

    return {
        col: float(
            pd.to_numeric(
                df[col].iloc[0],
                errors="coerce",
            )
        )
        for col in CLUSTER_WEIGHT_COLS
    }


# ============================================================
# LATITUDE
# ============================================================

def read_latitude(uploaded_file) -> float:

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8,
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        ["Lat"],
        "Forecast Config",
    )

    lat = pd.to_numeric(
        df["Lat"].iloc[0],
        errors="coerce",
    )

    if pd.isna(lat):
        raise ValueError("Latitude is not a valid number.")

    return float(lat)


# ============================================================
# TILT LOOKUP
# ============================================================

def read_tilt_lookup(uploaded_file) -> Dict:

    try:
        uploaded_file.seek(0)

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Config Tilt Angle",
            header=7,
        )

        df.columns = (
            df.columns.astype(str)
            .str.strip()
        )

        if "Fixed" not in df.columns:
            return {}

        null_idx = df[df["Fixed"].isna()].index

        if len(null_idx):
            first_idx = null_idx[0]
            pos = df.index.get_loc(first_idx)
            df = df.iloc[:pos]

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

        return (
            df.dropna(subset=["Month"])
            .set_index("Month")["Fixed"]
            .to_dict()
        )

    except Exception:
        return {}


# ============================================================
# INPUT DATA
# ============================================================

def load_input_data(
    uploaded_file,
    cluster: bool,
) -> pd.DataFrame:

    uploaded_file.seek(0)

    sheet_name = (
        "Fixed-CL1"
        if cluster
        else "Fixed"
    )

    df = pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name,
        header=1,
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    df = clean_data_rows(df)

    validate_columns(
        df,
        ["Actual"],
        sheet_name,
    )

    if cluster:
        try:
            uploaded_file.seek(0)

            result = pd.read_excel(
                uploaded_file,
                sheet_name="Result",
                usecols=range(6),
            ).fillna(0)

            for i, col in enumerate(CLUSTER_GHI_COLS):
                if (
                    col not in df.columns
                    and i < len(result.columns)
                ):
                    values = result.iloc[
                        :len(df),
                        i,
                    ].to_numpy()

                    if len(values) < len(df):
                        values = np.pad(
                            values,
                            (
                                0,
                                len(df) - len(values),
                            ),
                            constant_values=0,
                        )

                    df[col] = values

        except Exception:
            pass

        validate_columns(
            df,
            CLUSTER_GHI_COLS,
            "Cluster Forecast",
        )

    else:
        validate_columns(
            df,
            ["GHI_Forecast"],
            "Fixed Forecast",
        )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    if cluster:
        for col in CLUSTER_GHI_COLS:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)
    else:
        df["GHI_Forecast"] = pd.to_numeric(
            df["GHI_Forecast"],
            errors="coerce",
        ).fillna(0)

    return df


# ============================================================
# SOLAR ANGLES
# ============================================================

def prepare_solar_angles(
    df: pd.DataFrame,
    lat: float,
    tilt_lookup: Optional[Dict] = None,
    tracking: bool = False,
) -> pd.DataFrame:

    result = df.copy()

    # Preserve workbook Date when available.
    # If no usable Date exists, use today's date.
    if "Date" in result.columns:
        parsed_date = pd.to_datetime(
            result["Date"],
            errors="coerce",
        )
        fallback = pd.Timestamp.today().normalize()
        result["Date"] = parsed_date.fillna(fallback)
    else:
        result["Date"] = pd.Timestamp.today().normalize()

    day_number = (
        result["Date"].dt.dayofyear
    )

    result["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360 * (284 + day_number) / 365
            )
        )
    )

    result["Elevation angle a"] = (
        90
        - lat
        + result["Declination Angle ∆"]
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
        np.radians(result["a+b"])
    )

    result["Sin(a)"] = np.sin(
        np.radians(result["Elevation angle a"])
    ).clip(lower=1e-6)

    return result


# ============================================================
# EFFICIENCY LOSS
# ============================================================

def calculate_efficiency_loss(
    df: pd.DataFrame,
    poa,
    actual,
) -> float:

    standard = numeric_series(
        df["Standard PV Efficiency (%)"]
    )

    area = numeric_series(
        df["Total area(m2)"]
    )

    poa = np.asarray(
        poa,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    valid = (
        np.isfinite(poa)
        & np.isfinite(actual)
    )

    if not np.any(valid):
        return 0.0

    valid_poa = poa[valid]
    valid_actual = actual[valid]

    poa_peak = float(np.max(valid_poa))

    if poa_peak <= 0:
        return 0.0

    actual_peak = float(np.max(valid_actual))

    base_area = float(
        np.sum(
            area * standard / 100
        )
    )

    loss_coeff = float(
        np.sum(area / 100)
    )

    if loss_coeff <= 0:
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

    return float(
        np.clip(
            loss,
            0,
            np.nanmin(standard),
        )
    )


def apply_efficiency_loss(
    df: pd.DataFrame,
    loss: float,
) -> pd.DataFrame:

    result = df.copy()

    result["Efficiency Losses(%)"] = float(loss)

    result["Net Efficiency (%)"] = (
        result["Standard PV Efficiency (%)"]
        - float(loss)
    ).clip(lower=0)

    result["Eff Area"] = (
        result["Total area(m2)"]
        * result["Net Efficiency (%)"]
        / 100
    )

    return result


# ============================================================
# BACKEND BLOCKS
# ============================================================

def read_tracking_blocks(
    uploaded_file,
    cluster: bool,
) -> np.ndarray:

    sheet_name = (
        "Backend Cal CL1"
        if cluster
        else "Backend Cal"
    )

    uploaded_file.seek(0)

    backend = pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name,
    )

    backend.columns = (
        backend.columns.astype(str)
        .str.strip()
    )

    validate_columns(
        backend,
        ["Block No."],
        sheet_name,
    )

    blocks = pd.to_numeric(
        backend["Block No."],
        errors="coerce",
    ).to_numpy(float)

    return blocks


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df: pd.DataFrame,
    input_df: pd.DataFrame,
    lat: float,
    tilt_lookup: Dict,
    cluster: bool,
) -> Tuple[np.ndarray, pd.DataFrame]:

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=False,
    )

    if cluster:
        weights = read_cluster_weights(
            st.session_state.uploaded_file
        )

        forecast = np.zeros(
            len(input_df),
            dtype=float,
        )

        for ghi_col, weight_col in zip(
            CLUSTER_GHI_COLS,
            CLUSTER_WEIGHT_COLS,
        ):
            poa = (
                solar[ghi_col]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            eff_area = (
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * weights[weight_col]
            ).sum()

            forecast += (
                poa.to_numpy(float)
                * eff_area
                / 1_000_000
            )

        return forecast, solar

    poa = (
        solar["GHI_Forecast"]
        * solar["SIN(a+b)"]
        / solar["Sin(a)"]
    )

    forecast = (
        poa.to_numpy(float)
        * df["Eff Area"].sum()
        / 1_000_000
    )

    return forecast, solar


# ============================================================
# TRACKING GEOMETRY
# ============================================================

def tracking_geometry(
    blocks,
    params: Dict,
) -> Tuple[np.ndarray, np.ndarray]:

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    DHI = int(params["DHI"])
    start = int(params["start"])
    end = int(params["end"])
    max_block = int(params["max"])
    east = int(params["east"])
    west = int(params["west"])

    if not (
        start < max_block < end
    ):
        raise ValueError(
            "Starting Block < Max Block < Ending Block is required."
        )

    d1 = start - 1 - max_block
    d2 = end + 1 - max_block

    if d1 == 0 or d2 == 0:
        raise ValueError(
            "Invalid tracking block configuration."
        )

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

    cos_alpha = np.clip(
        np.cos(
            np.radians(panel)
        ),
        1e-6,
        None,
    )

    # DHI is returned separately.
    # This is deliberate: DHI must be applied exactly once.
    dhi_factor = 1 - DHI / 100

    return cos_alpha, np.full(
        len(blocks),
        dhi_factor,
        dtype=float,
    )


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    blocks,
    weighted_ghi,
    params,
) -> np.ndarray:

    weighted_ghi = np.asarray(
        weighted_ghi,
        dtype=float,
    )

    cos_alpha, dhi_factor = tracking_geometry(
        blocks,
        params,
    )

    # IMPORTANT:
    # Error/DHI correction is applied exactly ONCE here.
    forecast = (
        weighted_ghi
        * dhi_factor
        / cos_alpha
        / 1_000_000
    )

    return np.asarray(
        forecast,
        dtype=float,
    )


# ============================================================
# TRACKING OPTIMIZER
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
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

    actual_peak = float(np.max(actual))
    actual_energy = float(np.sum(actual))

    if (
        actual_peak <= 0
        or actual_energy <= 0
    ):
        raise ValueError(
            "Actual power data is invalid."
        )

    def objective(x):

        params = {
            "DHI": int(round(x[0])),
            "start": int(round(x[1])),
            "end": int(round(x[2])),
            "max": int(round(x[3])),
            "east": int(round(x[4])),
            "west": int(round(x[5])),
        }

        if not (
            params["start"]
            < params["max"]
            < params["end"]
        ):
            return 1e9

        try:
            prediction = calculate_tracking_forecast(
                blocks,
                weighted_ghi,
                params,
            )
        except Exception:
            return 1e9

        if not np.all(
            np.isfinite(prediction)
        ):
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual - prediction
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
# WEIGHTED GHI
# ============================================================

def calculate_weighted_ghi(
    df: pd.DataFrame,
    input_df: pd.DataFrame,
    uploaded_file,
    cluster: bool,
) -> np.ndarray:

    if not cluster:
        return (
            input_df["GHI_Forecast"]
            .to_numpy(float)
            * df["Eff Area"].sum()
        )

    weights = read_cluster_weights(
        uploaded_file
    )

    weighted_ghi = np.zeros(
        len(input_df),
        dtype=float,
    )

    for ghi_col, weight_col in zip(
        CLUSTER_GHI_COLS,
        CLUSTER_WEIGHT_COLS,
    ):

        eff_area = (
            df["Total area(m2)"]
            * df["Net Efficiency (%)"]
            / 100
            * weights[weight_col]
        ).sum()

        weighted_ghi += (
            input_df[ghi_col].to_numpy(float)
            * eff_area
        )

    return weighted_ghi


# ============================================================
# INPUT DATA EDITOR
# ============================================================

def show_input_editor(
    input_df: pd.DataFrame,
    cluster: bool,
) -> pd.DataFrame:

    st.markdown(
        '<div class="section-title">📊 Input GHI & Actual Power</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Edit the forecast GHI and Actual power below. "
        "Workbook configuration, efficiency, latitude and tracking data "
        "are read automatically."
    )

    if cluster:
        edit_cols = [
            "Actual",
            *CLUSTER_GHI_COLS,
        ]
    else:
        edit_cols = [
            "GHI_Forecast",
            "Actual",
        ]

    available = [
        col
        for col in edit_cols
        if col in input_df.columns
    ]

    display_df = input_df[available].copy()

    edited = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key=f"input_editor_{st.session_state.input_editor_version}",
        height=360,
        column_config={
            col: st.column_config.NumberColumn(
                col,
                step=0.01,
                format="%.2f",
            )
            for col in available
        },
    )

    result = input_df.copy()

    for col in available:
        result[col] = pd.to_numeric(
            edited[col],
            errors="coerce",
        ).fillna(0)

    return result


# ============================================================
# TRACKING PARAMETER UI
# ============================================================

def show_tracking_parameters(
    params: Dict,
) -> Dict:

    st.markdown(
        '<div class="section-title">⚙️ Tracking Parameters</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Optimizer values are editable. "
        "Changing them recalculates the tracking forecast."
    )

    c1, c2, c3 = st.columns(3)

    dhi = c1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=10,
        value=int(params["DHI"]),
        step=1,
        key="tracking_dhi",
    )

    start = c2.number_input(
        "Starting Block",
        min_value=0,
        max_value=30,
        value=int(params["start"]),
        step=1,
        key="tracking_start",
    )

    end = c3.number_input(
        "Ending Block",
        min_value=65,
        max_value=80,
        value=int(params["end"]),
        step=1,
        key="tracking_end",
    )

    c1, c2, c3 = st.columns(3)

    max_block = c1.number_input(
        "Max Block",
        min_value=44,
        max_value=60,
        value=int(params["max"]),
        step=1,
        key="tracking_max",
    )

    east = c2.number_input(
        "East Limit",
        min_value=0,
        max_value=70,
        value=int(params["east"]),
        step=1,
        key="tracking_east",
    )

    west = c3.number_input(
        "West Limit",
        min_value=0,
        max_value=70,
        value=int(params["west"]),
        step=1,
        key="tracking_west",
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
# EFFICIENCY UI
# ============================================================

def show_efficiency_control(
    df: pd.DataFrame,
    auto_loss: float,
    key_prefix: str,
) -> pd.DataFrame:

    st.markdown(
        '<div class="section-title">📉 Efficiency Loss</div>',
        unsafe_allow_html=True,
    )

    max_loss = float(
        df[
            "Standard PV Efficiency (%)"
        ].min()
    )

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=max_loss,
        value=float(auto_loss),
        step=0.1,
        format="%.2f",
        key=f"{key_prefix}_loss",
    )

    return apply_efficiency_loss(
        df,
        loss,
    )


def show_efficiency_table(
    df: pd.DataFrame,
):

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

    display = df[cols].copy()

    numeric_cols = display.select_dtypes(
        include="number"
    ).columns

    display[numeric_cols] = (
        display[numeric_cols].round(3)
    )

    with st.expander(
        "🔍 View Efficiency Calculations",
        expanded=False,
    ):
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CHARTS
# ============================================================

def build_forecast_chart(
    forecast,
    actual,
    title: str,
    tracking: bool = False,
):

    n = min(
        len(forecast),
        len(actual),
    )

    x = np.arange(
        1,
        n + 1,
    )

    forecast = np.asarray(
        forecast[:n],
        dtype=float,
    )

    actual = np.asarray(
        actual[:n],
        dtype=float,
    )

    error = actual - forecast

    fig = go.Figure()

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
            y=error,
            mode="lines",
            name="Actual - Forecast",
            line=dict(
                color="#F59E0B",
                width=1.5,
                dash="dot",
            ),
            visible="legendonly",
        )
    )

    fig.update_layout(
        title=title,
        height=500,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            title="15 Minute Block",
            gridcolor="rgba(128,128,128,.15)",
        ),
        yaxis=dict(
            title="Power (MW)",
            gridcolor="rgba(128,128,128,.15)",
        ),
    )

    return fig


def show_forecast_chart(
    forecast,
    actual,
    title: str,
):

    fig = build_forecast_chart(
        forecast,
        actual,
        title,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"forecast_chart_{st.session_state.input_editor_version}",
    )


# ============================================================
# METRICS UI
# ============================================================

def metric_card(
    label: str,
    value: str,
):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_metrics(metrics: Dict):

    st.markdown(
        '<div class="section-title">📈 Forecast Performance</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        metric_card(
            "Actual Peak",
            f"{metrics['actual_peak']:.2f} MW",
        )

    with c2:
        metric_card(
            "Forecast Peak",
            f"{metrics['forecast_peak']:.2f} MW",
        )

    with c3:
        metric_card(
            "Peak Error",
            f"{metrics['peak_error']:.2f}%",
        )

    with c4:
        metric_card(
            "Actual Energy",
            f"{metrics['actual_energy']:.2f}",
        )

    with c5:
        metric_card(
            "Forecast Energy",
            f"{metrics['forecast_energy']:.2f}",
        )

    with c6:
        metric_card(
            "Energy Error",
            f"{metrics['energy_error']:.2f}%",
        )


# ============================================================
# MODEL EXECUTION
# ============================================================

def run_fixed_model(
    uploaded_file,
    df,
    input_df,
    lat,
    tilt_lookup,
    cluster,
):

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=False,
    )

    if cluster:
        loss_poa = (
            solar["CL1-GHI"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )
    else:
        loss_poa = (
            solar["GHI_Forecast"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

    auto_loss = calculate_efficiency_loss(
        df,
        loss_poa,
        input_df["Actual"],
    )

    df = show_efficiency_control(
        df,
        auto_loss,
        "fixed",
    )

    if cluster:
        weights = read_cluster_weights(
            uploaded_file
        )

        forecast = np.zeros(
            len(input_df),
            dtype=float,
        )

        for ghi_col, weight_col in zip(
            CLUSTER_GHI_COLS,
            CLUSTER_WEIGHT_COLS,
        ):

            poa = (
                solar[ghi_col]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            eff_area = (
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * weights[weight_col]
            ).sum()

            forecast += (
                poa.to_numpy(float)
                * eff_area
                / 1_000_000
            )

    else:
        poa = (
            solar["GHI_Forecast"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

        forecast = (
            poa.to_numpy(float)
            * df["Eff Area"].sum()
            / 1_000_000
        )

    actual = numeric_series(
        input_df["Actual"]
    )

    metrics = calculate_metrics(
        forecast,
        actual,
    )

    show_metrics(metrics)

    st.markdown(
        '<div class="section-title">📊 Forecast vs Actual</div>',
        unsafe_allow_html=True,
    )

    show_forecast_chart(
        forecast,
        actual,
        "🏗️ Fixed Forecast vs Actual",
    )

    show_efficiency_table(df)

    return {
        "forecast": forecast,
        "actual": actual,
        "metrics": metrics,
        "efficiency": df,
    }


def run_tracking_model(
    uploaded_file,
    df,
    input_df,
    lat,
    tilt_lookup,
    cluster,
):

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=True,
    )

    if cluster:
        loss_poa = (
            solar["CL1-GHI"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )
    else:
        loss_poa = (
            solar["GHI_Forecast"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

    auto_loss = calculate_efficiency_loss(
        df,
        loss_poa,
        input_df["Actual"],
    )

    df = show_efficiency_control(
        df,
        auto_loss,
        "tracking",
    )

    weighted_ghi = calculate_weighted_ghi(
        df,
        input_df,
        uploaded_file,
        cluster,
    )

    blocks = read_tracking_blocks(
        uploaded_file,
        cluster,
    )

    actual = numeric_series(
        input_df["Actual"]
    )

    n = min(
        len(blocks),
        len(weighted_ghi),
        len(actual),
    )

    blocks = blocks[:n]
    weighted_ghi = weighted_ghi[:n]
    actual = actual[:n]

    # --------------------------------------------------------
    # OPTIMIZE ONCE FOR CURRENT INPUT DATA
    # --------------------------------------------------------

    input_signature = (
        tuple(np.round(weighted_ghi, 8)),
        tuple(np.round(actual, 8)),
        tuple(np.round(blocks, 8)),
    )

    if (
        st.session_state.tracking_params is None
        or st.session_state.last_run_signature != input_signature
    ):

        with st.spinner(
            "🔄 Optimizing tracking parameters..."
        ):

            params = optimize_tracking_cached(
                tuple(blocks),
                tuple(weighted_ghi),
                tuple(actual),
            )

        st.session_state.tracking_params = params
        st.session_state.last_run_signature = input_signature

    params = show_tracking_parameters(
        st.session_state.tracking_params
    )

    if not (
        params["start"]
        < params["max"]
        < params["end"]
    ):
        st.error(
            "Starting Block must be less than Max Block, "
            "and Max Block must be less than Ending Block."
        )
        return None

    forecast = calculate_tracking_forecast(
        blocks,
        weighted_ghi,
        params,
    )

    metrics = calculate_metrics(
        forecast,
        actual,
    )

    show_metrics(metrics)

    st.markdown(
        '<div class="section-title">📊 Forecast vs Actual</div>',
        unsafe_allow_html=True,
    )

    show_forecast_chart(
        forecast,
        actual,
        "🔄 Tracking Forecast vs Actual",
    )

    st.markdown(
        '<div class="success-box">'
        '<b>Tracking correction applied once.</b><br>'
        'DHI / Error correction is applied only in the final '
        'tracking prediction step.'
        '</div>',
        unsafe_allow_html=True,
    )

    show_efficiency_table(df)

    return {
        "forecast": forecast,
        "actual": actual,
        "metrics": metrics,
        "efficiency": df,
        "tracking_params": params,
        "blocks": blocks,
    }


# ============================================================
# HEADER
# ============================================================

def show_header():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">☀️ Solar Loss Correction</div>
            <div class="hero-subtitle">
                Fixed and tracking forecast correction with editable
                GHI, actual power, efficiency losses and tracking parameters.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# WORKBOOK SUMMARY
# ============================================================

def show_workbook_summary(
    uploaded_file,
    cluster,
    lat,
):

    sheets = get_sheet_names(
        uploaded_file
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Workbook",
            st.session_state.input_file_name or "Uploaded",
        )

    with c2:
        metric_card(
            "Plant Structure",
            "Cluster" if cluster else "Single Plant",
        )

    with c3:
        metric_card(
            "Latitude",
            f"{lat:.4f}°",
        )

    with c4:
        metric_card(
            "Sheets",
            str(len(sheets)),
        )


# ============================================================
# MAIN
# ============================================================

def main():

    show_header()

    # ========================================================
    # FILE
    # ========================================================

    st.markdown(
        '<div class="section-title">📁 Input Workbook</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel workbook",
        type=["xlsx", "xls"],
        help=(
            "Upload the same workbook used by the Jupyter model. "
            "Workbook parameters are read automatically."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Upload your Excel workbook to start the model."
        )
        return

    st.session_state.uploaded_file = uploaded_file
    st.session_state.input_file_name = uploaded_file.name

    try:
        cluster = is_cluster_workbook(
            uploaded_file
        )

        lat = read_latitude(
            uploaded_file
        )

        df = read_area_efficiency(
            uploaded_file,
            cluster,
        )

        tilt_lookup = read_tilt_lookup(
            uploaded_file
        )

        if (
            st.session_state.input_df is None
            or st.session_state.input_file_name
            != uploaded_file.name
        ):
            st.session_state.input_df = load_input_data(
                uploaded_file,
                cluster,
            )
            reset_model()

    except Exception as e:
        st.error(
            f"Unable to load workbook: {e}"
        )
        return

    # ========================================================
    # WORKBOOK SUMMARY
    # ========================================================

    show_workbook_summary(
        uploaded_file,
        cluster,
        lat,
    )

    # ========================================================
    # PLANT TYPE
    # ========================================================

    st.markdown(
        '<div class="section-title">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    plant_type = st.segmented_control(
        "Select plant type",
        options=[
            "🏗️ Fixed",
            "🔄 Tracking",
        ],
        default=st.session_state.plant_type,
        selection_mode="single",
        key="plant_selector",
        label_visibility="collapsed",
        width="stretch",
    )

    if plant_type != st.session_state.plant_type:
        st.session_state.plant_type = plant_type
        reset_model()

    # ========================================================
    # INPUT DATA
    # ========================================================

    with st.container(border=True):

        edited_df = show_input_editor(
            st.session_state.input_df,
            cluster,
        )

    # Detect whether data changed.
    old_df = st.session_state.input_df

    if not edited_df.equals(old_df):
        st.session_state.input_df = edited_df
        st.session_state.input_editor_version += 1
        reset_model()

    input_df = st.session_state.input_df

    # ========================================================
    # RUN MODEL
    # ========================================================

    st.markdown("")

    run_clicked = st.button(
        "🚀 Run Loss Correction",
        type="primary",
        use_container_width=True,
        key="run_model_button",
    )

    if run_clicked:
        reset_model()

        try:

            with st.spinner(
                "Calculating loss correction..."
            ):

                if plant_type == "🏗️ Fixed":

                    result = run_fixed_model(
                        uploaded_file,
                        df,
                        input_df,
                        lat,
                        tilt_lookup,
                        cluster,
                    )

                else:

                    result = run_tracking_model(
                        uploaded_file,
                        df,
                        input_df,
                        lat,
                        tilt_lookup,
                        cluster,
                    )

            st.session_state.model_result = result

        except Exception as e:

            st.error(
                "❌ Loss correction failed."
            )
            st.exception(e)

    # ========================================================
    # RE-RENDER RESULTS
    # ========================================================

    # Streamlit reruns after widgets change. If model result
    # already exists, render the result again using current
    # editable parameters.
    if (
        st.session_state.model_result is not None
        and not run_clicked
    ):

        try:
            with st.container(border=True):

                if plant_type == "🏗️ Fixed":
                    result = run_fixed_model(
                        uploaded_file,
                        df,
                        input_df,
                        lat,
                        tilt_lookup,
                        cluster,
                    )
                else:
                    result = run_tracking_model(
                        uploaded_file,
                        df,
                        input_df,
                        lat,
                        tilt_lookup,
                        cluster,
                    )

                st.session_state.model_result = result

        except Exception as e:
            st.error(
                "❌ Unable to refresh model results."
            )
            st.exception(e)

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("")
    st.caption(
        "Solar Loss Correction Model • Fixed / Tracking • "
        "Input-driven forecast calculation"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
