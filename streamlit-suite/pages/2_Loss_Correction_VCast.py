# ============================================================
# LOSS CORRECTION MODEL
# Supports:
#   1. Fixed      -> Non-Cluster
#   2. Fixed-CL1  -> Cluster
#   3. Fixed-C11  -> VCast
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Loss Correction",
    page_icon="☀️",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

CLUSTERS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
]

VCast_GHI_COLUMNS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

VCast_FIXED_AREA_ROWS = range(2, 7)       # P3:P7
VCast_TRACKING_AREA_ROWS = range(28, 33) # P29:P33


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        opacity: 0.65;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }

    div[data-testid="stDataEditor"] {
        border-radius: 12px;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }

    details {
        border-radius: 12px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <h1 style='text-align:center;
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    font-size:40px;
    font-weight:800;'>
    ⚡ Solar Suite
    </h1>

    <p style='text-align:center;color:gray;font-size:14px'>
    Forecast Correction Platform
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {

    "loss_model_type": None,

    "uploaded_file_name": None,

    "run_model": False,

    # VCast
    "vcast_fixed_loss_auto": None,
    "vcast_fixed_loss": None,

    "vcast_tracking_params": None,

    # Current editable VCast parameters
    "vcast_dhi": None,
    "vcast_start": None,
    "vcast_end": None,
    "vcast_max": None,
    "vcast_east": None,
    "vcast_west": None,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def reset_model_state():

    keys = [
        "run_model",
        "vcast_fixed_loss_auto",
        "vcast_fixed_loss",
        "vcast_tracking_params",
        "vcast_dhi",
        "vcast_start",
        "vcast_end",
        "vcast_max",
        "vcast_east",
        "vcast_west",
    ]

    for key in keys:
        if key in st.session_state:
            st.session_state[key] = None

    st.session_state.run_model = False


def get_sheet_names(uploaded_file):

    uploaded_file.seek(0)

    xls = pd.ExcelFile(uploaded_file)

    return xls.sheet_names


def detect_workbook_type(uploaded_file):

    sheets = get_sheet_names(uploaded_file)

    # --------------------------------------------------------
    # IMPORTANT ORDER
    # --------------------------------------------------------

    if "Fixed-C11" in sheets:

        return "VCast"

    elif "Fixed-CL1" in sheets:

        return "Cluster"

    elif "Fixed" in sheets:

        return "Non-Cluster"

    return None


def validate_columns(
    df,
    required,
    name="DataFrame",
):

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} missing columns: "
            + ", ".join(missing)
        )


def clean_date_rows(
    df,
    date_col="Date",
):

    df = df.copy()

    if date_col not in df.columns:
        return df.reset_index(drop=True)

    valid = df[date_col].notna()

    if not valid.any():

        raise ValueError(
            f"No valid {date_col} rows found."
        )

    invalid_positions = np.where(
        ~valid.to_numpy()
    )[0]

    if len(invalid_positions):

        df = df.iloc[
            :invalid_positions[0]
        ]

    else:

        df = df.loc[valid]

    return df.reset_index(drop=True)


# ============================================================
# VCAST INPUT
# ============================================================

def read_vcast_input(uploaded_file):

    # --------------------------------------------------------
    # Fixed-C11
    # --------------------------------------------------------

    uploaded_file.seek(0)

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-C11",
        header=1,
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df_fix,
        [
            "Date",
            "Actual",
        ],
        "Fixed-C11",
    )

    df_fix = clean_date_rows(
        df_fix,
        "Date",
    )

    df_fix = df_fix.iloc[:96].copy()

    df_fix["Actual"] = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    uploaded_file.seek(0)

    df_ghi = pd.read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=range(6),
    )

    df_ghi.columns = [
        "Block",
        *VCast_GHI_COLUMNS,
    ]

    df_ghi["Block"] = pd.to_numeric(
        df_ghi["Block"],
        errors="coerce",
    )

    df_ghi = df_ghi.dropna(
        subset=["Block"]
    )

    for col in VCast_GHI_COLUMNS:

        df_ghi[col] = pd.to_numeric(
            df_ghi[col],
            errors="coerce",
        ).fillna(0)

    # --------------------------------------------------------
    # Align
    # --------------------------------------------------------

    n = min(
        len(df_fix),
        len(df_ghi),
    )

    df_fix = df_fix.iloc[:n].copy()
    df_ghi = df_ghi.iloc[:n].copy()

    actual = df_fix["Actual"].to_numpy(
        dtype=float
    )

    ghi_matrix = np.column_stack(
        [
            df_ghi[col].to_numpy(
                dtype=float
            )
            for col in VCast_GHI_COLUMNS
        ]
    )

    blocks = df_ghi["Block"].to_numpy(
        dtype=float
    )

    dates = pd.to_datetime(
        df_fix["Date"],
        errors="coerce",
    )

    if dates.isna().any():

        raise ValueError(
            "Invalid dates found in Fixed-C11."
        )

    return (
        df_fix,
        df_ghi,
        actual,
        ghi_matrix,
        blocks,
        dates,
    )


# ============================================================
# VCAST AREA & EFFICIENCY
# ============================================================

def read_vcast_area_efficiency(
    uploaded_file
):

    uploaded_file.seek(0)

    area_df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    area_df.columns = (
        area_df.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False,
        )
        .str.strip()
    )

    area_df = area_df[
        area_df["S.No."].notna()
    ].copy()

    area_df.reset_index(
        drop=True,
        inplace=True,
    )

    validate_columns(
        area_df,
        [
            "Standard PV Efficiency (%)",
        ],
        "Area & Efficiency",
    )

    # --------------------------------------------------------
    # Total Area
    #
    # If Total area(m2) already exists, use it.
    # Otherwise calculate it.
    # --------------------------------------------------------

    if "Total area(m2)" in area_df.columns:

        area_df["Total area(m2)"] = pd.to_numeric(
            area_df["Total area(m2)"],
            errors="coerce",
        ).fillna(0)

    else:

        validate_columns(
            area_df,
            [
                "No of Module",
                "Area of 1 Module (m2)",
            ],
            "Area & Efficiency",
        )

        area_df["Total area(m2)"] = (
            pd.to_numeric(
                area_df["No of Module"],
                errors="coerce",
            ).fillna(0)
            *
            pd.to_numeric(
                area_df["Area of 1 Module (m2)"],
                errors="coerce",
            ).fillna(0)
        )

    standard_efficiency = pd.to_numeric(
        area_df[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    standard_efficiency = (
        standard_efficiency[:5]
    )

    # --------------------------------------------------------
    # Read P3:P7 and P29:P33
    # --------------------------------------------------------

    uploaded_file.seek(0)

    raw_area = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=None,
    )

    fixed_weights = pd.to_numeric(
        raw_area.iloc[
            2:7,
            15
        ],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    tracking_weights = pd.to_numeric(
        raw_area.iloc[
            28:33,
            15
        ],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    if len(fixed_weights) != 5:

        raise ValueError(
            "Unable to read VCast Fixed effective areas P3:P7."
        )

    if len(tracking_weights) != 5:

        raise ValueError(
            "Unable to read VCast Tracking effective areas P29:P33."
        )

    return (
        area_df,
        standard_efficiency,
        fixed_weights,
        tracking_weights,
    )


# ============================================================
# VCAST LATITUDE
# ============================================================

def read_vcast_latitude(
    uploaded_file
):

    uploaded_file.seek(0)

    df_config = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8,
    )

    df_config.columns = (
        df_config.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df_config,
        ["Lat"],
        "Forecast Config",
    )

    return float(
        df_config["Lat"].iloc[0]
    )


# ============================================================
# VCAST TILT
# ============================================================

def read_vcast_tilt(
    uploaded_file
):

    uploaded_file.seek(0)

    df_tilt = pd.read_excel(
        uploaded_file,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" not in df_tilt.columns:

        return {}

    df_tilt["Month_Num"] = pd.to_numeric(
        df_tilt.get(
            "Unnamed: 2"
        ),
        errors="coerce",
    )

    df_tilt["Fixed"] = pd.to_numeric(
        df_tilt["Fixed"],
        errors="coerce",
    )

    return (
        df_tilt
        .dropna(
            subset=[
                "Month_Num",
                "Fixed",
            ]
        )
        .set_index("Month_Num")["Fixed"]
        .to_dict()
    )


# ============================================================
# VCAST SOLAR ANGLES
# ============================================================

def calculate_vcast_fixed_poa(
    dates,
    ghi_matrix,
    lat,
    tilt_lookup,
):

    # --------------------------------------------------------
    # Workbook reference date
    # --------------------------------------------------------

    first_date = pd.Timestamp(
        year=2025,
        month=1,
        day=1,
    )

    day_offset = (
        dates - first_date
    ).dt.days.to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Declination
    # --------------------------------------------------------

    declination = (
        23.45
        *
        np.sin(
            np.radians(
                360
                *
                (
                    284
                    + day_offset
                    + 1
                )
                / 365
            )
        )
    )

    # --------------------------------------------------------
    # Elevation
    # --------------------------------------------------------

    elevation = (
        90
        - lat
        + declination
    )

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    months = dates.dt.month.to_numpy()

    tilt = np.array(
        [
            tilt_lookup.get(
                float(month),
                0,
            )
            for month in months
        ]
    )

    a_plus_b = (
        elevation
        + tilt
    )

    sin_a = np.sin(
        np.radians(
            elevation
        )
    )

    sin_ab = np.sin(
        np.radians(
            a_plus_b
        )
    )

    sin_a_safe = np.where(
        np.abs(sin_a) < 1e-8,
        1e-8,
        sin_a,
    )

    fixed_poa = (
        ghi_matrix
        *
        sin_ab[:, None]
        /
        sin_a_safe[:, None]
    )

    return fixed_poa


# ============================================================
# VCAST FIXED OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False
)
def optimize_vcast_fixed(
    actual_tuple,
    fixed_poa_tuple,
    standard_efficiency_tuple,
    fixed_weights_tuple,
):

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    fixed_poa = np.asarray(
        fixed_poa_tuple,
        dtype=float,
    )

    standard_efficiency = np.asarray(
        standard_efficiency_tuple,
        dtype=float,
    )

    fixed_weights = np.asarray(
        fixed_weights_tuple,
        dtype=float,
    )

    valid_mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    if not valid_mask.any():

        raise ValueError(
            "Actual power contains no valid non-zero values."
        )

    actual_day = actual[
        valid_mask
    ]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    max_loss = np.min(
        standard_efficiency
    )

    results = []

    for loss in np.arange(
        0,
        max_loss + 0.0001,
        0.1,
    ):

        net_efficiency = np.maximum(
            standard_efficiency - loss,
            0,
        )

        efficiency_factor = np.divide(
            net_efficiency,
            standard_efficiency,
            out=np.zeros_like(
                net_efficiency
            ),
            where=(
                standard_efficiency != 0
            ),
        )

        adjusted_weights = (
            fixed_weights
            * efficiency_factor
        )

        power_matrix = (
            fixed_poa
            *
            adjusted_weights[None, :]
            / 1_000_000
        )

        predicted = (
            power_matrix.sum(axis=1)
        )

        predicted_day = predicted[
            valid_mask
        ]

        if len(predicted_day) == 0:
            continue

        predicted_peak = (
            predicted_day.max()
        )

        peak_error = abs(
            actual_peak
            - predicted_peak
        )

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    - predicted_day
                )
            )
            / actual_peak
        )

        predicted_energy = (
            predicted_day.sum()
        )

        energy_error = abs(
            actual_energy
            - predicted_energy
        ) / actual_energy

        score = (
            0.80 * block_error
            +
            0.10 * (
                peak_error
                / actual_peak
            )
            +
            0.10 * energy_error
        )

        results.append(
            {
                "Error %": loss,
                "Actual Peak": actual_peak,
                "Predicted Peak": predicted_peak,
                "Peak Error": peak_error,
                "Peak Error (%)": (
                    peak_error
                    / actual_peak
                    * 100
                ),
                "Block Error": block_error,
                "Energy Error": energy_error,
                "Overall Score": score,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise ValueError(
            "Fixed VCast optimization produced no results."
        )

    # IMPORTANT:
    # Your reference chooses minimum Peak Error.
    best_row = results_df.loc[
        results_df[
            "Peak Error"
        ].idxmin()
    ]

    return (
        float(best_row["Error %"]),
        results_df,
    )


# ============================================================
# VCAST FIXED FORECAST
# ============================================================

def calculate_vcast_fixed_forecast(
    fixed_poa,
    standard_efficiency,
    fixed_weights,
    loss,
):

    net_efficiency = np.maximum(
        standard_efficiency - loss,
        0,
    )

    efficiency_factor = np.divide(
        net_efficiency,
        standard_efficiency,
        out=np.zeros_like(
            net_efficiency
        ),
        where=(
            standard_efficiency != 0
        ),
    )

    final_weights = (
        fixed_weights
        * efficiency_factor
    )

    power_matrix = (
        fixed_poa
        *
        final_weights[None, :]
        / 1_000_000
    )

    forecast = (
        power_matrix.sum(axis=1)
    )

    return (
        forecast,
        power_matrix,
        final_weights,
        net_efficiency,
    )


# ============================================================
# VCAST TRACKING CALCULATION
# ============================================================

def calculate_vcast_tracking(
    blocks,
    ghi_matrix,
    tracking_weights,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
):

    if not (
        start_block
        < max_block
        < end_block
    ):

        raise ValueError(
            "Tracking blocks must satisfy "
            "Starting < Max < Ending."
        )

    denominator_1 = (
        start_block
        - 1
        - max_block
    )

    denominator_2 = (
        end_block
        + 1
        - max_block
    )

    if (
        denominator_1 == 0
        or denominator_2 == 0
    ):

        raise ValueError(
            "Invalid tracking block configuration."
        )

    m1 = (
        90
        / denominator_1
    )

    m2 = (
        90
        / denominator_2
    )

    zenith = np.where(

        blocks <= max_block,

        np.minimum(
            89,
            m1
            *
            (
                blocks
                - max_block
            ),
        ),

        np.minimum(
            89,
            m2
            *
            (
                blocks
                - max_block
            ),
        ),
    )

    panel = np.where(

        blocks < max_block,

        np.where(
            zenith < abs(east_limit),
            zenith,
            abs(east_limit),
        ),

        np.where(
            (
                (blocks > max_block)
                &
                (zenith > west_limit)
            ),
            west_limit,
            zenith,
        ),
    )

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None,
    )

    dhi = (
        ghi_matrix
        *
        DHI
        / 100
    )

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    # IMPORTANT:
    # Original Tracking areas.
    # Fixed loss is NOT applied here.
    tracking_power_matrix = (
        dni
        *
        tracking_weights[None, :]
        / 1_000_000
    )

    forecast = (
        tracking_power_matrix.sum(
            axis=1
        )
    )

    return {
        "forecast": forecast,
        "power_matrix": tracking_power_matrix,
        "zenith": zenith,
        "panel": panel,
        "dni": dni,
    }


# ============================================================
# VCAST TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False
)
def optimize_vcast_tracking(
    actual_tuple,
    ghi_matrix_tuple,
    blocks_tuple,
    tracking_weights_tuple,
):

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    ghi_matrix = np.asarray(
        ghi_matrix_tuple,
        dtype=float,
    )

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    tracking_weights = np.asarray(
        tracking_weights_tuple,
        dtype=float,
    )

    valid_mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    actual_day = actual[
        valid_mask
    ]

    if len(actual_day) == 0:

        raise ValueError(
            "No valid Actual values for tracking."
        )

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    def objective(x):

        DHI = int(round(x[0]))
        start_block = int(round(x[1]))
        end_block = int(round(x[2]))
        max_block = int(round(x[3]))
        east_limit = int(round(x[4]))
        west_limit = int(round(x[5]))

        if not (
            start_block
            < max_block
            < end_block
        ):
            return 1e9

        try:

            result = calculate_vcast_tracking(
                blocks,
                ghi_matrix,
                tracking_weights,
                DHI,
                start_block,
                end_block,
                max_block,
                east_limit,
                west_limit,
            )

        except Exception:

            return 1e9

        prediction = result[
            "forecast"
        ]

        if not np.all(
            np.isfinite(prediction)
        ):
            return 1e9

        prediction_day = prediction[
            valid_mask
        ]

        if len(prediction_day) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    - prediction_day
                )
            )
            / actual_peak
        )

        peak_error = (
            abs(
                actual_peak
                - prediction_day.max()
            )
            / actual_peak
        )

        energy_error = (
            abs(
                actual_energy
                - prediction_day.sum()
            )
            / actual_energy
        )

        return (
            0.80 * block_error
            +
            0.10 * peak_error
            +
            0.10 * energy_error
        )

    result = differential_evolution(

        objective,

        bounds=[
            (0, 10),
            (10, 30),
            (65, 80),
            (47, 53),
            (10, 70),
            (10, 70),
        ],

        strategy="best1bin",
        maxiter=40,
        popsize=15,
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    best = np.rint(
        result.x
    ).astype(int)

    params = {
        "DHI": int(best[0]),
        "start": int(best[1]),
        "end": int(best[2]),
        "max": int(best[3]),
        "east": int(best[4]),
        "west": int(best[5]),
        "score": float(
            objective(best)
        ),
    }

    return params


# ============================================================
# VCAST METRICS
# ============================================================

def calculate_metrics(
    actual,
    forecast,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        np.isfinite(forecast)
        &
        (actual != 0)
    )

    act = actual[mask]
    pred = forecast[mask]

    if len(act) == 0:

        return {
            "block_error": np.nan,
            "peak_error": np.nan,
            "energy_error": np.nan,
            "score": np.nan,
            "actual_peak": np.nan,
            "forecast_peak": np.nan,
        }

    actual_peak = act.max()

    forecast_peak = pred.max()

    block_error = (
        np.mean(
            np.abs(
                act - pred
            )
        )
        / actual_peak
    )

    peak_error = (
        abs(
            actual_peak
            - forecast_peak
        )
        / actual_peak
    )

    energy_error = (
        abs(
            act.sum()
            - pred.sum()
        )
        / act.sum()
    )

    score = (
        0.80 * block_error
        +
        0.10 * peak_error
        +
        0.10 * energy_error
    )

    return {
        "block_error": block_error,
        "peak_error": peak_error,
        "energy_error": energy_error,
        "score": score,
        "actual_peak": actual_peak,
        "forecast_peak": forecast_peak,
    }


# ============================================================
# VCAST PAGE
# ============================================================

def run_vcast(
    uploaded_file
):

    st.markdown(
        "### 🏗️ VCast Loss Correction"
    )

    # --------------------------------------------------------
    # Read workbook
    # --------------------------------------------------------

    (
        df_fix,
        df_ghi,
        actual,
        ghi_matrix,
        blocks,
        dates,
    ) = read_vcast_input(
        uploaded_file
    )

    (
        area_df,
        standard_efficiency,
        fixed_weights,
        tracking_weights,
    ) = read_vcast_area_efficiency(
        uploaded_file
    )

    lat = read_vcast_latitude(
        uploaded_file
    )

    tilt_lookup = read_vcast_tilt(
        uploaded_file
    )

    # --------------------------------------------------------
    # Input editor
    #
    # Keep it limited to the VCast GHI + Actual values.
    # --------------------------------------------------------

    editor_df = df_ghi[
        VCast_GHI_COLUMNS
    ].copy()

    editor_df["Actual"] = actual

    edited_df = st.data_editor(

        editor_df,

        use_container_width=True,

        hide_index=True,

        num_rows="fixed",

        key="vcast_editor",

        column_config={
            col: st.column_config.NumberColumn(
                col,
                format="%.3f",
            )
            for col in (
                VCast_GHI_COLUMNS
                + ["Actual"]
            )
        },
    )

    for col in VCast_GHI_COLUMNS:

        edited_df[col] = pd.to_numeric(
            edited_df[col],
            errors="coerce",
        ).fillna(0)

    edited_df["Actual"] = pd.to_numeric(
        edited_df["Actual"],
        errors="coerce",
    ).fillna(0)

    actual = edited_df[
        "Actual"
    ].to_numpy(float)

    ghi_matrix = np.column_stack(
        [
            edited_df[col].to_numpy(float)
            for col in VCast_GHI_COLUMNS
        ]
    )

    # --------------------------------------------------------
    # Model selector
    # --------------------------------------------------------

    plant_mode = st.radio(
        "Calculation Type",
        [
            "Fixed",
            "Tracking",
        ],
        horizontal=True,
        key="vcast_mode",
    )

    # ========================================================
    # FIXED
    # ========================================================

    if plant_mode == "Fixed":

        fixed_poa = calculate_vcast_fixed_poa(
            dates,
            ghi_matrix,
            lat,
            tilt_lookup,
        )

        # ----------------------------------------------------
        # Automatic loss optimization
        # ----------------------------------------------------

        if (
            st.session_state
            .vcast_fixed_loss_auto
            is None
        ):

            with st.spinner(
                "🔄 Optimizing Fixed efficiency loss..."
            ):

                (
                    auto_loss,
                    fixed_results,
                ) = optimize_vcast_fixed(

                    tuple(actual),

                    tuple(
                        fixed_poa.ravel()
                    ),

                    tuple(
                        standard_efficiency
                    ),

                    tuple(
                        fixed_weights
                    ),
                )

            st.session_state.vcast_fixed_loss_auto = (
                auto_loss
            )

            st.session_state.vcast_fixed_loss = (
                auto_loss
            )

            st.session_state.vcast_fixed_results = (
                fixed_results
            )

        else:

            auto_loss = (
                st.session_state
                .vcast_fixed_loss_auto
            )

            fixed_results = (
                st.session_state
                .vcast_fixed_results
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # reshape cached POA
        # ----------------------------------------------------

        fixed_poa = np.asarray(
            fixed_poa,
            dtype=float,
        )

        # ----------------------------------------------------
        # User-adjustable loss
        # ----------------------------------------------------

        st.markdown(
            "### 📉 Efficiency Loss"
        )

        loss = st.number_input(
            "Efficiency Loss (%)",
            min_value=0.0,
            max_value=float(
                np.min(
                    standard_efficiency
                )
            ),
            value=float(
                st.session_state
                .vcast_fixed_loss
            ),
            step=0.1,
            format="%.2f",
            key="vcast_fixed_loss",
        )

        forecast, power_matrix, final_weights, net_efficiency = (
            calculate_vcast_fixed_forecast(

                fixed_poa,

                standard_efficiency,

                fixed_weights,

                loss,
            )
        )

        metrics = calculate_metrics(
            actual,
            forecast,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Auto Loss",
            f"{auto_loss:.2f}%",
        )

        c2.metric(
            "Current Loss",
            f"{loss:.2f}%",
        )

        c3.metric(
            "Peak Error",
            f"{metrics['peak_error'] * 100:.2f}%",
        )

        c4.metric(
            "Overall Score",
            f"{metrics['score']:.4f}",
        )

        # ----------------------------------------------------
        # Efficiency table
        # ----------------------------------------------------

        display_df = area_df.copy()

        display_df[
            "Efficiency Losses(%)"
        ] = loss

        display_df[
            "Net Efficiency (%)"
        ] = net_efficiency

        display_df[
            "Eff Area"
        ] = (
            display_df[
                "Total area(m2)"
            ]
            *
            net_efficiency
            / 100
        )

        cols = [
            col
            for col in [
                "Module Type",
                "Standard PV Efficiency (%)",
                "Efficiency Losses(%)",
                "Net Efficiency (%)",
                "Total area(m2)",
                "Eff Area",
            ]
            if col in display_df.columns
        ]

        with st.expander(
            "🔍 View Efficiency Calculations"
        ):

            st.dataframe(
                display_df[cols].round(4),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Fixed cluster output
        # ----------------------------------------------------

        fixed_output = pd.DataFrame(
            {
                f"{cl}_Fixed Power=I*Ƞ*A":
                    power_matrix[:, i]
                for i, cl in enumerate(
                    CLUSTERS
                )
            }
        )

        fixed_output[
            "Total Fixed Power"
        ] = forecast

        with st.expander(
            "📊 Fixed Power by Cluster"
        ):

            st.dataframe(
                fixed_output.round(6),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Chart
        # ----------------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=blocks,
                y=forecast,
                mode="lines",
                name="Fixed Forecast",
                line=dict(
                    color="#2563EB",
                    width=3,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=blocks,
                y=actual,
                mode="lines",
                name="Actual",
                line=dict(
                    color="#DC2626",
                    width=3,
                ),
            )
        )

        fig.update_layout(
            title="Fixed Forecast vs Actual",
            height=500,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h",
                y=1.08,
                x=0,
            ),
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
            xaxis_title="Block",
            yaxis_title="Power (MW)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

        return

    # ========================================================
    # TRACKING
    # ========================================================

    if plant_mode == "Tracking":

        # ----------------------------------------------------
        # Optimize Tracking only once
        # ----------------------------------------------------

        if (
            st.session_state
            .vcast_tracking_params
            is None
        ):

            with st.spinner(
                "🔄 Optimizing Tracking parameters..."
            ):

                tracking_params = (
                    optimize_vcast_tracking(

                        tuple(actual),

                        tuple(
                            ghi_matrix.ravel()
                        ),

                        tuple(blocks),

                        tuple(tracking_weights),
                    )
                )

            st.session_state.vcast_tracking_params = (
                tracking_params
            )

            # ------------------------------------------------
            # Automatically populate UI state
            # ------------------------------------------------

            st.session_state.vcast_dhi = (
                tracking_params["DHI"]
            )

            st.session_state.vcast_start = (
                tracking_params["start"]
            )

            st.session_state.vcast_end = (
                tracking_params["end"]
            )

            st.session_state.vcast_max = (
                tracking_params["max"]
            )

            st.session_state.vcast_east = (
                tracking_params["east"]
            )

            st.session_state.vcast_west = (
                tracking_params["west"]
            )

        params = (
            st.session_state
            .vcast_tracking_params
        )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        st.markdown(
            "### ⚙️ Tracking Parameters"
        )

        c1, c2, c3 = st.columns(3)

        DHI = c1.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            step=1,
            value=int(
                st.session_state
                .vcast_dhi
            ),
            key="vcast_dhi",
        )

        start_block = c2.number_input(
            "GHI Starting Block",
            min_value=1,
            max_value=96,
            step=1,
            value=int(
                st.session_state
                .vcast_start
            ),
            key="vcast_start",
        )

        end_block = c3.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            step=1,
            value=int(
                st.session_state
                .vcast_end
            ),
            key="vcast_end",
        )

        c1, c2, c3 = st.columns(3)

        max_block = c1.number_input(
            "GHI Max Block",
            min_value=1,
            max_value=96,
            step=1,
            value=int(
                st.session_state
                .vcast_max
            ),
            key="vcast_max",
        )

        east_limit = c2.number_input(
            "East Tracking Limit",
            min_value=0,
            max_value=70,
            step=1,
            value=int(
                st.session_state
                .vcast_east
            ),
            key="vcast_east",
        )

        west_limit = c3.number_input(
            "West Tracking Limit",
            min_value=0,
            max_value=70,
            step=1,
            value=int(
                st.session_state
                .vcast_west
            ),
            key="vcast_west",
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if not (
            start_block
            < max_block
            < end_block
        ):

            st.warning(
                "GHI Starting Block must be smaller than "
                "GHI Max Block, which must be smaller than "
                "GHI Ending Block."
            )

            st.stop()

        # ----------------------------------------------------
        # Final tracking calculation
        # ----------------------------------------------------

        tracking = calculate_vcast_tracking(

            blocks,

            ghi_matrix,

            tracking_weights,

            DHI,

            start_block,

            end_block,

            max_block,

            east_limit,

            west_limit,
        )

        forecast = tracking[
            "forecast"
        ]

        metrics = calculate_metrics(
            actual,
            forecast,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Optimization Score",
            f"{params['score']:.4f}",
        )

        c2.metric(
            "Peak Error",
            f"{metrics['peak_error'] * 100:.2f}%",
        )

        c3.metric(
            "Energy Error",
            f"{metrics['energy_error'] * 100:.2f}%",
        )

        c4.metric(
            "Peak Power",
            f"{metrics['forecast_peak']:.4f}",
        )

        # ----------------------------------------------------
        # Tracking important blocks
        # ----------------------------------------------------

        tracking_lookup = pd.DataFrame(
            {
                "Parameter": [
                    "GHI Starting Block",
                    "GHI Ending Block",
                    "GHI Max Block",
                ],
                "Block": [
                    start_block,
                    end_block,
                    max_block,
                ],
            }
        )

        # ----------------------------------------------------
        # Time blocks
        # ----------------------------------------------------

        def block_to_time(block):

            if block < 1 or block > 96:
                return "—"

            start_min = (
                (block - 1)
                * 15
            )

            end_min = (
                block
                * 15
            )

            start_h = start_min // 60
            start_m = start_min % 60

            end_h = end_min // 60
            end_m = end_min % 60

            return (
                f"{start_h:02d}:{start_m:02d}"
                f" - "
                f"{end_h:02d}:{end_m:02d}"
            )

        tracking_lookup[
            "Time Block"
        ] = tracking_lookup[
            "Block"
        ].map(
            block_to_time
        )

        with st.expander(
            "📅 Important Time Blocks"
        ):

            st.dataframe(
                tracking_lookup,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Tracking calculations
        # ----------------------------------------------------

        tracking_df = pd.DataFrame(
            {
                "Block": blocks,
                "Zenith Angle":
                    tracking["zenith"],
                "Panel Angle":
                    tracking["panel"],
                "Tracking Forecast":
                    forecast,
                "Actual":
                    actual,
            }
        )

        # ----------------------------------------------------
        # Cluster power
        # ----------------------------------------------------

        for i, cl in enumerate(
            CLUSTERS
        ):

            tracking_df[
                f"{cl}_Tracking Power=I*Ƞ*A"
            ] = tracking[
                "power_matrix"
            ][:, i]

        with st.expander(
            "🔍 View Tracking Calculations"
        ):

            st.dataframe(
                tracking_df.round(6),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Chart
        # ----------------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=blocks,
                y=forecast,
                mode="lines",
                name="Tracking Forecast",
                line=dict(
                    color="#2563EB",
                    width=3,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=blocks,
                y=actual,
                mode="lines",
                name="Actual",
                line=dict(
                    color="#DC2626",
                    width=3,
                ),
            )
        )

        fig.update_layout(
            title="Tracking Forecast vs Actual",
            height=500,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h",
                y=1.08,
                x=0,
            ),
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
            xaxis_title="Block",
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
# MAIN
# ============================================================

st.markdown(
    '<div class="main-title">☀️ Loss Correction Model</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Upload your solar workbook and optimize Fixed or Tracking loss correction."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Plant Excel Workbook",
    type=[
        "xlsx",
        "xls",
    ],
    key="loss_correction_uploader",
)


if uploaded_file is None:

    st.info(
        "👆 Upload your Excel workbook to start."
    )

    st.stop()


# ============================================================
# DETECT WORKBOOK
# ============================================================

try:

    workbook_type = detect_workbook_type(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


if workbook_type is None:

    st.error(
        """
        ❌ Unsupported workbook.

        The workbook must contain one of:

        • `Fixed` → Non-Cluster  
        • `Fixed-CL1` → Cluster  
        • `Fixed-C11` → VCast
        """
    )

    st.stop()


# ============================================================
# RESET WHEN NEW FILE IS UPLOADED
# ============================================================

if (
    st.session_state.uploaded_file_name
    != uploaded_file.name
):

    reset_model_state()

    st.session_state.uploaded_file_name = (
        uploaded_file.name
    )

    st.session_state.loss_model_type = (
        workbook_type
    )


# ============================================================
# WORKBOOK INFORMATION
# ============================================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "Workbook",
    uploaded_file.name,
)

c2.metric(
    "Detected Type",
    workbook_type,
)

c3.metric(
    "Calculation",
    "Fixed / Tracking",
)


# ============================================================
# SHEETS
# ============================================================

sheets = get_sheet_names(
    uploaded_file
)

with st.expander(
    "📑 Workbook Information"
):

    st.dataframe(
        pd.DataFrame(
            {
                "Available Sheets": sheets
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MODEL
# ============================================================

if workbook_type == "VCast":

    run_vcast(
        uploaded_file
    )

else:

    # ========================================================
    # IMPORTANT
    #
    # Keep your EXISTING Non-Cluster and Cluster calculation
    # functions here.
    #
    # Do NOT route these workbooks through the VCast model.
    # ========================================================

    st.info(
        f"""
        `{workbook_type}` workbook detected.

        This workbook should continue using the existing
        {workbook_type} Fixed/Tracking calculation module.

        VCast calculation is only used when `Fixed-C11`
        exists in the workbook.
        """
    )
