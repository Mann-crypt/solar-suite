# ============================================================
# STREAMLIT APP
# LOSS CORRECTION MODEL
# ============================================================

import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loss Correction Model",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
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

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

N_CLUSTERS = 5

TRACKING_BOUNDS = [
    (0, 10),       # DHI %
    (10, 30),      # Starting Block
    (65, 80),      # Ending Block
    (47, 53),      # Max Block
    (10, 70),      # East Limit
    (10, 70),      # West Limit
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #8b949e;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 1.20rem;
        font-weight: 650;
        margin: 18px 0 10px 0;
    }

    div.stButton > button {
        min-height: 50px;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 650;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 10px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "file_signature": None,
    "workbook_cache": None,
    "input_editor_data": None,
    "run_model": False,
    "plant_type": "🏗️ Fixed",
    "tracking_params": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GENERAL HELPERS
# ============================================================

def validate_columns(df, required, name="Data"):

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} is missing: "
            f"{', '.join(missing)}"
        )


def get_file_signature(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    digest = hashlib.md5(
        file_bytes
    ).hexdigest()

    return (
        uploaded_file.name,
        uploaded_file.size,
        digest,
    )


def clean_data_rows(
    df,
    date_column="Date",
):

    df = df.copy()

    if date_column in df.columns:

        valid = df[date_column].notna()

        if valid.any():

            first_blank = np.where(
                ~valid.to_numpy()
            )[0]

            if len(first_blank):

                df = df.iloc[
                    :first_blank[0]
                ]

            else:

                df = df.loc[valid]

    return df.reset_index(
        drop=True
    )


# ============================================================
# SHEET NAMES
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def get_sheet_names_cached(
    file_bytes,
):

    return tuple(
        pd.ExcelFile(
            BytesIO(file_bytes)
        ).sheet_names
    )


# ============================================================
# AREA & EFFICIENCY
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_area_efficiency(
    file_bytes,
):

    df = pd.read_excel(
        BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False,
        )
        .str.strip()
    )

    validate_columns(
        df,
        [
            "S.No.",
            "No of Module",
            "Area of 1 Module (m2)",
            "Standard PV Efficiency (%)",
        ],
        "Area & Efficiency",
    )

    df = df[
        df["S.No."].notna()
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True,
    )

    df["No of Module"] = pd.to_numeric(
        df["No of Module"],
        errors="coerce",
    ).fillna(0)

    df["Area of 1 Module (m2)"] = pd.to_numeric(
        df["Area of 1 Module (m2)"],
        errors="coerce",
    ).fillna(0)

    df["Total area (m2)"] = (
        df["No of Module"]
        *
        df["Area of 1 Module (m2)"]
    )

    df["Standard PV Efficiency (%)"] = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
        errors="coerce",
    )

    if len(df) < N_CLUSTERS:

        raise ValueError(
            "Area & Efficiency contains fewer "
            "than 5 cluster rows."
        )

    return df.reset_index(
        drop=True
    )


# ============================================================
# EFFECTIVE AREAS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_effective_areas(
    file_bytes,
):

    area_raw = pd.read_excel(
        BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=None,
    )

    if (
        area_raw.shape[0] < 33
        or area_raw.shape[1] < 16
    ):

        raise ValueError(
            "Area & Efficiency does not contain "
            "the expected effective-area cells."
        )

    # Excel P3:P7
    fixed_weights = (
        pd.to_numeric(
            area_raw.iloc[
                2:7,
                15,
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    # Excel P29:P33
    tracking_weights = (
        pd.to_numeric(
            area_raw.iloc[
                28:33,
                15,
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    if len(fixed_weights) != N_CLUSTERS:

        raise ValueError(
            "Could not read 5 Fixed effective "
            "areas from P3:P7."
        )

    if len(tracking_weights) != N_CLUSTERS:

        raise ValueError(
            "Could not read 5 Tracking effective "
            "areas from P29:P33."
        )

    return (
        fixed_weights,
        tracking_weights,
    )


# ============================================================
# LATITUDE
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_latitude(
    file_bytes,
):

    df = pd.read_excel(
        BytesIO(file_bytes),
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

    lat = pd.to_numeric(
        df["Lat"],
        errors="coerce",
    ).dropna()

    if lat.empty:

        raise ValueError(
            "Latitude could not be read."
        )

    return float(
        lat.iloc[0]
    )


# ============================================================
# TILT LOOKUP
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_tilt_lookup(
    file_bytes,
):

    try:

        df = pd.read_excel(
            BytesIO(file_bytes),
            sheet_name="Config Tilt Angle",
            header=7,
        )

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        df = df.rename(
            columns={
                "Unnamed: 2": "Month_Num",
                "Unnamed: 3": "Month",
            }
        )

        if "Fixed" not in df.columns:

            return {}

        if "Month_Num" not in df.columns:

            return {}

        df["Month_Num"] = pd.to_numeric(
            df["Month_Num"],
            errors="coerce",
        )

        df["Fixed"] = pd.to_numeric(
            df["Fixed"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "Month_Num",
                "Fixed",
            ]
        )

        return {
            float(k): float(v)
            for k, v in
            df.set_index(
                "Month_Num"
            )["Fixed"].to_dict().items()
        }

    except Exception:

        return {}


# ============================================================
# RESULT GHI
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_result_ghi(
    file_bytes,
):

    df = pd.read_excel(
        BytesIO(file_bytes),
        sheet_name="Result",
        usecols=range(6),
    )

    if len(df.columns) < 6:

        raise ValueError(
            "Result sheet must contain Block "
            "and five GHI columns."
        )

    df.columns = [
        "Block",
        *GHI_COLS,
    ]

    df["Block"] = pd.to_numeric(
        df["Block"],
        errors="coerce",
    )

    df = df[
        df["Block"].notna()
    ].copy()

    for col in GHI_COLS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

    return df.reset_index(
        drop=True
    )


# ============================================================
# FIXED-C11 INPUT
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def load_input_data(
    file_bytes,
):

    df = pd.read_excel(
        BytesIO(file_bytes),
        sheet_name="Fixed-C11",
        header=1,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        [
            "Date",
            "Actual",
        ],
        "Fixed-C11",
    )

    df = clean_data_rows(
        df,
        "Date",
    )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    return df.reset_index(
        drop=True
    )


# ============================================================
# COMPLETE WORKBOOK LOADER
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def load_workbook(
    file_bytes,
):

    sheets = get_sheet_names_cached(
        file_bytes
    )

    required_sheets = [
        "Area & Efficiency",
        "Forecast Config",
        "Config Tilt Angle",
        "Result",
        "Fixed-C11",
        "Tracking",
        "Backend Cal C11",
        "Backend Cal C12",
        "Backend Cal C13",
        "Backend Cal C14",
        "Backend Cal C15",
    ]

    missing = [
        s
        for s in required_sheets
        if s not in sheets
    ]

    if missing:

        raise ValueError(
            "Required sheets are missing: "
            + ", ".join(missing)
        )

    area_df = read_area_efficiency(
        file_bytes
    )

    (
        fixed_weights,
        tracking_weights,
    ) = read_effective_areas(
        file_bytes
    )

    lat = read_latitude(
        file_bytes
    )

    tilt_lookup = read_tilt_lookup(
        file_bytes
    )

    result_df = read_result_ghi(
        file_bytes
    )

    input_raw = load_input_data(
        file_bytes
    )

    return {
        "area_df": area_df,
        "fixed_weights": fixed_weights,
        "tracking_weights": tracking_weights,
        "lat": lat,
        "tilt_lookup": tilt_lookup,
        "result_df": result_df,
        "input_raw": input_raw,
    }


# ============================================================
# INPUT EDITOR DATA
# ============================================================

def build_input_editor_data(
    input_raw,
    result_df,
):

    n = min(
        len(input_raw),
        len(result_df),
    )

    if n == 0:

        raise ValueError(
            "No valid data rows found."
        )

    display = pd.DataFrame({

        "Date":
            input_raw[
                "Date"
            ]
            .iloc[:n]
            .values,

        "GHI C11":
            result_df[
                "GHI C11"
            ]
            .iloc[:n]
            .values,

        "GHI C12":
            result_df[
                "GHI C12"
            ]
            .iloc[:n]
            .values,

        "GHI C13":
            result_df[
                "GHI C13"
            ]
            .iloc[:n]
            .values,

        "GHI C14":
            result_df[
                "GHI C14"
            ]
            .iloc[:n]
            .values,

        "GHI C15":
            result_df[
                "GHI C15"
            ]
            .iloc[:n]
            .values,

        "Actual":
            input_raw[
                "Actual"
            ]
            .iloc[:n]
            .values,
    })

    display["Date"] = pd.to_datetime(
        display["Date"],
        errors="coerce",
    )

    return display


# ============================================================
# INPUT DATA EDITOR
# ============================================================

def input_data_editor(
    df,
):

    st.markdown(
        '<div class="section-title">'
        '📊 Input GHI and Power'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "GHI values are loaded from Result and "
        "Actual power is loaded from Fixed-C11. "
        "Modify values before running if required."
    )

    edited = st.data_editor(

        df,

        use_container_width=True,

        hide_index=True,

        num_rows="fixed",

        key="input_editor",

        height=500,

        column_config={

            "Date":
                st.column_config.DateColumn(
                    "Date",
                    disabled=True,
                ),

            "GHI C11":
                st.column_config.NumberColumn(
                    "GHI C11",
                    step=0.01,
                    format="%.2f",
                ),

            "GHI C12":
                st.column_config.NumberColumn(
                    "GHI C12",
                    step=0.01,
                    format="%.2f",
                ),

            "GHI C13":
                st.column_config.NumberColumn(
                    "GHI C13",
                    step=0.01,
                    format="%.2f",
                ),

            "GHI C14":
                st.column_config.NumberColumn(
                    "GHI C14",
                    step=0.01,
                    format="%.2f",
                ),

            "GHI C15":
                st.column_config.NumberColumn(
                    "GHI C15",
                    step=0.01,
                    format="%.2f",
                ),

            "Actual":
                st.column_config.NumberColumn(
                    "Actual",
                    step=0.01,
                    format="%.4f",
                ),
        },
    )

    result = edited.copy()

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    for col in GHI_COLS + ["Actual"]:

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce",
        ).fillna(0)

    return result


# ============================================================
# SOLAR CALCULATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def prepare_solar_data(
    dates_tuple,
    lat,
    tilt_items,
):

    dates = pd.to_datetime(
        list(dates_tuple),
        errors="coerce",
    )

    if dates.isna().any():

        raise ValueError(
            "Invalid dates found in Fixed-C11."
        )

    first_date = pd.Timestamp(
        year=2025,
        month=1,
        day=1,
    )

    day_offset = (
        dates
        - first_date
    ).dt.days.to_numpy(
        dtype=float
    )

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

    elevation = (
        90
        - lat
        + declination
    )

    months = (
        dates
        .dt.month
        .to_numpy()
    )

    tilt_lookup = dict(
        tilt_items
    )

    tilt = np.array([
        tilt_lookup.get(
            float(month),
            0,
        )
        for month in months
    ])

    a_plus_b = (
        elevation
        + tilt
    )

    sin_a = np.sin(
        np.radians(
            elevation
        )
    )

    sin_a_safe = np.where(
        np.abs(sin_a) < 1e-8,
        1e-8,
        sin_a,
    )

    sin_ab = np.sin(
        np.radians(
            a_plus_b
        )
    )

    return {

        "declination":
            declination,

        "elevation":
            elevation,

        "tilt":
            tilt,

        "sin_a":
            sin_a_safe,

        "sin_ab":
            sin_ab,
    }


# ============================================================
# FIXED LOSS OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_fixed_loss(
    standard_efficiency_tuple,
    fixed_weights_tuple,
    fixed_poa_tuple,
    actual_tuple,
):

    standard_efficiency = np.asarray(
        standard_efficiency_tuple,
        dtype=float,
    )

    fixed_weights = np.asarray(
        fixed_weights_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    fixed_poa = np.asarray(
        fixed_poa_tuple,
        dtype=float,
    )

    fixed_poa = fixed_poa.reshape(
        -1,
        N_CLUSTERS,
    )

    valid_mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    if not valid_mask.any():

        raise ValueError(
            "Actual power contains no valid "
            "non-zero values."
        )

    actual_day = (
        actual[
            valid_mask
        ]
    )

    actual_peak = np.max(
        actual_day
    )

    actual_energy = np.sum(
        actual_day
    )

    if actual_peak <= 0:

        raise ValueError(
            "Actual peak must be greater than zero."
        )

    if actual_energy <= 0:

        raise ValueError(
            "Actual energy must be greater than zero."
        )

    max_loss = np.min(
        standard_efficiency
    )

    results = []

    loss_values = np.arange(
        0,
        max_loss + 0.0001,
        0.1,
    )

    for loss in loss_values:

        net_efficiency = (
            standard_efficiency
            - loss
        )

        net_efficiency = np.maximum(
            net_efficiency,
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

        adjusted_fixed_weights = (
            fixed_weights
            * efficiency_factor
        )

        power_matrix = (
            fixed_poa
            * adjusted_fixed_weights[None, :]
            / 1_000_000
        )

        predicted = (
            power_matrix.sum(
                axis=1
            )
        )

        predicted_day = (
            predicted[
                valid_mask
            ]
        )

        predicted_peak = np.max(
            predicted_day
        )

        peak_error = abs(
            actual_peak
            - predicted_peak
        )

        peak_error_percent = (
            peak_error
            / actual_peak
            * 100
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
            np.sum(
                predicted_day
            )
        )

        energy_error = (
            abs(
                actual_energy
                - predicted_energy
            )
            / actual_energy
        )

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

        results.append({

            "Error %":
                loss,

            "Actual Peak":
                actual_peak,

            "Predicted Peak":
                predicted_peak,

            "Peak Error":
                peak_error,

            "Peak Error (%)":
                peak_error_percent,

            "Block Error":
                block_error,

            "Energy Error":
                energy_error,

            "Overall Score":
                score,
        })

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise ValueError(
            "Fixed efficiency optimization "
            "did not produce any results."
        )

    best_row = results_df.loc[
        results_df[
            "Peak Error"
        ].idxmin()
    ]

    best_loss = float(
        best_row[
            "Error %"
        ]
    )

    return (
        best_loss,
        results_df,
        valid_mask,
    )


# ============================================================
# APPLY FIXED LOSS
# ============================================================

def calculate_final_fixed(
    standard_efficiency,
    fixed_weights,
    fixed_poa,
    best_loss,
):

    net_efficiency = (
        standard_efficiency
        - best_loss
    )

    net_efficiency = np.maximum(
        net_efficiency,
        0,
    )

    efficiency_factor = np.divide(

        net_efficiency,

        standard_efficiency,

        out=np.zeros_like(
            standard_efficiency
        ),

        where=(
            standard_efficiency != 0
        ),
    )

    final_fixed_weights = (
        fixed_weights
        * efficiency_factor
    )

    final_power_matrix = (
        fixed_poa
        * final_fixed_weights[None, :]
        / 1_000_000
    )

    fixed_forecast = (
        final_power_matrix.sum(
            axis=1
        )
    )

    return (
        net_efficiency,
        final_fixed_weights,
        final_power_matrix,
        fixed_forecast,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    forecast,
    valid_mask,
):

    actual_day = (
        actual[
            valid_mask
        ]
    )

    forecast_day = (
        forecast[
            valid_mask
        ]
    )

    if len(actual_day) == 0:

        raise ValueError(
            "No valid actual values."
        )

    actual_peak = np.max(
        actual_day
    )

    actual_energy = np.sum(
        actual_day
    )

    if actual_peak <= 0:

        raise ValueError(
            "Actual peak must be greater than zero."
        )

    if actual_energy <= 0:

        raise ValueError(
            "Actual energy must be greater than zero."
        )

    forecast_peak = np.max(
        forecast_day
    )

    block_error = (
        np.mean(
            np.abs(
                actual_day
                - forecast_day
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
            actual_energy
            - np.sum(
                forecast_day
            )
        )
        / actual_energy
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return {

        "actual_peak":
            actual_peak,

        "forecast_peak":
            forecast_peak,

        "block_error":
            block_error,

        "peak_error":
            peak_error,

        "energy_error":
            energy_error,

        "score":
            score,
    }


# ============================================================
# EFFICIENCY CONTROL
# ============================================================

def fixed_loss_control(
    best_loss,
    min_efficiency,
):

    st.markdown(
        '<div class="section-title">'
        '📉 Efficiency Loss'
        '</div>',
        unsafe_allow_html=True,
    )

    loss = st.number_input(

        "Efficiency Loss (%)",

        min_value=0.0,

        max_value=float(
            min_efficiency
        ),

        value=float(
            best_loss
        ),

        step=0.1,

        format="%.2f",

        key="fixed_efficiency_loss",

        help=(
            "Automatically calculated by "
            "minimizing Peak Error. "
            "You can manually modify it."
        ),
    )

    return float(
        loss
    )


# ============================================================
# EFFICIENCY TABLE
# ============================================================

def show_efficiency_table(
    area_df,
    loss,
):

    display = area_df.copy()

    if "Standard PV Efficiency (%)" in display.columns:

        standard_eff = pd.to_numeric(
            display[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce",
        )

        # IMPORTANT:
        # loss is a scalar here.
        display["Error %"] = float(
            loss
        )

        display["Net Efficiency (%)"] = (
            standard_eff
            - float(loss)
        ).clip(
            lower=0
        )

    if (
        "No of Module" in display.columns
        and
        "Area of 1 Module (m2)"
        in display.columns
    ):

        no_module = pd.to_numeric(
            display[
                "No of Module"
            ],
            errors="coerce",
        ).fillna(0)

        module_area = pd.to_numeric(
            display[
                "Area of 1 Module (m2)"
            ],
            errors="coerce",
        ).fillna(0)

        display["Total area (m2)"] = (
            no_module
            * module_area
        )

    if (
        "Total area (m2)"
        in display.columns
        and
        "Net Efficiency (%)"
        in display.columns
    ):

        display["Eff Area"] = (

            pd.to_numeric(
                display[
                    "Total area (m2)"
                ],
                errors="coerce",
            ).fillna(0)

            *

            pd.to_numeric(
                display[
                    "Net Efficiency (%)"
                ],
                errors="coerce",
            ).fillna(0)

            / 100
        )

    preferred_cols = [

        "S.No.",

        "Module Type",

        "No of Module",

        "Area of 1 Module (m2)",

        "Total area (m2)",

        "Standard PV Efficiency (%)",

        "Error %",

        "Net Efficiency (%)",

        "Eff Area",
    ]

    cols = [
        c
        for c in preferred_cols
        if c in display.columns
    ]

    remaining = [
        c
        for c in display.columns
        if c not in cols
    ]

    display = display[
        cols + remaining
    ]

    numeric_cols = display.select_dtypes(
        include="number"
    ).columns

    display[
        numeric_cols
    ] = display[
        numeric_cols
    ].round(4)

    with st.expander(
        "🔍 View Efficiency Calculations",
        expanded=False,
    ):

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    return display


# ============================================================
# FIXED MODEL
# ============================================================

def run_fixed_model(
    area_df,
    input_df,
    ghi_matrix,
    solar,
    fixed_weights,
):

    standard_efficiency = (

        pd.to_numeric(
            area_df[
                "Standard PV Efficiency (%)"
            ]
            .iloc[
                :N_CLUSTERS
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    if np.any(
        standard_efficiency <= 0
    ):

        raise ValueError(
            "Standard PV Efficiency must be "
            "greater than zero for the first "
            "5 cluster rows."
        )

    fixed_poa = (

        ghi_matrix

        *

        solar[
            "sin_ab"
        ][:, None]

        /

        solar[
            "sin_a"
        ][:, None]
    )

    actual = (
        input_df[
            "Actual"
        ]
        .to_numpy(
            dtype=float
        )
    )

    with st.spinner(
        "⚙️ Optimizing Fixed efficiency loss..."
    ):

        (
            best_loss,
            results_df,
            valid_mask,
        ) = optimize_fixed_loss(

            tuple(
                standard_efficiency
            ),

            tuple(
                fixed_weights
            ),

            tuple(
                fixed_poa.ravel()
            ),

            tuple(
                actual
            ),
        )

    loss = fixed_loss_control(

        best_loss,

        np.min(
            standard_efficiency
        ),
    )

    (
        net_efficiency,
        final_fixed_weights,
        final_power_matrix,
        fixed_forecast,
    ) = calculate_final_fixed(

        standard_efficiency,

        fixed_weights,

        fixed_poa,

        loss,
    )

    metrics = calculate_metrics(

        actual,

        fixed_forecast,

        valid_mask,
    )

    # IMPORTANT FIX:
    # Pass scalar loss, not net_efficiency.
    show_efficiency_table(
        area_df,
        loss,
    )

    st.markdown(
        '<div class="section-title">'
        '📊 Fixed Model Results'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Efficiency Loss",
        f"{loss:.2f}%",
    )

    c2.metric(
        "Actual Peak",
        f"{metrics['actual_peak']:.4f}",
    )

    c3.metric(
        "Fixed Peak",
        f"{metrics['forecast_peak']:.4f}",
    )

    c4.metric(
        "Peak Error",
        f"{metrics['peak_error'] * 100:.3f}%",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Block Error",
        f"{metrics['block_error']:.6f}",
    )

    c2.metric(
        "Energy Error",
        f"{metrics['energy_error']:.6f}",
    )

    c3.metric(
        "Overall Score",
        f"{metrics['score']:.6f}",
    )

    cluster_power = pd.DataFrame()

    for i, cluster in enumerate(
        CLUSTERS
    ):

        cluster_power[
            f"{cluster} Fixed Power"
        ] = final_power_matrix[
            :, i
        ]

    cluster_power[
        "Total Fixed Power"
    ] = fixed_forecast

    with st.expander(
        "🔍 View Fixed Cluster Power"
    ):

        st.dataframe(
            cluster_power.round(6),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "📈 View Efficiency Loss Optimization"
    ):

        st.dataframe(
            results_df.round(6),
            use_container_width=True,
            hide_index=True,
        )

    return {

        "loss":
            loss,

        "forecast":
            fixed_forecast,

        "power_matrix":
            final_power_matrix,

        "net_efficiency":
            net_efficiency,

        "metrics":
            metrics,

        "results_df":
            results_df,

        "valid_mask":
            valid_mask,
    }


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(

    DHI,

    start_block,

    end_block,

    max_block,

    east_limit,

    west_limit,

    blocks,

    ghi_matrix,

    tracking_weights,
):

    if not (
        start_block
        < max_block
        < end_block
    ):

        return None

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

        return None

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

            zenith
            < abs(
                east_limit
            ),

            zenith,

            abs(
                east_limit
            ),
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
        np.radians(
            panel
        )
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None,
    )

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    tracking_power_matrix = (

        dni

        *

        tracking_weights[None, :]

        / 1_000_000
    )

    tracking_forecast = (
        tracking_power_matrix.sum(
            axis=1
        )
    )

    return (

        tracking_forecast,

        tracking_power_matrix,

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
def optimize_tracking_cached(

    blocks_tuple,

    ghi_tuple,

    actual_tuple,

    tracking_weights_tuple,
):

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    ghi_matrix = np.asarray(
        ghi_tuple,
        dtype=float,
    )

    ghi_matrix = ghi_matrix.reshape(
        -1,
        N_CLUSTERS,
    )

    actual = np.asarray(
        actual_tuple,
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

    if not valid_mask.any():

        raise ValueError(
            "Actual power contains no valid "
            "non-zero values."
        )

    actual_day = (
        actual[
            valid_mask
        ]
    )

    actual_peak = np.max(
        actual_day
    )

    actual_energy = np.sum(
        actual_day
    )

    if (
        actual_peak <= 0
        or actual_energy <= 0
    ):

        raise ValueError(
            "Actual power data is invalid."
        )

    def objective(x):

        DHI = int(
            round(
                x[0]
            )
        )

        start_block = int(
            round(
                x[1]
            )
        )

        end_block = int(
            round(
                x[2]
            )
        )

        max_block = int(
            round(
                x[3]
            )
        )

        east_limit = int(
            round(
                x[4]
            )
        )

        west_limit = int(
            round(
                x[5]
            )
        )

        if not (
            start_block
            < max_block
            < end_block
        ):

            return 1e9

        result = calculate_tracking(

            DHI,

            start_block,

            end_block,

            max_block,

            east_limit,

            west_limit,

            blocks,

            ghi_matrix,

            tracking_weights,
        )

        if result is None:

            return 1e9

        prediction = result[0]

        if not np.all(
            np.isfinite(
                prediction
            )
        ):

            return 1e9

        prediction_day = (
            prediction[
                valid_mask
            ]
        )

        if len(
            prediction_day
        ) == 0:

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

        bounds=TRACKING_BOUNDS,

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

    return {

        "DHI":
            int(best[0]),

        "start":
            int(best[1]),

        "end":
            int(best[2]),

        "max":
            int(best[3]),

        "east":
            int(best[4]),

        "west":
            int(best[5]),
    }


# ============================================================
# TRACKING PARAMETER UI
# ============================================================

def tracking_parameter_controls(
    params,
):

    st.markdown(
        '<div class="section-title">'
        '⚙️ Tracking Parameters'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Optimizer values are loaded automatically. "
        "You can manually modify them."
    )

    c1, c2, c3 = st.columns(3)

    DHI = c1.number_input(

        "DHI (%)",

        min_value=0,

        max_value=10,

        value=int(
            params["DHI"]
        ),

        step=1,

        key="tracking_dhi",
    )

    start = c2.number_input(

        "Starting Block",

        min_value=10,

        max_value=30,

        value=int(
            params["start"]
        ),

        step=1,

        key="tracking_start",
    )

    end = c3.number_input(

        "Ending Block",

        min_value=65,

        max_value=80,

        value=int(
            params["end"]
        ),

        step=1,

        key="tracking_end",
    )

    c1, c2, c3 = st.columns(3)

    max_block = c1.number_input(

        "Max Block",

        min_value=47,

        max_value=53,

        value=int(
            params["max"]
        ),

        step=1,

        key="tracking_max",
    )

    east = c2.number_input(

        "East Limit",

        min_value=10,

        max_value=70,

        value=int(
            params["east"]
        ),

        step=1,

        key="tracking_east",
    )

    west = c3.number_input(

        "West Limit",

        min_value=10,

        max_value=70,

        value=int(
            params["west"]
        ),

        step=1,

        key="tracking_west",
    )

    return {

        "DHI":
            int(DHI),

        "start":
            int(start),

        "end":
            int(end),

        "max":
            int(max_block),

        "east":
            int(east),

        "west":
            int(west),
    }


# ============================================================
# TRACKING MODEL
# ============================================================

def run_tracking_model(

    input_df,

    ghi_matrix,

    blocks,

    tracking_weights,
):

    actual = (
        input_df[
            "Actual"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if (
        st.session_state.tracking_params
        is None
    ):

        with st.spinner(
            "🔄 Optimizing tracking parameters..."
        ):

            st.session_state.tracking_params = (

                optimize_tracking_cached(

                    tuple(
                        blocks
                    ),

                    tuple(
                        ghi_matrix.ravel()
                    ),

                    tuple(
                        actual
                    ),

                    tuple(
                        tracking_weights
                    ),
                )
            )

    params = tracking_parameter_controls(
        st.session_state.tracking_params
    )

    if not (

        params["start"]
        <
        params["max"]
        <
        params["end"]
    ):

        st.error(
            "Starting Block < Max Block "
            "< Ending Block is required."
        )

        return None

    result = calculate_tracking(

        params["DHI"],

        params["start"],

        params["end"],

        params["max"],

        params["east"],

        params["west"],

        blocks,

        ghi_matrix,

        tracking_weights,
    )

    if result is None:

        st.error(
            "Unable to calculate "
            "tracking forecast."
        )

        return None

    (
        tracking_forecast,
        tracking_power_matrix,
        zenith,
        panel,
        dni,
    ) = result

    valid_mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    metrics = calculate_metrics(

        actual,

        tracking_forecast,

        valid_mask,
    )

    st.markdown(
        '<div class="section-title">'
        '🔄 Tracking Model Results'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "DHI",
        f"{params['DHI']}%",
    )

    c2.metric(
        "Actual Peak",
        f"{metrics['actual_peak']:.4f}",
    )

    c3.metric(
        "Tracking Peak",
        f"{metrics['forecast_peak']:.4f}",
    )

    c4.metric(
        "Peak Error",
        f"{metrics['peak_error'] * 100:.3f}%",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Block Error",
        f"{metrics['block_error']:.6f}",
    )

    c2.metric(
        "Energy Error",
        f"{metrics['energy_error']:.6f}",
    )

    c3.metric(
        "Overall Score",
        f"{metrics['score']:.6f}",
    )

    tracking_details = pd.DataFrame({

        "Block":
            blocks,

        "Zenith Angle":
            zenith,

        "Panel Angle":
            panel,
    })

    for i, cluster in enumerate(
        CLUSTERS
    ):

        tracking_details[
            f"{cluster} Tracking Power"
        ] = tracking_power_matrix[
            :, i
        ]

    tracking_details[
        "Total Tracking Power"
    ] = tracking_forecast

    with st.expander(
        "🔍 View Tracking Calculations"
    ):

        st.dataframe(
            tracking_details.round(6),
            use_container_width=True,
            hide_index=True,
        )

    return {

        "params":
            params,

        "forecast":
            tracking_forecast,

        "power_matrix":
            tracking_power_matrix,

        "zenith":
            zenith,

        "panel":
            panel,

        "dni":
            dni,

        "metrics":
            metrics,
    }


# ============================================================
# FORECAST CHART
# ============================================================

def show_forecast_chart(

    actual,

    fixed_forecast,

    tracking_forecast=None,
):

    n = min(

        len(actual),

        len(fixed_forecast),
    )

    if tracking_forecast is not None:

        n = min(

            n,

            len(
                tracking_forecast
            ),
        )

    x = np.arange(
        1,
        n + 1,
    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=x,

            y=np.asarray(
                actual[:n]
            ),

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

            y=np.asarray(
                fixed_forecast[:n]
            ),

            mode="lines",

            name="Fixed Forecast",

            line=dict(

                color="#3B82F6",

                width=2.5,
            ),
        )
    )

    if tracking_forecast is not None:

        fig.add_trace(

            go.Scatter(

                x=x,

                y=np.asarray(
                    tracking_forecast[:n]
                ),

                mode="lines",

                name="Tracking Forecast",

                line=dict(

                    color="#16A34A",

                    width=2.5,
                ),
            )
        )

    fig.update_layout(

        title=(

            "Actual vs Fixed vs Tracking"

            if tracking_forecast is not None

            else

            "Actual vs Fixed Forecast"
        ),

        height=500,

        hovermode="x unified",

        template="plotly_white",

        xaxis_title="15 Minute Block",

        yaxis_title="Power (MW)",

        margin=dict(

            l=20,

            r=20,

            t=60,

            b=20,
        ),
    )

    st.plotly_chart(

        fig,

        use_container_width=True,

        config={

            "displaylogo":
                False,

            "responsive":
                True,
        },
    )


# ============================================================
# PLANT SELECTOR
# ============================================================

def plant_selector():

    st.markdown(
        '<div class="section-title">'
        '🏭 Select Plant Type'
        '</div>',
        unsafe_allow_html=True,
    )

    plant_type = st.segmented_control(

        "Plant Type",

        options=[

            "🏗️ Fixed",

            "🔄 Tracking",
        ],

        default=st.session_state.plant_type,

        selection_mode="single",

        key="plant_type_selector",

        label_visibility="collapsed",

        width="stretch",
    )

    if plant_type is None:

        plant_type = "🏗️ Fixed"

    st.session_state.plant_type = (
        plant_type
    )

    return plant_type


# ============================================================
# SUMMARY
# ============================================================

def create_summary_dataframe(

    fixed_result,

    tracking_result,
):

    fixed_metrics = (
        fixed_result[
            "metrics"
        ]
    )

    tracking_metrics = (
        tracking_result[
            "metrics"
        ]
    )

    tracking_params = (
        tracking_result[
            "params"
        ]
    )

    return pd.DataFrame({

        "Metric": [

            "Efficiency Loss (%)",

            "DHI (%)",

            "GHI Starting Block",

            "GHI Ending Block",

            "GHI Max Block",

            "East Tracking Limit",

            "West Tracking Limit",

            "Block Error",

            "Peak Error",

            "Energy Error",

            "Overall Score",

            "Peak Power",
        ],

        "Fixed": [

            fixed_result[
                "loss"
            ],

            np.nan,

            np.nan,

            np.nan,

            np.nan,

            np.nan,

            np.nan,

            fixed_metrics[
                "block_error"
            ],

            fixed_metrics[
                "peak_error"
            ],

            fixed_metrics[
                "energy_error"
            ],

            fixed_metrics[
                "score"
            ],

            fixed_metrics[
                "forecast_peak"
            ],
        ],

        "Tracking": [

            fixed_result[
                "loss"
            ],

            tracking_params[
                "DHI"
            ],

            tracking_params[
                "start"
            ],

            tracking_params[
                "end"
            ],

            tracking_params[
                "max"
            ],

            tracking_params[
                "east"
            ],

            tracking_params[
                "west"
            ],

            tracking_metrics[
                "block_error"
            ],

            tracking_metrics[
                "peak_error"
            ],

            tracking_metrics[
                "energy_error"
            ],

            tracking_metrics[
                "score"
            ],

            tracking_metrics[
                "forecast_peak"
            ],
        ],
    })


def show_summary(

    fixed_result,

    tracking_result,
):

    st.markdown(
        '<div class="section-title">'
        '📋 Fixed vs Tracking Summary'
        '</div>',
        unsafe_allow_html=True,
    )

    summary = create_summary_dataframe(

        fixed_result,

        tracking_result,
    )

    st.dataframe(

        summary.round(6),

        use_container_width=True,

        hide_index=True,
    )

    return summary


# ============================================================
# EXCEL REPORT
# ============================================================

def create_excel_report(

    fixed_result,

    tracking_result=None,
):

    output = BytesIO()

    with pd.ExcelWriter(

        output,

        engine="openpyxl",
    ) as writer:

        # ----------------------------------------------------
        # Fixed optimization
        # ----------------------------------------------------

        fixed_result[
            "results_df"
        ].round(6).to_excel(

            writer,

            sheet_name="Fixed Optimization",

            index=False,
        )

        # ----------------------------------------------------
        # Fixed power
        # ----------------------------------------------------

        fixed_power = pd.DataFrame()

        for i, cluster in enumerate(
            CLUSTERS
        ):

            fixed_power[
                f"{cluster} Fixed Power"
            ] = fixed_result[
                "power_matrix"
            ][:, i]

        fixed_power[
            "Total Fixed Power"
        ] = fixed_result[
            "forecast"
        ]

        fixed_power.round(6).to_excel(

            writer,

            sheet_name="Fixed Power",

            index=False,
        )

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        if tracking_result is not None:

            tracking_power = pd.DataFrame()

            for i, cluster in enumerate(
                CLUSTERS
            ):

                tracking_power[
                    f"{cluster} Tracking Power"
                ] = tracking_result[
                    "power_matrix"
                ][:, i]

            tracking_power[
                "Total Tracking Power"
            ] = tracking_result[
                "forecast"
            ]

            tracking_power.round(6).to_excel(

                writer,

                sheet_name="Tracking Power",

                index=False,
            )

            # ------------------------------------------------
            # Tracking calculations
            # ------------------------------------------------

            tracking_calc = pd.DataFrame({

                "Block":
                    tracking_result[
                        "zenith"
                    ] * 0
                    + np.arange(
                        1,
                        len(
                            tracking_result[
                                "zenith"
                            ]
                        ) + 1
                    ),

                "Zenith Angle":
                    tracking_result[
                        "zenith"
                    ],

                "Panel Angle":
                    tracking_result[
                        "panel"
                    ],
            })

            tracking_calc.round(6).to_excel(

                writer,

                sheet_name="Tracking Calculations",

                index=False,
            )

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            summary = create_summary_dataframe(

                fixed_result,

                tracking_result,
            )

            summary.round(6).to_excel(

                writer,

                sheet_name="Summary",

                index=False,
            )

        else:

            # ------------------------------------------------
            # Fixed-only summary
            # ------------------------------------------------

            fm = fixed_result[
                "metrics"
            ]

            fixed_summary = pd.DataFrame({

                "Metric": [

                    "Efficiency Loss (%)",

                    "Actual Peak",

                    "Fixed Peak",

                    "Block Error",

                    "Peak Error",

                    "Energy Error",

                    "Overall Score",
                ],

                "Value": [

                    fixed_result[
                        "loss"
                    ],

                    fm[
                        "actual_peak"
                    ],

                    fm[
                        "forecast_peak"
                    ],

                    fm[
                        "block_error"
                    ],

                    fm[
                        "peak_error"
                    ],

                    fm[
                        "energy_error"
                    ],

                    fm[
                        "score"
                    ],
                ],
            })

            fixed_summary.round(6).to_excel(

                writer,

                sheet_name="Summary",

                index=False,
            )

    output.seek(0)

    return output.getvalue()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">'
        '☀️ Loss Correction Model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload the Excel workbook, modify GHI and Actual values, '
        'select Fixed or Tracking and run the correction.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📁 Input Sheet'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(

        "Upload Excel File",

        type=[
            "xlsx",
            "xls",
        ],

        key="workbook_uploader",
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload the plant Excel file to begin."
        )

        return

    file_bytes = (
        uploaded_file.getvalue()
    )

    signature = (
        get_file_signature(
            uploaded_file
        )
    )

    # ========================================================
    # NEW FILE DETECTION
    # ========================================================

    if (
        st.session_state.file_signature
        != signature
    ):

        st.session_state.file_signature = (
            signature
        )

        st.session_state.workbook_cache = None

        st.session_state.input_editor_data = None

        st.session_state.run_model = False

        st.session_state.tracking_params = None

    # ========================================================
    # LOAD WORKBOOK
    # ========================================================

    if (
        st.session_state.workbook_cache
        is None
    ):

        try:

            with st.spinner(
                "📖 Reading workbook..."
            ):

                st.session_state.workbook_cache = (
                    load_workbook(
                        file_bytes
                    )
                )

        except Exception as e:

            st.error(
                f"Unable to load workbook: {e}"
            )

            st.exception(e)

            return

    workbook = (
        st.session_state.workbook_cache
    )

    area_df = (
        workbook[
            "area_df"
        ]
    )

    fixed_weights = (
        workbook[
            "fixed_weights"
        ]
    )

    tracking_weights = (
        workbook[
            "tracking_weights"
        ]
    )

    lat = (
        workbook[
            "lat"
        ]
    )

    tilt_lookup = (
        workbook[
            "tilt_lookup"
        ]
    )

    result_df = (
        workbook[
            "result_df"
        ]
    )

    input_raw = (
        workbook[
            "input_raw"
        ]
    )

    # ========================================================
    # BUILD EDITOR
    # ========================================================

    if (
        st.session_state.input_editor_data
        is None
    ):

        try:

            st.session_state.input_editor_data = (
                build_input_editor_data(

                    input_raw,

                    result_df,
                )
            )

        except Exception as e:

            st.error(
                f"Unable to prepare input data: {e}"
            )

            return

    # ========================================================
    # EDITOR
    # ========================================================

    input_df = input_data_editor(

        st.session_state.input_editor_data
    )

    # Store edited data.
    st.session_state.input_editor_data = (
        input_df.copy()
    )

    # ========================================================
    # PLANT TYPE
    # ========================================================

    plant_type = plant_selector()

    st.markdown("")

    # ========================================================
    # RUN
    # ========================================================

    run_clicked = st.button(

        "🚀 RUN LOSS CORRECTION",

        type="primary",

        use_container_width=True,

        key="run_loss_correction",
    )

    if run_clicked:

        st.session_state.run_model = True

        # Tracking optimization depends on
        # current edited GHI + Actual values.
        st.session_state.tracking_params = None

    if not (
        st.session_state.run_model
    ):

        st.info(
            "Select the plant type and click "
            "**Run Loss Correction** to start."
        )

        return

    # ========================================================
    # PREPARE MODEL DATA
    # ========================================================

    try:

        dates = pd.to_datetime(

            input_df[
                "Date"
            ],

            errors="coerce",
        )

        if dates.isna().any():

            raise ValueError(
                "Invalid dates found in Fixed-C11."
            )

        # ----------------------------------------------------
        # GHI MATRIX
        # ----------------------------------------------------

        ghi_matrix = np.column_stack([

            pd.to_numeric(

                input_df[
                    col
                ],

                errors="coerce",

            )
            .fillna(0)
            .to_numpy(
                dtype=float
            )

            for col in GHI_COLS
        ])

        # ----------------------------------------------------
        # ACTUAL
        # ----------------------------------------------------

        actual = pd.to_numeric(

            input_df[
                "Actual"
            ],

            errors="coerce",

        ).fillna(0).to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # BLOCKS
        # ----------------------------------------------------

        blocks = pd.to_numeric(

            result_df[
                "Block"
            ].iloc[
                :len(input_df)
            ],

            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if (
            len(blocks)
            != len(input_df)
        ):

            raise ValueError(
                "Input rows and Result blocks "
                "are not aligned."
            )

        # ----------------------------------------------------
        # SOLAR
        # ----------------------------------------------------

        solar = prepare_solar_data(

            tuple(
                dates.astype(str)
            ),

            lat,

            tuple(
                sorted(
                    tilt_lookup.items()
                )
            ),
        )

    except Exception as e:

        st.error(
            f"Unable to prepare model data: {e}"
        )

        st.exception(e)

        return

    # ========================================================
    # FIXED MODEL
    # ========================================================

    try:

        fixed_result = run_fixed_model(

            area_df,

            input_df,

            ghi_matrix,

            solar,

            fixed_weights,
        )

    except Exception as e:

        st.error(
            "❌ Fixed model failed."
        )

        st.exception(e)

        return

    # ========================================================
    # TRACKING
    # ========================================================

    tracking_result = None

    if plant_type == "🔄 Tracking":

        try:

            tracking_result = run_tracking_model(

                input_df,

                ghi_matrix,

                blocks,

                tracking_weights,
            )

            if tracking_result is None:

                return

        except Exception as e:

            st.error(
                "❌ Tracking model failed."
            )

            st.exception(e)

            return

    # ========================================================
    # FORECAST CHART
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Forecast Comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    show_forecast_chart(

        actual,

        fixed_result[
            "forecast"
        ],

        (
            tracking_result[
                "forecast"
            ]

            if tracking_result is not None

            else None
        ),
    )

    # ========================================================
    # SUMMARY + REPORT
    # ========================================================

    if tracking_result is not None:

        show_summary(

            fixed_result,

            tracking_result,
        )

        report_bytes = create_excel_report(

            fixed_result,

            tracking_result,
        )

        st.download_button(

            "⬇️ Download Final Report",

            data=report_bytes,

            file_name=(
                "Loss_Correction_Final_Report.xlsx"
            ),

            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),

            use_container_width=True,
        )

    else:

        report_bytes = create_excel_report(

            fixed_result,

            None,
        )

        st.download_button(

            "⬇️ Download Fixed Report",

            data=report_bytes,

            file_name=(
                "Loss_Correction_Fixed_Report.xlsx"
            ),

            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),

            use_container_width=True,
        )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    main()
