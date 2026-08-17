# ============================================================
# STREAMLIT APP
# FIXED VS TRACKING LOSS CORRECTION MODEL
# ============================================================

import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fixed vs Tracking Loss Correction",
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

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

N_CLUSTERS = len(CLUSTERS)

MAX_OPT_ITER = 40
OPT_POPSIZE = 15

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
        font-size: 1.25rem;
        font-weight: 650;
        margin: 18px 0 10px 0;
    }

    div.stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 650;
        transition: 0.15s ease;
    }

    .selected-model {
        padding: 12px 16px;
        border-radius: 12px;
        margin-top: 8px;
        font-weight: 600;
        text-align: center;
        background: rgba(37, 99, 235, 0.12);
        border: 1px solid rgba(37, 99, 235, 0.45);
        color: #60a5fa;
    }

    .input-card {
        padding: 12px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 12px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "input_df": None,
    "input_context": None,
    "model_results": None,
    "tracking_params": None,
    "fixed_loss": None,
    "run_model": False,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HELPERS
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


def get_sheet_names(uploaded_file):

    uploaded_file.seek(0)

    return pd.ExcelFile(
        uploaded_file
    ).sheet_names


def clean_data_rows(
    df,
    date_column="Date"
):

    df = df.copy()

    if date_column in df.columns:

        idx = df[
            df[date_column].isna()
        ].index

        if len(idx):

            pos = df.index.get_loc(
                idx[0]
            )

            df = df.iloc[:pos]

    return df.reset_index(
        drop=True
    )


# ============================================================
# WORKBOOK DETECTION
# ============================================================

def detect_workbook(uploaded_file):

    sheets = get_sheet_names(
        uploaded_file
    )

    required = [
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
        s for s in required
        if s not in sheets
    ]

    if missing:

        raise ValueError(
            "Workbook is missing required sheets: "
            + ", ".join(missing)
        )

    return sheets


# ============================================================
# 1. AREA & EFFICIENCY
# ============================================================

def read_area_efficiency(
    uploaded_file
):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
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
            regex=False
        )
        .str.strip()
    )

    validate_columns(
        df,
        [
            "S.No.",
            "Standard PV Efficiency (%)",
        ],
        "Area & Efficiency",
    )

    df = df[
        df["S.No."].notna()
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# ============================================================
# 2. EFFECTIVE AREAS
# ============================================================

def read_effective_areas(
    uploaded_file
):

    uploaded_file.seek(0)

    area_df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=None,
    )

    fixed_weights = (
        pd.to_numeric(
            area_df.iloc[
                2:7,
                15
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    tracking_weights = (
        pd.to_numeric(
            area_df.iloc[
                28:33,
                15
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
            "Could not read 5 fixed effective areas."
        )

    if len(tracking_weights) != N_CLUSTERS:

        raise ValueError(
            "Could not read 5 tracking effective areas."
        )

    return (
        fixed_weights,
        tracking_weights,
    )


# ============================================================
# 3. STANDARD PV EFFICIENCY
# ============================================================

def read_standard_efficiency(
    df
):

    standard_efficiency = (
        pd.to_numeric(
            df[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    if len(
        standard_efficiency
    ) < N_CLUSTERS:

        raise ValueError(
            "Less than 5 Standard PV Efficiency "
            "values found."
        )

    return standard_efficiency[
        :N_CLUSTERS
    ]


# ============================================================
# 4. FORECAST CONFIG
# ============================================================

def read_latitude(
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
        df_config.loc[
            0,
            "Lat"
        ]
    )


# ============================================================
# 5. CONFIG TILT ANGLE
# ============================================================

def read_tilt_lookup(
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

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    validate_columns(
        df_tilt,
        ["Month_Num", "Fixed"],
        "Config Tilt Angle",
    )

    df_tilt = df_tilt.dropna(
        subset=["Fixed"]
    ).copy()

    df_tilt["Month_Num"] = (
        pd.to_numeric(
            df_tilt["Month_Num"],
            errors="coerce",
        )
    )

    df_tilt["Fixed"] = (
        pd.to_numeric(
            df_tilt["Fixed"],
            errors="coerce",
        )
    )

    return (
        df_tilt
        .dropna(
            subset=["Month_Num"]
        )
        .set_index("Month_Num")[
            "Fixed"
        ]
        .to_dict()
    )


# ============================================================
# 6. RESULT / GHI
# ============================================================

def read_ghi_data(
    uploaded_file
):

    uploaded_file.seek(0)

    df_ghi = pd.read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=range(6),
    )

    df_ghi.columns = [
        "Block",
        *GHI_COLS,
    ]

    df_ghi = df_ghi[
        pd.to_numeric(
            df_ghi["Block"],
            errors="coerce",
        ).notna()
    ].copy()

    for col in GHI_COLS:

        df_ghi[col] = (
            pd.to_numeric(
                df_ghi[col],
                errors="coerce",
            )
            .fillna(0)
        )

    blocks = (
        pd.to_numeric(
            df_ghi["Block"],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    ghi_matrix = np.column_stack(
        [
            df_ghi[col].to_numpy(
                dtype=float
            )
            for col in GHI_COLS
        ]
    )

    return (
        df_ghi,
        blocks,
        ghi_matrix,
    )


# ============================================================
# 7. FIXED-C11
# ============================================================

def read_fixed_data(
    uploaded_file
):

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

    date_valid = (
        df_fix["Date"].notna()
    )

    if not date_valid.any():

        raise ValueError(
            "No valid Date rows found in Fixed-C11."
        )

    first_blank = np.where(
        ~date_valid.to_numpy()
    )[0]

    if len(first_blank) > 0:

        df_fix = df_fix.iloc[
            :first_blank[0]
        ].copy()

    else:

        df_fix = df_fix.loc[
            date_valid
        ].copy()

    df_fix.reset_index(
        drop=True,
        inplace=True
    )

    return df_fix


# ============================================================
# 8. ALIGN DATA
# ============================================================

def prepare_model_data(
    df_fix,
    df_ghi,
    blocks,
    ghi_matrix,
):

    actual_full = (
        pd.to_numeric(
            df_fix["Actual"],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    n = min(
        len(df_fix),
        len(df_ghi),
    )

    if n == 0:

        raise ValueError(
            "No valid forecast rows available."
        )

    df_fix = df_fix.iloc[
        :n
    ].copy()

    df_ghi = df_ghi.iloc[
        :n
    ].copy()

    actual = actual_full[
        :n
    ]

    ghi_matrix = ghi_matrix[
        :n
    ]

    blocks = blocks[
        :n
    ]

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
# 9. SOLAR CALCULATIONS
# ============================================================

def calculate_solar_angles(
    dates,
    lat,
    tilt_lookup,
):

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

    declination = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
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
        dates.dt.month.to_numpy()
    )

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

    return (
        declination,
        elevation,
        tilt,
        sin_a_safe,
        sin_ab,
    )


# ============================================================
# 10. FIXED POA
# ============================================================

def calculate_fixed_poa(
    ghi_matrix,
    sin_a_safe,
    sin_ab,
):

    return (
        ghi_matrix
        * sin_ab[:, None]
        / sin_a_safe[:, None]
    )


# ============================================================
# 11. VALID ACTUAL DATA
# ============================================================

def prepare_actual_metrics(
    actual
):

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

    actual_day = actual[
        valid_mask
    ]

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

    return (
        valid_mask,
        actual_day,
        actual_peak,
        actual_energy,
    )


# ============================================================
# 12. FIXED EFFICIENCY LOSS OPTIMIZATION
# ============================================================

def optimize_fixed_loss(
    standard_efficiency,
    fixed_weights,
    fixed_poa,
    actual,
    valid_mask,
    actual_day,
    actual_peak,
    actual_energy,
):

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

        efficiency_factor = (
            np.divide(
                net_efficiency,
                standard_efficiency,
                out=np.zeros_like(
                    net_efficiency
                ),
                where=(
                    standard_efficiency
                    != 0
                ),
            )
        )

        adjusted_fixed_weights = (
            fixed_weights
            * efficiency_factor
        )

        power_matrix = (
            fixed_poa
            * adjusted_fixed_weights[
                None,
                :
            ]
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

        if len(predicted_day) == 0:

            continue

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

        energy_error = abs(
            actual_energy
            - predicted_energy
        ) / actual_energy

        score = (
            0.80 * block_error
            + 0.10 * (
                peak_error
                / actual_peak
            )
            + 0.10 * energy_error
        )

        results.append(
            {
                "Error %": loss,
                "Actual Peak": actual_peak,
                "Predicted Peak": predicted_peak,
                "Peak Error": peak_error,
                "Peak Error (%)": peak_error_percent,
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
            "Fixed efficiency optimization "
            "did not produce any results."
        )

    best_row = results_df.loc[
        results_df[
            "Peak Error"
        ].idxmin()
    ]

    best_loss = float(
        best_row["Error %"]
    )

    return (
        results_df,
        best_row,
        best_loss,
    )


# ============================================================
# 13. APPLY FIXED LOSS
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

    efficiency_factor = (
        np.divide(
            net_efficiency,
            standard_efficiency,
            out=np.zeros_like(
                standard_efficiency
            ),
            where=(
                standard_efficiency
                != 0
            ),
        )
    )

    final_fixed_weights = (
        fixed_weights
        * efficiency_factor
    )

    final_fixed_power_matrix = (
        fixed_poa
        * final_fixed_weights[
            None,
            :
        ]
        / 1_000_000
    )

    fixed_forecast = (
        final_fixed_power_matrix.sum(
            axis=1
        )
    )

    return (
        net_efficiency,
        final_fixed_weights,
        final_fixed_power_matrix,
        fixed_forecast,
    )


# ============================================================
# 14. FINAL FIXED METRICS
# ============================================================

def calculate_metrics(
    forecast,
    actual_day,
    actual_peak,
    actual_energy,
    valid_mask,
):

    forecast_day = (
        forecast[
            valid_mask
        ]
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
            - forecast_day.max()
        )
        / actual_peak
    )

    energy_error = (
        abs(
            actual_energy
            - forecast_day.sum()
        )
        / actual_energy
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return {
        "Block Error": block_error,
        "Peak Error": peak_error,
        "Energy Error": energy_error,
        "Overall Score": score,
        "Peak Power": forecast_day.max(),
    }


# ============================================================
# 15. TRACKING CALCULATION
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
        np.where(
            zenith
            < abs(east_limit),
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
        * DHI
        / 100
    )

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    tracking_power_matrix = (
        dni
        * tracking_weights[
            None,
            :
        ]
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
# 16. TRACKING OBJECTIVE
# ============================================================

def make_tracking_objective(
    blocks,
    ghi_matrix,
    tracking_weights,
    actual_day,
    actual_peak,
    actual_energy,
    valid_mask,
):

    def objective(x):

        DHI = int(
            round(x[0])
        )

        start_block = int(
            round(x[1])
        )

        end_block = int(
            round(x[2])
        )

        max_block = int(
            round(x[3])
        )

        east_limit = int(
            round(x[4])
        )

        west_limit = int(
            round(x[5])
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
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    return objective


# ============================================================
# 17. TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_tracking_cached(
    blocks_tuple,
    ghi_tuple,
    tracking_weights_tuple,
    actual_day_tuple,
    actual_peak,
    actual_energy,
    valid_mask_tuple,
):

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    ghi_matrix = np.asarray(
        ghi_tuple,
        dtype=float,
    )

    tracking_weights = np.asarray(
        tracking_weights_tuple,
        dtype=float,
    )

    actual_day = np.asarray(
        actual_day_tuple,
        dtype=float,
    )

    valid_mask = np.asarray(
        valid_mask_tuple,
        dtype=bool,
    )

    objective = make_tracking_objective(
        blocks,
        ghi_matrix,
        tracking_weights,
        actual_day,
        actual_peak,
        actual_energy,
        valid_mask,
    )

    result = differential_evolution(
        objective,
        bounds=TRACKING_BOUNDS,
        strategy="best1bin",
        maxiter=MAX_OPT_ITER,
        popsize=OPT_POPSIZE,
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
        "DHI": int(best[0]),
        "start": int(best[1]),
        "end": int(best[2]),
        "max": int(best[3]),
        "east": int(best[4]),
        "west": int(best[5]),
        "optimizer_score": float(
            result.fun
        ),
        "rounded_score": float(
            objective(best)
        ),
    }


# ============================================================
# 18. INPUT DATA EDITOR
# ============================================================

def input_data_editor(
    df_fix,
    df_ghi,
):

    st.markdown(
        '<div class="section-title">'
        '📊 Editable GHI and Actual Power'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Modify GHI and Actual values below. "
        "The original calculation methodology is preserved."
    )

    n = min(
        len(df_fix),
        len(df_ghi),
    )

    display = pd.DataFrame()

    display["Date"] = (
        df_fix["Date"]
        .iloc[:n]
        .values
    )

    display["Block"] = (
        df_ghi["Block"]
        .iloc[:n]
        .values
    )

    for col in GHI_COLS:

        display[col] = (
            df_ghi[col]
            .iloc[:n]
            .values
        )

    display["Actual"] = (
        df_fix["Actual"]
        .iloc[:n]
        .values
    )

    edited = st.data_editor(
        display,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="model_input_editor",
        column_config={
            "Date": st.column_config.DateColumn(
                "Date",
                disabled=True,
            ),
            "Block": st.column_config.NumberColumn(
                "Block",
                disabled=True,
            ),
            **{
                col: st.column_config.NumberColumn(
                    col,
                    step=0.01,
                    format="%.4f",
                )
                for col in GHI_COLS
            },
            "Actual": st.column_config.NumberColumn(
                "Actual",
                step=0.01,
                format="%.4f",
            ),
        },
    )

    return edited


# ============================================================
# 19. FIXED LOSS CONTROL
# ============================================================

def fixed_loss_control(
    best_loss,
    max_loss,
):

    st.markdown(
        '<div class="section-title">'
        '📉 Fixed Efficiency Loss'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Optimizer value is loaded automatically. "
        "You can manually modify it before recalculating."
    )

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=float(max_loss),
        value=float(best_loss),
        step=0.1,
        format="%.2f",
        key="fixed_efficiency_loss",
    )

    return float(loss)


# ============================================================
# 20. TRACKING PARAMETER CONTROLS
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
        "You can manually modify them before recalculating."
    )

    c1, c2, c3 = st.columns(3)

    DHI = c1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=10,
        value=int(params["DHI"]),
        step=1,
        key="tracking_dhi",
    )

    start = c2.number_input(
        "Starting Block",
        min_value=10,
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
        min_value=47,
        max_value=53,
        value=int(params["max"]),
        step=1,
        key="tracking_max",
    )

    east = c2.number_input(
        "East Limit",
        min_value=10,
        max_value=70,
        value=int(params["east"]),
        step=1,
        key="tracking_east",
    )

    west = c3.number_input(
        "West Limit",
        min_value=10,
        max_value=70,
        value=int(params["west"]),
        step=1,
        key="tracking_west",
    )

    return {
        "DHI": int(DHI),
        "start": int(start),
        "end": int(end),
        "max": int(max_block),
        "east": int(east),
        "west": int(west),
    }


# ============================================================
# 21. SUMMARY METRICS
# ============================================================

def show_metrics(
    fixed_metrics,
    tracking_metrics,
):

    st.markdown(
        '<div class="section-title">'
        '📊 Final Model Metrics'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            "### 🏗️ Fixed"
        )

        m1, m2 = st.columns(2)

        m1.metric(
            "Peak Power",
            f"{fixed_metrics['Peak Power']:.6f}",
        )

        m2.metric(
            "Overall Score",
            f"{fixed_metrics['Overall Score']:.6f}",
        )

        m1.metric(
            "Block Error",
            f"{fixed_metrics['Block Error']:.6f}",
        )

        m2.metric(
            "Peak Error",
            f"{fixed_metrics['Peak Error']:.6f}",
        )

        st.metric(
            "Energy Error",
            f"{fixed_metrics['Energy Error']:.6f}",
        )

    with c2:

        st.markdown(
            "### 🔄 Tracking"
        )

        m1, m2 = st.columns(2)

        m1.metric(
            "Peak Power",
            f"{tracking_metrics['Peak Power']:.6f}",
        )

        m2.metric(
            "Overall Score",
            f"{tracking_metrics['Overall Score']:.6f}",
        )

        m1.metric(
            "Block Error",
            f"{tracking_metrics['Block Error']:.6f}",
        )

        m2.metric(
            "Peak Error",
            f"{tracking_metrics['Peak Error']:.6f}",
        )

        st.metric(
            "Energy Error",
            f"{tracking_metrics['Energy Error']:.6f}",
        )


# ============================================================
# 22. FORECAST CHART
# ============================================================

def show_forecast_chart(
    actual,
    fixed_forecast,
    tracking_forecast,
):

    n = min(
        len(actual),
        len(fixed_forecast),
        len(tracking_forecast),
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
        title="Actual vs Fixed vs Tracking Forecast",
        height=500,
        hovermode="x unified",
        template="plotly_white",
        xaxis_title="Block",
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
    )


# ============================================================
# 23. EFFICIENCY LOSS RESULTS
# ============================================================

def show_efficiency_results(
    results_df,
):

    st.markdown(
        '<div class="section-title">'
        '📉 Efficiency Loss Test Results'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander(
        "🔍 View All Efficiency Loss Tests"
    ):

        display = results_df.copy()

        numeric_cols = (
            display
            .select_dtypes(
                include="number"
            )
            .columns
        )

        display[numeric_cols] = (
            display[numeric_cols]
            .round(6)
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 24. FIXED POWER TABLE
# ============================================================

def show_fixed_output(
    df_fix,
):

    st.markdown(
        '<div class="section-title">'
        '🏗️ Fixed Model Output'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander(
        "🔍 View Fixed Power by Cluster"
    ):

        cols = [
            "Date",
            "Actual",
            *[
                f"{cl}_Fixed Power=I*Ƞ*A"
                for cl in CLUSTERS
            ],
            "Total Power (CL1+CL2+…)",
        ]

        cols = [
            c
            for c in cols
            if c in df_fix.columns
        ]

        st.dataframe(
            df_fix[cols],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 25. TRACKING OUTPUT TABLE
# ============================================================

def show_tracking_output(
    df_trac,
):

    st.markdown(
        '<div class="section-title">'
        '🔄 Tracking Model Output'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander(
        "🔍 View Tracking Power by Cluster"
    ):

        cols = [
            "Zenith Angle",
            "Panel Angle",
            *[
                f"{cl}_Tracking Power=I*Ƞ*A"
                for cl in CLUSTERS
            ],
            "Tracking Power=I*Ƞ*A",
        ]

        cols = [
            c
            for c in cols
            if c in df_trac.columns
        ]

        st.dataframe(
            df_trac[cols],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 26. SUMMARY TABLE
# ============================================================

def build_summary(
    best_loss,
    DHI,
    start,
    end,
    max_block,
    east,
    west,
    fixed_metrics,
    tracking_metrics,
):

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

            best_loss,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            fixed_metrics[
                "Block Error"
            ],
            fixed_metrics[
                "Peak Error"
            ],
            fixed_metrics[
                "Energy Error"
            ],
            fixed_metrics[
                "Overall Score"
            ],
            fixed_metrics[
                "Peak Power"
            ],
        ],

        "Tracking": [

            best_loss,
            DHI,
            start,
            end,
            max_block,
            east,
            west,
            tracking_metrics[
                "Block Error"
            ],
            tracking_metrics[
                "Peak Error"
            ],
            tracking_metrics[
                "Energy Error"
            ],
            tracking_metrics[
                "Overall Score"
            ],
            tracking_metrics[
                "Peak Power"
            ],
        ],
    })


# ============================================================
# 27. OPTIMIZED PARAMETERS
# ============================================================

def build_optimized_parameters(
    best_loss,
    actual_peak,
    fixed_metrics,
    DHI,
    start,
    end,
    max_block,
    east,
    west,
    tracking_metrics,
):

    return pd.DataFrame({

        "Parameter": [

            "Fixed Efficiency Loss (%)",
            "Fixed Actual Peak",
            "Fixed Predicted Peak",
            "Fixed Peak Error (%)",
            "Fixed Block Error",
            "Fixed Energy Error",
            "Fixed Overall Score",

            "Tracking DHI (%)",
            "Tracking GHI Starting Block",
            "Tracking GHI Ending Block",
            "Tracking GHI Max Block",
            "Tracking East Limit",
            "Tracking West Limit",
            "Tracking Actual Peak",
            "Tracking Predicted Peak",
            "Tracking Peak Error",
            "Tracking Block Error",
            "Tracking Energy Error",
            "Tracking Overall Score",
        ],

        "Value": [

            best_loss,
            actual_peak,
            fixed_metrics[
                "Peak Power"
            ],
            fixed_metrics[
                "Peak Error"
            ] * 100,
            fixed_metrics[
                "Block Error"
            ],
            fixed_metrics[
                "Energy Error"
            ],
            fixed_metrics[
                "Overall Score"
            ],

            DHI,
            start,
            end,
            max_block,
            east,
            west,
            actual_peak,
            tracking_metrics[
                "Peak Power"
            ],
            tracking_metrics[
                "Peak Error"
            ],
            tracking_metrics[
                "Block Error"
            ],
            tracking_metrics[
                "Energy Error"
            ],
            tracking_metrics[
                "Overall Score"
            ],
        ],
    })


# ============================================================
# 28. DOWNLOAD REPORT
# ============================================================

def create_excel_report(
    df,
    df_fix,
    df_trac,
    summary,
    results_df,
    optimized_parameters,
):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Area & Efficiency",
            index=False,
        )

        df_fix.to_excel(
            writer,
            sheet_name="Fixed Output",
            index=False,
        )

        df_trac.to_excel(
            writer,
            sheet_name="Tracking Output",
            index=False,
        )

        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        results_df.to_excel(
            writer,
            sheet_name="Efficiency Tests",
            index=False,
        )

        optimized_parameters.to_excel(
            writer,
            sheet_name="Optimized Parameters",
            index=False,
        )

    output.seek(0)

    return output


# ============================================================
# 29. MAIN
# ============================================================

def main():

    st.markdown(
        '<div class="main-title">'
        '☀️ Fixed vs Tracking Loss Correction Model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Upload the Excel workbook, edit GHI/Actual data "
        "and optimized parameters, then run the model."
        '</div>',
        unsafe_allow_html=True,
    )


    # ========================================================
    # INPUT EXCEL
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📁 Input Workbook'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=[
            "xlsx",
            "xls",
        ],
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload the plant Excel file to begin."
        )

        return


    # ========================================================
    # LOAD WORKBOOK
    # ========================================================

    try:

        detect_workbook(
            uploaded_file
        )

        df = read_area_efficiency(
            uploaded_file
        )

        fixed_weights, tracking_weights = (
            read_effective_areas(
                uploaded_file
            )
        )

        standard_efficiency = (
            read_standard_efficiency(
                df
            )
        )

        lat = read_latitude(
            uploaded_file
        )

        tilt_lookup = read_tilt_lookup(
            uploaded_file
        )

        (
            df_ghi,
            blocks_result,
            ghi_matrix,
        ) = read_ghi_data(
            uploaded_file
        )

        df_fix = read_fixed_data(
            uploaded_file
        )

        (
            df_fix,
            df_ghi,
            actual,
            ghi_matrix,
            blocks,
            dates,
        ) = prepare_model_data(
            df_fix,
            df_ghi,
            blocks_result,
            ghi_matrix,
        )

    except Exception as e:

        st.error(
            f"Unable to load workbook: {e}"
        )

        return


    # ========================================================
    # WORKBOOK INFORMATION
    # ========================================================

    with st.expander(
        "📋 Workbook Configuration",
        expanded=False,
    ):

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Latitude",
            f"{lat:.4f}",
        )

        c2.metric(
            "Forecast Blocks",
            len(blocks),
        )

        c3.metric(
            "Clusters",
            N_CLUSTERS,
        )

        c4.metric(
            "Max PV Efficiency",
            f"{standard_efficiency.min():.2f}%",
        )


    # ========================================================
    # INPUT DATA EDITOR
    # ========================================================

    edited = input_data_editor(
        df_fix,
        df_ghi,
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

        st.session_state.run_model = True
        st.session_state.tracking_params = None
        st.session_state.fixed_loss = None


    if not st.session_state.run_model:

        st.info(
            "Edit the input values if required and "
            "click **Run Loss Correction** to start."
        )

        return


    # ========================================================
    # UPDATE EDITED DATA
    # ========================================================

    actual = pd.to_numeric(
        edited["Actual"],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    for col in GHI_COLS:

        ghi_matrix[
            :len(edited),
            GHI_COLS.index(col)
        ] = pd.to_numeric(
            edited[col],
            errors="coerce",
        ).fillna(0).to_numpy(
            dtype=float
        )


    # ========================================================
    # ACTUAL METRICS
    # ========================================================

    try:

        (
            valid_mask,
            actual_day,
            actual_peak,
            actual_energy,
        ) = prepare_actual_metrics(
            actual
        )

    except Exception as e:

        st.error(
            str(e)
        )

        return


    # ========================================================
    # SOLAR CALCULATIONS
    # ========================================================

    (
        declination,
        elevation,
        tilt,
        sin_a_safe,
        sin_ab,
    ) = calculate_solar_angles(
        dates,
        lat,
        tilt_lookup,
    )


    fixed_poa = calculate_fixed_poa(
        ghi_matrix,
        sin_a_safe,
        sin_ab,
    )


    # ========================================================
    # FIXED OPTIMIZATION
    # ========================================================

    with st.spinner(
        "🔄 Optimizing Fixed efficiency loss..."
    ):

        try:

            (
                results_df,
                best_row,
                auto_best_loss,
            ) = optimize_fixed_loss(
                standard_efficiency,
                fixed_weights,
                fixed_poa,
                actual,
                valid_mask,
                actual_day,
                actual_peak,
                actual_energy,
            )

        except Exception as e:

            st.error(
                f"Fixed optimization failed: {e}"
            )

            return


    # ========================================================
    # FIXED LOSS CONTROL
    # ========================================================

    best_loss = fixed_loss_control(
        auto_best_loss,
        np.min(
            standard_efficiency
        ),
    )


    # ========================================================
    # FINAL FIXED MODEL
    # ========================================================

    (
        net_efficiency_fixed,
        final_fixed_weights,
        final_fixed_power_matrix,
        fixed_forecast,
    ) = calculate_final_fixed(
        standard_efficiency,
        fixed_weights,
        fixed_poa,
        best_loss,
    )


    fixed_metrics = calculate_metrics(
        fixed_forecast,
        actual_day,
        actual_peak,
        actual_energy,
        valid_mask,
    )


    # ========================================================
    # TRACKING OPTIMIZATION
    # ========================================================

    if st.session_state.tracking_params is None:

        with st.spinner(
            "🔄 Optimizing Tracking parameters..."
        ):

            try:

                st.session_state.tracking_params = (
                    optimize_tracking_cached(
                        tuple(blocks),
                        tuple(
                            ghi_matrix.flatten()
                        ),
                        tuple(
                            tracking_weights
                        ),
                        tuple(
                            actual_day
                        ),
                        float(actual_peak),
                        float(actual_energy),
                        tuple(valid_mask),
                    )
                )

            except Exception as e:

                st.error(
                    f"Tracking optimization failed: {e}"
                )

                return


    # ========================================================
    # TRACKING PARAMETER CONTROLS
    # ========================================================

    tracking_params = (
        tracking_parameter_controls(
            st.session_state.tracking_params
        )
    )


    # ========================================================
    # VALIDATE TRACKING PARAMETERS
    # ========================================================

    if not (
        tracking_params["start"]
        < tracking_params["max"]
        < tracking_params["end"]
    ):

        st.error(
            "Tracking parameters must satisfy: "
            "**Starting Block < Max Block < Ending Block**."
        )

        return


    # ========================================================
    # FINAL TRACKING MODEL
    # ========================================================

    try:

        (
            tracking_forecast,
            tracking_power_matrix,
            zenith,
            panel,
            dni,
        ) = calculate_tracking(
            tracking_params["DHI"],
            tracking_params["start"],
            tracking_params["end"],
            tracking_params["max"],
            tracking_params["east"],
            tracking_params["west"],
            blocks,
            ghi_matrix,
            tracking_weights,
        )

    except Exception as e:

        st.error(
            f"Tracking calculation failed: {e}"
        )

        return


    tracking_metrics = calculate_metrics(
        tracking_forecast,
        actual_day,
        actual_peak,
        actual_energy,
        valid_mask,
    )


    # ========================================================
    # UPDATE FIXED DATAFRAME
    # ========================================================

    df_fix_output = df_fix.copy()

    for i, cl in enumerate(
        CLUSTERS
    ):

        df_fix_output[
            f"{cl}_Fixed Power=I*Ƞ*A"
        ] = (
            final_fixed_power_matrix[
                :len(df_fix_output),
                i
            ]
        )

    df_fix_output[
        "Total Power (CL1+CL2+…)"
    ] = fixed_forecast

    df_fix_output[
        "Actual"
    ] = actual


    # ========================================================
    # UPDATE AREA & EFFICIENCY
    # ========================================================

    df_output = df.copy()

    if (
        "No of Module"
        in df_output.columns
        and
        "Area of 1 Module (m2)"
        in df_output.columns
    ):

        df_output[
            "Total area (m2)"
        ] = (
            pd.to_numeric(
                df_output[
                    "No of Module"
                ],
                errors="coerce",
            )
            *
            pd.to_numeric(
                df_output[
                    "Area of 1 Module (m2)"
                ],
                errors="coerce",
            )
        )

    df_output[
        "Error %"
    ] = best_loss

    df_output[
        "Net Efficiency (%)"
    ] = (
        pd.to_numeric(
            df_output[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce",
        )
        - best_loss
    )


    # ========================================================
    # UPDATE TRACKING DATAFRAME
    # ========================================================

    uploaded_file.seek(0)

    df_trac = pd.read_excel(
        uploaded_file,
        sheet_name="Tracking",
        header=1,
    )

    df_trac = df_trac.iloc[
        :len(blocks)
    ].copy()

    df_trac.reset_index(
        drop=True,
        inplace=True,
    )

    df_trac[
        "Zenith Angle"
    ] = zenith

    df_trac[
        "Panel Angle"
    ] = panel

    for i, cl in enumerate(
        CLUSTERS
    ):

        df_trac[
            f"{cl}_Tracking Power=I*Ƞ*A"
        ] = tracking_power_matrix[
            :len(df_trac),
            i
        ]

    df_trac[
        "Tracking Power=I*Ƞ*A"
    ] = tracking_forecast


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = build_summary(
        best_loss,
        tracking_params["DHI"],
        tracking_params["start"],
        tracking_params["end"],
        tracking_params["max"],
        tracking_params["east"],
        tracking_params["west"],
        fixed_metrics,
        tracking_metrics,
    )


    # ========================================================
    # OPTIMIZED PARAMETERS
    # ========================================================

    optimized_parameters = (
        build_optimized_parameters(
            best_loss,
            actual_peak,
            fixed_metrics,
            tracking_params["DHI"],
            tracking_params["start"],
            tracking_params["end"],
            tracking_params["max"],
            tracking_params["east"],
            tracking_params["west"],
            tracking_metrics,
        )
    )


    # ========================================================
    # OUTPUTS
    # ========================================================

    show_metrics(
        fixed_metrics,
        tracking_metrics,
    )

    show_forecast_chart(
        actual,
        fixed_forecast,
        tracking_forecast,
    )

    show_efficiency_results(
        results_df
    )

    show_fixed_output(
        df_fix_output
    )

    show_tracking_output(
        df_trac
    )


    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Fixed vs Tracking Summary'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        summary.round(6),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # OPTIMIZED PARAMETERS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '⚙️ Final Optimized Parameters'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        optimized_parameters.round(6),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    output = create_excel_report(
        df_output,
        df_fix_output,
        df_trac,
        summary,
        results_df,
        optimized_parameters,
    )

    st.markdown(
        '<div class="section-title">'
        '📥 Final Report'
        '</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇️ Download Final Excel Report",
        data=output,
        file_name=(
            "Fixed_vs_Tracking_"
            "Loss_Correction_Report.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        use_container_width=True,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
