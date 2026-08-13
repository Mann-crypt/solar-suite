# ============================================================
# STREAMLIT APP
# LOSS CORRECTION MODEL
# NON-FREEZING VERSION
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
    page_title="Loss Correction Model",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_OPT_ITER = 30
OPT_POPSIZE = 8

PARAM_BOUNDS = [
    (0, 10),      # DHI
    (0, 30),      # Starting Block
    (65, 80),     # Ending Block
    (44, 60),     # Max Block
    (0, 70),      # East Limit
    (0, 70),      # West Limit
]

TRACKING_KEYS = [
    "DHI",
    "start",
    "end",
    "max",
    "east",
    "west",
]

QUOTES = [
    "☀️ Solar power loading...",
    "🔍 Finding the best correction parameters...",
    "📊 Comparing forecast against actual...",
    "⚙️ Optimizing tracking parameters...",
    "📈 Checking peak and energy error...",
    "🧮 Calculating the best combination...",
    "🌤️ Understanding the plant behavior...",
    "🚀 Almost there...",
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "plant_type": "🏗️ Fixed",
    "model_context": None,
    "tracking_params": None,
    "tracking_optimization_done": False,
    "input_data": None,
    "input_data_context": None,
    "auto_loss": None,
    "manual_loss": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ------------------------------
       Main header
    ------------------------------ */

    .main-title {
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* ------------------------------
       Section titles
    ------------------------------ */

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }

    /* ------------------------------
       Plant selection buttons
       Theme-safe
    ------------------------------ */

    .plant-label {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        min-height: 58px;
        border-radius: 14px;
        font-size: 16px;
        font-weight: 700;

        background-color: transparent;
        color: inherit;

        border: 1px solid rgba(128, 128, 128, 0.45);

        transition:
            transform 0.15s ease,
            border-color 0.15s ease,
            background-color 0.15s ease;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
        background-color: rgba(59, 130, 246, 0.08);
    }

    div[data-testid="stButton"] > button:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.18);
    }

    /* ------------------------------
       Cards
       ------------------------------ */

    .info-card {
        padding: 15px 18px;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin: 10px 0 16px 0;
    }

    .success-card {
        padding: 14px 18px;
        border-radius: 14px;
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.25);
    }

    .warning-card {
        padding: 14px 18px;
        border-radius: 14px;
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    /* ------------------------------
       Metrics
       ------------------------------ */

    div[data-testid="stMetric"] {
        border-radius: 14px;
        padding: 10px;
    }

    /* ------------------------------
       Hide excessive Streamlit spacing
       ------------------------------ */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def validate_columns(df, required_columns, dataframe_name="Data"):

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required "
            f"column(s): {', '.join(missing)}"
        )


def clean_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


def clean_data_rows(df, date_column="Date"):

    df = df.copy()

    if date_column in df.columns:

        null_indices = df[
            df[date_column].isna()
        ].index

        if len(null_indices) > 0:

            first_null_pos = (
                df.index.get_loc(
                    null_indices[0]
                )
            )

            df = df.iloc[
                :first_null_pos
            ]

    return df.reset_index(drop=True)


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


# ============================================================
# EXCEL SHEETS
# ============================================================

def get_sheet_names(uploaded_file):

    uploaded_file.seek(0)

    return pd.ExcelFile(
        uploaded_file
    ).sheet_names


def detect_cluster_model(uploaded_file):

    sheets = get_sheet_names(
        uploaded_file
    )

    return "Fixed" not in sheets


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def read_area_efficiency(
    uploaded_file,
    cluster=False,
):

    uploaded_file.seek(0)

    if cluster:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Area & Efficiency",
            header=1,
            usecols=range(8),
        )

    else:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Area & Efficiency",
            header=1,
        )

    df = clean_columns(df)

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

        null_indices = df[
            df["Module Type"].isna()
        ].index

        if len(null_indices) > 0:

            first_null_pos = (
                df.index.get_loc(
                    null_indices[0]
                )
            )

            df = df.iloc[
                :first_null_pos
            ]

    df = df.dropna(
        subset=[
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        how="all",
    )

    df[
        "Standard PV Efficiency (%)"
    ] = safe_numeric(
        df[
            "Standard PV Efficiency (%)"
        ]
    )

    df["Total area(m2)"] = safe_numeric(
        df["Total area(m2)"]
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# CLUSTER WEIGHTS
# ============================================================

def read_cluster_weights(
    uploaded_file,
):

    uploaded_file.seek(0)

    df_w = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[12, 13, 14, 15, 16],
    )

    df_w = clean_columns(df_w)

    required = [
        "CL-1",
        "CL-2",
        "CL-3",
        "CL-4",
        "CL-5",
    ]

    validate_columns(
        df_w,
        required,
        "Cluster Weights",
    )

    weights = {}

    for col in required:

        value = pd.to_numeric(
            df_w[col].iloc[0],
            errors="coerce",
        )

        weights[col] = (
            0.0
            if pd.isna(value)
            else float(value)
        )

    return weights


# ============================================================
# LATITUDE
# ============================================================

def read_latitude(
    uploaded_file,
):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8,
    )

    df = clean_columns(df)

    validate_columns(
        df,
        ["Lat"],
        "Forecast Config",
    )

    value = pd.to_numeric(
        df["Lat"].iloc[0],
        errors="coerce",
    )

    if pd.isna(value):
        raise ValueError(
            "Latitude value is invalid."
        )

    return float(value)


# ============================================================
# TILT LOOKUP
# ============================================================

def read_tilt_lookup(
    uploaded_file,
):

    uploaded_file.seek(0)

    try:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Config Tilt Angle",
            header=7,
        )

    except Exception:

        return {}

    df = clean_columns(df)

    if "Fixed" not in df.columns:
        return {}

    null_indices = df[
        df["Fixed"].isna()
    ].index

    if len(null_indices) > 0:

        first_null_pos = (
            df.index.get_loc(
                null_indices[0]
            )
        )

        df = df.iloc[
            :first_null_pos
        ]

    df = df.dropna(
        how="all",
        axis=1,
    )

    df = df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    if (
        "Month" not in df.columns
        or "Fixed" not in df.columns
    ):
        return {}

    result = {}

    for _, row in df.dropna(
        subset=["Month"]
    ).iterrows():

        month = str(
            row["Month"]
        ).strip()

        value = pd.to_numeric(
            row["Fixed"],
            errors="coerce",
        )

        if not pd.isna(value):

            result[month] = float(value)

    return result


# ============================================================
# SOLAR ANGLES
# ============================================================

def prepare_solar_angles(
    df_fix,
    lat,
    tilt_lookup=None,
    tracking=False,
):

    df_fix = df_fix.copy()

    today = pd.Timestamp.today().normalize()

    df_fix["Date"] = today

    first_date = today.replace(
        month=1,
        day=1,
    )

    day_number = (
        df_fix["Date"]
        - first_date
    ).dt.days + 1

    df_fix[
        "Declination Angle ∆"
    ] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + day_number)
                / 365
            )
        )
    )

    df_fix[
        "Elevation angle a"
    ] = (
        90
        - lat
        + df_fix[
            "Declination Angle ∆"
        ]
    )

    if tracking:

        df_fix[
            "Tilt Angle b"
        ] = 0.0

    else:

        if not tilt_lookup:

            df_fix[
                "Tilt Angle b"
            ] = 0.0

        else:

            df_fix[
                "Tilt Angle b"
            ] = (
                df_fix[
                    "Date"
                ]
                .dt.strftime("%B")
                .map(tilt_lookup)
                .fillna(0)
            )

    df_fix["a+b"] = (
        df_fix["Elevation angle a"]
        + df_fix["Tilt Angle b"]
    )

    df_fix["SIN(a+b)"] = np.sin(
        np.radians(
            df_fix["a+b"]
        )
    )

    df_fix["Sin(a)"] = np.sin(
        np.radians(
            df_fix["Elevation angle a"]
        )
    )

    df_fix["Sin(a)"] = (
        df_fix["Sin(a)"]
        .clip(lower=1e-6)
    )

    return df_fix


# ============================================================
# EFFICIENCY LOSS
# ============================================================

def calculate_efficiency_loss(
    df,
    poa,
    actual,
):

    standard_eff = df[
        "Standard PV Efficiency (%)"
    ].to_numpy(
        dtype=float
    )

    area = df[
        "Total area(m2)"
    ].to_numpy(
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    poa = np.asarray(
        poa,
        dtype=float,
    )

    valid_actual = actual[
        np.isfinite(actual)
    ]

    valid_poa = poa[
        np.isfinite(poa)
    ]

    if len(valid_actual) == 0:
        return 0.0

    if len(valid_poa) == 0:
        return 0.0

    poa_peak = float(
        np.nanmax(valid_poa)
    )

    actual_peak = float(
        np.nanmax(valid_actual)
    )

    if poa_peak <= 0:
        return 0.0

    max_loss = float(
        np.nanmin(
            standard_eff
        )
    )

    base_area = np.sum(
        area
        * standard_eff
        / 100
    )

    loss_area_coefficient = np.sum(
        area / 100
    )

    if loss_area_coefficient <= 0:
        return 0.0

    target_area = (
        actual_peak
        * 1_000_000
        / poa_peak
    )

    best_loss = (
        base_area
        - target_area
    ) / loss_area_coefficient

    best_loss = np.clip(
        best_loss,
        0,
        max_loss,
    )

    return float(
        best_loss
    )


def apply_efficiency_loss(
    df,
    loss,
):

    df = df.copy()

    df[
        "Efficiency Losses(%)"
    ] = float(loss)

    df[
        "Net Efficiency (%)"
    ] = (
        df[
            "Standard PV Efficiency (%)"
        ]
        - float(loss)
    )

    df["Eff Area"] = (
        df[
            "Total area(m2)"
        ]
        * df[
            "Net Efficiency (%)"
        ]
        / 100
    )

    return df


# ============================================================
# LOAD NON-CLUSTER FIXED DATA
# ============================================================

def load_noncluster_data(
    uploaded_file,
):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed",
        header=1,
    )

    df = clean_columns(df)

    df = clean_data_rows(df)

    validate_columns(
        df,
        [
            "GHI_Forecast",
            "Actual",
        ],
        "Fixed",
    )

    df["GHI_Forecast"] = safe_numeric(
        df["GHI_Forecast"]
    )

    df["Actual"] = safe_numeric(
        df["Actual"]
    )

    return df


# ============================================================
# LOAD CLUSTER DATA
# ============================================================

def load_cluster_data(
    uploaded_file,
):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-CL1",
        header=1,
    )

    df = clean_columns(df)

    df = clean_data_rows(df)

    validate_columns(
        df,
        ["Actual"],
        "Fixed-CL1",
    )

    df["Actual"] = safe_numeric(
        df["Actual"]
    )

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

    # ------------------------------------------
    # Try Result sheet if cluster GHI columns
    # are not already present
    # ------------------------------------------

    missing_ghi = [
        col
        for col in ghi_columns
        if col not in df.columns
    ]

    if missing_ghi:

        try:

            uploaded_file.seek(0)

            result = pd.read_excel(
                uploaded_file,
                sheet_name="Result",
                usecols=range(6),
            )

            result = result.fillna(0)

            for i, col in enumerate(
                ghi_columns
            ):

                if (
                    col not in df.columns
                    and i < result.shape[1]
                ):

                    values = (
                        result.iloc[
                            :len(df),
                            i
                        ]
                        .to_numpy()
                    )

                    if len(values) < len(df):

                        values = np.pad(
                            values,
                            (
                                0,
                                len(df)
                                - len(values),
                            ),
                            constant_values=0,
                        )

                    df[col] = values

        except Exception:
            pass

    validate_columns(
        df,
        ghi_columns,
        "Cluster GHI",
    )

    for col in ghi_columns:

        df[col] = safe_numeric(
            df[col]
        )

    return df


# ============================================================
# USER INPUT TABLE
# ============================================================

def show_input_editor(
    df,
    cluster=False,
):

    st.markdown(
        '<div class="section-title">✏️ Input Data</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "You can manually change only the GHI forecast values and Actual power. "
        "All other workbook parameters remain protected."
    )

    if cluster:

        editable_columns = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
            "Actual",
        ]

    else:

        editable_columns = [
            "GHI_Forecast",
            "Actual",
        ]

    display_df = df[
        editable_columns
    ].copy()

    display_df.insert(
        0,
        "Block",
        np.arange(
            1,
            len(display_df) + 1
        ),
    )

    edited = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key=(
            "cluster_input_editor"
            if cluster
            else "fixed_input_editor"
        ),
        column_config={
            "Block": st.column_config.NumberColumn(
                "Block",
                disabled=True,
            ),
        },
    )

    edited = edited.copy()

    edited["Block"] = np.arange(
        1,
        len(edited) + 1
    )

    return edited


# ============================================================
# APPLY USER INPUT CHANGES
# ============================================================

def apply_user_input(
    original_df,
    edited_df,
    cluster=False,
):

    df = original_df.copy()

    if cluster:

        ghi_columns = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

        for col in ghi_columns:

            if col in edited_df.columns:

                df[col] = safe_numeric(
                    edited_df[col]
                ).to_numpy()

        df["Actual"] = safe_numeric(
            edited_df["Actual"]
        ).to_numpy()

    else:

        df["GHI_Forecast"] = safe_numeric(
            edited_df["GHI_Forecast"]
        ).to_numpy()

        df["Actual"] = safe_numeric(
            edited_df["Actual"]
        ).to_numpy()

    return df


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df,
    df_fix,
    cluster=False,
    cluster_weights=None,
):

    if cluster:

        ghi_columns = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

        weight_columns = [
            "CL-1",
            "CL-2",
            "CL-3",
            "CL-4",
            "CL-5",
        ]

        forecast = np.zeros(
            len(df_fix),
            dtype=float,
        )

        for ghi_col, weight_col in zip(
            ghi_columns,
            weight_columns,
        ):

            poa = (
                df_fix[ghi_col].to_numpy(
                    dtype=float
                )
                * df_fix[
                    "SIN(a+b)"
                ].to_numpy(
                    dtype=float
                )
                / df_fix[
                    "Sin(a)"
                ].to_numpy(
                    dtype=float
                )
            )

            eff_area = (
                df["Eff Area"]
                * cluster_weights[
                    weight_col
                ]
            ).sum()

            forecast += (
                poa
                * eff_area
                / 1_000_000
            )

        return forecast

    poa = (
        df_fix["GHI_Forecast"].to_numpy(
            dtype=float
        )
        * df_fix["SIN(a+b)"].to_numpy(
            dtype=float
        )
        / df_fix["Sin(a)"].to_numpy(
            dtype=float
        )
    )

    return (
        poa
        * df["Eff Area"].sum()
        / 1_000_000
    )


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    blocks,
    weighted_ghi,
    params,
):

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    weighted_ghi = np.asarray(
        weighted_ghi,
        dtype=float,
    )

    DHI = int(params["DHI"])
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
            "Required condition: "
            "Starting Block < Max Block < Ending Block."
        )

    denominator_1 = (
        start
        - 1
        - max_block
    )

    denominator_2 = (
        end
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

    m1 = 90 / denominator_1
    m2 = 90 / denominator_2

    zenith = np.where(
        blocks <= max_block,

        np.minimum(
            89,
            m1 * (
                blocks
                - max_block
            ),
        ),

        np.minimum(
            89,
            m2 * (
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
                (blocks > max_block)
                & (zenith > west)
            ),
            west,
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

    dhi_factor = (
        1
        - DHI / 100
    )

    forecast = (
        weighted_ghi
        * dhi_factor
        / cos_alpha
        / 1_000_000
    )

    return forecast


# ============================================================
# TRACKING OPTIMIZATION
# IMPORTANT:
# This function has NO Streamlit calls.
# Therefore it is safe to cache.
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_tracking_cached(
    blocks_tuple,
    weighted_ghi_tuple,
    actual_tuple,
    maxiter=MAX_OPT_ITER,
    popsize=OPT_POPSIZE,
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
        np.isfinite(actual)
        & np.isfinite(weighted_ghi)
        & (actual != 0)
    )

    actual_day = actual[mask]
    weighted_day = weighted_ghi[mask]
    blocks_day = blocks[mask]

    if len(actual_day) == 0:
        raise ValueError(
            "No valid non-zero Actual power values found."
        )

    actual_peak = np.max(
        actual_day
    )

    actual_energy = np.sum(
        actual_day
    )

    if actual_peak <= 0:
        raise ValueError(
            "Actual peak power is zero."
        )

    if actual_energy <= 0:
        raise ValueError(
            "Actual energy is zero."
        )

    def objective(x):

        DHI = int(round(x[0]))
        start = int(round(x[1]))
        end = int(round(x[2]))
        max_block = int(round(x[3]))
        east = int(round(x[4]))
        west = int(round(x[5]))

        if not (
            start
            < max_block
            < end
        ):
            return 1e9

        denominator_1 = (
            start
            - 1
            - max_block
        )

        denominator_2 = (
            end
            + 1
            - max_block
        )

        if (
            denominator_1 == 0
            or denominator_2 == 0
        ):
            return 1e9

        m1 = 90 / denominator_1
        m2 = 90 / denominator_2

        zenith = np.where(
            blocks_day <= max_block,

            np.minimum(
                89,
                m1
                * (
                    blocks_day
                    - max_block
                ),
            ),

            np.minimum(
                89,
                m2
                * (
                    blocks_day
                    - max_block
                ),
            ),
        )

        panel = np.where(
            blocks_day < max_block,

            np.minimum(
                zenith,
                abs(east),
            ),

            np.where(
                (
                    (blocks_day > max_block)
                    & (
                        zenith > west
                    )
                ),
                west,
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

        prediction = (
            weighted_day
            * (1 - DHI / 100)
            / cos_alpha
            / 1_000_000
        )

        if (
            np.isnan(prediction).any()
            or np.isinf(prediction).any()
        ):
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual_day
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

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return float(score)

    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
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
        "score": float(
            result.fun
        ),
        "DHI": int(best[0]),
        "start": int(best[1]),
        "end": int(best[2]),
        "max": int(best[3]),
        "east": int(best[4]),
        "west": int(best[5]),
    }


# ============================================================
# RUN OPTIMIZATION
# User explicitly starts it.
# No background thread.
# ============================================================

def run_tracking_optimization(
    blocks,
    weighted_ghi,
    actual,
):

    st.markdown(
        "### ⚙️ Tracking Optimization"
    )

    st.info(
        "Optimization runs only after you click the button. "
        "This prevents Streamlit from repeatedly running the optimizer."
    )

    if not st.button(
        "🚀 Run Tracking Optimization",
        use_container_width=True,
        type="primary",
        key="run_tracking_optimizer",
    ):
        return None

    progress = st.progress(
        0,
        text="Preparing optimization...",
    )

    status = st.empty()

    # ------------------------------------------
    # Simpler and safer than ThreadPoolExecutor
    # ------------------------------------------

    try:

        status.info(
            "⚙️ Optimizing tracking parameters..."
        )

        progress.progress(
            0.15,
            text="Running optimizer..."
        )

        result = optimize_tracking_cached(
            tuple(
                np.asarray(
                    blocks,
                    dtype=float,
                )
            ),
            tuple(
                np.asarray(
                    weighted_ghi,
                    dtype=float,
                )
            ),
            tuple(
                np.asarray(
                    actual,
                    dtype=float,
                )
            ),
        )

        progress.progress(
            1.0,
            text="Optimization completed."
        )

        status.success(
            "✅ Tracking optimization completed."
        )

        return result

    except Exception as e:

        progress.empty()

        status.error(
            f"Optimization failed: {e}"
        )

        return None


# ============================================================
# BACKEND BLOCKS
# ============================================================

def read_blocks(
    uploaded_file,
    sheet_name,
):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name,
    )

    df = clean_columns(df)

    validate_columns(
        df,
        ["Block No."],
        sheet_name,
    )

    return safe_numeric(
        df["Block No."]
    ).to_numpy(
        dtype=float
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

    available = [
        col
        for col in columns
        if col in df.columns
    ]

    display_df = df[
        available
    ].copy()

    numeric_cols = (
        display_df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    display_df[
        numeric_cols
    ] = display_df[
        numeric_cols
    ].round(3)

    with st.expander(
        "🔍 View Efficiency Calculations",
        expanded=False,
    ):

        st.dataframe(
            display_df,
            use_container_width=True,
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

    n = min(
        len(forecast),
        len(actual),
    )

    forecast = np.asarray(
        forecast[:n],
        dtype=float,
    )

    actual = np.asarray(
        actual[:n],
        dtype=float,
    )

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
                width=3,
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
                color="#DC2626",
                width=3,
            ),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        hovermode="x unified",

        xaxis=dict(
            title="15 Minute Block",
            dtick=4,
        ),

        yaxis=dict(
            title="Power (MW)",
        ),

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
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# PLANT TYPE BUTTONS
# ============================================================

def plant_type_selector():

    st.markdown(
        '<div class="plant-label">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        2,
        gap="medium",
    )

    with col1:

        if st.button(
            "🏗️  FIXED PLANT",
            key="fixed_plant_button",
            use_container_width=True,
        ):

            st.session_state.plant_type = (
                "🏗️ Fixed"
            )

            st.session_state.tracking_params = None
            st.session_state.tracking_optimization_done = False

            st.rerun()

    with col2:

        if st.button(
            "🔄  TRACKING PLANT",
            key="tracking_plant_button",
            use_container_width=True,
        ):

            st.session_state.plant_type = (
                "🔄 Tracking"
            )

            st.session_state.tracking_params = None
            st.session_state.tracking_optimization_done = False

            st.rerun()

    current = st.session_state.plant_type

    if current == "🏗️ Fixed":

        st.markdown(
            """
            <div class="success-card">
            <b>🏗️ Fixed Plant Selected</b><br>
            Efficiency loss will be automatically calculated and can then be adjusted manually.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="info-card">
            <b>🔄 Tracking Plant Selected</b><br>
            Tracking parameters will be optimized only when you start optimization.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return current


# ============================================================
# FIXED PLANT
# ============================================================

def process_fixed(
    uploaded_file,
    df,
    input_df,
    lat,
    tilt_lookup,
    cluster=False,
):

    if cluster:

        cluster_weights = (
            read_cluster_weights(
                uploaded_file
            )
        )

        df_fix = input_df.copy()

        df_fix = prepare_solar_angles(
            df_fix,
            lat,
            tilt_lookup,
            tracking=False,
        )

        poa = (
            df_fix["CL1-GHI"].to_numpy(
                dtype=float
            )
            * df_fix[
                "SIN(a+b)"
            ].to_numpy(
                dtype=float
            )
            / df_fix[
                "Sin(a)"
            ].to_numpy(
                dtype=float
            )
        )

    else:

        df_fix = input_df.copy()

        df_fix = prepare_solar_angles(
            df_fix,
            lat,
            tilt_lookup,
            tracking=False,
        )

        df_fix[
            "POA fixed"
        ] = (
            df_fix[
                "GHI_Forecast"
            ].to_numpy(
                dtype=float
            )
            * df_fix[
                "SIN(a+b)"
            ].to_numpy(
                dtype=float
            )
            / df_fix[
                "Sin(a)"
            ].to_numpy(
                dtype=float
            )
        )

        poa = df_fix[
            "POA fixed"
        ].to_numpy(
            dtype=float
        )

    # ------------------------------------------
    # AUTO LOSS
    # ------------------------------------------

    auto_loss = calculate_efficiency_loss(
        df,
        poa,
        df_fix[
            "Actual"
        ].to_numpy(
            dtype=float
        ),
    )

    max_loss = float(
        df[
            "Standard PV Efficiency (%)"
        ].min()
    )

    # ------------------------------------------
    # LOSS INPUT
    # ------------------------------------------

    st.markdown(
        "### 📉 Efficiency Loss"
    )

    st.caption(
        "The model calculates an initial efficiency loss automatically. "
        "You can manually change it below."
    )

    if st.session_state.manual_loss is None:

        st.session_state.manual_loss = (
            float(auto_loss)
        )

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=max_loss,
        value=float(
            st.session_state.manual_loss
        ),
        step=0.1,
        format="%.2f",
        key=(
            "fixed_loss_input"
            if not cluster
            else "cluster_fixed_loss_input"
        ),
    )

    st.session_state.manual_loss = loss

    df = apply_efficiency_loss(
        df,
        loss,
    )

    # ------------------------------------------
    # FORECAST
    # ------------------------------------------

    if cluster:

        forecast = calculate_fixed_forecast(
            df,
            df_fix,
            cluster=True,
            cluster_weights=cluster_weights,
        )

    else:

        forecast = calculate_fixed_forecast(
            df,
            df_fix,
            cluster=False,
        )

    c1, c2 = st.columns(2)

    c1.metric(
        "🤖 Auto Loss",
        f"{auto_loss:.2f}%",
    )

    c2.metric(
        "✏️ Loss Used",
        f"{loss:.2f}%",
    )

    show_efficiency_table(
        df
    )

    show_forecast_chart(
        forecast,
        df_fix[
            "Actual"
        ].to_numpy(
            dtype=float
        ),
        (
            "🏗️ Fixed Cluster Forecast vs Actual"
            if cluster
            else
            "🏗️ Fixed Forecast vs Actual"
        ),
    )


# ============================================================
# TRACKING PLANT
# ============================================================

def process_tracking(
    uploaded_file,
    df,
    input_df,
    lat,
    tilt_lookup,
    cluster=False,
):

    if cluster:

        cluster_weights = (
            read_cluster_weights(
                uploaded_file
            )
        )

        df_fix = input_df.copy()

        ghi_columns = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

        df_fix = prepare_solar_angles(
            df_fix,
            lat,
            tilt_lookup,
            tracking=True,
        )

        # --------------------------------------
        # Weighted GHI
        # --------------------------------------

        weighted_ghi = np.zeros(
            len(df_fix),
            dtype=float,
        )

        for ghi_col, weight_key in zip(
            ghi_columns,
            [
                "CL-1",
                "CL-2",
                "CL-3",
                "CL-4",
                "CL-5",
            ],
        ):

            ghi = df_fix[
                ghi_col
            ].to_numpy(
                dtype=float
            )

            eff_area = (
                df[
                    "Total area(m2)"
                ]
                * df[
                    "Standard PV Efficiency (%)"
                ]
                / 100
                * cluster_weights[
                    weight_key
                ]
            ).sum()

            weighted_ghi += (
                ghi
                * eff_area
            )

        backend_sheets = [
            "Backend Cal CL1",
            "Backend Cal CL2",
            "Backend Cal CL3",
            "Backend Cal CL4",
            "Backend Cal CL5",
        ]

        uploaded_file.seek(0)

        backend = pd.read_excel(
            uploaded_file,
            sheet_name=backend_sheets[0],
        )

        backend = clean_columns(
            backend
        )

        validate_columns(
            backend,
            ["Block No."],
            backend_sheets[0],
        )

        blocks = safe_numeric(
            backend[
                "Block No."
            ]
        ).to_numpy(
            dtype=float
        )

    else:

        df_fix = input_df.copy()

        df_fix = prepare_solar_angles(
            df_fix,
            lat,
            tilt_lookup,
            tracking=True,
        )

        weighted_ghi = (
            df_fix[
                "GHI_Forecast"
            ].to_numpy(
                dtype=float
            )
            * (
                df[
                    "Total area(m2)"
                ]
                * df[
                    "Standard PV Efficiency (%)"
                ]
                / 100
            ).sum()
        )

        blocks = read_blocks(
            uploaded_file,
            "Backend Cal",
        )

    # ------------------------------------------
    # AUTO EFFICIENCY LOSS
    # ------------------------------------------

    poa_reference = (
        (
            df_fix[
                "CL1-GHI"
            ].to_numpy(
                dtype=float
            )
            if cluster
            else
            df_fix[
                "GHI_Forecast"
            ].to_numpy(
                dtype=float
            )
        )
        * df_fix[
            "SIN(a+b)"
        ].to_numpy(
            dtype=float
        )
        / df_fix[
            "Sin(a)"
        ].to_numpy(
            dtype=float
        )
    )

    auto_loss = calculate_efficiency_loss(
        df,
        poa_reference,
        df_fix[
            "Actual"
        ].to_numpy(
            dtype=float
        ),
    )

    max_loss = float(
        df[
            "Standard PV Efficiency (%)"
        ].min()
    )

    # ------------------------------------------
    # EFFICIENCY LOSS
    # ------------------------------------------

    st.markdown(
        "### 📉 Efficiency Loss"
    )

    st.caption(
        "The initial loss is calculated automatically. "
        "You have full manual control over the value."
    )

    if st.session_state.manual_loss is None:

        st.session_state.manual_loss = (
            float(auto_loss)
        )

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=max_loss,
        value=float(
            st.session_state.manual_loss
        ),
        step=0.1,
        format="%.2f",
        key=(
            "tracking_loss_input"
            if not cluster
            else "cluster_tracking_loss_input"
        ),
    )

    st.session_state.manual_loss = loss

    df = apply_efficiency_loss(
        df,
        loss,
    )

    # ------------------------------------------
    # RECALCULATE WEIGHTED GHI USING
    # USER-SELECTED EFFICIENCY LOSS
    # ------------------------------------------

    if cluster:

        weighted_ghi = np.zeros(
            len(df_fix),
            dtype=float,
        )

        for ghi_col, weight_key in zip(
            [
                "CL1-GHI",
                "CL2-GHI",
                "CL3-GHI",
                "CL4-GHI",
                "CL5-GHI",
            ],
            [
                "CL-1",
                "CL-2",
                "CL-3",
                "CL-4",
                "CL-5",
            ],
        ):

            ghi = df_fix[
                ghi_col
            ].to_numpy(
                dtype=float
            )

            eff_area = (
                df["Eff Area"]
                * cluster_weights[
                    weight_key
                ]
            ).sum()

            weighted_ghi += (
                ghi
                * eff_area
            )

    else:

        weighted_ghi = (
            df_fix[
                "GHI_Forecast"
            ].to_numpy(
                dtype=float
            )
            * df[
                "Eff Area"
            ].sum()
        )

    actual = df_fix[
        "Actual"
    ].to_numpy(
        dtype=float
    )

    # ------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------

    if (
        st.session_state.tracking_params
        is None
    ):

        result = run_tracking_optimization(
            blocks,
            weighted_ghi,
            actual,
        )

        if result is not None:

            st.session_state.tracking_params = (
                result
            )

            st.session_state.tracking_optimization_done = (
                True
            )

            st.rerun()

        else:

            st.warning(
                "Click **Run Tracking Optimization** "
                "to calculate the initial tracking parameters."
            )

            return

    params = (
        st.session_state.tracking_params
    )

    # ------------------------------------------
    # PARAMETER EDITING
    # ------------------------------------------

    st.markdown(
        "### ⚙️ Tracking Parameters"
    )

    st.caption(
        "These values were calculated automatically. "
        "You can manually change them."
    )

    c1, c2, c3 = st.columns(3)

    DHI = c1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=10,
        value=int(params["DHI"]),
        step=1,
        key=(
            "tracking_DHI"
            if not cluster
            else "cluster_tracking_DHI"
        ),
    )

    start = c2.number_input(
        "Starting Block",
        min_value=0,
        max_value=30,
        value=int(params["start"]),
        step=1,
        key=(
            "tracking_start"
            if not cluster
            else "cluster_tracking_start"
        ),
    )

    end = c3.number_input(
        "Ending Block",
        min_value=65,
        max_value=80,
        value=int(params["end"]),
        step=1,
        key=(
            "tracking_end"
            if not cluster
            else "cluster_tracking_end"
        ),
    )

    c1, c2, c3 = st.columns(3)

    max_block = c1.number_input(
        "Max Block",
        min_value=44,
        max_value=60,
        value=int(params["max"]),
        step=1,
        key=(
            "tracking_max"
            if not cluster
            else "cluster_tracking_max"
        ),
    )

    east = c2.number_input(
        "East Limit",
        min_value=0,
        max_value=70,
        value=int(params["east"]),
        step=1,
        key=(
            "tracking_east"
            if not cluster
            else "cluster_tracking_east"
        ),
    )

    west = c3.number_input(
        "West Limit",
        min_value=0,
        max_value=70,
        value=int(params["west"]),
        step=1,
        key=(
            "tracking_west"
            if not cluster
            else "cluster_tracking_west"
        ),
    )

    final_params = {
        "DHI": int(DHI),
        "start": int(start),
        "end": int(end),
        "max": int(max_block),
        "east": int(east),
        "west": int(west),
    }

    # ------------------------------------------
    # FORECAST
    # ------------------------------------------

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            final_params,
        )

    except Exception as e:

        st.error(
            f"❌ Invalid tracking parameters: {e}"
        )

        return

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🤖 Auto Loss",
        f"{auto_loss:.2f}%",
    )

    c2.metric(
        "✏️ Loss Used",
        f"{loss:.2f}%",
    )

    c3.metric(
        "🎯 Optimizer Score",
        f"{params['score']:.5f}",
    )

    show_efficiency_table(
        df
    )

    show_forecast_chart(
        forecast,
        actual,
        (
            "🔄 Tracking Cluster Forecast vs Actual"
            if cluster
            else
            "🔄 Tracking Forecast vs Actual"
        ),
    )


# ============================================================
# RESET WHEN NEW FILE / MODEL IS SELECTED
# ============================================================

def handle_context(
    uploaded_file,
    plant_type,
    is_cluster,
):

    context = (
        uploaded_file.name,
        uploaded_file.size,
        plant_type,
        is_cluster,
    )

    if (
        st.session_state.model_context
        != context
    ):

        st.session_state.model_context = (
            context
        )

        st.session_state.tracking_params = None
        st.session_state.tracking_optimization_done = False
        st.session_state.manual_loss = None
        st.session_state.input_data = None
        st.session_state.input_data_context = None


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">☀️ Loss Correction Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Upload the plant workbook. Configuration is extracted automatically, "
        "while GHI Forecast and Actual remain editable."
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # INPUT DATA
    # ========================================================

    st.markdown(
        "### 📁 Input Data"
    )

    uploaded_file = st.file_uploader(
        "Upload Excel Workbook",
        type=[
            "xlsx",
            "xls",
        ],
        help=(
            "The workbook should contain Area & Efficiency, "
            "Forecast Config and the relevant Fixed / Backend sheets."
        ),
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload your Excel workbook to begin."
        )

        st.stop()

    # ========================================================
    # DETECT WORKBOOK
    # ========================================================

    try:

        sheets = get_sheet_names(
            uploaded_file
        )

        is_cluster = (
            "Fixed" not in sheets
        )

    except Exception as e:

        st.error(
            f"Unable to read workbook: {e}"
        )

        st.stop()

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    try:

        df = read_area_efficiency(
            uploaded_file,
            cluster=is_cluster,
        )

        lat = read_latitude(
            uploaded_file
        )

        tilt_lookup = read_tilt_lookup(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to load configuration: {e}"
        )

        st.stop()

    # ========================================================
    # LOAD USER-EDITABLE INPUT
    # ========================================================

    input_context = (
        uploaded_file.name,
        uploaded_file.size,
        is_cluster,
    )

    if (
        st.session_state.input_data_context
        != input_context
    ):

        try:

            if is_cluster:

                st.session_state.input_data = (
                    load_cluster_data(
                        uploaded_file
                    )
                )

            else:

                st.session_state.input_data = (
                    load_noncluster_data(
                        uploaded_file
                    )
                )

            st.session_state.input_data_context = (
                input_context
            )

        except Exception as e:

            st.error(
                f"Unable to load input data: {e}"
            )

            st.stop()

    original_input = (
        st.session_state.input_data
    )

    # ========================================================
    # INPUT EDITOR
    # ========================================================

    edited_input = show_input_editor(
        original_input,
        cluster=is_cluster,
    )

    # ========================================================
    # APPLY USER CHANGES
    # ========================================================

    try:

        input_df = apply_user_input(
            original_input,
            edited_input,
            cluster=is_cluster,
        )

    except Exception as e:

        st.error(
            f"Invalid input data: {e}"
        )

        st.stop()

    # ========================================================
    # PLANT TYPE
    # BELOW INPUT DATA AS REQUESTED
    # ========================================================

    st.markdown("---")

    plant_type = plant_type_selector()

    handle_context(
        uploaded_file,
        plant_type,
        is_cluster,
    )

    # ========================================================
    # WORKBOOK STATUS
    # ========================================================

    if is_cluster:

        st.info(
            "🏢 **Cluster workbook detected**"
        )

    else:

        st.success(
            "🏭 **Non-cluster workbook detected**"
        )

    # ========================================================
    # MODEL EXECUTION
    # ========================================================

    st.markdown("---")

    try:

        if plant_type == "🏗️ Fixed":

            process_fixed(
                uploaded_file,
                df,
                input_df,
                lat,
                tilt_lookup,
                cluster=is_cluster,
            )

        else:

            process_tracking(
                uploaded_file,
                df,
                input_df,
                lat,
                tilt_lookup,
                cluster=is_cluster,
            )

    except Exception as e:

        st.error(
            "❌ Model calculation failed."
        )

        with st.expander(
            "Technical details"
        ):

            st.exception(e)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
