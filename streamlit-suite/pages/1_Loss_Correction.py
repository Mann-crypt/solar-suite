# ============================================================
# STREAMLIT APP
# LOSS CORRECTION MODEL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.optimize import differential_evolution
from concurrent.futures import ThreadPoolExecutor
import time


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
    (0, 10),      # DHI %
    (0, 30),      # Starting block
    (65, 80),     # Ending block
    (44, 60),     # Max block
    (0, 70),      # East tracking limit
    (0, 70),      # West tracking limit
]

QUOTES = [
    "☕ Vo kehte the kya ho tum, aaj hum kehte hai tum kya ho be?",
    "🌦 Aapka mann nahi kar raha bahar jaane ka?",
    "😊 Jinke ghar sheeshe ke bane hote hai vo basement mai kapde change krte h...",
    "😋 Aromatic Rose Latte with Frothy Milk pine ka mann ho raha hai na...",
    "🥛 Garmi mai daalo dudh mai Ice 🧊 Dudh bangya Very Nice...",
    "🌟 Aapke face pr toh Modiji se bhi jyada glow hai..",
    "😁 Ho raha hai benstokes, kaan mai ghus jao insaan ke...",
    "😗 Muskuraiye, aap MAL mai hai...",
    "🥱 Hum na hote toh Operations ka kya hota?",
    "😎 6:30 hote hi Billu MAL se faraar...",
    "😇 Guruji ne ek baat kahi thi...",
    "🎼 Karna hai kuchh kaam, M se gaao...",
    "😠 Nahi karni Loss Correction, now what to do?",
    "💸 Iss job ko chhod aur chhod kar ameer ho...",
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "plant_type": "Fixed",
    "tracking_params": None,
    "model_context": None,
    "input_data": None,
    "input_context": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       MAIN HEADER
    -------------------------------------------------------- */

    .main-title {
        font-size: 2.3rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }


    /* --------------------------------------------------------
       SECTION TITLES
    -------------------------------------------------------- */

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 12px;
    }


    /* --------------------------------------------------------
       PLANT TYPE CARDS
    -------------------------------------------------------- */

    .plant-wrapper {
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .plant-label {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
    }


    /* Buttons work in both light and dark themes */

    div.stButton > button {
        width: 100%;
        min-height: 62px;
        border-radius: 14px;

        font-size: 16px;
        font-weight: 700;

        background-color: transparent;
        color: inherit;

        border: 1px solid rgba(128,128,128,0.35);

        transition:
            all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #4f8cff;
        color: #4f8cff;
        transform: translateY(-1px);
    }


    /* --------------------------------------------------------
       STATUS BOX
    -------------------------------------------------------- */

    .status-card {
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 15px;

        border: 1px solid rgba(128,128,128,0.25);
    }


    /* --------------------------------------------------------
       INFO CARDS
    -------------------------------------------------------- */

    .info-card {
        padding: 16px;
        border-radius: 14px;

        border: 1px solid rgba(128,128,128,0.25);

        margin-bottom: 15px;
    }


    /* --------------------------------------------------------
       METRIC CARDS
    -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 12px;
        border-radius: 12px;
    }


    /* --------------------------------------------------------
       DATA EDITOR
    -------------------------------------------------------- */

    div[data-testid="stDataEditor"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* --------------------------------------------------------
       DOWNLOAD BUTTON
    -------------------------------------------------------- */

    div.stDownloadButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 650;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def validate_columns(df, required_columns, dataframe_name="Data"):

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required column(s): "
            f"{', '.join(missing)}"
        )


def clean_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


def get_sheet_names(uploaded_file):

    uploaded_file.seek(0)

    excel = pd.ExcelFile(uploaded_file)

    return excel.sheet_names


# ============================================================
# CLUSTER DETECTION
# ============================================================

def detect_cluster_model(uploaded_file):

    sheets = get_sheet_names(uploaded_file)

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

    return df.reset_index(drop=True)


# ============================================================
# CLUSTER WEIGHTS
# ============================================================

def read_cluster_weights(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[12, 13, 14, 15, 16],
    )

    df = clean_columns(df)

    required = [
        "CL-1",
        "CL-2",
        "CL-3",
        "CL-4",
        "CL-5",
    ]

    validate_columns(
        df,
        required,
        "Cluster Weights",
    )

    return {
        col: float(df[col].iloc[0])
        for col in required
    }


# ============================================================
# LATITUDE
# ============================================================

def read_latitude(uploaded_file):

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

    return float(df["Lat"].iloc[0])


# ============================================================
# TILT LOOKUP
# ============================================================

def read_tilt_lookup(uploaded_file):

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

    return (
        df
        .dropna(subset=["Month"])
        .set_index("Month")["Fixed"]
        .to_dict()
    )


# ============================================================
# CLEAN DATA ROWS
# ============================================================

def clean_data_rows(
    df,
    date_column="Date",
):

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

    df_fix["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + day_number)
                / 365
            )
        )
    )

    df_fix["Elevation angle a"] = (
        90
        - lat
        + df_fix[
            "Declination Angle ∆"
        ]
    )

    if tracking:

        df_fix["Tilt Angle b"] = 0

    else:

        if not tilt_lookup:

            df_fix["Tilt Angle b"] = 0

        else:

            df_fix["Tilt Angle b"] = (
                df_fix["Date"]
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

    standard_eff = (
        pd.to_numeric(
            df[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    area = (
        pd.to_numeric(
            df[
                "Total area(m2)"
            ],
            errors="coerce",
        )
        .to_numpy(dtype=float)
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

    actual_peak = np.nanmax(
        valid_actual
    )

    poa_peak = np.nanmax(
        valid_poa
    )

    if poa_peak <= 0:
        return 0.0

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

    max_loss = float(
        np.nanmin(
            standard_eff
        )
    )

    best_loss = np.clip(
        best_loss,
        0,
        max_loss,
    )

    return float(best_loss)


def apply_efficiency_loss(
    df,
    loss,
):

    df = df.copy()

    df[
        "Efficiency Losses(%)"
    ] = loss

    df[
        "Net Efficiency (%)"
    ] = (
        df[
            "Standard PV Efficiency (%)"
        ]
        - loss
    )

    df["Eff Area"] = (
        df["Total area(m2)"]
        * df["Net Efficiency (%)"]
        / 100
    )

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

        ghi_cols = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

        weight_cols = [
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
            ghi_cols,
            weight_cols,
        ):

            poa = (
                df_fix[ghi_col]
                * df_fix["SIN(a+b)"]
                / df_fix["Sin(a)"]
            )

            eff_area = (
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * cluster_weights[
                    weight_col
                ]
            ).sum()

            forecast += (
                poa.to_numpy()
                * eff_area
                / 1_000_000
            )

        return forecast

    validate_columns(
        df_fix,
        ["GHI_Forecast"],
        "Fixed Forecast",
    )

    poa = (
        df_fix["GHI_Forecast"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    return (
        poa.to_numpy()
        * df["Eff Area"].sum()
        / 1_000_000
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    blocks,
    weighted_ghi,
    actual,
):

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    weighted_ghi = np.asarray(
        weighted_ghi,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        & np.isfinite(weighted_ghi)
        & (actual != 0)
    )

    actual_day = actual[mask]
    ghi_day = weighted_ghi[mask]
    blocks_day = blocks[mask]

    if len(actual_day) == 0:

        raise ValueError(
            "No valid Actual power values found."
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
            start < max_block < end
        ):
            return 1e9

        denominator_1 = (
            start - 1 - max_block
        )

        denominator_2 = (
            end + 1 - max_block
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
            1 - DHI / 100
        )

        prediction = (
            ghi_day
            * dhi_factor
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

        return float(
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
        "score": float(result.fun),
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

def calculate_tracking_forecast(
    blocks,
    weighted_ghi,
    params,
):

    DHI = params["DHI"]
    start = params["start"]
    end = params["end"]
    max_block = params["max"]
    east = params["east"]
    west = params["west"]

    if not (
        start < max_block < end
    ):
        raise ValueError(
            "Starting Block < Max Block < Ending Block required."
        )

    denominator_1 = (
        start - 1 - max_block
    )

    denominator_2 = (
        end + 1 - max_block
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

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    weighted_ghi = np.asarray(
        weighted_ghi,
        dtype=float,
    )

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

    return (
        weighted_ghi
        * (1 - DHI / 100)
        / cos_alpha
        / 1_000_000
    )


# ============================================================
# PLANT TYPE BUTTONS
# ============================================================

def plant_type_selector():

    st.markdown(
        '<div class="plant-label">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        fixed_label = (
            "🏗️  FIXED PLANT"
            if st.session_state.plant_type != "Fixed"
            else "✅  FIXED PLANT  •  SELECTED"
        )

        if st.button(
            fixed_label,
            key="fixed_button",
            use_container_width=True,
        ):

            st.session_state.plant_type = "Fixed"
            st.session_state.tracking_params = None
            st.rerun()

    with col2:

        tracking_label = (
            "🔄  TRACKING PLANT"
            if st.session_state.plant_type != "Tracking"
            else "✅  TRACKING PLANT  •  SELECTED"
        )

        if st.button(
            tracking_label,
            key="tracking_button",
            use_container_width=True,
        ):

            st.session_state.plant_type = "Tracking"
            st.session_state.tracking_params = None
            st.rerun()

    if st.session_state.plant_type == "Fixed":

        st.success(
            "🏗️ Fixed plant selected"
        )

    else:

        st.info(
            "🔄 Tracking plant selected"
        )

    return st.session_state.plant_type


# ============================================================
# INPUT DATA EDITOR
# ============================================================

def edit_input_data(
    df_fix,
    editable_columns,
):

    st.markdown(
        "### ✏️ Input Data"
    )

    st.caption(
        "Only GHI Forecast and Actual values are editable. "
        "All other parameters are extracted automatically from the workbook."
    )

    display_columns = [
        col
        for col in editable_columns
        if col in df_fix.columns
    ]

    if not display_columns:
        st.warning(
            "No editable GHI/Actual columns were found."
        )
        return df_fix

    editor_df = df_fix[
        display_columns
    ].copy()

    for col in display_columns:

        editor_df[col] = pd.to_numeric(
            editor_df[col],
            errors="coerce",
        )

    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=False,
        num_rows="fixed",
        key="input_data_editor",
        column_config={
            col: st.column_config.NumberColumn(
                col,
                format="%.2f",
            )
            for col in display_columns
        },
    )

    return edited_df


# ============================================================
# EFFICIENCY DISPLAY
# ============================================================

def show_efficiency_table(df):

    columns = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)",
    ]

    columns = [
        col for col in columns
        if col in df.columns
    ]

    display_df = df[
        columns
    ].copy()

    numeric_cols = (
        display_df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    display_df[numeric_cols] = (
        display_df[numeric_cols]
        .round(2)
    )

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
# READ NON-CLUSTER FIXED
# ============================================================

def read_noncluster_fixed(
    uploaded_file,
):

    uploaded_file.seek(0)

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed",
        header=1,
    )

    df_fix = clean_columns(
        df_fix
    )

    df_fix = clean_data_rows(
        df_fix
    )

    validate_columns(
        df_fix,
        [
            "GHI_Forecast",
            "Actual",
        ],
        "Fixed",
    )

    df_fix["GHI_Forecast"] = pd.to_numeric(
        df_fix["GHI_Forecast"],
        errors="coerce",
    ).fillna(0)

    df_fix["Actual"] = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

    return df_fix


# ============================================================
# READ CLUSTER DATA
# ============================================================

def read_cluster_fixed(
    uploaded_file,
):

    uploaded_file.seek(0)

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-CL1",
        header=1,
    )

    df_fix = clean_columns(
        df_fix
    )

    df_fix = clean_data_rows(
        df_fix
    )

    validate_columns(
        df_fix,
        ["Actual"],
        "Fixed-CL1",
    )

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

    # Try Result sheet if cluster GHI
    # columns are not already in Fixed-CL1

    missing_ghi = [
        col for col in ghi_columns
        if col not in df_fix.columns
    ]

    if missing_ghi:

        uploaded_file.seek(0)

        try:

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
                    col not in df_fix.columns
                    and i < len(result.columns)
                ):

                    values = (
                        pd.to_numeric(
                            result.iloc[
                                :len(df_fix),
                                i,
                            ],
                            errors="coerce",
                        )
                        .fillna(0)
                        .to_numpy()
                    )

                    if len(values) < len(df_fix):

                        values = np.pad(
                            values,
                            (
                                0,
                                len(df_fix)
                                - len(values),
                            ),
                            constant_values=0,
                        )

                    df_fix[col] = values

        except Exception:
            pass

    validate_columns(
        df_fix,
        ghi_columns,
        "Cluster GHI",
    )

    df_fix["Actual"] = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

    for col in ghi_columns:

        df_fix[col] = pd.to_numeric(
            df_fix[col],
            errors="coerce",
        ).fillna(0)

    return df_fix


# ============================================================
# TRACKING BACKEND
# ============================================================

def read_tracking_blocks(
    uploaded_file,
    cluster=False,
):

    uploaded_file.seek(0)

    if cluster:

        sheet = "Backend Cal CL1"

    else:

        sheet = "Backend Cal"

    df = pd.read_excel(
        uploaded_file,
        sheet_name=sheet,
    )

    df = clean_columns(df)

    validate_columns(
        df,
        ["Block No."],
        sheet,
    )

    return df[
        "Block No."
    ].to_numpy(
        dtype=float
    )


# ============================================================
# TRACKING OPTIMIZATION UI
# ============================================================

def run_tracking_optimization(
    blocks,
    weighted_ghi,
    actual,
):

    st.markdown(
        "### ⚙️ Tracking Optimization"
    )

    progress = st.progress(0)

    status = st.empty()

    state = {
        "generation": 0
    }

    def callback(
        generation,
        total,
    ):

        state[
            "generation"
        ] = generation

    with ThreadPoolExecutor(
        max_workers=1
    ) as executor:

        future = executor.submit(
            optimize_tracking,
            blocks,
            weighted_ghi,
            actual,
        )

        last_generation = -1

        while not future.done():

            generation = state[
                "generation"
            ]

            if generation != last_generation:

                last_generation = generation

                progress_value = min(
                    generation
                    / MAX_OPT_ITER,
                    0.99,
                )

                progress.progress(
                    progress_value
                )

                quote = QUOTES[
                    generation
                    % len(QUOTES)
                ]

                status.info(
                    f"{quote}\n\n"
                    f"Generation {generation} / "
                    f"{MAX_OPT_ITER}"
                )

            time.sleep(0.1)

        result = future.result()

    progress.progress(1.0)

    status.success(
        "✅ Tracking optimization completed."
    )

    time.sleep(0.4)

    progress.empty()
    status.empty()

    return result


# ============================================================
# NON-CLUSTER FIXED
# ============================================================

def process_noncluster_fixed(
    uploaded_file,
    df,
    lat,
    tilt_lookup,
):

    df_fix = read_noncluster_fixed(
        uploaded_file
    )

    # --------------------------------------------------------
    # USER EDITABLE INPUT
    # --------------------------------------------------------

    edited = edit_input_data(
        df_fix,
        [
            "GHI_Forecast",
            "Actual",
        ],
    )

    df_fix[
        "GHI_Forecast"
    ] = edited[
        "GHI_Forecast"
    ].to_numpy()

    df_fix[
        "Actual"
    ] = edited[
        "Actual"
    ].to_numpy()

    # --------------------------------------------------------
    # SOLAR CALCULATION
    # --------------------------------------------------------

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=False,
    )

    df_fix["POA fixed"] = (
        df_fix["GHI_Forecast"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    # --------------------------------------------------------
    # AUTOMATIC LOSS
    # --------------------------------------------------------

    loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    df = apply_efficiency_loss(
        df,
        loss,
    )

    forecast = (
        df_fix["POA fixed"].to_numpy()
        * df["Eff Area"].sum()
        / 1_000_000
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Efficiency Loss",
        f"{loss:.2f}%",
    )

    c2.metric(
        "Peak Actual",
        f"{np.max(df_fix['Actual']):.2f} MW",
    )

    c3.metric(
        "Peak Forecast",
        f"{np.max(forecast):.2f} MW",
    )

    show_efficiency_table(
        df
    )

    show_forecast_chart(
        forecast,
        df_fix["Actual"].to_numpy(),
        "🏗️ Fixed Forecast vs Actual",
    )


# ============================================================
# NON-CLUSTER TRACKING
# ============================================================

def process_noncluster_tracking(
    uploaded_file,
    df,
    lat,
    tilt_lookup,
):

    df_fix = read_noncluster_fixed(
        uploaded_file
    )

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    edited = edit_input_data(
        df_fix,
        [
            "GHI_Forecast",
            "Actual",
        ],
    )

    df_fix[
        "GHI_Forecast"
    ] = edited[
        "GHI_Forecast"
    ].to_numpy()

    df_fix[
        "Actual"
    ] = edited[
        "Actual"
    ].to_numpy()

    # --------------------------------------------------------
    # SOLAR ANGLES
    # --------------------------------------------------------

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=True,
    )

    df_fix["POA fixed"] = (
        df_fix["GHI_Forecast"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    # --------------------------------------------------------
    # AUTOMATIC EFFICIENCY LOSS
    # --------------------------------------------------------

    loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    df = apply_efficiency_loss(
        df,
        loss,
    )

    weighted_ghi = (
        df_fix["GHI_Forecast"].to_numpy(
            dtype=float
        )
        * df["Eff Area"].sum()
    )

    # --------------------------------------------------------
    # BACKEND BLOCKS
    # --------------------------------------------------------

    blocks = read_tracking_blocks(
        uploaded_file,
        cluster=False,
    )

    actual = df_fix[
        "Actual"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    context = (
        uploaded_file.name,
        uploaded_file.size,
        "noncluster_tracking",
        len(df_fix),
    )

    if (
        st.session_state.tracking_params is None
        or st.session_state.model_context
        != context
    ):

        result = run_tracking_optimization(
            blocks,
            weighted_ghi,
            actual,
        )

        st.session_state.tracking_params = result
        st.session_state.model_context = context

    params = st.session_state.tracking_params

    # --------------------------------------------------------
    # SHOW OPTIMIZED PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Optimized Tracking Parameters"
    )

    st.caption(
        "These parameters are automatically optimized from the uploaded GHI and Actual data."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "DHI",
        f"{params['DHI']}%",
    )

    c2.metric(
        "Starting Block",
        params["start"],
    )

    c3.metric(
        "Ending Block",
        params["end"],
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Max Block",
        params["max"],
    )

    c2.metric(
        "East Limit",
        f"{params['east']}°",
    )

    c3.metric(
        "West Limit",
        f"{params['west']}°",
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            params,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Efficiency Loss",
            f"{loss:.2f}%",
        )

        c2.metric(
            "Peak Actual",
            f"{np.max(actual):.2f} MW",
        )

        c3.metric(
            "Peak Forecast",
            f"{np.max(forecast):.2f} MW",
        )

        show_efficiency_table(
            df
        )

        show_forecast_chart(
            forecast,
            actual,
            "🔄 Tracking Forecast vs Actual",
        )

    except Exception as e:

        st.error(
            f"Unable to calculate tracking forecast: {e}"
        )


# ============================================================
# CLUSTER FIXED
# ============================================================

def process_cluster_fixed(
    uploaded_file,
    df,
    lat,
    tilt_lookup,
):

    cluster_weights = read_cluster_weights(
        uploaded_file
    )

    df_fix = read_cluster_fixed(
        uploaded_file
    )

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

    # --------------------------------------------------------
    # USER EDITABLE INPUT
    # --------------------------------------------------------

    editable_columns = (
        ghi_columns
        + ["Actual"]
    )

    edited = edit_input_data(
        df_fix,
        editable_columns,
    )

    for col in editable_columns:

        if col in edited.columns:

            df_fix[col] = (
                pd.to_numeric(
                    edited[col],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy()
            )

    # --------------------------------------------------------
    # SOLAR
    # --------------------------------------------------------

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=False,
    )

    df_fix["POA fixed"] = (
        df_fix["CL1-GHI"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    df = apply_efficiency_loss(
        df,
        loss,
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast = calculate_fixed_forecast(
        df,
        df_fix,
        cluster=True,
        cluster_weights=cluster_weights,
    )

    actual = df_fix[
        "Actual"
    ].to_numpy(
        dtype=float
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Efficiency Loss",
        f"{loss:.2f}%",
    )

    c2.metric(
        "Peak Actual",
        f"{np.max(actual):.2f} MW",
    )

    c3.metric(
        "Peak Forecast",
        f"{np.max(forecast):.2f} MW",
    )

    show_efficiency_table(
        df
    )

    show_forecast_chart(
        forecast,
        actual,
        "🏗️ Fixed Cluster Forecast vs Actual",
    )


# ============================================================
# CLUSTER TRACKING
# ============================================================

def process_cluster_tracking(
    uploaded_file,
    df,
    lat,
    tilt_lookup,
):

    cluster_weights = read_cluster_weights(
        uploaded_file
    )

    df_fix = read_cluster_fixed(
        uploaded_file
    )

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    edited = edit_input_data(
        df_fix,
        ghi_columns + ["Actual"],
    )

    for col in ghi_columns + ["Actual"]:

        df_fix[col] = (
            pd.to_numeric(
                edited[col],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy()
        )

    # --------------------------------------------------------
    # SOLAR ANGLES
    # --------------------------------------------------------

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=True,
    )

    df_fix["POA fixed"] = (
        df_fix["CL1-GHI"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    # --------------------------------------------------------
    # AUTOMATIC LOSS
    # --------------------------------------------------------

    loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    df = apply_efficiency_loss(
        df,
        loss,
    )

    # --------------------------------------------------------
    # WEIGHTED GHI
    # --------------------------------------------------------

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
            df["Total area(m2)"]
            * df["Net Efficiency (%)"]
            / 100
            * cluster_weights[
                weight_key
            ]
        ).sum()

        weighted_ghi += (
            ghi
            * eff_area
        )

    # --------------------------------------------------------
    # BACKEND
    # --------------------------------------------------------

    blocks = read_tracking_blocks(
        uploaded_file,
        cluster=True,
    )

    actual = df_fix[
        "Actual"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    context = (
        uploaded_file.name,
        uploaded_file.size,
        "cluster_tracking",
        len(df_fix),
    )

    if (
        st.session_state.tracking_params is None
        or st.session_state.model_context
        != context
    ):

        result = run_tracking_optimization(
            blocks,
            weighted_ghi,
            actual,
        )

        st.session_state.tracking_params = result
        st.session_state.model_context = context

    params = st.session_state.tracking_params

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Optimized Tracking Parameters"
    )

    st.caption(
        "These values are automatically optimized from your GHI and Actual inputs."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "DHI",
        f"{params['DHI']}%",
    )

    c2.metric(
        "Starting Block",
        params["start"],
    )

    c3.metric(
        "Ending Block",
        params["end"],
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Max Block",
        params["max"],
    )

    c2.metric(
        "East Limit",
        f"{params['east']}°",
    )

    c3.metric(
        "West Limit",
        f"{params['west']}°",
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            params,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Efficiency Loss",
            f"{loss:.2f}%",
        )

        c2.metric(
            "Peak Actual",
            f"{np.max(actual):.2f} MW",
        )

        c3.metric(
            "Peak Forecast",
            f"{np.max(forecast):.2f} MW",
        )

        show_efficiency_table(
            df
        )

        show_forecast_chart(
            forecast,
            actual,
            "🔄 Tracking Cluster Forecast vs Actual",
        )

    except Exception as e:

        st.error(
            f"Unable to calculate tracking forecast: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">'
        "☀️ Loss Correction Model"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Upload your Excel workbook. Plant configuration and "
        "model parameters are extracted automatically."
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    st.markdown(
        "### 📁 Upload Plant Data"
    )

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx", "xls"],
        help=(
            "Upload the complete plant Excel workbook. "
            "You only need to edit GHI and Actual values inside the app."
        ),
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload your Excel workbook to start the model."
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

    if is_cluster:

        st.info(
            "🏢 **Cluster workbook detected**"
            "\n\n"
            "The workbook does not contain a `Fixed` sheet."
        )

    else:

        st.success(
            "🏭 **Non-cluster workbook detected**"
            "\n\n"
            "The workbook contains a `Fixed` sheet."
        )

    # ========================================================
    # PLANT TYPE
    # ========================================================

    plant_type = plant_type_selector()

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
            f"Unable to load plant configuration: {e}"
        )

        st.stop()

    # ========================================================
    # MODEL CONTEXT
    # ========================================================

    current_context = (
        uploaded_file.name,
        uploaded_file.size,
        plant_type,
        is_cluster,
    )

    if (
        st.session_state.model_context
        != current_context
    ):

        st.session_state.tracking_params = None
        st.session_state.model_context = (
            current_context
        )

    # ========================================================
    # SHOW EXTRACTED PARAMETERS
    # ========================================================

    with st.expander(
        "📋 View Extracted Plant Configuration",
        expanded=False,
    ):

        c1, c2 = st.columns(2)

        c1.metric(
            "Plant Type",
            plant_type,
        )

        c2.metric(
            "Latitude",
            f"{lat:.4f}°",
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # MODEL EXECUTION
    # ========================================================

    try:

        if not is_cluster:

            if plant_type == "Fixed":

                process_noncluster_fixed(
                    uploaded_file,
                    df,
                    lat,
                    tilt_lookup,
                )

            else:

                process_noncluster_tracking(
                    uploaded_file,
                    df,
                    lat,
                    tilt_lookup,
                )

        else:

            if plant_type == "Fixed":

                process_cluster_fixed(
                    uploaded_file,
                    df,
                    lat,
                    tilt_lookup,
                )

            else:

                process_cluster_tracking(
                    uploaded_file,
                    df,
                    lat,
                    tilt_lookup,
                )

    except Exception as e:

        st.error(
            "❌ Model calculation failed."
        )

        st.exception(e)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
