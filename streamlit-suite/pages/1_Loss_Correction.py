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
    (0, 10),     # DHI %
    (0, 30),     # Starting block
    (65, 80),    # Ending block
    (44, 60),    # Max block
    (0, 70),     # East tracking limit
    (0, 70),     # West tracking limit
]

QUOTES = [
    "☕ Vo kehte the kya ho tum, aaj hum kehte hai tum kya ho be?",
    "🌦 Aapka mann nahi kar raha bahar jaane ka?..",
    "😊 Jinke ghar sheeshe ke bane hote hai vo basement mai kapde change krte h...",
    "😋 Aromatic Rose Latte with Frothy Milk pine ka mann hor hai na...",
    "🥛 Garmi mai daalo dudh mai Ice 🧊 Dudh bangya Very Nice...",
    "🌟 Aapke face pr toh Modiji se bhi jyada glow hai...",
    "😁 Horaha hai benstokes Kaan mai ghusjao insaan ke...",
    "😗 Muskuraiye aap MAL mai hai...",
    "🥱 Hum na hote toh Operations ka kya hota?..",
    "😎 6:30 hote hi Billu MAL se faraar...",
    "😇 Guruji ne ek baat kahi thi....",
    "🎼 Karna hai kuchh kaam M se gaao...",
    "😠 Nahi karni Loss Correction, Now what to do?...",
    "💸 Iss Job ko chhod or chhod kar ameer ho..",
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "plant_type": "Fixed",
    "input_mode": "Excel Workbook",
    "tracking_params": None,
    "model_context": None,

    # Manual efficiency losses
    "manual_fixed_loss": None,
    "manual_tracking_loss": None,

    # Manual data
    "manual_input_df": None,
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

    /* =====================================================
       GENERAL
       ===================================================== */

    .main-title {
        font-size: 2.3rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 12px;
    }


    /* =====================================================
       PLANT TYPE CARDS
       ===================================================== */

    .plant-card {
        border: 2px solid #334155;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
        background: #111827;
        transition: all 0.2s ease;
        margin-bottom: 8px;
    }

    .plant-card:hover {
        border-color: #60a5fa;
        transform: translateY(-2px);
    }

    .plant-card-selected {
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
        background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.22),
            rgba(59, 130, 246, 0.08)
        );
        box-shadow: 0 0 0 1px rgba(59,130,246,0.15);
        margin-bottom: 8px;
    }

    .plant-icon {
        font-size: 32px;
        margin-bottom: 5px;
    }

    .plant-title {
        font-size: 18px;
        font-weight: 750;
    }

    .plant-description {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 5px;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    div.stButton > button {
        min-height: 46px;
        border-radius: 12px;
        font-weight: 650;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #3b82f6;
        color: #60a5fa;
    }


    /* =====================================================
       METRIC / STATUS CARDS
       ===================================================== */

    .status-card {
        padding: 15px 18px;
        border-radius: 14px;
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid #334155;
        margin-bottom: 15px;
    }


    /* =====================================================
       INPUT DATA CARD
       ===================================================== */

    .input-card {
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #334155;
        background: rgba(15, 23, 42, 0.65);
        margin-bottom: 15px;
    }


    /* =====================================================
       EXPANDER
       ===================================================== */

    [data-testid="stExpander"] {
        border-radius: 12px;
    }


    /* =====================================================
       NUMBER INPUT
       ===================================================== */

    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }


    /* =====================================================
       DATAFRAME
       ===================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER
# ============================================================

def validate_columns(
    df,
    required_columns,
    dataframe_name="Data",
):

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


# ============================================================
# SHEET HELPERS
# ============================================================

def get_sheet_names(uploaded_file):

    uploaded_file.seek(0)

    excel_file = pd.ExcelFile(
        uploaded_file
    )

    return excel_file.sheet_names


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

    null_indices = (
        df[
            df["Module Type"].isna()
        ].index
    )

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
    ] = pd.to_numeric(
        df[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    )

    df[
        "Total area(m2)"
    ] = pd.to_numeric(
        df[
            "Total area(m2)"
        ],
        errors="coerce",
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

    df_w.columns = (
        df_w.columns
        .astype(str)
        .str.strip()
    )

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

    return {
        col: float(
            pd.to_numeric(
                df_w[col].iloc[0],
                errors="coerce",
            )
        )
        for col in required
    }


# ============================================================
# LATITUDE
# ============================================================

def read_latitude(
    uploaded_file,
):

    uploaded_file.seek(0)

    df_st = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8,
    )

    df_st.columns = (
        df_st.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df_st,
        ["Lat"],
        "Forecast Config",
    )

    return float(
        pd.to_numeric(
            df_st["Lat"].iloc[0],
            errors="coerce",
        )
    )


# ============================================================
# TILT
# ============================================================

def read_tilt_lookup(
    uploaded_file,
):

    uploaded_file.seek(0)

    try:

        df_tilt = pd.read_excel(
            uploaded_file,
            sheet_name="Config Tilt Angle",
            header=7,
        )

    except Exception:

        return {}

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" not in df_tilt.columns:

        return {}

    df_tilt = df_tilt.dropna(
        how="all"
    )

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    if (
        "Month" not in df_tilt.columns
        or "Fixed" not in df_tilt.columns
    ):

        return {}

    return (
        df_tilt
        .dropna(
            subset=["Month"]
        )
        .set_index("Month")["Fixed"]
        .to_dict()
    )


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data_rows(
    df,
    date_column="Date",
):

    df = df.copy()

    if date_column in df.columns:

        null_indices = (
            df[
                df[date_column].isna()
            ].index
        )

        if len(null_indices) > 0:

            first_null_pos = (
                df.index.get_loc(
                    null_indices[0]
                )
            )

            df = df.iloc[
                :first_null_pos
            ]

    return df.reset_index(
        drop=True
    )


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
        ] = 0

    else:

        if not tilt_lookup:

            df_fix[
                "Tilt Angle b"
            ] = 0

        else:

            df_fix[
                "Tilt Angle b"
            ] = (
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
            df_fix[
                "Elevation angle a"
            ]
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
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    area = (
        pd.to_numeric(
            df[
                "Total area(m2)"
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    poa = np.asarray(
        poa,
        dtype=float,
    )

    valid = (
        np.isfinite(actual)
        & np.isfinite(poa)
        & (actual >= 0)
        & (poa > 0)
    )

    if not valid.any():

        return 0.0

    actual_valid = actual[valid]
    poa_valid = poa[valid]

    actual_peak = np.max(
        actual_valid
    )

    poa_peak = np.max(
        poa_valid
    )

    if actual_peak <= 0 or poa_peak <= 0:

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

    best_loss = np.clip(
        best_loss,
        0,
        np.min(standard_eff),
    )

    return float(
        best_loss
    )


# ============================================================
# APPLY EFFICIENCY LOSS
# ============================================================

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

    df["Net Efficiency (%)"] = (
        df["Net Efficiency (%)"]
        .clip(lower=0)
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
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df,
    df_fix,
    cluster=False,
    cluster_weights=None,
):

    df = df.copy()
    df_fix = df_fix.copy()

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

        validate_columns(
            df_fix,
            ghi_cols,
            "Cluster Forecast",
        )

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

    else:

        validate_columns(
            df_fix,
            ["GHI_Forecast"],
            "Fixed Forecast",
        )

        df_fix[
            "POA fixed"
        ] = (
            df_fix["GHI_Forecast"]
            * df_fix["SIN(a+b)"]
            / df_fix["Sin(a)"]
        )

        forecast = (
            df_fix["POA fixed"].to_numpy()
            * df["Eff Area"].sum()
            / 1_000_000
        )

    return forecast


# ============================================================
# TRACKING CALCULATION
# ============================================================

def tracking_core(
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
        start < max_block < end
    ):

        raise ValueError(
            "Starting Block < Max Block < Ending Block is required."
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

    m1 = (
        90 / denominator_1
    )

    m2 = (
        90 / denominator_2
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

    dhi_factor = (
        1 - DHI / 100
    )

    forecast = (
        weighted_ghi
        * dhi_factor
        / cos_alpha
        / 1_000_000
    )

    return forecast


def calculate_tracking_forecast(
    blocks,
    weighted_ghi,
    params,
):

    return tracking_core(
        blocks,
        weighted_ghi,
        params,
    )


# ============================================================
# TRACKING OPTIMIZER
# ============================================================

def optimize_tracking(
    blocks,
    weighted_ghi,
    actual,
    maxiter=40,
    popsize=10,
    progress_callback=None,
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
    weighted_ghi_day = weighted_ghi[mask]
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

        params = {
            "DHI": int(round(x[0])),
            "start": int(round(x[1])),
            "end": int(round(x[2])),
            "max": int(round(x[3])),
            "east": int(round(x[4])),
            "west": int(round(x[5])),
        }

        try:

            prediction = tracking_core(
                blocks_day,
                weighted_ghi_day,
                params,
            )

        except Exception:

            return 1e9

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

    generation = {
        "count": 0
    }

    def callback(
        xk,
        convergence,
    ):

        generation["count"] += 1

        if progress_callback:

            progress_callback(
                generation["count"],
                maxiter,
            )

        return False

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

        callback=callback,

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

    progress_bar = st.progress(
        0
    )

    status_box = st.empty()

    progress_state = {
        "generation": 0
    }

    def update_progress(
        generation,
        total,
    ):

        progress_state[
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
            MAX_OPT_ITER,
            OPT_POPSIZE,
            update_progress,
        )

        last_generation = -1

        while not future.done():

            generation = (
                progress_state[
                    "generation"
                ]
            )

            if (
                generation
                != last_generation
            ):

                last_generation = (
                    generation
                )

                progress = min(
                    generation
                    / MAX_OPT_ITER,
                    0.99,
                )

                progress_bar.progress(
                    progress
                )

                quote = QUOTES[
                    generation
                    % len(QUOTES)
                ]

                status_box.info(
                    f"{quote}\n\n"
                    f"Generation "
                    f"{generation} / "
                    f"{MAX_OPT_ITER}"
                )

            time.sleep(
                0.10
            )

        result = future.result()

    progress_bar.progress(
        1.0
    )

    status_box.success(
        "✅ Tracking optimization completed!"
    )

    time.sleep(
        0.5
    )

    progress_bar.empty()
    status_box.empty()

    return result


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
    ]

    display_df = df[
        columns
    ].copy()

    num_cols = (
        display_df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    display_df[
        num_cols
    ] = (
        display_df[
            num_cols
        ].round(2)
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
                color="#3B82F6",
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
                color="#EF4444",
                width=3,
            ),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
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
# PLANT TYPE SELECTOR
# ============================================================

def plant_type_selector():

    st.markdown(
        '<div class="section-title">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    current = st.session_state.plant_type

    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    with col1:

        if current == "Fixed":

            st.markdown(
                """
                <div class="plant-card-selected">
                    <div class="plant-icon">🏗️</div>
                    <div class="plant-title">Fixed Plant</div>
                    <div class="plant-description">
                        Fixed-tilt solar plant
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="plant-card">
                    <div class="plant-icon">🏗️</div>
                    <div class="plant-title">Fixed Plant</div>
                    <div class="plant-description">
                        Fixed-tilt solar plant
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button(
            "Select Fixed",
            key="select_fixed",
            use_container_width=True,
        ):

            st.session_state.plant_type = "Fixed"

            st.session_state.tracking_params = None

            st.rerun()

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    with col2:

        if current == "Tracking":

            st.markdown(
                """
                <div class="plant-card-selected">
                    <div class="plant-icon">🔄</div>
                    <div class="plant-title">Tracking Plant</div>
                    <div class="plant-description">
                        Single-axis tracking solar plant
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="plant-card">
                    <div class="plant-icon">🔄</div>
                    <div class="plant-title">Tracking Plant</div>
                    <div class="plant-description">
                        Single-axis tracking solar plant
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button(
            "Select Tracking",
            key="select_tracking",
            use_container_width=True,
        ):

            st.session_state.plant_type = "Tracking"

            st.session_state.tracking_params = None

            st.rerun()

    return st.session_state.plant_type


# ============================================================
# INPUT MODE
# ============================================================

def input_mode_selector():

    st.markdown(
        '<div class="section-title">📥 Input Data</div>',
        unsafe_allow_html=True,
    )

    input_mode = st.segmented_control(
        "Select input method",
        [
            "Excel Workbook",
            "Manual Data",
        ],
        default=st.session_state.input_mode,
        key="input_mode_control",
    )

    if input_mode is None:

        input_mode = "Excel Workbook"

    st.session_state.input_mode = input_mode

    return input_mode


# ============================================================
# MANUAL DATA INPUT
# ============================================================

def manual_input_data(
    plant_type,
):

    st.markdown(
        """
        <div class="input-card">
            <b>📝 Manual Forecast Input</b><br>
            Enter or paste your forecast data below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if plant_type == "Tracking":

        columns = [
            "Block No.",
            "GHI Forecast",
            "Actual",
        ]

        default_rows = 96

    else:

        columns = [
            "Block No.",
            "GHI Forecast",
            "Actual",
        ]

        default_rows = 96

    if (
        st.session_state.manual_input_df
        is None
    ):

        default_df = pd.DataFrame(
            {
                "Block No.": np.arange(
                    1,
                    default_rows + 1,
                ),
                "GHI Forecast": np.zeros(
                    default_rows
                ),
                "Actual": np.zeros(
                    default_rows
                ),
            }
        )

        st.session_state.manual_input_df = (
            default_df
        )

    edited_df = st.data_editor(

        st.session_state.manual_input_df,

        num_rows="dynamic",

        use_container_width=True,

        hide_index=True,

        column_config={

            "Block No.": st.column_config.NumberColumn(
                "Block No.",
                min_value=1,
                step=1,
            ),

            "GHI Forecast": st.column_config.NumberColumn(
                "GHI Forecast",
                help="Forecast GHI value",
                format="%.2f",
            ),

            "Actual": st.column_config.NumberColumn(
                "Actual Power (MW)",
                help="Actual generated power",
                format="%.3f",
            ),
        },

        key="manual_data_editor",
    )

    st.session_state.manual_input_df = (
        edited_df
    )

    if st.button(
        "🗑️ Clear Manual Data",
        use_container_width=False,
    ):

        st.session_state.manual_input_df = None

        st.rerun()

    return edited_df.copy()


# ============================================================
# EFFICIENCY LOSS CONTROL
# ============================================================

def efficiency_loss_control(
    df,
    auto_loss,
    plant_type,
):

    max_loss = float(
        df[
            "Standard PV Efficiency (%)"
        ].min()
    )

    key = (
        "manual_fixed_loss"
        if plant_type == "Fixed"
        else "manual_tracking_loss"
    )

    # --------------------------------------------------------
    # Initialize manual value ONLY once
    # --------------------------------------------------------

    if (
        st.session_state[key]
        is None
    ):

        st.session_state[key] = float(
            auto_loss
        )

    st.markdown(
        "### 📉 Efficiency Loss"
    )

    st.caption(
        "The model first calculates the automatic efficiency loss. "
        "You can manually change the value below and the forecast "
        "will update immediately."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Auto Calculated Loss",
            f"{auto_loss:.2f}%",
        )

    with col2:

        st.metric(
            "Current Loss",
            f"{st.session_state[key]:.2f}%",
        )

    loss = st.number_input(

        "Efficiency Loss (%)",

        min_value=0.0,

        max_value=max_loss,

        value=float(
            st.session_state[key]
        ),

        step=0.1,

        format="%.2f",

        key=f"{plant_type.lower()}_efficiency_loss_input",
    )

    st.session_state[key] = float(
        loss
    )

    if abs(
        loss - auto_loss
    ) > 0.001:

        st.info(
            f"✏️ Manual efficiency loss override is active: "
            f"**{loss:.2f}%**"
        )

    else:

        st.success(
            f"🤖 Automatic efficiency loss is being used: "
            f"**{loss:.2f}%**"
        )

    return float(loss)


# ============================================================
# NON-CLUSTER FIXED
# ============================================================

def process_noncluster_fixed(
    uploaded_file,
    df,
    lat,
    tilt_lookup,
    manual_data=None,
):

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    if manual_data is not None:

        df_fix = manual_data.copy()

        validate_columns(
            df_fix,
            [
                "GHI Forecast",
                "Actual",
            ],
            "Manual Input",
        )

        df_fix = df_fix.rename(
            columns={
                "GHI Forecast": "GHI_Forecast",
            }
        )

    else:

        uploaded_file.seek(0)

        df_fix = pd.read_excel(
            uploaded_file,
            sheet_name="Fixed",
            header=1,
        )

        df_fix.columns = (
            df_fix.columns
            .astype(str)
            .str.strip()
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

    # --------------------------------------------------------
    # NUMERIC DATA
    # --------------------------------------------------------

    df_fix["Actual"] = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

    df_fix["GHI_Forecast"] = pd.to_numeric(
        df_fix["GHI_Forecast"],
        errors="coerce",
    ).fillna(0)

    # --------------------------------------------------------
    # SOLAR ANGLES
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

    auto_loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    # --------------------------------------------------------
    # MANUAL LOSS CONTROL
    # --------------------------------------------------------

    loss = efficiency_loss_control(
        df,
        auto_loss,
        "Fixed",
    )

    df = apply_efficiency_loss(
        df,
        loss,
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast = (
        df_fix["POA fixed"].to_numpy()
        * df["Eff Area"].sum()
        / 1_000_000
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
    manual_data=None,
):

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    if manual_data is not None:

        df_fix = manual_data.copy()

        validate_columns(
            df_fix,
            [
                "Block No.",
                "GHI Forecast",
                "Actual",
            ],
            "Manual Input",
        )

        df_fix = df_fix.rename(
            columns={
                "GHI Forecast": "GHI_Forecast",
            }
        )

    else:

        uploaded_file.seek(0)

        df_fix = pd.read_excel(
            uploaded_file,
            sheet_name="Fixed",
            header=1,
        )

        df_fix.columns = (
            df_fix.columns
            .astype(str)
            .str.strip()
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

        uploaded_file.seek(0)

        df_bcal = pd.read_excel(
            uploaded_file,
            sheet_name="Backend Cal",
        )

        validate_columns(
            df_bcal,
            ["Block No."],
            "Backend Cal",
        )

        df_fix["Block No."] = (
            df_bcal["Block No."]
            .iloc[
                :len(df_fix)
            ]
            .to_numpy()
        )

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    df_fix["Actual"] = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

    df_fix["GHI_Forecast"] = pd.to_numeric(
        df_fix["GHI_Forecast"],
        errors="coerce",
    ).fillna(0)

    df_fix["Block No."] = pd.to_numeric(
        df_fix["Block No."],
        errors="coerce",
    )

    df_fix = df_fix.dropna(
        subset=["Block No."]
    ).reset_index(
        drop=True
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
        df_fix["GHI_Forecast"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    # --------------------------------------------------------
    # AUTOMATIC EFFICIENCY LOSS
    # --------------------------------------------------------

    auto_loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    loss = efficiency_loss_control(
        df,
        auto_loss,
        "Tracking",
    )

    df = apply_efficiency_loss(
        df,
        loss,
    )

    # --------------------------------------------------------
    # WEIGHTED GHI
    # --------------------------------------------------------

    weighted_ghi = (
        df_fix["GHI_Forecast"].to_numpy(
            dtype=float
        )
        * df["Eff Area"].sum()
    )

    blocks = df_fix[
        "Block No."
    ].to_numpy(
        dtype=float
    )

    actual = df_fix[
        "Actual"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    context_key = (
        "noncluster_tracking",
        len(df_fix),
    )

    if (
        st.session_state.tracking_params
        is None
    ):

        result = run_tracking_optimization(
            blocks,
            weighted_ghi,
            actual,
        )

        st.session_state.tracking_params = (
            result
        )

    params = (
        st.session_state.tracking_params
    )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Tracking Parameters"
    )

    st.caption(
        "The optimizer provides the initial parameters. "
        "You can manually change any parameter below."
    )

    col1, col2, col3 = st.columns(3)

    DHI = col1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=10,
        value=int(params["DHI"]),
        step=1,
        key="noncluster_dhi",
    )

    start = col2.number_input(
        "Starting Block",
        min_value=0,
        max_value=30,
        value=int(params["start"]),
        step=1,
        key="noncluster_start",
    )

    end = col3.number_input(
        "Ending Block",
        min_value=65,
        max_value=80,
        value=int(params["end"]),
        step=1,
        key="noncluster_end",
    )

    col1, col2, col3 = st.columns(3)

    max_block = col1.number_input(
        "Max Block",
        min_value=44,
        max_value=60,
        value=int(params["max"]),
        step=1,
        key="noncluster_max",
    )

    east = col2.number_input(
        "East Limit",
        min_value=0,
        max_value=70,
        value=int(params["east"]),
        step=1,
        key="noncluster_east",
    )

    west = col3.number_input(
        "West Limit",
        min_value=0,
        max_value=70,
        value=int(params["west"]),
        step=1,
        key="noncluster_west",
    )

    final_params = {
        "DHI": int(DHI),
        "start": int(start),
        "end": int(end),
        "max": int(max_block),
        "east": int(east),
        "west": int(west),
    }

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            final_params,
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
# CLUSTER GHI LOADER
# ============================================================

def load_cluster_ghi(
    uploaded_file,
    df_fix,
):

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

    uploaded_file.seek(0)

    try:

        df_ghi = pd.read_excel(
            uploaded_file,
            sheet_name="Result",
            usecols=[
                0,
                1,
                2,
                3,
                4,
                5,
            ],
        )

        df_ghi = df_ghi.fillna(0)

    except Exception:

        df_ghi = None

    if df_ghi is not None:

        for i, col in enumerate(
            ghi_columns
        ):

            if col not in df_fix.columns:

                if i < len(
                    df_ghi.columns
                ):

                    values = (
                        df_ghi.iloc[
                            :len(df_fix),
                            i,
                        ]
                        .to_numpy()
                    )

                    if len(values) < len(
                        df_fix
                    ):

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

    validate_columns(
        df_fix,
        ghi_columns,
        "Cluster GHI",
    )

    for col in ghi_columns:

        df_fix[col] = pd.to_numeric(
            df_fix[col],
            errors="coerce",
        ).fillna(0)

    return df_fix


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

    uploaded_file.seek(0)

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-CL1",
        header=1,
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    df_fix = clean_data_rows(
        df_fix
    )

    validate_columns(
        df_fix,
        ["Actual"],
        "Fixed-CL1",
    )

    df_fix = load_cluster_ghi(
        uploaded_file,
        df_fix,
    )

    df_fix["Actual"] = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

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

    auto_loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    loss = efficiency_loss_control(
        df,
        auto_loss,
        "Fixed",
    )

    df = apply_efficiency_loss(
        df,
        loss,
    )

    forecast = calculate_fixed_forecast(
        df,
        df_fix,
        cluster=True,
        cluster_weights=cluster_weights,
    )

    show_efficiency_table(
        df
    )

    show_forecast_chart(
        forecast,
        df_fix["Actual"].to_numpy(),
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

    uploaded_file.seek(0)

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-CL1",
        header=1,
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    df_fix = clean_data_rows(
        df_fix
    )

    validate_columns(
        df_fix,
        ["Actual"],
        "Fixed-CL1",
    )

    df_fix = load_cluster_ghi(
        uploaded_file,
        df_fix,
    )

    df_fix["Actual"] = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

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
    # EFFICIENCY LOSS
    # --------------------------------------------------------

    auto_loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    loss = efficiency_loss_control(
        df,
        auto_loss,
        "Tracking",
    )

    df = apply_efficiency_loss(
        df,
        loss,
    )

    # --------------------------------------------------------
    # BACKEND SHEETS
    # --------------------------------------------------------

    backend_sheets = [
        "Backend Cal CL1",
        "Backend Cal CL2",
        "Backend Cal CL3",
        "Backend Cal CL4",
        "Backend Cal CL5",
    ]

    uploaded_file.seek(0)

    backend_list = []

    for sheet in backend_sheets:

        backend_list.append(
            pd.read_excel(
                uploaded_file,
                sheet_name=sheet,
            )
        )

    validate_columns(
        backend_list[0],
        ["Block No."],
        backend_sheets[0],
    )

    blocks = (
        backend_list[0][
            "Block No."
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # WEIGHTED GHI
    # --------------------------------------------------------

    weighted_ghi = np.zeros(
        len(df_fix),
        dtype=float,
    )

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

    weight_keys = [
        "CL-1",
        "CL-2",
        "CL-3",
        "CL-4",
        "CL-5",
    ]

    for ghi_col, weight_key in zip(
        ghi_columns,
        weight_keys,
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

    actual = df_fix[
        "Actual"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # OPTIMIZE
    # --------------------------------------------------------

    if (
        st.session_state.tracking_params
        is None
    ):

        result = run_tracking_optimization(
            blocks,
            weighted_ghi,
            actual,
        )

        st.session_state.tracking_params = (
            result
        )

    params = (
        st.session_state.tracking_params
    )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Tracking Parameters"
    )

    st.caption(
        "Optimizer values are automatically generated. "
        "You can manually change them below."
    )

    col1, col2, col3 = st.columns(3)

    DHI = col1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=10,
        value=int(params["DHI"]),
        step=1,
        key="cluster_dhi",
    )

    start = col2.number_input(
        "Starting Block",
        min_value=0,
        max_value=30,
        value=int(params["start"]),
        step=1,
        key="cluster_start",
    )

    end = col3.number_input(
        "Ending Block",
        min_value=65,
        max_value=80,
        value=int(params["end"]),
        step=1,
        key="cluster_end",
    )

    col1, col2, col3 = st.columns(3)

    max_block = col1.number_input(
        "Max Block",
        min_value=44,
        max_value=60,
        value=int(params["max"]),
        step=1,
        key="cluster_max",
    )

    east = col2.number_input(
        "East Limit",
        min_value=0,
        max_value=70,
        value=int(params["east"]),
        step=1,
        key="cluster_east",
    )

    west = col3.number_input(
        "West Limit",
        min_value=0,
        max_value=70,
        value=int(params["west"]),
        step=1,
        key="cluster_west",
    )

    final_params = {
        "DHI": int(DHI),
        "start": int(start),
        "end": int(end),
        "max": int(max_block),
        "east": int(east),
        "west": int(west),
    }

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            final_params,
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
# EXCEL INPUT
# ============================================================

def process_excel_input(
    uploaded_file,
    plant_type,
):

    try:

        is_cluster = detect_cluster_model(
            uploaded_file
        )

        # ----------------------------------------------------
        # LOAD COMMON CONFIG
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # WORKBOOK TYPE
        # ----------------------------------------------------

        if is_cluster:

            st.info(
                "🏢 **Cluster workbook detected**"
            )

        else:

            st.success(
                "🏭 **Non-cluster workbook detected**"
            )

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

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
# MANUAL MODEL
# ============================================================

def process_manual_input(
    manual_data,
    plant_type,
):

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    required = [
        "GHI Forecast",
        "Actual",
    ]

    if plant_type == "Tracking":

        required.append(
            "Block No."
        )

    missing = [
        col
        for col in required
        if col not in manual_data.columns
    ]

    if missing:

        st.error(
            "Missing columns: "
            + ", ".join(missing)
        )

        return

    data = manual_data.copy()

    for col in required:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    data = data.dropna(
        subset=required
    ).reset_index(
        drop=True
    )

    if len(data) == 0:

        st.warning(
            "Please enter valid input data."
        )

        return

    # --------------------------------------------------------
    # MANUAL PLANT ASSUMPTIONS
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Manual Model Configuration"
    )

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude (°)",
            min_value=-90.0,
            max_value=90.0,
            value=28.60,
            step=0.1,
        )

    with col2:

        tilt = st.number_input(
            "Tilt Angle (°)",
            min_value=0.0,
            max_value=90.0,
            value=20.0,
            step=0.5,
        )

    # --------------------------------------------------------
    # MANUAL MODULE CONFIGURATION
    # --------------------------------------------------------

    st.markdown(
        "### 🔋 Module Configuration"
    )

    module_col1, module_col2 = st.columns(2)

    with module_col1:

        module_eff = st.number_input(
            "Standard PV Efficiency (%)",
            min_value=1.0,
            max_value=40.0,
            value=20.0,
            step=0.1,
        )

    with module_col2:

        area = st.number_input(
            "Total Area (m²)",
            min_value=1.0,
            max_value=10_000_000.0,
            value=1000.0,
            step=100.0,
        )

    df = pd.DataFrame(
        {
            "Module Type": [
                "Manual Module"
            ],

            "Standard PV Efficiency (%)": [
                module_eff
            ],

            "Total area(m2)": [
                area
            ],
        }
    )

    # --------------------------------------------------------
    # ANGLES
    # --------------------------------------------------------

    df_angle = pd.DataFrame(
        {
            "Actual": data[
                "Actual"
            ].to_numpy(),

            "GHI_Forecast": data[
                "GHI Forecast"
            ].to_numpy(),
        }
    )

    if plant_type == "Tracking":

        df_angle["Block No."] = data[
            "Block No."
        ].to_numpy()

    # Use manual date/angle calculations
    today = pd.Timestamp.today().normalize()

    df_angle["Date"] = today

    day_number = (
        today.dayofyear
    )

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

    df_angle[
        "Declination Angle ∆"
    ] = declination

    df_angle[
        "Elevation angle a"
    ] = elevation

    df_angle[
        "Tilt Angle b"
    ] = (
        0
        if plant_type == "Tracking"
        else tilt
    )

    df_angle["a+b"] = (
        df_angle["Elevation angle a"]
        + df_angle["Tilt Angle b"]
    )

    df_angle["SIN(a+b)"] = np.sin(
        np.radians(
            df_angle["a+b"]
        )
    )

    df_angle["Sin(a)"] = max(
        np.sin(
            np.radians(
                elevation
            )
        ),
        1e-6,
    )

    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    if plant_type == "Fixed":

        df_angle[
            "POA fixed"
        ] = (
            df_angle[
                "GHI_Forecast"
            ]
            * df_angle[
                "SIN(a+b)"
            ]
            / df_angle[
                "Sin(a)"
            ]
        )

        auto_loss = calculate_efficiency_loss(
            df,
            df_angle["POA fixed"],
            df_angle["Actual"],
        )

        loss = efficiency_loss_control(
            df,
            auto_loss,
            "Fixed",
        )

        df = apply_efficiency_loss(
            df,
            loss,
        )

        forecast = (
            df_angle[
                "POA fixed"
            ].to_numpy()
            * df["Eff Area"].sum()
            / 1_000_000
        )

        show_efficiency_table(
            df
        )

        show_forecast_chart(
            forecast,
            df_angle["Actual"].to_numpy(),
            "🏗️ Manual Fixed Forecast vs Actual",
        )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    else:

        blocks = df_angle[
            "Block No."
        ].to_numpy(
            dtype=float
        )

        df_angle[
            "POA fixed"
        ] = (
            df_angle[
                "GHI_Forecast"
            ]
            * df_angle[
                "SIN(a+b)"
            ]
            / df_angle[
                "Sin(a)"
            ]
        )

        auto_loss = calculate_efficiency_loss(
            df,
            df_angle["POA fixed"],
            df_angle["Actual"],
        )

        loss = efficiency_loss_control(
            df,
            auto_loss,
            "Tracking",
        )

        df = apply_efficiency_loss(
            df,
            loss,
        )

        weighted_ghi = (
            df_angle[
                "GHI_Forecast"
            ].to_numpy()
            * df[
                "Eff Area"
            ].sum()
        )

        actual = df_angle[
            "Actual"
        ].to_numpy(
            dtype=float
        )

        if (
            st.session_state.tracking_params
            is None
        ):

            result = run_tracking_optimization(
                blocks,
                weighted_ghi,
                actual,
            )

            st.session_state.tracking_params = (
                result
            )

        params = (
            st.session_state.tracking_params
        )

        st.markdown(
            "### ⚙️ Tracking Parameters"
        )

        col1, col2, col3 = st.columns(3)

        DHI = col1.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            value=int(params["DHI"]),
            step=1,
            key="manual_tracking_dhi",
        )

        start = col2.number_input(
            "Starting Block",
            min_value=0,
            max_value=30,
            value=int(params["start"]),
            step=1,
            key="manual_tracking_start",
        )

        end = col3.number_input(
            "Ending Block",
            min_value=65,
            max_value=80,
            value=int(params["end"]),
            step=1,
            key="manual_tracking_end",
        )

        col1, col2, col3 = st.columns(3)

        max_block = col1.number_input(
            "Max Block",
            min_value=44,
            max_value=60,
            value=int(params["max"]),
            step=1,
            key="manual_tracking_max",
        )

        east = col2.number_input(
            "East Limit",
            min_value=0,
            max_value=70,
            value=int(params["east"]),
            step=1,
            key="manual_tracking_east",
        )

        west = col3.number_input(
            "West Limit",
            min_value=0,
            max_value=70,
            value=int(params["west"]),
            step=1,
            key="manual_tracking_west",
        )

        final_params = {
            "DHI": int(DHI),
            "start": int(start),
            "end": int(end),
            "max": int(max_block),
            "east": int(east),
            "west": int(west),
        }

        try:

            forecast = calculate_tracking_forecast(
                blocks,
                weighted_ghi,
                final_params,
            )

            show_efficiency_table(
                df
            )

            show_forecast_chart(
                forecast,
                actual,
                "🔄 Manual Tracking Forecast vs Actual",
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
        '☀️ Loss Correction Model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Solar forecast correction using efficiency-loss '
        'optimization and tracking parameter optimization.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # PLANT TYPE
    # ========================================================

    plant_type = plant_type_selector()

    st.divider()

    # ========================================================
    # INPUT MODE
    # ========================================================

    input_mode = input_mode_selector()

    # ========================================================
    # MANUAL DATA
    # ========================================================

    if input_mode == "Manual Data":

        manual_data = manual_input_data(
            plant_type
        )

        st.divider()

        process_manual_input(
            manual_data,
            plant_type,
        )

        return

    # ========================================================
    # EXCEL WORKBOOK
    # ========================================================

    st.markdown(
        "### 📁 Excel Workbook"
    )

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=[
            "xlsx",
            "xls",
        ],
        help=(
            "Upload the plant workbook containing "
            "Area & Efficiency, Fixed/cluster sheets, "
            "Forecast Config and Backend Cal."
        ),
        key="excel_uploader",
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload an Excel workbook to start."
        )

        return

    # ========================================================
    # MODEL CONTEXT
    # ========================================================

    current_context = (
        uploaded_file.name,
        uploaded_file.size,
        plant_type,
        input_mode,
    )

    if (
        st.session_state.model_context
        != current_context
    ):

        st.session_state.tracking_params = None

        st.session_state.manual_fixed_loss = None

        st.session_state.manual_tracking_loss = None

        st.session_state.model_context = (
            current_context
        )

    # ========================================================
    # RUN EXCEL MODEL
    # ========================================================

    process_excel_input(
        uploaded_file,
        plant_type,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
