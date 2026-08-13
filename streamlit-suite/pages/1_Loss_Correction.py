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
    (0, 30),     # GHI Starting Block
    (65, 80),    # GHI Ending Block
    (44, 60),    # GHI Max Block
    (0, 70),     # East Tracking Limit
    (0, 70),     # West Tracking Limit
]

QUOTES = [
    "☕ Vo kehte the kya ho tum, aaj hum kehte hai tum kya ho be?",
    "🌦 Aapka mann nahi kar raha bahar jaane ka?..",
    "😊 Jinke ghar sheeshe ke bane hote hai vo basement mai kapde change krte h...",
    "😋 Aromatic Rose Latte with Frothy Milk pine ka mann hor hai na...",
    "🥛 Garmi mai daalo dudh mai Ice🧊 Dudh bangya Very Nice...",
    "🌟 Aapke face pr toh Modiji se bhi jyda glow hai..",
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN
       ======================================================== */

    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: var(--text-color);
        opacity: 0.65;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    /* ========================================================
       PLANT TYPE BUTTONS
       ======================================================== */

    div.stButton > button {
        width: 100%;
        min-height: 58px;

        border-radius: 14px;

        font-size: 16px;
        font-weight: 700;

        background-color: var(--secondary-background-color);
        color: var(--text-color);

        border: 1px solid rgba(128, 128, 128, 0.35);

        transition:
            all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #3b82f6;
        background-color: rgba(59, 130, 246, 0.10);
        color: var(--text-color);

        transform: translateY(-1px);
    }

    div.stButton > button:active {
        transform: translateY(0px);
    }

    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border-radius: 12px;

        padding: 12px;

        border: 1px solid rgba(128, 128, 128, 0.20);
    }

    /* ========================================================
       EXPANDER
       ======================================================== */

    div[data-testid="stExpander"] {
        border-radius: 12px;
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
    "tracking_params": None,
    "model_context": None,

    # Efficiency losses
    "fixed_noncluster_loss": None,
    "tracking_noncluster_loss": None,
    "fixed_cluster_loss": None,
    "tracking_cluster_loss": None,

    # Tracking parameter values
    "noncluster_dhi": None,
    "noncluster_start": None,
    "noncluster_end": None,
    "noncluster_max": None,
    "noncluster_east": None,
    "noncluster_west": None,

    "cluster_dhi": None,
    "cluster_start": None,
    "cluster_end": None,
    "cluster_max": None,
    "cluster_east": None,
    "cluster_west": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER
# REQUIRED COLUMN CHECK
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
# EXCEL SHEET DETECTION
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
# READ AREA & EFFICIENCY
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

    weights = {}

    for col in required:

        value = pd.to_numeric(
            df_w[col].iloc[0],
            errors="coerce",
        )

        if pd.isna(value):
            value = 0.0

        weights[col] = float(value)

    return weights


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

    latitude = pd.to_numeric(
        df_st["Lat"].iloc[0],
        errors="coerce",
    )

    if pd.isna(latitude):

        raise ValueError(
            "Latitude value is invalid."
        )

    return float(latitude)


# ============================================================
# TILT LOOKUP
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

    null_indices = (
        df_tilt[
            df_tilt["Fixed"].isna()
        ].index
    )

    if len(null_indices) > 0:

        first_null_pos = (
            df_tilt.index.get_loc(
                null_indices[0]
            )
        )

        df_tilt = df_tilt.iloc[
            :first_null_pos
        ]

    df_tilt = df_tilt.dropna(
        how="all",
        axis=1,
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

    result = (
        df_tilt
        .dropna(
            subset=["Month"]
        )
        .set_index("Month")["Fixed"]
        .to_dict()
    )

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
            df_fix["Elevation angle a"]
        )
    )

    df_fix["Sin(a)"] = (
        df_fix["Sin(a)"]
        .clip(lower=1e-6)
    )

    return df_fix


# ============================================================
# REMOVE EMPTY ROWS
# ============================================================

def clean_data_rows(
    df,
    date_column="Date",
):

    df = df.copy()

    if date_column in df.columns:

        null_indices = (
            df[
                df[
                    date_column
                ].isna()
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
# EFFICIENCY LOSS CALCULATION
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

    valid_eff = standard_eff[
        np.isfinite(standard_eff)
    ]

    if len(valid_eff) == 0:
        return 0.0

    max_loss = float(
        np.nanmin(valid_eff)
    )

    actual_peak = float(
        np.nanmax(valid_actual)
    )

    poa_peak = float(
        np.nanmax(valid_poa)
    )

    if poa_peak <= 0:
        return 0.0

    base_area = np.nansum(
        area
        * standard_eff
        / 100
    )

    loss_area_coefficient = np.nansum(
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


# ============================================================
# EFFICIENCY LOSS CONTROL
# ============================================================

def efficiency_loss_control(
    auto_loss,
    df,
    key,
):
    """
    Displays automatic efficiency loss and allows
    manual modification.

    The selected manual value is retained during
    Streamlit reruns.
    """

    max_loss = float(
        pd.to_numeric(
            df[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce",
        ).min()
    )

    if not np.isfinite(max_loss):
        max_loss = 20.0

    auto_loss = float(
        np.clip(
            auto_loss,
            0,
            max_loss,
        )
    )

    # Initialize only once
    if st.session_state.get(key) is None:

        st.session_state[key] = auto_loss

    st.markdown(
        "### 📉 Efficiency Loss"
    )

    st.caption(
        "The model automatically calculates the efficiency loss. "
        "You can manually change the value below and the forecast "
        "will update automatically."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🤖 Auto Calculated Loss",
            f"{auto_loss:.2f}%",
        )

    with col2:

        loss = st.number_input(
            "🎛️ Efficiency Loss Used (%)",
            min_value=0.0,
            max_value=max_loss,
            step=0.1,
            format="%.2f",
            key=key,
        )

    if abs(
        float(loss)
        - auto_loss
    ) > 0.001:

        st.warning(
            f"⚠️ Manual loss is being used: "
            f"**{loss:.2f}%** "
            f"(automatic: {auto_loss:.2f}%)"
        )

    else:

        st.success(
            f"✅ Using automatic efficiency loss: "
            f"**{loss:.2f}%**"
        )

    return float(loss)


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
        pd.to_numeric(
            df[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce",
        )
        - float(loss)
    )

    df["Eff Area"] = (
        pd.to_numeric(
            df[
                "Total area(m2)"
            ],
            errors="coerce",
        )
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
            df_fix[
                "POA fixed"
            ].to_numpy()
            * df["Eff Area"].sum()
            / 1_000_000
        )

    return forecast


# ============================================================
# TRACKING OBJECTIVE
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
            "No non-zero Actual power values found."
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

        m1 = (
            90
            / denominator_1
        )

        m2 = (
            90
            / denominator_2
        )

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

        dhi_factor = (
            1
            - DHI / 100
        )

        prediction = (
            weighted_ghi_day
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

    DHI = int(
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

        np.minimum(
            zenith,
            abs(east),
        ),

        np.where(

            (
                (blocks > max_block)
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
# ATTRACTIVE PLANT TYPE SELECTOR
# ============================================================

def reset_model_controls():

    st.session_state.tracking_params = None

    loss_keys = [
        "fixed_noncluster_loss",
        "tracking_noncluster_loss",
        "fixed_cluster_loss",
        "tracking_cluster_loss",
    ]

    for key in loss_keys:

        st.session_state[key] = None

    parameter_keys = [
        "noncluster_dhi",
        "noncluster_start",
        "noncluster_end",
        "noncluster_max",
        "noncluster_east",
        "noncluster_west",
        "cluster_dhi",
        "cluster_start",
        "cluster_end",
        "cluster_max",
        "cluster_east",
        "cluster_west",
    ]

    for key in parameter_keys:

        st.session_state[key] = None


def plant_type_selector():

    st.markdown(
        '<div class="section-title">🏭 Select Plant Type</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🏗️  FIXED PLANT",
            key="fixed_plant_button",
            use_container_width=True,
        ):

            if (
                st.session_state.plant_type
                != "🏗️ Fixed"
            ):

                st.session_state.plant_type = (
                    "🏗️ Fixed"
                )

                reset_model_controls()

                st.rerun()

    with col2:

        if st.button(
            "🔄  TRACKING PLANT",
            key="tracking_plant_button",
            use_container_width=True,
        ):

            if (
                st.session_state.plant_type
                != "🔄 Tracking"
            ):

                st.session_state.plant_type = (
                    "🔄 Tracking"
                )

                reset_model_controls()

                st.rerun()

    current = (
        st.session_state.plant_type
    )

    if current == "🏗️ Fixed":

        st.success(
            "🏗️ Fixed plant selected",
            icon="🏗️",
        )

    else:

        st.info(
            "🔄 Tracking plant selected",
            icon="🔄",
        )

    return current


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
# TRACKING PARAMETER UI
# ============================================================

def tracking_parameter_controls(
    params,
    prefix,
):

    st.markdown(
        "### ⚙️ Tracking Parameters"
    )

    st.caption(
        "The optimizer provides the initial values. "
        "You can manually adjust any parameter and the "
        "forecast will update automatically."
    )

    key_dhi = f"{prefix}_dhi"
    key_start = f"{prefix}_start"
    key_end = f"{prefix}_end"
    key_max = f"{prefix}_max"
    key_east = f"{prefix}_east"
    key_west = f"{prefix}_west"

    # Initialize session state
    if st.session_state[key_dhi] is None:
        st.session_state[key_dhi] = int(params["DHI"])

    if st.session_state[key_start] is None:
        st.session_state[key_start] = int(params["start"])

    if st.session_state[key_end] is None:
        st.session_state[key_end] = int(params["end"])

    if st.session_state[key_max] is None:
        st.session_state[key_max] = int(params["max"])

    if st.session_state[key_east] is None:
        st.session_state[key_east] = int(params["east"])

    if st.session_state[key_west] is None:
        st.session_state[key_west] = int(params["west"])

    col1, col2, col3 = st.columns(3)

    with col1:

        DHI = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            step=1,
            key=key_dhi,
        )

    with col2:

        start = st.number_input(
            "Starting Block",
            min_value=0,
            max_value=30,
            step=1,
            key=key_start,
        )

    with col3:

        end = st.number_input(
            "Ending Block",
            min_value=65,
            max_value=80,
            step=1,
            key=key_end,
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        max_block = st.number_input(
            "Max Block",
            min_value=44,
            max_value=60,
            step=1,
            key=key_max,
        )

    with col2:

        east = st.number_input(
            "East Limit",
            min_value=0,
            max_value=70,
            step=1,
            key=key_east,
        )

    with col3:

        west = st.number_input(
            "West Limit",
            min_value=0,
            max_value=70,
            step=1,
            key=key_west,
        )

    final_params = {
        "DHI": int(DHI),
        "start": int(start),
        "end": int(end),
        "max": int(max_block),
        "east": int(east),
        "west": int(west),
    }

    return final_params


# ============================================================
# FIXED NON-CLUSTER
# ============================================================

def process_noncluster_fixed(
    uploaded_file,
    df,
    lat,
    tilt_lookup,
):

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

    df_fix["Actual"] = (
        pd.to_numeric(
            df_fix["Actual"],
            errors="coerce",
        )
        .fillna(0)
    )

    df_fix["GHI_Forecast"] = (
        pd.to_numeric(
            df_fix["GHI_Forecast"],
            errors="coerce",
        )
        .fillna(0)
    )

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=False,
    )

    df_fix[
        "POA fixed"
    ] = (
        df_fix["GHI_Forecast"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    auto_loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    # --------------------------------------------------------
    # EFFICIENCY LOSS CONTROL
    # --------------------------------------------------------

    loss = efficiency_loss_control(
        auto_loss=auto_loss,
        df=df,
        key="fixed_noncluster_loss",
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
# TRACKING NON-CLUSTER
# ============================================================

def process_noncluster_tracking(
    uploaded_file,
    df,
    lat,
    tilt_lookup,
):

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

    df_fix["Actual"] = (
        pd.to_numeric(
            df_fix["Actual"],
            errors="coerce",
        )
        .fillna(0)
    )

    df_fix["GHI_Forecast"] = (
        pd.to_numeric(
            df_fix["GHI_Forecast"],
            errors="coerce",
        )
        .fillna(0)
    )

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=True,
    )

    df_fix[
        "POA fixed"
    ] = (
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

    # --------------------------------------------------------
    # MANUAL EFFICIENCY LOSS
    # --------------------------------------------------------

    loss = efficiency_loss_control(
        auto_loss=auto_loss,
        df=df,
        key="tracking_noncluster_loss",
    )

    df = apply_efficiency_loss(
        df,
        loss,
    )

    # --------------------------------------------------------
    # BACKEND DATA
    # --------------------------------------------------------

    uploaded_file.seek(0)

    df_bcal = pd.read_excel(
        uploaded_file,
        sheet_name="Backend Cal",
    )

    df_bcal.columns = (
        df_bcal.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df_bcal,
        ["Block No."],
        "Backend Cal",
    )

    blocks = (
        pd.to_numeric(
            df_bcal["Block No."],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # WEIGHTED GHI
    # --------------------------------------------------------

    weighted_ghi = (
        df_fix["GHI_Forecast"]
        .to_numpy(
            dtype=float
        )
        * df["Eff Area"].sum()
    )

    actual = (
        df_fix["Actual"]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # TRACKING OPTIMIZATION
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

        st.session_state.tracking_params = result

    params = (
        st.session_state.tracking_params
    )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    final_params = tracking_parameter_controls(
        params,
        prefix="noncluster",
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            final_params,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Auto Efficiency Loss",
            f"{auto_loss:.2f}%",
        )

        c2.metric(
            "Current Efficiency Loss",
            f"{loss:.2f}%",
        )

        c3.metric(
            "Tracking Score",
            f"{params['score']:.4f}",
        )

        # ----------------------------------------------------
        # RE-OPTIMIZE
        # ----------------------------------------------------

        if (
            abs(
                loss
                - auto_loss
            ) > 0.001
        ):

            st.warning(
                "The efficiency loss has been manually changed. "
                "The forecast uses your new loss, while the tracking "
                "parameters are still from the previous optimization."
            )

            if st.button(
                "🔄 Re-optimize Tracking Parameters",
                key="reoptimize_noncluster",
                use_container_width=True,
            ):

                st.session_state.tracking_params = None

                st.rerun()

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

    # --------------------------------------------------------
    # GHI DATA
    # --------------------------------------------------------

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

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

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
        "Fixed-CL1",
    )

    df_fix["Actual"] = (
        pd.to_numeric(
            df_fix["Actual"],
            errors="coerce",
        )
        .fillna(0)
    )

    for col in ghi_columns:

        df_fix[col] = (
            pd.to_numeric(
                df_fix[col],
                errors="coerce",
            )
            .fillna(0)
        )

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=False,
    )

    df_fix[
        "POA fixed"
    ] = (
        df_fix["CL1-GHI"]
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
    # MANUAL LOSS
    # --------------------------------------------------------

    loss = efficiency_loss_control(
        auto_loss=auto_loss,
        df=df,
        key="fixed_cluster_loss",
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

    c1, c2 = st.columns(2)

    c1.metric(
        "Auto Calculated Loss",
        f"{auto_loss:.2f}%",
    )

    c2.metric(
        "Current Loss Used",
        f"{loss:.2f}%",
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

    # --------------------------------------------------------
    # CLUSTER GHI
    # --------------------------------------------------------

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
        "Fixed-CL1",
    )

    df_fix["Actual"] = (
        pd.to_numeric(
            df_fix["Actual"],
            errors="coerce",
        )
        .fillna(0)
    )

    for col in ghi_columns:

        df_fix[col] = (
            pd.to_numeric(
                df_fix[col],
                errors="coerce",
            )
            .fillna(0)
        )

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=True,
    )

    # --------------------------------------------------------
    # AUTOMATIC LOSS
    # --------------------------------------------------------

    df_fix[
        "POA fixed"
    ] = (
        df_fix["CL1-GHI"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    auto_loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    # --------------------------------------------------------
    # MANUAL LOSS
    # --------------------------------------------------------

    loss = efficiency_loss_control(
        auto_loss=auto_loss,
        df=df,
        key="tracking_cluster_loss",
    )

    df = apply_efficiency_loss(
        df,
        loss,
    )

    # --------------------------------------------------------
    # BACKEND DATA
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

        temp = pd.read_excel(
            uploaded_file,
            sheet_name=sheet,
        )

        temp.columns = (
            temp.columns
            .astype(str)
            .str.strip()
        )

        backend_list.append(
            temp
        )

    validate_columns(
        backend_list[0],
        ["Block No."],
        "Backend Cal CL1",
    )

    blocks = (
        pd.to_numeric(
            backend_list[0]["Block No."],
            errors="coerce",
        )
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

        ghi = (
            df_fix[ghi_col]
            .to_numpy(
                dtype=float
            )
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

    actual = (
        df_fix["Actual"]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # OPTIMIZATION
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
    # TRACKING PARAMETERS
    # --------------------------------------------------------

    final_params = tracking_parameter_controls(
        params,
        prefix="cluster",
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            final_params,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Auto Efficiency Loss",
            f"{auto_loss:.2f}%",
        )

        c2.metric(
            "Current Efficiency Loss",
            f"{loss:.2f}%",
        )

        c3.metric(
            "Tracking Score",
            f"{params['score']:.4f}",
        )

        # ----------------------------------------------------
        # RE-OPTIMIZE
        # ----------------------------------------------------

        if (
            abs(
                loss
                - auto_loss
            ) > 0.001
        ):

            st.warning(
                "The efficiency loss has been manually changed. "
                "The forecast uses your new loss, while the tracking "
                "parameters are still from the previous optimization."
            )

            if st.button(
                "🔄 Re-optimize Tracking Parameters",
                key="reoptimize_cluster",
                use_container_width=True,
            ):

                st.session_state.tracking_params = None

                st.rerun()

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
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '☀️ Loss Correction Model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Upload your plant Excel file, select Fixed or Tracking, "
        "and the model will calculate the forecast automatically."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # FILE INPUT
    # --------------------------------------------------------

    st.markdown(
        "### 📁 Input Data"
    )

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=[
            "xlsx",
            "xls",
        ],
        help=(
            "Upload the Excel workbook containing "
            "the Area & Efficiency and forecast sheets."
        ),
    )

    if uploaded_file is None:

        st.info(
            "👆 Please upload the Excel file to start."
        )

        st.stop()

    # --------------------------------------------------------
    # DETECT CLUSTER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL DETECTION
    # --------------------------------------------------------

    if is_cluster:

        st.info(
            "🏢 **Cluster workbook detected** "
            "because the workbook does not contain a `Fixed` sheet."
        )

    else:

        st.success(
            "🏭 **Non-cluster workbook detected** "
            "because the workbook contains a `Fixed` sheet."
        )

    # --------------------------------------------------------
    # PLANT TYPE
    # --------------------------------------------------------

    plant_type = plant_type_selector()

    # --------------------------------------------------------
    # MODEL CONTEXT
    # --------------------------------------------------------

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

        st.session_state.fixed_noncluster_loss = None
        st.session_state.tracking_noncluster_loss = None
        st.session_state.fixed_cluster_loss = None
        st.session_state.tracking_cluster_loss = None

        st.session_state.noncluster_dhi = None
        st.session_state.noncluster_start = None
        st.session_state.noncluster_end = None
        st.session_state.noncluster_max = None
        st.session_state.noncluster_east = None
        st.session_state.noncluster_west = None

        st.session_state.cluster_dhi = None
        st.session_state.cluster_start = None
        st.session_state.cluster_end = None
        st.session_state.cluster_max = None
        st.session_state.cluster_east = None
        st.session_state.cluster_west = None

        st.session_state.model_context = (
            current_context
        )

    # --------------------------------------------------------
    # LOAD COMMON DATA
    # --------------------------------------------------------

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
            f"Unable to load model configuration: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # MODEL EXECUTION
    # --------------------------------------------------------

    try:

        # ====================================================
        # NON-CLUSTER
        # ====================================================

        if not is_cluster:

            if plant_type == "🏗️ Fixed":

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

        # ====================================================
        # CLUSTER
        # ====================================================

        else:

            if plant_type == "🏗️ Fixed":

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
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()
