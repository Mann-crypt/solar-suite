# ============================================================
# LOSS CORRECTION MODEL PAGE
# Clean Streamlit UI + Optimized Backend
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
    page_icon="🔧",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_OPT_ITER = 40
OPT_POPSIZE = 10

PARAM_BOUNDS = [
    (0, 10),      # DHI %
    (0, 30),      # GHI Starting Block
    (65, 80),     # GHI Ending Block
    (44, 60),     # GHI Max Block
    (0, 70),      # East Tracking Limit
    (0, 70),      # West Tracking Limit
]

QUOTES = [
    "☕ Vo kehte the kya ho tum, aaj hum kehte hai tum kya ho be?",
    "🌦 Aapka mann nahi kar raha bahar jaane ka?..",
    "😊 Jinke ghar sheeshe ke bane hote hai vo basement mai kapde change krte h...",
    "😋 Aromatic Rose Latte with Frothy Milk pine ka mann hor hai na...",
    "🥛 Garmi mai daalo dudh mai Ice🧊 Dudh bangya Very Nice...",
    "🌟 Aapke face pr toh Modiji se bhi jyada glow hai..",
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
    "loss_model_context": None,
    "tracking_params": None,
    "model_result": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🔧 Loss Correction Model")

st.caption(
    "Calculate efficiency losses, optimize tracking parameters "
    "and compare forecasted power against actual generation."
)

st.divider()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_area_efficiency(uploaded_file, cluster=False):

    if cluster:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Area & Efficiency",
            header=1,
            usecols=range(8)
        )

    else:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Area & Efficiency",
            header=1
        )

    df.columns = df.columns.astype(str).str.strip()

    if "Module Type" in df.columns:

        null_indices = (
            df[df["Module Type"].isna()].index
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

    return df.reset_index(drop=True)


# ------------------------------------------------------------


def read_cluster_weights(uploaded_file):

    df_w = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[12, 13, 14, 15, 16]
    )

    return {
        "CL-1": float(df_w["CL-1"].iloc[0]),
        "CL-2": float(df_w["CL-2"].iloc[0]),
        "CL-3": float(df_w["CL-3"].iloc[0]),
        "CL-4": float(df_w["CL-4"].iloc[0]),
        "CL-5": float(df_w["CL-5"].iloc[0]),
    }


# ------------------------------------------------------------


def read_latitude(uploaded_file):

    df_st = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8
    )

    return float(
        df_st.loc[0, "Lat"]
    )


# ------------------------------------------------------------


def read_tilt_lookup(uploaded_file):

    df_tilt = pd.read_excel(
        uploaded_file,
        sheet_name="Config Tilt Angle",
        header=7
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" in df_tilt.columns:

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
        axis=1
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
        .set_index("Month")["Fixed"]
        .to_dict()
    )


# ============================================================
# SOLAR ANGLES
# ============================================================

def prepare_solar_angles(
    df_fix,
    lat,
    tilt_lookup=None,
    tracking=False
):

    df_fix = df_fix.copy()

    today = pd.Timestamp.today().normalize()

    df_fix["Date"] = today

    first_date = today.replace(
        month=1,
        day=1
    )

    day_number = (
        df_fix["Date"] - first_date
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
        + df_fix["Declination Angle ∆"]
    )

    if tracking:

        df_fix["Tilt Angle b"] = 0

    else:

        if tilt_lookup:

            df_fix["Tilt Angle b"] = (
                df_fix["Date"]
                .dt.strftime("%B")
                .map(tilt_lookup)
            )

        else:

            df_fix["Tilt Angle b"] = 0

    df_fix["Tilt Angle b"] = (
        df_fix["Tilt Angle b"]
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
    actual
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

    max_loss = float(
        np.nanmin(
            standard_eff
        )
    )

    actual = np.asarray(
        actual,
        dtype=float
    )

    poa = np.asarray(
        poa,
        dtype=float
    )

    actual_peak = float(
        np.nanmax(actual)
    )

    poa_peak = float(
        np.nanmax(poa)
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

    best_loss = np.clip(
        best_loss,
        0,
        max_loss
    )

    return float(
        best_loss
    )


# ------------------------------------------------------------


def apply_efficiency_loss(
    df,
    best_loss
):

    df = df.copy()

    df["Efficiency Losses(%)"] = (
        best_loss
    )

    df["Net Efficiency (%)"] = (
        df[
            "Standard PV Efficiency (%)"
        ]
        - best_loss
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
    cluster_weights=None
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

        forecast = np.zeros(
            len(df_fix),
            dtype=float
        )

        for ghi_col, weight_col in zip(
            ghi_cols,
            weight_cols
        ):

            poa = (
                df_fix[ghi_col].to_numpy(
                    dtype=float
                )
                * df_fix["SIN(a+b)"].to_numpy(
                    dtype=float
                )
                / df_fix["Sin(a)"].to_numpy(
                    dtype=float
                )
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
                poa
                * eff_area
                / 1_000_000
            )

    else:

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

        forecast = (
            poa
            * df["Eff Area"].sum()
            / 1_000_000
        )

    return forecast


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    blocks,
    weighted_ghi,
    actual,
    maxiter=40,
    popsize=10,
    progress_callback=None
):

    blocks = np.asarray(
        blocks,
        dtype=float
    )

    weighted_ghi = np.asarray(
        weighted_ghi,
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float
    )

    mask = actual != 0

    actual_day = actual[mask]

    weighted_ghi_day = (
        weighted_ghi[mask]
    )

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

        DHI = int(x[0])
        start = int(x[1])
        end = int(x[2])
        max_block = int(x[3])
        east = int(x[4])
        west = int(x[5])

        # ----------------------------------------------------
        # Block validation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Zenith
        # ----------------------------------------------------

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
                )
            ),

            np.minimum(
                89,
                m2
                * (
                    blocks_day
                    - max_block
                )
            )
        )

        # ----------------------------------------------------
        # Panel angle
        # ----------------------------------------------------

        panel = np.where(

            blocks_day < max_block,

            np.minimum(
                zenith,
                abs(east)
            ),

            np.where(

                (
                    (blocks_day > max_block)
                    & (zenith > west)
                ),

                west,

                zenith
            )
        )

        # ----------------------------------------------------
        # DNI / GHI correction
        # ----------------------------------------------------

        cos_alpha = np.cos(
            np.radians(panel)
        )

        cos_alpha = np.clip(
            cos_alpha,
            1e-6,
            None
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

        # ----------------------------------------------------
        # Error metrics
        # ----------------------------------------------------

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

        return float(
            score
        )

    # --------------------------------------------------------
    # Optimization callback
    # --------------------------------------------------------

    generation = {
        "count": 0
    }

    def callback(
        xk,
        convergence
    ):

        generation["count"] += 1

        if progress_callback:

            progress_callback(
                generation["count"],
                maxiter
            )

        return False

    # --------------------------------------------------------
    # Differential Evolution
    # --------------------------------------------------------

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
        ]
    )

    best = np.rint(
        result.x
    ).astype(int)

    return {

        "score": float(
            result.fun
        ),

        "DHI": int(
            best[0]
        ),

        "start": int(
            best[1]
        ),

        "end": int(
            best[2]
        ),

        "max": int(
            best[3]
        ),

        "east": int(
            best[4]
        ),

        "west": int(
            best[5]
        ),
    }


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    blocks,
    weighted_ghi,
    params
):

    blocks = np.asarray(
        blocks,
        dtype=float
    )

    weighted_ghi = np.asarray(
        weighted_ghi,
        dtype=float
    )

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
            )
        ),

        np.minimum(
            89,
            m2
            * (
                blocks
                - max_block
            )
        )
    )

    panel = np.where(

        blocks < max_block,

        np.minimum(
            zenith,
            abs(east)
        ),

        np.where(

            (
                (blocks > max_block)
                & (zenith > west)
            ),

            west,

            zenith
        )
    )

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None
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
# FILE CLEANING
# ============================================================

def clean_date_rows(df):

    df = df.copy()

    if "Date" in df.columns:

        null_indices = (
            df[
                df["Date"].isna()
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
# EFFICIENCY TABLE
# ============================================================

def show_efficiency_table(df):

    required_columns = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)",
    ]

    available = [
        c
        for c in required_columns
        if c in df.columns
    ]

    display_df = df[
        available
    ].copy()

    numeric_columns = (
        display_df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    display_df[
        numeric_columns
    ] = (
        display_df[
            numeric_columns
        ]
        .round(2)
    )

    with st.expander(
        "🔍 View Efficiency Calculations",
        expanded=False
    ):

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FORECAST METRICS
# ============================================================

def calculate_metrics(
    forecast,
    actual
):

    forecast = np.asarray(
        forecast,
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float
    )

    n = min(
        len(forecast),
        len(actual)
    )

    forecast = forecast[:n]
    actual = actual[:n]

    mask = actual != 0

    if not mask.any():

        return {
            "forecast_peak": 0,
            "actual_peak": 0,
            "peak_error": 0,
            "energy_error": 0,
            "mae": 0,
        }

    forecast_day = forecast[mask]
    actual_day = actual[mask]

    actual_peak = (
        np.max(actual_day)
    )

    forecast_peak = (
        np.max(forecast_day)
    )

    peak_error = (
        abs(
            actual_peak
            - forecast_peak
        )
        / actual_peak
        * 100
        if actual_peak != 0
        else 0
    )

    energy_error = (
        abs(
            np.sum(actual_day)
            - np.sum(forecast_day)
        )
        / np.sum(actual_day)
        * 100
        if np.sum(actual_day) != 0
        else 0
    )

    mae = (
        np.mean(
            np.abs(
                actual_day
                - forecast_day
            )
        )
    )

    return {
        "forecast_peak": forecast_peak,
        "actual_peak": actual_peak,
        "peak_error": peak_error,
        "energy_error": energy_error,
        "mae": mae,
    }


# ============================================================
# FORECAST CHART
# ============================================================

def show_forecast_chart(
    forecast,
    actual,
    title="Forecast vs Actual Power"
):

    n = min(
        len(forecast),
        len(actual)
    )

    forecast = np.asarray(
        forecast[:n],
        dtype=float
    )

    actual = np.asarray(
        actual[:n],
        dtype=float
    )

    x = np.arange(
        1,
        n + 1
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
                width=3
            )
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
                width=3
            )
        )
    )

    fig.update_layout(

        title=title,

        template="plotly_white",

        height=500,

        hovermode="x unified",

        xaxis=dict(
            title="15 Minute Block",
            dtick=4
        ),

        yaxis=dict(
            title="Power (MW)"
        ),

        legend=dict(
            orientation="h",
            y=1.08,
            x=0
        ),

        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# METRIC DISPLAY
# ============================================================

def show_forecast_metrics(
    forecast,
    actual,
    efficiency_loss
):

    metrics = calculate_metrics(
        forecast,
        actual
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    col1.metric(
        "Efficiency Loss",
        f"{efficiency_loss:.2f}%"
    )

    col2.metric(
        "Forecast Peak",
        f"{metrics['forecast_peak']:.2f} MW"
    )

    col3.metric(
        "Actual Peak",
        f"{metrics['actual_peak']:.2f} MW"
    )

    col4.metric(
        "Peak Error",
        f"{metrics['peak_error']:.2f}%"
    )

    col5.metric(
        "Energy Error",
        f"{metrics['energy_error']:.2f}%"
    )

    return metrics


# ============================================================
# OPTIMIZATION UI
# ============================================================

def run_tracking_optimization(
    blocks,
    weighted_ghi,
    actual
):

    st.subheader(
        "⚙️ Tracking Optimization"
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
        total
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
            update_progress
        )

        last_generation = -1

        while not future.done():

            generation = (
                progress_state[
                    "generation"
                ]
            )

            if generation != last_generation:

                last_generation = (
                    generation
                )

                progress = min(
                    generation
                    / MAX_OPT_ITER,
                    0.99
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
                0.15
            )

        result = future.result()

    progress_bar.progress(
        1.0
    )

    status_box.success(
        "✅ Optimization completed!"
    )

    time.sleep(
        0.5
    )

    progress_bar.empty()
    status_box.empty()

    return result


# ============================================================
# TRACKING PARAMETERS UI
# ============================================================

def show_tracking_parameters(
    params,
    efficiency_loss,
    key_prefix
):

    st.subheader(
        "🎯 Tracking Parameters"
    )

    st.caption(
        "Parameters are updated automatically. "
        "Change any value below and the forecast will refresh."
    )

    # --------------------------------------------------------
    # Efficiency loss
    # --------------------------------------------------------

    efficiency_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(efficiency_loss),
        step=0.1,
        format="%.2f",
        key=f"{key_prefix}_loss"
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    DHI = col1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=10,
        value=int(params["DHI"]),
        step=1,
        key=f"{key_prefix}_dhi"
    )

    start = col2.number_input(
        "Starting Block",
        min_value=0,
        max_value=100,
        value=int(params["start"]),
        step=1,
        key=f"{key_prefix}_start"
    )

    end = col3.number_input(
        "Ending Block",
        min_value=0,
        max_value=100,
        value=int(params["end"]),
        step=1,
        key=f"{key_prefix}_end"
    )

    col1, col2, col3 = st.columns(3)

    max_block = col1.number_input(
        "Max Block",
        min_value=0,
        max_value=100,
        value=int(params["max"]),
        step=1,
        key=f"{key_prefix}_max"
    )

    east = col2.number_input(
        "East Tracking Limit",
        min_value=0,
        max_value=70,
        value=int(params["east"]),
        step=1,
        key=f"{key_prefix}_east"
    )

    west = col3.number_input(
        "West Tracking Limit",
        min_value=0,
        max_value=70,
        value=int(params["west"]),
        step=1,
        key=f"{key_prefix}_west"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not (
        start < max_block < end
    ):

        st.error(
            "❌ Invalid tracking blocks. "
            "Required: Starting Block < Max Block < Ending Block."
        )

        return None, efficiency_loss

    final_params = {

        "DHI": int(DHI),

        "start": int(start),

        "end": int(end),

        "max": int(max_block),

        "east": int(east),

        "west": int(west),
    }

    return (
        final_params,
        efficiency_loss
    )


# ============================================================
# INPUT UI
# ============================================================

st.subheader(
    "📂 Model Configuration"
)

col1, col2, col3 = st.columns(
    [2, 1, 1]
)

with col1:

    uploaded_file = st.file_uploader(
        "Upload Solar Plant Excel File",
        type=["xlsx", "xls"],
        help=(
            "Upload the Excel workbook containing "
            "Area & Efficiency, Forecast Config, "
            "Fixed/Tracking and Backend calculation sheets."
        )
    )

with col2:

    plant_type = st.selectbox(
        "Plant Type",
        [
            "🏗️ Fixed",
            "🔄 Tracking"
        ]
    )

with col3:

    is_cluster = st.toggle(
        "Cluster Model",
        value=False
    )


# ============================================================
# FILE CHECK
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload an Excel file to start the Loss Correction Model."
    )

    st.stop()


# ============================================================
# MODEL CONTEXT
# ============================================================

current_context = (
    uploaded_file.name,
    uploaded_file.size,
    plant_type,
    bool(is_cluster),
)


# Reset tracking optimization when model changes

if (
    st.session_state.loss_model_context
    != current_context
):

    st.session_state.tracking_params = None

    st.session_state.model_result = None

    st.session_state.loss_model_context = (
        current_context
    )


# ============================================================
# FILE STATUS
# ============================================================

st.success(
    f"📄 **{uploaded_file.name}** loaded successfully"
)


# ============================================================
# RUN MODEL
# ============================================================

run_model = st.button(
    "🚀 Run Loss Correction",
    type="primary",
    use_container_width=True
)


# ============================================================
# MODEL EXECUTION
# ============================================================

if run_model:

    st.session_state.tracking_params = None
    st.session_state.model_result = None

    st.session_state.run_model = True


# If model has never been run

if (
    st.session_state.model_result is None
    and not run_model
):

    st.info(
        "Click **Run Loss Correction** to calculate the model."
    )

    st.stop()


# ============================================================
# MODEL START
# ============================================================

with st.spinner(
    "Loading workbook and preparing model..."
):

    # --------------------------------------------------------
    # Common inputs
    # --------------------------------------------------------

    df = read_area_efficiency(
        uploaded_file,
        cluster=is_cluster
    )

    lat = read_latitude(
        uploaded_file
    )

    tilt_lookup = read_tilt_lookup(
        uploaded_file
    )


# ============================================================
# CLUSTER MODEL
# ============================================================

if is_cluster:

    # --------------------------------------------------------
    # Cluster weights
    # --------------------------------------------------------

    cluster_weights = (
        read_cluster_weights(
            uploaded_file
        )
    )

    # --------------------------------------------------------
    # GHI Result sheet
    # --------------------------------------------------------

    df_ghi = pd.read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5]
    )

    df_ghi = df_ghi.fillna(0)

    # --------------------------------------------------------
    # Fixed-CL1
    # --------------------------------------------------------

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-CL1",
        header=1
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    df_fix = clean_date_rows(
        df_fix
    )

    df_fix["Actual"] = (
        df_fix["Actual"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Add cluster GHI columns if required
    # --------------------------------------------------------

    if plant_type == "🏗️ Fixed":

        ghi_columns = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

        for i, col in enumerate(
            ghi_columns
        ):

            if col not in df_fix.columns:

                df_fix[col] = (
                    df_ghi
                    .iloc[:, i]
                    .to_numpy()
                )

    # --------------------------------------------------------
    # Solar angles
    # --------------------------------------------------------

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=(
            plant_type
            == "🔄 Tracking"
        )
    )

    # --------------------------------------------------------
    # Efficiency Loss
    # --------------------------------------------------------

    df_fix["POA fixed"] = (
        df_fix["CL1-GHI"].to_numpy(
            dtype=float
        )
        * df_fix["SIN(a+b)"].to_numpy(
            dtype=float
        )
        / df_fix["Sin(a)"].to_numpy(
            dtype=float
        )
    )

    best_loss = (
        calculate_efficiency_loss(
            df,
            df_fix["POA fixed"],
            df_fix["Actual"]
        )
    )

    df = apply_efficiency_loss(
        df,
        best_loss
    )

    # ========================================================
    # CLUSTER FIXED
    # ========================================================

    if plant_type == "🏗️ Fixed":

        forecast = calculate_fixed_forecast(
            df,
            df_fix,
            cluster=True,
            cluster_weights=cluster_weights
        )

        st.divider()

        st.header(
            "📊 Fixed Cluster Results"
        )

        show_forecast_metrics(
            forecast,
            df_fix["Actual"].to_numpy(),
            best_loss
        )

        show_forecast_chart(
            forecast,
            df_fix["Actual"].to_numpy(),
            "Fixed Cluster Forecast vs Actual"
        )

        show_efficiency_table(
            df
        )

        st.session_state.model_result = {
            "forecast": forecast,
            "actual": df_fix[
                "Actual"
            ].to_numpy(),
            "efficiency_loss": best_loss,
        }

    # ========================================================
    # CLUSTER TRACKING
    # ========================================================

    else:

        # ----------------------------------------------------
        # Backend sheets
        # ----------------------------------------------------

        backend_sheets = [
            "Backend Cal CL1",
            "Backend Cal CL2",
            "Backend Cal CL3",
            "Backend Cal CL4",
            "Backend Cal CL5",
        ]

        backend_list = []

        for sheet in backend_sheets:

            backend_list.append(
                pd.read_excel(
                    uploaded_file,
                    sheet_name=sheet
                )
            )

        blocks = (
            backend_list[0][
                "Block No."
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Weighted GHI
        # ----------------------------------------------------

        weighted_ghi = np.zeros(
            len(df_fix),
            dtype=float
        )

        ghi_cols = [
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
            ghi_cols,
            weight_keys
        ):

            ghi = (
                df_fix[
                    ghi_col
                ]
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

        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

        if (
            st.session_state.tracking_params
            is None
        ):

            result = run_tracking_optimization(
                blocks,
                weighted_ghi,
                df_fix[
                    "Actual"
                ].to_numpy(
                    dtype=float
                )
            )

            st.session_state.tracking_params = (
                result
            )

        params = (
            st.session_state.tracking_params
        )

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        st.divider()

        st.header(
            "🔄 Tracking Model"
        )

        final_params, user_loss = (
            show_tracking_parameters(
                params,
                best_loss,
                "cluster_tracking"
            )
        )

        if final_params is None:

            st.stop()

        # ----------------------------------------------------
        # Apply edited efficiency loss
        # ----------------------------------------------------

        df = apply_efficiency_loss(
            df,
            user_loss
        )

        # ----------------------------------------------------
        # Final forecast
        # ----------------------------------------------------

        try:

            forecast = (
                calculate_tracking_forecast(
                    blocks,
                    weighted_ghi,
                    final_params
                )
            )

        except Exception as e:

            st.error(
                f"Unable to calculate forecast: {e}"
            )

            st.stop()

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.subheader(
            "📊 Forecast Results"
        )

        show_forecast_metrics(
            forecast,
            df_fix["Actual"].to_numpy(),
            user_loss
        )

        show_forecast_chart(
            forecast,
            df_fix["Actual"].to_numpy(),
            "Tracking Cluster Forecast vs Actual"
        )

        # ----------------------------------------------------
        # Optimizer result
        # ----------------------------------------------------

        with st.expander(
            "🧠 Optimization Details"
        ):

            col1, col2 = st.columns(2)

            col1.metric(
                "Optimization Score",
                f"{params['score']:.5f}"
            )

            col2.write(
                "The optimized values are shown above and "
                "can be edited directly."
            )

        # ----------------------------------------------------
        # Efficiency
        # ----------------------------------------------------

        show_efficiency_table(
            df
        )

        st.session_state.model_result = {
            "forecast": forecast,
            "actual": df_fix[
                "Actual"
            ].to_numpy(),
            "efficiency_loss": user_loss,
        }


# ============================================================
# NON-CLUSTER MODEL
# ============================================================

else:

    # --------------------------------------------------------
    # Fixed sheet
    # --------------------------------------------------------

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed",
        header=1
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    df_fix = clean_date_rows(
        df_fix
    )

    df_fix["Actual"] = (
        df_fix["Actual"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Solar angles
    # --------------------------------------------------------

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=(
            plant_type
            == "🔄 Tracking"
        )
    )

    # ========================================================
    # NON-CLUSTER FIXED
    # ========================================================

    if plant_type == "🏗️ Fixed":

        # ----------------------------------------------------
        # POA
        # ----------------------------------------------------

        df_fix["POA fixed"] = (
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

        # ----------------------------------------------------
        # Efficiency loss
        # ----------------------------------------------------

        best_loss = (
            calculate_efficiency_loss(
                df,
                df_fix[
                    "POA fixed"
                ],
                df_fix[
                    "Actual"
                ]
            )
        )

        df = apply_efficiency_loss(
            df,
            best_loss
        )

        # ----------------------------------------------------
        # Forecast
        # ----------------------------------------------------

        forecast = (
            df_fix[
                "POA fixed"
            ].to_numpy(
                dtype=float
            )
            * df[
                "Eff Area"
            ].sum()
            / 1_000_000
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.divider()

        st.header(
            "🏗️ Fixed Plant Results"
        )

        show_forecast_metrics(
            forecast,
            df_fix[
                "Actual"
            ].to_numpy(),
            best_loss
        )

        show_forecast_chart(
            forecast,
            df_fix[
                "Actual"
            ].to_numpy(),
            "Fixed Forecast vs Actual"
        )

        show_efficiency_table(
            df
        )

        st.session_state.model_result = {
            "forecast": forecast,
            "actual": df_fix[
                "Actual"
            ].to_numpy(),
            "efficiency_loss": best_loss,
        }

    # ========================================================
    # NON-CLUSTER TRACKING
    # ========================================================

    else:

        # ----------------------------------------------------
        # Backend Cal
        # ----------------------------------------------------

        df_bcal = pd.read_excel(
            uploaded_file,
            sheet_name="Backend Cal"
        )

        blocks = (
            df_bcal[
                "Block No."
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # POA used for efficiency loss
        # ----------------------------------------------------

        df_fix["POA fixed"] = (
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

        best_loss = (
            calculate_efficiency_loss(
                df,
                df_fix[
                    "POA fixed"
                ],
                df_fix[
                    "Actual"
                ]
            )
        )

        df = apply_efficiency_loss(
            df,
            best_loss
        )

        # ----------------------------------------------------
        # Weighted GHI
        # ----------------------------------------------------

        eff_area = (
            df[
                "Eff Area"
            ].sum()
        )

        weighted_ghi = (
            df_fix[
                "GHI_Forecast"
            ]
            .to_numpy(
                dtype=float
            )
            * eff_area
        )

        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

        if (
            st.session_state.tracking_params
            is None
        ):

            result = run_tracking_optimization(
                blocks,
                weighted_ghi,
                df_fix[
                    "Actual"
                ].to_numpy(
                    dtype=float
                )
            )

            st.session_state.tracking_params = (
                result
            )

        params = (
            st.session_state.tracking_params
        )

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        st.divider()

        st.header(
            "🔄 Tracking Model"
        )

        final_params, user_loss = (
            show_tracking_parameters(
                params,
                best_loss,
                "tracking"
            )
        )

        if final_params is None:

            st.stop()

        # ----------------------------------------------------
        # Apply edited loss
        # ----------------------------------------------------

        df = apply_efficiency_loss(
            df,
            user_loss
        )

        # ----------------------------------------------------
        # Recalculate weighted GHI
        #
        # Important:
        # efficiency loss affects Eff Area,
        # so weighted GHI must also update.
        # ----------------------------------------------------

        eff_area = (
            df[
                "Eff Area"
            ].sum()
        )

        weighted_ghi = (
            df_fix[
                "GHI_Forecast"
            ]
            .to_numpy(
                dtype=float
            )
            * eff_area
        )

        # ----------------------------------------------------
        # Final forecast
        # ----------------------------------------------------

        try:

            forecast = (
                calculate_tracking_forecast(
                    blocks,
                    weighted_ghi,
                    final_params
                )
            )

        except Exception as e:

            st.error(
                f"Unable to calculate forecast: {e}"
            )

            st.stop()

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.subheader(
            "📊 Forecast Results"
        )

        show_forecast_metrics(
            forecast,
            df_fix[
                "Actual"
            ].to_numpy(),
            user_loss
        )

        show_forecast_chart(
            forecast,
            df_fix[
                "Actual"
            ].to_numpy(),
            "Tracking Forecast vs Actual"
        )

        # ----------------------------------------------------
        # Optimization Details
        # ----------------------------------------------------

        with st.expander(
            "🧠 Optimization Details"
        ):

            col1, col2 = st.columns(2)

            col1.metric(
                "Optimization Score",
                f"{params['score']:.5f}"
            )

            col2.write(
                "The optimized values are shown above. "
                "You can modify them directly and the "
                "forecast updates automatically."
            )

        # ----------------------------------------------------
        # Efficiency table
        # ----------------------------------------------------

        show_efficiency_table(
            df
        )

        st.session_state.model_result = {
            "forecast": forecast,
            "actual": df_fix[
                "Actual"
            ].to_numpy(),
            "efficiency_loss": user_loss,
        }


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Loss Correction Model • Solar Forecasting & Performance Analysis"
)
