# ============================================================
# LOSS CORRECTION MODEL PAGE
# Optimized for Streamlit performance and stability
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.optimize import differential_evolution
from concurrent.futures import ThreadPoolExecutor
import time
import random


# ============================================================
# CONSTANTS
# ============================================================

MAX_OPT_ITER = 40
OPT_POPSIZE = 10

PARAM_BOUNDS = [
    (0, 10),    # DHI %
    (0, 30),    # GHI Starting Block
    (65, 80),   # GHI Ending Block
    (44, 60),   # GHI Max Block
    (0, 70),    # East Tracking Limit
    (0, 70),    # West Tracking Limit
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
# SESSION STATE
# ============================================================

if "tracking_params" not in st.session_state:
    st.session_state.tracking_params = None

if "model_context" not in st.session_state:
    st.session_state.model_context = None


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

    df.columns = df.columns.str.strip()

    if "Module Type" in df.columns:

        null_indices = df[df["Module Type"].isna()].index

        if len(null_indices) > 0:
            first_null_pos = df.index.get_loc(null_indices[0])
            df = df.iloc[:first_null_pos]

    return df.reset_index(drop=True)


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


def read_latitude(uploaded_file):

    df_st = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8
    )

    return float(df_st.loc[0, "Lat"])


def read_tilt_lookup(uploaded_file):

    df_tilt = pd.read_excel(
        uploaded_file,
        sheet_name="Config Tilt Angle",
        header=7
    )

    df_tilt.columns = df_tilt.columns.str.strip()

    if "Fixed" in df_tilt.columns:

        null_indices = df_tilt[df_tilt["Fixed"].isna()].index

        if len(null_indices) > 0:
            first_null_pos = df_tilt.index.get_loc(null_indices[0])
            df_tilt = df_tilt.iloc[:first_null_pos]

    df_tilt = df_tilt.dropna(how="all", axis=1)

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    if "Month" not in df_tilt.columns or "Fixed" not in df_tilt.columns:
        return {}

    return df_tilt.set_index("Month")["Fixed"].to_dict()


def prepare_solar_angles(df_fix, lat, tilt_lookup=None, tracking=False):

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
                360 * (284 + day_number) / 365
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

        df_fix["Tilt Angle b"] = (
            df_fix["Date"]
            .dt.strftime("%B")
            .map(tilt_lookup)
        )

    df_fix["a+b"] = (
        df_fix["Elevation angle a"]
        + df_fix["Tilt Angle b"]
    )

    df_fix["SIN(a+b)"] = np.sin(
        np.radians(df_fix["a+b"])
    )

    df_fix["Sin(a)"] = np.sin(
        np.radians(df_fix["Elevation angle a"])
    )

    # Avoid division by zero
    df_fix["Sin(a)"] = df_fix["Sin(a)"].clip(
        lower=1e-6
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
    """
    Directly calculates the loss required to match
    actual peak instead of testing every 0.1%.

    This replaces the expensive:

        for loss in np.arange(...)

    loop.
    """

    standard_eff = df[
        "Standard PV Efficiency (%)"
    ].to_numpy(dtype=float)

    area = df[
        "Total area(m2)"
    ].to_numpy(dtype=float)

    max_loss = float(
        np.nanmin(standard_eff)
    )

    actual_peak = float(
        np.nanmax(actual)
    )

    # Power = POA * SUM(area * (eff-loss)/100)
    #
    # Peak occurs at maximum POA.
    poa_peak = float(
        np.nanmax(poa)
    )

    if poa_peak <= 0:
        return 0.0

    base_area = np.sum(
        area * standard_eff / 100
    )

    loss_area_coefficient = np.sum(
        area / 100
    )

    if loss_area_coefficient <= 0:
        return 0.0

    # Target:
    #
    # poa_peak * (
    #     base_area
    #     - loss * loss_area_coefficient
    # ) / 1e6
    #
    # = actual_peak

    target_area = (
        actual_peak
        * 1_000_000
        / poa_peak
    )

    best_loss = (
        base_area - target_area
    ) / loss_area_coefficient

    best_loss = np.clip(
        best_loss,
        0,
        max_loss
    )

    return float(best_loss)


def apply_efficiency_loss(df, best_loss):

    df = df.copy()

    df["Efficiency Losses(%)"] = best_loss

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - best_loss
    )

    df["Eff Area"] = (
        df["Total area(m2)"]
        * df["Net Efficiency (%)"]
        / 100
    )

    return df


# ============================================================
# FIXED MODEL
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

        for ghi_col, weight_col in zip(
            ghi_cols,
            weight_cols
        ):

            df_fix[f"POA_{weight_col}"] = (
                df_fix[ghi_col]
                * df_fix["SIN(a+b)"]
                / df_fix["Sin(a)"]
            )

        weights = {}

        for weight_col in weight_cols:

            weights[weight_col] = (
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * cluster_weights[weight_col]
            ).sum()

        forecast = np.zeros(
            len(df_fix),
            dtype=float
        )

        for weight_col in weight_cols:

            forecast += (
                df_fix[f"POA_{weight_col}"].to_numpy()
                * weights[weight_col]
                / 1_000_000
            )

    else:

        df_fix["POA fixed"] = (
            df_fix["GHI_Forecast"]
            * df_fix["SIN(a+b)"]
            / df_fix["Sin(a)"]
        )

        eff_area = df["Eff Area"].sum()

        forecast = (
            df_fix["POA fixed"].to_numpy()
            * eff_area
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
        dtype=np.float64
    )

    weighted_ghi = np.asarray(
        weighted_ghi,
        dtype=np.float64
    )

    actual = np.asarray(
        actual,
        dtype=np.float64
    )

    # Daylight mask
    mask = actual != 0

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

        DHI = int(x[0])
        start = int(x[1])
        end = int(x[2])
        max_block = int(x[3])
        east = int(x[4])
        west = int(x[5])

        # -----------------------------------------------
        # Validate block sequence
        # -----------------------------------------------

        if (
            start >= max_block
            or max_block >= end
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

        # -----------------------------------------------
        # Zenith
        # -----------------------------------------------

        zenith = np.where(
            blocks_day <= max_block,

            np.minimum(
                89,
                m1 * (
                    blocks_day
                    - max_block
                )
            ),

            np.minimum(
                89,
                m2 * (
                    blocks_day
                    - max_block
                )
            )
        )

        # -----------------------------------------------
        # Panel angle
        # -----------------------------------------------

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

        # -----------------------------------------------
        # DNI
        # -----------------------------------------------

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
        )

        # -----------------------------------------------
        # Power scaling
        #
        # weighted_ghi already contains
        # efficiency-weighted area.
        # -----------------------------------------------

        prediction = (
            prediction
            / 1_000_000
        )

        if (
            np.isnan(prediction).any()
            or np.isinf(prediction).any()
        ):
            return 1e9

        # -----------------------------------------------
        # Error metrics
        # -----------------------------------------------

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

    # -------------------------------------------------------
    # Callback
    # -------------------------------------------------------

    generation = {
        "count": 0
    }

    def callback(xk, convergence):

        generation["count"] += 1

        if progress_callback is not None:

            progress_callback(
                generation["count"],
                maxiter
            )

        return False

    # -------------------------------------------------------
    # Integer optimization
    # -------------------------------------------------------

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

        # Newer scipy versions support this.
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
        "score": float(result.fun),
        "DHI": int(best[0]),
        "start": int(best[1]),
        "end": int(best[2]),
        "max": int(best[3]),
        "east": int(best[4]),
        "west": int(best[5]),
    }


# ============================================================
# FINAL TRACKING FORECAST
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

    if (
        start >= max_block
        or max_block >= end
    ):
        raise ValueError(
            "Starting Block < Max Block < Ending Block is required."
        )

    m1 = 90 / (
        start - 1 - max_block
    )

    m2 = 90 / (
        end + 1 - max_block
    )

    zenith = np.where(

        blocks <= max_block,

        np.minimum(
            89,
            m1 * (
                blocks
                - max_block
            )
        ),

        np.minimum(
            89,
            m2 * (
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
# DISPLAY EFFICIENCY TABLE
# ============================================================

def show_efficiency_table(df):

    display_df = df[
        [
            "Module Type",
            "Standard PV Efficiency (%)",
            "Efficiency Losses(%)",
            "Net Efficiency (%)",
            "Total area(m2)",
        ]
    ].copy()

    num_cols = display_df.select_dtypes(
        include="number"
    ).columns

    display_df[num_cols] = (
        display_df[num_cols]
        .round(2)
    )

    with st.expander(
        "🔍 View Efficiency Calculations"
    ):

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FORECAST VS ACTUAL CHART
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
            t=60,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# MODEL PAGE
# ============================================================

if st.session_state.get(
    "run_model",
    False
):

    if uploaded_file is None:

        st.warning(
            "Please upload the Excel file first."
        )

        st.stop()

    # ========================================================
    # MODEL CONTEXT
    # ========================================================

    current_context = (
        uploaded_file.name,
        uploaded_file.size,
        plant_type,
        bool(is_cluster),
    )

    # Reset optimizer when model/file changes
    if (
        st.session_state.model_context
        != current_context
    ):

        st.session_state.tracking_params = None

        st.session_state.model_context = (
            current_context
        )

    # ========================================================
    # LOAD AREA & EFFICIENCY
    # ========================================================

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

    # ========================================================
    # CLUSTER MODEL
    # ========================================================

    if is_cluster:

        cluster_weights = (
            read_cluster_weights(
                uploaded_file
            )
        )

        # ----------------------------------------------------
        # GHI DATA
        # ----------------------------------------------------

        df_ghi = pd.read_excel(
            uploaded_file,
            sheet_name="Result",
            usecols=[0, 1, 2, 3, 4, 5]
        )

        df_ghi = df_ghi.fillna(0)

        # ----------------------------------------------------
        # FIXED-CL1 DATA
        # ----------------------------------------------------

        df_fix = pd.read_excel(
            uploaded_file,
            sheet_name="Fixed-CL1",
            header=1
        )

        df_fix.columns = (
            df_fix.columns.str.strip()
        )

        if "Date" in df_fix.columns:

            null_indices = (
                df_fix[
                    df_fix["Date"].isna()
                ].index
            )

            if len(null_indices) > 0:

                first_null_pos = (
                    df_fix.index.get_loc(
                        null_indices[0]
                    )
                )

                df_fix = (
                    df_fix.iloc[
                        :first_null_pos
                    ]
                )

        df_fix = prepare_solar_angles(
            df_fix,
            lat,
            tilt_lookup,
            tracking=(
                plant_type
                == "🔄 Tracking"
            )
        )

        # ----------------------------------------------------
        # CLUSTER GHI
        # ----------------------------------------------------

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
                        df_ghi.iloc[:, i]
                        .to_numpy()
                    )

        # ====================================================
        # EFFICIENCY LOSS
        # ====================================================

        if plant_type == "🏗️ Fixed":

            # Calculate POA using CL1 as
            # reference for efficiency optimization.

            df_fix["POA fixed"] = (
                df_fix["CL1-GHI"]
                * df_fix["SIN(a+b)"]
                / df_fix["Sin(a)"]
            )

            best_loss = (
                calculate_efficiency_loss(
                    df,
                    df_fix["POA fixed"],
                    df_fix["Actual"]
                )
            )

        else:

            # Tracking uses zero tilt.
            # GHI itself is used for the loss scaling.

            df_fix["POA fixed"] = (
                df_fix["CL1-GHI"]
                * df_fix["SIN(a+b)"]
                / df_fix["Sin(a)"]
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

        # ====================================================
        # FIXED PLANT
        # ====================================================

        if plant_type == "🏗️ Fixed":

            forecast = calculate_fixed_forecast(
                df,
                df_fix,
                cluster=True,
                cluster_weights=cluster_weights
            )

            st.metric(
                "Efficiency Loss",
                f"{best_loss:.2f}%"
            )

            show_efficiency_table(
                df
            )

            show_forecast_chart(
                forecast,
                df_fix["Actual"].to_numpy(),
                "Fixed Cluster Forecast vs Actual"
            )

        # ====================================================
        # TRACKING PLANT
        # ====================================================

        else:

            # ------------------------------------------------
            # BACKEND CAL
            # ------------------------------------------------

            backend_sheets = [
                "Backend Cal CL1",
                "Backend Cal CL2",
                "Backend Cal CL3",
                "Backend Cal CL4",
                "Backend Cal CL5",
            ]

            backend_list = [
                pd.read_excel(
                    uploaded_file,
                    sheet_name=sheet
                )
                for sheet in backend_sheets
            ]

            blocks = (
                backend_list[0][
                    "Block No."
                ]
                .to_numpy(
                    dtype=float
                )
            )

            ghi_cols = [
                "CL1-GHI",
                "CL2-GHI",
                "CL3-GHI",
                "CL4-GHI",
                "CL5-GHI",
            ]

            # ------------------------------------------------
            # WEIGHTED GHI
            # ------------------------------------------------

            weighted_ghi = np.zeros(
                len(df_fix),
                dtype=float
            )

            for ghi_col, weight_key in zip(
                ghi_cols,
                [
                    "CL-1",
                    "CL-2",
                    "CL-3",
                    "CL-4",
                    "CL-5",
                ]
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
                    ghi * eff_area
                )

            # ------------------------------------------------
            # RUN OPTIMIZATION ONLY ONCE
            # ------------------------------------------------

            if (
                st.session_state.tracking_params
                is None
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

                # ------------------------------------------------
                # Run optimizer in background thread
                # ------------------------------------------------

                with ThreadPoolExecutor(
                    max_workers=1
                ) as executor:

                    future = executor.submit(
                        optimize_tracking,
                        blocks,
                        weighted_ghi,
                        df_fix["Actual"]
                        .to_numpy(
                            dtype=float
                        ),
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

                st.session_state.tracking_params = (
                    result
                )

                time.sleep(
                    0.3
                )

                progress_bar.empty()
                status_box.empty()

            # ------------------------------------------------
            # GET OPTIMIZED PARAMETERS
            # ------------------------------------------------

            params = (
                st.session_state.tracking_params
            )

            st.subheader(
                "Optimized Parameters"
            )

            # ------------------------------------------------
            # PARAMETER EDITING
            # ------------------------------------------------

            best_loss = st.number_input(
                "Efficiency Loss (%)",
                value=float(best_loss),
                step=0.1,
                key="tracking_loss_input"
            )

            col1, col2, col3 = st.columns(3)

            DHI = col1.number_input(
                "DHI (%)",
                value=int(params["DHI"]),
                step=1,
                key="tracking_dhi_input"
            )

            GHI_Starting_Block = col2.number_input(
                "Starting Block",
                value=int(params["start"]),
                step=1,
                key="tracking_start_input"
            )

            GHI_Ending_Block = col3.number_input(
                "Ending Block",
                value=int(params["end"]),
                step=1,
                key="tracking_end_input"
            )

            col1, col2, col3 = st.columns(3)

            GHI_Max_Block = col1.number_input(
                "Max Block",
                value=int(params["max"]),
                step=1,
                key="tracking_max_input"
            )

            Tracking_angle_lim_E = col2.number_input(
                "East Limit",
                value=int(params["east"]),
                step=1,
                key="tracking_east_input"
            )

            Tracking_angle_lim_W = col3.number_input(
                "West Limit",
                value=int(params["west"]),
                step=1,
                key="tracking_west_input"
            )

            # =================================================
            # APPLY USER EDITED LOSS
            # =================================================

            df = apply_efficiency_loss(
                df,
                best_loss
            )

            show_efficiency_table(
                df
            )

            # =================================================
            # FINAL TRACKING FORECAST
            # =================================================

            final_params = {
                "DHI": int(DHI),
                "start": int(
                    GHI_Starting_Block
                ),
                "end": int(
                    GHI_Ending_Block
                ),
                "max": int(
                    GHI_Max_Block
                ),
                "east": int(
                    Tracking_angle_lim_E
                ),
                "west": int(
                    Tracking_angle_lim_W
                ),
            }

            try:

                forecast = (
                    calculate_tracking_forecast(
                        blocks,
                        weighted_ghi,
                        final_params
                    )
                )

                show_forecast_chart(
                    forecast,
                    df_fix["Actual"].to_numpy(),
                    "Tracking Forecast vs Actual"
                )

            except Exception as e:

                st.error(
                    f"Unable to calculate forecast: {e}"
                )

    # ========================================================
    # NON-CLUSTER MODEL
    # ========================================================

    else:

        # ----------------------------------------------------
        # AREA & EFFICIENCY
        # ----------------------------------------------------

        df = read_area_efficiency(
            uploaded_file,
            cluster=False
        )

        # ----------------------------------------------------
        # FORECAST CONFIG
        # ----------------------------------------------------

        lat = read_latitude(
            uploaded_file
        )

        tilt_lookup = read_tilt_lookup(
            uploaded_file
        )

        # ----------------------------------------------------
        # FIXED SHEET
        # ----------------------------------------------------

        df_fix = pd.read_excel(
            uploaded_file,
            sheet_name="Fixed",
            header=1
        )

        df_fix.columns = (
            df_fix.columns.str.strip()
        )

        # ----------------------------------------------------
        # USE EDITED DATA WHEN AVAILABLE
        # ----------------------------------------------------

        if (
            "edited_df" in locals()
            and edited_df is not None
        ):

            if (
                "GHI_Forecast"
                in edited_df.columns
            ):

                df_fix["GHI_Forecast"] = (
                    edited_df[
                        "GHI_Forecast"
                    ].to_numpy()
                )

            if (
                "Actual"
                in edited_df.columns
            ):

                df_fix["Actual"] = (
                    edited_df[
                        "Actual"
                    ].to_numpy()
                )

        # ----------------------------------------------------
        # CUT EMPTY ROWS
        # ----------------------------------------------------

        if "Date" in df_fix.columns:

            null_indices = (
                df_fix[
                    df_fix["Date"].isna()
                ].index
            )

            if len(null_indices) > 0:

                first_null_pos = (
                    df_fix.index.get_loc(
                        null_indices[0]
                    )
                )

                df_fix = (
                    df_fix.iloc[
                        :first_null_pos
                    ]
                )

        df_fix["Actual"] = (
            df_fix["Actual"]
            .fillna(0)
        )

        # ====================================================
        # SOLAR ANGLES
        # ====================================================

        df_fix = prepare_solar_angles(
            df_fix,
            lat,
            tilt_lookup,
            tracking=(
                plant_type
                == "🔄 Tracking"
            )
        )

        # ====================================================
        # FIXED
        # ====================================================

        if plant_type == "🏗️ Fixed":

            df_fix["POA fixed"] = (
                df_fix["GHI_Forecast"]
                * df_fix["SIN(a+b)"]
                / df_fix["Sin(a)"]
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

            forecast = (
                df_fix["POA fixed"]
                .to_numpy()
                * df["Eff Area"].sum()
                / 1_000_000
            )

            st.metric(
                "Efficiency Loss",
                f"{best_loss:.2f}%"
            )

            show_efficiency_table(
                df
            )

            show_forecast_chart(
                forecast,
                df_fix["Actual"].to_numpy(),
                "Fixed Forecast vs Actual"
            )

        # ====================================================
        # TRACKING
        # ====================================================

        else:

            # ------------------------------------------------
            # BACKEND CAL
            # ------------------------------------------------

            df_bcal = pd.read_excel(
                uploaded_file,
                sheet_name="Backend Cal"
            )

            df_trac = pd.read_excel(
                uploaded_file,
                sheet_name="Tracking",
                header=1
            )

            blocks = (
                df_bcal[
                    "Block No."
                ].to_numpy(
                    dtype=float
                )
            )

            # ------------------------------------------------
            # EFFICIENCY LOSS
            # ------------------------------------------------

            df_fix["POA fixed"] = (
                df_fix["GHI_Forecast"]
                * df_fix["SIN(a+b)"]
                / df_fix["Sin(a)"]
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

            # ------------------------------------------------
            # WEIGHTED GHI
            # ------------------------------------------------

            eff_area = (
                df["Eff Area"].sum()
            )

            weighted_ghi = (
                df_fix[
                    "GHI_Forecast"
                ].to_numpy(
                    dtype=float
                )
                * eff_area
            )

            # ------------------------------------------------
            # OPTIMIZATION
            # ------------------------------------------------

            if (
                st.session_state.tracking_params
                is None
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
                        df_fix[
                            "Actual"
                        ].to_numpy(
                            dtype=float
                        ),
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

                st.session_state.tracking_params = (
                    result
                )

                time.sleep(
                    0.3
                )

                progress_bar.empty()
                status_box.empty()

            # ------------------------------------------------
            # PARAMETERS
            # ------------------------------------------------

            params = (
                st.session_state.tracking_params
            )

            st.subheader(
                "Optimized Parameters"
            )

            best_loss = st.number_input(
                "Efficiency Loss (%)",
                value=float(best_loss),
                step=0.1,
                key="noncluster_loss_input"
            )

            col1, col2, col3 = st.columns(3)

            DHI = col1.number_input(
                "DHI (%)",
                value=int(params["DHI"]),
                step=1,
                key="noncluster_dhi_input"
            )

            GHI_Starting_Block = col2.number_input(
                "Starting Block",
                value=int(params["start"]),
                step=1,
                key="noncluster_start_input"
            )

            GHI_Ending_Block = col3.number_input(
                "Ending Block",
                value=int(params["end"]),
                step=1,
                key="noncluster_end_input"
            )

            col1, col2, col3 = st.columns(3)

            GHI_Max_Block = col1.number_input(
                "Max Block",
                value=int(params["max"]),
                step=1,
                key="noncluster_max_input"
            )

            Tracking_angle_lim_E = col2.number_input(
                "East Limit",
                value=int(params["east"]),
                step=1,
                key="noncluster_east_input"
            )

            Tracking_angle_lim_W = col3.number_input(
                "West Limit",
                value=int(params["west"]),
                step=1,
                key="noncluster_west_input"
            )

            # ------------------------------------------------
            # APPLY EDITED LOSS
            # ------------------------------------------------

            df = apply_efficiency_loss(
                df,
                best_loss
            )

            show_efficiency_table(
                df
            )

            # ------------------------------------------------
            # FINAL FORECAST
            # ------------------------------------------------

            final_params = {
                "DHI": int(DHI),
                "start": int(
                    GHI_Starting_Block
                ),
                "end": int(
                    GHI_Ending_Block
                ),
                "max": int(
                    GHI_Max_Block
                ),
                "east": int(
                    Tracking_angle_lim_E
                ),
                "west": int(
                    Tracking_angle_lim_W
                ),
            }

            try:

                forecast = (
                    calculate_tracking_forecast(
                        blocks,
                        weighted_ghi,
                        final_params
                    )
                )

                show_forecast_chart(
                    forecast,
                    df_fix["Actual"].to_numpy(),
                    "Tracking Forecast vs Actual"
                )

            except Exception as e:

                st.error(
                    f"Unable to calculate forecast: {e}"
                )
