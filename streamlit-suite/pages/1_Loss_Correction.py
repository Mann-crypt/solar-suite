import io
import random
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Suite",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# AUTO REFRESH AFTER 10 MINUTES OF INACTIVITY
# ============================================================

components.html(
    """
    <script>
    let timer;

    function resetTimer() {
        clearTimeout(timer);

        timer = setTimeout(() => {
            window.location.reload();
        }, 600000);
    }

    ["mousemove", "mousedown", "keydown", "scroll", "touchstart"]
        .forEach(e => {
            document.addEventListener(e, resetTimer);
        });

    resetTimer();
    </script>
    """,
    height=0
)


# ============================================================
# SIDEBAR CSS
# ============================================================

st.markdown(
    """
    <style>

    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }

    .sidebar-bottom {
        margin-top: auto;
        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <h1 style='
        text-align:center;
        background: linear-gradient(90deg,#00c6ff,#0072ff);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        font-size:40px;
        font-weight:800;
    '>
        ⚡ Solar Suite
    </h1>

    <p style='
        text-align:center;
        color:gray;
        font-size:14px;
    '>
        Forecast Correction Platform
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()


# ============================================================
# NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Loss Correction"


if st.sidebar.button(
    "⛅ Loss Correction",
    use_container_width=True
):
    st.session_state.page = "Loss Correction"


if st.sidebar.button(
    "⏰ RT Correction",
    use_container_width=True
):
    st.session_state.page = "RT Correction"


if st.sidebar.button(
    "🐱‍🏍 Aeromal",
    use_container_width=True
):
    st.session_state.page = "Aeromal"


st.sidebar.divider()


# ============================================================
# AEROMAL LOGOUT
# ============================================================

if st.session_state.get("aeromal_auth", False):

    if st.sidebar.button(
        "🚪 Logout",
        use_container_width=True,
        key="logout"
    ):

        st.session_state.aeromal_auth = False
        st.session_state.page = "Loss Correction"

        st.rerun()


# ============================================================
# CREDITS
# ============================================================

st.sidebar.markdown(
    """
    <div style='
        text-align:center;
        color:gray;
        font-size:13px;
    '>

    Developed and Maintained by:<br>
    <b>Manjot Singh</b><br><br>

    Scripter Writer:<br>
    <b>Tushar Sharma</b><br><br>

    Challenger:<br>
    <b>Aarav Sharma</b><br><br>

    Tester:<br>
    <b>Jatin Chaturvedi</b><br><br>

    Improviser:<br>
    <b>Ujala Agrahari</b><br><br>

    Suggested by:<br>
    <b>Garima Bajetha</b>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EXCEL HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes):

    xls = pd.ExcelFile(io.BytesIO(file_bytes))

    return xls.sheet_names


@st.cache_data(show_spinner=False)
def read_excel_cached(
    file_bytes,
    sheet_name,
    header=0,
    usecols=None
):

    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
        header=header,
        usecols=usecols
    )


def trim_at_first_null(
    df,
    column
):

    df = df.copy()

    if column not in df.columns:
        return df

    null_indices = df[df[column].isna()].index

    if len(null_indices) > 0:

        first_null = null_indices[0]

        df = df.loc[:first_null - 1]

    return df.reset_index(drop=True)


def clean_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# DATE / SOLAR CALCULATION
# ============================================================

def add_solar_geometry(
    df,
    lat
):

    df = df.copy()

    today = pd.Timestamp.today().normalize()

    first_date = today.replace(
        month=1,
        day=1
    )

    days = (
        today - first_date
    ).days

    declination = 23.45 * np.sin(
        np.radians(
            360 * (284 + days + 1) / 365
        )
    )

    df["Date"] = today

    df["Declination Angle ∆"] = declination

    df["Elevation angle a"] = (
        90 - lat + declination
    )

    return df


# ============================================================
# FIXED SOLAR GEOMETRY
# ============================================================

def calculate_fixed_poa(
    df,
    ghi_column,
    tilt_angle
):

    df = df.copy()

    df["Tilt Angle b"] = tilt_angle

    df["a+b"] = (
        df["Elevation angle a"]
        + df["Tilt Angle b"]
    )

    df["SIN(a+b)"] = np.sin(
        np.radians(df["a+b"])
    )

    df["Sin(a)"] = np.sin(
        np.radians(df["Elevation angle a"])
    )

    safe_sin_a = np.where(
        np.abs(df["Sin(a)"]) < 1e-6,
        1e-6,
        df["Sin(a)"]
    )

    df["GHI*sin(a)"] = (
        df[ghi_column]
        * df["Sin(a)"]
    )

    df["GHI*sin(a+b)"] = (
        df[ghi_column]
        * df["SIN(a+b)"]
    )

    df["POA fixed"] = (
        df["GHI*sin(a+b)"]
        / safe_sin_a
    )

    return df


# ============================================================
# EFFICIENCY LOSS OPTIMIZATION
# ============================================================

def optimize_efficiency_loss(
    df,
    base_forecast,
    actual
):

    df = df.copy()

    standard_eff = (
        pd.to_numeric(
            df["Standard PV Efficiency (%)"],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    area = (
        pd.to_numeric(
            df["Total area(m2)"],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    max_loss = float(
        np.nanmin(standard_eff)
    )

    if max_loss <= 0:
        return 0.0

    actual_peak = float(
        np.nanmax(actual)
    )

    if actual_peak <= 0:
        return 0.0

    results = []

    # Base forecast at zero loss.
    #
    # Because efficiency loss reduces every module's
    # effective area proportionally, the complete forecast
    # can be scaled efficiently rather than rebuilding
    # every dataframe for every 0.1% loss.

    base_eff_area = (
        area * standard_eff / 100
    )

    base_eff_area_sum = np.sum(
        base_eff_area
    )

    if base_eff_area_sum <= 0:
        return 0.0

    # Forecast corresponding to zero loss
    zero_loss_forecast = (
        base_forecast.copy()
    )

    zero_peak = float(
        np.nanmax(zero_loss_forecast)
    )

    if zero_peak <= 0:
        return 0.0

    losses = np.arange(
        0,
        max_loss + 0.0001,
        0.1
    )

    for loss in losses:

        net_eff = (
            standard_eff - loss
        )

        eff_area_sum = np.sum(
            area * net_eff / 100
        )

        scale = (
            eff_area_sum
            / base_eff_area_sum
        )

        predicted_peak = (
            zero_peak * scale
        )

        peak_error = abs(
            actual_peak
            - predicted_peak
        )

        results.append(
            {
                "Efficiency Loss (%)": loss,
                "Actual Peak": actual_peak,
                "Predicted Peak": predicted_peak,
                "Peak Error": peak_error
            }
        )

    results_df = pd.DataFrame(
        results
    )

    best_loss = float(
        results_df.loc[
            results_df["Peak Error"].idxmin(),
            "Efficiency Loss (%)"
        ]
    )

    return best_loss


# ============================================================
# BUILD EFFICIENCY DATA
# ============================================================

def apply_efficiency_loss(
    df,
    loss
):

    df = df.copy()

    df["Efficiency Losses(%)"] = loss

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - df["Efficiency Losses(%)"]
    )

    df["Eff Area"] = (
        df["Total area(m2)"]
        * df["Net Efficiency (%)"]
        / 100
    )

    return df


# ============================================================
# DISPLAY EFFICIENCY TABLE
# ============================================================

def show_efficiency_table(
    df
):

    display_df = df[
        [
            "Module Type",
            "Standard PV Efficiency (%)",
            "Efficiency Losses(%)",
            "Net Efficiency (%)",
            "Total area(m2)"
        ]
    ].copy()

    numeric_columns = (
        display_df
        .select_dtypes(include="number")
        .columns
    )

    display_df[numeric_columns] = (
        display_df[numeric_columns]
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
# CLUSTER WEIGHTS
# ============================================================

def calculate_cluster_weights(
    df,
    df_w
):

    weights = {}

    for cluster_number in range(1, 6):

        efficiency_area = (
            df["Total area(m2)"]
            * df["Net Efficiency (%)"]
            / 100
        )

        column = f"CL-{cluster_number}"

        weight_factor = float(
            df_w[column]
            .iloc[0]
        )

        weights[column] = (
            efficiency_area
            * weight_factor
        )

    return weights


# ============================================================
# CLUSTER FIXED FORECAST
# ============================================================

def calculate_cluster_fixed_forecast(
    df_fix,
    df,
    df_w,
    ghi_cols,
    lat,
    tilt
):

    temp = add_solar_geometry(
        df_fix,
        lat
    )

    forecasts = {}

    for i, ghi_column in enumerate(
        ghi_cols,
        start=1
    ):

        poa_df = calculate_fixed_poa(
            temp,
            ghi_column,
            tilt
        )

        weight_column = (
            f"CL-{i}"
        )

        eff_area = (
            df["Total area(m2)"]
            * df["Net Efficiency (%)"]
            / 100
        )

        weight_factor = float(
            df_w[weight_column]
            .iloc[0]
        )

        total_weight = (
            eff_area
            * weight_factor
        ).sum()

        forecasts[
            f"CL{i}"
        ] = (
            poa_df["POA fixed"].to_numpy()
            * total_weight
            / 1_000_000
        )

    forecast = np.sum(
        list(forecasts.values()),
        axis=0
    )

    return forecast


# ============================================================
# NON-CLUSTER FIXED FORECAST
# ============================================================

def calculate_noncluster_fixed_forecast(
    df_fix,
    df,
    lat,
    tilt
):

    temp = add_solar_geometry(
        df_fix,
        lat
    )

    temp = calculate_fixed_poa(
        temp,
        "GHI_Forecast",
        tilt
    )

    eff_area = (
        df["Total area(m2)"]
        * df["Net Efficiency (%)"]
        / 100
    ).sum()

    forecast = (
        temp["POA fixed"].to_numpy()
        * eff_area
        / 1_000_000
    )

    return forecast


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking_forecast(
    ghi_arrays,
    blocks,
    weights,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit
):

    start_block = int(
        round(start_block)
    )

    end_block = int(
        round(end_block)
    )

    max_block = int(
        round(max_block)
    )

    east_limit = int(
        round(east_limit)
    )

    west_limit = int(
        round(west_limit)
    )

    # Invalid configuration
    if (
        start_block >= max_block
        or max_block >= end_block
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

    blocks = np.asarray(
        blocks,
        dtype=float
    )

    zenith = np.where(
        blocks <= max_block,

        np.minimum(
            89.0,
            m1 * (
                blocks
                - max_block
            )
        ),

        np.minimum(
            89.0,
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
            abs(east_limit)
        ),

        np.where(
            (
                (blocks > max_block)
                &
                (zenith > west_limit)
            ),

            west_limit,

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

    DHI = float(DHI)

    forecast = np.zeros(
        len(blocks),
        dtype=float
    )

    for i, ghi in enumerate(
        ghi_arrays
    ):

        ghi = np.asarray(
            ghi,
            dtype=float
        )

        dhi = (
            ghi
            * DHI
            / 100
        )

        dni = (
            ghi - dhi
        ) / cos_alpha

        forecast += (
            dni
            * weights[i]
            / 1_000_000
        )

    return forecast


# ============================================================
# TRACKING OBJECTIVE FUNCTION
# ============================================================

def optimize_tracking_parameters(
    actual,
    ghi_arrays,
    blocks,
    weights,
    max_iterations=100
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    ghi_arrays = [
        np.asarray(
            x,
            dtype=float
        )
        for x in ghi_arrays
    ]

    blocks = np.asarray(
        blocks,
        dtype=float
    )

    # Use only non-zero actual blocks
    daylight_mask = (
        actual != 0
    )

    actual_day = actual[
        daylight_mask
    ]

    if len(actual_day) == 0:
        return None

    actual_max = float(
        np.max(actual_day)
    )

    actual_sum = float(
        np.sum(actual_day)
    )

    if (
        actual_max <= 0
        or actual_sum <= 0
    ):

        return None

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

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

        forecast = calculate_tracking_forecast(
            ghi_arrays=ghi_arrays,
            blocks=blocks,
            weights=weights,
            DHI=DHI,
            start_block=start_block,
            end_block=end_block,
            max_block=max_block,
            east_limit=east_limit,
            west_limit=west_limit
        )

        if forecast is None:
            return 1e9

        if (
            np.isnan(forecast).any()
            or np.isinf(forecast).any()
        ):

            return 1e9

        forecast_day = forecast[
            daylight_mask
        ]

        if len(forecast_day) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    - forecast_day
                )
            )
            / actual_max
        )

        peak_error = (
            abs(
                actual_max
                - np.max(forecast_day)
            )
            / actual_max
        )

        energy_error = (
            abs(
                actual_sum
                - np.sum(forecast_day)
            )
            / actual_sum
        )

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return float(score)

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    bounds = [
        (0, 10),       # DHI
        (0, 30),       # Starting block
        (65, 80),      # Ending block
        (44, 60),      # Maximum block
        (0, 70),       # East limit
        (0, 70)        # West limit
    ]

    # --------------------------------------------------------
    # Differential Evolution
    # --------------------------------------------------------

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=max_iterations,
        popsize=15,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
        updating="immediate"
    )

    best = np.round(
        result.x
    ).astype(int)

    return {
        "DHI": int(best[0]),
        "start": int(best[1]),
        "end": int(best[2]),
        "max": int(best[3]),
        "east": int(best[4]),
        "west": int(best[5]),
        "score": float(result.fun)
    }


# ============================================================
# CACHED TRACKING OPTIMIZATION
#
# IMPORTANT:
# The optimizer is cached.
#
# Changing number_input values does NOT execute this again.
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=20
)
def cached_tracking_optimization(
    actual,
    ghi_arrays,
    blocks,
    weights
):

    return optimize_tracking_parameters(
        actual=np.array(actual),
        ghi_arrays=[
            np.array(x)
            for x in ghi_arrays
        ],
        blocks=np.array(blocks),
        weights=np.array(weights),
        max_iterations=100
    )


# ============================================================
# OPTIMIZATION STATUS MESSAGES
# ============================================================

QUOTES = [
    "☕ Vo kehte the kya ho tum, aaj hum kehte hai tum kya ho be?",
    "🌦 Aapka mann nahi kar raha bahar jaane ka?..",
    "😊 Jinke ghar sheeshe ke bane hote hai vo basement mai kapde change krte h...",
    "😋 Aromatic Rose Latte with Frothy Milk pine ka mann hor hai na...",
    "🥛 Garmi mai daalo dudh mai Ice🧊 Dudh bangya Very Nice - Dudh Dudh Dudh Dudh...",
    "🌟 Aapke face pr toh Modiji se bhi jyda glow hai..",
    "😁 Horaha hai benstokes Kaan mai ghusjao insaan ke...",
    "😗 Muskuraiye aap MAL mai hai...",
    "🥱 Hum na hote toh Operations ka kya hota?..",
    "😎 6:30 hote hi Billu MAL se faraar...",
    "😇 Guruji ne ek baat kahi thi....",
    "🎼 Karna hai kuchh kaam M se gaao...",
    "😠 Nahi karni Loss Correction, Now what to do?...",
    "💸 Iss Job ko chhod or chhod kar ameer ho.."
]


# ============================================================
# SHOW OPTIMIZATION
# ============================================================

def run_tracking_optimizer(
    actual,
    ghi_arrays,
    blocks,
    weights
):

    quote = random.choice(
        QUOTES
    )

    with st.spinner(
        f"{quote}\n\n"
        "Optimization ho raha hai... "
        "Aap tab tak saath waale se baat karlo 🗣"
    ):

        result = cached_tracking_optimization(
            tuple(
                np.asarray(actual)
                .round(8)
            ),

            tuple(
                tuple(
                    np.asarray(x)
                    .round(8)
                )
                for x in ghi_arrays
            ),

            tuple(
                np.asarray(blocks)
                .round(8)
            ),

            tuple(
                np.asarray(weights)
                .round(8)
            )
        )

    return result


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_forecast_vs_actual(
    forecast,
    actual,
    title="Forecast vs Actual Power"
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
# LOSS CORRECTION PAGE
# ============================================================

def loss_correction_page():

    st.title(
        "Pakima Pakam Ravi, "
        "3-4 Loss Correction kar chuke hai!!😎"
    )

    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Yaha Feko!!",
        type=["xlsx"],
        key="excel_uploader"
    )

    if uploaded_file is None:

        st.info(
            "Pehle File toh upload karo!!!"
        )

        st.stop()

    # --------------------------------------------------------
    # Convert upload to bytes
    # --------------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    # File signature
    file_signature = (
        uploaded_file.name,
        len(file_bytes),
        hash(file_bytes)
    )

    # --------------------------------------------------------
    # Reset ONLY when a different workbook is uploaded
    # --------------------------------------------------------

    if (
        "file_signature"
        not in st.session_state
        or
        st.session_state.file_signature
        != file_signature
    ):

        # Remove old optimization state
        for key in list(
            st.session_state.keys()
        ):

            if key.startswith(
                "loss_"
            ):

                del st.session_state[key]

        st.session_state.file_signature = (
            file_signature
        )

    # --------------------------------------------------------
    # Workbook detection
    # --------------------------------------------------------

    sheet_names = get_sheet_names(
        file_bytes
    )

    is_cluster = (
        "Fixed-CL1"
        in sheet_names
    )

    if is_cluster:

        main_sheet = "Fixed-CL1"

        ghi_cols = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI"
        ]

    else:

        main_sheet = "Fixed"

        ghi_cols = [
            "GHI_Forecast"
        ]

    # --------------------------------------------------------
    # Read initial input sheet
    # --------------------------------------------------------

    df_input = read_excel_cached(
        file_bytes,
        main_sheet,
        header=[1]
    )

    df_input = clean_columns(
        df_input
    )

    if "Actual" in df_input.columns:

        df_input["Actual"] = (
            pd.to_numeric(
                df_input["Actual"],
                errors="coerce"
            )
            .fillna(0)
        )

    # Remove empty rows
    df_input = trim_at_first_null(
        df_input,
        "Date"
    )

    # Keep only first 96 blocks
    df_input = (
        df_input
        .iloc[:96]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Input editor
    # --------------------------------------------------------

    st.subheader(
        "Input Data"
    )

    input_df = df_input[
        ghi_cols + ["Actual"]
    ].copy()

    original_df = input_df.copy()

    edited_df = st.data_editor(
        input_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="solar_input_editor"
    )

    edited_df = (
        edited_df
        .iloc[:96]
        .copy()
        .reset_index(drop=True)
    )

    for column in ghi_cols:

        edited_df[column] = (
            pd.to_numeric(
                edited_df[column],
                errors="coerce"
            )
            .fillna(0)
        )

    edited_df["Actual"] = (
        pd.to_numeric(
            edited_df["Actual"],
            errors="coerce"
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Show edit notification
    # --------------------------------------------------------

    try:

        changed_rows = (
            edited_df
            .ne(
                original_df
                .fillna(0)
            )
        )
        .any(axis=1)

        if changed_rows.any():

            st.toast(
                f"✨ {changed_rows.sum()} rows updated successfully!",
                icon="✅"
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # Plant type
    # --------------------------------------------------------

    plant_type = st.pills(
        "Select Plant Type",
        [
            "🏗️ Fixed",
            "🔄 Tracking"
        ],
        default="🏗️ Fixed"
    )

    # ========================================================
    # NON-CLUSTER
    # ========================================================

    if not is_cluster:

        process_noncluster(
            file_bytes=file_bytes,
            edited_df=edited_df,
            plant_type=plant_type,
            sheet_names=sheet_names
        )

    # ========================================================
    # CLUSTER
    # ========================================================

    else:

        process_cluster(
            file_bytes=file_bytes,
            edited_df=edited_df,
            plant_type=plant_type,
            sheet_names=sheet_names
        )


# ============================================================
# NON-CLUSTER PROCESSING
# ============================================================

def process_noncluster(
    file_bytes,
    edited_df,
    plant_type,
    sheet_names
):

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    df = read_excel_cached(
        file_bytes,
        "Area & Efficiency",
        header=[1]
    )

    df = clean_columns(df)

    df = trim_at_first_null(
        df,
        "Module Type"
    )

    # --------------------------------------------------------
    # Forecast Config
    # --------------------------------------------------------

    df_st = read_excel_cached(
        file_bytes,
        "Forecast Config",
        header=[8]
    )

    df_st = clean_columns(
        df_st
    )

    lat = float(
        df_st.loc[
            0,
            "Lat"
        ]
    )

    # --------------------------------------------------------
    # Tilt configuration
    # --------------------------------------------------------

    df_tilt = read_excel_cached(
        file_bytes,
        "Config Tilt Angle",
        header=[7]
    )

    df_tilt = clean_columns(
        df_tilt
    )

    if "Fixed" not in df_tilt.columns:

        st.error(
            "Config Tilt Angle sheet mein 'Fixed' column nahi mila."
        )

        return

    df_tilt["Fixed"] = (
        pd.to_numeric(
            df_tilt["Fixed"],
            errors="coerce"
        )
    )

    df_tilt = trim_at_first_null(
        df_tilt,
        "Fixed"
    )

    df_tilt = df_tilt.dropna(
        how="all",
        axis=1
    )

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month"
        }
    )

    # --------------------------------------------------------
    # Current month tilt
    # --------------------------------------------------------

    current_month = (
        pd.Timestamp.today()
        .strftime("%B")
    )

    month_lookup = (
        df_tilt
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    tilt = month_lookup.get(
        current_month,
        0
    )

    # ========================================================
    # FIXED
    # ========================================================

    if plant_type == "🏗️ Fixed":

        df_fix = read_excel_cached(
            file_bytes,
            "Fixed",
            header=[1]
        )

        df_fix = clean_columns(
            df_fix
        )

        # Use edited input
        df_fix["GHI_Forecast"] = (
            edited_df["GHI_Forecast"].values
        )

        df_fix["Actual"] = (
            edited_df["Actual"].values
        )

        df_fix = trim_at_first_null(
            df_fix,
            "Date"
        )

        df_fix = (
            df_fix
            .iloc[:96]
            .copy()
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Solar geometry
        # ----------------------------------------------------

        df_fix = add_solar_geometry(
            df_fix,
            lat
        )

        df_fix = calculate_fixed_poa(
            df_fix,
            "GHI_Forecast",
            tilt
        )

        actual = (
            df_fix["Actual"]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Base forecast
        # ----------------------------------------------------

        base_eff_area = (
            df["Total area(m2)"]
            * df["Standard PV Efficiency (%)"]
            / 100
        ).sum()

        base_forecast = (
            df_fix["POA fixed"]
            .to_numpy(
                dtype=float
            )
            * base_eff_area
            / 1_000_000
        )

        # ----------------------------------------------------
        # Efficiency optimization
        # ----------------------------------------------------

        cache_key = (
            "noncluster_fixed_loss",
            st.session_state.file_signature
        )

        if (
            "loss_result"
            not in st.session_state
            or
            st.session_state.get(
                "loss_result_key"
            ) != cache_key
        ):

            with st.spinner(
                "Efficiency Loss calculate ho raha hai... ⏳"
            ):

                best_loss = (
                    optimize_efficiency_loss(
                        df=df,
                        base_forecast=base_forecast,
                        actual=actual
                    )
                )

            st.session_state.loss_result = (
                best_loss
            )

            st.session_state.loss_result_key = (
                cache_key
            )

        else:

            best_loss = (
                st.session_state.loss_result
            )

        # ----------------------------------------------------
        # Efficiency controls
        # ----------------------------------------------------

        st.subheader(
            "Optimized Parameters"
        )

        loss_key = (
            "loss_noncluster_fixed"
        )

        if loss_key not in st.session_state:

            st.session_state[
                loss_key
            ] = float(best_loss)

        best_loss = st.number_input(
            "Efficiency Loss (%)",
            min_value=0.0,
            step=0.1,
            key=loss_key
        )

        # ----------------------------------------------------
        # Final calculation
        # ----------------------------------------------------

        df_final = apply_efficiency_loss(
            df,
            best_loss
        )

        forecast = (
            df_fix["POA fixed"]
            .to_numpy(
                dtype=float
            )
            * df_final["Eff Area"].sum()
            / 1_000_000
        )

        st.metric(
            "Efficiency Loss",
            f"{best_loss:.2f}%"
        )

        show_efficiency_table(
            df_final
        )

        plot_forecast_vs_actual(
            forecast,
            actual
        )

    # ========================================================
    # TRACKING
    # ========================================================

    else:

        df_fix = read_excel_cached(
            file_bytes,
            "Fixed",
            header=[1]
        )

        df_fix = clean_columns(
            df_fix
        )

        df_fix["GHI_Forecast"] = (
            edited_df["GHI_Forecast"].values
        )

        df_fix["Actual"] = (
            edited_df["Actual"].values
        )

        df_fix = trim_at_first_null(
            df_fix,
            "Date"
        )

        df_fix = (
            df_fix
            .iloc[:96]
            .copy()
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Solar geometry
        # ----------------------------------------------------

        df_fix = add_solar_geometry(
            df_fix,
            lat
        )

        # Tracking uses zero tilt
        df_fix["Tilt Angle b"] = 0

        df_fix["a+b"] = (
            df_fix["Elevation angle a"]
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

        # ----------------------------------------------------
        # Backend Cal
        # ----------------------------------------------------

        df_bcal = read_excel_cached(
            file_bytes,
            "Backend Cal"
        )

        df_bcal = clean_columns(
            df_bcal
        )

        blocks = (
            df_bcal["Block No."]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Efficiency base
        # ----------------------------------------------------

        base_eff_area = (
            df["Total area(m2)"]
            * df["Standard PV Efficiency (%)"]
            / 100
        ).sum()

        ghi = (
            df_fix["GHI_Forecast"]
            .to_numpy(
                dtype=float
            )
        )

        # Base tracking forecast.
        #
        # This is only used to optimize efficiency loss.
        # Tracking parameters are handled separately.

        base_tracking = (
            ghi
            * base_eff_area
            / 1_000_000
        )

        actual = (
            df_fix["Actual"]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Efficiency loss
        # ----------------------------------------------------

        cache_key = (
            "noncluster_tracking_loss",
            st.session_state.file_signature
        )

        if (
            "loss_result"
            not in st.session_state
            or
            st.session_state.get(
                "loss_result_key"
            ) != cache_key
        ):

            with st.spinner(
                "Efficiency Loss calculate ho raha hai... ⏳"
            ):

                best_loss = (
                    optimize_efficiency_loss(
                        df=df,
                        base_forecast=base_tracking,
                        actual=actual
                    )
                )

            st.session_state.loss_result = (
                best_loss
            )

            st.session_state.loss_result_key = (
                cache_key
            )

        else:

            best_loss = (
                st.session_state.loss_result
            )

        # ----------------------------------------------------
        # Efficiency
        # ----------------------------------------------------

        df_final = apply_efficiency_loss(
            df,
            best_loss
        )

        eff_area = (
            df_final["Eff Area"]
            .sum()
        )

        # ----------------------------------------------------
        # Optimize tracking parameters
        # ----------------------------------------------------

        optimization_key = (
            "tracking_noncluster",
            st.session_state.file_signature
        )

        if (
            "tracking_params"
            not in st.session_state
            or
            st.session_state.get(
                "tracking_params_key"
            ) != optimization_key
        ):

            weights = np.array(
                [eff_area],
                dtype=float
            )

            optimization_result = (
                run_tracking_optimizer(
                    actual=actual,
                    ghi_arrays=[
                        ghi
                    ],
                    blocks=blocks,
                    weights=weights
                )
            )

            if optimization_result is None:

                st.error(
                    "Tracking optimization failed."
                )

                return

            st.session_state.tracking_params = (
                optimization_result
            )

            st.session_state.tracking_params_key = (
                optimization_key
            )

        optimization_result = (
            st.session_state.tracking_params
        )

        # ----------------------------------------------------
        # Parameter controls
        # ----------------------------------------------------

        st.subheader(
            "Optimized Parameters"
        )

        params = optimization_result

        if "loss_noncluster_tracking" not in st.session_state:
            st.session_state.loss_noncluster_tracking = (
                float(best_loss)
            )

        if "dhi_noncluster_tracking" not in st.session_state:
            st.session_state.dhi_noncluster_tracking = (
                int(params["DHI"])
            )

        if "start_noncluster_tracking" not in st.session_state:
            st.session_state.start_noncluster_tracking = (
                int(params["start"])
            )

        if "end_noncluster_tracking" not in st.session_state:
            st.session_state.end_noncluster_tracking = (
                int(params["end"])
            )

        if "max_noncluster_tracking" not in st.session_state:
            st.session_state.max_noncluster_tracking = (
                int(params["max"])
            )

        if "east_noncluster_tracking" not in st.session_state:
            st.session_state.east_noncluster_tracking = (
                int(params["east"])
            )

        if "west_noncluster_tracking" not in st.session_state:
            st.session_state.west_noncluster_tracking = (
                int(params["west"])
            )

        best_loss = st.number_input(
            "Efficiency Loss (%)",
            min_value=0.0,
            step=0.1,
            key="loss_noncluster_tracking"
        )

        col1, col2, col3 = st.columns(3)

        DHI = col1.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            step=1,
            key="dhi_noncluster_tracking"
        )

        GHI_Starting_Block = col2.number_input(
            "Starting Block",
            min_value=0,
            max_value=96,
            step=1,
            key="start_noncluster_tracking"
        )

        GHI_Ending_Block = col3.number_input(
            "Ending Block",
            min_value=1,
            max_value=96,
            step=1,
            key="end_noncluster_tracking"
        )

        col1, col2, col3 = st.columns(3)

        GHI_Max_Block = col1.number_input(
            "Max Block",
            min_value=0,
            max_value=96,
            step=1,
            key="max_noncluster_tracking"
        )

        Tracking_angle_lim_E = col2.number_input(
            "East Limit",
            min_value=0,
            max_value=90,
            step=1,
            key="east_noncluster_tracking"
        )

        Tracking_angle_lim_W = col3.number_input(
            "West Limit",
            min_value=0,
            max_value=90,
            step=1,
            key="west_noncluster_tracking"
        )

        # ----------------------------------------------------
        # Recalculate efficiency with user's edited loss
        # ----------------------------------------------------

        df_final = apply_efficiency_loss(
            df,
            best_loss
        )

        eff_area = (
            df_final["Eff Area"]
            .sum()
        )

        # ----------------------------------------------------
        # FAST FINAL TRACKING CALCULATION
        # ----------------------------------------------------

        forecast = calculate_tracking_forecast(
            ghi_arrays=[
                ghi
            ],
            blocks=blocks,
            weights=np.array(
                [eff_area],
                dtype=float
            ),
            DHI=DHI,
            start_block=GHI_Starting_Block,
            end_block=GHI_Ending_Block,
            max_block=GHI_Max_Block,
            east_limit=Tracking_angle_lim_E,
            west_limit=Tracking_angle_lim_W
        )

        if forecast is None:

            st.error(
                "Invalid tracking parameters. "
                "Please make sure Starting < Max < Ending."
            )

            return

        show_efficiency_table(
            df_final
        )

        plot_forecast_vs_actual(
            forecast,
            actual
        )


# ============================================================
# CLUSTER PROCESSING
# ============================================================

def process_cluster(
    file_bytes,
    edited_df,
    plant_type,
    sheet_names
):

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    df = read_excel_cached(
        file_bytes,
        "Area & Efficiency",
        header=[1],
        usecols=range(8)
    )

    df = clean_columns(
        df
    )

    df = trim_at_first_null(
        df,
        "Module Type"
    )

    # --------------------------------------------------------
    # Weight data
    # --------------------------------------------------------

    df_w = read_excel_cached(
        file_bytes,
        "Area & Efficiency",
        header=2,
        usecols=[
            12,
            13,
            14,
            15,
            16
        ]
    )

    df_w = clean_columns(
        df_w
    )

    # --------------------------------------------------------
    # Forecast Config
    # --------------------------------------------------------

    df_st = read_excel_cached(
        file_bytes,
        "Forecast Config",
        header=[8]
    )

    df_st = clean_columns(
        df_st
    )

    lat = float(
        df_st.loc[
            0,
            "Lat"
        ]
    )

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    df_tilt = read_excel_cached(
        file_bytes,
        "Config Tilt Angle",
        header=[7]
    )

    df_tilt = clean_columns(
        df_tilt
    )

    df_tilt = trim_at_first_null(
        df_tilt,
        "Fixed"
    )

    df_tilt = df_tilt.dropna(
        how="all",
        axis=1
    )

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month"
        }
    )

    month_lookup = (
        df_tilt
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    current_month = (
        pd.Timestamp.today()
        .strftime("%B")
    )

    tilt = month_lookup.get(
        current_month,
        0
    )

    # ========================================================
    # CLUSTER FIXED
    # ========================================================

    if plant_type == "🏗️ Fixed":

        df_fix = read_excel_cached(
            file_bytes,
            "Fixed-CL1",
            header=[1]
        )

        df_fix = clean_columns(
            df_fix
        )

        # ----------------------------------------------------
        # Replace GHI and Actual with edited values
        # ----------------------------------------------------

        for column in [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI"
        ]:

            if column in edited_df.columns:

                df_fix[column] = (
                    edited_df[column]
                    .values
                )

        df_fix["Actual"] = (
            edited_df["Actual"]
            .values
        )

        df_fix = trim_at_first_null(
            df_fix,
            "Date"
        )

        df_fix = (
            df_fix
            .iloc[:96]
            .copy()
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Solar geometry
        # ----------------------------------------------------

        df_fix = add_solar_geometry(
            df_fix,
            lat
        )

        # ----------------------------------------------------
        # Build POA for every cluster
        # ----------------------------------------------------

        poa_arrays = []

        for i, ghi_column in enumerate(
            [
                "CL1-GHI",
                "CL2-GHI",
                "CL3-GHI",
                "CL4-GHI",
                "CL5-GHI"
            ],
            start=1
        ):

            poa_df = calculate_fixed_poa(
                df_fix,
                ghi_column,
                tilt
            )

            poa_arrays.append(
                poa_df["POA fixed"]
                .to_numpy(
                    dtype=float
                )
            )

        # ----------------------------------------------------
        # Base forecast
        # ----------------------------------------------------

        standard_eff = (
            df["Standard PV Efficiency (%)"]
            .to_numpy(
                dtype=float
            )
        )

        area = (
            df["Total area(m2)"]
            .to_numpy(
                dtype=float
            )
        )

        base_eff_area = (
            area
            * standard_eff
            / 100
        )

        base_forecast = np.zeros(
            len(df_fix)
        )

        for i in range(5):

            cluster_weight = float(
                df_w[
                    f"CL-{i+1}"
                ].iloc[0]
            )

            total_weight = (
                base_eff_area
                * cluster_weight
            ).sum()

            base_forecast += (
                poa_arrays[i]
                * total_weight
                / 1_000_000
            )

        actual = (
            df_fix["Actual"]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Efficiency loss
        # ----------------------------------------------------

        cache_key = (
            "cluster_fixed_loss",
            st.session_state.file_signature
        )

        if (
            "loss_result"
            not in st.session_state
            or
            st.session_state.get(
                "loss_result_key"
            ) != cache_key
        ):

            with st.spinner(
                "Efficiency Loss calculate ho raha hai... ⏳"
            ):

                best_loss = (
                    optimize_efficiency_loss(
                        df=df,
                        base_forecast=base_forecast,
                        actual=actual
                    )
                )

            st.session_state.loss_result = (
                best_loss
            )

            st.session_state.loss_result_key = (
                cache_key
            )

        else:

            best_loss = (
                st.session_state.loss_result
            )

        # ----------------------------------------------------
        # Loss input
        # ----------------------------------------------------

        if "loss_cluster_fixed" not in st.session_state:

            st.session_state.loss_cluster_fixed = (
                float(best_loss)
            )

        best_loss = st.number_input(
            "Efficiency Loss (%)",
            min_value=0.0,
            step=0.1,
            key="loss_cluster_fixed"
        )

        # ----------------------------------------------------
        # Final efficiency
        # ----------------------------------------------------

        df_final = apply_efficiency_loss(
            df,
            best_loss
        )

        # ----------------------------------------------------
        # Final cluster forecast
        # ----------------------------------------------------

        final_weights = calculate_cluster_weights(
            df_final,
            df_w
        )

        forecast = np.zeros(
            len(df_fix)
        )

        for i in range(5):

            weight_column = (
                f"CL-{i+1}"
            )

            forecast += (
                poa_arrays[i]
                * final_weights[
                    weight_column
                ].sum()
                / 1_000_000
            )

        st.metric(
            "Efficiency Loss",
            f"{best_loss:.2f}%"
        )

        show_efficiency_table(
            df_final
        )

        plot_forecast_vs_actual(
            forecast,
            actual
        )

    # ========================================================
    # CLUSTER TRACKING
    # ========================================================

    else:

        # ----------------------------------------------------
        # Backend sheets
        # ----------------------------------------------------

        backend_list = []

        for i in range(1, 6):

            sheet_name = (
                f"Backend Cal CL{i}"
            )

            if sheet_name not in sheet_names:

                st.error(
                    f"{sheet_name} sheet not found."
                )

                return

            backend_df = read_excel_cached(
                file_bytes,
                sheet_name
            )

            backend_df = clean_columns(
                backend_df
            )

            backend_list.append(
                backend_df
            )

        # ----------------------------------------------------
        # Tracking sheet
        # ----------------------------------------------------

        df_trac = read_excel_cached(
            file_bytes,
            "Tracking",
            header=[1]
        )

        df_trac = clean_columns(
            df_trac
        )

        # ----------------------------------------------------
        # Solar geometry
        # ----------------------------------------------------

        df_fix = read_excel_cached(
            file_bytes,
            "Fixed-CL1",
            header=[1]
        )

        df_fix = clean_columns(
            df_fix
        )

        for column in [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI"
        ]:

            df_fix[column] = (
                edited_df[column]
                .values
            )

        df_fix["Actual"] = (
            edited_df["Actual"]
            .values
        )

        df_fix = trim_at_first_null(
            df_fix,
            "Date"
        )

        df_fix = (
            df_fix
            .iloc[:96]
            .copy()
            .reset_index(drop=True)
        )

        df_fix = add_solar_geometry(
            df_fix,
            lat
        )

        # ----------------------------------------------------
        # Tracking uses zero tilt
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Base efficiency area
        # ----------------------------------------------------

        standard_eff = (
            df["Standard PV Efficiency (%)"]
            .to_numpy(
                dtype=float
            )
        )

        area = (
            df["Total area(m2)"]
            .to_numpy(
                dtype=float
            )
        )

        base_eff_area = (
            area
            * standard_eff
            / 100
        )

        # ----------------------------------------------------
        # Initial efficiency forecast
        # ----------------------------------------------------

        ghi_arrays = [
            df_fix[
                column
            ].to_numpy(
                dtype=float
            )

            for column in [
                "CL1-GHI",
                "CL2-GHI",
                "CL3-GHI",
                "CL4-GHI",
                "CL5-GHI"
            ]
        ]

        # For efficiency optimization we use a
        # simple irradiance proxy.
        #
        # The actual tracking parameters are optimized
        # separately.

        base_forecast = np.zeros(
            len(df_fix)
        )

        for i in range(5):

            cluster_weight = float(
                df_w[
                    f"CL-{i+1}"
                ].iloc[0]
            )

            total_weight = (
                base_eff_area
                * cluster_weight
            ).sum()

            base_forecast += (
                ghi_arrays[i]
                * total_weight
                / 1_000_000
            )

        actual = (
            df_fix["Actual"]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Efficiency loss
        # ----------------------------------------------------

        cache_key = (
            "cluster_tracking_loss",
            st.session_state.file_signature
        )

        if (
            "loss_result"
            not in st.session_state
            or
            st.session_state.get(
                "loss_result_key"
            ) != cache_key
        ):

            with st.spinner(
                "Efficiency Loss calculate ho raha hai... ⏳"
            ):

                best_loss = (
                    optimize_efficiency_loss(
                        df=df,
                        base_forecast=base_forecast,
                        actual=actual
                    )
                )

            st.session_state.loss_result = (
                best_loss
            )

            st.session_state.loss_result_key = (
                cache_key
            )

        else:

            best_loss = (
                st.session_state.loss_result
            )

        # ----------------------------------------------------
        # Apply efficiency
        # ----------------------------------------------------

        if "loss_cluster_tracking" not in st.session_state:

            st.session_state.loss_cluster_tracking = (
                float(best_loss)
            )

        best_loss = st.number_input(
            "Efficiency Loss (%)",
            min_value=0.0,
            step=0.1,
            key="loss_cluster_tracking"
        )

        df_final = apply_efficiency_loss(
            df,
            best_loss
        )

        cluster_weights = calculate_cluster_weights(
            df_final,
            df_w
        )

        weight_array = np.array(
            [
                cluster_weights[
                    "CL-1"
                ].sum(),

                cluster_weights[
                    "CL-2"
                ].sum(),

                cluster_weights[
                    "CL-3"
                ].sum(),

                cluster_weights[
                    "CL-4"
                ].sum(),

                cluster_weights[
                    "CL-5"
                ].sum()
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # Blocks
        # ----------------------------------------------------

        blocks = (
            backend_list[0]["Block No."]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Optimize tracking
        # ----------------------------------------------------

        optimization_key = (
            "tracking_cluster",
            st.session_state.file_signature
        )

        if (
            "tracking_params"
            not in st.session_state
            or
            st.session_state.get(
                "tracking_params_key"
            ) != optimization_key
        ):

            optimization_result = (
                run_tracking_optimizer(
                    actual=actual,
                    ghi_arrays=ghi_arrays,
                    blocks=blocks,
                    weights=weight_array
                )
            )

            if optimization_result is None:

                st.error(
                    "Tracking optimization failed."
                )

                return

            st.session_state.tracking_params = (
                optimization_result
            )

            st.session_state.tracking_params_key = (
                optimization_key
            )

        params = (
            st.session_state.tracking_params
        )

        # ----------------------------------------------------
        # Parameter state
        # ----------------------------------------------------

        parameter_defaults = {

            "dhi_cluster_tracking":
                int(params["DHI"]),

            "start_cluster_tracking":
                int(params["start"]),

            "end_cluster_tracking":
                int(params["end"]),

            "max_cluster_tracking":
                int(params["max"]),

            "east_cluster_tracking":
                int(params["east"]),

            "west_cluster_tracking":
                int(params["west"])
        }

        for key, value in (
            parameter_defaults.items()
        ):

            if key not in st.session_state:

                st.session_state[
                    key
                ] = value

        st.subheader(
            "Optimized Parameters"
        )

        col1, col2, col3 = st.columns(3)

        DHI = col1.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            step=1,
            key="dhi_cluster_tracking"
        )

        GHI_Starting_Block = col2.number_input(
            "Starting Block",
            min_value=0,
            max_value=96,
            step=1,
            key="start_cluster_tracking"
        )

        GHI_Ending_Block = col3.number_input(
            "Ending Block",
            min_value=1,
            max_value=96,
            step=1,
            key="end_cluster_tracking"
        )

        col1, col2, col3 = st.columns(3)

        GHI_Max_Block = col1.number_input(
            "Max Block",
            min_value=0,
            max_value=96,
            step=1,
            key="max_cluster_tracking"
        )

        Tracking_angle_lim_E = col2.number_input(
            "East Limit",
            min_value=0,
            max_value=90,
            step=1,
            key="east_cluster_tracking"
        )

        Tracking_angle_lim_W = col3.number_input(
            "West Limit",
            min_value=0,
            max_value=90,
            step=1,
            key="west_cluster_tracking"
        )

        # ----------------------------------------------------
        # Final cluster tracking forecast
        # ----------------------------------------------------

        forecast = calculate_tracking_forecast(
            ghi_arrays=ghi_arrays,
            blocks=blocks,
            weights=weight_array,
            DHI=DHI,
            start_block=GHI_Starting_Block,
            end_block=GHI_Ending_Block,
            max_block=GHI_Max_Block,
            east_limit=Tracking_angle_lim_E,
            west_limit=Tracking_angle_lim_W
        )

        if forecast is None:

            st.error(
                "Invalid tracking parameters. "
                "Please make sure Starting < Max < Ending."
            )

            return

        show_efficiency_table(
            df_final
        )

        plot_forecast_vs_actual(
            forecast,
            actual
        )
