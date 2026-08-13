# ============================================================
# LOSS CORRECTION MODEL PAGE
# Complete Streamlit Page
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
    layout="wide"
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
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "tracking_params": None,
    "model_context": None,
    "input_data": None,
    "model_result": None,
    "optimization_running": False,
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

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .result-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #f9fafb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">☀️ Loss Correction Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Upload your solar plant Excel file, select the plant configuration,
    edit forecast data if required, and calculate the corrected forecast.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📁 1. Input Data</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"],
    help="Upload the Excel workbook containing Area & Efficiency, Forecast Config and forecast data."
)


if uploaded_file is None:

    st.info(
        "👆 Please upload the Excel file to start the Loss Correction Model."
    )

    st.stop()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ 2. Model Configuration</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    plant_type = st.radio(
        "Plant Type",
        [
            "🏗️ Fixed",
            "🔄 Tracking"
        ],
        horizontal=True
    )


with col2:

    model_type = st.radio(
        "Plant Configuration",
        [
            "Non-Cluster",
            "Cluster"
        ],
        horizontal=True
    )


is_cluster = (
    model_type == "Cluster"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def read_area_efficiency(file_bytes, cluster=False):

    from io import BytesIO

    file = BytesIO(file_bytes)

    if cluster:

        df = pd.read_excel(
            file,
            sheet_name="Area & Efficiency",
            header=1,
            usecols=range(8)
        )

    else:

        df = pd.read_excel(
            file,
            sheet_name="Area & Efficiency",
            header=1
        )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

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


@st.cache_data(show_spinner=False)
def read_cluster_weights(file_bytes):

    from io import BytesIO

    file = BytesIO(file_bytes)

    df_w = pd.read_excel(
        file,
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


@st.cache_data(show_spinner=False)
def read_latitude(file_bytes):

    from io import BytesIO

    file = BytesIO(file_bytes)

    df_st = pd.read_excel(
        file,
        sheet_name="Forecast Config",
        header=8
    )

    return float(
        df_st.loc[0, "Lat"]
    )


@st.cache_data(show_spinner=False)
def read_tilt_lookup(file_bytes):

    from io import BytesIO

    file = BytesIO(file_bytes)

    df_tilt = pd.read_excel(
        file,
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
    tilt_lookup,
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

    standard_eff = (
        df[
            "Standard PV Efficiency (%)"
        ]
        .to_numpy(dtype=float)
    )

    area = (
        df[
            "Total area(m2)"
        ]
        .to_numpy(dtype=float)
    )

    max_loss = float(
        np.nanmin(
            standard_eff
        )
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

    return float(best_loss)


def apply_efficiency_loss(
    df,
    best_loss
):

    df = df.copy()

    df["Efficiency Losses(%)"] = (
        best_loss
    )

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
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    blocks,
    weighted_ghi,
    actual,
    maxiter=40,
    popsize=10
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

    def objective(x):

        DHI = int(x[0])
        start = int(x[1])
        end = int(x[2])
        max_block = int(x[3])
        east = int(x[4])
        west = int(x[5])

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

        m1 = (
            90 / denominator_1
        )

        m2 = (
            90 / denominator_2
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

        panel = np.where(

            blocks_day < max_block,

            np.minimum(
                zenith,
                abs(east)
            ),

            np.where(
                (
                    blocks_day > max_block
                )
                & (
                    zenith > west
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

        prediction = (
            weighted_ghi_day
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

        return float(
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

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
            True
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

    m1 = (
        90
        / (
            start
            - 1
            - max_block
        )
    )

    m2 = (
        90
        / (
            end
            + 1
            - max_block
        )
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
                blocks > max_block
            )
            & (
                zenith > west
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

    forecast = (
        weighted_ghi
        * (
            1 - DHI / 100
        )
        / cos_alpha
        / 1_000_000
    )

    return forecast


# ============================================================
# EFFICIENCY TABLE
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
        c for c in columns
        if c in df.columns
    ]

    display_df = (
        df[columns]
        .copy()
    )

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
        "🔍 View Efficiency Calculations"
    ):

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FORECAST CHART
# ============================================================

def show_forecast_chart(
    forecast,
    actual,
    title
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
# LOAD EXCEL
# ============================================================

file_bytes = uploaded_file.getvalue()

try:

    df = read_area_efficiency(
        file_bytes,
        cluster=is_cluster
    )

    lat = read_latitude(
        file_bytes
    )

    tilt_lookup = read_tilt_lookup(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to read the Excel configuration: {e}"
    )

    st.stop()


# ============================================================
# INPUT DATA SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📊 3. Forecast Input Data</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Determine Forecast Sheet
# ------------------------------------------------------------

if is_cluster:

    forecast_sheet = "Fixed-CL1"

else:

    forecast_sheet = "Fixed"


try:

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name=forecast_sheet,
        header=1
    )

except Exception as e:

    st.error(
        f"Unable to read '{forecast_sheet}' sheet: {e}"
    )

    st.stop()


df_fix.columns = (
    df_fix.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# CLEAN DATA
# ============================================================

if "Date" in df_fix.columns:

    df_fix = df_fix[
        df_fix["Date"].notna()
    ].copy()


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

required_columns = [
    "GHI_Forecast",
    "Actual"
]

missing_columns = [
    c for c in required_columns
    if c not in df_fix.columns
]


if missing_columns:

    st.error(
        "The following required columns are missing: "
        + ", ".join(missing_columns)
    )

    st.stop()


df_fix["GHI_Forecast"] = pd.to_numeric(
    df_fix["GHI_Forecast"],
    errors="coerce"
).fillna(0)

df_fix["Actual"] = pd.to_numeric(
    df_fix["Actual"],
    errors="coerce"
).fillna(0)


# ============================================================
# EDITABLE INPUT TABLE
# ============================================================

st.info(
    "💡 You can directly edit GHI Forecast and Actual values below. "
    "The model uses these edited values when calculating the result."
)


input_columns = [
    "GHI_Forecast",
    "Actual"
]


if "Date" in df_fix.columns:

    input_columns = [
        "Date"
    ] + input_columns


input_df = df_fix[
    input_columns
].copy()


edited_df = st.data_editor(
    input_df,
    use_container_width=True,
    hide_index=True,

    num_rows="fixed",

    column_config={

        "GHI_Forecast": st.column_config.NumberColumn(
            "GHI Forecast",
            help="Forecast GHI value",
            format="%.2f"
        ),

        "Actual": st.column_config.NumberColumn(
            "Actual Power",
            help="Actual power value",
            format="%.4f"
        ),

    },

    disabled=[
        c for c in input_df.columns
        if c == "Date"
    ],

    key="loss_correction_input_data"
)


# ============================================================
# USE EDITED DATA
# ============================================================

df_fix["GHI_Forecast"] = (
    edited_df["GHI_Forecast"]
    .to_numpy()
)

df_fix["Actual"] = (
    edited_df["Actual"]
    .to_numpy()
)


# ============================================================
# START MODEL
# ============================================================

st.markdown(
    '<div class="section-title">🚀 4. Run Loss Correction</div>',
    unsafe_allow_html=True
)


run_model = st.button(
    "🚀 Run Loss Correction",
    type="primary",
    use_container_width=True
)


if run_model:

    # Reset tracking optimization
    st.session_state.tracking_params = None

    # Reset model result
    st.session_state.model_result = None

    st.session_state.model_context = (
        uploaded_file.name,
        uploaded_file.size,
        plant_type,
        is_cluster
    )


# ============================================================
# MODEL EXECUTION
# ============================================================

if not run_model and st.session_state.model_result is None:

    st.info(
        "Edit the input data if required, then click "
        "**Run Loss Correction**."
    )

    st.stop()


# ============================================================
# SOLAR ANGLES
# ============================================================

df_fix = prepare_solar_angles(
    df_fix,
    lat,
    tilt_lookup,
    tracking=(
        plant_type == "🔄 Tracking"
    )
)


# ============================================================
# CLUSTER DATA
# ============================================================

if is_cluster:

    cluster_weights = (
        read_cluster_weights(
            file_bytes
        )
    )

    df_ghi = pd.read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5]
    )

    df_ghi = df_ghi.fillna(0)

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

    if plant_type == "🏗️ Fixed":

        for i, col in enumerate(
            ghi_columns
        ):

            if col not in df_fix.columns:

                df_fix[col] = (
                    df_ghi.iloc[:, i]
                    .to_numpy()
                )


# ============================================================
# EFFICIENCY LOSS
# ============================================================

if plant_type == "🏗️ Fixed":

    if is_cluster:

        df_fix["POA fixed"] = (
            df_fix["CL1-GHI"]
            * df_fix["SIN(a+b)"]
            / df_fix["Sin(a)"]
        )

    else:

        df_fix["POA fixed"] = (
            df_fix["GHI_Forecast"]
            * df_fix["SIN(a+b)"]
            / df_fix["Sin(a)"]
        )


else:

    df_fix["POA fixed"] = (
        df_fix["GHI_Forecast"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )


best_loss = calculate_efficiency_loss(
    df,
    df_fix["POA fixed"],
    df_fix["Actual"]
)


df = apply_efficiency_loss(
    df,
    best_loss
)


# ============================================================
# FIXED MODEL
# ============================================================

if plant_type == "🏗️ Fixed":

    st.markdown(
        '<div class="section-title">📈 Fixed Plant Results</div>',
        unsafe_allow_html=True
    )

    if is_cluster:

        weights = {}

        for weight_col in [
            "CL-1",
            "CL-2",
            "CL-3",
            "CL-4",
            "CL-5"
        ]:

            weights[weight_col] = (
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * cluster_weights[
                    weight_col
                ]
            ).sum()

        forecast = np.zeros(
            len(df_fix),
            dtype=float
        )

        for ghi_col, weight_col in zip(
            ghi_columns,
            [
                "CL-1",
                "CL-2",
                "CL-3",
                "CL-4",
                "CL-5"
            ]
        ):

            poa = (
                df_fix[ghi_col]
                * df_fix["SIN(a+b)"]
                / df_fix["Sin(a)"]
            )

            forecast += (
                poa.to_numpy()
                * weights[weight_col]
                / 1_000_000
            )

        chart_title = (
            "Fixed Cluster Forecast vs Actual"
        )

    else:

        forecast = (
            df_fix["POA fixed"]
            .to_numpy()
            * df["Eff Area"].sum()
            / 1_000_000
        )

        chart_title = (
            "Fixed Forecast vs Actual"
        )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Efficiency Loss",
        f"{best_loss:.2f}%"
    )

    col2.metric(
        "Forecast Peak",
        f"{np.max(forecast):.3f} MW"
    )

    col3.metric(
        "Actual Peak",
        f"{np.max(df_fix['Actual']):.3f} MW"
    )


    show_efficiency_table(
        df
    )

    show_forecast_chart(
        forecast,
        df_fix["Actual"].to_numpy(),
        chart_title
    )


# ============================================================
# TRACKING MODEL
# ============================================================

else:

    st.markdown(
        '<div class="section-title">🔄 Tracking Model</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # BACKEND CAL
    # ========================================================

    if is_cluster:

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
            backend_list[0]["Block No."]
            .to_numpy(dtype=float)
        )

    else:

        df_bcal = pd.read_excel(
            uploaded_file,
            sheet_name="Backend Cal"
        )

        blocks = (
            df_bcal["Block No."]
            .to_numpy(dtype=float)
        )


    # ========================================================
    # WEIGHTED GHI
    # ========================================================

    if is_cluster:

        weighted_ghi = np.zeros(
            len(df_fix),
            dtype=float
        )

        for ghi_col, weight_key in zip(
            ghi_columns,
            [
                "CL-1",
                "CL-2",
                "CL-3",
                "CL-4",
                "CL-5"
            ]
        ):

            eff_area = (
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * cluster_weights[
                    weight_key
                ]
            ).sum()

            weighted_ghi += (
                df_fix[ghi_col]
                .to_numpy(dtype=float)
                * eff_area
            )

    else:

        weighted_ghi = (
            df_fix["GHI_Forecast"]
            .to_numpy(dtype=float)
            * df["Eff Area"].sum()
        )


    # ========================================================
    # OPTIMIZATION
    # ========================================================

    if st.session_state.tracking_params is None:

        st.subheader(
            "⚙️ Tracking Optimization"
        )

        progress_bar = st.progress(
            0
        )

        status_box = st.empty()

        with st.spinner(
            "Optimizing tracking parameters..."
        ):

            result = optimize_tracking(
                blocks,
                weighted_ghi,
                df_fix["Actual"]
                .to_numpy(dtype=float),
                MAX_OPT_ITER,
                OPT_POPSIZE
            )

        progress_bar.progress(
            1.0
        )

        status_box.success(
            "✅ Optimization completed!"
        )

        time.sleep(0.5)

        progress_bar.empty()
        status_box.empty()

        st.session_state.tracking_params = (
            result
        )


    params = (
        st.session_state.tracking_params
    )


    # ========================================================
    # OPTIMIZED PARAMETERS
    # ========================================================

    st.subheader(
        "🎯 Optimized Parameters"
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        DHI = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            value=int(params["DHI"]),
            step=1
        )

    with col2:

        start = st.number_input(
            "Starting Block",
            min_value=0,
            max_value=30,
            value=int(params["start"]),
            step=1
        )

    with col3:

        end = st.number_input(
            "Ending Block",
            min_value=65,
            max_value=80,
            value=int(params["end"]),
            step=1
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        max_block = st.number_input(
            "Max Block",
            min_value=44,
            max_value=60,
            value=int(params["max"]),
            step=1
        )

    with col2:

        east = st.number_input(
            "East Tracking Limit",
            min_value=0,
            max_value=70,
            value=int(params["east"]),
            step=1
        )

    with col3:

        west = st.number_input(
            "West Tracking Limit",
            min_value=0,
            max_value=70,
            value=int(params["west"]),
            step=1
        )


    # ========================================================
    # EFFICIENCY LOSS INPUT
    # ========================================================

    best_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(best_loss),
        step=0.1,
        format="%.2f"
    )


    # ========================================================
    # APPLY LOSS
    # ========================================================

    df = apply_efficiency_loss(
        df,
        best_loss
    )


    show_efficiency_table(
        df
    )


    # ========================================================
    # FINAL TRACKING FORECAST
    # ========================================================

    final_params = {

        "DHI": int(DHI),

        "start": int(start),

        "end": int(end),

        "max": int(max_block),

        "east": int(east),

        "west": int(west)
    }


    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            final_params
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Efficiency Loss",
            f"{best_loss:.2f}%"
        )

        col2.metric(
            "Forecast Peak",
            f"{np.max(forecast):.3f} MW"
        )

        col3.metric(
            "Actual Peak",
            f"{np.max(df_fix['Actual']):.3f} MW"
        )

        col4.metric(
            "Optimization Score",
            f"{params['score']:.4f}"
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
