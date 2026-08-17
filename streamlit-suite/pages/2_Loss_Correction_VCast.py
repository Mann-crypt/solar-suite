# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# JUPYTER-MATCHED CALCULATION
# ============================================================

import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Forecast Correction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SIMPLE CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1 {
        font-size: 2rem !important;
        margin-bottom: 0.2rem !important;
    }

    h2 {
        font-size: 1.25rem !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border-radius: 10px;
        padding: 10px;
    }

    .result-box {
        padding: 14px 18px;
        border-radius: 10px;
        background: rgba(128,128,128,0.08);
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

TRACKING_BOUNDS = [
    (0, 10),      # DHI
    (10, 30),     # GHI Starting Block
    (65, 80),     # GHI Ending Block
    (47, 53),     # GHI Max Block
    (10, 70),     # Tracking East Limit
    (10, 70),     # Tracking West Limit
]


# ============================================================
# HELPERS
# ============================================================

def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )
    return df


def trim_at_blank(df, column):
    df = df.copy()

    if column not in df.columns:
        return df.reset_index(drop=True)

    null_rows = df[df[column].isna()].index

    if len(null_rows):
        first_pos = df.index.get_loc(null_rows[0])
        df = df.iloc[:first_pos]

    return df.reset_index(drop=True)


def numeric_series(df, column):
    if column not in df.columns:
        return pd.Series(
            np.zeros(len(df)),
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(0.0)


def file_hash(uploaded_file):
    data = uploaded_file.getvalue()
    return hashlib.md5(data).hexdigest()


# ============================================================
# READ EXCEL
# ============================================================

@st.cache_data(show_spinner=False)
def load_workbook(file_bytes):

    excel = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    df_area = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df_area = clean_columns(df_area)
    df_area = trim_at_blank(df_area, "S.No.")

    # --------------------------------------------------------
    # Cluster mapping
    # --------------------------------------------------------

    df_w = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df_w = clean_columns(df_w)
    df_w = trim_at_blank(df_w, "Clusters")

    # --------------------------------------------------------
    # Forecast Config
    # --------------------------------------------------------

    df_config = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )

    lat = float(
        pd.to_numeric(
            df_config.loc[0, "Lat"],
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    df_tilt = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df_tilt = clean_columns(df_tilt)

    if "Fixed" in df_tilt.columns:
        df_tilt = trim_at_blank(
            df_tilt,
            "Fixed",
        )

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

    month_lookup = {}

    if {
        "Month",
        "Fixed",
    }.issubset(df_tilt.columns):

        month_lookup = (
            df_tilt
            .dropna(subset=["Month"])
            .set_index("Month")["Fixed"]
            .to_dict()
        )

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    df_ghi = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df_ghi = df_ghi.fillna(0)

    # --------------------------------------------------------
    # Fixed data
    # --------------------------------------------------------

    df_fix = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Fixed-C11",
        header=1,
    )

    df_fix = clean_columns(df_fix)
    df_fix = trim_at_blank(
        df_fix,
        "Date",
    )

    # --------------------------------------------------------
    # Tracking backend
    # --------------------------------------------------------

    backend = []

    for cluster in ["C11", "C12", "C13", "C14", "C15"]:

        sheet = f"Backend Cal {cluster}"

        temp = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet,
        )

        temp = clean_columns(temp)

        backend.append(temp)

    # --------------------------------------------------------
    # Tracking output sheet
    # --------------------------------------------------------

    df_tracking = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Tracking",
        header=1,
    )

    df_tracking = clean_columns(
        df_tracking
    )

    return {
        "area": df_area,
        "weights": df_w,
        "config": df_config,
        "lat": lat,
        "month_lookup": month_lookup,
        "ghi": df_ghi,
        "fix": df_fix,
        "backend": backend,
        "tracking": df_tracking,
    }


# ============================================================
# CALCULATE EFFECTIVE AREAS
#
# IMPORTANT:
# Error % is applied HERE ONCE.
# Tracking must use the resulting effective areas directly.
# ============================================================

def calculate_effective_areas(
    df_area,
    df_w,
    error_percent,
):

    df = df_area.copy()
    weights = df_w.copy()

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    standard_eff = numeric_series(
        df,
        "Standard PV Efficiency (%)",
    )

    modules = numeric_series(
        df,
        "No of Module",
    )

    module_area = numeric_series(
        df,
        "Area of 1 Module (m2)",
    )

    # --------------------------------------------------------
    # Total physical area
    # --------------------------------------------------------

    df["Total area (m2)"] = (
        modules * module_area
    )

    # --------------------------------------------------------
    # APPLY ERROR %
    #
    # THIS IS THE ONLY PLACE WHERE ERROR % IS APPLIED.
    # --------------------------------------------------------

    df["Error %"] = float(
        error_percent
    )

    df["Net Efficiency (%)"] = (
        standard_eff
        - float(error_percent)
    )

    # --------------------------------------------------------
    # Effective area
    # --------------------------------------------------------

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100.0
    )

    # --------------------------------------------------------
    # Cluster effective area
    # --------------------------------------------------------

    cluster_sums = (
        df.groupby(
            "Clusters"
        )["Eff Area"]
        .sum()
    )

    weights["Eff Area(m2)"] = (
        weights["Clusters"]
        .map(cluster_sums)
        .fillna(0.0)
    )

    return df, weights


# ============================================================
# PREPARE FIXED SOLAR GEOMETRY
# ============================================================

def prepare_fixed_geometry(
    df_fix,
    df_ghi,
    lat,
    month_lookup,
):

    result = df_fix.copy()

    n = min(
        len(result),
        len(df_ghi),
    )

    result = result.iloc[:n].copy()
    ghi = df_ghi.iloc[:n].copy()

    # --------------------------------------------------------
    # Preserve the original Jupyter calculation
    # --------------------------------------------------------

    result["Date"] = pd.Timestamp.today()

    first_date = (
        pd.Timestamp.today()
        .replace(
            month=1,
            day=1,
        )
        .normalize()
    )

    result["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + (
                        result["Date"]
                        - first_date
                    ).dt.days
                    + 1
                )
                / 365
            )
        )
    )

    result["Elevation angle a"] = (
        90
        - lat
        + result[
            "Declination Angle ∆"
        ]
    )

    result["Tilt Angle b"] = (
        result["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

    result["Tilt Angle b"] = (
        pd.to_numeric(
            result["Tilt Angle b"],
            errors="coerce",
        ).fillna(0)
    )

    result["a+b"] = (
        result["Elevation angle a"]
        + result["Tilt Angle b"]
    )

    result["SIN(a+b)"] = np.sin(
        np.radians(
            result["a+b"]
        )
    )

    result["Sin(a)"] = np.sin(
        np.radians(
            result["Elevation angle a"]
        )
    )

    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    sin_a = result["Sin(a)"].replace(
        0,
        np.nan,
    )

    for i, col in enumerate(GHI_COLS):

        ghi_values = pd.to_numeric(
            ghi[col],
            errors="coerce",
        ).fillna(0).to_numpy()

        if i == 0:

            result["GHI*sin(a)"] = (
                ghi_values
                * result["Sin(a)"]
            )

            result["GHI*sin(a+b)"] = (
                ghi_values
                * result["SIN(a+b)"]
            )

            result["POA fixed"] = (
                result["GHI*sin(a+b)"]
                / sin_a
            )

        else:

            cluster = i + 1

            result[
                f"GHI*sin(a)-CL{cluster}"
            ] = (
                ghi_values
                * result["Sin(a)"]
            )

            result[
                f"GHI*sin(a+b)-CL{cluster}"
            ] = (
                ghi_values
                * result["SIN(a+b)"]
            )

            result[
                f"POA Fixed-C{cluster}"
            ] = (
                result[
                    f"GHI*sin(a+b)-CL{cluster}"
                ]
                / sin_a
            )

    return result


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df_fix,
    df_weights,
):

    result = df_fix.copy()

    poa_cols = [
        "POA fixed",
        "POA Fixed-C12",
        "POA Fixed-C13",
        "POA Fixed-C14",
        "POA Fixed-C15",
    ]

    power_cols = []

    for i in range(5):

        col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        result[col] = (
            result[poa_cols[i]]
            * float(
                df_weights.iloc[i][
                    "Eff Area(m2)"
                ]
            )
            / 1_000_000
        )

        power_cols.append(col)

    result[
        "Total Power (CL1+CL2+…)"
    ] = result[power_cols].sum(
        axis=1
    )

    return result


# ============================================================
# FIXED OPTIMIZATION
# ============================================================

def optimize_fixed_error(
    df_area,
    df_weights,
    df_geometry,
    actual,
):

    actual = pd.to_numeric(
        actual,
        errors="coerce",
    ).fillna(0).to_numpy()

    actual_peak = (
        np.max(actual)
        if len(actual)
        else 0
    )

    if actual_peak <= 0:
        return 0.0

    best_error = 0.0
    best_score = np.inf

    # --------------------------------------------------------
    # SAME RANGE AS JUPYTER
    # --------------------------------------------------------

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        _, weights = (
            calculate_effective_areas(
                df_area,
                df_weights,
                error,
            )
        )

        forecast_df = (
            calculate_fixed_forecast(
                df_geometry,
                weights,
            )
        )

        forecast = (
            forecast_df[
                "Total Power (CL1+CL2+…)"
            ]
            .to_numpy()
        )

        calculated_peak = (
            np.nanmax(forecast)
        )

        peak_error = abs(
            calculated_peak
            - actual_peak
        )

        if actual_peak:

            score = (
                peak_error
                / actual_peak
            )

        else:
            score = np.inf

        if score < best_score:

            best_score = score
            best_error = float(
                round(error, 1)
            )

    return best_error


# ============================================================
# TRACKING ANGLES
# ============================================================

def calculate_tracking_angles(
    blocks,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
):

    if not (
        start_block
        < max_block
        < end_block
    ):
        raise ValueError(
            "GHI Max Block must be between "
            "GHI Starting Block and "
            "GHI Ending Block."
        )

    # --------------------------------------------------------
    # SAME JUPYTER FORMULAS
    # --------------------------------------------------------

    m1 = 90 / (
        start_block
        - 1
        - max_block
    )

    m2 = 90 / (
        end_block
        + 1
        - max_block
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
            abs(east_limit),
        ),

        np.where(
            (
                (blocks > max_block)
                & (
                    zenith
                    > west_limit
                )
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

    return (
        zenith,
        panel,
        cos_alpha,
    )


# ============================================================
# TRACKING FORECAST
#
# IMPORTANT:
# df_weights already contains Error %.
# NO ERROR % IS APPLIED HERE.
# ============================================================

def calculate_tracking_forecast(
    df_ghi,
    df_backend,
    df_weights,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
):

    n = min(
        len(df_ghi),
        len(df_backend[0]),
    )

    ghi_matrix = np.column_stack(
        [
            pd.to_numeric(
                df_ghi.iloc[:n][col],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(
                dtype=float
            )
            for col in GHI_COLS
        ]
    )

    blocks = pd.to_numeric(
        df_backend[0]
        .iloc[:n]["Block No."],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    weights = pd.to_numeric(
        df_weights.iloc[:5][
            "Eff Area(m2)"
        ],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    zenith, panel, cos_alpha = (
        calculate_tracking_angles(
            blocks,
            DHI,
            start_block,
            end_block,
            max_block,
            east_limit,
            west_limit,
        )
    )

    # --------------------------------------------------------
    # SAME JUPYTER CALCULATION
    # --------------------------------------------------------

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    forecast = (
        dni @ weights
    ) / 1_000_000

    forecast = np.nan_to_num(
        forecast,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return (
        forecast,
        zenith,
        panel,
    )


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def tracking_objective(
    x,
    ghi_matrix,
    blocks,
    weights,
    actual,
):

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

    try:

        _, _, cos_alpha = (
            calculate_tracking_angles(
                blocks,
                DHI,
                start_block,
                end_block,
                max_block,
                east_limit,
                west_limit,
            )
        )

    except Exception:
        return 1e9

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    prediction = (
        dni @ weights
    ) / 1_000_000

    if (
        np.isnan(prediction).any()
        or np.isinf(prediction).any()
    ):
        return 1e9

    actual = np.asarray(
        actual,
        dtype=float,
    )

    mask = actual != 0

    if not mask.any():
        return 1e9

    actual_valid = actual[mask]
    prediction_valid = prediction[mask]

    actual_max = (
        actual_valid.max()
    )

    actual_sum = (
        actual_valid.sum()
    )

    if (
        actual_max <= 0
        or actual_sum <= 0
    ):
        return 1e9

    block_error = (
        np.mean(
            np.abs(
                actual_valid
                - prediction_valid
            )
        )
        / actual_max
    )

    peak_error = (
        abs(
            actual_max
            - prediction_valid.max()
        )
        / actual_max
    )

    energy_error = (
        abs(
            actual_sum
            - prediction_valid.sum()
        )
        / actual_sum
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return float(score)


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
)
def optimize_tracking_cached(
    ghi_matrix,
    blocks,
    weights,
    actual,
):

    result = differential_evolution(
        lambda x: tracking_objective(
            x,
            ghi_matrix,
            blocks,
            weights,
            actual,
        ),
        bounds=TRACKING_BOUNDS,
        strategy="best1bin",
        maxiter=40,
        popsize=15,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    return tuple(
        np.round(
            result.x
        ).astype(int)
    )


def optimize_tracking(
    df_ghi,
    backend,
    weights,
    actual,
):

    n = min(
        len(df_ghi),
        len(backend[0]),
        len(actual),
    )

    ghi_matrix = np.column_stack(
        [
            pd.to_numeric(
                df_ghi.iloc[:n][col],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(
                dtype=float
            )
            for col in GHI_COLS
        ]
    )

    blocks = pd.to_numeric(
        backend[0]
        .iloc[:n]["Block No."],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    weights_array = pd.to_numeric(
        weights.iloc[:5][
            "Eff Area(m2)"
        ],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    actual_array = pd.to_numeric(
        actual.iloc[:n],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    return optimize_tracking_cached(
        ghi_matrix,
        blocks,
        weights_array,
        actual_array,
    )


# ============================================================
# INPUT DATA PREPARATION
# ============================================================

def prepare_input_dataframe(
    df_ghi,
    df_fix,
):

    n = min(
        len(df_ghi),
        len(df_fix),
    )

    data = pd.DataFrame()

    for col in GHI_COLS:

        if col in df_ghi.columns:

            data[col] = pd.to_numeric(
                df_ghi[col].iloc[:n],
                errors="coerce",
            ).fillna(0)

        else:

            data[col] = 0.0

    if "Actual" in df_fix.columns:

        data["Actual"] = (
            pd.to_numeric(
                df_fix[
                    "Actual"
                ].iloc[:n],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        data["Actual"] = 0.0

    return data.reset_index(
        drop=True
    )


# ============================================================
# GRAPH
# ============================================================

def show_forecast_graph(
    actual,
    forecast,
    title,
):

    n = min(
        len(actual),
        len(forecast),
    )

    actual = np.asarray(
        actual[:n],
        dtype=float,
    )

    forecast = np.asarray(
        forecast[:n],
        dtype=float,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(
                width=2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                width=2,
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
        xaxis_title="15-Minute Block",
        yaxis_title="Power",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# SESSION STATE
# ============================================================

if "calculated" not in st.session_state:
    st.session_state.calculated = False

if "result_forecast" not in st.session_state:
    st.session_state.result_forecast = None

if "result_actual" not in st.session_state:
    st.session_state.result_actual = None

if "best_error" not in st.session_state:
    st.session_state.best_error = 0.0

if "tracking_params" not in st.session_state:
    st.session_state.tracking_params = (
        1,
        30,
        79,
        53,
        11,
        23,
    )


# ============================================================
# HEADER
# ============================================================

st.title("☀️ Solar Forecast Correction")

st.caption(
    "Fixed and Tracking plant forecast correction"
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"],
)


if uploaded_file is None:

    st.info(
        "Upload your Excel workbook to begin."
    )

    st.stop()


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    data = load_workbook(
        uploaded_file.getvalue()
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


# ============================================================
# INPUT DATA
# ============================================================

st.subheader("Input Data")

default_input = (
    prepare_input_dataframe(
        data["ghi"],
        data["fix"],
    )
)

edited_input = st.data_editor(
    default_input,
    use_container_width=True,
    height=300,
    num_rows="dynamic",
    hide_index=True,
    key="input_dataframe",
)


# ============================================================
# PLANT TYPE
# ============================================================

st.subheader("Plant Type")

plant_type = st.segmented_control(
    "Select Plant Type",
    options=[
        "Fixed",
        "Tracking",
    ],
    default="Fixed",
)


if plant_type is None:
    plant_type = "Fixed"


# ============================================================
# PARAMETERS
# ============================================================

st.subheader("Parameters")


if plant_type == "Fixed":

    col1, col2 = st.columns(
        [1, 3]
    )

    with col1:

        auto_fixed = st.checkbox(
            "Automatic Error %",
            value=True,
        )

    with col2:

        if auto_fixed:

            error_input = st.number_input(
                "Error %",
                min_value=0.0,
                max_value=10.0,
                value=float(
                    st.session_state.best_error
                ),
                step=0.1,
            )

        else:

            error_input = st.number_input(
                "Error %",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
            )


else:

    p1, p2, p3 = st.columns(3)
    p4, p5, p6 = st.columns(3)

    default_params = (
        st.session_state.tracking_params
    )

    with p1:

        dhi_input = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            value=int(
                default_params[0]
            ),
            step=1,
        )

    with p2:

        start_input = st.number_input(
            "GHI Starting Block",
            min_value=1,
            max_value=95,
            value=int(
                default_params[1]
            ),
            step=1,
        )

    with p3:

        end_input = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                default_params[2]
            ),
            step=1,
        )

    with p4:

        max_input = st.number_input(
            "GHI Max Block",
            min_value=1,
            max_value=96,
            value=int(
                default_params[3]
            ),
            step=1,
        )

    with p5:

        east_input = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=70,
            value=int(
                default_params[4]
            ),
            step=1,
        )

    with p6:

        west_input = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=70,
            value=int(
                default_params[5]
            ),
            step=1,
        )


# ============================================================
# CALCULATE BUTTON
# ============================================================

calculate_clicked = st.button(
    "Calculate Forecast",
    type="primary",
    use_container_width=True,
)


# ============================================================
# CALCULATION
# ============================================================

if calculate_clicked:

    try:

        # ----------------------------------------------------
        # USER INPUT
        # ----------------------------------------------------

        input_ghi = edited_input[
            GHI_COLS
        ].copy()

        input_actual = edited_input[
            "Actual"
        ].copy()

        # ----------------------------------------------------
        # UPDATE SOURCE DATA
        # ----------------------------------------------------

        data["ghi"] = input_ghi

        data["fix"] = data["fix"].iloc[
            :len(input_actual)
        ].copy()

        data["fix"]["Actual"] = (
            input_actual.to_numpy()
        )

        # ====================================================
        # FIXED
        # ====================================================

        if plant_type == "Fixed":

            if auto_fixed:

                with st.spinner(
                    "Calculating optimal Error %..."
                ):

                    geometry = (
                        prepare_fixed_geometry(
                            data["fix"],
                            data["ghi"],
                            data["lat"],
                            data[
                                "month_lookup"
                            ],
                        )
                    )

                    best_error = (
                        optimize_fixed_error(
                            data["area"],
                            data["weights"],
                            geometry,
                            input_actual,
                        )
                    )

                st.session_state.best_error = (
                    best_error
                )

                error_input = best_error

            else:

                best_error = (
                    float(error_input)
                )

            # ------------------------------------------------
            # APPLY ERROR % ONCE
            # ------------------------------------------------

            area_result, weights_result = (
                calculate_effective_areas(
                    data["area"],
                    data["weights"],
                    best_error,
                )
            )

            geometry = (
                prepare_fixed_geometry(
                    data["fix"],
                    data["ghi"],
                    data["lat"],
                    data["month_lookup"],
                )
            )

            fixed_result = (
                calculate_fixed_forecast(
                    geometry,
                    weights_result,
                )
            )

            forecast = (
                fixed_result[
                    "Total Power (CL1+CL2+…)"
                ]
                .to_numpy()
            )

            st.session_state.best_error = (
                best_error
            )

        # ====================================================
        # TRACKING
        # ====================================================

        else:

            # ------------------------------------------------
            # FIRST:
            # calculate effective area ONCE
            #
            # Default Error % = existing UI value
            # ------------------------------------------------

            tracking_error = st.number_input(
                "Error %",
                min_value=0.0,
                max_value=10.0,
                value=float(
                    st.session_state.best_error
                ),
                step=0.1,
                key="tracking_error_runtime",
            )

            area_result, weights_result = (
                calculate_effective_areas(
                    data["area"],
                    data["weights"],
                    tracking_error,
                )
            )

            actual_array = (
                input_actual.to_numpy(
                    dtype=float
                )
            )

            # ------------------------------------------------
            # AUTOMATIC OPTIMIZATION
            #
            # Tracking optimizer receives already corrected
            # effective areas.
            #
            # It NEVER applies Error % itself.
            # ------------------------------------------------

            with st.spinner(
                "Optimizing tracking parameters..."
            ):

                params = optimize_tracking(
                    data["ghi"],
                    data["backend"],
                    weights_result,
                    pd.Series(
                        actual_array
                    ),
                )

            (
                DHI,
                start_block,
                end_block,
                max_block,
                east_limit,
                west_limit,
            ) = params

            st.session_state.tracking_params = (
                DHI,
                start_block,
                end_block,
                max_block,
                east_limit,
                west_limit,
            )

            # ------------------------------------------------
            # FINAL TRACKING CALCULATION
            # ------------------------------------------------

            (
                forecast,
                _,
                _,
            ) = calculate_tracking_forecast(
                data["ghi"],
                data["backend"],
                weights_result,
                DHI,
                start_block,
                end_block,
                max_block,
                east_limit,
                west_limit,
            )

            st.session_state.best_error = (
                tracking_error
            )

        # ====================================================
        # SAVE RESULTS
        # ====================================================

        st.session_state.result_forecast = (
            forecast
        )

        st.session_state.result_actual = (
            input_actual.to_numpy(
                dtype=float
            )
        )

        st.session_state.calculated = True

        st.success(
            "Forecast calculation completed."
        )

    except Exception as e:

        st.session_state.calculated = False

        st.error(
            f"Calculation failed: {e}"
        )


# ============================================================
# RESULT
# ============================================================

if st.session_state.calculated:

    st.subheader(
        f"{plant_type} Forecast"
    )

    show_forecast_graph(
        st.session_state.result_actual,
        st.session_state.result_forecast,
        f"{plant_type} Forecast vs Actual",
    )
