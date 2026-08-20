# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# STREAMLIT PAGE
#
# Calculation logic preserved.
# UI cleaned and optimized.
# ============================================================

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
# CSS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .app-title {
            font-size: 2rem;
            font-weight: 750;
            letter-spacing: -0.5px;
            margin-bottom: 2px;
        }

        .app-subtitle {
            color: #6b7280;
            font-size: 0.92rem;
            margin-bottom: 1.2rem;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin: 1.2rem 0 0.65rem 0;
        }

        .card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.035);
        }

        .card-label {
            color: #6b7280;
            font-size: 0.78rem;
            margin-bottom: 3px;
        }

        .card-value {
            font-size: 1.45rem;
            font-weight: 750;
        }

        div[data-testid="stDataEditor"] {
            border-radius: 10px;
        }

        div[data-testid="stFileUploader"] {
            border-radius: 12px;
        }

        .stButton > button {
            min-height: 42px;
            border-radius: 10px;
            font-weight: 650;
        }

        div[data-baseweb="segmented-control"] {
            margin-bottom: 4px;
        }

        .result-header {
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 1.3rem;
            margin-bottom: 0.7rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

POA_COLS = [
    "POA fixed",
    "POA Fixed-C12",
    "POA Fixed-C13",
    "POA Fixed-C14",
    "POA Fixed-C15",
]

POWER_COL = "Total Power (CL1+CL2+…)"


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "plant_type" not in st.session_state:
    st.session_state.plant_type = "Fixed"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Forecast correction and parameter optimization for Fixed and Tracking plants"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def read_excel(uploaded_file, **kwargs):
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, **kwargs)


def numeric(series):
    """
    Safe numeric conversion.
    Always returns a Series.
    """
    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )
    return df


def trim_at_first_null(df, column):
    if column not in df.columns:
        return df

    null_idx = df[df[column].isna()].index

    if len(null_idx):
        pos = df.index.get_loc(null_idx[0])
        df = df.iloc[:pos].copy()

    return df


# ============================================================
# LOAD AREA & EFFICIENCY
# ============================================================

def load_area_efficiency(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df = clean_columns(df)
    df = trim_at_first_null(df, "S.No.")

    required = [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in Area & Efficiency sheet."
            )

        df[col] = numeric(df[col])

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df


# ============================================================
# LOAD CLUSTER AREA
# ============================================================

def load_cluster_table(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df = clean_columns(df)
    df = trim_at_first_null(df, "Clusters")

    return df.reset_index(drop=True)


# ============================================================
# LOAD GHI
# ============================================================

def load_ghi(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df = df.fillna(0)

    for col in GHI_COLS:
        if col in df.columns:
            df[col] = numeric(df[col])

    return df


# ============================================================
# LOAD LATITUDE
# ============================================================

def load_latitude(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8,
    )

    if "Lat" not in df.columns:
        raise ValueError(
            "Lat column not found in Forecast Config."
        )

    lat = pd.to_numeric(
        df.loc[0, "Lat"],
        errors="coerce",
    )

    if pd.isna(lat):
        raise ValueError(
            "Latitude value is invalid."
        )

    return float(lat)


# ============================================================
# LOAD TILT
# ============================================================

def load_tilt(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df = clean_columns(df)
    df = trim_at_first_null(df, "Fixed")

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

    if "Month" not in df.columns or "Fixed" not in df.columns:
        raise ValueError(
            "Month / Fixed columns not found in Config Tilt Angle."
        )

    df["Fixed"] = numeric(df["Fixed"])

    return (
        df.set_index("Month")["Fixed"]
        .to_dict()
    )


# ============================================================
# LOAD FIXED DATA
# ============================================================

def load_fixed_data(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Fixed-C11",
        header=1,
    )

    df = clean_columns(df)
    df = trim_at_first_null(df, "Date")

    if "Actual" not in df.columns:
        raise ValueError(
            "Actual column not found in Fixed-C11."
        )

    df["Actual"] = numeric(df["Actual"])

    return df.reset_index(drop=True)


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def prepare_fixed_geometry(
    df_fix,
    df_ghi,
    lat,
    month_lookup,
):

    df = df_fix.copy()

    today = pd.Timestamp.today()

    df["Date"] = today

    first_date = (
        today
        .replace(
            month=1,
            day=1,
        )
        .normalize()
    )

    df["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + (
                        df["Date"]
                        - first_date
                    ).dt.days
                    + 1
                )
                / 365
            )
        )
    )

    df["Elevation angle a"] = (
        90
        - lat
        + df["Declination Angle ∆"]
    )

    df["Tilt Angle b"] = (
        df["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

    df["a+b"] = (
        df["Elevation angle a"]
        + df["Tilt Angle b"]
    )

    df["SIN(a+b)"] = np.sin(
        np.radians(df["a+b"])
    )

    df["Sin(a)"] = np.sin(
        np.radians(
            df["Elevation angle a"]
        )
    )

    sin_a = df["Sin(a)"].replace(
        0,
        np.nan,
    )

    cluster_map = {
        "C11": ("GHI C11", "GHI*sin(a)", "GHI*sin(a+b)", "POA fixed"),
        "C12": ("GHI C12", "GHI*sin(a)-CL2", "GHI*sin(a+b)-CL2", "POA Fixed-C12"),
        "C13": ("GHI C13", "GHI*sin(a)-CL3", "GHI*sin(a+b)-CL3", "POA Fixed-C13"),
        "C14": ("GHI C14", "GHI*sin(a)-CL4", "GHI*sin(a+b)-CL4", "POA Fixed-C14"),
        "C15": ("GHI C15", "GHI*sin(a)-CL5", "GHI*sin(a+b)-CL5", "POA Fixed-C15"),
    }

    for cluster, (
        ghi_col,
        ghi_a_col,
        ghi_ab_col,
        poa_col,
    ) in cluster_map.items():

        ghi = numeric(
            df_ghi[ghi_col]
        )

        df[ghi_a_col] = (
            ghi
            * df["Sin(a)"]
        )

        df[ghi_ab_col] = (
            ghi
            * df["SIN(a+b)"]
        )

        df[poa_col] = (
            df[ghi_ab_col]
            / sin_a
        )

    return df


# ============================================================
# EFFECTIVE AREA
#
# ERROR % APPLIED ONLY ONCE.
# ============================================================

def calculate_effective_area(
    df_original,
    df_w_original,
    error,
):

    df = df_original.copy()
    df_w = df_w_original.copy()

    df["Error %"] = error

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - error
    )

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

    cluster_sums = (
        df.groupby("Clusters")["Eff Area"]
        .sum()
    )

    df_w["Eff Area(m2)"] = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0)
    )

    return df, df_w


# ============================================================
# FIXED POWER
# ============================================================

def calculate_fixed_power(
    df_fix,
    df_w,
):

    df = df_fix.copy()
    power_cols = []

    for i, poa_col in enumerate(POA_COLS):

        power_col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        area = pd.to_numeric(
            df_w.iloc[i]["Eff Area(m2)"],
            errors="coerce",
        )

        area = 0 if pd.isna(area) else float(area)

        df[power_col] = (
            numeric(df[poa_col])
            * area
            / 1_000_000
        )

        power_cols.append(power_col)

    df[POWER_COL] = df[power_cols].sum(
        axis=1
    )

    return df


# ============================================================
# AUTOMATIC ERROR OPTIMIZATION
# ============================================================

def optimize_error(
    df_original,
    df_w_original,
    df_fix,
):

    actual = numeric(
        df_fix["Actual"]
    )

    actual_peak = actual.max()

    if actual_peak <= 0:
        raise ValueError(
            "No non-zero Actual values found."
        )

    results = []

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        _, df_w = calculate_effective_area(
            df_original,
            df_w_original,
            error,
        )

        calculated = calculate_fixed_power(
            df_fix,
            df_w,
        )

        calculated_peak = (
            numeric(
                calculated[POWER_COL]
            ).max()
        )

        peak_error = abs(
            calculated_peak
            - actual_peak
        )

        results.append(
            {
                "Error %": round(error, 1),
                "Calculated Peak": calculated_peak,
                "Actual Peak": actual_peak,
                "Peak Error": peak_error,
                "Peak Error %": (
                    peak_error
                    / actual_peak
                    * 100
                ),
            }
        )

    result_df = pd.DataFrame(results)

    best_row = result_df.loc[
        result_df["Peak Error"].idxmin()
    ]

    return (
        float(best_row["Error %"]),
        result_df,
    )


# ============================================================
# TRACKING DATA
# ============================================================

def load_tracking_data(
    uploaded_file,
):

    backend_list = []

    for cluster in CLUSTERS:

        df_backend = read_excel(
            uploaded_file,
            sheet_name=f"Backend Cal {cluster}",
        )

        backend_list.append(
            df_backend
        )

    df_trac = read_excel(
        uploaded_file,
        sheet_name="Tracking",
        header=1,
    )

    df_trac = clean_columns(df_trac)

    return backend_list, df_trac


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def create_tracking_objective(
    backend_list,
    df_ghi,
    df_fix,
    df_w,
):

    cl_weights = (
        pd.to_numeric(
            df_w.iloc[:5]["Eff Area(m2)"],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    ghi_matrix = np.column_stack(
        [
            numeric(
                df_ghi[col]
            ).to_numpy(dtype=float)
            for col in GHI_COLS
        ]
    )

    blocks = numeric(
        backend_list[0]["Block No."]
    ).to_numpy(dtype=float)

    actual_full = numeric(
        df_fix["Actual"]
    ).to_numpy(dtype=float)

    if len(actual_full) == 0:
        raise ValueError(
            "Actual data is empty."
        )

    if len(blocks) != len(ghi_matrix):
        raise ValueError(
            "Tracking Block No. and GHI data "
            "have different lengths."
        )

    if len(actual_full) != len(blocks):
        raise ValueError(
            "Tracking Actual and Block No. "
            "have different lengths."
        )

    mask = actual_full != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual = actual_full[mask]

    actual_max = actual.max()
    actual_sum = actual.sum()

    def objective(x):

        DHI = int(round(x[0]))
        start = int(round(x[1]))
        end = int(round(x[2]))
        maximum = int(round(x[3]))
        east = int(round(x[4]))
        west = int(round(x[5]))

        if not (
            start
            < maximum
            < end
        ):
            return 1e9

        denominator_1 = (
            start
            - 1
            - maximum
        )

        denominator_2 = (
            end
            + 1
            - maximum
        )

        if denominator_1 == 0 or denominator_2 == 0:
            return 1e9

        m1 = 90 / denominator_1
        m2 = 90 / denominator_2

        zenith = np.where(
            blocks <= maximum,

            np.minimum(
                89,
                m1 * (
                    blocks
                    - maximum
                ),
            ),

            np.minimum(
                89,
                m2 * (
                    blocks
                    - maximum
                ),
            ),
        )

        panel = np.where(
            blocks < maximum,

            np.minimum(
                zenith,
                abs(east),
            ),

            np.where(
                (
                    (blocks > maximum)
                    &
                    (zenith > west)
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

        dhi = (
            ghi_matrix
            * DHI
            / 100
        )

        dni = (
            ghi_matrix
            - dhi
        ) / cos_alpha[:, None]

        prediction_full = (
            dni @ cl_weights
        ) / 1_000_000

        if (
            np.isnan(prediction_full).any()
            or np.isinf(prediction_full).any()
        ):
            return 1e9

        prediction = prediction_full[mask]

        if len(prediction) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual
                    - prediction
                )
            )
            / actual_max
        )

        peak_error = (
            abs(
                actual_max
                - prediction.max()
            )
            / actual_max
        )

        energy_error = (
            abs(
                actual_sum
                - prediction.sum()
            )
            / actual_sum
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    return (
        objective,
        blocks,
        ghi_matrix,
        actual_full,
        cl_weights,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    backend_list,
    df_ghi,
    df_fix,
    df_w,
):

    (
        objective,
        blocks,
        ghi_matrix,
        actual_full,
        cl_weights,
    ) = create_tracking_objective(
        backend_list,
        df_ghi,
        df_fix,
        df_w,
    )

    bounds = [
        (0, 10),
        (10, 30),
        (65, 80),
        (47, 53),
        (10, 70),
        (10, 70),
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
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

    best = np.round(
        result.x
    ).astype(int)

    parameters = {
        "DHI": int(best[0]),
        "GHI Starting Block": int(best[1]),
        "GHI Ending Block": int(best[2]),
        "GHI Max Block": int(best[3]),
        "Tracking East Limit": int(best[4]),
        "Tracking West Limit": int(best[5]),
    }

    return (
        parameters,
        blocks,
        ghi_matrix,
        actual_full,
        cl_weights,
        result.fun,
    )


# ============================================================
# FINAL TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    blocks,
    ghi_matrix,
    cl_weights,
    DHI,
    GHI_Starting_Block,
    GHI_Ending_Block,
    GHI_Max_Block,
    Tracking_angle_lim_E,
    Tracking_angle_lim_W,
):

    denominator_1 = (
        GHI_Starting_Block
        - 1
        - GHI_Max_Block
    )

    denominator_2 = (
        GHI_Ending_Block
        + 1
        - GHI_Max_Block
    )

    if denominator_1 == 0:
        raise ValueError(
            "Invalid Tracking parameters: "
            "East slope denominator is zero."
        )

    if denominator_2 == 0:
        raise ValueError(
            "Invalid Tracking parameters: "
            "West slope denominator is zero."
        )

    m1 = 90 / denominator_1
    m2 = 90 / denominator_2

    zenith = np.where(
        blocks <= GHI_Max_Block,

        np.minimum(
            89,
            m1 * (
                blocks
                - GHI_Max_Block
            ),
        ),

        np.minimum(
            89,
            m2 * (
                blocks
                - GHI_Max_Block
            ),
        ),
    )

    panel = np.where(
        blocks < GHI_Max_Block,

        np.minimum(
            zenith,
            abs(
                Tracking_angle_lim_E
            ),
        ),

        np.where(
            (
                (blocks > GHI_Max_Block)
                &
                (
                    zenith
                    > Tracking_angle_lim_W
                )
            ),

            Tracking_angle_lim_W,
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

    forecast = (
        dni @ cl_weights
    ) / 1_000_000

    return forecast


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    forecast,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    actual_peak = (
        np.max(actual)
        if len(actual)
        else 0
    )

    forecast_peak = (
        np.max(forecast)
        if len(forecast)
        else 0
    )

    actual_energy = np.sum(actual)
    forecast_energy = np.sum(forecast)

    peak_error = abs(
        forecast_peak
        - actual_peak
    )

    energy_error = abs(
        forecast_energy
        - actual_energy
    )

    return {
        "Actual Peak": actual_peak,
        "Forecast Peak": forecast_peak,
        "Peak Error": peak_error,
        "Energy Error": energy_error,
    }


# ============================================================
# GRAPH
# ============================================================

def build_graph(
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

    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(width=2.5),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
        ),
        height=450,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30,
        ),
        xaxis=dict(
            title="Block",
            dtick=5,
        ),
        yaxis=dict(
            title="Power",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
    )

    return fig


# ============================================================
# INPUT FILE
# ============================================================

st.markdown(
    '<div class="section-title">📁 Input Data</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Solar Excel File",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)

if uploaded_file is None:

    st.info(
        "Upload your Solar Excel file to begin."
    )

    st.stop()


# ============================================================
# LOAD SOURCE DATA ONCE
# ============================================================

try:

    df_original = load_area_efficiency(
        uploaded_file
    )

    df_w_original = load_cluster_table(
        uploaded_file
    )

    df_ghi_original = load_ghi(
        uploaded_file
    )

    df_fix_original = load_fixed_data(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read the Excel file: {e}"
    )

    st.stop()


# ============================================================
# INPUT DATA EDITORS + PLANT SELECTOR
#
# FORM PREVENTS EVERY EDIT FROM TRIGGERING
# THE FULL PAGE CALCULATION.
# ============================================================

with st.form(
    "solar_input_form",
    clear_on_submit=False,
):

    st.markdown(
        "#### GHI Forecast"
    )

    st.caption(
        "Edit the forecast GHI values below before running the calculation."
    )

    ghi_editor = st.data_editor(
        df_ghi_original[
            GHI_COLS
        ],
        use_container_width=True,
        num_rows="fixed",
        hide_index=False,
        key="ghi_editor",
    )

    st.markdown(
        "#### Actual Power"
    )

    st.caption(
        "Edit the Actual Power values used for comparison."
    )

    actual_editor = st.data_editor(
        pd.DataFrame(
            {
                "Actual": numeric(
                    df_fix_original["Actual"]
                )
            }
        ),
        use_container_width=True,
        num_rows="fixed",
        hide_index=False,
        key="actual_editor",
    )

    st.markdown(
        '<div class="section-title">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    plant_type = st.segmented_control(
        "Plant Type",
        options=[
            "Fixed",
            "Tracking",
        ],
        default=st.session_state.plant_type,
        label_visibility="collapsed",
    )

    if plant_type is None:
        plant_type = "Fixed"

    st.session_state.plant_type = plant_type

    st.markdown("")

    run_calculation = st.form_submit_button(
        "⚡ Run Calculation",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# RUN CALCULATION
# ============================================================

if run_calculation:

    try:

        # ----------------------------------------------------
        # Validate edited input lengths
        # ----------------------------------------------------

        if len(ghi_editor) != len(df_ghi_original):
            raise ValueError(
                "GHI Forecast row count cannot be changed."
            )

        if len(actual_editor) != len(df_fix_original):
            raise ValueError(
                "Actual Power row count cannot be changed."
            )

        # ----------------------------------------------------
        # Use USER-EDITED GHI
        # ----------------------------------------------------

        df_ghi = df_ghi_original.copy()

        for col in GHI_COLS:
            df_ghi[col] = numeric(
                ghi_editor[col]
            ).to_numpy()

        # ----------------------------------------------------
        # Use USER-EDITED ACTUAL
        # ----------------------------------------------------

        df_fix = df_fix_original.copy()

        df_fix["Actual"] = numeric(
            actual_editor["Actual"]
        ).to_numpy()

        # ----------------------------------------------------
        # Common configuration
        # ----------------------------------------------------

        with st.spinner(
            "Preparing solar calculation..."
        ):

            lat = load_latitude(
                uploaded_file
            )

            month_lookup = load_tilt(
                uploaded_file
            )

            df_fix = prepare_fixed_geometry(
                df_fix,
                df_ghi,
                lat,
                month_lookup,
            )

        # ----------------------------------------------------
        # Automatic Error %
        # ----------------------------------------------------

        with st.spinner(
            "Optimizing efficiency correction..."
        ):

            (
                best_error,
                error_results,
            ) = optimize_error(
                df_original,
                df_w_original,
                df_fix,
            )

        # ----------------------------------------------------
        # Apply Error % ONCE
        # ----------------------------------------------------

        (
            df_final,
            df_w_final,
        ) = calculate_effective_area(
            df_original,
            df_w_original,
            best_error,
        )

        # ----------------------------------------------------
        # Fixed forecast
        # ----------------------------------------------------

        fixed_final = calculate_fixed_power(
            df_fix,
            df_w_final,
        )

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        backend_list, df_trac = (
            load_tracking_data(
                uploaded_file
            )
        )

        with st.spinner(
            "Optimizing tracking parameters..."
        ):

            (
                tracking_parameters,
                blocks,
                ghi_matrix,
                actual_tracking,
                cl_weights,
                tracking_score,
            ) = optimize_tracking(
                backend_list,
                df_ghi,
                df_fix,
                df_w_final,
            )

        tracking_forecast = (
            calculate_tracking_forecast(
                blocks,
                ghi_matrix,
                cl_weights,
                tracking_parameters["DHI"],
                tracking_parameters[
                    "GHI Starting Block"
                ],
                tracking_parameters[
                    "GHI Ending Block"
                ],
                tracking_parameters[
                    "GHI Max Block"
                ],
                tracking_parameters[
                    "Tracking East Limit"
                ],
                tracking_parameters[
                    "Tracking West Limit"
                ],
            )
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        st.session_state.result = {
            "df_original": df_original,
            "df_w_original": df_w_original,
            "df_final": df_final,
            "df_w_final": df_w_final,
            "df_ghi": df_ghi,
            "df_fix": df_fix,
            "fixed_final": fixed_final,
            "backend_list": backend_list,
            "df_trac": df_trac,
            "blocks": blocks,
            "ghi_matrix": ghi_matrix,
            "actual_tracking": actual_tracking,
            "cl_weights": cl_weights,
            "best_error": best_error,
            "tracking_parameters": tracking_parameters,
            "tracking_score": tracking_score,
            "tracking_forecast": tracking_forecast,
            "error_results": error_results,
        }

        st.success(
            "Calculation completed successfully."
        )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )

        st.stop()


# ============================================================
# NO RESULTS YET
# ============================================================

if st.session_state.result is None:

    st.info(
        "Edit the input data if required, select the plant type, "
        "then click **Run Calculation**."
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

data = st.session_state.result
plant_type = st.session_state.plant_type


# ============================================================
# PARAMETER DISPLAY
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Optimized Parameters</div>',
    unsafe_allow_html=True,
)

if plant_type == "Fixed":

    p1, p2 = st.columns(2)

    with p1:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Efficiency Error</div>
                <div class="card-value">
                    {data["best_error"]:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p2:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Plant Type</div>
                <div class="card-value">
                    Fixed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    params = data["tracking_parameters"]

    cols = st.columns(6)

    parameter_items = [
        ("DHI", params["DHI"]),
        (
            "GHI Start",
            params["GHI Starting Block"],
        ),
        (
            "GHI End",
            params["GHI Ending Block"],
        ),
        (
            "GHI Max",
            params["GHI Max Block"],
        ),
        (
            "East Limit",
            params["Tracking East Limit"],
        ),
        (
            "West Limit",
            params["Tracking West Limit"],
        ),
    ]

    for col, (
        label,
        value,
    ) in zip(
        cols,
        parameter_items,
    ):

        with col:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">
                        {label}
                    </div>
                    <div class="card-value">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FORECAST DATA
# ============================================================

if plant_type == "Fixed":

    actual = numeric(
        data["df_fix"]["Actual"]
    ).to_numpy()

    forecast = numeric(
        data["fixed_final"][POWER_COL]
    ).to_numpy()

    title = (
        "Fixed Plant | Actual vs Forecast"
    )

else:

    # IMPORTANT:
    # actual_tracking is already a NumPy array
    # returned by optimize_tracking.
    #
    # Do NOT call .fillna() directly on it.

    actual = np.asarray(
        data["actual_tracking"],
        dtype=float,
    )

    forecast = np.asarray(
        data["tracking_forecast"],
        dtype=float,
    )

    title = (
        "Tracking Plant | Actual vs Forecast"
    )


# ============================================================
# ALIGN LENGTHS SAFELY
# ============================================================

n = min(
    len(actual),
    len(forecast),
)

actual = np.nan_to_num(
    actual[:n],
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

forecast = np.nan_to_num(
    forecast[:n],
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)


# ============================================================
# METRICS
# ============================================================

metrics = calculate_metrics(
    actual,
    forecast,
)


# ============================================================
# RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Results</div>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)

with m1:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">
                Plant Type
            </div>
            <div class="card-value">
                {plant_type}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">
                Actual Peak
            </div>
            <div class="card-value">
                {metrics["Actual Peak"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">
                Forecast Peak
            </div>
            <div class="card-value">
                {metrics["Forecast Peak"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FORECAST GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)

fig = build_graph(
    actual,
    forecast,
    title,
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        "responsive": True,
    },
)


# ============================================================
# OPTIONAL DATA PREVIEW
# ============================================================

with st.expander(
    "View Forecast Data",
    expanded=False,
):

    preview = pd.DataFrame(
        {
            "Block": np.arange(
                len(actual)
            ),
            "Actual": actual,
            "Forecast": forecast,
        }
    )

    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True,
    )
