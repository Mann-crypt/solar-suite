# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# COMPACT STREAMLIT PAGE
#
# CALCULATION LOGIC PRESERVED
# ERROR % APPLIED ONLY ONCE
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE
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

    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1450px;
        padding: 1.4rem 2rem 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #ffffff, #f1f6ff);
        border: 1px solid #e3e8f0;
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 18px;
        box-shadow: 0 5px 20px rgba(20, 40, 80, 0.06);
    }

    .hero-title {
        font-size: 30px;
        font-weight: 750;
        color: #172033;
        margin: 0;
    }

    .hero-subtitle {
        margin-top: 5px;
        color: #6b7280;
        font-size: 14px;
    }

    .section {
        background: white;
        border: 1px solid #e3e8f0;
        border-radius: 15px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 3px 14px rgba(20, 40, 80, 0.045);
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 13px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e3e8f0;
        border-radius: 14px;
        padding: 15px 18px;
        box-shadow: 0 3px 12px rgba(20, 40, 80, 0.04);
    }

    .metric-label {
        font-size: 12px;
        color: #737b89;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 750;
        color: #172033;
    }

    div[data-testid="stFileUploader"] {
        border: 1px solid #dfe5ee;
        border-radius: 12px;
        background: #fafbfe;
    }

    div[data-testid="stDataEditor"] {
        border-radius: 12px;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
    }

    div[data-baseweb="input"] input {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "calculated": False,
    "calculation_data": None,
    "plant_type": "Fixed",
    "uploaded_name": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">☀️ Solar Forecast Correction</div>
        <div class="hero-subtitle">
            Forecast correction and parameter optimization for Fixed and Tracking plants
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def numeric(value):
    """
    Safely convert Series / ndarray / list / scalar to numeric ndarray.
    Fixes the ndarray.fillna() issue.
    """
    if isinstance(value, pd.Series):
        return (
            pd.to_numeric(value, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )

    if isinstance(value, pd.DataFrame):
        return (
            value.apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )

    arr = np.asarray(value)

    if arr.ndim == 0:
        try:
            return np.array([float(arr)])
        except Exception:
            return np.array([0.0])

    return pd.to_numeric(
        pd.Series(arr.reshape(-1)),
        errors="coerce",
    ).fillna(0).to_numpy(dtype=float)


def read_excel(uploaded_file, **kwargs):
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, **kwargs)


# ============================================================
# EXCEL LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_area_efficiency(file_bytes):

    df = pd.read_excel(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df.columns = (
        df.columns.astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    if "S.No." in df.columns:
        nulls = df[df["S.No."].isna()].index
        if len(nulls):
            df = df.iloc[:df.index.get_loc(nulls[0])].copy()

    for col in [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df


@st.cache_data(show_spinner=False)
def load_cluster_table(file_bytes):

    df = pd.read_excel(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Clusters" in df.columns:
        nulls = df[df["Clusters"].isna()].index
        if len(nulls):
            df = df.iloc[:df.index.get_loc(nulls[0])].copy()

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_ghi(file_bytes):

    df = pd.read_excel(
        file_bytes,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    ).fillna(0)

    for col in [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

    return df


@st.cache_data(show_spinner=False)
def load_latitude(file_bytes):

    df = pd.read_excel(
        file_bytes,
        sheet_name="Forecast Config",
        header=8,
    )

    return float(
        pd.to_numeric(
            df.loc[0, "Lat"],
            errors="coerce",
        )
    )


@st.cache_data(show_spinner=False)
def load_tilt(file_bytes):

    df = pd.read_excel(
        file_bytes,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Fixed" in df.columns:
        nulls = df[df["Fixed"].isna()].index
        if len(nulls):
            df = df.iloc[:df.index.get_loc(nulls[0])].copy()

    df = df.dropna(how="all", axis=1)

    df = df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    return (
        df.set_index("Month")["Fixed"]
        .to_dict()
    )


@st.cache_data(show_spinner=False)
def load_fixed_data(file_bytes):

    df = pd.read_excel(
        file_bytes,
        sheet_name="Fixed-C11",
        header=1,
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Date" in df.columns:
        nulls = df[df["Date"].isna()].index
        if len(nulls):
            df = df.iloc[:df.index.get_loc(nulls[0])].copy()

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_tracking_data(file_bytes):

    backend_list = []

    for cluster in ["C11", "C12", "C13", "C14", "C15"]:

        backend_list.append(
            pd.read_excel(
                file_bytes,
                sheet_name=f"Backend Cal {cluster}",
            )
        )

    df_trac = pd.read_excel(
        file_bytes,
        sheet_name="Tracking",
        header=1,
    )

    df_trac.columns = (
        df_trac.columns.astype(str)
        .str.strip()
    )

    return backend_list, df_trac


# ============================================================
# GEOMETRY
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
                    + (df["Date"] - first_date).dt.days
                    + 1
                )
                / 365
            )
        )
    )

    df["Elevation angle a"] = (
        90 - lat + df["Declination Angle ∆"]
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
        np.radians(df["Elevation angle a"])
    )

    cluster_map = {
        "C11": ("GHI C11", "GHI*sin(a)", "GHI*sin(a+b)", "POA fixed"),
        "C12": ("GHI C12", "GHI*sin(a)-CL2", "GHI*sin(a+b)-CL2", "POA Fixed-C12"),
        "C13": ("GHI C13", "GHI*sin(a)-CL3", "GHI*sin(a+b)-CL3", "POA Fixed-C13"),
        "C14": ("GHI C14", "GHI*sin(a)-CL4", "GHI*sin(a+b)-CL4", "POA Fixed-C14"),
        "C15": ("GHI C15", "GHI*sin(a)-CL5", "GHI*sin(a+b)-CL5", "POA Fixed-C15"),
    }

    for ghi_col, mult_a, mult_ab, poa_col in cluster_map.values():

        df[mult_a] = (
            df_ghi[ghi_col]
            * df["Sin(a)"]
        )

        df[mult_ab] = (
            df_ghi[ghi_col]
            * df["SIN(a+b)"]
        )

        df[poa_col] = (
            df[mult_ab]
            / df["Sin(a"].replace(0, np.nan)
            if False
            else df[mult_ab]
            / df["Sin(a)"].replace(0, np.nan)
        )

    return df


# ============================================================
# EFFECTIVE AREA
# ============================================================

def calculate_effective_area(
    df_original,
    df_w_original,
    error,
):

    df = df_original.copy()
    df_w = df_w_original.copy()

    df["Error %"] = float(error)

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - float(error)
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

    poa_cols = [
        "POA fixed",
        "POA Fixed-C12",
        "POA Fixed-C13",
        "POA Fixed-C14",
        "POA Fixed-C15",
    ]

    power_cols = []

    for i, poa_col in enumerate(poa_cols):

        power_col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        df[power_col] = (
            df[poa_col]
            * float(
                pd.to_numeric(
                    df_w.iloc[i]["Eff Area(m2)"],
                    errors="coerce",
                )
                if pd.notna(
                    df_w.iloc[i]["Eff Area(m2)"]
                )
                else 0
            )
            / 1_000_000
        )

        power_cols.append(power_col)

    df["Total Power (CL1+CL2+…)"] = (
        df[power_cols]
        .sum(axis=1)
    )

    return df


# ============================================================
# ERROR OPTIMIZATION
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

    rows = []

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

        forecast_peak = numeric(
            calculated[
                "Total Power (CL1+CL2+…)"
            ]
        ).max()

        peak_error = abs(
            forecast_peak - actual_peak
        )

        rows.append(
            {
                "Error %": round(error, 1),
                "Calculated Peak": forecast_peak,
                "Actual Peak": actual_peak,
                "Peak Error": peak_error,
                "Peak Error %": (
                    peak_error
                    / actual_peak
                    * 100
                ),
            }
        )

    result_df = pd.DataFrame(rows)

    best = result_df.loc[
        result_df["Peak Error"].idxmin()
    ]

    return (
        float(best["Error %"]),
        result_df,
    )


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def create_tracking_objective(
    backend_list,
    df_ghi,
    df_fix,
    df_w,
):

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]

    cl_weights = numeric(
        df_w.iloc[:5]["Eff Area(m2)"]
    )

    ghi_matrix = np.column_stack(
        [
            numeric(df_ghi[col])
            for col in ghi_cols
        ]
    )

    blocks = numeric(
        backend_list[0]["Block No."]
    )

    actual_full = numeric(
        df_fix["Actual"]
    )

    if len(actual_full) == 0:
        raise ValueError(
            "Actual data is empty."
        )

    if len(blocks) != len(ghi_matrix):
        raise ValueError(
            "Tracking Block No. and GHI data have different lengths."
        )

    if len(actual_full) != len(blocks):
        raise ValueError(
            "Tracking Actual and Block No. have different lengths."
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
        max_block = int(round(x[3]))
        east = int(round(x[4]))
        west = int(round(x[5]))

        if not (
            start < max_block < end
        ):
            return 1e9

        den1 = (
            start - 1 - max_block
        )

        den2 = (
            end + 1 - max_block
        )

        if den1 == 0 or den2 == 0:
            return 1e9

        m1 = 90 / den1
        m2 = 90 / den2

        zenith = np.where(
            blocks <= max_block,

            np.minimum(
                89,
                m1 * (blocks - max_block),
            ),

            np.minimum(
                89,
                m2 * (blocks - max_block),
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

        cos_alpha = np.clip(
            np.cos(
                np.radians(panel)
            ),
            1e-6,
            None,
        )

        dhi = (
            ghi_matrix
            * DHI
            / 100
        )

        dni = (
            ghi_matrix - dhi
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

        block_error = (
            np.mean(
                np.abs(
                    actual - prediction
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
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    blocks,
    ghi_matrix,
    cl_weights,
    DHI,
    start,
    end,
    max_block,
    east,
    west,
):

    den1 = (
        start - 1 - max_block
    )

    den2 = (
        end + 1 - max_block
    )

    if den1 == 0 or den2 == 0:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    m1 = 90 / den1
    m2 = 90 / den2

    zenith = np.where(
        blocks <= max_block,

        np.minimum(
            89,
            m1 * (blocks - max_block),
        ),

        np.minimum(
            89,
            m2 * (blocks - max_block),
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

    cos_alpha = np.clip(
        np.cos(
            np.radians(panel)
        ),
        1e-6,
        None,
    )

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    dni = (
        ghi_matrix - dhi
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

    actual = numeric(actual)
    forecast = numeric(forecast)

    if len(actual) != len(forecast):
        raise ValueError(
            "Actual and Forecast lengths do not match."
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

    return {
        "Actual Peak": actual_peak,
        "Forecast Peak": forecast_peak,
    }


# ============================================================
# GRAPH
# ============================================================

def build_graph(
    actual,
    forecast,
    title,
):

    actual = numeric(actual)
    forecast = numeric(forecast)

    n = min(
        len(actual),
        len(forecast),
    )

    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual[:n],
            mode="lines",
            name="Actual",
            line=dict(width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast[:n],
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
            l=25,
            r=25,
            t=55,
            b=25,
        ),
        xaxis_title="Block",
        yaxis_title="Power",
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
# INPUT DATA
# ============================================================

st.markdown(
    '<div class="section"><div class="section-title">📂 Input Data</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Solar Excel File",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
    key="solar_excel_uploader",
)

st.markdown("</div>", unsafe_allow_html=True)


if uploaded_file is None:
    st.info(
        "Upload the Solar Excel file to begin."
    )
    st.stop()


# ============================================================
# RESET WHEN FILE CHANGES
# ============================================================

file_name = uploaded_file.name

if (
    st.session_state.uploaded_name
    != file_name
):

    st.session_state.calculated = False
    st.session_state.calculation_data = None
    st.session_state.uploaded_name = file_name


file_bytes = uploaded_file.getvalue()


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section"><div class="section-title">🌱 Plant Type</div>',
    unsafe_allow_html=True,
)

plant_type = st.radio(
    "Plant Type",
    ["Fixed", "Tracking"],
    horizontal=True,
    label_visibility="collapsed",
    key="plant_type_selector",
)

st.session_state.plant_type = plant_type

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# RUN CALCULATION
# ============================================================

st.markdown(
    '<div class="section">',
    unsafe_allow_html=True,
)

run_clicked = st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    use_container_width=True,
    key="run_calculation",
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

if run_clicked:

    try:

        with st.spinner(
            "Calculating forecast parameters..."
        ):

            # ------------------------------------------------
            # LOAD
            # ------------------------------------------------

            df_original = load_area_efficiency(
                file_bytes
            )

            df_w_original = load_cluster_table(
                file_bytes
            )

            df_ghi = load_ghi(
                file_bytes
            )

            lat = load_latitude(
                file_bytes
            )

            month_lookup = load_tilt(
                file_bytes
            )

            df_fix_raw = load_fixed_data(
                file_bytes
            )

            df_fix = prepare_fixed_geometry(
                df_fix_raw,
                df_ghi,
                lat,
                month_lookup,
            )

            # ------------------------------------------------
            # OPTIMIZE ERROR
            # ------------------------------------------------

            (
                best_error,
                error_results,
            ) = optimize_error(
                df_original,
                df_w_original,
                df_fix,
            )

            # ------------------------------------------------
            # APPLY ERROR ONCE
            # ------------------------------------------------

            (
                df_final,
                df_w_final,
            ) = calculate_effective_area(
                df_original,
                df_w_original,
                best_error,
            )

            # ------------------------------------------------
            # FIXED
            # ------------------------------------------------

            fixed_final = calculate_fixed_power(
                df_fix,
                df_w_final,
            )

            # ------------------------------------------------
            # TRACKING
            # ------------------------------------------------

            (
                backend_list,
                df_trac,
            ) = load_tracking_data(
                file_bytes
            )

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

            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            st.session_state.calculation_data = {

                "df_original":
                    df_original,

                "df_w_original":
                    df_w_original,

                "df_final":
                    df_final,

                "df_w_final":
                    df_w_final,

                "df_ghi":
                    df_ghi.copy(),

                "df_fix":
                    df_fix.copy(),

                "fixed_final":
                    fixed_final,

                "backend_list":
                    backend_list,

                "df_trac":
                    df_trac,

                "blocks":
                    blocks,

                "ghi_matrix":
                    ghi_matrix,

                "actual_tracking":
                    numeric(actual_tracking),

                "cl_weights":
                    cl_weights,

                "best_error":
                    best_error,

                "tracking_parameters":
                    tracking_parameters,

                "tracking_score":
                    tracking_score,

                "tracking_forecast":
                    tracking_forecast,

                "lat":
                    lat,

                "month_lookup":
                    month_lookup,

                "error_results":
                    error_results,
            }

            st.session_state.calculated = True

        st.success(
            "Calculation completed successfully."
        )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )


# ============================================================
# STOP
# ============================================================

if not st.session_state.calculated:
    st.stop()


data = st.session_state.calculation_data


# ============================================================
# EDITABLE INPUT DATA
#
# User can modify:
#   GHI C11...C15
#   Actual
#
# No Apply button.
#
# Changes are used when Run Calculation is pressed again.
# ============================================================

st.markdown(
    '<div class="section"><div class="section-title">✏️ Editable Input Data</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Edit GHI forecast values and Actual power directly below. "
    "Run the calculation again to use the updated values."
)

input_df = data["df_ghi"].copy()

input_df["Actual"] = numeric(
    data["df_fix"]["Actual"]
)

editable_cols = [
    col
    for col in [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
        "Actual",
    ]
    if col in input_df.columns
]

edited_input = st.data_editor(
    input_df[editable_cols],
    use_container_width=True,
    height=330,
    hide_index=True,
    num_rows="fixed",
    column_config={
        col: st.column_config.NumberColumn(
            col,
            format="%.3f",
        )
        for col in editable_cols
    },
    key="solar_input_editor",
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SAVE EDITED INPUT DATA
# ============================================================

for col in [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]:

    if col in edited_input.columns:
        data["df_ghi"][col] = numeric(
            edited_input[col]
        )

if "Actual" in edited_input.columns:

    data["df_fix"]["Actual"] = numeric(
        edited_input["Actual"]
    )

data["actual_tracking"] = numeric(
    data["df_fix"]["Actual"]
)

st.session_state.calculation_data = data


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section"><div class="section-title">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# ERROR
# ------------------------------------------------------------

error_col, _ = st.columns([1, 3])

with error_col:

    error_value = st.number_input(
        "Error %",
        min_value=0.0,
        max_value=20.0,
        value=float(
            data["best_error"]
        ),
        step=0.1,
        format="%.1f",
        key="error_parameter",
    )


# ------------------------------------------------------------
# TRACKING
# ------------------------------------------------------------

if plant_type == "Tracking":

    params = data[
        "tracking_parameters"
    ]

    st.markdown(
        "##### Tracking Optimization Parameters"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi_value = st.number_input(
            "DHI (%)",
            0,
            100,
            int(params["DHI"]),
            1,
            key="tracking_dhi",
        )

        start_value = st.number_input(
            "GHI Starting Block",
            0,
            95,
            int(
                params[
                    "GHI Starting Block"
                ]
            ),
            1,
            key="tracking_start",
        )

    with c2:

        end_value = st.number_input(
            "GHI Ending Block",
            1,
            96,
            int(
                params[
                    "GHI Ending Block"
                ]
            ),
            1,
            key="tracking_end",
        )

        max_value = st.number_input(
            "GHI Max Block",
            0,
            95,
            int(
                params[
                    "GHI Max Block"
                ]
            ),
            1,
            key="tracking_max",
        )

    with c3:

        east_value = st.number_input(
            "Tracking East Limit",
            0,
            90,
            int(
                params[
                    "Tracking East Limit"
                ]
            ),
            1,
            key="tracking_east",
        )

        west_value = st.number_input(
            "Tracking West Limit",
            0,
            90,
            int(
                params[
                    "Tracking West Limit"
                ]
            ),
            1,
            key="tracking_west",
        )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# LIVE PARAMETER CALCULATION
#
# No Apply Parameters button.
# ============================================================

try:

    (
        df_final,
        df_w_final,
    ) = calculate_effective_area(
        data["df_original"],
        data["df_w_original"],
        error_value,
    )

    fixed_final = calculate_fixed_power(
        data["df_fix"],
        df_w_final,
    )

    if plant_type == "Tracking":

        tracking_weights = numeric(
            df_w_final.iloc[:5][
                "Eff Area(m2)"
            ]
        )

        tracking_forecast = (
            calculate_tracking_forecast(
                data["blocks"],
                data["df_ghi"][
                    [
                        "GHI C11",
                        "GHI C12",
                        "GHI C13",
                        "GHI C14",
                        "GHI C15",
                    ]
                ].to_numpy(dtype=float),
                tracking_weights,
                int(dhi_value),
                int(start_value),
                int(end_value),
                int(max_value),
                int(east_value),
                int(west_value),
            )
        )

    else:

        tracking_forecast = data[
            "tracking_forecast"
        ]

except Exception as e:

    st.error(
        f"Parameter calculation failed: {e}"
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

if plant_type == "Fixed":

    actual = numeric(
        data["df_fix"]["Actual"]
    )

    forecast = numeric(
        fixed_final[
            "Total Power (CL1+CL2+…)"
        ]
    )

    title = (
        "Fixed Plant | Actual vs Forecast"
    )

else:

    actual = numeric(
        data["actual_tracking"]
    )

    forecast = numeric(
        tracking_forecast
    )

    title = (
        "Tracking Plant | Actual vs Forecast"
    )


metrics = calculate_metrics(
    actual,
    forecast,
)


# ============================================================
# RESULT CARDS
# ============================================================

st.markdown(
    '<div class="section"><div class="section-title">📊 Results</div>',
    unsafe_allow_html=True,
)

m1, m2 = st.columns(2)

with m1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Actual Peak</div>
            <div class="metric-value">
                {metrics["Actual Peak"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Forecast Peak</div>
            <div class="metric-value">
                {metrics["Forecast Peak"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# GRAPH
# ============================================================

st.markdown(
    '<div class="section"><div class="section-title">📈 Forecast Comparison</div>',
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
        "displayModeBar": False,
        "responsive": True,
    },
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FINAL STATUS
# ============================================================

st.caption(
    f"Plant: {plant_type}  •  "
    f"Error correction: {error_value:.1f}%  •  "
    f"Rows: {len(actual):,}"
)
