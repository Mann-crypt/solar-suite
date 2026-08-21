# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# STREAMLIT PAGE
#
# CLEAN + NON-FREEZING VERSION
#
# CALCULATION LOGIC PRESERVED
# ============================================================

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
        font-size: 30px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 15px 17px;
        min-height: 88px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.035);
    }

    .result-label {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 5px;
    }

    .result-value {
        font-size: 23px;
        font-weight: 750;
    }

    div[data-testid="stFileUploader"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 6px;
        background: white;
    }

    div[data-testid="stSegmentedControl"] {
        width: 100%;
    }

    div[data-testid="stSegmentedControl"] > div {
        width: 100%;
    }

    button[kind="segmented_control"] {
        flex: 1;
    }

    .stButton > button {
        min-height: 42px;
        border-radius: 9px;
        font-weight: 650;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "calculated": False,
    "results": None,
    "input_df": None,
    "plant_type": "Fixed",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Forecast correction with automatic optimization and editable parameters"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def numeric(series):
    """
    Always return a pandas Series.
    Prevents pandas / numpy AttributeError issues.
    """
    return pd.to_numeric(
        pd.Series(series),
        errors="coerce",
    ).fillna(0)


def as_array(values):
    """
    Convert any pandas/numpy input to clean float array.
    """
    return (
        pd.to_numeric(
            pd.Series(values),
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )


def read_excel(file_bytes, **kwargs):
    """
    Cached-compatible Excel reader.
    """
    return pd.read_excel(
        io.BytesIO(file_bytes),
        **kwargs,
    )


# ============================================================
# CACHED WORKBOOK LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_area_efficiency(file_bytes):

    df = read_excel(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    if "S.No." in df.columns:

        mask = df["S.No."].isna()

        if mask.any():
            first = np.flatnonzero(mask.to_numpy())[0]
            df = df.iloc[:first].copy()

    df["Standard PV Efficiency (%)"] = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
        errors="coerce",
    )

    df["No of Module"] = pd.to_numeric(
        df["No of Module"],
        errors="coerce",
    )

    df["Area of 1 Module (m2)"] = pd.to_numeric(
        df["Area of 1 Module (m2)"],
        errors="coerce",
    )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_cluster_table(file_bytes):

    df = read_excel(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "Clusters" in df.columns:

        mask = df["Clusters"].isna()

        if mask.any():
            first = np.flatnonzero(mask.to_numpy())[0]
            df = df.iloc[:first].copy()

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_ghi(file_bytes):

    df = read_excel(
        file_bytes,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df = df.fillna(0)

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]

    for col in ghi_cols:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

    return df


@st.cache_data(show_spinner=False)
def load_latitude(file_bytes):

    df = read_excel(
        file_bytes,
        sheet_name="Forecast Config",
        header=8,
    )

    lat = pd.to_numeric(
        df.loc[0, "Lat"],
        errors="coerce",
    )

    if pd.isna(lat):
        raise ValueError("Latitude could not be read.")

    return float(lat)


@st.cache_data(show_spinner=False)
def load_tilt(file_bytes):

    df = read_excel(
        file_bytes,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" in df.columns:

        mask = df["Fixed"].isna()

        if mask.any():
            first = np.flatnonzero(mask.to_numpy())[0]
            df = df.iloc[:first].copy()

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

    return (
        df.set_index("Month")["Fixed"]
        .to_dict()
    )


@st.cache_data(show_spinner=False)
def load_fixed_data(file_bytes):

    df = read_excel(
        file_bytes,
        sheet_name="Fixed-C11",
        header=1,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "Date" in df.columns:

        mask = df["Date"].isna()

        if mask.any():
            first = np.flatnonzero(mask.to_numpy())[0]
            df = df.iloc[:first].copy()

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_tracking_data(file_bytes):

    backend_list = []

    for cluster in [
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
    ]:

        df = read_excel(
            file_bytes,
            sheet_name=f"Backend Cal {cluster}",
        )

        backend_list.append(df)

    df_tracking = read_excel(
        file_bytes,
        sheet_name="Tracking",
        header=1,
    )

    df_tracking.columns = (
        df_tracking.columns
        .astype(str)
        .str.strip()
    )

    return backend_list, df_tracking


# ============================================================
# FIXED GEOMETRY
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
        np.radians(df["Elevation angle a"])
    )

    clusters = [
        ("C11", "GHI*sin(a)", "GHI*sin(a+b)", "POA fixed"),
        ("C12", "GHI*sin(a)-CL2", "GHI*sin(a+b)-CL2", "POA Fixed-C12"),
        ("C13", "GHI*sin(a)-CL3", "GHI*sin(a+b)-CL3", "POA Fixed-C13"),
        ("C14", "GHI*sin(a)-CL4", "GHI*sin(a+b)-CL4", "POA Fixed-C14"),
        ("C15", "GHI*sin(a)-CL5", "GHI*sin(a+b)-CL5", "POA Fixed-C15"),
    ]

    for ghi_cluster, col_a, col_ab, poa_col in clusters:

        ghi = numeric(
            df_ghi[f"GHI {ghi_cluster}"]
        ).to_numpy()

        df[col_a] = (
            ghi
            * df["Sin(a)"]
        )

        df[col_ab] = (
            ghi
            * df["SIN(a+b)"]
        )

        df[poa_col] = (
            df[col_ab]
            / df["Sin(a)"].replace(
                0,
                np.nan,
            )
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

    result = df_fix.copy()

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

        result[power_col] = (
            result[poa_col]
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
        ) / 1_000_000

        power_cols.append(power_col)

    result[
        "Total Power (CL1+CL2+…)"
    ] = result[power_cols].sum(axis=1)

    return result


# ============================================================
# ERROR OPTIMIZATION
# ============================================================

def optimize_error(
    df_original,
    df_w_original,
    df_fix,
):

    actual = as_array(
        df_fix["Actual"]
    )

    actual_peak = actual.max()

    if actual_peak <= 0:
        raise ValueError(
            "No non-zero Actual values found."
        )

    best_error = 0.0
    best_error_value = np.inf

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

        forecast = as_array(
            calculated[
                "Total Power (CL1+CL2+…)"
            ]
        )

        peak_error = abs(
            forecast.max()
            - actual_peak
        )

        if peak_error < best_error_value:

            best_error_value = peak_error
            best_error = float(
                round(error, 1)
            )

    return best_error


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
    blocks,
    ghi_matrix,
    tracking_weights,
):

    if not (
        start_block
        < max_block
        < end_block
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

    if denominator_1 == 0:
        return None

    if denominator_2 == 0:
        return None

    m1 = 90 / denominator_1
    m2 = 90 / denominator_2

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
                &
                (
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

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    power_matrix = (
        dni
        * tracking_weights[None, :]
        / 1_000_000
    )

    forecast = (
        power_matrix.sum(axis=1)
    )

    return (
        forecast,
        power_matrix,
        zenith,
        panel,
        dni,
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

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]

    ghi_matrix = np.column_stack(
        [
            as_array(
                df_ghi[col]
            )
            for col in ghi_cols
        ]
    )

    blocks = as_array(
        backend_list[0]["Block No."]
    )

    actual_full = as_array(
        df_fix["Actual"]
    )

    if len(blocks) != len(ghi_matrix):
        raise ValueError(
            "Tracking Block No. and GHI data have different lengths."
        )

    if len(actual_full) != len(blocks):
        raise ValueError(
            "Tracking Actual and Block No. have different lengths."
        )

    valid_mask = actual_full != 0

    if not valid_mask.any():
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual_day = actual_full[valid_mask]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    tracking_weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    def objective(x):

        DHI = int(round(x[0]))
        start_block = int(round(x[1]))
        end_block = int(round(x[2]))
        max_block = int(round(x[3]))
        east_limit = int(round(x[4]))
        west_limit = int(round(x[5]))

        result = calculate_tracking(
            DHI,
            start_block,
            end_block,
            max_block,
            east_limit,
            west_limit,
            blocks,
            ghi_matrix,
            tracking_weights,
        )

        if result is None:
            return 1e9

        prediction = result[0]

        if not np.all(
            np.isfinite(prediction)
        ):
            return 1e9

        prediction_day = (
            prediction[valid_mask]
        )

        if len(prediction_day) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    - prediction_day
                )
            )
            / actual_peak
        )

        peak_error = (
            abs(
                actual_peak
                - prediction_day.max()
            )
            / actual_peak
        )

        energy_error = (
            abs(
                actual_energy
                - prediction_day.sum()
            )
            / actual_energy
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
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
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    best = np.rint(
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

    return {
        "parameters": parameters,
        "blocks": blocks,
        "ghi_matrix": ghi_matrix,
        "actual": actual_full,
        "tracking_weights": tracking_weights,
        "score": float(result.fun),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    forecast,
):

    actual = as_array(actual)
    forecast = as_array(forecast)

    actual_peak = (
        actual.max()
        if len(actual)
        else 0
    )

    forecast_peak = (
        forecast.max()
        if len(forecast)
        else 0
    )

    actual_energy = actual.sum()
    forecast_energy = forecast.sum()

    peak_error_pct = (
        abs(
            forecast_peak
            - actual_peak
        )
        / actual_peak
        * 100
        if actual_peak
        else np.nan
    )

    energy_error_pct = (
        abs(
            forecast_energy
            - actual_energy
        )
        / actual_energy
        * 100
        if actual_energy
        else np.nan
    )

    return {
        "Actual Peak": actual_peak,
        "Forecast Peak": forecast_peak,
        "Peak Error %": peak_error_pct,
        "Energy Error %": energy_error_pct,
    }


# ============================================================
# GRAPH
# ============================================================

def build_graph(
    actual,
    forecast,
    title,
):

    actual = as_array(actual)
    forecast = as_array(forecast)

    n = min(
        len(actual),
        len(forecast),
    )

    actual = actual[:n]
    forecast = forecast[:n]

    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(
                width=2.4,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                width=2.4,
            ),
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
        },
        height=450,
        template="plotly_white",
        hovermode="x unified",
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
        "Upload the Solar Excel workbook to start."
    )

    st.stop()


# ============================================================
# FILE BYTES
# ============================================================

file_bytes = uploaded_file.getvalue()


# ============================================================
# LOAD INPUT DATA
# ============================================================

try:

    raw_ghi = load_ghi(
        file_bytes
    )

    raw_fixed = load_fixed_data(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to read input workbook: {e}"
    )

    st.stop()


# ============================================================
# EDITABLE INPUT DATA
#
# User can change:
# GHI C11 ... GHI C15
# Actual
# ============================================================

input_cols = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

input_df = pd.DataFrame()

for col in input_cols:

    if col in raw_ghi.columns:

        input_df[col] = numeric(
            raw_ghi[col]
        )

    else:

        input_df[col] = 0.0


input_df["Actual"] = numeric(
    raw_fixed["Actual"]
)


# Match common length

n_rows = min(
    len(input_df),
    96,
)

input_df = (
    input_df
    .iloc[:n_rows]
    .reset_index(drop=True)
)


edited_input = st.data_editor(
    input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    height=310,
    column_config={
        "GHI C11": st.column_config.NumberColumn(
            "GHI C11",
            format="%.3f",
        ),
        "GHI C12": st.column_config.NumberColumn(
            "GHI C12",
            format="%.3f",
        ),
        "GHI C13": st.column_config.NumberColumn(
            "GHI C13",
            format="%.3f",
        ),
        "GHI C14": st.column_config.NumberColumn(
            "GHI C14",
            format="%.3f",
        ),
        "GHI C15": st.column_config.NumberColumn(
            "GHI C15",
            format="%.3f",
        ),
        "Actual": st.column_config.NumberColumn(
            "Actual",
            format="%.3f",
        ),
    },
    key="solar_input_editor",
)


# ============================================================
# PLANT TYPE
# ============================================================

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
    selection_mode="single",
    label_visibility="collapsed",
    width="stretch",
)

if plant_type is None:
    plant_type = "Fixed"

st.session_state.plant_type = plant_type


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)


# ============================================================
# INITIAL VALUES
# ============================================================

if st.session_state.results is not None:

    saved = st.session_state.results

    default_error = float(
        saved.get(
            "error",
            0.0,
        )
    )

    saved_tracking = saved.get(
        "tracking_parameters",
        {
            "DHI": 0,
            "GHI Starting Block": 20,
            "GHI Ending Block": 70,
            "GHI Max Block": 50,
            "Tracking East Limit": 30,
            "Tracking West Limit": 30,
        },
    )

else:

    default_error = 0.0

    saved_tracking = {
        "DHI": 0,
        "GHI Starting Block": 20,
        "GHI Ending Block": 70,
        "GHI Max Block": 50,
        "Tracking East Limit": 30,
        "Tracking West Limit": 30,
    }


# ============================================================
# EDITABLE ERROR
# ============================================================

error_value = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=20.0,
    value=default_error,
    step=0.1,
    format="%.1f",
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

if plant_type == "Tracking":

    st.markdown(
        "#### Tracking Parameters"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi_value = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(
                saved_tracking["DHI"]
            ),
            step=1,
        )

        start_value = st.number_input(
            "GHI Starting Block",
            min_value=0,
            max_value=95,
            value=int(
                saved_tracking[
                    "GHI Starting Block"
                ]
            ),
            step=1,
        )

    with c2:

        end_value = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                saved_tracking[
                    "GHI Ending Block"
                ]
            ),
            step=1,
        )

        max_value = st.number_input(
            "GHI Max Block",
            min_value=0,
            max_value=95,
            value=int(
                saved_tracking[
                    "GHI Max Block"
                ]
            ),
            step=1,
        )

    with c3:

        east_value = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            value=int(
                saved_tracking[
                    "Tracking East Limit"
                ]
            ),
            step=1,
        )

        west_value = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            value=int(
                saved_tracking[
                    "Tracking West Limit"
                ]
            ),
            step=1,
        )

else:

    st.caption(
        "Error % is applied to PV efficiency once."
    )


# ============================================================
# RUN BUTTON
#
# IMPORTANT:
# Expensive optimization happens ONLY here.
# ============================================================

st.markdown("")

run_clicked = st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    width="stretch",
)


# ============================================================
# RUN CALCULATION
# ============================================================

if run_clicked:

    try:

        with st.spinner(
            "Running solar forecast calculation..."
        ):

            # ------------------------------------------------
            # COMMON DATA
            # ------------------------------------------------

            df_original = (
                load_area_efficiency(
                    file_bytes
                )
            )

            df_w_original = (
                load_cluster_table(
                    file_bytes
                )
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

            # ------------------------------------------------
            # REPLACE GHI + ACTUAL WITH USER INPUT
            # ------------------------------------------------

            df_ghi = raw_ghi.copy()

            for col in input_cols:

                df_ghi[col] = (
                    edited_input[col]
                    .to_numpy()
                )

            df_fix = df_fix_raw.copy()

            df_fix["Actual"] = (
                edited_input["Actual"]
                .to_numpy()
            )

            # ------------------------------------------------
            # GEOMETRY
            # ------------------------------------------------

            df_fix = prepare_fixed_geometry(
                df_fix,
                df_ghi,
                lat,
                month_lookup,
            )

            # ------------------------------------------------
            # ERROR OPTIMIZATION
            #
            # This is required once per Run.
            # ------------------------------------------------

            if st.session_state.results is None:

                calculated_error = (
                    optimize_error(
                        df_original,
                        df_w_original,
                        df_fix,
                    )
                )

            else:

                # User-editable error is respected.
                calculated_error = float(
                    error_value
                )

            # ------------------------------------------------
            # USER ERROR OVERRIDE
            # ------------------------------------------------

            calculated_error = float(
                error_value
            )

            # ------------------------------------------------
            # EFFECTIVE AREA
            # ------------------------------------------------

            (
                df_final,
                df_w_final,
            ) = calculate_effective_area(
                df_original,
                df_w_original,
                calculated_error,
            )

            # ------------------------------------------------
            # FIXED FORECAST
            # ------------------------------------------------

            fixed_final = (
                calculate_fixed_power(
                    df_fix,
                    df_w_final,
                )
            )

            # ------------------------------------------------
            # BASE RESULT
            # ------------------------------------------------

            results = {
                "df_original":
                    df_original,

                "df_w_original":
                    df_w_original,

                "df_final":
                    df_final,

                "df_w_final":
                    df_w_final,

                "df_ghi":
                    df_ghi,

                "df_fix":
                    df_fix,

                "fixed_final":
                    fixed_final,

                "error":
                    calculated_error,
            }

            # =================================================
            # TRACKING
            #
            # CRITICAL:
            # Only execute this section when Tracking.
            # =================================================

            if plant_type == "Tracking":

                (
                    backend_list,
                    df_tracking,
                ) = load_tracking_data(
                    file_bytes
                )

                # ---------------------------------------------
                # EXPENSIVE OPTIMIZATION
                # ONLY runs here.
                # ---------------------------------------------

                tracking_result = (
                    optimize_tracking(
                        backend_list,
                        df_ghi,
                        df_fix,
                        df_w_final,
                    )
                )

                tracking_params = (
                    tracking_result[
                        "parameters"
                    ]
                )

                blocks = (
                    tracking_result[
                        "blocks"
                    ]
                )

                ghi_matrix = (
                    tracking_result[
                        "ghi_matrix"
                    ]
                )

                tracking_weights = (
                    tracking_result[
                        "tracking_weights"
                    ]
                )

                # ---------------------------------------------
                # FINAL TRACKING FORECAST
                # ---------------------------------------------

                tracking_result_final = (
                    calculate_tracking(
                        tracking_params["DHI"],
                        tracking_params[
                            "GHI Starting Block"
                        ],
                        tracking_params[
                            "GHI Ending Block"
                        ],
                        tracking_params[
                            "GHI Max Block"
                        ],
                        tracking_params[
                            "Tracking East Limit"
                        ],
                        tracking_params[
                            "Tracking West Limit"
                        ],
                        blocks,
                        ghi_matrix,
                        tracking_weights,
                    )
                )

                if tracking_result_final is None:

                    raise ValueError(
                        "Invalid tracking parameters."
                    )

                (
                    tracking_forecast,
                    tracking_power_matrix,
                    zenith,
                    panel,
                    dni,
                ) = tracking_result_final

                results.update(
                    {
                        "backend_list":
                            backend_list,

                        "df_tracking":
                            df_tracking,

                        "tracking_parameters":
                            tracking_params,

                        "blocks":
                            blocks,

                        "ghi_matrix":
                            ghi_matrix,

                        "tracking_weights":
                            tracking_weights,

                        "tracking_forecast":
                            tracking_forecast,

                        "tracking_power_matrix":
                            tracking_power_matrix,

                        "zenith":
                            zenith,

                        "panel":
                            panel,

                        "dni":
                            dni,
                    }
                )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            st.session_state.results = results
            st.session_state.calculated = True
            st.session_state.input_df = edited_input.copy()

        st.success(
            "Calculation completed successfully."
        )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )

        st.stop()


# ============================================================
# NOTHING TO DISPLAY UNTIL RUN
# ============================================================

if not st.session_state.calculated:

    st.info(
        "Edit GHI / Actual values and parameters, "
        "then click Run Automatic Calculation."
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.results


# ============================================================
# IMPORTANT:
# Parameter edits AFTER optimization
#
# We recalculate only the final forecast.
# We NEVER rerun Differential Evolution here.
# ============================================================

if plant_type == "Fixed":

    (
        df_final,
        df_w_final,
    ) = calculate_effective_area(
        results["df_original"],
        results["df_w_original"],
        error_value,
    )

    fixed_final = (
        calculate_fixed_power(
            results["df_fix"],
            df_w_final,
        )
    )

    actual = as_array(
        results["df_fix"]["Actual"]
    )

    forecast = as_array(
        fixed_final[
            "Total Power (CL1+CL2+…)"
        ]
    )

    title = (
        "Fixed Plant | Actual vs Forecast"
    )

else:

    tracking_params = {
        "DHI":
            int(dhi_value),

        "GHI Starting Block":
            int(start_value),

        "GHI Ending Block":
            int(end_value),

        "GHI Max Block":
            int(max_value),

        "Tracking East Limit":
            int(east_value),

        "Tracking West Limit":
            int(west_value),
    }

    (
        df_final,
        df_w_final,
    ) = calculate_effective_area(
        results["df_original"],
        results["df_w_original"],
        error_value,
    )

    tracking_weights = (
        pd.to_numeric(
            df_w_final.iloc[:5, 1],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    tracking_final = calculate_tracking(
        tracking_params["DHI"],
        tracking_params[
            "GHI Starting Block"
        ],
        tracking_params[
            "GHI Ending Block"
        ],
        tracking_params[
            "GHI Max Block"
        ],
        tracking_params[
            "Tracking East Limit"
        ],
        tracking_params[
            "Tracking West Limit"
        ],
        results["blocks"],
        results["ghi_matrix"],
        tracking_weights,
    )

    if tracking_final is None:

        st.error(
            "Invalid Tracking parameters. "
            "GHI Starting Block < GHI Max Block < GHI Ending Block is required."
        )

        st.stop()

    (
        forecast,
        _,
        _,
        _,
        _,
    ) = tracking_final

    actual = as_array(
        results["df_fix"]["Actual"]
    )

    title = (
        "Tracking Plant | Actual vs Forecast"
    )


# ============================================================
# RESULT METRICS
# ============================================================

metrics = calculate_metrics(
    actual,
    forecast,
)

st.markdown(
    '<div class="section-title">📊 Results</div>',
    unsafe_allow_html=True,
)

m1, m2 = st.columns(2)

with m1:

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">
                Actual Peak
            </div>
            <div class="result-value">
                {metrics["Actual Peak"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">
                Forecast Peak
            </div>
            <div class="result-value">
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
)
