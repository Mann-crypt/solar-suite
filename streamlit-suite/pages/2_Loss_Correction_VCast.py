# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# CLEAN STREAMLIT UI
#
# CALCULATION LOGIC PRESERVED
# ERROR % IS APPLIED ONLY ONCE
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
    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .section {
        font-size: 19px;
        font-weight: 650;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 15px 17px;
        min-height: 82px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 23px;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 6px;
    }

    .stButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 600;
    }

    div[data-testid="stRadio"] {
        margin-bottom: -5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_TRACKING = {
    "DHI": 0,
    "GHI Starting Block": 20,
    "GHI Ending Block": 72,
    "GHI Max Block": 50,
    "Tracking East Limit": 30,
    "Tracking West Limit": 30,
}

if "calculated" not in st.session_state:
    st.session_state.calculated = False

if "data" not in st.session_state:
    st.session_state.data = None

if "file_key" not in st.session_state:
    st.session_state.file_key = None

if "plant_type" not in st.session_state:
    st.session_state.plant_type = "Fixed"

if "error_value" not in st.session_state:
    st.session_state.error_value = 0.0

for key, value in DEFAULT_TRACKING.items():
    state_key = f"param_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Automatic parameter optimization with editable correction parameters'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section">Input File</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Solar Excel File",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("Upload the Excel file to start.")
    st.stop()


# ============================================================
# FILE CHANGE DETECTION
# ============================================================

file_key = (
    uploaded_file.name,
    uploaded_file.size,
)

if file_key != st.session_state.file_key:
    st.session_state.file_key = file_key
    st.session_state.calculated = False
    st.session_state.data = None


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section">Plant Type</div>',
    unsafe_allow_html=True,
)

plant_type = st.radio(
    "Plant Type",
    ["Fixed", "Tracking"],
    horizontal=True,
    label_visibility="collapsed",
    key="plant_type",
)


# ============================================================
# EXCEL READER
# ============================================================

def read_excel(file, **kwargs):
    file.seek(0)
    return pd.read_excel(file, **kwargs)


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def load_area_efficiency(file):

    df = read_excel(
        file,
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
        idx = df[df["S.No."].isna()].index
        if len(idx):
            df = df.iloc[:df.index.get_loc(idx[0])].copy()

    for col in [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df


# ============================================================
# CLUSTER AREA
# ============================================================

def load_cluster_table(file):

    df = read_excel(
        file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Clusters" in df.columns:
        idx = df[df["Clusters"].isna()].index
        if len(idx):
            df = df.iloc[:df.index.get_loc(idx[0])].copy()

    return df.reset_index(drop=True)


# ============================================================
# GHI
# ============================================================

def load_ghi(file):

    df = read_excel(
        file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    ).fillna(0)

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


# ============================================================
# LATITUDE
# ============================================================

def load_latitude(file):

    df = read_excel(
        file,
        sheet_name="Forecast Config",
        header=8,
    )

    return float(
        pd.to_numeric(
            df.loc[0, "Lat"],
            errors="coerce",
        )
    )


# ============================================================
# TILT
# ============================================================

def load_tilt(file):

    df = read_excel(
        file,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Fixed" in df.columns:
        idx = df[df["Fixed"].isna()].index
        if len(idx):
            df = df.iloc[:df.index.get_loc(idx[0])].copy()

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


# ============================================================
# FIXED DATA
# ============================================================

def load_fixed_data(file):

    df = read_excel(
        file,
        sheet_name="Fixed-C11",
        header=1,
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Date" in df.columns:
        idx = df[df["Date"].isna()].index
        if len(idx):
            df = df.iloc[:df.index.get_loc(idx[0])].copy()

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    return df.reset_index(drop=True)


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
                    + (df["Date"] - first_date).dt.days
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

    denominator = df["Sin(a)"].replace(0, np.nan)

    for cluster, col_a, col_ab, poa_col in clusters:

        ghi = df_ghi[f"GHI {cluster}"]

        df[col_a] = ghi * df["Sin(a)"]
        df[col_ab] = ghi * df["SIN(a+b)"]
        df[poa_col] = df[col_ab] / denominator

    return df


# ============================================================
# EFFECTIVE AREA
#
# ERROR % APPLIED ONLY HERE
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
            * df_w.iloc[i]["Eff Area(m2)"]
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

    actual = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

    actual_peak = actual.max()

    if actual_peak <= 0:
        raise ValueError(
            "No non-zero Actual values found."
        )

    rows = []

    for error in np.arange(0, 10.01, 0.1):

        _, df_w = calculate_effective_area(
            df_original,
            df_w_original,
            error,
        )

        calculated = calculate_fixed_power(
            df_fix,
            df_w,
        )

        calculated_peak = calculated[
            "Total Power (CL1+CL2+…)"
        ].max()

        peak_error = abs(
            calculated_peak
            - actual_peak
        )

        rows.append(
            {
                "Error %": round(error, 1),
                "Peak Error": peak_error,
            }
        )

    result_df = pd.DataFrame(rows)

    best_error = float(
        result_df.loc[
            result_df["Peak Error"].idxmin(),
            "Error %",
        ]
    )

    return best_error, result_df


# ============================================================
# TRACKING DATA
# ============================================================

def load_tracking_data(file):

    backend_list = []

    for cluster in [
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
    ]:

        backend_list.append(
            read_excel(
                file,
                sheet_name=f"Backend Cal {cluster}",
            )
        )

    df_tracking = read_excel(
        file,
        sheet_name="Tracking",
        header=1,
    )

    df_tracking.columns = (
        df_tracking.columns.astype(str)
        .str.strip()
    )

    return backend_list, df_tracking


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

    cl_weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    ghi_matrix = np.column_stack(
        [
            pd.to_numeric(
                df_ghi[col],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(dtype=float)
            for col in ghi_cols
        ]
    )

    blocks = pd.to_numeric(
        backend_list[0]["Block No."],
        errors="coerce",
    ).to_numpy(dtype=float)

    actual_full = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0).to_numpy(dtype=float)

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
        maximum = int(round(x[3]))
        east = int(round(x[4]))
        west = int(round(x[5]))

        if not (
            start < maximum < end
        ):
            return 1e9

        d1 = start - 1 - maximum
        d2 = end + 1 - maximum

        if d1 == 0 or d2 == 0:
            return 1e9

        m1 = 90 / d1
        m2 = 90 / d2

        zenith = np.where(
            blocks <= maximum,
            np.minimum(
                89,
                m1 * (blocks - maximum),
            ),
            np.minimum(
                89,
                m2 * (blocks - maximum),
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
                    & (zenith > west)
                ),
                west,
                zenith,
            ),
        )

        cos_alpha = np.clip(
            np.cos(np.radians(panel)),
            1e-6,
            None,
        )

        dhi = ghi_matrix * DHI / 100

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

        if len(prediction) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(actual - prediction)
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
    maximum,
    east,
    west,
):

    d1 = start - 1 - maximum
    d2 = end + 1 - maximum

    if d1 == 0:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    if d2 == 0:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    m1 = 90 / d1
    m2 = 90 / d2

    zenith = np.where(
        blocks <= maximum,
        np.minimum(
            89,
            m1 * (blocks - maximum),
        ),
        np.minimum(
            89,
            m2 * (blocks - maximum),
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
                & (zenith > west)
            ),
            west,
            zenith,
        ),
    )

    cos_alpha = np.clip(
        np.cos(np.radians(panel)),
        1e-6,
        None,
    )

    dhi = ghi_matrix * DHI / 100

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

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    return {
        "Actual Peak": np.max(actual),
        "Forecast Peak": np.max(forecast),
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

    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual[:n],
            mode="lines",
            name="Actual",
            line=dict(width=2.3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast[:n],
            mode="lines",
            name="Forecast",
            line=dict(width=2.3),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
        ),
        height=430,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(
            l=30,
            r=30,
            t=55,
            b=30,
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
# INITIAL CALCULATION
# ============================================================

if not st.session_state.calculated:

    if st.button(
        "⚡ Run Automatic Calculation",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Running automatic optimization..."
            ):

                # -------------------------------
                # LOAD DATA
                # -------------------------------

                df_original = load_area_efficiency(
                    uploaded_file
                )

                df_w_original = load_cluster_table(
                    uploaded_file
                )

                df_ghi = load_ghi(
                    uploaded_file
                )

                lat = load_latitude(
                    uploaded_file
                )

                month_lookup = load_tilt(
                    uploaded_file
                )

                df_fix_raw = load_fixed_data(
                    uploaded_file
                )

                df_fix = prepare_fixed_geometry(
                    df_fix_raw,
                    df_ghi,
                    lat,
                    month_lookup,
                )

                # -------------------------------
                # AUTOMATIC ERROR
                # -------------------------------

                best_error, error_results = (
                    optimize_error(
                        df_original,
                        df_w_original,
                        df_fix,
                    )
                )

                # -------------------------------
                # APPLY ERROR ONCE
                # -------------------------------

                df_final, df_w_final = (
                    calculate_effective_area(
                        df_original,
                        df_w_original,
                        best_error,
                    )
                )

                # -------------------------------
                # FIXED
                # -------------------------------

                fixed_final = calculate_fixed_power(
                    df_fix,
                    df_w_final,
                )

                # -------------------------------
                # TRACKING
                # -------------------------------

                (
                    backend_list,
                    df_tracking,
                ) = load_tracking_data(
                    uploaded_file
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

                # -------------------------------
                # TRACKING FORECAST
                # -------------------------------

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

                # -------------------------------
                # SAVE
                # -------------------------------

                st.session_state.data = {
                    "df_original": df_original,
                    "df_w_original": df_w_original,
                    "df_final": df_final,
                    "df_w_final": df_w_final,
                    "df_ghi": df_ghi,
                    "df_fix": df_fix,
                    "fixed_final": fixed_final,
                    "backend_list": backend_list,
                    "df_tracking": df_tracking,
                    "blocks": blocks,
                    "ghi_matrix": ghi_matrix,
                    "actual_tracking": actual_tracking,
                    "cl_weights": cl_weights,
                    "best_error": best_error,
                    "tracking_parameters": tracking_parameters,
                    "tracking_score": tracking_score,
                    "tracking_forecast": tracking_forecast,
                }

                st.session_state.error_value = (
                    float(best_error)
                )

                for key, value in tracking_parameters.items():
                    st.session_state[
                        f"param_{key}"
                    ] = value

                st.session_state.calculated = True

            st.rerun()

        except Exception as e:

            st.error(
                f"Calculation failed: {e}"
            )

    st.info(
        "Upload the Excel file and run the automatic calculation."
    )

    st.stop()


# ============================================================
# DATA
# ============================================================

data = st.session_state.data


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)


# ============================================================
# ERROR %
# ============================================================

error_value = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=20.0,
    step=0.1,
    format="%.1f",
    key="error_value",
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

if plant_type == "Tracking":

    st.markdown(
        "##### Tracking Parameters"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            step=1,
            key="param_DHI",
        )

        st.number_input(
            "GHI Starting Block",
            min_value=0,
            max_value=95,
            step=1,
            key="param_GHI Starting Block",
        )

    with c2:

        st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            step=1,
            key="param_GHI Ending Block",
        )

        st.number_input(
            "GHI Max Block",
            min_value=0,
            max_value=95,
            step=1,
            key="param_GHI Max Block",
        )

    with c3:

        st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            step=1,
            key="param_Tracking East Limit",
        )

        st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            step=1,
            key="param_Tracking West Limit",
        )


# ============================================================
# AUTOMATIC PARAMETER UPDATE
# ============================================================

# Recalculate using edited values.
#
# No Apply button.
# No differential evolution here.
# Only the final forecast is recalculated.

df_final, df_w_final = calculate_effective_area(
    data["df_original"],
    data["df_w_original"],
    float(error_value),
)

fixed_final = calculate_fixed_power(
    data["df_fix"],
    df_w_final,
)

if plant_type == "Tracking":

    start = int(
        st.session_state["param_GHI Starting Block"]
    )

    end = int(
        st.session_state["param_GHI Ending Block"]
    )

    maximum = int(
        st.session_state["param_GHI Max Block"]
    )

    dhi = int(
        st.session_state["param_DHI"]
    )

    east = int(
        st.session_state["param_Tracking East Limit"]
    )

    west = int(
        st.session_state["param_Tracking West Limit"]
    )

    # --------------------------------------------
    # Validate editable parameters
    # --------------------------------------------

    if not (
        start < maximum < end
    ):

        st.warning(
            "Tracking parameters must satisfy: "
            "GHI Starting Block < GHI Max Block < GHI Ending Block."
        )

        st.stop()

    tracking_weights = (
        pd.to_numeric(
            df_w_final.iloc[:5, 1],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    tracking_forecast = calculate_tracking_forecast(
        data["blocks"],
        data["ghi_matrix"],
        tracking_weights,
        dhi,
        start,
        end,
        maximum,
        east,
        west,
    )

    actual = data["actual_tracking"]
    forecast = tracking_forecast

    graph_title = (
        "Tracking Plant | Actual vs Forecast"
    )

else:

    actual = (
        pd.to_numeric(
            data["df_fix"]["Actual"],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy()
    )

    forecast = (
        fixed_final[
            "Total Power (CL1+CL2+…)"
        ]
        .fillna(0)
        .to_numpy()
    )

    graph_title = (
        "Fixed Plant | Actual vs Forecast"
    )


# ============================================================
# RESULTS
# ============================================================

metrics = calculate_metrics(
    actual,
    forecast,
)

st.markdown(
    '<div class="section">📊 Results</div>',
    unsafe_allow_html=True,
)

m1, m2 = st.columns(2)

with m1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Actual Peak
            </div>
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
            <div class="metric-label">
                Forecast Peak
            </div>
            <div class="metric-value">
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
    '<div class="section">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)

st.plotly_chart(
    build_graph(
        actual,
        forecast,
        graph_title,
    ),
    use_container_width=True,
    config={
        "displayModeBar": False,
        "responsive": True,
    },
)


# ============================================================
# CURRENT PARAMETERS SUMMARY
# ============================================================

with st.expander(
    "View calculated parameters"
):

    if plant_type == "Fixed":

        st.write(
            f"**Error %:** {error_value:.1f}%"
        )

    else:

        st.write(
            f"**Error %:** {error_value:.1f}%"
        )

        st.write(
            f"**DHI:** {dhi}%"
        )

        st.write(
            f"**GHI Starting Block:** {start}"
        )

        st.write(
            f"**GHI Max Block:** {maximum}"
        )

        st.write(
            f"**GHI Ending Block:** {end}"
        )

        st.write(
            f"**Tracking East Limit:** {east}°"
        )

        st.write(
            f"**Tracking West Limit:** {west}°"
        )
