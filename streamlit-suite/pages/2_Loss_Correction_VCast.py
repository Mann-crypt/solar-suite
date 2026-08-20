# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# CLEAN + COMPACT STREAMLIT PAGE
#
# Error % is applied exactly ONCE.
# GHI + Actual are user-editable.
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

    .app-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 650;
        margin: 18px 0 10px 0;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 16px;
        min-height: 82px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 23px;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 6px;
        background: white;
    }

    div[data-testid="stDataEditor"] {
        border-radius: 10px;
        overflow: hidden;
    }

    .stButton > button {
        min-height: 40px;
        border-radius: 8px;
        font-weight: 600;
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

POA_COLS = [
    "POA fixed",
    "POA Fixed-C12",
    "POA Fixed-C13",
    "POA Fixed-C14",
    "POA Fixed-C15",
]

PARAM_BOUNDS = [
    (0, 10),    # DHI
    (10, 30),   # GHI Starting Block
    (65, 80),   # GHI Ending Block
    (47, 53),   # GHI Max Block
    (10, 70),   # Tracking East
    (10, 70),   # Tracking West
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "calculated": False,
    "data": None,
    "input_data": None,
    "last_file_name": None,
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
    "Fixed / Tracking forecast correction with editable GHI, Actual and parameters"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def numeric(value):
    """
    Safely convert Series / ndarray / list / scalar to float ndarray.

    Fixes:
        AttributeError:
        'numpy.ndarray' object has no attribute 'fillna'
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
        pd.Series(arr.ravel()),
        errors="coerce",
    ).fillna(0).to_numpy(dtype=float)


def read_excel(uploaded_file, **kwargs):
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, **kwargs)


def safe_divide(a, b):
    return np.divide(
        a,
        b,
        out=np.zeros_like(
            np.asarray(a, dtype=float),
            dtype=float,
        ),
        where=np.abs(b) > 1e-12,
    )


# ============================================================
# LOAD AREA + EFFICIENCY
# ============================================================

def load_area_efficiency(uploaded_file):

    df = read_excel(
        uploaded_file,
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
        idx = df["S.No."].isna()

        if idx.any():
            df = df.iloc[
                :df.index.get_loc(idx.idxmax())
            ].copy()

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

    return df.reset_index(drop=True)


# ============================================================
# CLUSTER TABLE
# ============================================================

def load_cluster_table(uploaded_file):

    df = read_excel(
        uploaded_file,
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

        valid = df["Clusters"].notna()

        if valid.any():
            df = df.iloc[
                :valid.sum()
            ].copy()

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

    for col in GHI_COLS:

        if col not in df.columns:
            raise ValueError(
                f"Missing column: {col}"
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

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
            "Invalid latitude."
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

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    df = df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    if "Fixed" not in df.columns:
        raise ValueError(
            "Fixed tilt column not found."
        )

    if "Month" not in df.columns:
        raise ValueError(
            "Month column not found."
        )

    df["Fixed"] = pd.to_numeric(
        df["Fixed"],
        errors="coerce",
    )

    return (
        df.dropna(subset=["Month"])
        .set_index("Month")["Fixed"]
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

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "Date" not in df.columns:
        raise ValueError(
            "Date column not found in Fixed-C11."
        )

    if "Actual" not in df.columns:
        raise ValueError(
            "Actual column not found in Fixed-C11."
        )

    valid = df["Date"].notna()

    if valid.any():
        df = df.iloc[
            :valid.sum()
        ].copy()

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

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

    today = pd.Timestamp.today().normalize()

    df["Date"] = today

    first_date = pd.Timestamp(
        year=today.year,
        month=1,
        day=1,
    )

    day_of_year = (
        df["Date"] - first_date
    ).dt.days + 1

    df["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_of_year
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
        .fillna(0)
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

    # --------------------------------------------------------
    # Each cluster
    # --------------------------------------------------------

    for i, ghi_col in enumerate(GHI_COLS):

        suffix = "" if i == 0 else f"-CL{i + 1}"

        sin_a_col = (
            "GHI*sin(a)"
            if i == 0
            else f"GHI*sin(a)-CL{i + 1}"
        )

        sin_ab_col = (
            "GHI*sin(a+b)"
            if i == 0
            else f"GHI*sin(a+b)-CL{i + 1}"
        )

        poa_col = POA_COLS[i]

        ghi = numeric(
            df_ghi[ghi_col]
        )

        df[sin_a_col] = (
            ghi
            * df["Sin(a)"].to_numpy()
        )

        df[sin_ab_col] = (
            ghi
            * df["SIN(a+b)"].to_numpy()
        )

        df[poa_col] = safe_divide(
            df[sin_ab_col].to_numpy(),
            df["Sin(a)"].to_numpy(),
        )

    return df


# ============================================================
# EFFECTIVE AREA
#
# Error % applied exactly ONCE.
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
        numeric(
            df["Standard PV Efficiency (%)"]
        )
        - float(error)
    )

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * numeric(
            df["Total area (m2)"]
        )
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

    areas = numeric(
        df_w.iloc[:5]["Eff Area(m2)"]
    )

    for i, poa_col in enumerate(POA_COLS):

        power_col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        poa = numeric(
            df[poa_col]
        )

        df[power_col] = (
            poa
            * areas[i]
            / 1_000_000
        )

        power_cols.append(power_col)

    df["Total Power (CL1+CL2+…)"] = (
        df[power_cols]
        .sum(axis=1)
        .fillna(0)
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

    best_error = 0
    best_peak_error = np.inf

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

        forecast = numeric(
            calculated[
                "Total Power (CL1+CL2+…)"
            ]
        )

        forecast_peak = forecast.max()

        peak_error = abs(
            forecast_peak
            - actual_peak
        )

        if peak_error < best_peak_error:

            best_peak_error = peak_error
            best_error = float(
                round(error, 1)
            )

    return best_error


# ============================================================
# TRACKING DATA
# ============================================================

def load_tracking_data(
    uploaded_file,
):

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
                uploaded_file,
                sheet_name=f"Backend Cal {cluster}",
            )
        )

    df_tracking = read_excel(
        uploaded_file,
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
# TRACKING OBJECTIVE
# ============================================================

def create_tracking_objective(
    backend_list,
    df_ghi,
    actual,
    df_w,
):

    ghi_matrix = np.column_stack(
        [
            numeric(df_ghi[col])
            for col in GHI_COLS
        ]
    )

    blocks = numeric(
        backend_list[0]["Block No."]
    )

    actual_full = numeric(
        actual
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
            "No non-zero Actual values found "
            "for Tracking."
        )

    actual_values = actual_full[mask]

    actual_max = actual_values.max()
    actual_sum = actual_values.sum()

    # Error % already applied.
    cl_weights = numeric(
        df_w.iloc[:5]["Eff Area(m2)"]
    )

    def objective(x):

        DHI = int(round(x[0]))
        start = int(round(x[1]))
        end = int(round(x[2]))
        max_block = int(round(x[3]))
        east = int(round(x[4]))
        west = int(round(x[5]))

        if not (
            start
            < max_block
            < end
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

        if denominator_1 == 0:
            return 1e9

        if denominator_2 == 0:
            return 1e9

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
                abs(east),
            ),

            np.where(
                (
                    (blocks > max_block)
                    &
                    (
                        zenith
                        > west
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
            np.isnan(
                prediction_full
            ).any()
            or np.isinf(
                prediction_full
            ).any()
        ):
            return 1e9

        prediction = (
            prediction_full[mask]
        )

        if len(prediction) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual_values
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
    actual,
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
        actual,
        df_w,
    )

    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
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

    if denominator_1 == 0:
        raise ValueError(
            "Invalid Tracking parameters: "
            "East denominator is zero."
        )

    if denominator_2 == 0:
        raise ValueError(
            "Invalid Tracking parameters: "
            "West denominator is zero."
        )

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
            abs(east),
        ),

        np.where(
            (
                (blocks > max_block)
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

    n = min(
        len(actual),
        len(forecast),
    )

    actual = actual[:n]
    forecast = forecast[:n]

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
                width=2.2,
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
                width=2.2,
            ),
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.01,
        },
        height=450,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
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
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Input File</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Solar Excel File",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)

if uploaded_file is None:

    st.info(
        "Upload the Excel file to start."
    )

    st.stop()


# ============================================================
# RESET WHEN NEW FILE IS UPLOADED
# ============================================================

if (
    st.session_state.last_file_name
    != uploaded_file.name
):

    st.session_state.calculated = False
    st.session_state.data = None
    st.session_state.input_data = None
    st.session_state.last_file_name = (
        uploaded_file.name
    )


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">Plant Type</div>',
    unsafe_allow_html=True,
)

plant_type = st.segmented_control(
    "Plant Type",
    options=[
        "Fixed",
        "Tracking",
    ],
    default="Fixed",
    label_visibility="collapsed",
)


# ============================================================
# LOAD INPUT DATA
# ============================================================

try:

    if st.session_state.input_data is None:

        ghi_df = load_ghi(
            uploaded_file
        )

        fixed_df = load_fixed_data(
            uploaded_file
        )

        n = min(
            len(ghi_df),
            len(fixed_df),
        )

        input_df = pd.DataFrame(
            {
                col: numeric(
                    ghi_df[col]
                )[:n]
                for col in GHI_COLS
            }
        )

        input_df["Actual"] = numeric(
            fixed_df["Actual"]
        )[:n]

        st.session_state.input_data = (
            input_df
        )

except Exception as e:

    st.error(
        f"Unable to load input data: {e}"
    )

    st.stop()


# ============================================================
# EDITABLE INPUT DATA
#
# Wrapped inside FORM so editing cells does not
# constantly rerun the application.
# ============================================================

st.markdown(
    '<div class="section-title">Input Data</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Edit GHI and Actual values below. "
    "Changes are used when you press Run Calculation."
)

with st.form(
    "input_data_form",
    clear_on_submit=False,
):

    edited_input = st.data_editor(
        st.session_state.input_data,
        use_container_width=True,
        height=310,
        num_rows="fixed",
        hide_index=True,
        column_config={
            col: st.column_config.NumberColumn(
                col,
                format="%.3f",
            )
            for col in GHI_COLS
        }
        | {
            "Actual": st.column_config.NumberColumn(
                "Actual",
                format="%.3f",
            )
        },
        key="solar_input_editor",
    )

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

        with st.spinner(
            "Running calculation..."
        ):

            # ------------------------------------------------
            # SAVE USER INPUT
            # ------------------------------------------------

            edited_input = edited_input.copy()

            for col in GHI_COLS + ["Actual"]:

                edited_input[col] = pd.to_numeric(
                    edited_input[col],
                    errors="coerce",
                ).fillna(0)

            st.session_state.input_data = (
                edited_input
            )

            # ------------------------------------------------
            # LOAD STATIC WORKBOOK DATA
            # ------------------------------------------------

            df_original = (
                load_area_efficiency(
                    uploaded_file
                )
            )

            df_w_original = (
                load_cluster_table(
                    uploaded_file
                )
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

            # ------------------------------------------------
            # REPLACE GHI + ACTUAL WITH USER INPUT
            # ------------------------------------------------

            n = min(
                len(df_fix_raw),
                len(edited_input),
            )

            df_fix_raw = (
                df_fix_raw
                .iloc[:n]
                .copy()
            )

            user_input = (
                edited_input
                .iloc[:n]
                .copy()
            )

            df_ghi = user_input[
                GHI_COLS
            ].copy()

            df_fix_raw["Actual"] = (
                user_input["Actual"]
                .to_numpy()
            )

            # ------------------------------------------------
            # GEOMETRY
            # ------------------------------------------------

            df_fix = prepare_fixed_geometry(
                df_fix_raw,
                df_ghi,
                lat,
                month_lookup,
            )

            # ------------------------------------------------
            # AUTOMATIC ERROR
            # ------------------------------------------------

            best_error = optimize_error(
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
            # FIXED FORECAST
            # ------------------------------------------------

            fixed_final = (
                calculate_fixed_power(
                    df_fix,
                    df_w_final,
                )
            )

            # ------------------------------------------------
            # TRACKING
            # ------------------------------------------------

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
            ) = optimize_tracking(
                backend_list,
                df_ghi,
                df_fix["Actual"],
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

            st.session_state.data = {

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

                "backend_list":
                    backend_list,

                "df_tracking":
                    df_tracking,

                "blocks":
                    blocks,

                "ghi_matrix":
                    ghi_matrix,

                "actual_tracking":
                    actual_tracking,

                "cl_weights":
                    cl_weights,

                "best_error":
                    best_error,

                "tracking_parameters":
                    tracking_parameters,

                "tracking_forecast":
                    tracking_forecast,
            }

            st.session_state.calculated = True

        st.success(
            "Calculation completed successfully."
        )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )

        st.stop()


# ============================================================
# STOP BEFORE FIRST CALCULATION
# ============================================================

if not st.session_state.calculated:

    st.info(
        "Edit the input data if required, then click "
        "**Run Calculation**."
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
    '<div class="section-title">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# ERROR
# ------------------------------------------------------------

p1, p2, p3 = st.columns(3)

with p1:

    error_value = st.number_input(
        "Error %",
        min_value=0.0,
        max_value=20.0,
        value=float(
            data["best_error"]
        ),
        step=0.1,
        format="%.1f",
    )


# ------------------------------------------------------------
# TRACKING PARAMETERS
# ------------------------------------------------------------

if plant_type == "Tracking":

    params = data[
        "tracking_parameters"
    ]

    with p2:

        dhi_value = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(
                params["DHI"]
            ),
            step=1,
        )

        start_value = st.number_input(
            "GHI Starting Block",
            min_value=0,
            max_value=95,
            value=int(
                params[
                    "GHI Starting Block"
                ]
            ),
            step=1,
        )

        end_value = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                params[
                    "GHI Ending Block"
                ]
            ),
            step=1,
        )

    with p3:

        max_value = st.number_input(
            "GHI Max Block",
            min_value=0,
            max_value=95,
            value=int(
                params[
                    "GHI Max Block"
                ]
            ),
            step=1,
        )

        east_value = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            value=int(
                params[
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
                params[
                    "Tracking West Limit"
                ]
            ),
            step=1,
        )


# ============================================================
# LIVE RECALCULATION FROM PARAMETERS
# ============================================================

try:

    # --------------------------------------------------------
    # Error % applied ONCE
    # --------------------------------------------------------

    (
        df_final,
        df_w_final,
    ) = calculate_effective_area(
        data["df_original"],
        data["df_w_original"],
        error_value,
    )

    # --------------------------------------------------------
    # Fixed
    # --------------------------------------------------------

    fixed_final = calculate_fixed_power(
        data["df_fix"],
        df_w_final,
    )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    if plant_type == "Tracking":

        if not (
            start_value
            < max_value
            < end_value
        ):
            st.error(
                "Invalid Tracking parameters. "
                "GHI Starting Block < GHI Max Block < GHI Ending Block."
            )
            st.stop()

        tracking_areas = numeric(
            df_w_final.iloc[:5][
                "Eff Area(m2)"
            ]
        )

        tracking_forecast = (
            calculate_tracking_forecast(
                data["blocks"],
                data["ghi_matrix"],
                tracking_areas,
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
# FINAL FORECAST
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
        data["df_fix"]["Actual"]
    )

    forecast = numeric(
        tracking_forecast
    )

    title = (
        "Tracking Plant | Actual vs Forecast"
    )


# ============================================================
# RESULTS
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
    '<div class="section-title">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)

st.plotly_chart(
    build_graph(
        actual,
        forecast,
        title,
    ),
    use_container_width=True,
    config={
        "displayModeBar": False,
        "responsive": True,
    },
)
