# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# Streamlit Page
#
# Calculation logic preserved.
# Error % is applied exactly once.
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
        padding: 1.2rem 2rem 2rem;
        max-width: 1550px;
    }

    .title {
        font-size: 30px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .section {
        font-size: 19px;
        font-weight: 700;
        margin: 18px 0 10px;
    }

    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,.04);
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 3px;
    }

    .metric-value {
        font-size: 22px;
        font-weight: 750;
    }

    div[data-testid="stDataEditor"] {
        border-radius: 10px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 650;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 8px;
    }

    div[data-testid="stRadio"] > div {
        gap: 8px;
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
    "input_df": None,
    "file_name": None,
    "plant_type": "Fixed",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
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

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]


def numeric(x):
    """
    Safe numeric conversion for Series, Index, ndarray and lists.
    """
    return np.nan_to_num(
        pd.to_numeric(
            np.asarray(x).reshape(-1),
            errors="coerce",
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def read_excel(file, **kwargs):
    file.seek(0)
    return pd.read_excel(file, **kwargs)


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

    idx = df[df[column].isna()].index

    if len(idx):
        return df.iloc[:df.index.get_loc(idx[0])].copy()

    return df


# ============================================================
# INPUT FILE
# ============================================================

st.markdown(
    '<div class="title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Fixed and Tracking forecast correction with editable inputs and parameters"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section">📁 Input Data</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Solar Excel File",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)


if uploaded_file is None:
    st.info("Upload the solar Excel file to begin.")
    st.stop()


# ============================================================
# LOAD INPUT DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_input_data(file_bytes):

    import io

    file = io.BytesIO(file_bytes)

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    df_original = pd.read_excel(
        file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df_original = clean_columns(df_original)
    df_original = trim_at_first_null(
        df_original,
        "S.No.",
    )

    for col in [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]:
        if col in df_original.columns:
            df_original[col] = pd.to_numeric(
                df_original[col],
                errors="coerce",
            )

    df_original["Total area (m2)"] = (
        df_original["No of Module"]
        * df_original["Area of 1 Module (m2)"]
    )

    # --------------------------------------------------------
    # Cluster table
    # --------------------------------------------------------

    df_w_original = pd.read_excel(
        file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df_w_original = clean_columns(
        df_w_original
    )

    df_w_original = trim_at_first_null(
        df_w_original,
        "Clusters",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    df_ghi = pd.read_excel(
        file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df_ghi = df_ghi.fillna(0)

    for col in GHI_COLS:
        if col in df_ghi.columns:
            df_ghi[col] = numeric(
                df_ghi[col]
            )

    # --------------------------------------------------------
    # Latitude
    # --------------------------------------------------------

    df_config = pd.read_excel(
        file,
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
        file,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df_tilt = clean_columns(df_tilt)
    df_tilt = trim_at_first_null(
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

    month_lookup = (
        df_tilt
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # Fixed data
    # --------------------------------------------------------

    df_fix = pd.read_excel(
        file,
        sheet_name="Fixed-C11",
        header=1,
    )

    df_fix = clean_columns(df_fix)
    df_fix = trim_at_first_null(
        df_fix,
        "Date",
    )

    df_fix["Actual"] = numeric(
        df_fix["Actual"]
    )

    # --------------------------------------------------------
    # Tracking backend
    # --------------------------------------------------------

    backend_list = []

    for cluster in CLUSTERS:

        backend = pd.read_excel(
            file,
            sheet_name=f"Backend Cal {cluster}",
        )

        backend_list.append(
            backend
        )

    df_tracking = pd.read_excel(
        file,
        sheet_name="Tracking",
        header=1,
    )

    df_tracking = clean_columns(
        df_tracking
    )

    return (
        df_original,
        df_w_original,
        df_ghi,
        lat,
        month_lookup,
        df_fix,
        backend_list,
        df_tracking,
    )


(
    df_original,
    df_w_original,
    df_ghi_original,
    lat,
    month_lookup,
    df_fix_original,
    backend_list,
    df_tracking,
) = load_input_data(
    uploaded_file.getvalue()
)


# ============================================================
# EDITABLE INPUT DATA
# ============================================================

input_df = pd.DataFrame(
    {
        "Actual": numeric(
            df_fix_original["Actual"]
        )
    }
)

for col in GHI_COLS:
    input_df[col] = numeric(
        df_ghi_original[col]
    )


# Keep original input data in session
if (
    st.session_state.input_df is None
    or st.session_state.file_name
    != uploaded_file.name
):

    st.session_state.input_df = (
        input_df.copy()
    )

    st.session_state.file_name = (
        uploaded_file.name
    )

    st.session_state.calculated = False
    st.session_state.calculation_data = None


# ============================================================
# HORIZONTAL INPUT EDITOR
# ============================================================

edited_input = st.data_editor(
    st.session_state.input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    height=260,
    column_config={
        "Actual": st.column_config.NumberColumn(
            "Actual Power",
            format="%.4f",
        ),
        **{
            col: st.column_config.NumberColumn(
                col,
                format="%.4f",
            )
            for col in GHI_COLS
        },
    },
    key="solar_input_editor",
)

st.session_state.input_df = edited_input


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section">🌱 Plant Type</div>',
    unsafe_allow_html=True,
)

plant_type = st.radio(
    "Plant Type",
    ["Fixed", "Tracking"],
    horizontal=True,
    index=(
        0
        if st.session_state.plant_type
        == "Fixed"
        else 1
    ),
    label_visibility="collapsed",
)

st.session_state.plant_type = plant_type


# ============================================================
# BUILD CURRENT INPUT DATA
# ============================================================

df_ghi = edited_input[
    GHI_COLS
].copy()

df_ghi = df_ghi.apply(
    pd.to_numeric,
    errors="coerce",
).fillna(0)

df_fix_raw = df_fix_original.copy()

df_fix_raw["Actual"] = numeric(
    edited_input["Actual"]
)

# Make sure lengths match
n = min(
    len(df_fix_raw),
    len(df_ghi),
)

df_fix_raw = df_fix_raw.iloc[
    :n
].reset_index(drop=True)

df_ghi = df_ghi.iloc[
    :n
].reset_index(drop=True)


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
        np.radians(
            df["a+b"]
        )
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

    mapping = {
        "C11": (
            "GHI*sin(a)",
            "GHI*sin(a+b)",
            "POA fixed",
        ),
        "C12": (
            "GHI*sin(a)-CL2",
            "GHI*sin(a+b)-CL2",
            "POA Fixed-C12",
        ),
        "C13": (
            "GHI*sin(a)-CL3",
            "GHI*sin(a+b)-CL3",
            "POA Fixed-C13",
        ),
        "C14": (
            "GHI*sin(a)-CL4",
            "GHI*sin(a+b)-CL4",
            "POA Fixed-C14",
        ),
        "C15": (
            "GHI*sin(a)-CL5",
            "GHI*sin(a+b)-CL5",
            "POA Fixed-C15",
        ),
    }

    for i, cluster in enumerate(
        CLUSTERS
    ):

        a_col, ab_col, poa_col = (
            mapping[cluster]
        )

        ghi = numeric(
            df_ghi[
                GHI_COLS[i]
            ]
        )

        df[a_col] = (
            ghi
            * df["Sin(a)"]
        )

        df[ab_col] = (
            ghi
            * df["SIN(a+b)"]
        )

        df[poa_col] = (
            df[ab_col]
            / sin_a
        )

    return df


df_fix = prepare_fixed_geometry(
    df_fix_raw,
    df_ghi,
    lat,
    month_lookup,
)


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
        df[
            "Standard PV Efficiency (%)"
        ]
        - float(error)
    )

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

    cluster_sums = (
        df.groupby("Clusters")[
            "Eff Area"
        ].sum()
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

    power_cols = []

    for i, poa_col in enumerate(
        POA_COLS
    ):

        power_col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        area = pd.to_numeric(
            df_w.iloc[i][
                "Eff Area(m2)"
            ],
            errors="coerce",
        )

        area = (
            0
            if pd.isna(area)
            else float(area)
        )

        result[power_col] = (
            result[poa_col]
            * area
            / 1_000_000
        )

        power_cols.append(
            power_col
        )

    result[
        "Total Power (CL1+CL2+…)"
    ] = result[power_cols].sum(
        axis=1
    )

    return result


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

    actual_peak = (
        actual.max()
    )

    if actual_peak <= 0:
        raise ValueError(
            "Actual Power contains no positive values."
        )

    best_error = 0
    best_error_value = np.inf

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        _, df_w = (
            calculate_effective_area(
                df_original,
                df_w_original,
                error,
            )
        )

        result = (
            calculate_fixed_power(
                df_fix,
                df_w,
            )
        )

        forecast_peak = numeric(
            result[
                "Total Power (CL1+CL2+…)"
            ]
        ).max()

        error_value = abs(
            forecast_peak
            - actual_peak
        )

        if (
            error_value
            < best_error_value
        ):

            best_error_value = (
                error_value
            )

            best_error = (
                round(
                    float(error),
                    1,
                )
            )

    return best_error


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def create_tracking_objective(
    backend_list,
    df_ghi,
    df_fix,
    df_w,
):

    ghi_matrix = np.column_stack(
        [
            numeric(
                df_ghi[col]
            )
            for col in GHI_COLS
        ]
    )

    blocks = numeric(
        backend_list[0][
            "Block No."
        ]
    )

    actual_full = numeric(
        df_fix["Actual"]
    )

    if len(blocks) != len(
        ghi_matrix
    ):
        raise ValueError(
            "Tracking Block No. and GHI data lengths do not match."
        )

    if len(actual_full) != len(
        blocks
    ):
        raise ValueError(
            "Tracking Actual and Block No. lengths do not match."
        )

    mask = actual_full != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual = actual_full[mask]

    actual_max = actual.max()
    actual_sum = actual.sum()

    cl_weights = numeric(
        df_w.iloc[:5][
            "Eff Area(m2)"
        ]
    )

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

        den1 = (
            start
            - 1
            - maximum
        )

        den2 = (
            end
            + 1
            - maximum
        )

        if den1 == 0 or den2 == 0:
            return 1e9

        m1 = 90 / den1
        m2 = 90 / den2

        zenith = np.where(
            blocks <= maximum,
            np.minimum(
                89,
                m1
                * (
                    blocks
                    - maximum
                ),
            ),
            np.minimum(
                89,
                m2
                * (
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
            ghi_matrix
            - dhi
        ) / cos_alpha[:, None]

        prediction = (
            dni @ cl_weights
        ) / 1_000_000

        if (
            np.isnan(prediction).any()
            or np.isinf(prediction).any()
        ):
            return 1e9

        prediction = (
            prediction[mask]
        )

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

    params = {
        "DHI": int(best[0]),
        "GHI Starting Block": int(
            best[1]
        ),
        "GHI Ending Block": int(
            best[2]
        ),
        "GHI Max Block": int(
            best[3]
        ),
        "Tracking East Limit": int(
            best[4]
        ),
        "Tracking West Limit": int(
            best[5]
        ),
    }

    return (
        params,
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
    maximum,
    east,
    west,
):

    den1 = (
        start
        - 1
        - maximum
    )

    den2 = (
        end
        + 1
        - maximum
    )

    if den1 == 0 or den2 == 0:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    m1 = 90 / den1
    m2 = 90 / den2

    zenith = np.where(
        blocks <= maximum,
        np.minimum(
            89,
            m1
            * (
                blocks
                - maximum
            ),
        ),
        np.minimum(
            89,
            m2
            * (
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
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    forecast = (
        dni @ cl_weights
    ) / 1_000_000

    return (
        forecast,
        zenith,
        panel,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    forecast,
):

    actual = numeric(actual)
    forecast = numeric(forecast)

    actual_peak = (
        actual.max()
    )

    forecast_peak = (
        forecast.max()
    )

    peak_error_pct = (
        abs(
            forecast_peak
            - actual_peak
        )
        / actual_peak
        * 100
        if actual_peak != 0
        else np.nan
    )

    return {
        "Actual Peak": actual_peak,
        "Forecast Peak": forecast_peak,
        "Peak Error %": peak_error_pct,
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

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(width=2.4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(width=2.4),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
        ),
        height=450,
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
            y=1.02,
            x=1,
            xanchor="right",
        ),
    )

    return fig


# ============================================================
# RUN CALCULATION
#
# IMPORTANT:
# Optimization happens ONLY here.
# Changing editable parameters afterwards does not
# rerun differential_evolution.
# ============================================================

if st.button(
    "⚡ Run Calculation",
    type="primary",
    use_container_width=True,
):

    try:

        with st.spinner(
            "Running solar forecast optimization..."
        ):

            # ------------------------------------------------
            # Geometry
            # ------------------------------------------------

            df_fix = (
                prepare_fixed_geometry(
                    df_fix_raw,
                    df_ghi,
                    lat,
                    month_lookup,
                )
            )

            # ------------------------------------------------
            # Automatic Error %
            # ------------------------------------------------

            best_error = optimize_error(
                df_original,
                df_w_original,
                df_fix,
            )

            # ------------------------------------------------
            # Apply Error %
            # ONLY ONCE
            # ------------------------------------------------

            df_final, df_w_final = (
                calculate_effective_area(
                    df_original,
                    df_w_original,
                    best_error,
                )
            )

            # ------------------------------------------------
            # Fixed
            # ------------------------------------------------

            fixed_final = (
                calculate_fixed_power(
                    df_fix,
                    df_w_final,
                )
            )

            # ------------------------------------------------
            # Tracking optimization
            # ------------------------------------------------

            (
                tracking_parameters,
                blocks,
                ghi_matrix,
                actual_tracking,
                cl_weights,
            ) = optimize_tracking(
                backend_list,
                df_ghi,
                df_fix,
                df_w_final,
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            st.session_state.calculation_data = {
                "df_original": df_original,
                "df_w_original": df_w_original,
                "df_final": df_final,
                "df_w_final": df_w_final,
                "df_ghi": df_ghi.copy(),
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
# RESULTS REQUIRE CALCULATION
# ============================================================

if not st.session_state.calculated:
    st.info(
        "Edit Actual/GHI values if required, select the plant type, then click Run Calculation."
    )
    st.stop()


data = st.session_state.calculation_data


# ============================================================
# PARAMETERS
#
# No Apply button.
# Changing a parameter automatically recalculates the
# lightweight forecast on the next Streamlit rerun.
# ============================================================

st.markdown(
    '<div class="section">⚙️ Editable Parameters</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Error
# ------------------------------------------------------------

error_value = st.number_input(
    "Efficiency Error (%)",
    min_value=0.0,
    max_value=20.0,
    value=float(
        data["best_error"]
    ),
    step=0.1,
    format="%.1f",
)


# ------------------------------------------------------------
# Tracking parameters
# ------------------------------------------------------------

if plant_type == "Tracking":

    params = data[
        "tracking_parameters"
    ]

    c1, c2, c3 = st.columns(3)

    with c1:

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

    with c2:

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

    with c3:

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

else:

    st.caption(
        "Efficiency Error controls the effective PV efficiency."
    )

    dhi_value = None
    start_value = None
    end_value = None
    max_value = None
    east_value = None
    west_value = None


# ============================================================
# LIGHTWEIGHT RECALCULATION
#
# No optimizer here.
# This prevents the UI from freezing whenever a parameter
# is changed.
# ============================================================

try:

    df_final, df_w_final = (
        calculate_effective_area(
            data["df_original"],
            data["df_w_original"],
            error_value,
        )
    )

    fixed_final = (
        calculate_fixed_power(
            data["df_fix"],
            df_w_final,
        )
    )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    if plant_type == "Tracking":

        tracking_weights = numeric(
            df_w_final.iloc[:5][
                "Eff Area(m2)"
            ]
        )

        (
            tracking_forecast,
            zenith,
            panel,
        ) = calculate_tracking_forecast(
            data["blocks"],
            data["ghi_matrix"],
            tracking_weights,
            int(dhi_value),
            int(start_value),
            int(end_value),
            int(max_value),
            int(east_value),
            int(west_value),
        )

        actual = numeric(
            data["actual_tracking"]
        )

        forecast = numeric(
            tracking_forecast
        )

    else:

        actual = numeric(
            data["df_fix"]["Actual"]
        )

        forecast = numeric(
            fixed_final[
                "Total Power (CL1+CL2+…)"
            ]
        )


except Exception as e:

    st.error(
        f"Parameter calculation failed: {e}"
    )

    st.stop()


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
        <div class="card">
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
        <div class="card">
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

title = (
    "Fixed Plant | Actual vs Forecast"
    if plant_type == "Fixed"
    else "Tracking Plant | Actual vs Forecast"
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
# FOOTER STATUS
# ============================================================

st.caption(
    f"Plant: {plant_type}  •  "
    f"Efficiency Error: {error_value:.1f}%  •  "
    "Forecast parameters are editable."
)
