# ============================================================
# 2_Loss_Correction_VCast.py
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# ============================================================

import io
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
            max-width: 1500px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .app-title {
            font-size: 30px;
            font-weight: 750;
            letter-spacing: -0.5px;
            margin-bottom: 2px;
        }

        .app-subtitle {
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 18px;
        }

        .section-title {
            font-size: 19px;
            font-weight: 700;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 14px 16px;
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

        div[data-testid="stSegmentedControl"] {
            width: 100%;
        }

        div[data-testid="stSegmentedControl"] > div {
            width: 100%;
        }

        div[data-testid="stSegmentedControl"] button {
            flex: 1;
        }

        .stButton > button {
            min-height: 42px;
            border-radius: 9px;
            font-weight: 650;
        }

        div[data-testid="stDataEditor"] {
            border-radius: 10px;
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

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

FIXED_POA_COLS = [
    "POA fixed",
    "POA Fixed-C12",
    "POA Fixed-C13",
    "POA Fixed-C14",
    "POA Fixed-C15",
]

POWER_TOTAL_COL = "Total Power (CL1+CL2+…)"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "plant_type": "Fixed",
    "has_run": False,
    "raw_data": None,
    "calc_data": None,
    "edited_input": None,
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
    "Forecast correction with editable input data and optimized parameters"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def numeric(series):
    """Convert values to numeric safely."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


def read_excel_bytes(file_bytes, **kwargs):
    """Read Excel from cached bytes."""
    return pd.read_excel(
        io.BytesIO(file_bytes),
        **kwargs,
    )


# ============================================================
# LOAD EXCEL DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_workbook(file_bytes):

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    df_original = read_excel_bytes(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df_original.columns = (
        df_original.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    if "S.No." in df_original.columns:
        idx = df_original["S.No."].isna()
        if idx.any():
            df_original = df_original.iloc[
                :df_original.index.get_loc(idx[idx].index[0])
            ]

    for col in [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]:
        if col in df_original.columns:
            df_original[col] = numeric(
                df_original[col]
            )

    df_original["Total area (m2)"] = (
        df_original["No of Module"]
        * df_original["Area of 1 Module (m2)"]
    )

    # --------------------------------------------------------
    # Cluster table
    # --------------------------------------------------------

    df_w_original = read_excel_bytes(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df_w_original.columns = (
        df_w_original.columns
        .astype(str)
        .str.strip()
    )

    if "Clusters" in df_w_original.columns:
        idx = df_w_original["Clusters"].isna()
        if idx.any():
            df_w_original = df_w_original.iloc[
                :df_w_original.index.get_loc(idx[idx].index[0])
            ]

    df_w_original = df_w_original.reset_index(drop=True)

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    df_ghi = read_excel_bytes(
        file_bytes,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df_ghi = df_ghi.fillna(0)

    for col in GHI_COLS:
        if col in df_ghi.columns:
            df_ghi[col] = numeric(df_ghi[col])

    # --------------------------------------------------------
    # Latitude
    # --------------------------------------------------------

    df_config = read_excel_bytes(
        file_bytes,
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

    df_tilt = read_excel_bytes(
        file_bytes,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" in df_tilt.columns:
        idx = df_tilt["Fixed"].isna()
        if idx.any():
            df_tilt = df_tilt.iloc[
                :df_tilt.index.get_loc(idx[idx].index[0])
            ]

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

    df_fix = read_excel_bytes(
        file_bytes,
        sheet_name="Fixed-C11",
        header=1,
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    if "Date" in df_fix.columns:
        idx = df_fix["Date"].isna()
        if idx.any():
            df_fix = df_fix.iloc[
                :df_fix.index.get_loc(idx[idx].index[0])
            ]

    df_fix["Actual"] = numeric(
        df_fix["Actual"]
    )

    df_fix = df_fix.reset_index(drop=True)

    # --------------------------------------------------------
    # Tracking backend
    # --------------------------------------------------------

    backend_list = []

    for cluster in CLUSTERS:

        backend = read_excel_bytes(
            file_bytes,
            sheet_name=f"Backend Cal {cluster}",
        )

        backend_list.append(
            backend
        )

    df_tracking = read_excel_bytes(
        file_bytes,
        sheet_name="Tracking",
        header=1,
    )

    df_tracking.columns = (
        df_tracking.columns
        .astype(str)
        .str.strip()
    )

    return {
        "df_original": df_original,
        "df_w_original": df_w_original,
        "df_ghi": df_ghi,
        "lat": lat,
        "month_lookup": month_lookup,
        "df_fix": df_fix,
        "backend_list": backend_list,
        "df_tracking": df_tracking,
    }


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

    day_number = (
        df["Date"] - first_date
    ).dt.days + 1

    df["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_number
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

    # --------------------------------------------------------
    # All clusters
    # --------------------------------------------------------

    for i, cluster in enumerate(CLUSTERS):

        ghi = numeric(
            df_ghi[GHI_COLS[i]]
        )

        suffix = "" if i == 0 else f"-CL{i + 1}"

        df[
            f"GHI*sin(a){suffix}"
        ] = (
            ghi
            * df["Sin(a)"]
        )

        df[
            f"GHI*sin(a+b){suffix}"
        ] = (
            ghi
            * df["SIN(a+b)"]
        )

        poa_name = (
            "POA fixed"
            if i == 0
            else f"POA Fixed-C{cluster[1:]}"
        )

        df[poa_name] = (
            df[
                f"GHI*sin(a+b){suffix}"
            ]
            / sin_a
        )

    return df


# ============================================================
# EFFECTIVE AREA
#
# ERROR % IS APPLIED ONLY ONCE
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
        * numeric(df["Total area (m2)"])
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

    power_cols = []

    for i, poa_col in enumerate(
        FIXED_POA_COLS
    ):

        power_col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        area = (
            pd.to_numeric(
                df_w.iloc[i]["Eff Area(m2)"],
                errors="coerce",
            )
        )

        if pd.isna(area):
            area = 0

        result[power_col] = (
            numeric(result[poa_col])
            * float(area)
            / 1_000_000
        )

        power_cols.append(
            power_col
        )

    result[POWER_TOTAL_COL] = (
        result[power_cols]
        .sum(axis=1)
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
    ).to_numpy()

    actual_peak = (
        actual.max()
        if len(actual)
        else 0
    )

    if actual_peak <= 0:
        raise ValueError(
            "No non-zero Actual values found."
        )

    best_error = 0
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

        peak = numeric(
            calculated[
                POWER_TOTAL_COL
            ]
        ).max()

        peak_error = abs(
            peak
            - actual_peak
        )

        if peak_error < best_error_value:
            best_error_value = peak_error
            best_error = float(
                round(error, 1)
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
            ).to_numpy(
                dtype=float
            )
            for col in GHI_COLS
        ]
    )

    blocks = numeric(
        backend_list[0]["Block No."]
    ).to_numpy(
        dtype=float
    )

    actual_full = numeric(
        df_fix["Actual"]
    ).to_numpy(
        dtype=float
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

    cl_weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
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

        d1 = (
            start
            - 1
            - max_block
        )

        d2 = (
            end
            + 1
            - max_block
        )

        if d1 == 0 or d2 == 0:
            return 1e9

        m1 = 90 / d1
        m2 = 90 / d2

        zenith = np.where(
            blocks <= max_block,

            np.minimum(
                89,
                m1
                * (
                    blocks
                    - max_block
                ),
            ),

            np.minimum(
                89,
                m2
                * (
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
                    & (
                        zenith > west
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

        if not len(prediction):
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

    params = {
        "DHI": int(best[0]),
        "GHI Starting Block": int(best[1]),
        "GHI Ending Block": int(best[2]),
        "GHI Max Block": int(best[3]),
        "Tracking East Limit": int(best[4]),
        "Tracking West Limit": int(best[5]),
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
    max_block,
    east,
    west,
):

    d1 = (
        start
        - 1
        - max_block
    )

    d2 = (
        end
        + 1
        - max_block
    )

    if d1 == 0:
        raise ValueError(
            "East slope denominator is zero."
        )

    if d2 == 0:
        raise ValueError(
            "West slope denominator is zero."
        )

    m1 = 90 / d1
    m2 = 90 / d2

    zenith = np.where(
        blocks <= max_block,

        np.minimum(
            89,
            m1
            * (
                blocks
                - max_block
            ),
        ),

        np.minimum(
            89,
            m2
            * (
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
                & (
                    zenith > west
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
                width=2.5,
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
                width=2.5,
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
        ),
        height=460,
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
# FILE INPUT
# ============================================================

st.markdown(
    '<div class="section-title">📂 Input Data</div>',
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
# READ WORKBOOK
# ============================================================

file_bytes = uploaded_file.getvalue()

try:

    if (
        st.session_state.raw_data is None
        or st.session_state.get(
            "file_signature"
        )
        != hash(file_bytes)
    ):

        st.session_state.raw_data = (
            load_workbook(file_bytes)
        )

        st.session_state.file_signature = (
            hash(file_bytes)
        )

        st.session_state.has_run = False
        st.session_state.calc_data = None
        st.session_state.edited_input = None

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


raw = st.session_state.raw_data


# ============================================================
# INPUT DATA EDITOR
# ============================================================

input_df = pd.DataFrame()

for col in GHI_COLS:
    input_df[col] = numeric(
        raw["df_ghi"][col]
    )

input_df["Actual"] = numeric(
    raw["df_fix"]["Actual"]
)

# Make all columns same length
input_df = input_df.iloc[
    :min(
        len(input_df),
        len(raw["df_fix"]),
    )
].reset_index(drop=True)

edited_input = st.data_editor(
    input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    height=260,
    key="solar_input_editor",
    column_config={
        col: st.column_config.NumberColumn(
            col,
            format="%.3f",
        )
        for col in GHI_COLS + ["Actual"]
    },
)

st.caption(
    "Edit GHI C11-C15 and Actual values directly above. "
    "The Run Calculation button uses the edited values."
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
    options=["Fixed", "Tracking"],
    default=st.session_state.plant_type,
    selection_mode="single",
    label_visibility="collapsed",
)

if plant_type is None:
    plant_type = st.session_state.plant_type

st.session_state.plant_type = plant_type


# ============================================================
# RUN CALCULATION
# ============================================================

st.markdown("")

run_clicked = st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN INITIAL OPTIMIZATION
# ============================================================

if run_clicked:

    try:

        with st.spinner(
            "Running automatic optimization..."
        ):

            # ------------------------------------------------
            # Edited input data
            # ------------------------------------------------

            df_ghi = raw["df_ghi"].copy()

            df_fix = raw["df_fix"].copy()

            n = min(
                len(df_ghi),
                len(edited_input),
            )

            df_ghi = df_ghi.iloc[
                :n
            ].copy()

            df_fix = df_fix.iloc[
                :n
            ].copy()

            for col in GHI_COLS:
                df_ghi[col] = numeric(
                    edited_input[col]
                ).to_numpy()

            df_fix["Actual"] = numeric(
                edited_input["Actual"]
            ).to_numpy()

            # ------------------------------------------------
            # Geometry
            # ------------------------------------------------

            df_fix = prepare_fixed_geometry(
                df_fix,
                df_ghi,
                raw["lat"],
                raw["month_lookup"],
            )

            # ------------------------------------------------
            # Optimize Error %
            # ------------------------------------------------

            best_error = optimize_error(
                raw["df_original"],
                raw["df_w_original"],
                df_fix,
            )

            # ------------------------------------------------
            # Apply Error %
            # ONLY ONCE
            # ------------------------------------------------

            (
                df_final,
                df_w_final,
            ) = calculate_effective_area(
                raw["df_original"],
                raw["df_w_original"],
                best_error,
            )

            # ------------------------------------------------
            # Fixed
            # ------------------------------------------------

            fixed_final = calculate_fixed_power(
                df_fix,
                df_w_final,
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
                raw["backend_list"],
                df_ghi,
                df_fix,
                df_w_final,
            )

            # ------------------------------------------------
            # Tracking forecast
            # ------------------------------------------------

            tracking_forecast = (
                calculate_tracking_forecast(
                    blocks,
                    ghi_matrix,
                    cl_weights,
                    tracking_parameters[
                        "DHI"
                    ],
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
            # Store
            # ------------------------------------------------

            st.session_state.calc_data = {
                "df_ghi": df_ghi,
                "df_fix": df_fix,
                "df_final": df_final,
                "df_w_final": df_w_final,
                "fixed_final": fixed_final,
                "best_error": best_error,
                "tracking_parameters":
                    tracking_parameters,
                "blocks": blocks,
                "ghi_matrix": ghi_matrix,
                "actual_tracking":
                    actual_tracking,
                "cl_weights": cl_weights,
                "tracking_forecast":
                    tracking_forecast,
            }

            st.session_state.has_run = True

        st.success(
            "Calculation completed successfully."
        )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )

        st.stop()


# ============================================================
# WAIT FOR RUN
# ============================================================

if not st.session_state.has_run:
    st.info(
        "Edit the input data if required, select the plant type, "
        "then click Run Automatic Calculation."
    )
    st.stop()


data = st.session_state.calc_data


# ============================================================
# CURRENT EDITED INPUT DATA
#
# Recalculate from current widgets automatically.
# No Apply Parameters button required.
# ============================================================

try:

    df_ghi_current = data["df_ghi"].copy()
    df_fix_current = data["df_fix"].copy()

    n = min(
        len(df_ghi_current),
        len(edited_input),
    )

    df_ghi_current = (
        df_ghi_current
        .iloc[:n]
        .copy()
    )

    df_fix_current = (
        df_fix_current
        .iloc[:n]
        .copy()
    )

    for col in GHI_COLS:
        df_ghi_current[col] = numeric(
            edited_input[col]
        ).to_numpy()

    df_fix_current["Actual"] = numeric(
        edited_input["Actual"]
    ).to_numpy()

    df_fix_current = prepare_fixed_geometry(
        df_fix_current,
        df_ghi_current,
        raw["lat"],
        raw["month_lookup"],
    )

except Exception as e:

    st.error(
        f"Input data error: {e}"
    )

    st.stop()


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Error
# ------------------------------------------------------------

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
# Tracking parameters
# ------------------------------------------------------------

if plant_type == "Tracking":

    params = data[
        "tracking_parameters"
    ]

    st.markdown(
        "#### Tracking Parameters"
    )

    p1, p2, p3 = st.columns(3)

    with p1:

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

    with p2:

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

    with p3:

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
# APPLY CURRENT PARAMETERS AUTOMATICALLY
# ============================================================

try:

    (
        df_final_current,
        df_w_final_current,
    ) = calculate_effective_area(
        raw["df_original"],
        raw["df_w_original"],
        error_value,
    )

    # --------------------------------------------------------
    # Fixed
    # --------------------------------------------------------

    fixed_final_current = (
        calculate_fixed_power(
            df_fix_current,
            df_w_final_current,
        )
    )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    if plant_type == "Tracking":

        current_weights = (
            pd.to_numeric(
                df_w_final_current.iloc[:5, 1],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(
                dtype=float
            )
        )

        tracking_forecast_current = (
            calculate_tracking_forecast(
                data["blocks"],
                np.column_stack(
                    [
                        numeric(
                            df_ghi_current[col]
                        ).to_numpy(
                            dtype=float
                        )
                        for col in GHI_COLS
                    ]
                ),
                current_weights,
                int(dhi_value),
                int(start_value),
                int(end_value),
                int(max_value),
                int(east_value),
                int(west_value),
            )
        )

    # --------------------------------------------------------
    # Fixed result
    # --------------------------------------------------------

    if plant_type == "Fixed":

        actual = numeric(
            df_fix_current["Actual"]
        ).to_numpy()

        forecast = numeric(
            fixed_final_current[
                POWER_TOTAL_COL
            ]
        ).to_numpy()

        title = (
            "Fixed Plant | Actual vs Forecast"
        )

    # --------------------------------------------------------
    # Tracking result
    # --------------------------------------------------------

    else:

        actual = numeric(
            df_fix_current["Actual"]
        ).to_numpy()

        forecast = np.asarray(
            tracking_forecast_current,
            dtype=float,
        )

        title = (
            "Tracking Plant | Actual vs Forecast"
        )

except Exception as e:

    st.error(
        f"Forecast calculation failed: {e}"
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Results</div>',
    unsafe_allow_html=True,
)

metrics = calculate_metrics(
    actual,
    forecast,
)

m1, m2, m3 = st.columns(3)

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

with m3:

    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">
                Peak Error
            </div>
            <div class="metric-value">
                {metrics["Peak Error %"]:.2f}%
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
        "displayModeBar": False,
        "responsive": True,
    },
)


# ============================================================
# FINAL NOTE
# ============================================================

st.caption(
    "Parameters and input values are editable. "
    "Changes are reflected automatically without requiring "
    "a separate Apply Parameters button."
)
