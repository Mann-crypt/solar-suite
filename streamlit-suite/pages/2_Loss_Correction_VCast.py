# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# Compact Streamlit Version
# ============================================================

import hashlib
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
    layout="wide"
)


# ============================================================
# SIMPLE UI
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-size: 2rem !important;
    }

    div[data-testid="stMetric"] {
        padding: 8px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("☀️ Solar Forecast Correction")
st.caption("Fixed / Tracking forecast correction")


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Upload your Excel file to start.")
    st.stop()


# ============================================================
# READ EXCEL
# ============================================================

@st.cache_data(show_spinner=False)
def load_excel(file_bytes):

    return pd.ExcelFile(file_bytes)


file_bytes = uploaded_file.getvalue()

try:
    excel = load_excel(file_bytes)
except Exception as e:
    st.error(f"Unable to read Excel file: {e}")
    st.stop()


# ============================================================
# REQUIRED SHEETS
# ============================================================

required_sheets = [
    "Area & Efficiency",
    "Forecast Config",
    "Config Tilt Angle",
    "Result",
    "Fixed-C11",
    "Backend Cal C11",
    "Backend Cal C12",
    "Backend Cal C13",
    "Backend Cal C14",
    "Backend Cal C15",
    "Tracking"
]

missing = [
    s for s in required_sheets
    if s not in excel.sheet_names
]

if missing:
    st.error(
        "Missing required sheets:\n\n"
        + "\n".join(f"- {x}" for x in missing)
    )
    st.stop()


# ============================================================
# INPUT DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_input_data(file_bytes):

    df_ghi = pd.read_excel(
        file_bytes,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5]
    ).fillna(0)

    df_fix = pd.read_excel(
        file_bytes,
        sheet_name="Fixed-C11",
        header=1
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    return df_ghi, df_fix


df_ghi_original, df_fix_original = load_input_data(file_bytes)


# ============================================================
# INPUT DATA
# ============================================================

st.subheader("Input Data")

ghi_cols = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15"
]

missing_ghi = [
    c for c in ghi_cols
    if c not in df_ghi_original.columns
]

if missing_ghi:
    st.error(
        "Missing GHI columns: "
        + ", ".join(missing_ghi)
    )
    st.stop()


if "Actual" not in df_fix_original.columns:
    st.error("Column 'Actual' was not found in Fixed-C11.")
    st.stop()


# Keep only useful rows
n = min(
    len(df_ghi_original),
    len(df_fix_original)
)

df_ghi_input = df_ghi_original[ghi_cols].iloc[:n].copy()

df_actual_input = pd.DataFrame({
    "Actual": pd.to_numeric(
        df_fix_original["Actual"].iloc[:n],
        errors="coerce"
    ).fillna(0)
})


col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**GHI Forecast**")

    df_ghi_input = st.data_editor(
        df_ghi_input,
        use_container_width=True,
        height=220,
        key="ghi_input"
    )

with col2:
    st.markdown("**Actual Power**")

    df_actual_input = st.data_editor(
        df_actual_input,
        use_container_width=True,
        height=220,
        key="actual_input"
    )


# ============================================================
# PLANT TYPE
# ============================================================

st.subheader("Plant Configuration")

plant_type = st.segmented_control(
    "Plant Type",
    ["Fixed", "Tracking"],
    default="Fixed"
)


if plant_type is None:
    st.stop()


# ============================================================
# LOAD AREA / EFFICIENCY
# ============================================================

@st.cache_data(show_spinner=False)
def load_area_data(file_bytes):

    df = pd.read_excel(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12)
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
            first_null = idx.idxmax()
            df = df.loc[:first_null - 1]

    df_w = pd.read_excel(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15]
    )

    df_w.columns = (
        df_w.columns
        .astype(str)
        .str.strip()
    )

    if "Clusters" in df_w.columns:
        idx = df_w["Clusters"].isna()

        if idx.any():
            first_null = idx.idxmax()
            df_w = df_w.loc[:first_null - 1]

    return df, df_w


df_area_original, df_w_original = load_area_data(file_bytes)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_area_columns = [
    "Clusters",
    "Standard PV Efficiency (%)",
    "No of Module",
    "Area of 1 Module (m2)"
]

missing_area = [
    c for c in required_area_columns
    if c not in df_area_original.columns
]

if missing_area:
    st.error(
        "Missing Area & Efficiency columns: "
        + ", ".join(missing_area)
    )
    st.stop()


# ============================================================
# LOAD LAT / TILT
# ============================================================

@st.cache_data(show_spinner=False)
def load_site_data(file_bytes):

    df_config = pd.read_excel(
        file_bytes,
        sheet_name="Forecast Config",
        header=8
    )

    lat = float(df_config.loc[0, "Lat"])

    df_tilt = pd.read_excel(
        file_bytes,
        sheet_name="Config Tilt Angle",
        header=7
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" in df_tilt.columns:

        idx = df_tilt["Fixed"].isna()

        if idx.any():
            first_null = idx.idxmax()
            df_tilt = df_tilt.loc[:first_null - 1]

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

    return lat, month_lookup


lat, month_lookup = load_site_data(file_bytes)


# ============================================================
# FIXED SOLAR GEOMETRY
# ============================================================

def prepare_fixed_geometry(
    df_ghi,
    df_actual,
    lat,
    month_lookup
):

    n = min(
        len(df_ghi),
        len(df_actual)
    )

    ghi = df_ghi.iloc[:n].copy()
    actual = pd.to_numeric(
        df_actual["Actual"].iloc[:n],
        errors="coerce"
    ).fillna(0).to_numpy(float)

    df = pd.DataFrame(index=np.arange(n))

    # Same date logic as original calculation
    today = pd.Timestamp.today()

    dates = pd.Series(
        [today] * n
    )

    first_date = today.replace(
        month=1,
        day=1
    ).normalize()

    day_no = (
        dates - first_date
    ).dt.days

    declination = 23.45 * np.sin(
        np.radians(
            360 * (284 + day_no + 1) / 365
        )
    )

    elevation = (
        90 - lat + declination
    )

    months = dates.dt.strftime("%B")

    tilt = months.map(
        month_lookup
    )

    a_plus_b = (
        elevation + tilt
    )

    sin_ab = np.sin(
        np.radians(a_plus_b)
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    # Avoid divide-by-zero
    sin_a_safe = np.where(
        np.abs(sin_a) < 1e-8,
        np.nan,
        sin_a
    )

    poa = {}

    for i, col in enumerate(ghi_cols):

        cluster = f"C{i + 1}"

        g = pd.to_numeric(
            ghi[col],
            errors="coerce"
        ).fillna(0).to_numpy(float)

        poa[cluster] = (
            g * sin_ab / sin_a_safe
        )

    return (
        np.asarray(actual, dtype=float),
        poa
    )


# ============================================================
# AREA CALCULATION
# ============================================================

def calculate_cluster_areas(
    error_percent,
    df_area,
    df_w
):

    area = df_area.copy()
    weights = df_w.copy()

    area["Standard PV Efficiency (%)"] = pd.to_numeric(
        area["Standard PV Efficiency (%)"],
        errors="coerce"
    )

    area["No of Module"] = pd.to_numeric(
        area["No of Module"],
        errors="coerce"
    )

    area["Area of 1 Module (m2)"] = pd.to_numeric(
        area["Area of 1 Module (m2)"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # ERROR % APPLIED HERE ONLY
    # --------------------------------------------------------

    area["Error %"] = error_percent

    area["Net Efficiency (%)"] = (
        area["Standard PV Efficiency (%)"]
        - error_percent
    )

    area["Total area (m2)"] = (
        area["No of Module"]
        * area["Area of 1 Module (m2)"]
    )

    area["Eff Area"] = (
        area["Net Efficiency (%)"]
        * area["Total area (m2)"]
        / 100
    )

    cluster_sums = (
        area
        .groupby("Clusters")["Eff Area"]
        .sum()
    )

    weights["Eff Area(m2)"] = (
        weights["Clusters"]
        .map(cluster_sums)
        .fillna(0)
    )

    return area, weights


# ============================================================
# AUTOMATIC ERROR %
# ============================================================

def find_best_error(
    actual,
    poa,
    df_area,
    df_w
):

    actual_peak = np.nanmax(actual)

    if not np.isfinite(actual_peak) or actual_peak <= 0:
        return 0.0

    results = []

    for error in np.arange(
        0,
        10.01,
        0.1
    ):

        _, weights = calculate_cluster_areas(
            error,
            df_area,
            df_w
        )

        cluster_weights = (
            pd.to_numeric(
                weights.iloc[:5]["Eff Area(m2)"],
                errors="coerce"
            )
            .fillna(0)
            .to_numpy(float)
        )

        forecast = np.zeros(len(actual))

        for i in range(5):
            forecast += (
                poa[f"C{i + 1}"]
                * cluster_weights[i]
                / 1_000_000
            )

        calculated_peak = np.nanmax(
            forecast
        )

        error_value = abs(
            calculated_peak
            - actual_peak
        )

        results.append(
            (
                error_value,
                error,
                calculated_peak
            )
        )

    return min(
        results,
        key=lambda x: x[0]
    )[1]


# ============================================================
# TRACKING DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_tracking_data(file_bytes):

    backend = []

    for cluster in range(11, 16):

        backend.append(
            pd.read_excel(
                file_bytes,
                sheet_name=f"Backend Cal C{cluster}"
            )
        )

    tracking = pd.read_excel(
        file_bytes,
        sheet_name="Tracking",
        header=1
    )

    return backend, tracking


backend_list, df_tracking_original = (
    load_tracking_data(file_bytes)
)


# ============================================================
# TRACKING OPTIMIZER
# ============================================================

def optimize_tracking(
    df_ghi,
    actual,
    weights,
    backend_list,
    initial_params=None
):

    n = min(
        len(actual),
        len(df_ghi),
        len(backend_list[0])
    )

    actual = actual[:n]

    ghi_matrix = np.column_stack([
        pd.to_numeric(
            df_ghi[col].iloc[:n],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(float)
        for col in ghi_cols
    ])

    blocks = pd.to_numeric(
        backend_list[0]["Block No."].iloc[:n],
        errors="coerce"
    ).fillna(0).to_numpy(float)

    cluster_weights = pd.to_numeric(
        weights.iloc[:5]["Eff Area(m2)"],
        errors="coerce"
    ).fillna(0).to_numpy(float)

    # --------------------------------------------------------
    # SAME MASK AS JUPYTER
    # --------------------------------------------------------

    mask = actual != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual_day = actual[mask]

    actual_max = actual_day.max()
    actual_sum = actual_day.sum()

    if actual_max <= 0 or actual_sum <= 0:
        raise ValueError(
            "Actual Power does not contain valid positive values."
        )

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

        denominator_1 = (
            start - 1 - max_block
        )

        denominator_2 = (
            end + 1 - max_block
        )

        if denominator_1 == 0 or denominator_2 == 0:
            return 1e9

        m1 = 90 / denominator_1
        m2 = 90 / denominator_2

        zenith = np.where(
            blocks <= max_block,

            np.minimum(
                89,
                m1 * (
                    blocks - max_block
                )
            ),

            np.minimum(
                89,
                m2 * (
                    blocks - max_block
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
                    (blocks > max_block)
                    & (zenith > west)
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

        dhi = (
            ghi_matrix * DHI / 100
        )

        dni = (
            ghi_matrix - dhi
        ) / cos_alpha[:, None]

        prediction_full = (
            dni @ cluster_weights
        ) / 1_000_000

        if (
            np.isnan(prediction_full).any()
            or np.isinf(prediction_full).any()
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
                    actual_day
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

    bounds = [
        (0, 10),
        (10, 30),
        (65, 80),
        (47, 53),
        (10, 70),
        (10, 70)
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
        workers=1
    )

    best = np.round(
        result.x
    ).astype(int)

    return best


# ============================================================
# CREATE TRACKING FORECAST
# ============================================================

def tracking_forecast(
    df_ghi,
    actual,
    weights,
    backend_list,
    params
):

    n = min(
        len(actual),
        len(df_ghi),
        len(backend_list[0])
    )

    ghi_matrix = np.column_stack([
        pd.to_numeric(
            df_ghi[col].iloc[:n],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(float)
        for col in ghi_cols
    ])

    blocks = pd.to_numeric(
        backend_list[0]["Block No."].iloc[:n],
        errors="coerce"
    ).fillna(0).to_numpy(float)

    cluster_weights = pd.to_numeric(
        weights.iloc[:5]["Eff Area(m2)"],
        errors="coerce"
    ).fillna(0)
    .to_numpy(float)

    DHI = int(params[0])
    start = int(params[1])
    end = int(params[2])
    max_block = int(params[3])
    east = int(params[4])
    west = int(params[5])

    m1 = 90 / (
        start - 1 - max_block
    )

    m2 = 90 / (
        end + 1 - max_block
    )

    zenith = np.where(
        blocks <= max_block,

        np.minimum(
            89,
            m1 * (
                blocks - max_block
            )
        ),

        np.minimum(
            89,
            m2 * (
                blocks - max_block
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
                (blocks > max_block)
                & (zenith > west)
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

    dhi = (
        ghi_matrix * DHI / 100
    )

    dni = (
        ghi_matrix - dhi
    ) / cos_alpha[:, None]

    forecast = (
        dni @ cluster_weights
    ) / 1_000_000

    return forecast


# ============================================================
# FIXED FORECAST
# ============================================================

def fixed_forecast(
    actual,
    poa,
    weights
):

    cluster_weights = pd.to_numeric(
        weights.iloc[:5]["Eff Area(m2)"],
        errors="coerce"
    ).fillna(0).to_numpy(float)

    forecast = np.zeros(
        len(actual)
    )

    for i in range(5):

        forecast += (
            poa[f"C{i + 1}"]
            * cluster_weights[i]
            / 1_000_000
        )

    return forecast


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "params" not in st.session_state:
    st.session_state.params = None

if "best_error" not in st.session_state:
    st.session_state.best_error = None

if "calculated" not in st.session_state:
    st.session_state.calculated = False


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

if st.button(
    "⚡ Run Automatic Calculation",
    type="primary"
):

    try:

        with st.spinner(
            "Running calculation..."
        ):

            actual, poa = prepare_fixed_geometry(
                df_ghi_input,
                df_actual_input,
                lat,
                month_lookup
            )

            best_error = find_best_error(
                actual,
                poa,
                df_area_original,
                df_w_original
            )

            area_final, weights_final = (
                calculate_cluster_areas(
                    best_error,
                    df_area_original,
                    df_w_original
                )
            )

            if plant_type == "Fixed":

                forecast = fixed_forecast(
                    actual,
                    poa,
                    weights_final
                )

                params = None

            else:

                params = optimize_tracking(
                    df_ghi_input,
                    actual,
                    weights_final,
                    backend_list
                )

                forecast = tracking_forecast(
                    df_ghi_input,
                    actual,
                    weights_final,
                    backend_list,
                    params
                )

            st.session_state.result = (
                actual,
                forecast
            )

            st.session_state.params = params
            st.session_state.best_error = best_error
            st.session_state.calculated = True

        st.success("Automatic calculation completed.")

    except Exception as e:

        st.session_state.calculated = False
        st.error(
            f"Calculation failed: {e}"
        )


# ============================================================
# EDITABLE PARAMETERS
# ============================================================

if st.session_state.calculated:

    st.subheader("Parameters")

    best_error = st.session_state.best_error

    if plant_type == "Fixed":

        error_percent = st.number_input(
            "Error %",
            min_value=0.0,
            max_value=10.0,
            value=float(best_error),
            step=0.1
        )

        if st.button(
            "Recalculate Forecast"
        ):

            try:

                actual, poa = prepare_fixed_geometry(
                    df_ghi_input,
                    df_actual_input,
                    lat,
                    month_lookup
                )

                _, weights = calculate_cluster_areas(
                    error_percent,
                    df_area_original,
                    df_w_original
                )

                forecast = fixed_forecast(
                    actual,
                    poa,
                    weights
                )

                st.session_state.result = (
                    actual,
                    forecast
                )

                st.session_state.best_error = (
                    error_percent
                )

                st.success(
                    "Forecast recalculated."
                )

            except Exception as e:
                st.error(
                    f"Recalculation failed: {e}"
                )

    else:

        params = st.session_state.params

        p1, p2, p3 = st.columns(3)

        with p1:

            dhi_value = st.number_input(
                "DHI (%)",
                min_value=0,
                max_value=10,
                value=int(params[0]),
                step=1
            )

            start_value = st.number_input(
                "GHI Starting Block",
                min_value=10,
                max_value=30,
                value=int(params[1]),
                step=1
            )

        with p2:

            end_value = st.number_input(
                "GHI Ending Block",
                min_value=65,
                max_value=80,
                value=int(params[2]),
                step=1
            )

            max_value = st.number_input(
                "GHI Max Block",
                min_value=47,
                max_value=53,
                value=int(params[3]),
                step=1
            )

        with p3:

            east_value = st.number_input(
                "Tracking East Limit",
                min_value=10,
                max_value=70,
                value=int(params[4]),
                step=1
            )

            west_value = st.number_input(
                "Tracking West Limit",
                min_value=10,
                max_value=70,
                value=int(params[5]),
                step=1
            )

        st.caption(
            f"Automatically calculated Error %: "
            f"**{best_error:.1f}%**"
        )

        if st.button(
            "Recalculate Forecast"
        ):

            try:

                actual, _ = prepare_fixed_geometry(
                    df_ghi_input,
                    df_actual_input,
                    lat,
                    month_lookup
                )

                # ------------------------------------------------
                # IMPORTANT:
                # Error % applied ONCE here.
                # Tracking optimizer receives the resulting
                # cluster effective areas.
                # ------------------------------------------------

                _, weights = calculate_cluster_areas(
                    best_error,
                    df_area_original,
                    df_w_original
                )

                new_params = np.array([
                    dhi_value,
                    start_value,
                    end_value,
                    max_value,
                    east_value,
                    west_value
                ])

                forecast = tracking_forecast(
                    df_ghi_input,
                    actual,
                    weights,
                    backend_list,
                    new_params
                )

                st.session_state.result = (
                    actual,
                    forecast
                )

                st.session_state.params = (
                    new_params
                )

                st.success(
                    "Forecast recalculated."
                )

            except Exception as e:

                st.error(
                    f"Recalculation failed: {e}"
                )


# ============================================================
# FORECAST GRAPH
# ============================================================

if st.session_state.result is not None:

    actual, forecast = (
        st.session_state.result
    )

    st.subheader(
        f"{plant_type} Forecast vs Actual"
    )

    n = min(
        len(actual),
        len(forecast)
    )

    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast[:n],
            mode="lines",
            name="Forecast",
            line=dict(
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual[:n],
            mode="lines",
            name="Actual",
            line=dict(
                width=2
            )
        )
    )

    fig.update_layout(
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),
        xaxis_title="15-Minute Block",
        yaxis_title="Power",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False
        }
    )
