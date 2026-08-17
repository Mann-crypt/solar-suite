# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# Compact + Optimized Streamlit App
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
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1 {
    margin-bottom: 0.2rem;
}

[data-testid="stMetricValue"] {
    font-size: 1.35rem;
}

.stButton > button {
    width: 100%;
}

div[data-testid="stDataEditor"] {
    border-radius: 8px;
}

.result-card {
    padding: 14px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


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

POWER_COLS = [
    "CL1 Power",
    "CL2 Power",
    "CL3 Power",
    "CL4 Power",
    "CL5 Power",
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
    """Keep rows until first blank in specified column."""
    df = df.copy()

    if column not in df.columns:
        return df

    valid = df[column].notna()

    if not valid.any():
        return df.iloc[0:0].copy()

    first_invalid = np.flatnonzero(~valid.to_numpy())

    if len(first_invalid):
        return df.iloc[:first_invalid[0]].copy()

    return df.copy()


def numeric_array(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).to_numpy(float)


# ============================================================
# LOAD WORKBOOK
# ============================================================

@st.cache_data(show_spinner=False)
def load_workbook(file_bytes):

    bio = io.BytesIO(file_bytes)

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    area = pd.read_excel(
        bio,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    area = clean_columns(area)
    area = trim_at_blank(area, "S.No.")

    # --------------------------------------------------------
    # Cluster mapping
    # --------------------------------------------------------

    bio.seek(0)

    cluster = pd.read_excel(
        bio,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    cluster = clean_columns(cluster)
    cluster = trim_at_blank(cluster, "Clusters")

    # --------------------------------------------------------
    # Forecast Config
    # --------------------------------------------------------

    bio.seek(0)

    config = pd.read_excel(
        bio,
        sheet_name="Forecast Config",
        header=8,
    )

    lat = float(config.loc[0, "Lat"])

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    bio.seek(0)

    tilt = pd.read_excel(
        bio,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    tilt = clean_columns(tilt)
    tilt = trim_at_blank(tilt, "Fixed")
    tilt = tilt.dropna(how="all", axis=1)

    tilt = tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    month_lookup = (
        tilt.set_index("Month")["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    bio.seek(0)

    ghi = pd.read_excel(
        bio,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    ghi = ghi.fillna(0)

    for col in GHI_COLS:
        if col not in ghi.columns:
            raise ValueError(
                f"Missing GHI column: {col}"
            )

    # --------------------------------------------------------
    # Actual
    # --------------------------------------------------------

    bio.seek(0)

    fixed = pd.read_excel(
        bio,
        sheet_name="Fixed-C11",
        header=1,
    )

    fixed = clean_columns(fixed)
    fixed = trim_at_blank(fixed, "Date")

    if "Actual" not in fixed.columns:
        raise ValueError(
            "Column 'Actual' not found in Fixed-C11."
        )

    actual = pd.to_numeric(
        fixed["Actual"],
        errors="coerce",
    ).fillna(0)

    # --------------------------------------------------------
    # Backend tracking blocks
    # --------------------------------------------------------

    backend_blocks = []

    for i in range(1, 6):

        bio.seek(0)

        try:
            backend = pd.read_excel(
                bio,
                sheet_name=f"Backend Cal C{i}",
            )

            backend = clean_columns(backend)

            if "Block No." in backend.columns:
                backend_blocks.append(
                    numeric_array(
                        backend["Block No."]
                    )
                )
            else:
                backend_blocks.append(None)

        except Exception:
            backend_blocks.append(None)

    # --------------------------------------------------------
    # Tracking sheet
    # --------------------------------------------------------

    bio.seek(0)

    try:
        tracking = pd.read_excel(
            bio,
            sheet_name="Tracking",
            header=1,
        )
        tracking = clean_columns(tracking)
    except Exception:
        tracking = pd.DataFrame()

    return {
        "area": area,
        "cluster": cluster,
        "lat": lat,
        "month_lookup": month_lookup,
        "ghi": ghi,
        "actual": actual.to_frame("Actual"),
        "backend_blocks": backend_blocks,
        "tracking": tracking,
    }


# ============================================================
# PREPARE PLANT DATA
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_plant_data(data):

    df = data["area"].copy()
    df_w = data["cluster"].copy()

    # --------------------------------------------------------
    # Initial efficiency
    # --------------------------------------------------------

    if "Error %" not in df.columns:
        df["Error %"] = 0.0

    if "Total area (m2)" not in df.columns:

        if (
            "No of Module" in df.columns
            and "Area of 1 Module (m2)" in df.columns
        ):
            df["Total area (m2)"] = (
                numeric_array(df["No of Module"])
                * numeric_array(
                    df["Area of 1 Module (m2)"]
                )
            )

    return df, df_w


# ============================================================
# EFFECTIVE AREA
# ============================================================

def calculate_cluster_area(
    df,
    df_w,
    error_percent,
):

    std_eff = numeric_array(
        df["Standard PV Efficiency (%)"]
    )

    total_area = numeric_array(
        df["Total area (m2)"]
    )

    clusters = df["Clusters"]

    # --------------------------------------------------------
    # Error applied EXACTLY ONCE
    # --------------------------------------------------------

    net_eff = std_eff - error_percent

    eff_area = (
        net_eff
        * total_area
        / 100.0
    )

    cluster_sum = (
        pd.DataFrame({
            "Clusters": clusters,
            "Eff Area": eff_area,
        })
        .groupby("Clusters")["Eff Area"]
        .sum()
    )

    cluster_area = (
        df_w["Clusters"]
        .map(cluster_sum)
        .fillna(0)
        .to_numpy(float)
    )

    return net_eff, eff_area, cluster_area


# ============================================================
# PREPARE FIXED CALCULATION
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_fixed_base(
    ghi_df,
    actual_df,
    lat,
    month_lookup,
):

    ghi = ghi_df[GHI_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0).to_numpy(float)

    actual = pd.to_numeric(
        actual_df["Actual"],
        errors="coerce",
    ).fillna(0).to_numpy(float)

    n = min(
        len(ghi),
        len(actual),
    )

    ghi = ghi[:n]
    actual = actual[:n]

    # --------------------------------------------------------
    # EXACT DATE LOGIC FROM JUPYTER
    # --------------------------------------------------------

    dates = pd.Series(
        pd.Timestamp.today(),
        index=np.arange(n),
    )

    first_date = (
        pd.Timestamp.today()
        .replace(
            month=1,
            day=1,
        )
        .normalize()
    )

    day_number = (
        dates - first_date
    ).dt.days.to_numpy()

    declination = 23.45 * np.sin(
        np.radians(
            360
            * (284 + day_number + 1)
            / 365
        )
    )

    elevation = (
        90
        - lat
        + declination
    )

    month_names = dates.dt.strftime("%B")

    tilt = np.array(
        [
            month_lookup.get(m, 0)
            for m in month_names
        ],
        dtype=float,
    )

    a_plus_b = elevation + tilt

    sin_ab = np.sin(
        np.radians(a_plus_b)
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    # Prevent divide-by-zero without changing normal daylight values
    safe_sin_a = np.where(
        np.abs(sin_a) < 1e-10,
        np.nan,
        sin_a,
    )

    poa = (
        ghi
        * sin_ab[:, None]
        / safe_sin_a[:, None]
    )

    return {
        "ghi": ghi,
        "actual": actual,
        "declination": declination,
        "elevation": elevation,
        "tilt": tilt,
        "sin_a": sin_a,
        "poa": poa,
    }


# ============================================================
# FIXED FORECAST
# ============================================================

def fixed_forecast(
    base,
    cluster_area,
):

    forecast = (
        np.nan_to_num(base["poa"], nan=0)
        @ cluster_area
    ) / 1_000_000

    return forecast


# ============================================================
# FIXED OPTIMIZATION
# ============================================================

@st.cache_data(show_spinner=False)
def optimize_fixed(
    df,
    df_w,
    base,
    error_min,
    error_max,
    error_step,
):

    errors = np.arange(
        error_min,
        error_max + error_step / 2,
        error_step,
    )

    actual = base["actual"]

    actual_peak = (
        np.max(actual)
        if len(actual)
        else 0
    )

    best_error = error_min
    best_score = np.inf
    best_forecast = None

    rows = []

    for error in errors:

        _, _, cluster_area = (
            calculate_cluster_area(
                df,
                df_w,
                float(error),
            )
        )

        forecast = fixed_forecast(
            base,
            cluster_area,
        )

        calculated_peak = (
            np.max(forecast)
            if len(forecast)
            else 0
        )

        peak_error = abs(
            calculated_peak
            - actual_peak
        )

        peak_error_pct = (
            peak_error
            / actual_peak
            * 100
            if actual_peak != 0
            else np.nan
        )

        rows.append({
            "Error %": error,
            "Calculated Peak": calculated_peak,
            "Actual Peak": actual_peak,
            "Peak Error": peak_error,
            "Peak Error %": peak_error_pct,
        })

        if peak_error < best_score:
            best_score = peak_error
            best_error = float(error)
            best_forecast = forecast.copy()

    return (
        best_error,
        best_forecast,
        pd.DataFrame(rows),
    )


# ============================================================
# TRACKING BASE
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_tracking_base(
    ghi_df,
    actual_df,
    backend_blocks,
):

    ghi = ghi_df[GHI_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0).to_numpy(float)

    actual = pd.to_numeric(
        actual_df["Actual"],
        errors="coerce",
    ).fillna(0).to_numpy(float)

    n = min(
        len(ghi),
        len(actual),
    )

    ghi = ghi[:n]
    actual = actual[:n]

    # --------------------------------------------------------
    # EXACT BLOCK LOGIC
    # --------------------------------------------------------

    blocks = None

    if backend_blocks:
        first = backend_blocks[0]

        if first is not None and len(first) >= n:
            blocks = first[:n]

    if blocks is None:
        blocks = np.arange(n, dtype=float)

    return {
        "ghi": ghi,
        "actual": actual,
        "blocks": blocks,
    }


# ============================================================
# TRACKING FORECAST
# ============================================================

def tracking_forecast(
    base,
    cluster_area,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
):

    blocks = base["blocks"]
    ghi = base["ghi"]

    # --------------------------------------------------------
    # Same formulas as Jupyter
    # --------------------------------------------------------

    if not (
        start_block
        < max_block
        < end_block
    ):
        return None

    den1 = (
        start_block
        - 1
        - max_block
    )

    den2 = (
        end_block
        + 1
        - max_block
    )

    if den1 == 0 or den2 == 0:
        return None

    m1 = 90 / den1
    m2 = 90 / den2

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
                & (zenith > west_limit)
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
        ghi * DHI / 100
    )

    dni = (
        ghi - dhi
    ) / cos_alpha[:, None]

    forecast = (
        dni @ cluster_area
    ) / 1_000_000

    if (
        np.isnan(forecast).any()
        or np.isinf(forecast).any()
    ):
        return None

    return {
        "forecast": forecast,
        "zenith": zenith,
        "panel": panel,
    }


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(show_spinner=False)
def optimize_tracking(
    base,
    cluster_area,
    bounds,
    maxiter,
    popsize,
    seed,
):

    actual_full = base["actual"]

    mask = actual_full != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual = actual_full[mask]

    actual_max = actual.max()
    actual_sum = actual.sum()

    if actual_max == 0 or actual_sum == 0:
        raise ValueError(
            "Actual data contains no usable non-zero values."
        )

    def objective(x):

        DHI = int(round(x[0]))
        start = int(round(x[1]))
        end = int(round(x[2]))
        max_block = int(round(x[3]))
        east = int(round(x[4]))
        west = int(round(x[5]))

        result = tracking_forecast(
            base,
            cluster_area,
            DHI,
            start,
            end,
            max_block,
            east,
            west,
        )

        if result is None:
            return 1e9

        prediction = result["forecast"][mask]

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

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=seed,
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

    return params, float(result.fun)


# ============================================================
# METRICS
# ============================================================

def metrics(forecast, actual):

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    n = min(
        len(forecast),
        len(actual),
    )

    forecast = forecast[:n]
    actual = actual[:n]

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

    peak_error = abs(
        forecast_peak
        - actual_peak
    )

    peak_error_pct = (
        peak_error
        / actual_peak
        * 100
        if actual_peak != 0
        else np.nan
    )

    energy_error_pct = (
        abs(
            forecast.sum()
            - actual.sum()
        )
        / actual.sum()
        * 100
        if actual.sum() != 0
        else np.nan
    )

    return {
        "Forecast Peak": forecast_peak,
        "Actual Peak": actual_peak,
        "Peak Error": peak_error,
        "Peak Error %": peak_error_pct,
        "Energy Error %": energy_error_pct,
    }


# ============================================================
# GRAPH
# ============================================================

def plot_forecast(
    forecast,
    actual,
    title,
):

    n = min(
        len(forecast),
        len(actual),
    )

    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast[:n],
            mode="lines",
            name="Forecast",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual[:n],
            mode="lines",
            name="Actual",
            line=dict(width=2),
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
        hovermode="x unified",
        xaxis_title="Block",
        yaxis_title="Power (MW)",
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
# HEADER
# ============================================================

st.title("☀️ Solar Forecast Correction")
st.caption(
    "Fixed / Tracking plant optimization with editable inputs"
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel Workbook",
    type=["xlsx", "xls"],
)


if uploaded_file is None:

    st.info(
        "Upload the Excel workbook to load GHI Forecast and Actual Power."
    )

    st.stop()


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    data = load_workbook(
        uploaded_file.getvalue()
    )

    plant_df, cluster_df = prepare_plant_data(
        data
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

c1, c2 = st.columns(2)

with c1:

    st.markdown("#### GHI Forecast")

    ghi_input = st.data_editor(
        data["ghi"].copy(),
        use_container_width=True,
        num_rows="fixed",
        key="ghi_input",
    )

with c2:

    st.markdown("#### Actual Power")

    actual_input = st.data_editor(
        data["actual"].copy(),
        use_container_width=True,
        num_rows="fixed",
        key="actual_input",
    )


# ============================================================
# VALIDATE INPUT
# ============================================================

missing_ghi = [
    c for c in GHI_COLS
    if c not in ghi_input.columns
]

if missing_ghi:

    st.error(
        "Missing GHI columns: "
        + ", ".join(missing_ghi)
    )

    st.stop()

if "Actual" not in actual_input.columns:

    st.error(
        "Actual column is required."
    )

    st.stop()


# ============================================================
# PLANT TYPE
# ============================================================

st.subheader("Plant Type")

plant_type = st.segmented_control(
    "Select Plant Type",
    ["Fixed", "Tracking"],
    default="Fixed",
)


# ============================================================
# PARAMETERS
# ============================================================

st.subheader("Optimization Parameters")


# ============================================================
# FIXED
# ============================================================

if plant_type == "Fixed":

    p1, p2, p3 = st.columns(3)

    with p1:
        error_min = st.number_input(
            "Error Min (%)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.1,
        )

    with p2:
        error_max = st.number_input(
            "Error Max (%)",
            min_value=0.1,
            max_value=50.0,
            value=10.0,
            step=0.1,
        )

    with p3:
        error_step = st.number_input(
            "Error Step (%)",
            min_value=0.01,
            max_value=5.0,
            value=0.1,
            step=0.1,
        )

    if error_max <= error_min:
        st.error(
            "Error Max must be greater than Error Min."
        )
        st.stop()

    # --------------------------------------------------------
    # Auto optimization
    # --------------------------------------------------------

    input_signature = (
        uploaded_file.name,
        len(ghi_input),
        len(actual_input),
        float(
            pd.to_numeric(
                ghi_input[GHI_COLS],
                errors="coerce",
            ).fillna(0).sum().sum()
        ),
        float(
            pd.to_numeric(
                actual_input["Actual"],
                errors="coerce",
            ).fillna(0).sum()
        ),
        error_min,
        error_max,
        error_step,
    )

    if st.session_state.get(
        "fixed_signature"
    ) != input_signature:

        with st.spinner(
            "Calculating best Error %..."
        ):

            base = prepare_fixed_base(
                ghi_input,
                actual_input,
                data["lat"],
                data["month_lookup"],
            )

            best_error, _, opt_table = (
                optimize_fixed(
                    plant_df,
                    cluster_df,
                    base,
                    error_min,
                    error_max,
                    error_step,
                )
            )

        st.session_state.fixed_signature = (
            input_signature
        )

        st.session_state.fixed_best_error = (
            best_error
        )

        st.session_state.fixed_opt_table = (
            opt_table
        )

    best_error = st.session_state.fixed_best_error

    # --------------------------------------------------------
    # Editable final Error %
    # --------------------------------------------------------

    selected_error = st.number_input(
        "Final Error (%)",
        min_value=error_min,
        max_value=error_max,
        value=float(best_error),
        step=error_step,
    )

    base = prepare_fixed_base(
        ghi_input,
        actual_input,
        data["lat"],
        data["month_lookup"],
    )

    _, eff_area, cluster_area = (
        calculate_cluster_area(
            plant_df,
            cluster_df,
            selected_error,
        )
    )

    forecast = fixed_forecast(
        base,
        cluster_area,
    )

    actual = base["actual"]

    result_metrics = metrics(
        forecast,
        actual,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.subheader("Results")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Error %",
        f"{selected_error:.2f}%",
    )

    m2.metric(
        "Forecast Peak",
        f"{result_metrics['Forecast Peak']:.3f}",
    )

    m3.metric(
        "Actual Peak",
        f"{result_metrics['Actual Peak']:.3f}",
    )

    m4.metric(
        "Peak Error",
        f"{result_metrics['Peak Error %']:.2f}%",
    )

    st.plotly_chart(
        plot_forecast(
            forecast,
            actual,
            "Fixed Plant: Forecast vs Actual",
        ),
        use_container_width=True,
    )

    with st.expander(
        "Optimization Details"
    ):

        st.dataframe(
            st.session_state.fixed_opt_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TRACKING
# ============================================================

else:

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        tracking_error = st.number_input(
            "Error (%)",
            min_value=0.0,
            max_value=50.0,
            value=4.9,
            step=0.1,
        )

    with c2:

        dhi_default = 1

        dhi_min = st.number_input(
            "DHI Min (%)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
        )

    # --------------------------------------------------------
    # Prepare base
    # --------------------------------------------------------

    tracking_base = prepare_tracking_base(
        ghi_input,
        actual_input,
        data["backend_blocks"],
    )

    # --------------------------------------------------------
    # Cluster effective area
    #
    # Error applied ONLY HERE
    # --------------------------------------------------------

    _, _, cluster_area = (
        calculate_cluster_area(
            plant_df,
            cluster_df,
            tracking_error,
        )
    )

    # --------------------------------------------------------
    # Optimization bounds
    # --------------------------------------------------------

    st.markdown("#### Tracking Optimization Bounds")

    b1, b2, b3 = st.columns(3)

    with b1:

        start_min = st.number_input(
            "Starting Block Min",
            0,
            95,
            10,
            1,
        )

        end_min = st.number_input(
            "Ending Block Min",
            1,
            96,
            65,
            1,
        )

    with b2:

        start_max = st.number_input(
            "Starting Block Max",
            1,
            95,
            30,
            1,
        )

        end_max = st.number_input(
            "Ending Block Max",
            1,
            96,
            80,
            1,
        )

    with b3:

        max_min = st.number_input(
            "Max Block Min",
            1,
            95,
            47,
            1,
        )

        max_max = st.number_input(
            "Max Block Max",
            1,
            95,
            53,
            1,
        )

    a1, a2 = st.columns(2)

    with a1:

        east_min = st.number_input(
            "East Limit Min",
            0,
            90,
            10,
            1,
        )

        east_max = st.number_input(
            "East Limit Max",
            1,
            90,
            70,
            1,
        )

    with a2:

        west_min = st.number_input(
            "West Limit Min",
            0,
            90,
            10,
            1,
        )

        west_max = st.number_input(
            "West Limit Max",
            1,
            90,
            70,
            1,
        )

    # --------------------------------------------------------
    # Optimizer settings
    # --------------------------------------------------------

    o1, o2, o3 = st.columns(3)

    with o1:

        maxiter = st.number_input(
            "Optimization Iterations",
            min_value=1,
            max_value=200,
            value=40,
            step=1,
        )

    with o2:

        popsize = st.number_input(
            "Population Size",
            min_value=3,
            max_value=50,
            value=15,
            step=1,
        )

    with o3:

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
        )

    # --------------------------------------------------------
    # Validate bounds
    # --------------------------------------------------------

    valid_bounds = (
        start_min < start_max
        and end_min < end_max
        and max_min < max_max
        and east_min < east_max
        and west_min < west_max
    )

    if not valid_bounds:

        st.error(
            "Invalid optimization bounds."
        )

        st.stop()

    # --------------------------------------------------------
    # Automatic tracking optimization
    # --------------------------------------------------------

    tracking_signature = (
        uploaded_file.name,
        len(ghi_input),
        len(actual_input),
        float(
            pd.to_numeric(
                ghi_input[GHI_COLS],
                errors="coerce",
            ).fillna(0).sum().sum()
        ),
        float(
            pd.to_numeric(
                actual_input["Actual"],
                errors="coerce",
            ).fillna(0).sum()
        ),
        tracking_error,
        start_min,
        start_max,
        end_min,
        end_max,
        max_min,
        max_max,
        east_min,
        east_max,
        west_min,
        west_max,
        maxiter,
        popsize,
        seed,
    )

    if st.session_state.get(
        "tracking_signature"
    ) != tracking_signature:

        bounds = [
            (dhi_min, 10),
            (start_min, start_max),
            (end_min, end_max),
            (max_min, max_max),
            (east_min, east_max),
            (west_min, west_max),
        ]

        with st.spinner(
            "Optimizing tracking parameters..."
        ):

            try:

                best_params, best_score = (
                    optimize_tracking(
                        tracking_base,
                        cluster_area,
                        bounds,
                        int(maxiter),
                        int(popsize),
                        int(seed),
                    )
                )

            except Exception as e:

                st.error(
                    f"Tracking optimization failed: {e}"
                )

                st.stop()

        st.session_state.tracking_signature = (
            tracking_signature
        )

        st.session_state.tracking_best = (
            best_params
        )

        st.session_state.tracking_score = (
            best_score
        )

    best = st.session_state.tracking_best

    # --------------------------------------------------------
    # Editable final parameters
    # --------------------------------------------------------

    st.markdown("#### Final Tracking Parameters")

    t1, t2, t3 = st.columns(3)

    with t1:

        DHI = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            value=int(best["DHI"]),
            step=1,
        )

        start_block = st.number_input(
            "GHI Starting Block",
            min_value=start_min,
            max_value=start_max,
            value=int(
                np.clip(
                    best["GHI Starting Block"],
                    start_min,
                    start_max,
                )
            ),
            step=1,
        )

    with t2:

        end_block = st.number_input(
            "GHI Ending Block",
            min_value=end_min,
            max_value=end_max,
            value=int(
                np.clip(
                    best["GHI Ending Block"],
                    end_min,
                    end_max,
                )
            ),
            step=1,
        )

        max_block = st.number_input(
            "GHI Max Block",
            min_value=max_min,
            max_value=max_max,
            value=int(
                np.clip(
                    best["GHI Max Block"],
                    max_min,
                    max_max,
                )
            ),
            step=1,
        )

    with t3:

        east_limit = st.number_input(
            "Tracking East Limit",
            min_value=east_min,
            max_value=east_max,
            value=int(
                np.clip(
                    best["Tracking East Limit"],
                    east_min,
                    east_max,
                )
            ),
            step=1,
        )

        west_limit = st.number_input(
            "Tracking West Limit",
            min_value=west_min,
            max_value=west_max,
            value=int(
                np.clip(
                    best["Tracking West Limit"],
                    west_min,
                    west_max,
                )
            ),
            step=1,
        )

    # --------------------------------------------------------
    # Final tracking calculation
    # --------------------------------------------------------

    if not (
        start_block
        < max_block
        < end_block
    ):

        st.error(
            "GHI Starting Block < GHI Max Block < GHI Ending Block is required."
        )

        st.stop()

    final = tracking_forecast(
        tracking_base,
        cluster_area,
        int(DHI),
        int(start_block),
        int(end_block),
        int(max_block),
        int(east_limit),
        int(west_limit),
    )

    if final is None:

        st.error(
            "Unable to calculate Tracking forecast with the selected parameters."
        )

        st.stop()

    forecast = final["forecast"]
    actual = tracking_base["actual"]

    result_metrics = metrics(
        forecast,
        actual,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.subheader("Results")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Error %",
        f"{tracking_error:.2f}%",
    )

    r2.metric(
        "Forecast Peak",
        f"{result_metrics['Forecast Peak']:.3f}",
    )

    r3.metric(
        "Actual Peak",
        f"{result_metrics['Actual Peak']:.3f}",
    )

    r4.metric(
        "Peak Error",
        f"{result_metrics['Peak Error %']:.2f}%",
    )

    st.plotly_chart(
        plot_forecast(
            forecast,
            actual,
            "Tracking Plant: Forecast vs Actual",
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Tracking angle graph
    # --------------------------------------------------------

    with st.expander(
        "Tracking Angles"
    ):

        fig_angle = go.Figure()

        fig_angle.add_trace(
            go.Scatter(
                x=np.arange(
                    len(final["zenith"])
                ),
                y=final["zenith"],
                mode="lines",
                name="Zenith Angle",
            )
        )

        fig_angle.add_trace(
            go.Scatter(
                x=np.arange(
                    len(final["panel"])
                ),
                y=final["panel"],
                mode="lines",
                name="Panel Angle",
            )
        )

        fig_angle.update_layout(
            height=350,
            xaxis_title="Block",
            yaxis_title="Angle (°)",
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_angle,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Optimization result
    # --------------------------------------------------------

    with st.expander(
        "Automatic Optimization Result"
    ):

        st.write(
            {
                "DHI": best["DHI"],
                "GHI Starting Block":
                    best["GHI Starting Block"],
                "GHI Ending Block":
                    best["GHI Ending Block"],
                "GHI Max Block":
                    best["GHI Max Block"],
                "Tracking East Limit":
                    best["Tracking East Limit"],
                "Tracking West Limit":
                    best["Tracking West Limit"],
                "Optimization Score":
                    round(
                        st.session_state.tracking_score,
                        6,
                    ),
            }
        )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Calculation uses the uploaded workbook as the default source. "
    "Edited GHI/Actual inputs are used for the current calculation."
)
