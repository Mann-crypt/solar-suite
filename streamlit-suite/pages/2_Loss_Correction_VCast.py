import io
import hashlib
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

POA_COLS = [
    "POA fixed",
    "POA Fixed-C12",
    "POA Fixed-C13",
    "POA Fixed-C14",
    "POA Fixed-C15",
]

POWER_COLS = [
    f"CL{i}_Fixed Power=I*Ƞ*A"
    for i in range(1, 6)
]

TOTAL_POWER = "Total Power (CL1+CL2+…)"


TRACKING_BOUNDS = [
    (0, 10),
    (10, 30),
    (65, 80),
    (47, 53),
    (10, 70),
    (10, 70),
]


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.block-container {
    max-width: 1500px;
    padding-top: 1rem;
}

.title {
    font-size: 30px;
    font-weight: 750;
}

.subtitle {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 15px;
}

.section {
    font-size: 19px;
    font-weight: 700;
    margin: 18px 0 9px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION
# ============================================================

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "input_df" not in st.session_state:
    st.session_state.input_df = None

if "results" not in st.session_state:
    st.session_state.results = None

if "plant" not in st.session_state:
    st.session_state.plant = "Fixed"


def reset():
    st.session_state.results = None


# ============================================================
# BASIC HELPERS
# ============================================================

def nums(x):
    return pd.to_numeric(x, errors="coerce").fillna(0)


def arr(x):
    return nums(pd.Series(x)).to_numpy(float)


def sf(x, default=0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def read(data, sheet, **kwargs):
    return pd.read_excel(
        io.BytesIO(data),
        sheet_name=sheet,
        **kwargs,
    )


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )
    return df


# ============================================================
# LOAD WORKBOOK
# ============================================================

@st.cache_data(show_spinner=False, max_entries=3)
def load_data(data):

    wb = {}

    wb["area"] = clean_columns(
        read(
            data,
            "Area & Efficiency",
            header=1,
            usecols=range(12),
        )
    )

    wb["cluster"] = clean_columns(
        read(
            data,
            "Area & Efficiency",
            header=1,
            usecols=[0, 14, 15],
        )
    )

    wb["ghi"] = clean_columns(
        read(
            data,
            "Result",
            usecols=[0, 1, 2, 3, 4, 5],
        )
    )

    wb["config"] = clean_columns(
        read(
            data,
            "Forecast Config",
            header=8,
        )
    )

    wb["tilt"] = clean_columns(
        read(
            data,
            "Config Tilt Angle",
            header=7,
        )
    )

    wb["fixed"] = clean_columns(
        read(
            data,
            "Fixed-C11",
            header=1,
        )
    )

    wb["backend"] = {}

    for c in CLUSTERS:
        wb["backend"][c] = clean_columns(
            read(
                data,
                f"Backend Cal {c}",
            )
        )

    return wb


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(wb):

    area = wb["area"].copy()

    required = [
        "Clusters",
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]

    for c in required:
        if c not in area.columns:
            raise ValueError(
                f"Missing '{c}' in Area & Efficiency"
            )

    for c in required[1:]:
        area[c] = nums(area[c])

    if "S.No." in area.columns:
        m = area["S.No."].isna()
        if m.any():
            area = area.iloc[:np.flatnonzero(m)[0]].copy()

    area["Total area (m2)"] = (
        area["No of Module"]
        * area["Area of 1 Module (m2)"]
    )

    cluster = wb["cluster"].copy()

    if "Clusters" not in cluster.columns:
        raise ValueError(
            "Clusters column missing."
        )

    cluster = cluster.dropna(
        subset=["Clusters"]
    ).reset_index(drop=True)

    ghi = wb["ghi"].copy()

    for c in GHI_COLS:
        if c not in ghi.columns:
            raise ValueError(
                f"Missing '{c}' in Result"
            )
        ghi[c] = nums(ghi[c])

    fixed = wb["fixed"].copy()

    if "Actual" not in fixed.columns:
        raise ValueError(
            "Actual column missing in Fixed-C11."
        )

    fixed["Actual"] = nums(
        fixed["Actual"]
    )

    n = min(
        len(ghi),
        len(fixed),
    )

    inp = ghi[GHI_COLS].iloc[:n].copy()

    inp["Actual"] = (
        fixed["Actual"]
        .iloc[:n]
        .to_numpy()
    )

    # Latitude
    if "Lat" not in wb["config"].columns:
        raise ValueError(
            "Lat column missing."
        )

    lat = sf(
        wb["config"]["Lat"].iloc[0],
        None,
    )

    if lat is None:
        raise ValueError(
            "Invalid latitude."
        )

    # Tilt
    tilt = {}

    month_col = next(
        (
            c for c in wb["tilt"].columns
            if str(c).strip().lower() == "month"
        ),
        None,
    )

    if month_col is None:
        raise ValueError(
            "Month column missing in Config Tilt Angle."
        )

    if "Fixed" not in wb["tilt"].columns:
        raise ValueError(
            "Fixed column missing in Config Tilt Angle."
        )

    for _, r in wb["tilt"].iterrows():

        m = str(r[month_col]).strip()

        if m:
            tilt[m] = sf(
                r["Fixed"],
                0,
            )

    return {
        "area": area,
        "cluster": cluster,
        "ghi": ghi.iloc[:n].reset_index(drop=True),
        "fixed": fixed.iloc[:n].reset_index(drop=True),
        "input": inp.reset_index(drop=True),
        "lat": lat,
        "tilt": tilt,
    }


# ============================================================
# GEOMETRY
# ============================================================

def geometry(ghi, fixed, lat, tilt):

    out = fixed.copy()

    n = min(
        len(out),
        len(ghi),
    )

    out = out.iloc[:n].copy()
    ghi = ghi.iloc[:n]

    today = pd.Timestamp.today()

    doy = today.dayofyear

    declination = (
        23.45
        * np.sin(
            np.radians(
                360 * (284 + doy) / 365
            )
        )
    )

    elevation = 90 - lat + declination

    month = today.strftime("%B")

    tilt_value = sf(
        tilt.get(month, 0),
        0,
    )

    out["Date"] = today
    out["Declination Angle ∆"] = declination
    out["Elevation angle a"] = elevation
    out["Tilt Angle b"] = tilt_value

    out["a+b"] = (
        elevation + tilt_value
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(
            elevation + tilt_value
        )
    )

    safe_sin_a = (
        sin_a
        if abs(sin_a) > 1e-8
        else 1e-8
    )

    out["SIN(a+b)"] = sin_ab
    out["Sin(a)"] = sin_a

    for i, c in enumerate(GHI_COLS):

        suffix = (
            ""
            if i == 0
            else f"-CL{i + 1}"
        )

        out[
            f"GHI*sin(a+b){suffix}"
        ] = (
            ghi[c].to_numpy()
            * sin_ab
        )

        out[
            f"GHI*sin(a){suffix}"
        ] = (
            ghi[c].to_numpy()
            * sin_a
        )

        out[
            POA_COLS[i]
        ] = (
            out[
                f"GHI*sin(a+b){suffix}"
            ]
            / safe_sin_a
        )

    return out


# ============================================================
# AREA + POWER
# ============================================================

def calculate_power(
    area,
    cluster,
    geometry_df,
    error,
):

    a = area.copy()
    c = cluster.copy()

    error = sf(error)

    a["Error %"] = error

    a["Net Efficiency (%)"] = (
        nums(a["Standard PV Efficiency (%)"])
        - error
    )

    a["Eff Area"] = (
        a["Net Efficiency (%)"]
        * nums(a["Total area (m2)"])
        / 100
    )

    sums = (
        a.groupby("Clusters")["Eff Area"]
        .sum()
    )

    c["Eff Area(m2)"] = (
        c["Clusters"]
        .map(sums)
        .fillna(0)
    )

    result = geometry_df.copy()

    if len(c) < 5:
        raise ValueError(
            "Five clusters are required."
        )

    for i in range(5):

        poa = POA_COLS[i]

        if poa not in result:
            raise ValueError(
                f"Missing POA '{poa}'."
            )

        result[POWER_COLS[i]] = (
            nums(result[poa])
            * sf(c.iloc[i]["Eff Area(m2)"])
            / 1_000_000
        )

    result[TOTAL_POWER] = (
        result[POWER_COLS]
        .sum(axis=1)
    )

    return a, c, result


# ============================================================
# AUTOMATIC ERROR %
# ============================================================

def find_best_error(
    area,
    cluster,
    geometry_df,
):

    actual = arr(
        geometry_df["Actual"]
    )

    peak = actual.max()

    if peak <= 0:
        raise ValueError(
            "Actual peak must be greater than zero."
        )

    best_error = 0
    best_difference = np.inf
    table = []

    # Keep automatic calculation lightweight
    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        _, c, result = calculate_power(
            area,
            cluster,
            geometry_df,
            error,
        )

        forecast_peak = arr(
            result[TOTAL_POWER]
        ).max()

        difference = abs(
            forecast_peak - peak
        )

        table.append(
            [
                round(error, 1),
                forecast_peak,
                peak,
                difference,
                difference / peak * 100,
            ]
        )

        if difference < best_difference:
            best_difference = difference
            best_error = error

    table = pd.DataFrame(
        table,
        columns=[
            "Error %",
            "Calculated Peak",
            "Actual Peak",
            "Peak Error",
            "Peak Error %",
        ],
    )

    return best_error, table


# ============================================================
# TRACKING
# ============================================================

def tracking_arrays(
    wb,
    ghi,
    fixed,
    cluster,
):

    backend = wb["backend"]["C11"]

    if "Block No." not in backend.columns:
        raise ValueError(
            "Block No. missing in Backend Cal C11."
        )

    n = min(
        len(backend),
        len(ghi),
        len(fixed),
    )

    blocks = arr(
        backend["Block No."]
    )[:n]

    matrix = np.column_stack([
        arr(ghi[c])[:n]
        for c in GHI_COLS
    ])

    actual = arr(
        fixed["Actual"]
    )[:n]

    weights = pd.to_numeric(
        cluster["Eff Area(m2)"].iloc[:5],
        errors="coerce",
    ).fillna(0).to_numpy(float)

    return blocks, matrix, actual, weights


def tracking_forecast(
    params,
    blocks,
    ghi,
    weights,
):

    dhi, start, end, maximum, east, west = (
        params
    )

    if not start < maximum < end:
        return None

    d1 = start - maximum - 1
    d2 = end - maximum + 1

    if d1 == 0 or d2 == 0:
        return None

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
            zenith > west,
            west,
            zenith,
        ),
    )

    cos_panel = np.clip(
        np.cos(np.radians(panel)),
        1e-6,
        None,
    )

    dhi_part = ghi * dhi / 100

    dni = (
        ghi - dhi_part
    ) / cos_panel[:, None]

    power = (
        dni
        * weights[None, :]
        / 1_000_000
    )

    return power.sum(axis=1)


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    blocks,
    ghi,
    actual,
    weights,
):

    mask = actual != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values."
        )

    actual_day = actual[mask]

    peak = actual_day.max()
    energy = actual_day.sum()

    def objective(x):

        x = np.rint(x).astype(int)

        forecast = tracking_forecast(
            x,
            blocks,
            ghi,
            weights,
        )

        if forecast is None:
            return 1e9

        if not np.all(
            np.isfinite(forecast)
        ):
            return 1e9

        pred = forecast[mask]

        block_error = (
            np.mean(
                np.abs(
                    actual_day - pred
                )
            )
            / peak
        )

        peak_error = (
            abs(
                peak - pred.max()
            )
            / peak
        )

        energy_error = (
            abs(
                energy - pred.sum()
            )
            / energy
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(
        objective,
        TRACKING_BOUNDS,
        maxiter=25,
        popsize=8,
        tol=0.002,
        seed=42,
        polish=False,
        workers=1,
    )

    x = np.rint(
        result.x
    ).astype(int)

    return x


# ============================================================
# MASTER CALCULATION
# ============================================================

def run_calculation(
    wb,
    prepared,
    edited_input,
    plant,
):

    # Apply edited values
    n = len(edited_input)

    ghi = prepared["ghi"].iloc[:n].copy()
    fixed = prepared["fixed"].iloc[:n].copy()

    for c in GHI_COLS:
        ghi[c] = nums(
            edited_input[c]
        ).to_numpy()

    fixed["Actual"] = nums(
        edited_input["Actual"]
    ).to_numpy()

    # Geometry
    geo = geometry(
        ghi,
        fixed,
        prepared["lat"],
        prepared["tilt"],
    )

    # Automatic Error %
    best_error, error_table = (
        find_best_error(
            prepared["area"],
            prepared["cluster"],
            geo,
        )
    )

    # Final automatic power
    final_area, final_cluster, fixed_result = (
        calculate_power(
            prepared["area"],
            prepared["cluster"],
            geo,
            best_error,
        )
    )

    tracking_params = None
    tracking_result = None

    if plant == "Tracking":

        blocks, matrix, actual, weights = (
            tracking_arrays(
                wb,
                ghi,
                geo,
                final_cluster,
            )
        )

        tracking_params = optimize_tracking(
            blocks,
            matrix,
            actual,
            weights,
        )

        tracking_result = tracking_forecast(
            tracking_params,
            blocks,
            matrix,
            weights,
        )

    return {
        "area": prepared["area"],
        "cluster": prepared["cluster"],
        "geo": geo,
        "ghi": ghi,
        "fixed": fixed,
        "best_error": best_error,
        "error_table": error_table,
        "final_area": final_area,
        "final_cluster": final_cluster,
        "fixed_result": fixed_result,
        "tracking_params": tracking_params,
        "tracking_result": tracking_result,
    }


# ============================================================
# GRAPH
# ============================================================

def graph(actual, forecast, title):

    n = min(
        len(actual),
        len(forecast),
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual[:n],
            name="Actual",
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=forecast[:n],
            name="Forecast",
            mode="lines",
        )
    )

    fig.update_layout(
        title=title,
        height=430,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Automatic optimization with editable parameters'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# FILE
# ============================================================

st.markdown(
    '<div class="section">📂 Input Data</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Solar Excel File",
    type=["xlsx", "xls"],
)


if uploaded is None:
    st.info(
        "Upload the Solar Excel file to start."
    )
    st.stop()


# ============================================================
# FILE HASH
# ============================================================

file_hash = hashlib.sha256(
    uploaded.getvalue()
).hexdigest()

if file_hash != st.session_state.file_hash:

    st.session_state.file_hash = file_hash
    st.session_state.input_df = None
    st.session_state.results = None


# ============================================================
# LOAD
# ============================================================

try:

    wb = load_data(
        uploaded.getvalue()
    )

    if st.session_state.input_df is None:

        prepared = prepare_input(wb)

        st.session_state.input_df = (
            prepared["input"]
        )

    else:

        prepared = prepare_input(wb)

except Exception as e:

    st.error(
        f"Input preparation failed: {e}"
    )

    st.stop()


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section">✏️ GHI / Actual Input</div>',
    unsafe_allow_html=True,
)

edited_input = st.data_editor(
    st.session_state.input_df,
    width="stretch",
    height=300,
    num_rows="fixed",
    hide_index=True,
    column_config={
        c: st.column_config.NumberColumn(
            c,
            format="%.3f",
        )
        for c in GHI_COLS + ["Actual"]
    },
    key="input_editor",
)


# ============================================================
# PLANT
# ============================================================

st.markdown(
    '<div class="section">🌱 Plant Type</div>',
    unsafe_allow_html=True,
)

plant = st.segmented_control(
    "Plant Type",
    ["Fixed", "Tracking"],
    default=st.session_state.plant,
    selection_mode="single",
    width="stretch",
    label_visibility="collapsed",
)

if plant is None:
    plant = "Fixed"

if plant != st.session_state.plant:

    st.session_state.plant = plant
    st.session_state.results = None


# ============================================================
# RUN
# ============================================================

if st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    width="stretch",
):

    try:

        with st.spinner(
            "Calculating..."
        ):

            st.session_state.results = (
                run_calculation(
                    wb,
                    prepared,
                    edited_input,
                    plant,
                )
            )

        st.success(
            "Calculation completed successfully."
        )

    except Exception as e:

        st.session_state.results = None

        st.error(
            f"Calculation failed: {e}"
        )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.results is None:

    st.caption(
        "Edit the input data, select Fixed or Tracking, "
        "then run the calculation."
    )

    st.stop()


data = st.session_state.results


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)

error = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=20.0,
    value=float(
        data["best_error"]
    ),
    step=0.1,
)


tracking_params = None

if plant == "Tracking":

    p = data["tracking_params"]

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi = st.number_input(
            "DHI (%)",
            0,
            100,
            int(p[0]),
        )

        start = st.number_input(
            "GHI Starting Block",
            0,
            95,
            int(p[1]),
        )

    with c2:

        end = st.number_input(
            "GHI Ending Block",
            1,
            96,
            int(p[2]),
        )

        maximum = st.number_input(
            "GHI Max Block",
            0,
            95,
            int(p[3]),
        )

    with c3:

        east = st.number_input(
            "Tracking East Limit",
            0,
            90,
            int(p[4]),
        )

        west = st.number_input(
            "Tracking West Limit",
            0,
            90,
            int(p[5]),
        )

    if not start < maximum < end:

        st.error(
            "Required: Starting Block < Max Block < Ending Block."
        )

        st.stop()

    _, final_cluster, fixed_result = (
        calculate_power(
            data["area"],
            data["cluster"],
            data["geo"],
            error,
        )
    )

    blocks, matrix, actual, weights = (
        tracking_arrays(
            wb,
            data["ghi"],
            data["geo"],
            final_cluster,
        )
    )

    forecast = tracking_forecast(
        np.array([
            dhi,
            start,
            end,
            maximum,
            east,
            west,
        ]),
        blocks,
        matrix,
        weights,
    )

    actual_values = actual

    title = (
        "Tracking Plant | Actual vs Forecast"
    )

else:

    _, _, fixed_result = (
        calculate_power(
            data["area"],
            data["cluster"],
            data["geo"],
            error,
        )
    )

    actual_values = arr(
        data["geo"]["Actual"]
    )

    forecast = arr(
        fixed_result[TOTAL_POWER]
    )

    title = (
        "Fixed Plant | Actual vs Forecast"
    )


# ============================================================
# RESULTS
# ============================================================

actual_peak = (
    actual_values.max()
    if len(actual_values)
    else 0
)

forecast_peak = (
    forecast.max()
    if len(forecast)
    else 0
)

st.markdown(
    '<div class="section">📊 Results</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

c1.metric(
    "Actual Peak",
    f"{actual_peak:.3f}",
)

c2.metric(
    "Forecast Peak",
    f"{forecast_peak:.3f}",
)


# ============================================================
# GRAPH
# ============================================================

st.markdown(
    '<div class="section">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)

st.plotly_chart(
    graph(
        actual_values,
        forecast,
        title,
    ),
    width="stretch",
    config={
        "displayModeBar": False,
        "responsive": True,
    },
)
