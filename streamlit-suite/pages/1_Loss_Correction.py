import io
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Loss Correction — Solar Suite",
    page_icon="⚡",
    layout="wide",
)


# ==========================================================
# IDLE RELOAD
# ==========================================================

st.components.v1.html("""
<script>
let t;
function r(){
    clearTimeout(t);
    t=setTimeout(()=>location.reload(),300000);
}
["mousemove","mousedown","keydown","scroll","touchstart"]
.forEach(e=>document.addEventListener(e,r));
r();
</script>
""", height=0)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("""
<h1 style='text-align:center;
background:linear-gradient(90deg,#00c6ff,#0072ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
font-size:40px;font-weight:800;'>
⚡ Solar Suite
</h1>
<p style='text-align:center;color:gray;font-size:14px;'>
Forecast Correction Platform
</p>
""", unsafe_allow_html=True)

st.sidebar.divider()


# ==========================================================
# STYLE
# ==========================================================

st.markdown("""
<style>
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{
    font-size:13px;
}
div[data-testid="metric-container"]{
    background:#111827;
    border:1px solid #1f2937;
    border-radius:10px;
    padding:12px 20px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# QUOTES
# ==========================================================

QUOTES = [
    "☕ Vo kehte the kya ho tum, aaj hum kehte hai tum kya ho be?",
    "🌦 Aapka mann nahi kar raha bahar jaane ka?..",
    "😊 Jinke ghar sheeshe ke bane hote hai vo basement mai kapde change krte h...",
    "😋 Aromatic Rose Latte with Frothy Milk pine ka mann hor hai na...",
    "🥛 Garmi mai daalo dudh mai Ice🧊 Dudh bangya Very Nice - Dudh Dudh Dudh Dudh...",
    "🌟 Aapke face pr toh Modiji se bhi jyda glow hai..",
    "😁 Horaha hai benstokes Kaan mai ghusjao insaan ke...",
    "😗 Muskuraiye aap MAL mai hai...",
    "🥱 Hum na hote toh Operations ka kya hota?..",
    "😎 6:30 hote hi Billu MAL se faraar...",
    "😇 Guruji ne ek baat kahi thi....",
    "🎼 Karna hai kuchh kaam M se gaao...",
    "😠 Nahi karni Loss Correction, Now what to do?...",
    "💸 Iss Job ko chhod or chhod kar ameer ho..",
]


# ==========================================================
# CACHED EXCEL READERS
# ==========================================================

@st.cache_data(show_spinner=False)
def excel_sheets(file_bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


@st.cache_data(show_spinner=False)
def detect_workbook(file_bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))

    cluster = "Fixed-CL1" in xls.sheet_names

    sheet = "Fixed-CL1" if cluster else "Fixed"

    ghi_cols = (
        ["CL1-GHI", "CL2-GHI", "CL3-GHI", "CL4-GHI", "CL5-GHI"]
        if cluster else
        ["GHI_Forecast"]
    )

    df = xls.parse(sheet_name=sheet, header=1)
    df.columns = df.columns.str.strip()

    df["Actual"] = pd.to_numeric(
        df["Actual"], errors="coerce"
    ).fillna(0)

    valid = df["Date"].notna()

    if valid.any():
        df = df.loc[:valid.idxmax() - 1] if not valid.all() else df

    df = df.iloc[:96]

    return cluster, ghi_cols, df[ghi_cols + ["Actual"]].copy()


@st.cache_data(show_spinner=False)
def read_area_efficiency(file_bytes):
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(8),
    )

    df.columns = df.columns.str.strip()

    valid = df["Module Type"].notna()

    return df.loc[valid].copy()


@st.cache_data(show_spinner=False)
def read_weights(file_bytes):
    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[12, 13, 14, 15, 16],
    )


@st.cache_data(show_spinner=False)
def read_forecast_config(file_bytes):
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )
    return float(df.loc[0, "Lat"])


@st.cache_data(show_spinner=False)
def read_tilt(file_bytes):
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df.columns = df.columns.str.strip()
    df["Fixed"] = df["Fixed"].fillna(0)

    valid = df["Fixed"] != 0
    df = df.loc[valid].dropna(how="all", axis=1)

    df = df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    return df.set_index("Month")["Fixed"].to_dict()


@st.cache_data(show_spinner=False)
def read_backend_cal(file_bytes, sheet):
    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet,
    )


@st.cache_data(show_spinner=False)
def read_tracking_sheet(file_bytes):
    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Tracking",
        header=1,
    )


# ==========================================================
# DATA PREPARATION
# ==========================================================

@st.cache_data(show_spinner=False)
def read_calculation_sheet(file_bytes, sheet):
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet,
        header=1,
    )

    df.columns = df.columns.str.strip()

    valid = df["Date"].notna()

    if valid.any():
        df = df.loc[valid].copy()

    return df.iloc[:96].copy()


def apply_editor_data(df, edited_df, ghi_cols):
    df = df.copy()

    for col in ghi_cols:
        df[col] = pd.to_numeric(
            edited_df[col].values[:len(df)],
            errors="coerce",
        )

    df["Actual"] = pd.to_numeric(
        edited_df["Actual"].values[:len(df)],
        errors="coerce",
    ).fillna(0)

    return df.iloc[:96].copy()


# ==========================================================
# SOLAR GEOMETRY
# ==========================================================

def add_geometry(df, lat, tilt):
    df = df.copy()

    today = pd.Timestamp.today().normalize()
    first = today.replace(month=1, day=1)

    days = (today - first).days + 1
    declination = 23.45 * np.sin(
        np.radians(360 * (284 + days) / 365)
    )

    elevation = 90 - lat + declination

    df["Date"] = today
    df["Elevation angle a"] = elevation

    if isinstance(tilt, dict):
        df["Tilt Angle b"] = (
            today.strftime("%B")
        )
        df["Tilt Angle b"] = df["Tilt Angle b"].map(tilt)
    else:
        df["Tilt Angle b"] = tilt

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

    return df


# ==========================================================
# POA
# ==========================================================

def add_poa(df, ghi_cols, cluster):
    df = df.copy()

    suffixes = (
        ["", "-CL2", "-CL3", "-CL4", "-CL5"]
        if cluster else [""]
    )

    for ghi, suffix in zip(ghi_cols, suffixes):
        df[f"GHI*sin(a){suffix}"] = (
            df[ghi] * df["Sin(a)"]
        )

        df[f"GHI*sin(a+b){suffix}"] = (
            df[ghi] * df["SIN(a+b)"]
        )

        df[f"POA fixed{suffix}"] = (
            df[f"GHI*sin(a+b){suffix}"]
            / df["Sin(a)"]
        )

    return df


# ==========================================================
# EFFICIENCY LOSS OPTIMIZATION
# ==========================================================

def optimize_loss(
    area_df,
    actual,
    poa,
    cluster=False,
    weights=None,
):
    std_eff = area_df[
        "Standard PV Efficiency (%)"
    ].to_numpy(float)

    area = area_df[
        "Total area(m2)"
    ].to_numpy(float)

    actual_peak = actual.max()

    losses = np.arange(
        0,
        std_eff.min() + 0.01,
        0.1,
    )

    best_loss = 0
    best_error = np.inf

    for loss in losses:

        eff_area = (
            area
            * (std_eff - loss)
            / 100
        )

        if cluster:

            pred = np.zeros(len(actual))

            for i in range(5):
                pred += (
                    poa[i]
                    * eff_area[i]
                    * weights[i]
                    / 1e6
                )

        else:

            pred = (
                poa
                * eff_area.sum()
                / 1e6
            )

        error = abs(
            actual_peak - pred.max()
        )

        if error < best_error:
            best_error = error
            best_loss = loss

    return float(best_loss)


def apply_efficiency(area_df, loss):
    df = area_df.copy()

    df["Efficiency Losses(%)"] = loss

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - loss
    )

    df["Eff Area"] = (
        df["Total area(m2)"]
        * df["Net Efficiency (%)"]
        / 100
    )

    return df


# ==========================================================
# EFFICIENCY DISPLAY
# ==========================================================

def show_efficiency(df):
    cols = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)",
    ]

    display = df[cols].copy()

    numeric = display.select_dtypes(
        include=np.number
    ).columns

    display[numeric] = display[numeric].round(2)

    with st.expander(
        "🔍 View Efficiency Calculations"
    ):
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# CHART
# ==========================================================

def make_chart(forecast, actual):
    x = np.arange(
        1,
        len(actual) + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line={
                "color": "#00c6ff",
                "width": 3,
            },
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line={
                "color": "#ef4444",
                "width": 3,
            },
        )
    )

    fig.update_layout(
        title="Forecast vs Actual Power",
        template="streamlit",
        height=500,
        hovermode="x unified",
        xaxis={
            "title": "15 Minute Block",
            "dtick": 4,
        },
        yaxis={
            "title": "Power (MW)"
        },
        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 0,
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20,
        },
    )

    return fig


# ==========================================================
# TRACKING OPTIMIZATION
# ==========================================================

def optimize_tracking(
    actual,
    blocks,
    ghi_arrays,
    weights,
    cluster=True,
):
    mask = actual != 0

    actual_m = actual[mask]

    bounds = [
        (0, 10),   # DHI
        (0, 30),   # Start
        (65, 80),  # End
        (44, 60),  # Max
        (0, 70),   # East
        (0, 70),   # West
    ]

    def objective(x):

        try:

            DHI, start, end, maximum, east, west = (
                int(round(v)) for v in x
            )

            if (
                start >= maximum
                or maximum >= end
            ):
                return 1e9

            m1 = 90 / (
                start - 1 - maximum
            )

            m2 = 90 / (
                end + 1 - maximum
            )

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
                    (blocks > maximum)
                    & (zenith > west),
                    west,
                    zenith,
                ),
            )

            cos_a = np.clip(
                np.cos(
                    np.radians(panel)
                ),
                1e-6,
                None,
            )

            prediction = np.zeros_like(
                blocks,
                dtype=float,
            )

            for ghi, weight in zip(
                ghi_arrays,
                weights,
            ):
                dhi = ghi * DHI / 100

                prediction += (
                    ((ghi - dhi) / cos_a)
                    * weight
                    / 1e6
                )

            prediction = prediction[mask]

            if (
                len(prediction) == 0
                or np.isnan(prediction).any()
                or np.isinf(prediction).any()
            ):
                return 1e9

            peak = actual_m.max()

            if peak == 0:
                return 1e9

            block_error = (
                np.mean(
                    np.abs(
                        actual_m - prediction
                    )
                ) / peak
            )

            peak_error = abs(
                peak - prediction.max()
            ) / peak

            energy_error = abs(
                actual_m.sum()
                - prediction.sum()
            ) / actual_m.sum()

            return (
                0.80 * block_error
                + 0.10 * peak_error
                + 0.10 * energy_error
            )

        except Exception:
            return 1e9

    progress = st.progress(0)
    status = st.empty()

    state = {
        "generation": 0,
        "quote": None,
    }

    max_iter = 100

    def callback(xk, convergence):

        state["generation"] += 1

        progress.progress(
            min(
                state["generation"]
                / max_iter,
                1.0,
            )
        )

        if (
            state["generation"] % 20 == 1
            or state["quote"] is None
        ):
            choices = [
                q for q in QUOTES
                if q != state["quote"]
            ]

            state["quote"] = random.choice(
                choices
            )

        status.info(
            f"{state['quote']}\n\n"
            f"Generation "
            f"{state['generation']} / {max_iter}"
        )

    status.info(
        random.choice(QUOTES)
    )

    with st.spinner(
        "Ho raha hai aap tab tak saath waale "
        "se baat karlo...🗣"
    ):
        result = differential_evolution(
            objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=max_iter,
            popsize=15,
            tol=0.001,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            polish=True,
            workers=1,
            callback=callback,
        )

    progress.empty()
    status.success(
        "✅ Dekha Kitni Jaldi Hogaya!"
    )

    return np.round(
        result.x
    ).astype(int)


# ==========================================================
# TRACKING FORECAST
# ==========================================================

def tracking_forecast(
    ghi_arrays,
    blocks,
    weights,
    DHI,
    start,
    end,
    maximum,
    east,
    west,
):
    m1 = 90 / (
        start - 1 - maximum
    )

    m2 = 90 / (
        end + 1 - maximum
    )

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
            (blocks > maximum)
            & (zenith > west),
            west,
            zenith,
        ),
    )

    cos_a = np.clip(
        np.cos(
            np.radians(panel)
        ),
        1e-6,
        None,
    )

    forecast = np.zeros(
        len(blocks),
        dtype=float,
    )

    for ghi, weight in zip(
        ghi_arrays,
        weights,
    ):
        dhi = ghi * DHI / 100

        forecast += (
            ((ghi - dhi) / cos_a)
            * weight
            / 1e6
        )

    return forecast


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "Pakima Pakam Ravi, 3-4 Loss Correction "
    "kar chuke hai!! 😎"
)


# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Yaha Feko!!",
    type=["xlsx"],
    key="lc_uploader",
)

if uploaded_file is None:
    st.info(
        "Pehle File toh upload karo!!!"
    )
    st.stop()

file_bytes = uploaded_file.getvalue()


# ==========================================================
# WORKBOOK DETECTION
# ==========================================================

is_cluster, ghi_cols, raw_df = (
    detect_workbook(file_bytes)
)


# ==========================================================
# INPUT EDITOR
# ==========================================================

st.subheader("Input Data")

original_df = raw_df.copy()

edited_df = st.data_editor(
    raw_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="lc_editor",
)

for col in ghi_cols:
    edited_df[col] = pd.to_numeric(
        edited_df[col],
        errors="coerce",
    ).fillna(0)

edited_df["Actual"] = pd.to_numeric(
    edited_df["Actual"],
    errors="coerce",
).fillna(0)

edited_df = (
    edited_df
    .iloc[:96]
    .reset_index(drop=True)
)

changed = (
    edited_df
    .ne(original_df.fillna(0))
    .any(axis=1)
)

if changed.any():
    st.toast(
        f"✨ {changed.sum()} rows updated successfully!",
        icon="✅",
    )


# ==========================================================
# PLANT TYPE
# ==========================================================

plant_type = st.pills(
    "Select Plant Type",
    ["🏗️ Fixed", "🔄 Tracking"],
    default="🏗️ Fixed",
)


# ==========================================================
# RUN
# ==========================================================

if "lc_run" not in st.session_state:
    st.session_state.lc_run = False

if st.button(
    "🚀 Dabao magar pyaar se!!",
    use_container_width=True,
    type="primary",
):
    st.session_state.lc_run = True
    st.session_state.pop(
        "lc_params",
        None,
    )

if not st.session_state.lc_run:
    st.stop()


# ==========================================================
# COMMON INPUTS
# ==========================================================

area_df = read_area_efficiency(
    file_bytes
)

lat = read_forecast_config(
    file_bytes
)

sheet = (
    "Fixed-CL1"
    if is_cluster
    else "Fixed"
)

df_fix = read_calculation_sheet(
    file_bytes,
    sheet,
)

df_fix = apply_editor_data(
    df_fix,
    edited_df,
    ghi_cols,
)

actual = df_fix[
    "Actual"
].to_numpy(float)


# ==========================================================
# FIXED PLANT
# ==========================================================

if plant_type == "🏗️ Fixed":

    tilt = (
        read_tilt(file_bytes)
        if not is_cluster
        else read_tilt(file_bytes)
    )

    df_fix = add_geometry(
        df_fix,
        lat,
        tilt,
    )

    df_fix = add_poa(
        df_fix,
        ghi_cols,
        is_cluster,
    )

    if is_cluster:

        weight_df = read_weights(
            file_bytes
        )

        weights = np.array([
            weight_df[f"CL-{i}"]
            .iloc[0]
            for i in range(1, 6)
        ])

        poa = [
            df_fix[
                f"POA fixed{s}"
            ].to_numpy(float)
            for s in [
                "",
                "-CL2",
                "-CL3",
                "-CL4",
                "-CL5",
            ]
        ]

    else:

        weight_df = None

        weights = None

        poa = df_fix[
            "POA fixed"
        ].to_numpy(float)

    best_loss = optimize_loss(
        area_df,
        actual,
        poa,
        cluster=is_cluster,
        weights=weights,
    )

    area_df = apply_efficiency(
        area_df,
        best_loss,
    )

    if is_cluster:

        forecast = np.zeros(
            len(df_fix)
        )

        for i in range(5):
            forecast += (
                poa[i]
                * area_df[
                    "Eff Area"
                ].iloc[i]
                * weights[i]
                / 1e6
            )

    else:

        forecast = (
            poa
            * area_df[
                "Eff Area"
            ].sum()
            / 1e6
        )

    st.metric(
        "Efficiency Loss",
        f"{best_loss:.2f}%",
    )

    show_efficiency(
        area_df
    )

    st.plotly_chart(
        make_chart(
            forecast,
            actual,
        ),
        use_container_width=True,
    )


# ==========================================================
# TRACKING PLANT
# ==========================================================

else:

    df_fix = add_geometry(
        df_fix,
        lat,
        0,
    )

    df_fix = add_poa(
        df_fix,
        ghi_cols,
        False,
    )

    if is_cluster:

        weight_df = read_weights(
            file_bytes
        )

        weights_area = np.array([
            weight_df[f"CL-{i}"]
            .iloc[0]
            for i in range(1, 6)
        ])

        ghi_arrays = [
            df_fix[col]
            .to_numpy(float)
            for col in ghi_cols
        ]

        area_poa = [
            df_fix[
                f"POA fixed{s}"
            ].to_numpy(float)
            for s in [
                "",
                "-CL2",
                "-CL3",
                "-CL4",
                "-CL5",
            ]
        ]

        best_loss = optimize_loss(
            area_df,
            actual,
            area_poa,
            cluster=True,
            weights=weights_area,
        )

        area_df = apply_efficiency(
            area_df,
            best_loss,
        )

        weights = np.array([
            area_df[
                "Eff Area"
            ].iloc[i]
            * weights_area[i]
            for i in range(5)
        ])

        backend = read_backend_cal(
            file_bytes,
            "Tracking",
        ) if "Tracking" else None

        tracking_sheet = (
            read_tracking_sheet(
                file_bytes
            )
        )

        block_df = read_backend_cal(
            file_bytes,
            "Backend Cal CL1",
        )

        blocks = block_df[
            "Block No."
        ].to_numpy(float)

    else:

        best_loss = optimize_loss(
            area_df,
            actual,
            df_fix[
                "POA fixed"
            ].to_numpy(float),
        )

        area_df = apply_efficiency(
            area_df,
            best_loss,
        )

        ghi_arrays = [
            df_fix[
                "GHI_Forecast"
            ].to_numpy(float)
        ]

        weights = [
            area_df[
                "Eff Area"
            ].sum()
        ]

        tracking_sheet = (
            read_tracking_sheet(
                file_bytes
            )
        )

        block_df = read_backend_cal(
            file_bytes,
            "Backend Cal",
        )

        blocks = block_df[
            "Block No."
        ].to_numpy(float)

    # ------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------

    if (
        "lc_params" not in st.session_state
        or st.session_state.get(
            "lc_file"
        ) != uploaded_file.name
    ):

        best = optimize_tracking(
            actual=actual,
            blocks=blocks,
            ghi_arrays=ghi_arrays,
            weights=weights,
            cluster=is_cluster,
        )

        st.session_state.lc_params = {
            "DHI": int(best[0]),
            "start": int(best[1]),
            "end": int(best[2]),
            "max": int(best[3]),
            "east": int(best[4]),
            "west": int(best[5]),
            "loss": float(best_loss),
        }

        st.session_state.lc_file = (
            uploaded_file.name
        )

    params = st.session_state.lc_params

    # ------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------

    st.subheader(
        "Optimized Parameters"
    )

    best_loss = st.number_input(
        "Efficiency Loss (%)",
        step=0.1,
        value=float(
            params["loss"]
        ),
        key="lc_loss",
    )

    c1, c2, c3 = st.columns(3)

    DHI = c1.number_input(
        "DHI (%)",
        step=1,
        value=params["DHI"],
        key="lc_dhi",
    )

    start = c2.number_input(
        "Starting Block",
        step=1,
        value=params["start"],
        key="lc_start",
    )

    end = c3.number_input(
        "Ending Block",
        step=1,
        value=params["end"],
        key="lc_end",
    )

    c1, c2, c3 = st.columns(3)

    maximum = c1.number_input(
        "Max Block",
        step=1,
        value=params["max"],
        key="lc_max",
    )

    east = c2.number_input(
        "East Limit",
        step=1,
        value=params["east"],
        key="lc_east",
    )

    west = c3.number_input(
        "West Limit",
        step=1,
        value=params["west"],
        key="lc_west",
    )

    # ------------------------------------------------------
    # UPDATED EFFICIENCY
    # ------------------------------------------------------

    area_df = apply_efficiency(
        area_df,
        best_loss,
    )

    show_efficiency(
        area_df
    )

    # ------------------------------------------------------
    # FINAL TRACKING FORECAST
    # ------------------------------------------------------

    forecast = tracking_forecast(
        ghi_arrays=ghi_arrays,
        blocks=blocks,
        weights=weights,
        DHI=DHI,
        start=start,
        end=end,
        maximum=maximum,
        east=east,
        west=west,
    )

    tracking_sheet[
        "Fixed Power=I*Ƞ*A"
    ] = forecast

    st.plotly_chart(
        make_chart(
            tracking_sheet[
                "Fixed Power=I*Ƞ*A"
            ].values,
            actual,
        ),
        use_container_width=True,
    )
