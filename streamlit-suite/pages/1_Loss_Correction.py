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
# SIDEBAR
# ==========================================================

st.sidebar.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()


# ==========================================================
# STYLE
# ==========================================================

st.markdown(
    """
    <style>

    div[data-testid="metric-container"] {
        background:#111827;
        border:1px solid #1f2937;
        border-radius:10px;
        padding:12px 20px;
    }

    div[data-testid="stDataEditor"] {
        border-radius:10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# QUOTES
# ==========================================================

QUOTES = [
    "☕ Vo kehte the kya ho tum, aaj hum kehte hai tum kya ho be?",
    "🌦 Aapka mann nahi kar raha bahar jaane ka?..",
    "😊 Jinke ghar sheeshe ke bane hote hai vo basement mai kapde change krte h...",
    "😋 Aromatic Rose Latte with Frothy Milk pine ka mann hor hai na...",
    "🥛 Garmi mai daalo dudh mai Ice🧊 Dudh bangya Very Nice...",
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
# EXCEL READERS
# ==========================================================

@st.cache_data(show_spinner=False)
def detect_workbook(file_bytes):

    xls = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    is_cluster = (
        "Fixed-CL1" in xls.sheet_names
    )

    sheet = (
        "Fixed-CL1"
        if is_cluster
        else "Fixed"
    )

    if is_cluster:

        ghi_cols = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

    else:

        ghi_cols = [
            "GHI_Forecast"
        ]

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet,
        header=1,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required = (
        ghi_cols
        + ["Actual", "Date"]
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    null_idx = df[
        df["Date"].isna()
    ].index

    if len(null_idx):

        df = df.iloc[
            :df.index.get_loc(
                null_idx[0]
            )
        ]

    df = df.iloc[:96]

    return (
        is_cluster,
        ghi_cols,
        df[
            ghi_cols + ["Actual"]
        ].copy(),
    )


# ==========================================================
# AREA & EFFICIENCY
# ==========================================================

@st.cache_data(show_spinner=False)
def read_area_efficiency(
    file_bytes
):

    for header in [1, 2, 0]:

        df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name="Area & Efficiency",
            header=header,
            usecols=range(8),
        )

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        if (
            "Module Type"
            not in df.columns
        ):
            continue

        if (
            "Standard PV Efficiency (%)"
            not in df.columns
        ):
            continue

        if (
            "Total area(m2)"
            not in df.columns
        ):
            continue

        df = df[
            df["Module Type"].notna()
        ].copy()

        for col in df.columns:

            if col != "Module Type":

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

        df = df.dropna(
            subset=[
                "Standard PV Efficiency (%)",
                "Total area(m2)",
            ]
        )

        df = df[
            df[
                "Standard PV Efficiency (%)"
            ].between(1, 50)
        ]

        if len(df):

            return df.reset_index(
                drop=True
            )

    raise ValueError(
        "Could not correctly read Area & Efficiency sheet."
    )


# ==========================================================
# WEIGHTS
# ==========================================================

@st.cache_data(show_spinner=False)
def read_weights(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[12, 13, 14, 15, 16],
    )

    return df


# ==========================================================
# FORECAST CONFIG
# ==========================================================

@st.cache_data(show_spinner=False)
def read_forecast_config(
    file_bytes
):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )

    if "Lat" not in df.columns:
        raise ValueError(
            "Lat column not found in Forecast Config."
        )

    return float(
        df.loc[0, "Lat"]
    )


# ==========================================================
# TILT
# ==========================================================

@st.cache_data(show_spinner=False)
def read_tilt(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" not in df.columns:
        raise ValueError(
            "Fixed column not found in Config Tilt Angle."
        )

    df["Fixed"] = df[
        "Fixed"
    ].fillna(0)

    df = df[
        df["Fixed"] != 0
    ]

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
        df
        .set_index("Month")["Fixed"]
        .to_dict()
    )


# ==========================================================
# BACKEND CAL
# ==========================================================

@st.cache_data(show_spinner=False)
def read_backend_cal(
    file_bytes,
    sheet,
):

    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet,
    )


# ==========================================================
# TRACKING SHEET
# ==========================================================

@st.cache_data(show_spinner=False)
def read_tracking_sheet(
    file_bytes
):

    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Tracking",
        header=1,
    )


# ==========================================================
# CALCULATION SHEET
# ==========================================================

@st.cache_data(show_spinner=False)
def read_calculation_sheet(
    file_bytes,
    sheet,
):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet,
        header=1,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "Date" in df.columns:

        null_idx = df[
            df["Date"].isna()
        ].index

        if len(null_idx):

            df = df.iloc[
                :df.index.get_loc(
                    null_idx[0]
                )
            ]

    return df.iloc[:96].copy()


# ==========================================================
# SOLAR GEOMETRY
# ==========================================================

@st.cache_data(show_spinner=False)
def solar_geometry(
    lat,
    tilt_dict=None,
    tracking=False,
):

    today = (
        pd.Timestamp
        .today()
        .normalize()
    )

    first = today.replace(
        month=1,
        day=1,
    )

    days = (
        today - first
    ).days + 1

    declination = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + days)
                / 365
            )
        )
    )

    elevation = (
        90
        - lat
        + declination
    )

    if tracking:

        tilt = 0

    else:

        month = today.strftime(
            "%B"
        )

        tilt = (
            tilt_dict.get(
                month,
                0,
            )
            if tilt_dict
            else 0
        )

    sin_a = np.sin(
        np.radians(
            elevation
        )
    )

    sin_ab = np.sin(
        np.radians(
            elevation + tilt
        )
    )

    return sin_ab, sin_a


# ==========================================================
# APPLY EFFICIENCY LOSS
# ==========================================================

def apply_efficiency(
    area_df,
    loss,
):

    df = area_df.copy()

    df[
        "Efficiency Losses(%)"
    ] = loss

    df[
        "Net Efficiency (%)"
    ] = (
        df[
            "Standard PV Efficiency (%)"
        ]
        - loss
    )

    df["Eff Area"] = (
        df["Total area(m2)"]
        * df["Net Efficiency (%)"]
        / 100
    )

    return df


# ==========================================================
# AUTOMATIC LOSS OPTIMIZATION
# ==========================================================

@st.cache_data(show_spinner=False)
def optimize_loss(
    std_eff,
    area,
    actual,
    poa,
    cluster=False,
    weights=None,
):

    std_eff = np.asarray(
        std_eff,
        dtype=float,
    )

    area = np.asarray(
        area,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    losses = np.arange(
        0,
        std_eff.min() + 0.01,
        0.1,
    )

    peak = actual.max()

    if peak == 0:
        return 0.0

    best_loss = 0.0
    best_error = np.inf

    for loss in losses:

        eff_area = (
            area
            * (std_eff - loss)
            / 100
        )

        if cluster:

            pred = np.zeros(
                len(actual)
            )

            for i in range(5):

                pred += (
                    np.asarray(
                        poa[i]
                    )
                    * eff_area[i]
                    * weights[i]
                    / 1e6
                )

        else:

            pred = (
                np.asarray(
                    poa[0]
                )
                * eff_area.sum()
                / 1e6
            )

        error = abs(
            peak
            - pred.max()
        )

        if error < best_error:

            best_error = error
            best_loss = loss

    return float(
        best_loss
    )


# ==========================================================
# TRACKING FORECAST
# ==========================================================

@st.cache_data(show_spinner=False)
def tracking_forecast(
    ghi_tuple,
    blocks_tuple,
    weights_tuple,
    dhi,
    start,
    end,
    maximum,
    east,
    west,
):

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    if (
        start >= maximum
        or maximum >= end
    ):
        return np.zeros(
            len(blocks)
        )

    m1 = 90 / (
        start
        - 1
        - maximum
    )

    m2 = 90 / (
        end
        + 1
        - maximum
    )

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
                blocks > maximum
            )
            & (
                zenith > west
            ),
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

    for ghi_t, weight in zip(
        ghi_tuple,
        weights_tuple,
    ):

        ghi = np.asarray(
            ghi_t,
            dtype=float,
        )

        dhi_value = (
            ghi
            * dhi
            / 100
        )

        forecast += (
            (
                (
                    ghi
                    - dhi_value
                )
                / cos_a
            )
            * weight
            / 1e6
        )

    return forecast


# ==========================================================
# DIFFERENTIAL EVOLUTION
# ==========================================================

def run_de(
    actual,
    blocks_tuple,
    ghi_tuple,
    weights_tuple,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    mask = actual != 0

    actual_m = actual[mask]

    bounds = [
        (0, 10),
        (0, 30),
        (65, 80),
        (44, 60),
        (0, 70),
        (0, 70),
    ]

    def objective(x):

        try:

            (
                DHI,
                start,
                end,
                maximum,
                east,
                west,
            ) = (
                int(round(v))
                for v in x
            )

            if (
                start >= maximum
                or maximum >= end
            ):
                return 1e9

            m1 = 90 / (
                start
                - 1
                - maximum
            )

            m2 = 90 / (
                end
                + 1
                - maximum
            )

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
                        blocks > maximum
                    )
                    & (
                        zenith > west
                    ),
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

            pred = np.zeros(
                len(blocks),
                dtype=float,
            )

            for ghi_t, weight in zip(
                ghi_tuple,
                weights_tuple,
            ):

                ghi = np.asarray(
                    ghi_t,
                    dtype=float,
                )

                dhi = (
                    ghi
                    * DHI
                    / 100
                )

                pred += (
                    (
                        (
                            ghi
                            - dhi
                        )
                        / cos_a
                    )
                    * weight
                    / 1e6
                )

            pred = pred[mask]

            if (
                len(pred) == 0
                or np.isnan(pred).any()
                or np.isinf(pred).any()
            ):
                return 1e9

            peak = actual_m.max()

            if peak == 0:
                return 1e9

            energy = actual_m.sum()

            if energy == 0:
                return 1e9

            return (
                0.80
                * np.mean(
                    np.abs(
                        actual_m
                        - pred
                    )
                )
                / peak
                +
                0.10
                * abs(
                    peak
                    - pred.max()
                )
                / peak
                +
                0.10
                * abs(
                    energy
                    - pred.sum()
                )
                / energy
            )

        except Exception:

            return 1e9

    progress = st.progress(
        0
    )

    status = st.empty()

    state = {
        "gen": 0,
        "quote": None,
    }

    MAX_ITER = 100

    def callback(
        xk,
        convergence,
    ):

        state["gen"] += 1

        progress.progress(
            min(
                state["gen"]
                / MAX_ITER,
                1.0,
            )
        )

        if (
            state["gen"] % 20 == 1
            or state["quote"] is None
        ):

            choices = [
                q
                for q in QUOTES
                if q
                != state["quote"]
            ]

            state["quote"] = (
                random.choice(
                    choices
                )
            )

        status.info(
            f"{state['quote']}\n\n"
            f"Generation "
            f"{state['gen']} / "
            f"{MAX_ITER}"
        )

    status.info(
        random.choice(
            QUOTES
        )
    )

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=MAX_ITER,
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
        "✅ Tracking optimization complete!"
    )

    return np.round(
        result.x
    ).astype(int)


# ==========================================================
# CHART
# ==========================================================

def make_chart(
    forecast,
    actual,
    title="Forecast vs Actual Power",
):

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
        title=title,
        template="streamlit",
        height=500,
        hovermode="x unified",
        xaxis={
            "title": "15 Minute Block",
            "dtick": 4,
        },
        yaxis={
            "title": "Power (MW)",
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
# PAGE
# ==========================================================

st.title(
    "Pakima Pakam Ravi, 3-4 Loss Correction kar chuke hai!! 😎"
)


# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded = st.file_uploader(
    "Yaha Feko!!",
    type=["xlsx"],
    key="lc_uploader",
)


if uploaded is None:

    st.info(
        "Pehle File toh upload karo!!!"
    )

    st.stop()


file_bytes = uploaded.getvalue()


# ==========================================================
# RESET WHEN NEW FILE UPLOADED
# ==========================================================

file_signature = (
    uploaded.name,
    len(file_bytes),
)

if (
    st.session_state.get(
        "lc_file_signature"
    )
    != file_signature
):

    keys_to_clear = [
        "lc_run",
        "lc_params",
        "lc_last_plant_type",
        "lc_auto_loss",
    ]

    for key in keys_to_clear:

        st.session_state.pop(
            key,
            None,
        )

    st.session_state[
        "lc_file_signature"
    ] = file_signature


# ==========================================================
# WORKBOOK DETECTION
# ==========================================================

try:

    (
        is_cluster,
        ghi_cols,
        raw_df,
    ) = detect_workbook(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Workbook read error: {e}"
    )

    st.stop()


# ==========================================================
# INPUT DATA
# ONLY GHI + ACTUAL ARE SHOWN
# ==========================================================

st.subheader(
    "📊 Input Data"
)

original_df = raw_df.copy()

edited_df = st.data_editor(
    raw_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="lc_editor",
    column_config={
        col: st.column_config.NumberColumn(
            col,
            format="%.2f",
        )
        for col in ghi_cols
    }
    | {
        "Actual": st.column_config.NumberColumn(
            "Actual",
            format="%.2f",
        )
    },
)


# ==========================================================
# CLEAN EDITED DATA
# ==========================================================

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


# ==========================================================
# PLANT TYPE
# BELOW INPUT TABLE
# ==========================================================

st.subheader(
    "🏭 Plant Type"
)

plant_type = st.pills(
    "Select Plant Type",
    [
        "🏗️ Fixed",
        "🔄 Tracking",
    ],
    default="🏗️ Fixed",
)


# ==========================================================
# RESET CALCULATION WHEN PLANT TYPE CHANGES
# ==========================================================

if (
    st.session_state.get(
        "lc_last_plant_type"
    )
    != plant_type
):

    st.session_state[
        "lc_run"
    ] = False

    st.session_state.pop(
        "lc_params",
        None,
    )

    st.session_state[
        "lc_last_plant_type"
    ] = plant_type


# ==========================================================
# RUN CALCULATION
# ==========================================================

if st.button(
    "🚀 Dabao magar pyaar se!!",
    use_container_width=True,
    type="primary",
):

    st.session_state[
        "lc_run"
    ] = True

    st.session_state.pop(
        "lc_params",
        None,
    )

    st.session_state.pop(
        "lc_auto_loss",
        None,
    )

    st.rerun()


if not st.session_state.get(
    "lc_run",
    False,
):

    st.info(
        "Input data check karo, Plant Type select karo aur calculation run karo."
    )

    st.stop()


# ==========================================================
# COMMON DATA
# ==========================================================

try:

    area_df = read_area_efficiency(
        file_bytes
    )

    lat = read_forecast_config(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Configuration read error: {e}"
    )

    st.stop()


sheet = (
    "Fixed-CL1"
    if is_cluster
    else "Fixed"
)


try:

    df_fix = read_calculation_sheet(
        file_bytes,
        sheet,
    )

except Exception as e:

    st.error(
        f"Calculation sheet error: {e}"
    )

    st.stop()


n = min(
    len(df_fix),
    len(edited_df),
)


df_fix = (
    df_fix
    .iloc[:n]
    .reset_index(drop=True)
)


for col in ghi_cols:

    df_fix[col] = (
        edited_df[col]
        .values[:n]
    )


df_fix["Actual"] = (
    edited_df["Actual"]
    .values[:n]
)


actual = df_fix[
    "Actual"
].to_numpy(
    dtype=float
)


# ==========================================================
# FIXED PLANT
# ==========================================================

if plant_type == "🏗️ Fixed":

    st.subheader(
        "🏗️ Fixed Plant Loss Correction"
    )

    # ------------------------------------------------------
    # Solar geometry
    # ------------------------------------------------------

    tilt_dict = read_tilt(
        file_bytes
    )

    sin_ab, sin_a = solar_geometry(
        lat,
        tilt_dict,
        tracking=False,
    )

    # ------------------------------------------------------
    # GHI -> POA
    # ------------------------------------------------------

    suffixes = (
        [
            "",
            "-CL2",
            "-CL3",
            "-CL4",
            "-CL5",
        ]
        if is_cluster
        else [""]
    )

    poa_list = []

    for col, suffix in zip(
        ghi_cols,
        suffixes,
    ):

        ghi = df_fix[
            col
        ].to_numpy(
            dtype=float
        )

        poa = (
            ghi
            * sin_ab
            / sin_a
        )

        poa_list.append(
            poa
        )

    # ------------------------------------------------------
    # Cluster setup
    # ------------------------------------------------------

    if is_cluster:

        weight_df = read_weights(
            file_bytes
        )

        weights_raw = tuple(
            float(
                weight_df[
                    f"CL-{i}"
                ].iloc[0]
            )
            for i in range(1, 6)
        )

        poa_tuple = tuple(
            tuple(x)
            for x in poa_list
        )

        # --------------------------------------------------
        # AUTOMATIC LOSS
        # --------------------------------------------------

        auto_loss = optimize_loss(
            tuple(
                area_df[
                    "Standard PV Efficiency (%)"
                ]
            ),
            tuple(
                area_df[
                    "Total area(m2)"
                ]
            ),
            tuple(actual),
            poa_tuple,
            True,
            weights_raw,
        )

        # Store automatic value only once
        if (
            "lc_auto_loss"
            not in st.session_state
        ):

            st.session_state[
                "lc_auto_loss"
            ] = float(
                auto_loss
            )

    else:

        poa_tuple = (
            tuple(
                poa_list[0]
            ),
        )

        # --------------------------------------------------
        # AUTOMATIC LOSS
        # --------------------------------------------------

        auto_loss = optimize_loss(
            tuple(
                area_df[
                    "Standard PV Efficiency (%)"
                ]
            ),
            tuple(
                area_df[
                    "Total area(m2)"
                ]
            ),
            tuple(actual),
            poa_tuple,
            False,
            None,
        )

        if (
            "lc_auto_loss"
            not in st.session_state
        ):

            st.session_state[
                "lc_auto_loss"
            ] = float(
                auto_loss
            )

        weights_raw = None

    # ------------------------------------------------------
    # MANUAL EFFICIENCY LOSS
    # ------------------------------------------------------

    st.markdown(
        "### ⚙️ Efficiency Loss Adjustment"
    )

    st.caption(
        "Program ne automatic loss calculate kiya hai. "
        "Aap is value ko manually change karke final forecast adjust kar sakte ho."
    )

    manual_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=float(
            area_df[
                "Standard PV Efficiency (%)"
            ].min()
        ),
        value=float(
            st.session_state[
                "lc_auto_loss"
            ]
        ),
        step=0.1,
        format="%.2f",
        key="fixed_manual_loss",
    )

    # ------------------------------------------------------
    # FINAL EFFICIENCY
    # ------------------------------------------------------

    area_eff = apply_efficiency(
        area_df,
        manual_loss,
    )

    # ------------------------------------------------------
    # FINAL FORECAST
    # ------------------------------------------------------

    if is_cluster:

        forecast = np.zeros(
            n,
            dtype=float,
        )

        for i in range(5):

            forecast += (
                poa_list[i]
                * area_eff[
                    "Eff Area"
                ].iloc[i]
                * weights_raw[i]
                / 1e6
            )

    else:

        forecast = (
            poa_list[0]
            * area_eff[
                "Eff Area"
            ].sum()
            / 1e6
        )

    # ------------------------------------------------------
    # RESULT
    # ------------------------------------------------------

    st.plotly_chart(
        make_chart(
            forecast,
            actual,
            "Fixed Plant — Forecast vs Actual",
        ),
        use_container_width=True,
    )

    # ------------------------------------------------------
    # Comparison metrics
    # ------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    peak_actual = (
        float(actual.max())
        if len(actual)
        else 0
    )

    peak_forecast = (
        float(forecast.max())
        if len(forecast)
        else 0
    )

    energy_actual = (
        float(actual.sum())
    )

    energy_forecast = (
        float(forecast.sum())
    )

    peak_error = (
        abs(
            peak_actual
            - peak_forecast
        )
        / peak_actual
        * 100
        if peak_actual != 0
        else 0
    )

    energy_error = (
        abs(
            energy_actual
            - energy_forecast
        )
        / energy_actual
        * 100
        if energy_actual != 0
        else 0
    )

    c1.metric(
        "Automatic Loss",
        f"{st.session_state['lc_auto_loss']:.2f}%",
    )

    c2.metric(
        "Applied Loss",
        f"{manual_loss:.2f}%",
    )

    c3.metric(
        "Peak Error",
        f"{peak_error:.2f}%",
    )

    st.caption(
        f"Actual Energy: {energy_actual:.2f} | "
        f"Forecast Energy: {energy_forecast:.2f} | "
        f"Energy Error: {energy_error:.2f}%"
    )


# ==========================================================
# TRACKING PLANT
# ==========================================================

else:

    st.subheader(
        "🔄 Tracking Plant Loss Correction"
    )

    # ------------------------------------------------------
    # Geometry
    # ------------------------------------------------------

    sin_ab, sin_a = solar_geometry(
        lat,
        tracking=True,
    )

    ghi_arrays = [
        df_fix[
            col
        ].to_numpy(
            dtype=float
        )
        for col in ghi_cols
    ]

    # ------------------------------------------------------
    # Cluster / non-cluster
    # ------------------------------------------------------

    if is_cluster:

        weight_df = read_weights(
            file_bytes
        )

        weights_raw = tuple(
            float(
                weight_df[
                    f"CL-{i}"
                ].iloc[0]
            )
            for i in range(1, 6)
        )

        poa_list = [
            ghi
            * sin_ab
            / sin_a
            for ghi in ghi_arrays
        ]

        poa_tuple = tuple(
            tuple(x)
            for x in poa_list
        )

        best_loss = optimize_loss(
            tuple(
                area_df[
                    "Standard PV Efficiency (%)"
                ]
            ),
            tuple(
                area_df[
                    "Total area(m2)"
                ]
            ),
            tuple(actual),
            poa_tuple,
            True,
            weights_raw,
        )

        area_eff = apply_efficiency(
            area_df,
            best_loss,
        )

        block_df = read_backend_cal(
            file_bytes,
            "Backend Cal CL1",
        )

    else:

        poa = (
            ghi_arrays[0]
            * sin_ab
            / sin_a
        )

        poa_tuple = (
            tuple(poa),
        )

        best_loss = optimize_loss(
            tuple(
                area_df[
                    "Standard PV Efficiency (%)"
                ]
            ),
            tuple(
                area_df[
                    "Total area(m2)"
                ]
            ),
            tuple(actual),
            poa_tuple,
            False,
            None,
        )

        area_eff = apply_efficiency(
            area_df,
            best_loss,
        )

        block_df = read_backend_cal(
            file_bytes,
            "Backend Cal",
        )

        weights_raw = None

    # ------------------------------------------------------
    # Backend blocks
    # ------------------------------------------------------

    blocks = block_df[
        "Block No."
    ].to_numpy(
        dtype=float
    )

    blocks_tuple = tuple(
        blocks
    )

    ghi_tuple = tuple(
        tuple(x)
        for x in ghi_arrays
    )

    # ------------------------------------------------------
    # Tracking optimization
    # ------------------------------------------------------

    if (
        "lc_params"
        not in st.session_state
    ):

        if is_cluster:

            de_weights = tuple(
                tuple(
                    area_eff[
                        "Eff Area"
                    ].iloc[i]
                    * weights_raw[i]
                    for i in range(5)
                )
            )

        else:

            de_weights = (
                float(
                    area_eff[
                        "Eff Area"
                    ].sum()
                ),
            )

        best = run_de(
            actual,
            blocks_tuple,
            ghi_tuple,
            de_weights,
        )

        st.session_state[
            "lc_params"
        ] = {

            "DHI": int(
                best[0]
            ),

            "start": int(
                best[1]
            ),

            "end": int(
                best[2]
            ),

            "max": int(
                best[3]
            ),

            "east": int(
                best[4]
            ),

            "west": int(
                best[5]
            ),

            "loss": float(
                best_loss
            ),
        }

    params = (
        st.session_state[
            "lc_params"
        ]
    )

    # ------------------------------------------------------
    # Manual tracking parameters
    # ------------------------------------------------------

    st.subheader(
        "⚙️ Tracking Parameters"
    )

    st.caption(
        "Optimized values automatically aaye hain. "
        "Aap values change karke Recalculate kar sakte ho."
    )

    with st.form(
        "tracking_parameters_form"
    ):

        c1, c2, c3 = st.columns(3)

        DHI = c1.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            step=1,
            value=params["DHI"],
        )

        start = c2.number_input(
            "Starting Block",
            min_value=0,
            max_value=95,
            step=1,
            value=params["start"],
        )

        end = c3.number_input(
            "Ending Block",
            min_value=1,
            max_value=96,
            step=1,
            value=params["end"],
        )

        c1, c2, c3 = st.columns(3)

        maximum = c1.number_input(
            "Max Block",
            min_value=1,
            max_value=96,
            step=1,
            value=params["max"],
        )

        east = c2.number_input(
            "East Limit",
            min_value=0,
            max_value=90,
            step=1,
            value=params["east"],
        )

        west = c3.number_input(
            "West Limit",
            min_value=0,
            max_value=90,
            step=1,
            value=params["west"],
        )

        recalc = st.form_submit_button(
            "🔄 Recalculate",
            use_container_width=True,
            type="primary",
        )

    # ------------------------------------------------------
    # Update parameters
    # ------------------------------------------------------

    if recalc:

        if (
            start >= maximum
            or maximum >= end
        ):

            st.error(
                "Starting Block < Max Block < Ending Block hona chahiye."
            )

        else:

            st.session_state[
                "lc_params"
            ].update(
                {
                    "DHI": DHI,
                    "start": start,
                    "end": end,
                    "max": maximum,
                    "east": east,
                    "west": west,
                }
            )

            params = (
                st.session_state[
                    "lc_params"
                ]
            )

            st.success(
                "✅ Tracking parameters updated."
            )

    # ------------------------------------------------------
    # Effective weights
    # ------------------------------------------------------

    area_eff = apply_efficiency(
        area_df,
        params["loss"],
    )

    if is_cluster:

        weights_eff = tuple(
            float(
                area_eff[
                    "Eff Area"
                ].iloc[i]
            )
            * weights_raw[i]
            for i in range(5)
        )

    else:

        weights_eff = (
            float(
                area_eff[
                    "Eff Area"
                ].sum()
            ),
        )

    # ------------------------------------------------------
    # FINAL TRACKING FORECAST
    # ------------------------------------------------------

    forecast = tracking_forecast(
        ghi_tuple,
        blocks_tuple,
        weights_eff,
        params["DHI"],
        params["start"],
        params["end"],
        params["max"],
        params["east"],
        params["west"],
    )

    # ------------------------------------------------------
    # Chart
    # ------------------------------------------------------

    st.plotly_chart(
        make_chart(
            forecast,
            actual,
            "Tracking Plant — Forecast vs Actual",
        ),
        use_container_width=True,
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    peak_actual = (
        float(actual.max())
        if len(actual)
        else 0
    )

    peak_forecast = (
        float(forecast.max())
        if len(forecast)
        else 0
    )

    energy_actual = (
        float(actual.sum())
    )

    energy_forecast = (
        float(forecast.sum())
    )

    peak_error = (
        abs(
            peak_actual
            - peak_forecast
        )
        / peak_actual
        * 100
        if peak_actual != 0
        else 0
    )

    energy_error = (
        abs(
            energy_actual
            - energy_forecast
        )
        / energy_actual
        * 100
        if energy_actual != 0
        else 0
    )

    c1.metric(
        "Efficiency Loss",
        f"{params['loss']:.2f}%",
    )

    c2.metric(
        "Peak Error",
        f"{peak_error:.2f}%",
    )

    c3.metric(
        "Energy Error",
        f"{energy_error:.2f}%",
    )

    st.caption(
        f"Actual Energy: {energy_actual:.2f} | "
        f"Forecast Energy: {energy_forecast:.2f}"
    )
