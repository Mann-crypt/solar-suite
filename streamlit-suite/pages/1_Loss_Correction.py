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

    div[data-testid="metric-container"]{
        background:#111827;
        border:1px solid #1f2937;
        border-radius:10px;
        padding:12px 20px;
    }

    div[data-testid="stDataEditor"]{
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
# FILE READERS
# ==========================================================

@st.cache_data(show_spinner=False)
def detect_workbook(file_bytes):

    xls = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    cluster = "Fixed-CL1" in xls.sheet_names

    sheet = (
        "Fixed-CL1"
        if cluster
        else "Fixed"
    )

    if cluster:

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

    # ------------------------------------------------------
    # Check columns
    # ------------------------------------------------------

    required = (
        ghi_cols
        + ["Actual"]
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Required columns missing: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------
    # Clean data
    # ------------------------------------------------------

    for col in ghi_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    # ------------------------------------------------------
    # Remove rows after first missing date
    # ------------------------------------------------------

    if "Date" in df.columns:

        null_idx = df[
            df["Date"].isna()
        ].index

        if len(null_idx):

            first_null = null_idx[0]

            df = df.iloc[
                :df.index.get_loc(
                    first_null
                )
            ]

    # ------------------------------------------------------
    # Exactly first 96 blocks
    # ------------------------------------------------------

    df = df.iloc[
        :96
    ].reset_index(
        drop=True
    )

    return (
        cluster,
        ghi_cols,
        df[
            ghi_cols + ["Actual"]
        ].copy(),
    )


@st.cache_data(show_spinner=False)
def read_area_efficiency(file_bytes):

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

        df = df[
            df["Module Type"].notna()
        ].copy()

        for col in df.columns:

            if col != "Module Type":

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

        if (
            "Standard PV Efficiency (%)"
            not in df.columns
        ):
            continue

        df = df.dropna(
            subset=[
                "Standard PV Efficiency (%)"
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
        "Could not correctly read "
        "Area & Efficiency sheet."
    )


@st.cache_data(show_spinner=False)
def read_weights(file_bytes):

    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[
            12,
            13,
            14,
            15,
            16,
        ],
    )


@st.cache_data(show_spinner=False)
def read_forecast_config(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )

    return float(
        df.loc[0, "Lat"]
    )


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

    df["Fixed"] = (
        df["Fixed"]
        .fillna(0)
    )

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

        month = (
            today
            .strftime("%B")
        )

        tilt = (
            tilt_dict.get(
                month,
                0,
            )
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

    return (
        sin_ab,
        sin_a,
    )


# ==========================================================
# EFFICIENCY
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

    df[
        "Eff Area"
    ] = (
        df[
            "Total area(m2)"
        ]
        * df[
            "Net Efficiency (%)"
        ]
        / 100
    )

    return df


def show_efficiency(df):

    cols = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)",
    ]

    disp = df[
        cols
    ].copy()

    num_cols = (
        disp
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    disp[
        num_cols
    ] = disp[
        num_cols
    ].round(2)

    with st.expander(
        "🔍 View Efficiency Calculations"
    ):

        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# LOSS OPTIMIZATION
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

    if peak <= 0:

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

    m1 = (
        90
        / (
            start
            - 1
            - maximum
        )
    )

    m2 = (
        90
        / (
            end
            + 1
            - maximum
        )
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
# TRACKING DE OPTIMIZATION
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

    actual_m = actual[
        mask
    ]

    if len(actual_m) == 0:

        return np.array(
            [0, 10, 70, 50, 30, 30]
        )

    bounds = [
        (0, 10),    # DHI
        (0, 30),    # Start
        (65, 80),   # End
        (44, 60),   # Maximum
        (0, 70),    # East
        (0, 70),    # West
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

            m1 = (
                90
                / (
                    start
                    - 1
                    - maximum
                )
            )

            m2 = (
                90
                / (
                    end
                    + 1
                    - maximum
                )
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

                dhi_value = (
                    ghi
                    * DHI
                    / 100
                )

                pred += (
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

            pred = pred[
                mask
            ]

            if (
                len(pred) == 0
                or np.isnan(pred).any()
                or np.isinf(pred).any()
            ):

                return 1e9

            peak = actual_m.max()

            if peak <= 0:

                return 1e9

            energy = actual_m.sum()

            if energy <= 0:

                return 1e9

            block_error = (
                np.mean(
                    np.abs(
                        actual_m
                        - pred
                    )
                )
                / peak
            )

            peak_error = (
                abs(
                    peak
                    - pred.max()
                )
                / peak
            )

            energy_error = (
                abs(
                    energy
                    - pred.sum()
                )
                / energy
            )

            return (
                0.80
                * block_error
                + 0.10
                * peak_error
                + 0.10
                * energy_error
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

    MAX_ITER = 60

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
            state["gen"] % 10 == 1
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

    with st.spinner(
        "Tracking calculation ho rahi hai... ⚙️"
    ):

        result = (
            differential_evolution(
                objective,
                bounds=bounds,
                strategy="best1bin",
                maxiter=MAX_ITER,
                popsize=10,
                tol=0.002,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                polish=True,
                workers=1,
                callback=callback,
            )
        )

    progress.empty()

    status.success(
        "✅ Tracking optimization complete"
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
        title="Forecast vs Actual Power",
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
# RESET WHEN NEW FILE
# ==========================================================

if (
    st.session_state.get(
        "lc_last_file"
    )
    != uploaded.name
):

    for key in [
        "lc_params",
        "lc_run",
        "lc_last_file",
    ]:

        st.session_state.pop(
            key,
            None,
        )

    st.session_state[
        "lc_last_file"
    ] = uploaded.name

    st.rerun()


# ==========================================================
# LOAD WORKBOOK
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
        f"Workbook read nahi ho paya: {e}"
    )

    st.stop()


# ==========================================================
# INPUT DATA
# ==========================================================

st.subheader(
    "📊 Input Data"
)

st.caption(
    "Sirf GHI aur Actual values edit karein."
)


# Only GHI + Actual are exposed to user
input_df = raw_df[
    ghi_cols + ["Actual"]
].copy()


edited_df = st.data_editor(
    input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="lc_editor",
)


# ==========================================================
# CLEAN USER INPUT
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
# MAIN CALCULATION BUTTON
# ==========================================================

if st.button(
    "🚀 Dabao magar pyaar se!!",
    use_container_width=True,
    type="primary",
):

    st.session_state.lc_run = True

    # Clear previous tracking parameters
    st.session_state.pop(
        "lc_params",
        None,
    )

    # Store plant type
    st.session_state[
        "lc_plant_type"
    ] = plant_type


if not st.session_state.get(
    "lc_run",
    False,
):

    st.stop()


# ==========================================================
# USE CURRENT PLANT TYPE
# ==========================================================

plant_type = st.session_state.get(
    "lc_plant_type",
    plant_type,
)


# ==========================================================
# COMMON DATA
# ==========================================================

area_df = read_area_efficiency(
    file_bytes
)

lat = read_forecast_config(
    file_bytes
)

actual = (
    edited_df[
        "Actual"
    ]
    .to_numpy(float)
)


# ==========================================================
# FIXED PLANT
# ==========================================================

if plant_type == "🏗️ Fixed":

    st.subheader(
        "🏗️ Fixed Plant Loss Correction"
    )

    # ------------------------------------------------------
    # Tilt
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

        ghi = (
            edited_df[
                col
            ]
            .to_numpy(float)
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
    # Weights
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

    else:

        weights_raw = None


    # ======================================================
    # MANUAL EFFICIENCY LOSS
    # ======================================================

    st.markdown(
        "### ⚙️ Efficiency Loss"
    )

    max_loss = max(
        0.1,
        float(
            area_df[
                "Standard PV Efficiency (%)"
            ].min()
        )
        - 0.1,
    )

    efficiency_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=float(
            max_loss
        ),
        value=0.0,
        step=0.1,
        format="%.1f",
        help=(
            "Manually enter the efficiency "
            "loss percentage."
        ),
    )


    # ------------------------------------------------------
    # Apply manual loss
    # ------------------------------------------------------

    area_eff = apply_efficiency(
        area_df,
        efficiency_loss,
    )


    # ------------------------------------------------------
    # Show efficiency
    # ------------------------------------------------------

    show_efficiency(
        area_eff
    )


    # ------------------------------------------------------
    # Forecast
    # ------------------------------------------------------

    if is_cluster:

        forecast = np.zeros(
            len(actual)
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
    # Metric
    # ------------------------------------------------------

    st.metric(
        "Efficiency Loss",
        f"{efficiency_loss:.1f}%",
    )


    # ------------------------------------------------------
    # Chart
    # ------------------------------------------------------

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

    st.subheader(
        "🔄 Tracking Plant Loss Correction"
    )

    # ------------------------------------------------------
    # Tracking geometry
    # ------------------------------------------------------

    sin_ab, sin_a = solar_geometry(
        lat,
        tracking=True,
    )


    # ------------------------------------------------------
    # GHI arrays
    # ------------------------------------------------------

    ghi_arrays = [
        edited_df[
            col
        ].to_numpy(float)
        for col in ghi_cols
    ]


    # ------------------------------------------------------
    # Area / weights
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

    else:

        weights_raw = None


    # ------------------------------------------------------
    # POA
    # ------------------------------------------------------

    poa_list = [
        ghi
        * sin_ab
        / sin_a
        for ghi in ghi_arrays
    ]


    # ------------------------------------------------------
    # Backend
    # ------------------------------------------------------

    if is_cluster:

        block_df = read_backend_cal(
            file_bytes,
            "Backend Cal CL1",
        )

    else:

        block_df = read_backend_cal(
            file_bytes,
            "Backend Cal",
        )


    blocks = block_df[
        "Block No."
    ].to_numpy(float)


    # ------------------------------------------------------
    # Optimize efficiency loss first
    # ------------------------------------------------------

    poa_tuple = tuple(
        tuple(x)
        for x in poa_list
    )


    if is_cluster:

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

    else:

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


    # ------------------------------------------------------
    # Apply optimized loss
    # ------------------------------------------------------

    area_eff = apply_efficiency(
        area_df,
        best_loss,
    )


    # ------------------------------------------------------
    # Tracking weights
    # ------------------------------------------------------

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
    # Initial DE optimization
    # ------------------------------------------------------

    if (
        "lc_params"
        not in st.session_state
    ):

        best = run_de(
            actual,
            tuple(blocks),
            tuple(
                tuple(x)
                for x in ghi_arrays
            ),
            weights_eff,
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


    params = (
        st.session_state.lc_params
    )


    # ======================================================
    # TRACKING PARAMETERS
    # ======================================================

    st.markdown(
        "### ⚙️ Tracking Parameters"
    )

    with st.form(
        "tracking_params_form"
    ):

        c1, c2, c3 = st.columns(3)

        loss_input = c1.number_input(
            "Efficiency Loss (%)",
            min_value=0.0,
            max_value=float(
                max(
                    0.1,
                    area_df[
                        "Standard PV Efficiency (%)"
                    ].min()
                    - 0.1,
                )
            ),
            value=float(
                params["loss"]
            ),
            step=0.1,
            format="%.1f",
        )

        DHI = c2.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(
                params["DHI"]
            ),
            step=1,
        )

        start = c3.number_input(
            "Starting Block",
            min_value=0,
            max_value=95,
            value=int(
                params["start"]
            ),
            step=1,
        )

        c1, c2, c3 = st.columns(3)

        end = c1.number_input(
            "Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                params["end"]
            ),
            step=1,
        )

        maximum = c2.number_input(
            "Max Block",
            min_value=1,
            max_value=96,
            value=int(
                params["max"]
            ),
            step=1,
        )

        east = c3.number_input(
            "East Limit",
            min_value=0,
            max_value=90,
            value=int(
                params["east"]
            ),
            step=1,
        )

        west = st.number_input(
            "West Limit",
            min_value=0,
            max_value=90,
            value=int(
                params["west"]
            ),
            step=1,
        )

        recalc = (
            st.form_submit_button(
                "🔄 Recalculate",
                use_container_width=True,
                type="primary",
            )
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
                "Invalid blocks: "
                "Starting Block < Max Block < Ending Block required."
            )

            st.stop()

        st.session_state.lc_params.update(
            {
                "loss": loss_input,
                "DHI": DHI,
                "start": start,
                "end": end,
                "max": maximum,
                "east": east,
                "west": west,
            }
        )

        params = (
            st.session_state
            .lc_params
        )


    # ------------------------------------------------------
    # Apply efficiency
    # ------------------------------------------------------

    area_eff = apply_efficiency(
        area_df,
        params["loss"],
    )


    # ------------------------------------------------------
    # Show efficiency
    # ------------------------------------------------------

    show_efficiency(
        area_eff
    )


    # ------------------------------------------------------
    # Effective weights
    # ------------------------------------------------------

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
    # Final tracking forecast
    # ------------------------------------------------------

    forecast = tracking_forecast(
        tuple(
            tuple(x)
            for x in ghi_arrays
        ),
        tuple(blocks),
        weights_eff,
        params["DHI"],
        params["start"],
        params["end"],
        params["max"],
        params["east"],
        params["west"],
    )


    # ------------------------------------------------------
    # Efficiency metric
    # ------------------------------------------------------

    st.metric(
        "Efficiency Loss",
        f'{params["loss"]:.1f}%',
    )


    # ------------------------------------------------------
    # Tracking chart
    # ------------------------------------------------------

    st.plotly_chart(
        make_chart(
            forecast,
            actual,
        ),
        use_container_width=True,
    )
