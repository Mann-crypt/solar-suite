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
# CONSTANTS
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
# CACHED FILE READERS
# ==========================================================

@st.cache_data(show_spinner=False)
def detect_workbook(file_bytes):

    xls = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    cluster = (
        "Fixed-CL1"
        in xls.sheet_names
    )

    sheet = (
        "Fixed-CL1"
        if cluster
        else "Fixed"
    )

    ghi_cols = (
        [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]
        if cluster
        else [
            "GHI_Forecast"
        ]
    )

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
        cluster,
        ghi_cols,
        df[
            ghi_cols + ["Actual"]
        ].copy(),
    )


@st.cache_data(show_spinner=False)
def read_area_efficiency(
    file_bytes
):

    for header in [
        1,
        2,
        0,
    ]:

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
            df["Module Type"]
            .notna()
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

    return pd.DataFrame()


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
def read_forecast_config(
    file_bytes
):

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
        pd.Timestamp.today()
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
            today.strftime("%B")
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
# EFFICIENCY CALCULATION
# ==========================================================

def apply_efficiency(
    area_df,
    loss,
):

    df = area_df.copy()

    df[
        "Net Efficiency (%)"
    ] = (
        df[
            "Standard PV Efficiency (%)"
        ]
        - loss
    )

    df[
        "Net Efficiency (%)"
    ] = np.clip(
        df[
            "Net Efficiency (%)"
        ],
        0,
        None,
    )

    df["Eff Area"] = (
        df["Total area(m2)"]
        * df[
            "Net Efficiency (%)"
        ]
        / 100
    )

    return df


# ==========================================================
# AUTOMATIC EFFICIENCY LOSS
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
            * (
                std_eff
                - loss
            )
            / 100
        )

        eff_area = np.clip(
            eff_area,
            0,
            None,
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
            np.radians(
                panel
            )
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

@st.cache_data(
    show_spinner=False
)
def optimize_tracking(
    actual_tuple,
    blocks_tuple,
    ghi_tuple,
    weights_tuple,
):

    actual = np.asarray(
        actual_tuple,
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

        return (
            20,
            10,
            70,
            50,
            30,
            30,
        )

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
                    np.radians(
                        panel
                    )
                ),
                1e-6,
                None,
            )

            pred = np.zeros(
                len(blocks)
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
                        ghi
                        - dhi
                    )
                    / cos_a
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

            energy = actual_m.sum()

            if (
                peak <= 0
                or energy <= 0
            ):

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

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=60,
        popsize=10,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    return tuple(
        np.round(
            result.x
        ).astype(int)
    )


# ==========================================================
# CHART
# ==========================================================

def make_chart(
    forecast,
    actual,
    title="Forecast vs Actual",
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
# TITLE
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
# FILE CHANGE RESET
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

    for key in [
        "lc_params",
        "lc_tracking_params",
        "lc_run",
        "lc_last_inputs",
    ]:

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

(
    is_cluster,
    ghi_cols,
    raw_df,
) = detect_workbook(
    file_bytes
)


# ==========================================================
# INPUT DATA
# ONLY GHI + ACTUAL
# ==========================================================

st.subheader(
    "📊 Input Data"
)

display_df = raw_df.copy()

display_df = display_df[
    ghi_cols + ["Actual"]
].copy()

display_df.columns = [
    "GHI"
    if col == "GHI_Forecast"
    else col
    for col in display_df.columns
]


# Keep original numeric values
for col in display_df.columns:

    display_df[col] = pd.to_numeric(
        display_df[col],
        errors="coerce",
    ).fillna(0)


edited_df = st.data_editor(
    display_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="lc_input_editor",
)


# ==========================================================
# PLANT TYPE
# ==========================================================

st.subheader(
    "🌞 Plant Type"
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
# RUN CALCULATION
# ==========================================================

if st.button(
    "🚀 Dabao magar pyaar se!!",
    use_container_width=True,
    type="primary",
):

    st.session_state.lc_run = True

    # Force optimization again for new input
    st.session_state.pop(
        "lc_params",
        None,
    )

    st.session_state.pop(
        "lc_tracking_params",
        None,
    )


if not st.session_state.get(
    "lc_run",
    False,
):

    st.stop()


# ==========================================================
# PREPARE INPUT ARRAYS
# ==========================================================

actual = pd.to_numeric(
    edited_df["Actual"],
    errors="coerce",
).fillna(0).to_numpy(
    dtype=float
)


# ==========================================================
# COMMON WORKBOOK DATA
# ==========================================================

area_df = read_area_efficiency(
    file_bytes
)

lat = read_forecast_config(
    file_bytes
)


# ==========================================================
# FIXED PLANT
# ==========================================================

if plant_type == "🏗️ Fixed":

    st.subheader(
        "🏗️ Fixed Plant"
    )

    tilt_dict = read_tilt(
        file_bytes
    )

    sin_ab, sin_a = solar_geometry(
        lat,
        tilt_dict,
        tracking=False,
    )

    # ------------------------------------------------------
    # GHI
    # ------------------------------------------------------

    ghi_arrays = []

    for col in ghi_cols:

        if col == "GHI_Forecast":

            ghi = edited_df[
                "GHI"
            ].to_numpy(float)

        else:

            ghi = edited_df[
                col
            ].to_numpy(float)

        ghi_arrays.append(
            ghi
        )

    # ------------------------------------------------------
    # POA
    # ------------------------------------------------------

    poa_list = []

    for ghi in ghi_arrays:

        poa_list.append(
            ghi
            * sin_ab
            / max(
                sin_a,
                1e-6,
            )
        )

    # ------------------------------------------------------
    # Cluster weights
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
    # Automatic efficiency loss
    # ------------------------------------------------------

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
        tuple(
            tuple(x)
            for x in poa_list
        ),
        is_cluster,
        weights_raw,
    )

    # ------------------------------------------------------
    # Manual efficiency loss
    # ------------------------------------------------------

    st.markdown(
        "### ⚙️ Efficiency Loss"
    )

    fixed_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=50.0,
        value=float(
            best_loss
        ),
        step=0.1,
        key="fixed_efficiency_loss",
        help=(
            "Automatically calculated value "
            "is provided. You can manually "
            "change it."
        ),
    )

    # ------------------------------------------------------
    # Apply efficiency
    # ------------------------------------------------------

    area_eff = apply_efficiency(
        area_df,
        fixed_loss,
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
    # Chart
    # ------------------------------------------------------

    st.plotly_chart(
        make_chart(
            forecast,
            actual,
            "Fixed Plant — Forecast vs Actual",
        ),
        use_container_width=True,
    )


# ==========================================================
# TRACKING PLANT
# ==========================================================

else:

    st.subheader(
        "🔄 Tracking Plant"
    )

    # ------------------------------------------------------
    # GHI
    # ------------------------------------------------------

    ghi_arrays = []

    for col in ghi_cols:

        if col == "GHI_Forecast":

            ghi = edited_df[
                "GHI"
            ].to_numpy(float)

        else:

            ghi = edited_df[
                col
            ].to_numpy(float)

        ghi_arrays.append(
            ghi
        )

    # ------------------------------------------------------
    # Tracking geometry
    # ------------------------------------------------------

    sin_ab, sin_a = solar_geometry(
        lat,
        tracking=True,
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

    else:

        weights_raw = None


    # ------------------------------------------------------
    # POA
    # ------------------------------------------------------

    poa_list = []

    for ghi in ghi_arrays:

        poa_list.append(
            ghi
            * sin_ab
            / max(
                sin_a,
                1e-6,
            )
        )


    # ------------------------------------------------------
    # Automatic efficiency loss
    # ------------------------------------------------------

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
        tuple(
            tuple(x)
            for x in poa_list
        ),
        is_cluster,
        weights_raw,
    )


    # ------------------------------------------------------
    # Manual efficiency loss
    # ------------------------------------------------------

    st.markdown(
        "### ⚙️ Efficiency Loss"
    )

    tracking_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=50.0,
        value=float(
            best_loss
        ),
        step=0.1,
        key="tracking_efficiency_loss",
        help=(
            "Automatically calculated value "
            "is provided. You can manually "
            "change it."
        ),
    )


    # ------------------------------------------------------
    # Apply efficiency
    # ------------------------------------------------------

    area_eff = apply_efficiency(
        area_df,
        tracking_loss,
    )


    # ------------------------------------------------------
    # Backend calculation
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

    blocks_tuple = tuple(
        blocks
    )

    ghi_tuple = tuple(
        tuple(x)
        for x in ghi_arrays
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
    # AUTOMATIC TRACKING PARAMETERS
    #
    # Only runs once and is cached.
    # ------------------------------------------------------

    if (
        "lc_tracking_params"
        not in st.session_state
    ):

        best = optimize_tracking(
            tuple(actual),
            blocks_tuple,
            ghi_tuple,
            weights_eff,
        )

        st.session_state[
            "lc_tracking_params"
        ] = {
            "DHI": int(best[0]),
            "start": int(best[1]),
            "end": int(best[2]),
            "max": int(best[3]),
            "east": int(best[4]),
            "west": int(best[5]),
        }


    params = st.session_state[
        "lc_tracking_params"
    ]


    # ------------------------------------------------------
    # Tracking parameters
    # ------------------------------------------------------

    st.markdown(
        "### 🎯 Tracking Parameters"
    )

    with st.form(
        "tracking_parameter_form"
    ):

        c1, c2, c3 = st.columns(
            3
        )

        DHI = c1.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(
                params["DHI"]
            ),
            step=1,
        )

        start = c2.number_input(
            "Starting Block",
            min_value=0,
            max_value=95,
            value=int(
                params["start"]
            ),
            step=1,
        )

        end = c3.number_input(
            "Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                params["end"]
            ),
            step=1,
        )

        c1, c2, c3 = st.columns(
            3
        )

        maximum = c1.number_input(
            "Max Block",
            min_value=1,
            max_value=96,
            value=int(
                params["max"]
            ),
            step=1,
        )

        east = c2.number_input(
            "East Limit",
            min_value=0,
            max_value=90,
            value=int(
                params["east"]
            ),
            step=1,
        )

        west = c3.number_input(
            "West Limit",
            min_value=0,
            max_value=90,
            value=int(
                params["west"]
            ),
            step=1,
        )

        recalc = st.form_submit_button(
            "🔄 Recalculate",
            use_container_width=True,
            type="primary",
        )


    if recalc:

        st.session_state[
            "lc_tracking_params"
        ] = {
            "DHI": int(DHI),
            "start": int(start),
            "end": int(end),
            "max": int(maximum),
            "east": int(east),
            "west": int(west),
        }

        params = (
            st.session_state[
                "lc_tracking_params"
            ]
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
