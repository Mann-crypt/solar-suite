# ============================================================
# STREAMLIT APP
# LOSS CORRECTION MODEL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loss Correction Model",
    page_icon="☀️",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_OPT_ITER = 40
OPT_POPSIZE = 15

CLUSTERS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

PARAM_BOUNDS = [
    (0, 10),      # DHI
    (10, 30),     # Starting block
    (65, 80),     # Ending block
    (47, 53),     # Max block
    (10, 70),     # East limit
    (10, 70),     # West limit
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #8b949e;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin: 18px 0 10px 0;
    }

    div.stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 650;
        transition: 0.15s ease;
    }

    .input-card {
        padding: 12px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "plant_type": "🏗️ Fixed",
    "tracking_params": None,
    "run_model": False,
    "input_df": None,
    "input_context": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def validate_columns(df, required, name="Data"):

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing: {', '.join(missing)}"
        )


def get_sheet_names(uploaded_file):

    uploaded_file.seek(0)

    return pd.ExcelFile(
        uploaded_file
    ).sheet_names


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def read_area_efficiency(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_excel(
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

    validate_columns(
        df,
        [
            "S.No.",
        ],
        "Area & Efficiency",
    )

    # Keep actual cluster/module rows only
    df = df[
        df["S.No."].notna()
    ].copy()

    return df.reset_index(drop=True)


# ============================================================
# FIXED EFFECTIVE AREAS
#
# Excel:
# P3:P7
# ============================================================

def read_fixed_effective_areas(uploaded_file):

    uploaded_file.seek(0)

    area_ws = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=None,
    )

    fixed_weights = pd.to_numeric(
        area_ws.iloc[2:7, 15],
        errors="coerce",
    ).fillna(0).to_numpy(dtype=float)

    if len(fixed_weights) != 5:
        raise ValueError(
            "Unable to read Fixed effective areas from P3:P7."
        )

    return dict(
        zip(
            CLUSTERS,
            fixed_weights,
        )
    )


# ============================================================
# TRACKING EFFECTIVE AREAS
#
# Excel:
# P29:P33
# ============================================================

def read_tracking_effective_areas(uploaded_file):

    uploaded_file.seek(0)

    area_ws = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=None,
    )

    tracking_weights = pd.to_numeric(
        area_ws.iloc[28:33, 15],
        errors="coerce",
    ).fillna(0).to_numpy(dtype=float)

    if len(tracking_weights) != 5:
        raise ValueError(
            "Unable to read Tracking effective areas from P29:P33."
        )

    return dict(
        zip(
            CLUSTERS,
            tracking_weights,
        )
    )


# ============================================================
# LATITUDE
# ============================================================

def read_latitude(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        ["Lat"],
        "Forecast Config",
    )

    return float(
        pd.to_numeric(
            df["Lat"].iloc[0],
            errors="coerce",
        )
    )


# ============================================================
# TILT LOOKUP
# ============================================================

def read_tilt_lookup(uploaded_file):

    try:

        uploaded_file.seek(0)

        df = pd.read_excel(
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
            return {}

        if "Month_Num" not in df.columns:
            return {}

        df = df.dropna(
            subset=["Fixed"]
        )

        df["Month_Num"] = pd.to_numeric(
            df["Month_Num"],
            errors="coerce",
        )

        df["Fixed"] = pd.to_numeric(
            df["Fixed"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "Month_Num",
                "Fixed",
            ]
        )

        return (
            df.set_index("Month_Num")["Fixed"]
            .to_dict()
        )

    except Exception:
        return {}


# ============================================================
# RESULT / GHI
# ============================================================

def read_result_ghi(uploaded_file):

    uploaded_file.seek(0)

    result = pd.read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=range(6),
    )

    if len(result.columns) < 6:
        raise ValueError(
            "Result sheet must contain Block + five GHI columns."
        )

    result.columns = [
        "Block",
        *GHI_COLS,
    ]

    result["Block"] = pd.to_numeric(
        result["Block"],
        errors="coerce",
    )

    for col in GHI_COLS:

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce",
        ).fillna(0)

    result = result.dropna(
        subset=["Block"]
    ).reset_index(drop=True)

    return result


# ============================================================
# FIXED-C11 INPUT
# ============================================================

def load_fixed_c11(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-C11",
        header=1,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        [
            "Date",
            "Actual",
        ],
        "Fixed-C11",
    )

    # --------------------------------------------------------
    # STOP AT FIRST BLANK DATE
    # --------------------------------------------------------

    date_valid = df["Date"].notna()

    if not date_valid.any():

        raise ValueError(
            "No valid Date rows found in Fixed-C11."
        )

    first_blank = np.where(
        ~date_valid.to_numpy()
    )[0]

    if len(first_blank) > 0:

        df = df.iloc[
            :first_blank[0]
        ].copy()

    else:

        df = df.loc[
            date_valid
        ].copy()

    # --------------------------------------------------------
    # CONVERT DATE / ACTUAL
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    if len(df) == 0:

        raise ValueError(
            "No valid forecast rows found in Fixed-C11."
        )

    return df.reset_index(drop=True)


# ============================================================
# ALIGN INPUT DATA
# ============================================================

def load_model_data(uploaded_file):

    fixed_c11 = load_fixed_c11(
        uploaded_file
    )

    result = read_result_ghi(
        uploaded_file
    )

    n = min(
        len(fixed_c11),
        len(result),
    )

    if n == 0:

        raise ValueError(
            "No common forecast rows available."
        )

    fixed_c11 = fixed_c11.iloc[
        :n
    ].copy()

    result = result.iloc[
        :n
    ].copy()

    fixed_c11["Block"] = result[
        "Block"
    ].to_numpy()

    for col in GHI_COLS:

        fixed_c11[col] = result[
            col
        ].to_numpy()

    return fixed_c11


# ============================================================
# INPUT DATA EDITOR
# ============================================================

def input_data_editor(df):

    st.markdown(
        '<div class="section-title">📊 Input GHI and Power</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "GHI values come from the Result sheet and Actual "
        "power comes from Fixed-C11. You can modify them here."
    )

    edit_cols = [
        "Actual",
        *GHI_COLS,
    ]

    display = df[
        edit_cols
    ].copy()

    edited = st.data_editor(
        display,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="input_editor",
        column_config={
            col: st.column_config.NumberColumn(
                col,
                step=0.01,
                format="%.2f",
            )
            for col in edit_cols
        },
    )

    result = df.copy()

    for col in edit_cols:

        result[col] = pd.to_numeric(
            edited[col],
            errors="coerce",
        ).fillna(0)

    return result


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def prepare_solar_geometry(
    df,
    lat,
    tilt_lookup,
    tracking=False,
):

    result = df.copy()

    dates = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    if dates.isna().any():

        raise ValueError(
            "Invalid dates found in Fixed-C11."
        )

    # --------------------------------------------------------
    # Excel reference date
    # 1-Jan-2025
    # --------------------------------------------------------

    first_date = pd.Timestamp(
        year=2025,
        month=1,
        day=1,
    )

    day_offset = (
        dates - first_date
    ).dt.days.to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # DECLINATION
    # --------------------------------------------------------

    declination = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_offset
                    + 1
                )
                / 365
            )
        )
    )

    # --------------------------------------------------------
    # ELEVATION
    # --------------------------------------------------------

    elevation = (
        90
        - lat
        + declination
    )

    # --------------------------------------------------------
    # TILT
    # --------------------------------------------------------

    if tracking:

        tilt = np.zeros(
            len(result),
            dtype=float,
        )

    else:

        months = dates.dt.month

        tilt = np.array(
            [
                tilt_lookup.get(
                    int(month),
                    0,
                )
                for month in months
            ],
            dtype=float,
        )

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(
            elevation + tilt
        )
    )

    sin_a_safe = np.where(
        np.abs(sin_a) > 1e-8,
        sin_a,
        1e-8,
    )

    result["Declination"] = declination
    result["Elevation"] = elevation
    result["Tilt"] = tilt
    result["Sin(a)"] = sin_a_safe
    result["Sin(a+b)"] = sin_ab

    return result


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df,
    fixed_effective_areas,
    lat,
    tilt_lookup,
):

    solar = prepare_solar_geometry(
        df,
        lat,
        tilt_lookup,
        tracking=False,
    )

    forecast_matrix = []

    for cluster in CLUSTERS:

        ghi_col = f"GHI {cluster}"

        poa = (
            solar[ghi_col].to_numpy(float)
            * solar["Sin(a+b)"].to_numpy(float)
            / solar["Sin(a)"].to_numpy(float)
        )

        effective_area = (
            fixed_effective_areas[cluster]
        )

        power = (
            poa
            * effective_area
            / 1_000_000
        )

        forecast_matrix.append(
            power
        )

    forecast_matrix = np.column_stack(
        forecast_matrix
    )

    forecast = np.sum(
        forecast_matrix,
        axis=1,
    )

    return (
        forecast,
        forecast_matrix,
        solar,
    )


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    df,
    tracking_effective_areas,
    params,
):

    DHI = int(params["DHI"])
    start_block = int(params["start"])
    end_block = int(params["end"])
    max_block = int(params["max"])
    east_limit = int(params["east"])
    west_limit = int(params["west"])

    blocks = df[
        "Block"
    ].to_numpy(float)

    if not (
        start_block
        < max_block
        < end_block
    ):

        raise ValueError(
            "Starting Block < Max Block < Ending Block is required."
        )

    denominator_1 = (
        start_block
        - 1
        - max_block
    )

    denominator_2 = (
        end_block
        + 1
        - max_block
    )

    if (
        denominator_1 == 0
        or denominator_2 == 0
    ):

        raise ValueError(
            "Invalid tracking block configuration."
        )

    m1 = (
        90
        / denominator_1
    )

    m2 = (
        90
        / denominator_2
    )

    # --------------------------------------------------------
    # ZENITH
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PANEL ANGLE
    # --------------------------------------------------------

    panel = np.where(

        blocks < max_block,

        np.where(
            zenith < abs(east_limit),
            zenith,
            abs(east_limit),
        ),

        np.where(

            (
                (blocks > max_block)
                &
                (zenith > west_limit)
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

    # --------------------------------------------------------
    # GHI MATRIX
    # --------------------------------------------------------

    ghi_matrix = df[
        GHI_COLS
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # DHI
    #
    # DHI = GHI * DHI%
    # --------------------------------------------------------

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    # --------------------------------------------------------
    # DNI
    #
    # DNI = (GHI - DHI) / COS(alpha)
    # --------------------------------------------------------

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    # --------------------------------------------------------
    # TRACKING POWER
    #
    # DNI * Tracking Effective Area
    # --------------------------------------------------------

    tracking_area = np.array(
        [
            tracking_effective_areas[c]
            for c in CLUSTERS
        ],
        dtype=float,
    )

    tracking_power_matrix = (
        dni
        * tracking_area[None, :]
        / 1_000_000
    )

    forecast = np.sum(
        tracking_power_matrix,
        axis=1,
    )

    return (
        forecast,
        tracking_power_matrix,
        zenith,
        panel,
        dni,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_tracking_cached(
    blocks_tuple,
    ghi_matrix_tuple,
    tracking_area_tuple,
    actual_tuple,
):

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    ghi_matrix = np.asarray(
        ghi_matrix_tuple,
        dtype=float,
    )

    tracking_area = np.asarray(
        tracking_area_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    # --------------------------------------------------------
    # DAYLIGHT / VALID MASK
    # --------------------------------------------------------

    valid_mask = (
        np.isfinite(actual)
        &
        np.all(
            np.isfinite(ghi_matrix),
            axis=1,
        )
        &
        (actual > 0)
    )

    actual_day = actual[
        valid_mask
    ]

    blocks_day = blocks[
        valid_mask
    ]

    ghi_day = ghi_matrix[
        valid_mask
    ]

    if len(actual_day) == 0:

        raise ValueError(
            "No valid daylight Actual values found."
        )

    actual_max = np.max(
        actual_day
    )

    actual_energy = np.sum(
        actual_day
    )

    if (
        actual_max <= 0
        or actual_energy <= 0
    ):

        raise ValueError(
            "Actual power data is invalid."
        )

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    def objective(x):

        DHI = int(
            round(x[0])
        )

        start_block = int(
            round(x[1])
        )

        end_block = int(
            round(x[2])
        )

        max_block = int(
            round(x[3])
        )

        east_limit = int(
            round(x[4])
        )

        west_limit = int(
            round(x[5])
        )

        if not (
            start_block
            < max_block
            < end_block
        ):

            return 1e9

        denominator_1 = (
            start_block
            - 1
            - max_block
        )

        denominator_2 = (
            end_block
            + 1
            - max_block
        )

        if (
            denominator_1 == 0
            or denominator_2 == 0
        ):

            return 1e9

        m1 = (
            90
            / denominator_1
        )

        m2 = (
            90
            / denominator_2
        )

        # ----------------------------------------------------
        # ZENITH
        # ----------------------------------------------------

        zenith = np.where(
            blocks_day <= max_block,

            np.minimum(
                89,
                m1
                * (
                    blocks_day
                    - max_block
                ),
            ),

            np.minimum(
                89,
                m2
                * (
                    blocks_day
                    - max_block
                ),
            ),
        )

        # ----------------------------------------------------
        # PANEL
        # ----------------------------------------------------

        panel = np.where(

            blocks_day < max_block,

            np.where(
                zenith < abs(east_limit),
                zenith,
                abs(east_limit),
            ),

            np.where(

                (
                    (blocks_day > max_block)
                    &
                    (zenith > west_limit)
                ),

                west_limit,

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

        # ----------------------------------------------------
        # DHI
        # ----------------------------------------------------

        dhi = (
            ghi_day
            * DHI
            / 100
        )

        # ----------------------------------------------------
        # DNI
        # ----------------------------------------------------

        dni = (
            ghi_day
            - dhi
        ) / cos_alpha[:, None]

        # ----------------------------------------------------
        # POWER
        # ----------------------------------------------------

        prediction_matrix = (
            dni
            * tracking_area[None, :]
            / 1_000_000
        )

        prediction = np.sum(
            prediction_matrix,
            axis=1,
        )

        if not np.all(
            np.isfinite(prediction)
        ):

            return 1e9

        # ----------------------------------------------------
        # ERROR METRICS
        # ----------------------------------------------------

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
                - np.max(prediction)
            )
            / actual_max
        )

        energy_error = (
            abs(
                actual_energy
                - np.sum(prediction)
            )
            / actual_energy
        )

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return score

    # --------------------------------------------------------
    # DIFFERENTIAL EVOLUTION
    # --------------------------------------------------------

    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
        strategy="best1bin",
        maxiter=MAX_OPT_ITER,
        popsize=OPT_POPSIZE,
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    best = np.rint(
        result.x
    ).astype(int)

    return {
        "DHI": int(best[0]),
        "start": int(best[1]),
        "end": int(best[2]),
        "max": int(best[3]),
        "east": int(best[4]),
        "west": int(best[5]),
    }


# ============================================================
# TRACKING PARAMETER UI
# ============================================================

def tracking_parameter_controls(
    params,
    prefix,
):

    st.markdown(
        '<div class="section-title">⚙️ Tracking Parameters</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Optimizer values are loaded automatically. "
        "You can manually modify them before recalculating."
    )

    c1, c2, c3 = st.columns(3)

    DHI = c1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=10,
        value=int(params["DHI"]),
        step=1,
        key=f"{prefix}_dhi",
    )

    start = c2.number_input(
        "Starting Block",
        min_value=10,
        max_value=30,
        value=int(params["start"]),
        step=1,
        key=f"{prefix}_start",
    )

    end = c3.number_input(
        "Ending Block",
        min_value=65,
        max_value=80,
        value=int(params["end"]),
        step=1,
        key=f"{prefix}_end",
    )

    c1, c2, c3 = st.columns(3)

    max_block = c1.number_input(
        "Max Block",
        min_value=47,
        max_value=53,
        value=int(params["max"]),
        step=1,
        key=f"{prefix}_max",
    )

    east = c2.number_input(
        "East Limit",
        min_value=10,
        max_value=70,
        value=int(params["east"]),
        step=1,
        key=f"{prefix}_east",
    )

    west = c3.number_input(
        "West Limit",
        min_value=10,
        max_value=70,
        value=int(params["west"]),
        step=1,
        key=f"{prefix}_west",
    )

    return {
        "DHI": int(DHI),
        "start": int(start),
        "end": int(end),
        "max": int(max_block),
        "east": int(east),
        "west": int(west),
    }


# ============================================================
# EFFICIENCY / AREA TABLE
# ============================================================

def show_effective_area_table(
    fixed_areas,
    tracking_areas,
):

    display = pd.DataFrame({
        "Cluster": CLUSTERS,
        "Fixed Effective Area (m²)": [
            fixed_areas[c]
            for c in CLUSTERS
        ],
        "Tracking Effective Area (m²)": [
            tracking_areas[c]
            for c in CLUSTERS
        ],
    })

    display[
        [
            "Fixed Effective Area (m²)",
            "Tracking Effective Area (m²)",
        ]
    ] = display[
        [
            "Fixed Effective Area (m²)",
            "Tracking Effective Area (m²)",
        ]
    ].round(4)

    with st.expander(
        "🔍 View Effective Area Calculations"
    ):

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CHART
# ============================================================

def show_forecast_chart(
    forecast,
    actual,
    title,
):

    n = min(
        len(forecast),
        len(actual),
    )

    x = np.arange(
        1,
        n + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(
                forecast[:n]
            ),
            mode="lines",
            name="Forecast",
            line=dict(
                color="#3B82F6",
                width=2.5,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(
                actual[:n]
            ),
            mode="lines",
            name="Actual",
            line=dict(
                color="#EF4444",
                width=2.5,
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=480,
        hovermode="x unified",
        template="plotly_white",
        xaxis_title="15 Minute Block",
        yaxis_title="Power (MW)",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# METRICS
# ============================================================

def show_metrics(
    forecast,
    actual,
):

    n = min(
        len(forecast),
        len(actual),
    )

    forecast = np.asarray(
        forecast[:n],
        dtype=float,
    )

    actual = np.asarray(
        actual[:n],
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        np.isfinite(forecast)
        &
        (actual > 0)
    )

    if not mask.any():
        return

    act = actual[mask]
    pred = forecast[mask]

    block_error = (
        np.mean(
            np.abs(
                act - pred
            )
        )
        / np.max(act)
    )

    peak_error = (
        abs(
            np.max(act)
            - np.max(pred)
        )
        / np.max(act)
    )

    energy_error = (
        abs(
            np.sum(act)
            - np.sum(pred)
        )
        / np.sum(act)
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Block Error",
        f"{block_error * 100:.2f}%",
    )

    c2.metric(
        "Peak Error",
        f"{peak_error * 100:.2f}%",
    )

    c3.metric(
        "Energy Error",
        f"{energy_error * 100:.2f}%",
    )


# ============================================================
# PLANT SELECTOR
# ============================================================

def plant_selector():

    st.markdown(
        '<div class="section-title">🏭 Select Plant Type</div>',
        unsafe_allow_html=True,
    )

    return st.segmented_control(
        "Plant Type",
        options=[
            "🏗️ Fixed",
            "🔄 Tracking",
        ],
        default="🏗️ Fixed",
        selection_mode="single",
        key="plant_type_selector",
        label_visibility="collapsed",
        width="stretch",
    )


# ============================================================
# RUN FIXED
# ============================================================

def run_fixed(
    input_df,
    fixed_areas,
    tracking_areas,
    lat,
    tilt_lookup,
):

    with st.spinner(
        "☀️ Calculating fixed forecast..."
    ):

        forecast, forecast_matrix, solar = (
            calculate_fixed_forecast(
                input_df,
                fixed_areas,
                lat,
                tilt_lookup,
            )
        )

    show_effective_area_table(
        fixed_areas,
        tracking_areas,
    )

    show_metrics(
        forecast,
        input_df["Actual"].to_numpy(),
    )

    show_forecast_chart(
        forecast,
        input_df["Actual"],
        "🏗️ Fixed Forecast vs Actual",
    )


# ============================================================
# RUN TRACKING
# ============================================================

def run_tracking(
    input_df,
    fixed_areas,
    tracking_areas,
):

    blocks = input_df[
        "Block"
    ].to_numpy(float)

    ghi_matrix = input_df[
        GHI_COLS
    ].to_numpy(float)

    actual = input_df[
        "Actual"
    ].to_numpy(float)

    tracking_area_array = np.array(
        [
            tracking_areas[c]
            for c in CLUSTERS
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # OPTIMIZE ONLY ON FIRST RUN
    # --------------------------------------------------------

    if (
        st.session_state.tracking_params
        is None
    ):

        with st.spinner(
            "🔄 Optimizing tracking parameters..."
        ):

            result = optimize_tracking_cached(
                tuple(blocks),
                tuple(
                    map(
                        tuple,
                        ghi_matrix,
                    )
                ),
                tuple(
                    tracking_area_array
                ),
                tuple(actual),
            )

        st.session_state.tracking_params = (
            result
        )

    params = tracking_parameter_controls(
        st.session_state.tracking_params,
        "tracking",
    )

    # --------------------------------------------------------
    # VALIDATE PARAMETERS
    # --------------------------------------------------------

    if not (
        params["start"]
        < params["max"]
        < params["end"]
    ):

        st.error(
            "Starting Block < Max Block < Ending Block is required."
        )

        return

    # --------------------------------------------------------
    # CALCULATE
    # --------------------------------------------------------

    try:

        (
            forecast,
            tracking_matrix,
            zenith,
            panel,
            dni,
        ) = calculate_tracking(
            input_df,
            tracking_areas,
            params,
        )

        show_effective_area_table(
            fixed_areas,
            tracking_areas,
        )

        show_metrics(
            forecast,
            actual,
        )

        show_forecast_chart(
            forecast,
            actual,
            "🔄 Tracking Forecast vs Actual",
        )

        # ----------------------------------------------------
        # TRACKING DETAILS
        # ----------------------------------------------------

        with st.expander(
            "🔍 View Tracking Calculations"
        ):

            tracking_detail = pd.DataFrame({
                "Block": blocks,
                "Zenith Angle": zenith,
                "Panel Angle": panel,
            })

            for i, cluster in enumerate(
                CLUSTERS
            ):

                tracking_detail[
                    f"DNI {cluster}"
                ] = dni[:, i]

                tracking_detail[
                    f"Power {cluster}"
                ] = tracking_matrix[:, i]

            st.dataframe(
                tracking_detail.round(4),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as e:

        st.error(
            f"Unable to calculate tracking forecast: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    st.markdown(
        '<div class="main-title">☀️ Loss Correction Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Upload the cluster workbook, modify GHI and Actual "
        "values, select the plant type and run the correction."
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    st.markdown(
        '<div class="section-title">📁 Input Workbook</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx", "xls"],
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload the plant Excel workbook to begin."
        )

        return

    # ========================================================
    # CHECK REQUIRED SHEETS
    # ========================================================

    try:

        sheets = get_sheet_names(
            uploaded_file
        )

        required_sheets = [
            "Area & Efficiency",
            "Forecast Config",
            "Config Tilt Angle",
            "Result",
            "Fixed-C11",
        ]

        missing = [
            s
            for s in required_sheets
            if s not in sheets
        ]

        if missing:

            st.error(
                "Required sheets are missing: "
                + ", ".join(missing)
            )

            return

    except Exception as e:

        st.error(
            f"Unable to read workbook: {e}"
        )

        return

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    try:

        area_efficiency = (
            read_area_efficiency(
                uploaded_file
            )
        )

        fixed_areas = (
            read_fixed_effective_areas(
                uploaded_file
            )
        )

        tracking_areas = (
            read_tracking_effective_areas(
                uploaded_file
            )
        )

        lat = read_latitude(
            uploaded_file
        )

        tilt_lookup = read_tilt_lookup(
            uploaded_file
        )

        input_df = load_model_data(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to load workbook configuration: {e}"
        )

        st.exception(e)

        return

    # ========================================================
    # BASIC INFO
    # ========================================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Forecast Blocks",
        len(input_df),
    )

    c2.metric(
        "Latitude",
        f"{lat:.4f}",
    )

    c3.metric(
        "Clusters",
        len(CLUSTERS),
    )

    # ========================================================
    # INPUT EDITOR
    # ========================================================

    input_df = input_data_editor(
        input_df
    )

    st.session_state.input_df = (
        input_df
    )

    # ========================================================
    # PLANT TYPE
    # ========================================================

    plant_type = plant_selector()

    # ========================================================
    # RUN BUTTON
    # ========================================================

    st.markdown("")

    run_clicked = st.button(
        "🚀 RUN LOSS CORRECTION",
        type="primary",
        use_container_width=True,
        key="run_loss_correction",
    )

    if run_clicked:

        # ----------------------------------------------------
        # Always reset optimizer when user explicitly runs
        # ----------------------------------------------------

        st.session_state.tracking_params = None
        st.session_state.run_model = True

    if not st.session_state.run_model:

        st.info(
            "Select the plant type and click "
            "**Run Loss Correction** to start."
        )

        return

    # ========================================================
    # MODEL
    # ========================================================

    try:

        if plant_type == "🏗️ Fixed":

            run_fixed(
                input_df,
                fixed_areas,
                tracking_areas,
                lat,
                tilt_lookup,
            )

        else:

            run_tracking(
                input_df,
                fixed_areas,
                tracking_areas,
            )

    except Exception as e:

        st.error(
            "❌ Loss correction failed."
        )

        st.exception(e)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
