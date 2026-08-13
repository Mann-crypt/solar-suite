# ============================================================
# STREAMLIT APP
# LOSS CORRECTION MODEL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.optimize import differential_evolution
import hashlib


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
OPT_POPSIZE = 10

PARAM_BOUNDS = [
    (0, 10),      # DHI %
    (0, 30),      # Starting block
    (65, 80),     # Ending block
    (44, 60),     # Max block
    (0, 70),      # East limit
    (0, 70),      # West limit
]

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
    "😎 6:30 hote hi Billu MAL se faraar",
    "😇 Guruji ne ek baat kahi thi....",
    "🎼 Karna hai kuchh kaam M se gaao...",
    "😠 Nahi karni Loss Correction, Now what to do?...",
    "💸 Iss Job ko chhod or chhod kar ameer ho..",
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "plant_type": "🏗️ Fixed",
    "model_context": None,
    "run_requested": False,
    "optimization_result": None,
    "calculation_result": None,
    "input_data": None,
    "input_signature": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.3rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.7rem;
    }

    .info-card {
        padding: 16px 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.08);
        margin-bottom: 15px;
    }

    .run-card {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(37,99,235,0.35);
        background: rgba(37,99,235,0.08);
        margin: 15px 0;
    }

    /* Plant buttons */
    div[data-testid="stButton"] > button {
        min-height: 62px;
        border-radius: 14px;
        font-size: 17px;
        font-weight: 700;
        transition: all 0.15s ease;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
    }

    /* Run button */
    .run-button div[data-testid="stButton"] > button {
        min-height: 58px;
        font-size: 18px;
        font-weight: 750;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def validate_columns(df, required_columns, dataframe_name="Data"):

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required column(s): "
            f"{', '.join(missing)}"
        )


def get_sheet_names(uploaded_file):

    uploaded_file.seek(0)

    return pd.ExcelFile(
        uploaded_file
    ).sheet_names


def clean_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


def clean_data_rows(df, date_column="Date"):

    df = df.copy()

    if date_column in df.columns:

        null_indices = df[
            df[date_column].isna()
        ].index

        if len(null_indices) > 0:

            first_null_position = (
                df.index.get_loc(
                    null_indices[0]
                )
            )

            df = df.iloc[
                :first_null_position
            ]

    return df.reset_index(drop=True)


def file_signature(uploaded_file):

    uploaded_file.seek(0)

    data = uploaded_file.getvalue()

    return hashlib.md5(data).hexdigest()


# ============================================================
# DETECT CLUSTER
# ============================================================

def detect_cluster_model(uploaded_file):

    sheets = get_sheet_names(
        uploaded_file
    )

    return "Fixed" not in sheets


# ============================================================
# READ AREA & EFFICIENCY
# ============================================================

def read_area_efficiency(
    uploaded_file,
    cluster=False,
):

    uploaded_file.seek(0)

    if cluster:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Area & Efficiency",
            header=1,
            usecols=range(8),
        )

    else:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Area & Efficiency",
            header=1,
        )

    df = clean_columns(df)

    validate_columns(
        df,
        [
            "Module Type",
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        "Area & Efficiency",
    )

    if "Module Type" in df.columns:

        null_indices = df[
            df["Module Type"].isna()
        ].index

        if len(null_indices) > 0:

            first_null_position = (
                df.index.get_loc(
                    null_indices[0]
                )
            )

            df = df.iloc[
                :first_null_position
            ]

    df = df.dropna(
        subset=[
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        how="all",
    )

    return df.reset_index(drop=True)


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

    df = clean_columns(df)

    validate_columns(
        df,
        ["Lat"],
        "Forecast Config",
    )

    return float(
        pd.to_numeric(
            df["Lat"],
            errors="coerce",
        ).dropna().iloc[0]
    )


# ============================================================
# TILT LOOKUP
# ============================================================

def read_tilt_lookup(uploaded_file):

    uploaded_file.seek(0)

    try:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Config Tilt Angle",
            header=7,
        )

    except Exception:

        return {}

    df = clean_columns(df)

    if "Fixed" not in df.columns:
        return {}

    null_indices = df[
        df["Fixed"].isna()
    ].index

    if len(null_indices) > 0:

        first_null_position = (
            df.index.get_loc(
                null_indices[0]
            )
        )

        df = df.iloc[
            :first_null_position
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

    if (
        "Month" not in df.columns
        or "Fixed" not in df.columns
    ):
        return {}

    result = (
        df.dropna(
            subset=["Month"]
        )
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    return result


# ============================================================
# CLUSTER WEIGHTS
# ============================================================

def read_cluster_weights(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[12, 13, 14, 15, 16],
    )

    df = clean_columns(df)

    required = [
        "CL-1",
        "CL-2",
        "CL-3",
        "CL-4",
        "CL-5",
    ]

    validate_columns(
        df,
        required,
        "Cluster Weights",
    )

    return {
        col: float(
            pd.to_numeric(
                df[col],
                errors="coerce",
            ).dropna().iloc[0]
        )
        for col in required
    }


# ============================================================
# SOLAR ANGLES
# ============================================================

def prepare_solar_angles(
    df_fix,
    lat,
    tilt_lookup=None,
    tracking=False,
):

    df_fix = df_fix.copy()

    today = pd.Timestamp.today().normalize()

    df_fix["Date"] = today

    first_date = today.replace(
        month=1,
        day=1,
    )

    day_number = (
        df_fix["Date"]
        - first_date
    ).dt.days + 1

    df_fix[
        "Declination Angle ∆"
    ] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + day_number)
                / 365
            )
        )
    )

    df_fix[
        "Elevation angle a"
    ] = (
        90
        - lat
        + df_fix[
            "Declination Angle ∆"
        ]
    )

    if tracking:

        df_fix["Tilt Angle b"] = 0

    else:

        if not tilt_lookup:

            df_fix["Tilt Angle b"] = 0

        else:

            df_fix["Tilt Angle b"] = (
                df_fix["Date"]
                .dt.strftime("%B")
                .map(tilt_lookup)
                .fillna(0)
            )

    df_fix["a+b"] = (
        df_fix["Elevation angle a"]
        + df_fix["Tilt Angle b"]
    )

    df_fix["SIN(a+b)"] = np.sin(
        np.radians(
            df_fix["a+b"]
        )
    )

    df_fix["Sin(a)"] = np.sin(
        np.radians(
            df_fix["Elevation angle a"]
        )
    )

    df_fix["Sin(a)"] = (
        df_fix["Sin(a)"]
        .clip(lower=1e-6)
    )

    return df_fix


# ============================================================
# EFFICIENCY LOSS
# ============================================================

def calculate_efficiency_loss(
    df,
    poa,
    actual,
):

    standard_eff = df[
        "Standard PV Efficiency (%)"
    ].to_numpy(dtype=float)

    area = df[
        "Total area(m2)"
    ].to_numpy(dtype=float)

    actual = np.asarray(
        actual,
        dtype=float,
    )

    poa = np.asarray(
        poa,
        dtype=float,
    )

    valid_actual = actual[
        np.isfinite(actual)
    ]

    valid_poa = poa[
        np.isfinite(poa)
    ]

    if len(valid_actual) == 0:
        return 0.0

    if len(valid_poa) == 0:
        return 0.0

    max_loss = float(
        np.nanmin(
            standard_eff
        )
    )

    actual_peak = float(
        np.nanmax(
            valid_actual
        )
    )

    poa_peak = float(
        np.nanmax(
            valid_poa
        )
    )

    if poa_peak <= 0:
        return 0.0

    base_area = np.sum(
        area
        * standard_eff
        / 100
    )

    loss_area_coefficient = np.sum(
        area / 100
    )

    if loss_area_coefficient <= 0:
        return 0.0

    target_area = (
        actual_peak
        * 1_000_000
        / poa_peak
    )

    best_loss = (
        base_area
        - target_area
    ) / loss_area_coefficient

    best_loss = np.clip(
        best_loss,
        0,
        max_loss,
    )

    return float(best_loss)


def apply_efficiency_loss(
    df,
    best_loss,
):

    df = df.copy()

    df[
        "Efficiency Losses(%)"
    ] = float(best_loss)

    df[
        "Net Efficiency (%)"
    ] = (
        df[
            "Standard PV Efficiency (%)"
        ]
        - float(best_loss)
    )

    df["Eff Area"] = (
        df[
            "Total area(m2)"
        ]
        * df[
            "Net Efficiency (%)"
        ]
        / 100
    )

    return df


# ============================================================
# INPUT DATA
# ============================================================

def load_noncluster_input(
    uploaded_file
):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed",
        header=1,
    )

    df = clean_columns(
        clean_data_rows(df)
    )

    validate_columns(
        df,
        [
            "GHI_Forecast",
            "Actual",
        ],
        "Fixed",
    )

    result = pd.DataFrame()

    result["GHI_Forecast"] = pd.to_numeric(
        df["GHI_Forecast"],
        errors="coerce",
    ).fillna(0)

    result["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    return result


def load_cluster_input(
    uploaded_file
):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-CL1",
        header=1,
    )

    df = clean_columns(
        clean_data_rows(df)
    )

    validate_columns(
        df,
        ["Actual"],
        "Fixed-CL1",
    )

    result = pd.DataFrame()

    result["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

    # Try Result sheet if cluster GHI isn't present
    uploaded_file.seek(0)

    try:

        result_ghi = pd.read_excel(
            uploaded_file,
            sheet_name="Result",
            usecols=[0, 1, 2, 3, 4, 5],
        )

        result_ghi = result_ghi.fillna(0)

    except Exception:

        result_ghi = None

    for i, col in enumerate(ghi_columns):

        if col in df.columns:

            values = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0).to_numpy()

        elif (
            result_ghi is not None
            and i < len(result_ghi.columns)
        ):

            values = pd.to_numeric(
                result_ghi.iloc[
                    :len(result),
                    i,
                ],
                errors="coerce",
            ).fillna(0).to_numpy()

            if len(values) < len(result):

                values = np.pad(
                    values,
                    (
                        0,
                        len(result) - len(values),
                    ),
                    constant_values=0,
                )

        else:

            values = np.zeros(
                len(result)
            )

        result[col] = values

    return result


# ============================================================
# INPUT DATA EDITOR
# ============================================================

def input_data_editor(
    uploaded_file,
    is_cluster,
):

    st.markdown(
        '<div class="section-title">📊 Input Data</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Only GHI Forecast and Actual values are editable. "
        "All other model parameters are extracted automatically from the workbook."
    )

    if st.session_state.input_data is None:

        if is_cluster:

            input_df = load_cluster_input(
                uploaded_file
            )

        else:

            input_df = load_noncluster_input(
                uploaded_file
            )

        st.session_state.input_data = input_df.copy()

    input_df = st.session_state.input_data.copy()

    editable_config = {
        col: st.column_config.NumberColumn(
            col,
            format="%.2f",
        )
        for col in input_df.columns
    }

    with st.form(
        "input_data_form",
        clear_on_submit=False,
    ):

        edited_df = st.data_editor(
            input_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config=editable_config,
            key="input_data_editor",
        )

        submitted = st.form_submit_button(
            "💾 Apply Input Data Changes",
            use_container_width=True,
        )

    if submitted:

        edited_df = edited_df.copy()

        for col in edited_df.columns:

            edited_df[col] = pd.to_numeric(
                edited_df[col],
                errors="coerce",
            ).fillna(0)

        st.session_state.input_data = (
            edited_df
        )

        # Existing optimization result becomes invalid
        st.session_state.optimization_result = None
        st.session_state.calculation_result = None

        st.success(
            "✅ Input data updated. "
            "Click 'Run Loss Correction' to recalculate."
        )

    return st.session_state.input_data.copy()


# ============================================================
# PLANT TYPE BUTTONS
# ============================================================

def plant_type_selector():

    st.markdown(
        '<div class="section-title">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        fixed_selected = (
            st.session_state.plant_type
            == "🏗️ Fixed"
        )

        label = (
            "🏗️ FIXED PLANT  ✓"
            if fixed_selected
            else "🏗️ FIXED PLANT"
        )

        if st.button(
            label,
            key="fixed_button",
            use_container_width=True,
        ):

            if st.session_state.plant_type != "🏗️ Fixed":

                st.session_state.plant_type = (
                    "🏗️ Fixed"
                )

                st.session_state.optimization_result = None
                st.session_state.calculation_result = None

                st.rerun()

    with col2:

        tracking_selected = (
            st.session_state.plant_type
            == "🔄 Tracking"
        )

        label = (
            "🔄 TRACKING PLANT  ✓"
            if tracking_selected
            else "🔄 TRACKING PLANT"
        )

        if st.button(
            label,
            key="tracking_button",
            use_container_width=True,
        ):

            if st.session_state.plant_type != "🔄 Tracking":

                st.session_state.plant_type = (
                    "🔄 Tracking"
                )

                st.session_state.optimization_result = None
                st.session_state.calculation_result = None

                st.rerun()

    if st.session_state.plant_type == "🏗️ Fixed":

        st.success(
            "🏗️ Fixed plant selected"
        )

    else:

        st.info(
            "🔄 Tracking plant selected"
        )

    return st.session_state.plant_type


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    blocks,
    weighted_ghi,
    actual,
):

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    weighted_ghi = np.asarray(
        weighted_ghi,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        & (actual != 0)
        & np.isfinite(weighted_ghi)
        & np.isfinite(blocks)
    )

    actual_day = actual[mask]
    weighted_ghi_day = weighted_ghi[mask]
    blocks_day = blocks[mask]

    if len(actual_day) == 0:

        raise ValueError(
            "No valid non-zero Actual power values found."
        )

    actual_peak = np.max(
        actual_day
    )

    actual_energy = np.sum(
        actual_day
    )

    if actual_peak <= 0:

        raise ValueError(
            "Actual peak power is zero."
        )

    if actual_energy <= 0:

        raise ValueError(
            "Actual energy is zero."
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

        if (
            denominator_1 == 0
            or denominator_2 == 0
        ):
            return 1e9

        m1 = 90 / denominator_1
        m2 = 90 / denominator_2

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

        panel = np.where(
            blocks_day < max_block,

            np.minimum(
                zenith,
                abs(east),
            ),

            np.where(
                (
                    (blocks_day > max_block)
                    & (zenith > west)
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

        dhi_factor = (
            1 - DHI / 100
        )

        prediction = (
            weighted_ghi_day
            * dhi_factor
            / cos_alpha
            / 1_000_000
        )

        if (
            np.isnan(prediction).any()
            or np.isinf(prediction).any()
        ):
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    - prediction
                )
            )
            / actual_peak
        )

        peak_error = (
            abs(
                actual_peak
                - np.max(prediction)
            )
            / actual_peak
        )

        energy_error = (
            abs(
                actual_energy
                - np.sum(prediction)
            )
            / actual_energy
        )

        return float(
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
        strategy="best1bin",
        maxiter=MAX_OPT_ITER,
        popsize=OPT_POPSIZE,
        tol=0.005,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=False,
        workers=1,
        integrality=[
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    )

    best = np.rint(
        result.x
    ).astype(int)

    return {
        "score": float(result.fun),
        "DHI": int(best[0]),
        "start": int(best[1]),
        "end": int(best[2]),
        "max": int(best[3]),
        "east": int(best[4]),
        "west": int(best[5]),
    }


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    blocks,
    weighted_ghi,
    params,
):

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    weighted_ghi = np.asarray(
        weighted_ghi,
        dtype=float,
    )

    DHI = int(params["DHI"])
    start = int(params["start"])
    end = int(params["end"])
    max_block = int(params["max"])
    east = int(params["east"])
    west = int(params["west"])

    if not (
        start < max_block < end
    ):

        raise ValueError(
            "Starting Block < Max Block < Ending Block is required."
        )

    denominator_1 = (
        start - 1 - max_block
    )

    denominator_2 = (
        end + 1 - max_block
    )

    if (
        denominator_1 == 0
        or denominator_2 == 0
    ):

        raise ValueError(
            "Invalid tracking block configuration."
        )

    m1 = 90 / denominator_1
    m2 = 90 / denominator_2

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
                & (zenith > west)
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

    dhi_factor = (
        1 - DHI / 100
    )

    return (
        weighted_ghi
        * dhi_factor
        / cos_alpha
        / 1_000_000
    )


# ============================================================
# EFFICIENCY TABLE
# ============================================================

def show_efficiency_table(df):

    columns = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)",
    ]

    display_df = df[
        columns
    ].copy()

    numeric_cols = (
        display_df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    display_df[
        numeric_cols
    ] = display_df[
        numeric_cols
    ].round(2)

    with st.expander(
        "🔍 View Efficiency Calculations",
        expanded=False,
    ):

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FORECAST CHART
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

    forecast = np.asarray(
        forecast[:n],
        dtype=float,
    )

    actual = np.asarray(
        actual[:n],
        dtype=float,
    )

    x = np.arange(
        1,
        n + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                color="#3B82F6",
                width=3,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(
                color="#EF4444",
                width=3,
            ),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        hovermode="x unified",
        xaxis=dict(
            title="15 Minute Block",
            dtick=4,
        ),
        yaxis=dict(
            title="Power (MW)",
        ),
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
        ),
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
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df,
    df_fix,
    cluster=False,
    cluster_weights=None,
):

    if cluster:

        ghi_columns = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

        weight_columns = [
            "CL-1",
            "CL-2",
            "CL-3",
            "CL-4",
            "CL-5",
        ]

        forecast = np.zeros(
            len(df_fix),
            dtype=float,
        )

        for ghi_col, weight_col in zip(
            ghi_columns,
            weight_columns,
        ):

            poa = (
                df_fix[ghi_col]
                * df_fix["SIN(a+b)"]
                / df_fix["Sin(a)"]
            )

            eff_area = (
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * cluster_weights[weight_col]
            ).sum()

            forecast += (
                poa.to_numpy()
                * eff_area
                / 1_000_000
            )

        return forecast

    df_fix["POA fixed"] = (
        df_fix["GHI_Forecast"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    return (
        df_fix["POA fixed"].to_numpy()
        * df["Eff Area"].sum()
        / 1_000_000
    )


# ============================================================
# TRACKING PARAMETER UI
# ============================================================

def tracking_parameter_editor(
    params,
    prefix,
):

    st.markdown(
        "### ⚙️ Tracking Parameters"
    )

    st.caption(
        "These values are automatically calculated by the optimizer. "
        "After calculation, you can manually adjust them."
    )

    with st.form(
        f"{prefix}_tracking_form"
    ):

        c1, c2, c3 = st.columns(3)

        DHI = c1.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            value=int(params["DHI"]),
            step=1,
        )

        start = c2.number_input(
            "Starting Block",
            min_value=0,
            max_value=30,
            value=int(params["start"]),
            step=1,
        )

        end = c3.number_input(
            "Ending Block",
            min_value=65,
            max_value=80,
            value=int(params["end"]),
            step=1,
        )

        c1, c2, c3 = st.columns(3)

        max_block = c1.number_input(
            "Max Block",
            min_value=44,
            max_value=60,
            value=int(params["max"]),
            step=1,
        )

        east = c2.number_input(
            "East Limit",
            min_value=0,
            max_value=70,
            value=int(params["east"]),
            step=1,
        )

        west = c3.number_input(
            "West Limit",
            min_value=0,
            max_value=70,
            value=int(params["west"]),
            step=1,
        )

        apply = st.form_submit_button(
            "🔄 Apply Tracking Parameter Changes",
            use_container_width=True,
        )

    final_params = {
        "DHI": int(DHI),
        "start": int(start),
        "end": int(end),
        "max": int(max_block),
        "east": int(east),
        "west": int(west),
    }

    return final_params, apply


# ============================================================
# EFFICIENCY LOSS EDITOR
# ============================================================

def efficiency_loss_editor(
    auto_loss,
    df,
    key_prefix,
):

    min_loss = 0.0

    max_loss = float(
        df[
            "Standard PV Efficiency (%)"
        ].min()
    )

    if key_prefix not in st.session_state:

        st.session_state[key_prefix] = (
            float(auto_loss)
        )

    st.markdown(
        "### 📉 Efficiency Loss"
    )

    st.caption(
        "The optimizer automatically calculates the initial loss. "
        "You can manually change it after calculation."
    )

    with st.form(
        f"{key_prefix}_form"
    ):

        loss = st.number_input(
            "Efficiency Loss (%)",
            min_value=min_loss,
            max_value=max_loss,
            value=float(
                st.session_state[key_prefix]
            ),
            step=0.10,
            format="%.2f",
        )

        apply = st.form_submit_button(
            "🔄 Apply Efficiency Loss",
            use_container_width=True,
        )

    if apply:

        st.session_state[key_prefix] = (
            float(loss)
        )

    return float(
        st.session_state[key_prefix]
    )


# ============================================================
# NON CLUSTER FIXED
# ============================================================

def run_noncluster_fixed(
    df,
    input_df,
    lat,
    tilt_lookup,
):

    df_fix = input_df.copy()

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=False,
    )

    df_fix["POA fixed"] = (
        df_fix["GHI_Forecast"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    auto_loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    loss = efficiency_loss_editor(
        auto_loss,
        df,
        "noncluster_fixed_loss",
    )

    df_final = apply_efficiency_loss(
        df,
        loss,
    )

    forecast = (
        df_fix["POA fixed"].to_numpy()
        * df_final["Eff Area"].sum()
        / 1_000_000
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Auto Calculated Loss",
        f"{auto_loss:.2f}%",
    )

    c2.metric(
        "Current Loss Used",
        f"{loss:.2f}%",
    )

    show_efficiency_table(
        df_final
    )

    show_forecast_chart(
        forecast,
        df_fix["Actual"].to_numpy(),
        "🏗️ Fixed Forecast vs Actual",
    )


# ============================================================
# NON CLUSTER TRACKING
# ============================================================

def run_noncluster_tracking(
    uploaded_file,
    df,
    input_df,
    lat,
    tilt_lookup,
):

    df_fix = input_df.copy()

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=True,
    )

    df_fix["POA fixed"] = (
        df_fix["GHI_Forecast"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    auto_loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    loss = efficiency_loss_editor(
        auto_loss,
        df,
        "noncluster_tracking_loss",
    )

    df_final = apply_efficiency_loss(
        df,
        loss,
    )

    weighted_ghi = (
        df_fix["GHI_Forecast"].to_numpy(
            dtype=float
        )
        * df_final["Eff Area"].sum()
    )

    uploaded_file.seek(0)

    df_backend = pd.read_excel(
        uploaded_file,
        sheet_name="Backend Cal",
    )

    df_backend = clean_columns(
        df_backend
    )

    validate_columns(
        df_backend,
        ["Block No."],
        "Backend Cal",
    )

    blocks = df_backend[
        "Block No."
    ].to_numpy(dtype=float)

    actual = df_fix[
        "Actual"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    if st.session_state.optimization_result is None:

        with st.spinner(
            "⚙️ Running tracking optimization... "
            "Please wait."
        ):

            result = optimize_tracking(
                blocks,
                weighted_ghi,
                actual,
            )

        st.session_state.optimization_result = (
            result
        )

        st.success(
            "✅ Tracking optimization completed."
        )

    params = (
        st.session_state.optimization_result
    )

    st.metric(
        "Optimization Score",
        f"{params['score']:.5f}",
    )

    final_params, changed = (
        tracking_parameter_editor(
            params,
            "noncluster",
        )
    )

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            final_params,
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Auto Efficiency Loss",
            f"{auto_loss:.2f}%",
        )

        c2.metric(
            "Current Efficiency Loss",
            f"{loss:.2f}%",
        )

        show_efficiency_table(
            df_final
        )

        show_forecast_chart(
            forecast,
            actual,
            "🔄 Tracking Forecast vs Actual",
        )

    except Exception as e:

        st.error(
            f"Unable to calculate tracking forecast: {e}"
        )


# ============================================================
# CLUSTER FIXED
# ============================================================

def run_cluster_fixed(
    uploaded_file,
    df,
    input_df,
    lat,
    tilt_lookup,
):

    cluster_weights = read_cluster_weights(
        uploaded_file
    )

    df_fix = input_df.copy()

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=False,
    )

    df_fix["POA fixed"] = (
        df_fix["CL1-GHI"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    auto_loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    loss = efficiency_loss_editor(
        auto_loss,
        df,
        "cluster_fixed_loss",
    )

    df_final = apply_efficiency_loss(
        df,
        loss,
    )

    forecast = calculate_fixed_forecast(
        df_final,
        df_fix,
        cluster=True,
        cluster_weights=cluster_weights,
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Auto Calculated Loss",
        f"{auto_loss:.2f}%",
    )

    c2.metric(
        "Current Loss Used",
        f"{loss:.2f}%",
    )

    show_efficiency_table(
        df_final
    )

    show_forecast_chart(
        forecast,
        df_fix["Actual"].to_numpy(),
        "🏗️ Fixed Cluster Forecast vs Actual",
    )


# ============================================================
# CLUSTER TRACKING
# ============================================================

def run_cluster_tracking(
    uploaded_file,
    df,
    input_df,
    lat,
    tilt_lookup,
):

    cluster_weights = read_cluster_weights(
        uploaded_file
    )

    df_fix = input_df.copy()

    df_fix = prepare_solar_angles(
        df_fix,
        lat,
        tilt_lookup,
        tracking=True,
    )

    df_fix["POA fixed"] = (
        df_fix["CL1-GHI"]
        * df_fix["SIN(a+b)"]
        / df_fix["Sin(a)"]
    )

    auto_loss = calculate_efficiency_loss(
        df,
        df_fix["POA fixed"],
        df_fix["Actual"],
    )

    loss = efficiency_loss_editor(
        auto_loss,
        df,
        "cluster_tracking_loss",
    )

    df_final = apply_efficiency_loss(
        df,
        loss,
    )

    # --------------------------------------------------------
    # BACKEND
    # --------------------------------------------------------

    backend_sheets = [
        "Backend Cal CL1",
        "Backend Cal CL2",
        "Backend Cal CL3",
        "Backend Cal CL4",
        "Backend Cal CL5",
    ]

    backend_list = []

    for sheet in backend_sheets:

        uploaded_file.seek(0)

        temp = pd.read_excel(
            uploaded_file,
            sheet_name=sheet,
        )

        temp = clean_columns(
            temp
        )

        backend_list.append(
            temp
        )

    validate_columns(
        backend_list[0],
        ["Block No."],
        backend_sheets[0],
    )

    blocks = backend_list[0][
        "Block No."
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # WEIGHTED GHI
    # --------------------------------------------------------

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

    weight_columns = [
        "CL-1",
        "CL-2",
        "CL-3",
        "CL-4",
        "CL-5",
    ]

    weighted_ghi = np.zeros(
        len(df_fix),
        dtype=float,
    )

    for ghi_col, weight_col in zip(
        ghi_columns,
        weight_columns,
    ):

        ghi = df_fix[
            ghi_col
        ].to_numpy(
            dtype=float
        )

        eff_area = (
            df_final["Total area(m2)"]
            * df_final["Net Efficiency (%)"]
            / 100
            * cluster_weights[weight_col]
        ).sum()

        weighted_ghi += (
            ghi
            * eff_area
        )

    actual = df_fix[
        "Actual"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    if st.session_state.optimization_result is None:

        with st.spinner(
            "⚙️ Running tracking optimization... "
            "Please wait."
        ):

            result = optimize_tracking(
                blocks,
                weighted_ghi,
                actual,
            )

        st.session_state.optimization_result = (
            result
        )

        st.success(
            "✅ Tracking optimization completed."
        )

    params = (
        st.session_state.optimization_result
    )

    st.metric(
        "Optimization Score",
        f"{params['score']:.5f}",
    )

    final_params, changed = (
        tracking_parameter_editor(
            params,
            "cluster",
        )
    )

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            final_params,
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Auto Efficiency Loss",
            f"{auto_loss:.2f}%",
        )

        c2.metric(
            "Current Efficiency Loss",
            f"{loss:.2f}%",
        )

        show_efficiency_table(
            df_final
        )

        show_forecast_chart(
            forecast,
            actual,
            "🔄 Tracking Cluster Forecast vs Actual",
        )

    except Exception as e:

        st.error(
            f"Unable to calculate tracking forecast: {e}"
        )


# ============================================================
# RUN LOSS CORRECTION BUTTON
# ============================================================

def run_loss_correction_button():

    st.markdown(
        """
        <div class="run-card">
            <b>🚀 Ready to run Loss Correction</b><br>
            <span style="opacity:0.75;">
            Input data and plant type are selected.
            Click the button below to start the calculation.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🚀 RUN LOSS CORRECTION",
        key="run_loss_correction",
        use_container_width=True,
        type="primary",
    ):

        st.session_state.run_requested = True

        # Every new run should calculate optimization again
        st.session_state.optimization_result = None
        st.session_state.calculation_result = None

        st.rerun()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">☀️ Loss Correction Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Upload the plant workbook, edit GHI Forecast / Actual data, "
        "select the plant type, then run Loss Correction."
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # FILE
    # ========================================================

    st.markdown(
        '<div class="section-title">📁 Input Workbook</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx", "xls"],
        help=(
            "The workbook should contain the required plant "
            "configuration sheets."
        ),
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload the Excel workbook to begin."
        )

        return

    # ========================================================
    # FILE CONTEXT
    # ========================================================

    current_file_signature = (
        file_signature(uploaded_file)
    )

    if (
        st.session_state.input_signature
        != current_file_signature
    ):

        st.session_state.input_signature = (
            current_file_signature
        )

        st.session_state.input_data = None
        st.session_state.optimization_result = None
        st.session_state.calculation_result = None
        st.session_state.run_requested = False

    # ========================================================
    # DETECT WORKBOOK
    # ========================================================

    try:

        sheets = get_sheet_names(
            uploaded_file
        )

        is_cluster = (
            "Fixed" not in sheets
        )

    except Exception as e:

        st.error(
            f"Unable to read workbook: {e}"
        )

        return

    if is_cluster:

        st.info(
            "🏢 **Cluster workbook detected**"
        )

    else:

        st.success(
            "🏭 **Non-cluster workbook detected**"
        )

    # ========================================================
    # COMMON CONFIGURATION
    # ========================================================

    try:

        df = read_area_efficiency(
            uploaded_file,
            cluster=is_cluster,
        )

        lat = read_latitude(
            uploaded_file
        )

        tilt_lookup = read_tilt_lookup(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to load workbook configuration: {e}"
        )

        return

    # ========================================================
    # INPUT DATA
    # ========================================================

    try:

        input_df = input_data_editor(
            uploaded_file,
            is_cluster,
        )

    except Exception as e:

        st.error(
            f"Unable to load input data: {e}"
        )

        return

    # ========================================================
    # PLANT TYPE
    # ========================================================

    plant_type = plant_type_selector()

    # ========================================================
    # RUN BUTTON
    # ========================================================

    run_loss_correction_button()

    # ========================================================
    # DON'T CALCULATE UNTIL BUTTON IS PRESSED
    # ========================================================

    if not st.session_state.run_requested:

        st.info(
            "👆 Review/edit the input data, select Fixed or Tracking, "
            "then click **RUN LOSS CORRECTION**."
        )

        return

    # ========================================================
    # CALCULATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Loss Correction Result</div>',
        unsafe_allow_html=True,
    )

    try:

        # ----------------------------------------------------
        # NON CLUSTER
        # ----------------------------------------------------

        if not is_cluster:

            if plant_type == "🏗️ Fixed":

                run_noncluster_fixed(
                    df,
                    input_df,
                    lat,
                    tilt_lookup,
                )

            else:

                run_noncluster_tracking(
                    uploaded_file,
                    df,
                    input_df,
                    lat,
                    tilt_lookup,
                )

        # ----------------------------------------------------
        # CLUSTER
        # ----------------------------------------------------

        else:

            if plant_type == "🏗️ Fixed":

                run_cluster_fixed(
                    uploaded_file,
                    df,
                    input_df,
                    lat,
                    tilt_lookup,
                )

            else:

                run_cluster_tracking(
                    uploaded_file,
                    df,
                    input_df,
                    lat,
                    tilt_lookup,
                )

    except Exception as e:

        st.error(
            "❌ Loss correction calculation failed."
        )

        st.exception(e)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
