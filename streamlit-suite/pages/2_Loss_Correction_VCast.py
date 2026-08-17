# ============================================================
# STREAMLIT APP
# SOLAR LOSS CORRECTION MODEL
# FIXED / TRACKING
#
# IMPORTANT TRACKING FIX:
# ------------------------------------------------------------
# Efficiency Loss is applied ONCE through Effective Area.
#
# DHI is a separate tracking parameter.
#
# The optimizer and final tracking forecast use the EXACT
# SAME tracking calculation function.
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Loss Correction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_OPT_ITER = 40
OPT_POPSIZE = 10

TRACKING_BOUNDS = [
    (0, 10),     # DHI
    (0, 30),     # Starting Block
    (65, 80),    # Ending Block
    (44, 60),    # Max Block
    (0, 70),     # East Limit
    (0, 70),     # West Limit
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 1.20rem;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 10px;
    }

    .metric-card {
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 14px;
        padding: 15px 18px;
        background: rgba(128,128,128,0.04);
    }

    .metric-label {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 750;
    }

    .run-box {
        border-radius: 14px;
        padding: 16px;
        background: rgba(37,99,235,0.07);
        border: 1px solid rgba(37,99,235,0.20);
        margin: 15px 0;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 46px;
        font-weight: 650;
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
    "last_file_name": None,
    "last_plant_type": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GENERAL HELPERS
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


def clean_data_rows(df, date_column="Date"):

    df = df.copy()

    if date_column in df.columns:

        idx = df[
            df[date_column].isna()
        ].index

        if len(idx):

            pos = df.index.get_loc(
                idx[0]
            )

            df = df.iloc[:pos]

    return df.reset_index(drop=True)


def get_sheet_names(uploaded_file):

    uploaded_file.seek(0)

    return pd.ExcelFile(
        uploaded_file
    ).sheet_names


# ============================================================
# WORKBOOK TYPE
# ============================================================

def detect_cluster(uploaded_file):

    sheets = get_sheet_names(
        uploaded_file
    )

    return "Fixed" not in sheets


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def read_area_efficiency(
    uploaded_file,
    cluster=False,
):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(8) if cluster else None,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

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

        idx = df[
            df["Module Type"].isna()
        ].index

        if len(idx):

            pos = df.index.get_loc(
                idx[0]
            )

            df = df.iloc[:pos]

    df = df.dropna(
        subset=[
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        how="all",
    )

    df[
        "Standard PV Efficiency (%)"
    ] = pd.to_numeric(
        df[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    )

    df[
        "Total area(m2)"
    ] = pd.to_numeric(
        df[
            "Total area(m2)"
        ],
        errors="coerce",
    )

    return df.reset_index(drop=True)


# ============================================================
# CLUSTER WEIGHTS
# ============================================================

def read_cluster_weights(
    uploaded_file,
):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[12, 13, 14, 15, 16],
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    cols = [
        "CL-1",
        "CL-2",
        "CL-3",
        "CL-4",
        "CL-5",
    ]

    validate_columns(
        df,
        cols,
        "Cluster Weights",
    )

    return {
        c: float(
            pd.to_numeric(
                df[c].iloc[0],
                errors="coerce",
            )
        )
        for c in cols
    }


# ============================================================
# LATITUDE
# ============================================================

def read_latitude(
    uploaded_file,
):

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

def read_tilt_lookup(
    uploaded_file,
):

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

        if "Fixed" not in df.columns:
            return {}

        idx = df[
            df["Fixed"].isna()
        ].index

        if len(idx):

            pos = df.index.get_loc(
                idx[0]
            )

            df = df.iloc[:pos]

        df = df.dropna(
            axis=1,
            how="all",
        )

        df = df.rename(
            columns={
                "Unnamed: 2": "Month_Num",
                "Unnamed: 3": "Month",
            }
        )

        if "Month" not in df.columns:
            return {}

        return (
            df
            .dropna(
                subset=["Month"]
            )
            .set_index("Month")["Fixed"]
            .to_dict()
        )

    except Exception:

        return {}


# ============================================================
# INPUT DATA
# ============================================================

def load_input_data(
    uploaded_file,
    cluster,
):

    uploaded_file.seek(0)

    if cluster:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Fixed-CL1",
            header=1,
        )

    else:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="Fixed",
            header=1,
        )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    df = clean_data_rows(df)

    validate_columns(
        df,
        ["Actual"],
        "Forecast Sheet",
    )

    if cluster:

        ghi_cols = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

        uploaded_file.seek(0)

        try:

            result = pd.read_excel(
                uploaded_file,
                sheet_name="Result",
                usecols=range(6),
            ).fillna(0)

            for i, col in enumerate(
                ghi_cols
            ):

                if (
                    col not in df.columns
                    and i < len(result.columns)
                ):

                    values = result.iloc[
                        :len(df),
                        i,
                    ].to_numpy()

                    if len(values) < len(df):

                        values = np.pad(
                            values,
                            (
                                0,
                                len(df)
                                - len(values),
                            ),
                            constant_values=0,
                        )

                    df[col] = values

        except Exception:
            pass

        validate_columns(
            df,
            ghi_cols,
            "Cluster Forecast",
        )

    else:

        validate_columns(
            df,
            ["GHI_Forecast"],
            "Fixed Forecast",
        )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    if not cluster:

        df["GHI_Forecast"] = pd.to_numeric(
            df["GHI_Forecast"],
            errors="coerce",
        ).fillna(0)

    else:

        for col in [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

    return df


# ============================================================
# INPUT DATA EDITOR
# ============================================================

def input_data_editor(
    df,
    cluster,
):

    st.markdown(
        '<div class="section-title">'
        '📊 Input GHI Forecast & Actual Power'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Edit the GHI forecast and Actual power directly below. "
        "All plant configuration parameters continue to come "
        "from the uploaded workbook."
    )

    if cluster:

        edit_cols = [
            "Actual",
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

    else:

        edit_cols = [
            "GHI_Forecast",
            "Actual",
        ]

    edit_cols = [
        c for c in edit_cols
        if c in df.columns
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
# SOLAR ANGLES
# ============================================================

def prepare_solar_angles(
    df,
    lat,
    tilt_lookup=None,
    tracking=False,
):

    result = df.copy()

    today = pd.Timestamp.today().normalize()

    result["Date"] = today

    first_date = today.replace(
        month=1,
        day=1,
    )

    day_number = (
        result["Date"]
        - first_date
    ).dt.days + 1

    result[
        "Declination Angle ∆"
    ] = (
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

    result[
        "Elevation angle a"
    ] = (
        90
        - lat
        + result[
            "Declination Angle ∆"
        ]
    )

    if tracking:

        result["Tilt Angle b"] = 0

    else:

        if tilt_lookup:

            result[
                "Tilt Angle b"
            ] = (
                result["Date"]
                .dt.strftime("%B")
                .map(tilt_lookup)
                .fillna(0)
            )

        else:

            result["Tilt Angle b"] = 0

    result["a+b"] = (
        result["Elevation angle a"]
        + result["Tilt Angle b"]
    )

    result["SIN(a+b)"] = np.sin(
        np.radians(
            result["a+b"]
        )
    )

    result["Sin(a)"] = np.sin(
        np.radians(
            result["Elevation angle a"]
        )
    ).clip(
        lower=1e-6
    )

    return result


# ============================================================
# EFFICIENCY LOSS
# ============================================================

def calculate_efficiency_loss(
    df,
    poa,
    actual,
):

    standard = df[
        "Standard PV Efficiency (%)"
    ].to_numpy(float)

    area = df[
        "Total area(m2)"
    ].to_numpy(float)

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

    if (
        len(valid_actual) == 0
        or len(valid_poa) == 0
    ):
        return 0.0

    poa_peak = np.nanmax(
        valid_poa
    )

    actual_peak = np.nanmax(
        valid_actual
    )

    if poa_peak <= 0:
        return 0.0

    if actual_peak <= 0:
        return 0.0

    base_area = np.sum(
        area
        * standard
        / 100
    )

    loss_coeff = np.sum(
        area / 100
    )

    if loss_coeff <= 0:
        return 0.0

    target_area = (
        actual_peak
        * 1_000_000
        / poa_peak
    )

    loss = (
        base_area
        - target_area
    ) / loss_coeff

    return float(
        np.clip(
            loss,
            0,
            np.nanmin(standard),
        )
    )


def apply_efficiency_loss(
    df,
    loss,
):

    result = df.copy()

    result[
        "Efficiency Losses(%)"
    ] = loss

    result[
        "Net Efficiency (%)"
    ] = (
        result[
            "Standard PV Efficiency (%)"
        ]
        - loss
    )

    result["Eff Area"] = (
        result["Total area(m2)"]
        * result["Net Efficiency (%)"]
        / 100
    )

    return result


# ============================================================
# EFFICIENCY UI
# ============================================================

def efficiency_control(
    df,
    auto_loss,
    key,
):

    st.markdown(
        '<div class="section-title">'
        '📉 Efficiency Loss'
        '</div>',
        unsafe_allow_html=True,
    )

    max_loss = float(
        df[
            "Standard PV Efficiency (%)"
        ].min()
    )

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=max_loss,
        value=float(
            np.clip(
                auto_loss,
                0,
                max_loss,
            )
        ),
        step=0.1,
        format="%.2f",
        key=key,
    )

    return apply_efficiency_loss(
        df,
        loss,
    )


# ============================================================
# EFFICIENCY TABLE
# ============================================================

def show_efficiency_table(
    df,
):

    cols = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)",
        "Eff Area",
    ]

    cols = [
        c for c in cols
        if c in df.columns
    ]

    display = df[
        cols
    ].copy()

    nums = display.select_dtypes(
        include="number"
    ).columns

    display[nums] = (
        display[nums]
        .round(3)
    )

    with st.expander(
        "🔍 View Efficiency Calculations"
    ):

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FIXED FORECAST
# ============================================================

def fixed_forecast(
    df,
    input_df,
    lat,
    tilt_lookup,
    cluster=False,
    weights=None,
):

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=False,
    )

    if cluster:

        ghi_cols = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

        weight_cols = [
            "CL-1",
            "CL-2",
            "CL-3",
            "CL-4",
            "CL-5",
        ]

        forecast = np.zeros(
            len(input_df),
            dtype=float,
        )

        for ghi_col, weight_col in zip(
            ghi_cols,
            weight_cols,
        ):

            poa = (
                solar[ghi_col]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            eff_area = (
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * weights[weight_col]
            ).sum()

            forecast += (
                poa.to_numpy()
                * eff_area
                / 1_000_000
            )

        return forecast, solar

    poa = (
        solar["GHI_Forecast"]
        * solar["SIN(a+b)"]
        / solar["Sin(a)"]
    )

    forecast = (
        poa.to_numpy()
        * df["Eff Area"].sum()
        / 1_000_000
    )

    return forecast, solar


# ============================================================
# TRACKING CORE CALCULATION
#
# THIS IS NOW THE SINGLE SOURCE OF TRUTH.
# Optimizer and final forecast both call this function.
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

    DHI = float(
        params["DHI"]
    )

    start = int(
        params["start"]
    )

    end = int(
        params["end"]
    )

    max_block = int(
        params["max"]
    )

    east = int(
        params["east"]
    )

    west = int(
        params["west"]
    )

    if not (
        start
        < max_block
        < end
    ):

        raise ValueError(
            "Starting Block < Max Block < Ending Block is required."
        )

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

        raise ValueError(
            "Invalid tracking block configuration."
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
                & (zenith > west)
            ),

            west,

            zenith,
        ),
    )

    cos_alpha = np.clip(
        np.cos(
            np.radians(
                panel
            )
        ),
        1e-6,
        None,
    )

    # --------------------------------------------------------
    # DHI IS APPLIED ONLY HERE.
    #
    # Efficiency Loss is NOT applied here again.
    #
    # weighted_ghi already contains the single efficiency
    # correction through Effective Area.
    # --------------------------------------------------------

    forecast = (
        weighted_ghi
        * (1 - DHI / 100)
        / cos_alpha
        / 1_000_000
    )

    return forecast


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_tracking_cached(
    blocks_tuple,
    weighted_ghi_tuple,
    actual_tuple,
):

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    weighted_ghi = np.asarray(
        weighted_ghi_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    n = min(
        len(blocks),
        len(weighted_ghi),
        len(actual),
    )

    blocks = blocks[:n]
    weighted_ghi = weighted_ghi[:n]
    actual = actual[:n]

    mask = (
        np.isfinite(actual)
        & np.isfinite(weighted_ghi)
        & (actual != 0)
    )

    blocks = blocks[mask]
    weighted_ghi = weighted_ghi[mask]
    actual = actual[mask]

    if len(actual) == 0:

        raise ValueError(
            "No valid Actual power values found."
        )

    actual_peak = np.max(
        actual
    )

    actual_energy = np.sum(
        actual
    )

    if (
        actual_peak <= 0
        or actual_energy <= 0
    ):

        raise ValueError(
            "Actual power data is invalid."
        )

    def objective(x):

        params = {
            "DHI": int(
                round(x[0])
            ),
            "start": int(
                round(x[1])
            ),
            "end": int(
                round(x[2])
            ),
            "max": int(
                round(x[3])
            ),
            "east": int(
                round(x[4])
            ),
            "west": int(
                round(x[5])
            ),
        }

        if not (
            params["start"]
            < params["max"]
            < params["end"]
        ):

            return 1e9

        try:

            prediction = (
                calculate_tracking_forecast(
                    blocks,
                    weighted_ghi,
                    params,
                )
            )

        except Exception:

            return 1e9

        if not np.all(
            np.isfinite(
                prediction
            )
        ):

            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual
                    - prediction
                )
            )
            / actual_peak
        )

        peak_error = (
            abs(
                actual_peak
                - np.max(
                    prediction
                )
            )
            / actual_peak
        )

        energy_error = (
            abs(
                actual_energy
                - np.sum(
                    prediction
                )
            )
            / actual_energy
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(
        objective,
        bounds=TRACKING_BOUNDS,
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
        '<div class="section-title">'
        '⚙️ Tracking Parameters'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Optimizer values are loaded automatically. "
        "All values remain editable."
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
        "GHI Starting Block",
        min_value=0,
        max_value=30,
        value=int(params["start"]),
        step=1,
        key=f"{prefix}_start",
    )

    end = c3.number_input(
        "GHI Ending Block",
        min_value=65,
        max_value=80,
        value=int(params["end"]),
        step=1,
        key=f"{prefix}_end",
    )

    c1, c2, c3 = st.columns(3)

    max_block = c1.number_input(
        "GHI Max Block",
        min_value=44,
        max_value=60,
        value=int(params["max"]),
        step=1,
        key=f"{prefix}_max",
    )

    east = c2.number_input(
        "Tracking East Limit",
        min_value=0,
        max_value=70,
        value=int(params["east"]),
        step=1,
        key=f"{prefix}_east",
    )

    west = c3.number_input(
        "Tracking West Limit",
        min_value=0,
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
# METRICS
# ============================================================

def calculate_metrics(
    forecast,
    actual,
):

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

    valid = (
        np.isfinite(
            forecast
        )
        & np.isfinite(
            actual
        )
    )

    forecast = forecast[valid]
    actual = actual[valid]

    if len(actual) == 0:

        return {
            "peak_error": np.nan,
            "energy_error": np.nan,
            "mae": np.nan,
        }

    actual_peak = np.max(
        actual
    )

    forecast_peak = np.max(
        forecast
    )

    actual_energy = np.sum(
        actual
    )

    forecast_energy = np.sum(
        forecast
    )

    peak_error = (
        abs(
            forecast_peak
            - actual_peak
        )
        / actual_peak
        * 100
        if actual_peak > 0
        else np.nan
    )

    energy_error = (
        abs(
            forecast_energy
            - actual_energy
        )
        / actual_energy
        * 100
        if actual_energy > 0
        else np.nan
    )

    mae = np.mean(
        np.abs(
            forecast
            - actual
        )
    )

    return {
        "peak_error": peak_error,
        "energy_error": energy_error,
        "mae": mae,
    }


def show_metrics(
    forecast,
    actual,
):

    metrics = calculate_metrics(
        forecast,
        actual,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Peak Error",
        f"{metrics['peak_error']:.2f}%",
    )

    c2.metric(
        "Energy Error",
        f"{metrics['energy_error']:.2f}%",
    )

    c3.metric(
        "MAE",
        f"{metrics['mae']:.3f}",
    )


# ============================================================
# FORECAST GRAPH
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
                actual[:n]
            ),
            mode="lines",
            name="Actual",
            line=dict(
                width=2.5,
                color="#EF4444",
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(
                forecast[:n]
            ),
            mode="lines",
            name="Forecast",
            line=dict(
                width=2.5,
                color="#2563EB",
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=500,
        hovermode="x unified",
        template="plotly_white",
        xaxis_title="15 Minute Block",
        yaxis_title="Power",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=30,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(
        rangeslider_visible=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# PLANT SELECTOR
# ============================================================

def plant_selector():

    st.markdown(
        '<div class="section-title">'
        '🏭 Plant Type'
        '</div>',
        unsafe_allow_html=True,
    )

    plant_type = st.segmented_control(
        "Plant Type",
        options=[
            "🏗️ Fixed",
            "🔄 Tracking",
        ],
        default=st.session_state.plant_type,
        selection_mode="single",
        key="plant_type_selector",
        label_visibility="collapsed",
        width="stretch",
    )

    if plant_type is None:

        plant_type = "🏗️ Fixed"

    return plant_type


# ============================================================
# NON-CLUSTER FIXED
# ============================================================

def run_noncluster_fixed(
    df,
    input_df,
    lat,
    tilt_lookup,
):

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=False,
    )

    poa = (
        solar["GHI_Forecast"]
        * solar["SIN(a+b)"]
        / solar["Sin(a)"]
    )

    auto_loss = calculate_efficiency_loss(
        df,
        poa,
        input_df["Actual"],
    )

    df = efficiency_control(
        df,
        auto_loss,
        "noncluster_fixed_loss",
    )

    forecast = (
        poa.to_numpy()
        * df["Eff Area"].sum()
        / 1_000_000
    )

    show_metrics(
        forecast,
        input_df["Actual"],
    )

    show_efficiency_table(
        df
    )

    show_forecast_chart(
        forecast,
        input_df["Actual"],
        "🏗️ Fixed Forecast vs Actual",
    )


# ============================================================
# NON-CLUSTER TRACKING
# ============================================================

def run_noncluster_tracking(
    uploaded_file,
    df,
    input_df,
    lat,
    tilt_lookup,
):

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=True,
    )

    poa = (
        solar["GHI_Forecast"]
        * solar["SIN(a+b)"]
        / solar["Sin(a)"]
    )

    auto_loss = calculate_efficiency_loss(
        df,
        poa,
        input_df["Actual"],
    )

    # --------------------------------------------------------
    # Efficiency Loss applied ONCE here.
    # --------------------------------------------------------

    df = efficiency_control(
        df,
        auto_loss,
        "noncluster_tracking_loss",
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # weighted_ghi contains EFFECTIVE AREA.
    #
    # Therefore Efficiency Loss has already been included.
    # Do NOT multiply Efficiency Loss again later.
    # --------------------------------------------------------

    weighted_ghi = (
        input_df[
            "GHI_Forecast"
        ].to_numpy(float)
        * df["Eff Area"].sum()
    )

    uploaded_file.seek(0)

    backend = pd.read_excel(
        uploaded_file,
        sheet_name="Backend Cal",
    )

    backend.columns = (
        backend.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        backend,
        ["Block No."],
        "Backend Cal",
    )

    blocks = backend[
        "Block No."
    ].to_numpy(float)

    actual = input_df[
        "Actual"
    ].to_numpy(float)

    # --------------------------------------------------------
    # OPTIMIZE
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
                tuple(weighted_ghi),
                tuple(actual),
            )

        st.session_state.tracking_params = result

    params = tracking_parameter_controls(
        st.session_state.tracking_params,
        "noncluster",
    )

    # --------------------------------------------------------
    # FINAL FORECAST
    # --------------------------------------------------------

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            params,
        )

        show_metrics(
            forecast,
            actual,
        )

        show_efficiency_table(
            df
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

    weights = read_cluster_weights(
        uploaded_file
    )

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=False,
    )

    poa = (
        solar["CL1-GHI"]
        * solar["SIN(a+b)"]
        / solar["Sin(a)"]
    )

    auto_loss = calculate_efficiency_loss(
        df,
        poa,
        input_df["Actual"],
    )

    df = efficiency_control(
        df,
        auto_loss,
        "cluster_fixed_loss",
    )

    forecast, _ = fixed_forecast(
        df,
        input_df,
        lat,
        tilt_lookup,
        cluster=True,
        weights=weights,
    )

    show_metrics(
        forecast,
        input_df["Actual"],
    )

    show_efficiency_table(
        df
    )

    show_forecast_chart(
        forecast,
        input_df["Actual"],
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

    weights = read_cluster_weights(
        uploaded_file
    )

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=True,
    )

    ghi_cols = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI",
    ]

    weight_cols = [
        "CL-1",
        "CL-2",
        "CL-3",
        "CL-4",
        "CL-5",
    ]

    poa = (
        solar["CL1-GHI"]
        * solar["SIN(a+b)"]
        / solar["Sin(a)"]
    )

    auto_loss = calculate_efficiency_loss(
        df,
        poa,
        input_df["Actual"],
    )

    # --------------------------------------------------------
    # Efficiency Loss applied ONCE.
    # --------------------------------------------------------

    df = efficiency_control(
        df,
        auto_loss,
        "cluster_tracking_loss",
    )

    # --------------------------------------------------------
    # WEIGHTED GHI
    # --------------------------------------------------------

    weighted_ghi = np.zeros(
        len(input_df),
        dtype=float,
    )

    for ghi_col, weight_col in zip(
        ghi_cols,
        weight_cols,
    ):

        eff_area = (
            df["Total area(m2)"]
            * df["Net Efficiency (%)"]
            / 100
            * weights[weight_col]
        ).sum()

        weighted_ghi += (
            input_df[
                ghi_col
            ].to_numpy(float)
            * eff_area
        )

    # --------------------------------------------------------
    # BACKEND BLOCKS
    # --------------------------------------------------------

    uploaded_file.seek(0)

    backend = pd.read_excel(
        uploaded_file,
        sheet_name="Backend Cal CL1",
    )

    backend.columns = (
        backend.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        backend,
        ["Block No."],
        "Backend Cal CL1",
    )

    blocks = backend[
        "Block No."
    ].to_numpy(float)

    actual = input_df[
        "Actual"
    ].to_numpy(float)

    # --------------------------------------------------------
    # OPTIMIZATION
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
                tuple(weighted_ghi),
                tuple(actual),
            )

        st.session_state.tracking_params = result

    params = tracking_parameter_controls(
        st.session_state.tracking_params,
        "cluster",
    )

    # --------------------------------------------------------
    # FINAL FORECAST
    # --------------------------------------------------------

    try:

        forecast = calculate_tracking_forecast(
            blocks,
            weighted_ghi,
            params,
        )

        show_metrics(
            forecast,
            actual,
        )

        show_efficiency_table(
            df
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
# MAIN
# ============================================================

def main():

    st.markdown(
        '<div class="main-title">'
        '☀️ Solar Loss Correction Model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload your plant workbook, edit GHI Forecast and '
        'Actual power, select Fixed or Tracking, and run '
        'the correction model.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📁 Input Workbook'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=[
            "xlsx",
            "xls",
        ],
        label_visibility="collapsed",
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload the Excel plant workbook to begin."
        )

        return

    # ========================================================
    # RESET STATE WHEN NEW FILE IS UPLOADED
    # ========================================================

    file_name = uploaded_file.name

    if (
        st.session_state.last_file_name
        != file_name
    ):

        st.session_state.tracking_params = None
        st.session_state.run_model = False
        st.session_state.input_df = None
        st.session_state.last_file_name = file_name

        if "input_editor" in st.session_state:
            del st.session_state[
                "input_editor"
            ]

    # ========================================================
    # DETECT WORKBOOK
    # ========================================================

    try:

        is_cluster = detect_cluster(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to detect workbook type: {e}"
        )

        return

    if is_cluster:

        st.success(
            "📦 Cluster workbook detected"
        )

    else:

        st.success(
            "🏭 Fixed plant workbook detected"
        )

    # ========================================================
    # LOAD CONFIGURATION
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

        if st.session_state.input_df is None:

            input_df = load_input_data(
                uploaded_file,
                is_cluster,
            )

            st.session_state.input_df = input_df

        else:

            input_df = (
                st.session_state.input_df
            )

    except Exception as e:

        st.error(
            f"Unable to load workbook: {e}"
        )

        return

    # ========================================================
    # INPUT DATA
    # ========================================================

    edited_input_df = input_data_editor(
        input_df,
        is_cluster,
    )

    st.session_state.input_df = (
        edited_input_df
    )

    # ========================================================
    # PLANT TYPE
    # ========================================================

    plant_type = plant_selector()

    if (
        st.session_state.last_plant_type
        != plant_type
    ):

        st.session_state.tracking_params = None
        st.session_state.run_model = False

        st.session_state.last_plant_type = (
            plant_type
        )

    # ========================================================
    # RUN BUTTON
    # ========================================================

    st.markdown(
        '<div class="run-box">',
        unsafe_allow_html=True,
    )

    run_clicked = st.button(
        "🚀 RUN LOSS CORRECTION",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    if run_clicked:

        # Always optimize tracking again
        # when explicitly running the model.

        st.session_state.tracking_params = None

        st.session_state.run_model = True

    if not st.session_state.run_model:

        st.info(
            "Select Fixed or Tracking and click "
            "**Run Loss Correction**."
        )

        return

    # ========================================================
    # RUN MODEL
    # ========================================================

    try:

        if not is_cluster:

            if plant_type == "🏗️ Fixed":

                run_noncluster_fixed(
                    df,
                    edited_input_df,
                    lat,
                    tilt_lookup,
                )

            else:

                run_noncluster_tracking(
                    uploaded_file,
                    df,
                    edited_input_df,
                    lat,
                    tilt_lookup,
                )

        else:

            if plant_type == "🏗️ Fixed":

                run_cluster_fixed(
                    uploaded_file,
                    df,
                    edited_input_df,
                    lat,
                    tilt_lookup,
                )

            else:

                run_cluster_tracking(
                    uploaded_file,
                    df,
                    edited_input_df,
                    lat,
                    tilt_lookup,
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
