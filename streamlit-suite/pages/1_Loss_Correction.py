```python
# ============================================================
# LOSS CORRECTION MODEL
# Fixed / Tracking
# Compact + Non-Freezing Streamlit Page
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
    (0, 10),      # DHI
    (0, 30),      # Start
    (65, 80),     # End
    (44, 60),     # Max
    (0, 70),      # East
    (0, 70),      # West
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 2rem;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #8b949e;
        margin-bottom: 22px;
    }

    .section {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 18px 0 8px;
    }

    .card {
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.22);
        background: rgba(128,128,128,.035);
        margin-bottom: 14px;
    }

    div[data-testid="stSegmentedControl"] {
        width: 100%;
    }

    div[data-testid="stSegmentedControl"] > div {
        width: 100%;
    }

    div[data-testid="stSegmentedControl"] button {
        flex: 1;
        font-weight: 650;
        min-height: 44px;
    }

    div.stButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "plant_type": "🏗️ Fixed",
    "input_df": None,
    "input_signature": None,
    "workbook_signature": None,
    "model_result": None,
    "tracking_params": None,
    "auto_loss": None,
    "run_id": 0,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GENERIC HELPERS
# ============================================================

def numeric(series):
    """
    Safe numeric conversion.
    Handles Series, lists and numpy arrays.
    """
    if isinstance(series, pd.Series):
        return pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(0)

    return pd.Series(
        pd.to_numeric(
            np.asarray(series),
            errors="coerce",
        )
    ).fillna(0)


def validate_columns(df, required, name="Data"):
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required column(s): "
            f"{', '.join(missing)}"
        )


def clean_data_rows(df, date_column="Date"):
    df = df.copy()

    if date_column in df.columns:
        null_rows = df[df[date_column].isna()].index

        if len(null_rows):
            pos = df.index.get_loc(null_rows[0])
            df = df.iloc[:pos]

    return df.reset_index(drop=True)


def signature(df):
    """
    Small signature used to detect changed input data.
    """
    return (
        len(df),
        tuple(df.columns),
        float(
            pd.to_numeric(
                df.select_dtypes(include="number"),
                errors="coerce",
            )
            .fillna(0)
            .to_numpy()
            .sum()
        )
        if not df.select_dtypes(include="number").empty
        else 0.0,
    )


# ============================================================
# CACHED EXCEL HELPERS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def workbook_sheets(file_bytes):
    return tuple(
        pd.ExcelFile(
            io.BytesIO(file_bytes)
        ).sheet_names
    )


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_area_efficiency_cached(
    file_bytes,
    cluster,
):
    df = pd.read_excel(
        io.BytesIO(file_bytes),
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
        null_rows = df[
            df["Module Type"].isna()
        ].index

        if len(null_rows):
            pos = df.index.get_loc(null_rows[0])
            df = df.iloc[:pos]

    df = df.dropna(
        subset=[
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        how="all",
    )

    return df.reset_index(drop=True)


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_cluster_weights_cached(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
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
        col: float(
            pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0).iloc[0]
        )
        for col in cols
    }


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_latitude_cached(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
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
            df["Lat"],
            errors="coerce",
        ).dropna().iloc[0]
    )


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_tilt_lookup_cached(file_bytes):

    try:
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
            return {}

        null_rows = df[
            df["Fixed"].isna()
        ].index

        if len(null_rows):
            pos = df.index.get_loc(null_rows[0])
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

        df["Fixed"] = pd.to_numeric(
            df["Fixed"],
            errors="coerce",
        )

        return (
            df.dropna(subset=["Month"])
            .set_index("Month")["Fixed"]
            .dropna()
            .to_dict()
        )

    except Exception:
        return {}


# ============================================================
# INPUT DATA
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def load_input_data_cached(
    file_bytes,
    cluster,
):

    sheet = (
        "Fixed-CL1"
        if cluster
        else "Fixed"
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

        try:
            result = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name="Result",
                usecols=range(6),
            )

            result = result.fillna(0)

            for i, col in enumerate(ghi_cols):

                if (
                    col not in df.columns
                    and i < len(result.columns)
                ):

                    values = pd.to_numeric(
                        result.iloc[:len(df), i],
                        errors="coerce",
                    ).fillna(0).to_numpy()

                    if len(values) < len(df):
                        values = np.pad(
                            values,
                            (
                                0,
                                len(df) - len(values),
                            ),
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

    df["Actual"] = numeric(
        df["Actual"]
    ).to_numpy()

    if not cluster:

        df["GHI_Forecast"] = numeric(
            df["GHI_Forecast"]
        ).to_numpy()

    else:

        for col in [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]:
            df[col] = numeric(
                df[col]
            ).to_numpy()

    return df


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

    first_date = pd.Timestamp(
        year=today.year,
        month=1,
        day=1,
    )

    day_number = (
        result["Date"] - first_date
    ).dt.days + 1

    result["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + day_number)
                / 365
            )
        )
    )

    result["Elevation angle a"] = (
        90
        - lat
        + result[
            "Declination Angle ∆"
        ]
    )

    if tracking:

        result["Tilt Angle b"] = 0

    elif tilt_lookup:

        result["Tilt Angle b"] = (
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
    ).clip(lower=1e-6)

    return result


# ============================================================
# EFFICIENCY
# ============================================================

def calculate_efficiency_loss(
    df,
    poa,
    actual,
):

    standard = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
        errors="coerce",
    ).fillna(0).to_numpy(float)

    area = pd.to_numeric(
        df["Total area(m2)"],
        errors="coerce",
    ).fillna(0).to_numpy(float)

    actual = np.asarray(
        actual,
        dtype=float,
    )

    poa = np.asarray(
        poa,
        dtype=float,
    )

    actual = actual[
        np.isfinite(actual)
    ]

    poa = poa[
        np.isfinite(poa)
    ]

    if not len(actual) or not len(poa):
        return 0.0

    poa_peak = np.nanmax(poa)
    actual_peak = np.nanmax(actual)

    if poa_peak <= 0 or actual_peak <= 0:
        return 0.0

    base_area = np.sum(
        area * standard / 100
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

    result["Efficiency Losses(%)"] = loss

    result["Net Efficiency (%)"] = (
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
# TRACKING OPTIMIZER
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=8,
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

    actual_peak = np.max(actual)
    actual_energy = np.sum(actual)

    if actual_peak <= 0:
        raise ValueError(
            "Actual peak power is invalid."
        )

    if actual_energy <= 0:
        raise ValueError(
            "Actual energy is invalid."
        )

    def objective(x):

        (
            DHI,
            start,
            end,
            max_block,
            east,
            west,
        ) = np.rint(x).astype(int)

        if not (
            start < max_block < end
        ):
            return 1e9

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
            return 1e9

        m1 = 90 / d1
        m2 = 90 / d2

        zenith = np.where(
            blocks <= max_block,
            np.minimum(
                89,
                m1
                * (blocks - max_block),
            ),
            np.minimum(
                89,
                m2
                * (blocks - max_block),
            ),
        )

        panel = np.where(
            blocks < max_block,
            np.minimum(
                zenith,
                abs(east),
            ),
            np.where(
                (blocks > max_block)
                & (zenith > west),
                west,
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

        prediction = (
            weighted_ghi
            * (1 - DHI / 100)
            / cos_alpha
            / 1_000_000
        )

        if not np.all(
            np.isfinite(prediction)
        ):
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual - prediction
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

        return (
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

def tracking_forecast(
    blocks,
    weighted_ghi,
    params,
):

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
            m1 * (blocks - max_block),
        ),
        np.minimum(
            89,
            m2 * (blocks - max_block),
        ),
    )

    panel = np.where(
        blocks < max_block,
        np.minimum(
            zenith,
            abs(east),
        ),
        np.where(
            (blocks > max_block)
            & (zenith > west),
            west,
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

    return (
        weighted_ghi
        * (1 - DHI / 100)
        / cos_alpha
        / 1_000_000
    )


# ============================================================
# INPUT EDITOR
# ============================================================

def input_data_editor(
    df,
    cluster,
):

    st.markdown(
        '<div class="section">📊 Input GHI & Actual Power</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Edit the GHI forecast and Actual values, then click "
        "**RUN LOSS CORRECTION**. Excel parameters are calculated automatically."
    )

    if cluster:
        columns = [
            "Actual",
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]
    else:
        columns = [
            "GHI_Forecast",
            "Actual",
        ]

    columns = [
        c for c in columns
        if c in df.columns
    ]

    display = df[columns].copy()

    for col in columns:
        display[col] = numeric(
            display[col]
        ).to_numpy()

    with st.form(
        "input_data_form",
        clear_on_submit=False,
        border=False,
    ):

        edited = st.data_editor(
            display,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="loss_input_editor",
            column_config={
                col: st.column_config.NumberColumn(
                    col,
                    step=0.01,
                    format="%.2f",
                )
                for col in columns
            },
        )

        submitted = st.form_submit_button(
            "✓ Update Input Data",
            type="secondary",
            use_container_width=False,
        )

    if submitted:
        result = df.copy()

        for col in columns:
            result[col] = numeric(
                edited[col]
            ).to_numpy()

        st.session_state.input_df = result
        st.session_state.model_result = None
        st.session_state.tracking_params = None
        st.session_state.auto_loss = None

        st.success(
            "Input data updated. Click RUN LOSS CORRECTION."
        )

        return result

    if st.session_state.input_df is not None:
        return st.session_state.input_df

    return df.copy()


# ============================================================
# PLANT SELECTOR
# ============================================================

def plant_selector():

    st.markdown(
        '<div class="section">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    return st.segmented_control(
        "Plant Type",
        [
            "🏗️ Fixed",
            "🔄 Tracking",
        ],
        default=st.session_state.plant_type,
        selection_mode="single",
        key="plant_type",
        label_visibility="collapsed",
        width="stretch",
    )


# ============================================================
# TRACKING PARAMETERS
# ============================================================

def tracking_parameter_controls(
    params,
    prefix,
):

    st.markdown(
        '<div class="section">⚙️ Tracking Parameters</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Values are automatically optimized. You can edit them directly."
    )

    c1, c2, c3 = st.columns(3)

    dhi = c1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=10,
        value=int(params["DHI"]),
        step=1,
        key=f"{prefix}_dhi",
    )

    start = c2.number_input(
        "Starting Block",
        min_value=0,
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
        min_value=44,
        max_value=60,
        value=int(params["max"]),
        step=1,
        key=f"{prefix}_max",
    )

    east = c2.number_input(
        "East Limit",
        min_value=0,
        max_value=70,
        value=int(params["east"]),
        step=1,
        key=f"{prefix}_east",
    )

    west = c3.number_input(
        "West Limit",
        min_value=0,
        max_value=70,
        value=int(params["west"]),
        step=1,
        key=f"{prefix}_west",
    )

    return {
        "DHI": int(dhi),
        "start": int(start),
        "end": int(end),
        "max": int(max_block),
        "east": int(east),
        "west": int(west),
    }


# ============================================================
# EFFICIENCY CONTROL
# ============================================================

def efficiency_control(
    df,
    auto_loss,
    key,
):

    max_loss = float(
        pd.to_numeric(
            df[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce",
        ).min()
    )

    if auto_loss is None:
        auto_loss = 0.0

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=max_loss,
        value=float(auto_loss),
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

def show_efficiency_table(df):

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

    display = df[cols].copy()

    numeric_cols = display.select_dtypes(
        include="number"
    ).columns

    display[numeric_cols] = (
        display[numeric_cols]
        .round(2)
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
# FORECAST CHART
# ============================================================

def show_forecast_chart(
    forecast,
    actual,
    title,
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

    if n == 0:
        st.warning(
            "No forecast data available."
        )
        return

    x = np.arange(
        1,
        n + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast[:n],
            mode="lines",
            name="Forecast",
            line=dict(
                color="#2563EB",
                width=2.5,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual[:n],
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
        height=430,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
        ),
        xaxis_title="15 Minute Block",
        yaxis_title="Power (MW)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


# ============================================================
# BACKEND BLOCKS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_backend_blocks_cached(
    file_bytes,
    cluster,
):

    sheet = (
        "Backend Cal CL1"
        if cluster
        else "Backend Cal"
    )

    backend = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet,
    )

    backend.columns = (
        backend.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        backend,
        ["Block No."],
        sheet,
    )

    return numeric(
        backend["Block No."]
    ).to_numpy()


# ============================================================
# FIXED FORECAST
# ============================================================

def run_fixed(
    df,
    input_df,
    lat,
    tilt_lookup,
    cluster,
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

        poa_for_loss = (
            solar["CL1-GHI"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

        auto_loss = calculate_efficiency_loss(
            df,
            poa_for_loss,
            input_df["Actual"],
        )

        loss_key = "cluster_fixed_loss"

        df = efficiency_control(
            df,
            auto_loss,
            loss_key,
        )

        forecast = np.zeros(
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

            poa = (
                solar[ghi_col]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            forecast += (
                poa.to_numpy()
                * eff_area
                / 1_000_000
            )

    else:

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
            "fixed_loss",
        )

        forecast = (
            poa.to_numpy()
            * df["Eff Area"].sum()
            / 1_000_000
        )

    show_efficiency_table(df)

    show_forecast_chart(
        forecast,
        input_df["Actual"].to_numpy(float),
        (
            "🏗️ Fixed Cluster Forecast vs Actual"
            if cluster
            else "🏗️ Fixed Forecast vs Actual"
        ),
    )


# ============================================================
# TRACKING FORECAST
# ============================================================

def run_tracking(
    df,
    input_df,
    lat,
    tilt_lookup,
    file_bytes,
    cluster,
):

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
        tracking=True,
    )

    if cluster:

        weights = read_cluster_weights_cached(
            file_bytes
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

        poa_for_loss = (
            solar["CL1-GHI"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

        auto_loss = calculate_efficiency_loss(
            df,
            poa_for_loss,
            input_df["Actual"],
        )

        df = efficiency_control(
            df,
            auto_loss,
            "cluster_tracking_loss",
        )

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
                input_df[ghi_col]
                .to_numpy(float)
                * eff_area
            )

        prefix = "cluster_tracking"

    else:

        poa_for_loss = (
            solar["GHI_Forecast"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

        auto_loss = calculate_efficiency_loss(
            df,
            poa_for_loss,
            input_df["Actual"],
        )

        df = efficiency_control(
            df,
            auto_loss,
            "tracking_loss",
        )

        weighted_ghi = (
            input_df["GHI_Forecast"]
            .to_numpy(float)
            * df["Eff Area"].sum()
        )

        prefix = "tracking"

    blocks = read_backend_blocks_cached(
        file_bytes,
        cluster,
    )

    actual = input_df[
        "Actual"
    ].to_numpy(float)

    # --------------------------------------------------------
    # AUTOMATIC OPTIMIZATION
    # --------------------------------------------------------

    if (
        st.session_state.tracking_params is None
    ):

        with st.spinner(
            "🔄 Optimizing tracking parameters..."
        ):

            params = optimize_tracking_cached(
                tuple(blocks),
                tuple(weighted_ghi),
                tuple(actual),
            )

        st.session_state.tracking_params = params

    else:

        params = (
            st.session_state.tracking_params
        )

    # --------------------------------------------------------
    # EDITABLE PARAMETERS
    # --------------------------------------------------------

    params = tracking_parameter_controls(
        params,
        prefix,
    )

    st.session_state.tracking_params = params

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    try:

        forecast = tracking_forecast(
            blocks,
            weighted_ghi,
            params,
        )

        show_efficiency_table(df)

        show_forecast_chart(
            forecast,
            actual,
            (
                "🔄 Tracking Cluster Forecast vs Actual"
                if cluster
                else "🔄 Tracking Forecast vs Actual"
            ),
        )

    except Exception as exc:

        st.error(
            f"Unable to calculate tracking forecast: {exc}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="title">☀️ Loss Correction Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Upload your workbook, edit the input data if required, "
        "select the plant type and run the correction."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # FILE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">📁 Input Excel</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel Workbook",
        type=["xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload the Excel workbook to begin."
        )

        return

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        st.error("The uploaded file is empty.")
        return

    # --------------------------------------------------------
    # WORKBOOK
    # --------------------------------------------------------

    try:

        sheets = workbook_sheets(
            file_bytes
        )

        cluster = "Fixed" not in sheets

    except Exception as exc:

        st.error(
            f"Unable to read workbook: {exc}"
        )

        return

    workbook_sig = (
        len(file_bytes),
        hash(file_bytes),
    )

    if (
        st.session_state.workbook_signature
        != workbook_sig
    ):

        st.session_state.workbook_signature = workbook_sig

        st.session_state.input_df = None
        st.session_state.input_signature = None
        st.session_state.model_result = None
        st.session_state.tracking_params = None
        st.session_state.auto_loss = None
        st.session_state.run_id = 0

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    try:

        base_df = read_area_efficiency_cached(
            file_bytes,
            cluster,
        )

        lat = read_latitude_cached(
            file_bytes
        )

        tilt_lookup = read_tilt_lookup_cached(
            file_bytes
        )

        if cluster:
            weights = read_cluster_weights_cached(
                file_bytes
            )
        else:
            weights = None

        original_input = load_input_data_cached(
            file_bytes,
            cluster,
        )

    except Exception as exc:

        st.error(
            f"Unable to load workbook parameters: {exc}"
        )

        return

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    input_df = input_data_editor(
        original_input,
        cluster,
    )

    st.session_state.input_df = input_df

    # --------------------------------------------------------
    # PLANT TYPE
    # --------------------------------------------------------

    plant_type = plant_selector()

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    st.markdown("")

    run_clicked = st.button(
        "🚀 RUN LOSS CORRECTION",
        type="primary",
        use_container_width=True,
        key="run_loss_correction",
    )

    if run_clicked:

        # New explicit run.
        # Tracking optimizer will execute again.
        st.session_state.tracking_params = None
        st.session_state.auto_loss = None
        st.session_state.model_result = None

        st.session_state.run_id += 1

    # --------------------------------------------------------
    # NOTHING TO RUN
    # --------------------------------------------------------

    if (
        st.session_state.run_id == 0
        and st.session_state.model_result is None
    ):

        st.info(
            "Select the plant type and click "
            "**RUN LOSS CORRECTION**."
        )

        return

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    try:

        if plant_type == "🏗️ Fixed":

            run_fixed(
                base_df,
                input_df,
                lat,
                tilt_lookup,
                cluster,
                weights,
            )

        else:

            run_tracking(
                base_df,
                input_df,
                lat,
                tilt_lookup,
                file_bytes,
                cluster,
            )

        st.session_state.model_result = True

    except Exception as exc:

        st.error(
            "❌ Loss correction failed."
        )

        st.exception(exc)


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()
```
