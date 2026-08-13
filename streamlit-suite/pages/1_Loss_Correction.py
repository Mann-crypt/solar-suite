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
OPT_POPSIZE = 10

PARAM_BOUNDS = [
    (0, 10),     # DHI
    (0, 30),     # Starting block
    (65, 80),    # Ending block
    (44, 60),    # Max block
    (0, 70),     # East limit
    (0, 70),     # West limit
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

    /* Plant selector buttons */
    div.stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 650;
        transition: 0.15s ease;
    }

    /* Selected button */
    .selected-plant {
        padding: 12px 16px;
        border-radius: 12px;
        margin-top: 8px;
        font-weight: 600;
        text-align: center;
        background: rgba(37, 99, 235, 0.12);
        border: 1px solid rgba(37, 99, 235, 0.45);
        color: #60a5fa;
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
    "model_context": None,
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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing: {', '.join(missing)}"
        )


def get_sheet_names(uploaded_file):
    uploaded_file.seek(0)
    return pd.ExcelFile(uploaded_file).sheet_names


def clean_data_rows(df, date_column="Date"):
    df = df.copy()

    if date_column in df.columns:
        idx = df[df[date_column].isna()].index

        if len(idx):
            pos = df.index.get_loc(idx[0])
            df = df.iloc[:pos]

    return df.reset_index(drop=True)


# ============================================================
# WORKBOOK DETECTION
# ============================================================

def detect_cluster(uploaded_file):
    return "Fixed" not in get_sheet_names(uploaded_file)


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def read_area_efficiency(uploaded_file, cluster=False):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(8) if cluster else None,
    )

    df.columns = df.columns.astype(str).str.strip()

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
        idx = df[df["Module Type"].isna()].index

        if len(idx):
            pos = df.index.get_loc(idx[0])
            df = df.iloc[:pos]

    df = df.dropna(
        subset=[
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        how="all",
    )

    return df.reset_index(drop=True)


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

    df.columns = df.columns.astype(str).str.strip()

    cols = ["CL-1", "CL-2", "CL-3", "CL-4", "CL-5"]

    validate_columns(df, cols, "Cluster Weights")

    return {
        c: float(df[c].iloc[0])
        for c in cols
    }


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

    df.columns = df.columns.astype(str).str.strip()

    validate_columns(df, ["Lat"], "Forecast Config")

    return float(df["Lat"].iloc[0])


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

        df.columns = df.columns.astype(str).str.strip()

        if "Fixed" not in df.columns:
            return {}

        idx = df[df["Fixed"].isna()].index

        if len(idx):
            pos = df.index.get_loc(idx[0])
            df = df.iloc[:pos]

        df = df.dropna(axis=1, how="all")

        df = df.rename(
            columns={
                "Unnamed: 2": "Month_Num",
                "Unnamed: 3": "Month",
            }
        )

        if "Month" not in df.columns:
            return {}

        return (
            df.dropna(subset=["Month"])
            .set_index("Month")["Fixed"]
            .to_dict()
        )

    except Exception:
        return {}


# ============================================================
# SOLAR ANGLES
# ============================================================

def prepare_solar_angles(
    df,
    lat,
    tilt_lookup=None,
    tracking=False,
):

    df = df.copy()

    today = pd.Timestamp.today().normalize()
    df["Date"] = today

    first_date = today.replace(month=1, day=1)

    day_number = (
        df["Date"] - first_date
    ).dt.days + 1

    df["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360 * (284 + day_number) / 365
            )
        )
    )

    df["Elevation angle a"] = (
        90 - lat + df["Declination Angle ∆"]
    )

    if tracking:
        df["Tilt Angle b"] = 0
    else:
        if tilt_lookup:
            df["Tilt Angle b"] = (
                df["Date"]
                .dt.strftime("%B")
                .map(tilt_lookup)
                .fillna(0)
            )
        else:
            df["Tilt Angle b"] = 0

    df["a+b"] = (
        df["Elevation angle a"]
        + df["Tilt Angle b"]
    )

    df["SIN(a+b)"] = np.sin(
        np.radians(df["a+b"])
    )

    df["Sin(a)"] = np.sin(
        np.radians(df["Elevation angle a"])
    ).clip(lower=1e-6)

    return df


# ============================================================
# EFFICIENCY LOSS
# ============================================================

def calculate_efficiency_loss(df, poa, actual):

    standard = df[
        "Standard PV Efficiency (%)"
    ].to_numpy(float)

    area = df[
        "Total area(m2)"
    ].to_numpy(float)

    actual = np.asarray(actual, float)
    poa = np.asarray(poa, float)

    valid_actual = actual[np.isfinite(actual)]
    valid_poa = poa[np.isfinite(poa)]

    if not len(valid_actual) or not len(valid_poa):
        return 0.0

    poa_peak = np.nanmax(valid_poa)

    if poa_peak <= 0:
        return 0.0

    actual_peak = np.nanmax(valid_actual)

    base_area = np.sum(
        area * standard / 100
    )

    loss_coeff = np.sum(area / 100)

    if loss_coeff <= 0:
        return 0.0

    target_area = (
        actual_peak * 1_000_000 / poa_peak
    )

    loss = (
        base_area - target_area
    ) / loss_coeff

    return float(
        np.clip(
            loss,
            0,
            np.nanmin(standard),
        )
    )


def apply_efficiency_loss(df, loss):

    df = df.copy()

    df["Efficiency Losses(%)"] = loss

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"] - loss
    )

    df["Eff Area"] = (
        df["Total area(m2)"]
        * df["Net Efficiency (%)"]
        / 100
    )

    return df


# ============================================================
# INPUT DATA EDITOR
# ============================================================

def load_input_data(uploaded_file, cluster):

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

    df.columns = df.columns.astype(str).str.strip()

    df = clean_data_rows(df)

    validate_columns(
        df,
        ["Actual"],
        "Forecast Sheet",
    )

    # --------------------------------------------------------
    # CLUSTER GHI
    # --------------------------------------------------------

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

            for i, col in enumerate(ghi_cols):

                if col not in df.columns and i < len(result.columns):

                    values = result.iloc[
                        :len(df), i
                    ].to_numpy()

                    if len(values) < len(df):

                        values = np.pad(
                            values,
                            (0, len(df) - len(values)),
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

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

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


def input_data_editor(df, cluster):

    st.markdown(
        '<div class="section-title">📊 Input GHI and Power</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "All plant parameters are extracted from Excel. "
        "You only need to modify GHI Forecast and Actual values."
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

    available = [
        c for c in edit_cols
        if c in df.columns
    ]

    display = df[available].copy()

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
            for col in available
        },
    )

    result = df.copy()

    for col in available:

        result[col] = pd.to_numeric(
            edited[col],
            errors="coerce",
        ).fillna(0)

    return result


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
            len(solar),
            dtype=float,
        )

        for ghi, weight in zip(
            ghi_cols,
            weight_cols,
        ):

            poa = (
                solar[ghi]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            eff_area = (
                df["Total area(m2)"]
                * df["Net Efficiency (%)"]
                / 100
                * weights[weight]
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

    mask = (
        np.isfinite(actual)
        & np.isfinite(weighted_ghi)
        & (actual != 0)
    )

    actual = actual[mask]
    weighted_ghi = weighted_ghi[mask]
    blocks = blocks[mask]

    if len(actual) == 0:
        raise ValueError(
            "No valid Actual power values found."
        )

    actual_peak = np.max(actual)
    actual_energy = np.sum(actual)

    if actual_peak <= 0 or actual_energy <= 0:
        raise ValueError(
            "Actual power data is invalid."
        )

    def objective(x):

        DHI, start, end, max_block, east, west = (
            np.rint(x).astype(int)
        )

        if not (
            start < max_block < end
        ):
            return 1e9

        d1 = start - 1 - max_block
        d2 = end + 1 - max_block

        if d1 == 0 or d2 == 0:
            return 1e9

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
            np.minimum(zenith, abs(east)),
            np.where(
                (blocks > max_block)
                & (zenith > west),
                west,
                zenith,
            ),
        )

        cos_alpha = np.clip(
            np.cos(np.radians(panel)),
            1e-6,
            None,
        )

        prediction = (
            weighted_ghi
            * (1 - DHI / 100)
            / cos_alpha
            / 1_000_000
        )

        if not np.all(np.isfinite(prediction)):
            return 1e9

        block_error = (
            np.mean(np.abs(actual - prediction))
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

    best = np.rint(result.x).astype(int)

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

    d1 = start - 1 - max_block
    d2 = end + 1 - max_block

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
        np.minimum(zenith, abs(east)),
        np.where(
            (blocks > max_block)
            & (zenith > west),
            west,
            zenith,
        ),
    )

    cos_alpha = np.clip(
        np.cos(np.radians(panel)),
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
# EFFICIENCY UI
# ============================================================

def efficiency_control(
    df,
    auto_loss,
    key,
):

    st.markdown(
        '<div class="section-title">📉 Efficiency Loss</div>',
        unsafe_allow_html=True,
    )

    max_loss = float(
        df["Standard PV Efficiency (%)"].min()
    )

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=max_loss,
        value=float(auto_loss),
        step=0.1,
        format="%.2f",
        key=key,
        help="Automatically calculated initially. You can manually change it.",
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

    nums = display.select_dtypes(
        include="number"
    ).columns

    display[nums] = display[nums].round(2)

    with st.expander(
        "🔍 View Efficiency Calculations"
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

    x = np.arange(1, n + 1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(forecast[:n]),
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
            y=np.asarray(actual[:n]),
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
# PLANT TYPE SELECTOR
# ============================================================

def plant_selector():

    st.markdown(
        '<div class="section-title">🏭 Plant Type</div>',
        unsafe_allow_html=True,
    )

    selected = st.session_state.plant_type

    c1, c2 = st.columns(2)

    # ========================================================
    # FIXED PLANT
    # ========================================================

    with c1:

        if selected == "🏗️ Fixed":

            st.markdown(
                """
                <div style="
                    border: 2px solid #3b82f6;
                    background: rgba(59,130,246,0.15);
                    border-radius: 14px;
                    padding: 14px 18px;
                    margin-bottom: 8px;
                    min-height: 82px;
                ">
                    <div style="
                        font-size: 17px;
                        font-weight: 700;
                        color: #3b82f6;
                    ">
                        🏗️ Fixed Plant
                    </div>

                    <div style="
                        font-size: 13px;
                        opacity: 0.75;
                        margin-top: 5px;
                    ">
                        Panels remain at a fixed tilt angle
                        throughout the day.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "✓ Selected",
                key="fixed_active",
                use_container_width=True,
            ):
                pass

        else:

            st.markdown(
                """
                <div style="
                    padding: 14px 18px 5px 18px;
                    min-height: 82px;
                ">
                    <div style="
                        font-size: 17px;
                        font-weight: 700;
                    ">
                        🏗️ Fixed Plant
                    </div>

                    <div style="
                        font-size: 13px;
                        opacity: 0.65;
                        margin-top: 5px;
                    ">
                        Panels remain at a fixed tilt angle
                        throughout the day.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "Select Fixed Plant",
                key="fixed_button",
                use_container_width=True,
            ):

                st.session_state.plant_type = "🏗️ Fixed"
                st.session_state.tracking_params = None
                st.session_state.run_model = False
                st.rerun()

    # ========================================================
    # TRACKING PLANT
    # ========================================================

    with c2:

        if selected == "🔄 Tracking":

            st.markdown(
                """
                <div style="
                    border: 2px solid #10b981;
                    background: rgba(16,185,129,0.15);
                    border-radius: 14px;
                    padding: 14px 18px;
                    margin-bottom: 8px;
                    min-height: 82px;
                ">
                    <div style="
                        font-size: 17px;
                        font-weight: 700;
                        color: #10b981;
                    ">
                        🔄 Tracking Plant
                    </div>

                    <div style="
                        font-size: 13px;
                        opacity: 0.75;
                        margin-top: 5px;
                    ">
                        Panels adjust their angle during the day
                        to follow the sun.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "✓ Selected",
                key="tracking_active",
                use_container_width=True,
            ):
                pass

        else:

            st.markdown(
                """
                <div style="
                    padding: 14px 18px 5px 18px;
                    min-height: 82px;
                ">
                    <div style="
                        font-size: 17px;
                        font-weight: 700;
                    ">
                        🔄 Tracking Plant
                    </div>

                    <div style="
                        font-size: 13px;
                        opacity: 0.65;
                        margin-top: 5px;
                    ">
                        Panels adjust their angle during the day
                        to follow the sun.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "Select Tracking Plant",
                key="tracking_button",
                use_container_width=True,
            ):

                st.session_state.plant_type = "🔄 Tracking"
                st.session_state.tracking_params = None
                st.session_state.run_model = False
                st.rerun()

    return selected

# ============================================================
# TRACKING PARAMETERS
# ============================================================

def tracking_parameter_controls(params, prefix):

    st.markdown(
        '<div class="section-title">⚙️ Tracking Parameters</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Optimizer values are loaded automatically. "
        "You can manually modify them before recalculating the forecast."
    )

    c1, c2, c3 = st.columns(3)

    DHI = c1.number_input(
        "DHI (%)",
        0,
        10,
        int(params["DHI"]),
        1,
        key=f"{prefix}_dhi",
    )

    start = c2.number_input(
        "Starting Block",
        0,
        30,
        int(params["start"]),
        1,
        key=f"{prefix}_start",
    )

    end = c3.number_input(
        "Ending Block",
        65,
        80,
        int(params["end"]),
        1,
        key=f"{prefix}_end",
    )

    c1, c2, c3 = st.columns(3)

    max_block = c1.number_input(
        "Max Block",
        44,
        60,
        int(params["max"]),
        1,
        key=f"{prefix}_max",
    )

    east = c2.number_input(
        "East Limit",
        0,
        70,
        int(params["east"]),
        1,
        key=f"{prefix}_east",
    )

    west = c3.number_input(
        "West Limit",
        0,
        70,
        int(params["west"]),
        1,
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
# NON-CLUSTER FIXED
# ============================================================

def run_noncluster_fixed(
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

    show_efficiency_table(df)

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

    df = efficiency_control(
        df,
        auto_loss,
        "noncluster_tracking_loss",
    )

    weighted_ghi = (
        input_df["GHI_Forecast"].to_numpy(float)
        * df["Eff Area"].sum()
    )

    uploaded_file.seek(0)

    backend = pd.read_excel(
        uploaded_file,
        sheet_name="Backend Cal",
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
    # OPTIMIZATION ONLY WHEN BUTTON WAS PRESSED
    # --------------------------------------------------------

    if st.session_state.tracking_params is None:

        with st.spinner(
            "🔄 Optimizing tracking parameters... "
            "Please wait."
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

    show_efficiency_table(df)

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
            input_df[ghi_col].to_numpy(float)
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
    # OPTIMIZE ONLY WHEN REQUIRED
    # --------------------------------------------------------

    if st.session_state.tracking_params is None:

        with st.spinner(
            "🔄 Optimizing tracking parameters... "
            "Please wait."
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
        '<div class="main-title">☀️ Loss Correction Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Upload the Excel workbook, modify GHI Forecast and Actual, "
        "select the plant type and run the correction."
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # INPUT EXCEL
    # ========================================================

    st.markdown(
        '<div class="section-title">📁 Input Sheet</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx", "xls"],
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload the plant Excel file to begin."
        )

        return

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

    # ========================================================
    # LOAD EXCEL PARAMETERS
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

        input_df = load_input_data(
            uploaded_file,
            is_cluster,
        )

    except Exception as e:

        st.error(
            f"Unable to load workbook: {e}"
        )

        return

    # ========================================================
    # INPUT DATA EDITOR
    # ========================================================

    input_df = input_data_editor(
        input_df,
        is_cluster,
    )

    # Keep edited data in session
    st.session_state.input_df = input_df

    # ========================================================
    # PLANT TYPE
    # ========================================================

    plant_type = plant_selector()

    # ========================================================
    # RUN BUTTON
    # ========================================================

    st.markdown("")

    run_clicked = st.button(
        "🚀  RUN LOSS CORRECTION",
        type="primary",
        use_container_width=True,
        key="run_loss_correction",
    )

    if run_clicked:

        # Reset optimization when user explicitly runs again
        if plant_type == "🏗️ Fixed":

            st.session_state.tracking_params = None

        else:

            st.session_state.tracking_params = None

        st.session_state.run_model = True

    if not st.session_state.run_model:

        st.info(
            "Select the plant type and click "
            "**Run Loss Correction** to start."
        )

        return

    # ========================================================
    # LOAD PARAMETERS
    # ========================================================

    # Re-read workbook parameters because the Excel file
    # itself is the source of truth.
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
            f"Unable to load model configuration: {e}"
        )

        return

    # ========================================================
    # MODEL
    # ========================================================

    try:

        if not is_cluster:

            if plant_type == "🏗️ Fixed":

                run_noncluster_fixed(
                    uploaded_file,
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
            "❌ Loss correction failed."
        )

        st.exception(e)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
