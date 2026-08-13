import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datetime import datetime, timedelta
from scipy.optimize import differential_evolution


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="RT Correction — Solar Suite",
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
# CONSTANTS
# ==========================================================

BLOCKS = 96

DEFAULT_PARAMS = {
    "w": 0.30,
    "n1": 29,
    "n2": 72,
    "b": 39,
}

# Smaller optimization settings to prevent Streamlit freezing
OPT_MAXITER = 40
OPT_POPSIZE = 10


# ==========================================================
# SESSION STATE
# ==========================================================

if "rt_actual" not in st.session_state:
    st.session_state.rt_actual = [0.0] * BLOCKS

if "rt_trend" not in st.session_state:
    st.session_state.rt_trend = [0.0] * BLOCKS

if "rt_params" not in st.session_state:
    st.session_state.rt_params = DEFAULT_PARAMS.copy()

if "rt_optimized" not in st.session_state:
    st.session_state.rt_optimized = False


# ==========================================================
# CSS
# ==========================================================

st.markdown(
    """
    <style>

    div[data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px 20px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 44px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# HEAVY OPTIMIZATION
# ==========================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def cached_rt_optimize(
    actual_tuple,
    trend_tuple,
):

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    trend = np.asarray(
        trend_tuple,
        dtype=float,
    )

    blocks = np.arange(
        1,
        BLOCKS + 1,
        dtype=float,
    )

    # ------------------------------------------------------
    # Daylight / valid mask
    # ------------------------------------------------------

    mask = (
        np.isfinite(actual)
        & np.isfinite(trend)
        & (actual > 0.5)
    )

    if not np.any(mask):
        raise ValueError(
            "No valid Actual values greater than 0.5 were found."
        )

    actual_valid = actual[mask]
    trend_valid = trend[mask]

    actual_peak = np.max(
        actual_valid
    )

    actual_energy = np.sum(
        actual_valid
    )

    if (
        actual_peak <= 0
        or actual_energy <= 0
    ):
        raise ValueError(
            "Actual generation data is invalid."
        )

    # ------------------------------------------------------
    # Objective
    # ------------------------------------------------------

    def objective(x):

        w = float(x[0])

        n1 = int(
            round(x[1])
        )

        n2 = int(
            round(x[2])
        )

        b = int(
            round(x[3])
        )

        # ----------------------------------------------
        # Structural validation
        # ----------------------------------------------

        if not (
            n1 < b < n2
        ):
            return 1e9

        denominator = (
            (n1 - b)
            * (n2 - b)
        )

        if denominator == 0:
            return 1e9

        # ----------------------------------------------
        # Peak generation
        # ----------------------------------------------

        peak_mask = np.isin(
            blocks,
            [
                b - 1,
                b,
                b + 1,
            ],
        )

        peak_values = actual[
            peak_mask
        ]

        if len(peak_values) == 0:
            return 1e9

        peak_power = np.mean(
            peak_values
        )

        # ----------------------------------------------
        # Parabolic projection
        # ----------------------------------------------

        projection = (
            peak_power
            * (
                (n1 - blocks)
                * (n2 - blocks)
            )
            / denominator
        )

        projection = np.maximum(
            projection,
            0,
        )

        # ----------------------------------------------
        # Blend projection + trend
        # ----------------------------------------------

        prediction = (
            w * projection
            + (1 - w) * trend
        )

        prediction_valid = prediction[
            mask
        ]

        if not np.all(
            np.isfinite(
                prediction_valid
            )
        ):
            return 1e9

        # ----------------------------------------------
        # Errors
        # ----------------------------------------------

        block_error = (
            np.mean(
                np.abs(
                    actual_valid
                    - prediction_valid
                )
            )
            / actual_peak
        )

        peak_error = (
            abs(
                actual_peak
                - np.max(
                    prediction_valid
                )
            )
            / actual_peak
        )

        prediction_energy = np.sum(
            prediction_valid
        )

        if prediction_energy <= 0:
            return 1e9

        energy_error = (
            abs(
                actual_energy
                - prediction_energy
            )
            / actual_energy
        )

        # --------------------------------------------------
        # Final score
        # --------------------------------------------------

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    # ------------------------------------------------------
    # Differential evolution
    # ------------------------------------------------------

    result = differential_evolution(
        objective,

        bounds=[
            (0.0, 1.0),   # Weight
            (5, 40),      # n1
            (55, 95),     # n2
            (35, 45),     # peak block
        ],

        strategy="best1bin",

        maxiter=OPT_MAXITER,

        popsize=OPT_POPSIZE,

        tol=0.01,

        mutation=(0.5, 1.0),

        recombination=0.7,

        seed=42,

        polish=False,

        workers=1,

        updating="immediate",
    )

    return {
        "w": float(
            np.clip(
                result.x[0],
                0,
                1,
            )
        ),

        "n1": int(
            round(result.x[1])
        ),

        "n2": int(
            round(result.x[2])
        ),

        "b": int(
            round(result.x[3])
        ),
    }


# ==========================================================
# FINAL CURVE
# ==========================================================

def calculate_rt_curve(
    actual,
    trend,
    w,
    n1,
    n2,
    b,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    trend = np.asarray(
        trend,
        dtype=float,
    )

    blocks = np.arange(
        1,
        BLOCKS + 1,
        dtype=float,
    )

    # ------------------------------------------------------
    # Validate parameters
    # ------------------------------------------------------

    if not (
        n1 < b < n2
    ):
        raise ValueError(
            "Parameter condition must be: "
            "n1 < Peak Block < n2."
        )

    denominator = (
        (n1 - b)
        * (n2 - b)
    )

    if denominator == 0:
        raise ValueError(
            "Invalid parameter configuration."
        )

    # ------------------------------------------------------
    # Find peak power
    # ------------------------------------------------------

    peak_mask = np.isin(
        blocks,
        [
            b - 1,
            b,
            b + 1,
        ],
    )

    peak_values = actual[
        peak_mask
    ]

    if len(peak_values) == 0:
        raise ValueError(
            "Unable to determine peak generation."
        )

    peak_power = np.mean(
        peak_values
    )

    # ------------------------------------------------------
    # Parabolic projection
    # ------------------------------------------------------

    projection = (
        peak_power
        * (
            (n1 - blocks)
            * (n2 - blocks)
        )
        / denominator
    )

    projection = np.maximum(
        projection,
        0,
    )

    # ------------------------------------------------------
    # RT CORRECTION
    #
    # w = projection contribution
    # 1-w = trend contribution
    # ------------------------------------------------------

    corrected = (
        w * projection
        + (1 - w) * trend
    )

    corrected = np.maximum(
        corrected,
        0,
    )

    return corrected


# ==========================================================
# TIME BLOCK LOOKUP
# ==========================================================

@st.cache_data(
    show_spinner=False,
)
def time_block_lookup(
    n1,
    n2,
):

    start = datetime.strptime(
        "00:00",
        "%H:%M",
    )

    time_blocks = [
        (
            start
            + timedelta(
                minutes=15 * i
            )
        ).strftime("%H:%M")
        + " - "
        + (
            start
            + timedelta(
                minutes=15 * (i + 1)
            )
        ).strftime("%H:%M")

        for i in range(BLOCKS)
    ]

    rows = [
        (
            "Parabolic Power Generation Starting Block",
            n1,
            (
                time_blocks[n1 - 1]
                if 0 < n1 <= BLOCKS
                else "—"
            ),
        ),

        (
            "Parabolic Power Generation Ending Block",
            n2,
            (
                time_blocks[n2 - 1]
                if 0 < n2 <= BLOCKS
                else "—"
            ),
        ),

        (
            "Actual Generation Available Block (Lower Limit)",
            n1 + 3,
            (
                time_blocks[n1 + 2]
                if n1 + 3 <= BLOCKS
                else "—"
            ),
        ),

        (
            "Actual Generation Effective Block (Upper Limit)",
            n2 - 3,
            (
                time_blocks[n2 - 4]
                if n2 - 3 >= 1
                else "—"
            ),
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Parameter",
            "Block",
            "Time Block",
        ],
    )


# ==========================================================
# PAGE HEADER
# ==========================================================

st.title(
    "Guruji ne kaha tha RT Correct kardo bhayi 🛐!!"
)

st.caption(
    "Adjust Actual and Trend values, optimize the RT curve, "
    "then manually fine-tune the correction parameters."
)


# ==========================================================
# INPUT DATA
# ==========================================================

st.subheader(
    "Input Data"
)

input_df = pd.DataFrame(
    {
        "Actual": st.session_state.rt_actual,
        "Trend": st.session_state.rt_trend,
    }
)


# ==========================================================
# DATA EDITOR FORM
# ==========================================================

with st.form(
    "rt_data_form",
    clear_on_submit=False,
):

    edited_df = st.data_editor(
        input_df,

        use_container_width=True,

        hide_index=True,

        num_rows="fixed",

        height=420,

        key="rt_editor",

        column_config={
            "Actual": st.column_config.NumberColumn(
                "Actual",
                format="%.2f",
                step=0.01,
            ),

            "Trend": st.column_config.NumberColumn(
                "Trend",
                format="%.2f",
                step=0.01,
            ),
        },
    )

    data_submit = st.form_submit_button(
        "💾 Apply Data Changes",
        use_container_width=True,
    )


# ==========================================================
# APPLY DATA CHANGES
# ==========================================================

if data_submit:

    edited_df = (
        edited_df
        .iloc[:BLOCKS]
        .reset_index(drop=True)
    )

    edited_df["Actual"] = pd.to_numeric(
        edited_df["Actual"],
        errors="coerce",
    ).fillna(0)

    edited_df["Trend"] = pd.to_numeric(
        edited_df["Trend"],
        errors="coerce",
    ).fillna(0)

    new_actual = (
        edited_df["Actual"]
        .tolist()
    )

    new_trend = (
        edited_df["Trend"]
        .tolist()
    )

    changed = (
        new_actual
        != st.session_state.rt_actual
        or
        new_trend
        != st.session_state.rt_trend
    )

    if changed:

        st.session_state.rt_actual = (
            new_actual
        )

        st.session_state.rt_trend = (
            new_trend
        )

        # Existing optimization is no longer valid
        st.session_state.rt_optimized = False

        st.toast(
            "Input data updated.",
            icon="✅",
        )

        st.rerun()

    else:

        st.info(
            "No data changes detected."
        )


# ==========================================================
# CURRENT DATA
# ==========================================================

actual = np.asarray(
    st.session_state.rt_actual,
    dtype=float,
)

trend = np.asarray(
    st.session_state.rt_trend,
    dtype=float,
)


# ==========================================================
# OPTIMIZATION
# ==========================================================

st.subheader(
    "RT Optimization"
)

if st.button(
    "🚀 Optimize RT Parameters",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Optimizing RT curve... "
        "This may take a few seconds."
    ):

        try:

            params = cached_rt_optimize(
                tuple(actual),
                tuple(trend),
            )

            st.session_state.rt_params = (
                params
            )

            st.session_state.rt_optimized = (
                True
            )

            st.success(
                "Optimization completed successfully."
            )

        except Exception as e:

            st.error(
                f"Optimization failed: {e}"
            )


# ==========================================================
# PARAMETERS
# ==========================================================

st.subheader(
    "Parameters"
)

p = st.session_state.rt_params


# ----------------------------------------------------------
# Parameter form
# ----------------------------------------------------------

with st.form(
    "rt_parameter_form",
    clear_on_submit=False,
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        new_w = st.number_input(
            "Weight",
            min_value=0.0,
            max_value=1.0,
            value=float(
                p["w"]
            ),
            step=0.01,
            format="%.2f",
        )

    with col2:

        new_n1 = st.number_input(
            "n1",
            min_value=1,
            max_value=95,
            value=int(
                p["n1"]
            ),
            step=1,
        )

    with col3:

        new_n2 = st.number_input(
            "n2",
            min_value=2,
            max_value=96,
            value=int(
                p["n2"]
            ),
            step=1,
        )

    with col4:

        new_b = st.number_input(
            "Peak Block",
            min_value=2,
            max_value=95,
            value=int(
                p["b"]
            ),
            step=1,
        )

    parameter_submit = st.form_submit_button(
        "🔧 Apply Parameter Changes",
        use_container_width=True,
    )


# ==========================================================
# APPLY PARAMETER CHANGES
# ==========================================================

if parameter_submit:

    if not (
        new_n1
        < new_b
        < new_n2
    ):

        st.error(
            "Parameter condition must be: "
            "**n1 < Peak Block < n2**."
        )

    else:

        st.session_state.rt_params = {
            "w": float(new_w),
            "n1": int(new_n1),
            "n2": int(new_n2),
            "b": int(new_b),
        }

        st.toast(
            "Parameters updated.",
            icon="🔧",
        )

        st.rerun()


# ==========================================================
# CURRENT PARAMETERS
# ==========================================================

p = st.session_state.rt_params


# ==========================================================
# PARAMETER SUMMARY
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Weight",
    f"{p['w']:.2f}",
)

c2.metric(
    "n1",
    p["n1"],
)

c3.metric(
    "Peak Block",
    p["b"],
)

c4.metric(
    "n2",
    p["n2"],
)


# ==========================================================
# CALCULATE FINAL CURVE
# ==========================================================

try:

    projection = calculate_rt_curve(
        actual=actual,
        trend=trend,
        w=p["w"],
        n1=p["n1"],
        n2=p["n2"],
        b=p["b"],
    )

except Exception as e:

    st.error(
        f"Unable to calculate RT curve: {e}"
    )

    st.stop()


# ==========================================================
# IMPORTANT TIME BLOCKS
# ==========================================================

with st.expander(
    "📅 Important Time Blocks"
):

    st.dataframe(
        time_block_lookup(
            p["n1"],
            p["n2"],
        ),

        use_container_width=True,

        hide_index=True,
    )


# ==========================================================
# ERROR METRICS
# ==========================================================

mask = (
    np.isfinite(actual)
    & np.isfinite(projection)
    & (actual > 0.5)
)

if np.any(mask):

    actual_valid = actual[mask]
    projection_valid = projection[mask]

    actual_peak = np.max(
        actual_valid
    )

    actual_energy = np.sum(
        actual_valid
    )

    block_error = (
        np.mean(
            np.abs(
                actual_valid
                - projection_valid
            )
        )
        / actual_peak
        * 100
    )

    peak_error = (
        abs(
            actual_peak
            - np.max(
                projection_valid
            )
        )
        / actual_peak
        * 100
    )

    energy_error = (
        abs(
            actual_energy
            - np.sum(
                projection_valid
            )
        )
        / actual_energy
        * 100
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Block Error",
        f"{block_error:.2f}%",
    )

    c2.metric(
        "Peak Error",
        f"{peak_error:.2f}%",
    )

    c3.metric(
        "Energy Error",
        f"{energy_error:.2f}%",
    )


# ==========================================================
# CHART
# ==========================================================

st.subheader(
    "RT Corrected Curve"
)

blocks = np.arange(
    1,
    BLOCKS + 1,
)


fig = go.Figure()


# ----------------------------------------------------------
# Projection
# ----------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=projection,
        name="RT Corrected",
        mode="lines",
        line=dict(
            color="#00c6ff",
            width=3,
        ),
        hovertemplate=(
            "<b>RT Corrected</b>"
            "<br>Block: %{x}"
            "<br>Power: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)


# ----------------------------------------------------------
# Actual
# ----------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=actual,
        name="Actual",
        mode="lines",
        line=dict(
            color="#ef4444",
            width=3,
        ),
        hovertemplate=(
            "<b>Actual</b>"
            "<br>Block: %{x}"
            "<br>Power: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)


# ----------------------------------------------------------
# Trend
# ----------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=trend,
        name="Trend",
        mode="lines",
        line=dict(
            color="#9ca3af",
            width=2,
            dash="dot",
        ),
        hovertemplate=(
            "<b>Trend</b>"
            "<br>Block: %{x}"
            "<br>Power: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)


fig.update_layout(
    height=520,

    template="streamlit",

    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,

        # Click legend = isolate
        itemclick="toggleothers",

        # Double click = normal toggle
        itemdoubleclick="toggle",
    ),

    xaxis=dict(
        title="15 Minute Block",
        fixedrange=True,
    ),

    yaxis=dict(
        title="Power",
        fixedrange=True,
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),

    uirevision="rt_chart",
)


st.plotly_chart(
    fig,
    use_container_width=True,
    key="rt_correction_chart",
)
