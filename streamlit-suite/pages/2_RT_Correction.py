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
    page_title="RT Correction - Solar Suite",
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
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 10px 15px;
    }

    .param-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 12px 15px;
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# CONSTANTS
# ==========================================================

N_BLOCKS = 96


# ==========================================================
# TIME BLOCKS
# ==========================================================

def create_time_blocks():

    start = datetime.strptime(
        "00:00",
        "%H:%M",
    )

    return [
        (
            start
            + timedelta(minutes=15 * i)
        ).strftime("%H:%M")
        + " - "
        + (
            start
            + timedelta(minutes=15 * (i + 1))
        ).strftime("%H:%M")
        for i in range(N_BLOCKS)
    ]


TIME_BLOCKS = create_time_blocks()


# ==========================================================
# EXACT RT CALCULATION
# ==========================================================

def calculate_rt(
    actual,
    trend,
    w,
    n1,
    n2,
    b,
):
    """
    Exact RT / parabolic calculation from
    the reference code.

    IMPORTANT:
    Projection and Prediction are intentionally
    calculated separately.
    """

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
        N_BLOCKS + 1,
        dtype=float,
    )

    # ------------------------------------------------------
    # Peak value
    # ------------------------------------------------------

    peak_mask = np.isin(
        blocks,
        [b - 1, b, b + 1],
    )

    p = actual[peak_mask].mean()

    # ------------------------------------------------------
    # Parabolic calculation
    # ------------------------------------------------------

    denominator = (
        (n1 - b)
        * (n2 - b)
    )

    # Invalid parameter combination
    if denominator == 0:

        projection = np.zeros(
            N_BLOCKS,
            dtype=float,
        )

    else:

        calc = p * (
            (
                (n1 - blocks)
                * (n2 - blocks)
            )
            /
            denominator
        )

        # EXACTLY as reference
        projection = np.where(
            calc < 0,
            0,
            calc,
        )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    # EXACTLY as reference
    prediction = np.where(
        blocks > b,
        w * projection
        + (1 - w) * trend,
        trend,
    )

    return (
        projection,
        prediction,
    )


# ==========================================================
# OBJECTIVE
# ==========================================================

def rt_objective(
    x,
    actual,
    trend,
):
    """
    Exact optimization objective from reference code.

    NOTE:
    The objective compares Projection against Actual.
    Prediction is NOT used here.
    """

    w, n1, n2, b = x

    n1 = int(round(n1))
    n2 = int(round(n2))
    b = int(round(b))

    # ------------------------------------------------------
    # Parameter relationship
    # ------------------------------------------------------

    if not (
        n1 < b < n2
    ):
        return 1e6

    blocks = np.arange(
        1,
        N_BLOCKS + 1,
        dtype=float,
    )

    # ------------------------------------------------------
    # Peak value
    # ------------------------------------------------------

    peak_mask = np.isin(
        blocks,
        [b - 1, b, b + 1],
    )

    p = actual[peak_mask].mean()

    # ------------------------------------------------------
    # Projection
    # ------------------------------------------------------

    denominator = (
        (n1 - b)
        * (n2 - b)
    )

    if denominator == 0:

        return 1e6

    calc = p * (
        (
            (n1 - blocks)
            * (n2 - blocks)
        )
        /
        denominator
    )

    projection = np.where(
        calc < 0,
        0,
        calc,
    )

    # ------------------------------------------------------
    # Daylight mask
    # ------------------------------------------------------

    mask = actual > 0.5

    pred = projection[mask]
    act = actual[mask]

    if len(act) == 0:

        return 1e6

    if act.max() == 0:

        return 1e6

    if act.sum() == 0:

        return 1e6

    # ------------------------------------------------------
    # Errors
    # ------------------------------------------------------

    block_error = (
        np.mean(
            np.abs(
                act - pred
            )
        )
        / act.max()
    )

    peak_error = (
        abs(
            act.max()
            - pred.max()
        )
        / act.max()
    )

    energy_error = (
        abs(
            act.sum()
            - pred.sum()
        )
        / act.sum()
    )

    # ------------------------------------------------------
    # EXACT SCORE
    # ------------------------------------------------------

    return (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )


# ==========================================================
# CACHED OPTIMIZATION
# ==========================================================

@st.cache_data(
    show_spinner=False,
)
def optimize_rt(
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

    result = differential_evolution(
        lambda x: rt_objective(
            x,
            actual,
            trend,
        ),
        bounds=[
            (0.3, 0.3),
            (5, 40),
            (55, 95),
            (35, 40),
        ],
        popsize=20,
        maxiter=100,
        polish=True,
        seed=42,
    )

    w, n1, n2, b = result.x

    return {
        "w": float(w),
        "n1": int(round(n1)),
        "n2": int(round(n2)),
        "b": int(round(b)),
        "score": float(result.fun),
    }


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(
    actual,
    projection,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    projection = np.asarray(
        projection,
        dtype=float,
    )

    mask = actual > 0.5

    act = actual[mask]
    pred = projection[mask]

    if len(act) == 0:

        return {
            "block_error": 0,
            "peak_error": 0,
            "energy_error": 0,
            "score": 0,
        }

    max_actual = act.max()

    if max_actual == 0:

        return {
            "block_error": 0,
            "peak_error": 0,
            "energy_error": 0,
            "score": 0,
        }

    block_error = (
        np.mean(
            np.abs(
                act - pred
            )
        )
        / max_actual
    )

    peak_error = (
        abs(
            act.max()
            - pred.max()
        )
        / max_actual
    )

    if act.sum() == 0:

        energy_error = 0

    else:

        energy_error = (
            abs(
                act.sum()
                - pred.sum()
            )
            / act.sum()
        )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return {
        "block_error": block_error,
        "peak_error": peak_error,
        "energy_error": energy_error,
        "score": score,
    }


# ==========================================================
# SESSION STATE
# ==========================================================

if "rt_input" not in st.session_state:

    st.session_state.rt_input = pd.DataFrame(
        {
            "Actual": np.zeros(N_BLOCKS),
            "Trend": np.zeros(N_BLOCKS),
        }
    )


if "rt_params" not in st.session_state:

    st.session_state.rt_params = {
        "w": 0.3,
        "n1": 29,
        "n2": 72,
        "b": 39,
    }


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "Guruji ne kaha tha RT Correct kardo bhyii 🛐!!"
)

st.caption(
    "RT parabolic projection and trend correction"
)


# ==========================================================
# DATA INPUT
# ==========================================================

st.subheader(
    "Input Data"
)

edited_df = st.data_editor(
    st.session_state.rt_input,
    key="rt_editor",
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    height=420,
)

edited_df = (
    edited_df
    .iloc[:N_BLOCKS]
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


# ----------------------------------------------------------
# Keep input in session state
# ----------------------------------------------------------

st.session_state.rt_input = (
    edited_df.copy()
)


# ==========================================================
# PREPARE DATA
# ==========================================================

actual = (
    edited_df["Actual"]
    .to_numpy(dtype=float)
)

trend = (
    edited_df["Trend"]
    .to_numpy(dtype=float)
)

blocks = np.arange(
    1,
    N_BLOCKS + 1,
)


# ==========================================================
# INPUT SUMMARY
# ==========================================================

m1, m2, m3 = st.columns(3)

m1.metric(
    "Actual Peak",
    f"{actual.max():,.2f}",
)

m2.metric(
    "Trend Peak",
    f"{trend.max():,.2f}",
)

m3.metric(
    "Daylight Blocks",
    int(
        np.sum(
            actual > 0.5
        )
    ),
)


# ==========================================================
# OPTIMIZATION
# ==========================================================

st.subheader(
    "RT Optimization"
)

if st.button(
    "🚀 Dabaiye na!!",
    type="primary",
    use_container_width=True,
):

    if np.all(actual <= 0.5):

        st.warning(
            "Actual data does not contain "
            "enough daylight values for optimization."
        )

    else:

        with st.spinner(
            "Optimizing RT parameters..."
        ):

            optimized = optimize_rt(
                tuple(actual.tolist()),
                tuple(trend.tolist()),
            )

        st.session_state.rt_params = {
            "w": optimized["w"],
            "n1": optimized["n1"],
            "n2": optimized["n2"],
            "b": optimized["b"],
        }

        st.session_state.rt_optimization_score = (
            optimized["score"]
        )

        st.rerun()


# ==========================================================
# PARAMETERS
# ==========================================================

st.subheader(
    "Parameters"
)

p = st.session_state.rt_params

param_col1, param_col2 = st.columns(2)


# ----------------------------------------------------------
# LEFT
# ----------------------------------------------------------

with param_col1:

    w = st.number_input(
        "Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(p["w"]),
        step=0.01,
        key="rt_weight",
    )

    n2 = st.number_input(
        "n2",
        min_value=1,
        max_value=96,
        value=int(p["n2"]),
        step=1,
        key="rt_n2",
    )


# ----------------------------------------------------------
# RIGHT
# ----------------------------------------------------------

with param_col2:

    n1 = st.number_input(
        "n1",
        min_value=1,
        max_value=96,
        value=int(p["n1"]),
        step=1,
        key="rt_n1",
    )

    b = st.number_input(
        "Peak Block",
        min_value=1,
        max_value=96,
        value=int(p["b"]),
        step=1,
        key="rt_peak",
    )


# ==========================================================
# VALIDATE PARAMETERS
# ==========================================================

if not (
    n1 < b < n2
):

    st.warning(
        "Parameter relationship must satisfy: "
        "**n1 < Peak Block < n2**."
    )

    valid_parameters = False

else:

    valid_parameters = True


# ==========================================================
# SAVE CURRENT PARAMETERS
# ==========================================================

st.session_state.rt_params = {
    "w": float(w),
    "n1": int(n1),
    "n2": int(n2),
    "b": int(b),
}


# ==========================================================
# STOP BEFORE CALCULATION IF INVALID
# ==========================================================

if not valid_parameters:

    st.stop()


# ==========================================================
# EXACT FINAL RT CALCULATION
# ==========================================================

projection, prediction = calculate_rt(
    actual=actual,
    trend=trend,
    w=float(w),
    n1=int(n1),
    n2=int(n2),
    b=int(b),
)


# ==========================================================
# METRICS
# ==========================================================

metrics = calculate_metrics(
    actual,
    projection,
)

e1, e2, e3, e4 = st.columns(4)

e1.metric(
    "Block Error",
    f"{metrics['block_error'] * 100:.2f}%",
)

e2.metric(
    "Peak Error",
    f"{metrics['peak_error'] * 100:.2f}%",
)

e3.metric(
    "Energy Error",
    f"{metrics['energy_error'] * 100:.2f}%",
)

e4.metric(
    "RT Score",
    f"{metrics['score'] * 100:.2f}%",
)


# ==========================================================
# IMPORTANT TIME BLOCKS
# ==========================================================

st.subheader(
    "Important Time Blocks"
)

lookup_blocks = [
    int(n1),
    int(n2),
    int(n1 + 3),
    int(n2 - 3),
]

lookup_names = [
    "Parabolic Power Generation Starting Block",
    "Parabolic Power Generation Ending Block",
    "Actual Generation Available Block (Lower Limit)",
    "Actual Generation Effective Block (Upper Limit)",
]

lookup_rows = []

for name, block in zip(
    lookup_names,
    lookup_blocks,
):

    if 1 <= block <= 96:

        time_block = TIME_BLOCKS[
            block - 1
        ]

    else:

        time_block = "—"

    lookup_rows.append(
        {
            "Parameter": name,
            "Block": block,
            "Time Block": time_block,
        }
    )

lookup_df = pd.DataFrame(
    lookup_rows
)

with st.expander(
    "📅 View Important Time Blocks"
):

    st.dataframe(
        lookup_df,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# RT GRAPH
# ==========================================================

st.subheader(
    "RT Correction Curve"
)

fig = go.Figure()


# ----------------------------------------------------------
# Projection
# ----------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=projection,
        name="Projection",
        mode="lines",
        line=dict(
            color="#00c6ff",
            width=3,
        ),
    )
)


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=prediction,
        name="Prediction",
        mode="lines",
        line=dict(
            color="#a855f7",
            width=3,
            dash="dash",
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
            color="#22c55e",
            width=2,
            dash="dot",
        ),
    )
)


# ----------------------------------------------------------
# Peak block
# ----------------------------------------------------------

fig.add_vline(
    x=b,
    line_width=1.5,
    line_dash="dash",
    line_color="#f59e0b",
    annotation_text=f"Peak Block {b}",
    annotation_position="top",
)


fig.update_layout(
    height=550,
    template="streamlit",
    hovermode="x unified",
    margin=dict(
        l=20,
        r=20,
        t=70,
        b=20,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    xaxis_title="15 Minute Block",
    yaxis_title="Generation",
)


st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
    },
)


# ==========================================================
# OUTPUT DATA
# ==========================================================

output_df = pd.DataFrame(
    {
        "Blocks": blocks,
        "Time-Blocks": TIME_BLOCKS,
        "Actual": actual,
        "Trend": trend,
        "Projection": projection,
        "Prediction": prediction,
    }
)


with st.expander(
    "📊 View RT Calculation Data"
):

    st.dataframe(
        output_df,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# OPTIMIZATION INFORMATION
# ==========================================================

if "rt_optimization_score" in st.session_state:

    st.caption(
        "Last optimization score: "
        f"**{st.session_state.rt_optimization_score:.6f}**"
    )
