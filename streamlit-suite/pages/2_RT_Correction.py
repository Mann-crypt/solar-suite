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

st.sidebar.markdown("""
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
""", unsafe_allow_html=True)

st.sidebar.divider()


# ==========================================================
# SESSION STATE
# ==========================================================

if "rt_actual" not in st.session_state:
    st.session_state.rt_actual = [0.0] * 96

if "rt_trend" not in st.session_state:
    st.session_state.rt_trend = [0.0] * 96

if "rt_params" not in st.session_state:
    st.session_state.rt_params = {
        "w": 0.30,
        "n1": 29,
        "n2": 72,
        "b": 39,
    }


# ==========================================================
# 1. RTF OPTIMIZATION
# ==========================================================
#
# IMPORTANT:
# This is RTF optimization.
#
# It is NOT the projection calculation.
#
# Weight, Trend, etc. belong here and are not used
# inside the parabolic projection function below.
# ==========================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def cached_rtf_optimize(
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
        97,
        dtype=float,
    )

    mask = actual > 5

    if not np.any(mask):
        return {
            "w": 0.30,
            "n1": 29,
            "n2": 72,
            "b": 39,
        }

    def objective(x):

        w, n1, n2, b = x

        n1 = int(round(n1))
        n2 = int(round(n2))
        b = int(round(b))

        # ----------------------------------------------
        # RTF parameter validation
        # ----------------------------------------------

        if not (
            1 <= n1 < b < n2 <= 96
        ):
            return 1e6

        # ----------------------------------------------
        # Peak value around peak block
        # ----------------------------------------------

        peak_mask = np.isin(
            blocks,
            [b - 1, b, b + 1],
        )

        if not np.any(peak_mask):
            return 1e6

        peak = actual[
            peak_mask
        ].mean()

        # ----------------------------------------------
        # RTF / projection used by optimization
        # ----------------------------------------------

        denominator = (
            (n1 - b)
            * (n2 - b)
        )

        if denominator == 0:
            return 1e6

        calc = peak * (
            (
                (n1 - blocks)
                * (n2 - blocks)
            )
            / denominator
        )

        proj = np.maximum(
            calc,
            0,
        )

        # ----------------------------------------------
        # RTF correction
        #
        # This remains separate from the projection
        # function used for normal UI interaction.
        # ----------------------------------------------

        prediction = np.where(
            blocks > b,
            w * proj
            + (1 - w) * trend,
            proj,
        )

        pred = prediction[mask]
        act = actual[mask]

        if len(act) == 0:
            return 1e6

        actual_max = act.max()
        actual_sum = act.sum()

        if actual_max <= 0:
            return 1e6

        # ----------------------------------------------
        # Error components
        # ----------------------------------------------

        block_error = (
            np.mean(
                np.abs(
                    act - pred
                )
            )
            / actual_max
        )

        peak_error = (
            abs(
                actual_max
                - pred.max()
            )
            / actual_max
        )

        energy_error = (
            abs(
                actual_sum
                - pred.sum()
            )
            / actual_sum
        )

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return score

    # ----------------------------------------------
    # Differential Evolution
    # ----------------------------------------------

    result = differential_evolution(
        objective,

        bounds=[
            (0.3, 0.3),
            (5, 40),
            (55, 95),
            (35, 40),
        ],

        popsize=12,
        maxiter=50,
        polish=True,
        seed=42,
        workers=1,
    )

    w, n1, n2, b = result.x

    return {
        "w": float(w),
        "n1": int(round(n1)),
        "n2": int(round(n2)),
        "b": int(round(b)),
    }


# ==========================================================
# 2. PARABOLIC PROJECTION
# ==========================================================
#
# IMPORTANT:
#
# This function is ONLY projection.
#
# No Weight.
# No Trend.
# No RTF blending.
#
# Projection depends only on:
#
# Actual
# n1
# n2
# Peak Block
# ==========================================================

@st.cache_data(
    show_spinner=False,
    max_entries=50,
)
def cached_projection(
    actual_tuple,
    n1,
    n2,
    b,
):

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    blocks = np.arange(
        1,
        97,
        dtype=float,
    )

    # ----------------------------------------------
    # Peak block
    # ----------------------------------------------

    peak_mask = np.isin(
        blocks,
        [b - 1, b, b + 1],
    )

    if not np.any(peak_mask):
        return actual.tolist()

    peak = actual[
        peak_mask
    ].mean()

    # ----------------------------------------------
    # Parabolic denominator
    # ----------------------------------------------

    denominator = (
        (n1 - b)
        * (n2 - b)
    )

    if denominator == 0:
        return actual.tolist()

    # ----------------------------------------------
    # Parabolic projection
    # ----------------------------------------------

    calc = peak * (
        (
            (n1 - blocks)
            * (n2 - blocks)
        )
        / denominator
    )

    projection = np.maximum(
        calc,
        0,
    )

    return projection.tolist()


# ==========================================================
# 3. RTF CURVE
# ==========================================================
#
# This is kept separate from projection.
#
# It uses:
#   Projection
#   Trend
#   Weight
#
# The projection itself is NEVER modified.
# ==========================================================

@st.cache_data(
    show_spinner=False,
    max_entries=50,
)
def cached_rtf_curve(
    projection_tuple,
    trend_tuple,
    w,
    b,
):

    projection = np.asarray(
        projection_tuple,
        dtype=float,
    )

    trend = np.asarray(
        trend_tuple,
        dtype=float,
    )

    blocks = np.arange(
        1,
        97,
        dtype=float,
    )

    # ----------------------------------------------
    # RTF correction
    # ----------------------------------------------

    rtf_curve = np.where(
        blocks > b,
        w * projection
        + (1 - w) * trend,
        projection,
    )

    return rtf_curve.tolist()


# ==========================================================
# 4. TIME BLOCK LOOKUP
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
        for i in range(96)
    ]

    rows = [
        (
            "Parabolic Power Generation Starting Block",
            n1,
            time_blocks[n1 - 1]
            if 1 <= n1 <= 96
            else "—",
        ),

        (
            "Parabolic Power Generation Ending Block",
            n2,
            time_blocks[n2 - 1]
            if 1 <= n2 <= 96
            else "—",
        ),

        (
            "Actual Generation Available Block (Lower Limit)",
            n1 + 3,
            time_blocks[n1 + 2]
            if n1 + 3 <= 96
            else "—",
        ),

        (
            "Actual Generation Effective Block (Upper Limit)",
            n2 - 3,
            time_blocks[n2 - 4]
            if n2 - 3 >= 1
            else "—",
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


# ==========================================================
# DATA INPUT
# ==========================================================

input_df = pd.DataFrame({
    "Actual": st.session_state.rt_actual,
    "Trend": st.session_state.rt_trend,
})


# ==========================================================
# DATA EDITOR
# ==========================================================

edited_df = st.data_editor(
    input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    height=420,
    key="rt_editor",
)


# ==========================================================
# CLEAN EDITED DATA
# ==========================================================

edited_df = (
    edited_df
    .iloc[:96]
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


actual_list = (
    edited_df["Actual"]
    .round(6)
    .tolist()
)

trend_list = (
    edited_df["Trend"]
    .round(6)
    .tolist()
)


# ==========================================================
# LIVE DATA STATE
# ==========================================================
#
# No Apply button.
# No st.rerun().
#
# Streamlit automatically reruns when the editor changes.
# ==========================================================

st.session_state.rt_actual = actual_list
st.session_state.rt_trend = trend_list


# ==========================================================
# OPTIMIZE RTF
# ==========================================================

if st.button(
    "🚀 Dabaiye na!!",
    use_container_width=True,
    type="primary",
):

    with st.spinner(
        "Optimizing RTF parameters..."
    ):

        optimized = cached_rtf_optimize(
            tuple(actual_list),
            tuple(trend_list),
        )

    st.session_state.rt_params = optimized

    # ----------------------------------------------
    # Store widget values
    # ----------------------------------------------

    st.session_state.rt_w = (
        optimized["w"]
    )

    st.session_state.rt_n1 = (
        optimized["n1"]
    )

    st.session_state.rt_n2 = (
        optimized["n2"]
    )

    st.session_state.rt_b = (
        optimized["b"]
    )


# ==========================================================
# PARAMETERS
# ==========================================================

st.subheader(
    "Parameters"
)


p = st.session_state.rt_params


# ==========================================================
# PARAMETER INPUTS
# ==========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    w = st.number_input(
        "Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(
            p["w"]
        ),
        step=0.01,
        key="rt_w",
    )


with col2:

    n1 = st.number_input(
        "n1",
        min_value=1,
        max_value=95,
        value=int(
            p["n1"]
        ),
        step=1,
        key="rt_n1",
    )


with col3:

    n2 = st.number_input(
        "n2",
        min_value=2,
        max_value=96,
        value=int(
            p["n2"]
        ),
        step=1,
        key="rt_n2",
    )


with col4:

    b = st.number_input(
        "Peak Block",
        min_value=2,
        max_value=95,
        value=int(
            p["b"]
        ),
        step=1,
        key="rt_b",
    )


# ==========================================================
# LIVE PARAMETER STATE
# ==========================================================

st.session_state.rt_params = {
    "w": float(w),
    "n1": int(n1),
    "n2": int(n2),
    "b": int(b),
}


# ==========================================================
# PARAMETER VALIDATION
# ==========================================================

parameters_valid = (
    n1 < b < n2
)


if not parameters_valid:

    st.warning(
        "Required relationship: "
        "**n1 < Peak Block < n2**"
    )

    st.stop()


# ==========================================================
# PROJECTION
# ==========================================================
#
# ONLY n1 / n2 / b affect projection.
#
# Weight and Trend do NOT enter here.
# ==========================================================

projection = cached_projection(
    tuple(actual_list),
    int(n1),
    int(n2),
    int(b),
)


# ==========================================================
# RTF CURVE
# ==========================================================
#
# Separate calculation.
# ==========================================================

rtf_curve = cached_rtf_curve(
    tuple(projection),
    tuple(trend_list),
    float(w),
    int(b),
)


# ==========================================================
# TIME BLOCK INFORMATION
# ==========================================================

with st.expander(
    "📅 Important Time Blocks"
):

    st.dataframe(
        time_block_lookup(
            int(n1),
            int(n2),
        ),
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# CHART
# ==========================================================

blocks = list(
    range(1, 97)
)


fig = go.Figure()


# ==========================================================
# PROJECTION
# ==========================================================

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
        hovertemplate=(
            "Block: %{x}<br>"
            "Projection: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)


# ==========================================================
# ACTUAL
# ==========================================================

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=actual_list,
        name="Actual",
        mode="lines",
        line=dict(
            color="#ef4444",
            width=3,
        ),
        hovertemplate=(
            "Block: %{x}<br>"
            "Actual: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)


# ==========================================================
# RTF
# ==========================================================

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=rtf_curve,
        name="RTF",
        mode="lines",
        line=dict(
            color="#22c55e",
            width=3,
            dash="dash",
        ),
        hovertemplate=(
            "Block: %{x}<br>"
            "RTF: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)


# ==========================================================
# CHART LAYOUT
# ==========================================================

fig.update_layout(
    height=550,

    template="streamlit",

    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),

    xaxis=dict(
        title="15 Minute Block",
        fixedrange=True,
    ),

    yaxis=dict(
        title="Power",
        fixedrange=True,
    ),
)


# ==========================================================
# DISPLAY
# ==========================================================

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
        "scrollZoom": False,
    },
)
