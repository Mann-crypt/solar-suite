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
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 10px 16px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "rt_input" not in st.session_state:

    st.session_state.rt_input = pd.DataFrame(
        {
            "Actual": np.zeros(96),
            "Trend": np.zeros(96),
        }
    )


if "rt_params" not in st.session_state:

    st.session_state.rt_params = {
        "w": 0.30,
        "n1": 29,
        "n2": 72,
        "b": 39,
    }


if "rt_optimized" not in st.session_state:

    st.session_state.rt_optimized = False


# ==========================================================
# HELPER: CREATE TIME BLOCKS
# ==========================================================

def create_time_blocks():

    start = datetime.strptime(
        "00:00",
        "%H:%M",
    )

    return [
        (
            f"{(start + timedelta(minutes=15 * i)).strftime('%H:%M')}"
            f" - "
            f"{(start + timedelta(minutes=15 * (i + 1))).strftime('%H:%M')}"
        )
        for i in range(96)
    ]


# ==========================================================
# HELPER: CALCULATE PROJECTION
# ==========================================================

def calculate_projection(
    actual,
    trend,
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
        97,
        dtype=float,
    )

    # ------------------------------------------------------
    # Peak reference
    # ------------------------------------------------------

    peak_mask = np.isin(
        blocks,
        [b - 1, b, b + 1],
    )

    p = actual[peak_mask].mean()

    # ------------------------------------------------------
    # Parabolic projection
    # ------------------------------------------------------

    denominator = (
        (n1 - b)
        * (n2 - b)
    )

    if denominator == 0:

        return np.zeros(96)

    calc = p * (
        (
            (n1 - blocks)
            * (n2 - blocks)
        )
        / denominator
    )

    projection = np.where(
        calc < 0,
        0,
        calc,
    )

    return projection


# ==========================================================
# HELPER: CALCULATE PREDICTION
# ==========================================================

def calculate_prediction(
    projection,
    trend,
    w,
    b,
):

    blocks = np.arange(
        1,
        97,
        dtype=float,
    )

    prediction = np.where(
        blocks > b,
        w * projection
        + (1 - w) * trend,
        trend,
    )

    return prediction


# ==========================================================
# OBJECTIVE FUNCTION
# ==========================================================

def make_objective(
    actual,
    trend,
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
        97,
        dtype=float,
    )

    mask = actual > 0.5

    def objective(x):

        w, n1, n2, b = x

        n1 = int(round(n1))
        n2 = int(round(n2))
        b = int(round(b))

        # --------------------------------------------------
        # Parameter relationship
        # --------------------------------------------------

        if not (
            n1 < b < n2
        ):

            return 1e6

        # --------------------------------------------------
        # Peak reference
        # --------------------------------------------------

        peak_mask = np.isin(
            blocks,
            [b - 1, b, b + 1],
        )

        p = actual[peak_mask].mean()

        if np.isnan(p):

            return 1e6

        # --------------------------------------------------
        # Projection
        # --------------------------------------------------

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
            / denominator
        )

        projection = np.where(
            calc < 0,
            0,
            calc,
        )

        # --------------------------------------------------
        # Prediction
        #
        # IMPORTANT:
        # This is kept exactly as your reference.
        # --------------------------------------------------

        prediction = np.where(
            blocks > b,
            w * projection
            + (1 - w) * trend,
            trend,
        )

        # --------------------------------------------------
        # Metrics use PROJECTION
        # exactly as reference code
        # --------------------------------------------------

        pred = projection[mask]
        act = actual[mask]

        if len(act) == 0:

            return 1e6

        if act.max() == 0:

            return 1e6

        if act.sum() == 0:

            return 1e6

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

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return score

    return objective


# ==========================================================
# PAGE HEADER
# ==========================================================

st.title(
    "Guruji ne kaha tha RT Correct kardo bhyii🛐!!"
)

st.caption(
    "Parabolic RT projection with trend-weighted correction."
)


# ==========================================================
# INPUT DATA
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
    height=360,
    column_config={
        "Actual": st.column_config.NumberColumn(
            "Actual",
            format="%.2f",
        ),
        "Trend": st.column_config.NumberColumn(
            "Trend",
            format="%.2f",
        ),
    },
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
).fillna(0.0)


edited_df["Trend"] = pd.to_numeric(
    edited_df["Trend"],
    errors="coerce",
).fillna(0.0)


# ==========================================================
# DETECT DATA CHANGES
# ==========================================================

old_df = st.session_state.rt_input

changed = not edited_df.equals(
    old_df.reset_index(drop=True)
)


if changed:

    changed_cells = (
        edited_df
        .ne(old_df.reset_index(drop=True))
        .sum()
        .sum()
    )

    st.toast(
        f"✨ {int(changed_cells)} cells updated!",
        icon="✅",
    )

    st.session_state.rt_input = (
        edited_df.copy()
    )


df = edited_df.copy()


# ==========================================================
# TIME BLOCKS
# ==========================================================

df["Time-Blocks"] = (
    create_time_blocks()
)

df["Blocks"] = np.arange(
    1,
    97,
)


# ==========================================================
# ARRAYS
# ==========================================================

actual = df[
    "Actual"
].to_numpy(
    dtype=float
)

trend = df[
    "Trend"
].to_numpy(
    dtype=float
)

blocks = df[
    "Blocks"
].to_numpy(
    dtype=float
)


# ==========================================================
# DATA SUMMARY
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
    int(np.sum(actual > 0.5)),
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

    if np.max(actual) <= 0:

        st.error(
            "Actual generation data is empty."
        )

    else:

        objective = make_objective(
            actual,
            trend,
        )

        with st.spinner(
            "Optimizing RT parameters..."
        ):

            result = differential_evolution(
                objective,
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

        optimized_w = float(
            result.x[0]
        )

        optimized_n1 = int(
            round(result.x[1])
        )

        optimized_n2 = int(
            round(result.x[2])
        )

        optimized_b = int(
            round(result.x[3])
        )

        st.session_state.rt_params = {
            "w": optimized_w,
            "n1": optimized_n1,
            "n2": optimized_n2,
            "b": optimized_b,
        }

        st.session_state.rt_optimized = True

        st.success(
            "Optimization completed successfully."
        )

        st.rerun()


# ==========================================================
# PARAMETERS
# ==========================================================

st.subheader(
    "Parameters"
)

p = st.session_state.rt_params


param1, param2 = st.columns(2)


with param1:

    w = st.number_input(
        "Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(p["w"]),
        step=0.01,
        format="%.2f",
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


with param2:

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
# PARAMETER VALIDATION
# ==========================================================

valid_parameters = (
    n1 < b < n2
)


if not valid_parameters:

    st.warning(
        "Parameter relationship must satisfy "
        "**n1 < Peak Block < n2**."
    )


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
# FINAL CALCULATION
# ==========================================================

if valid_parameters:

    projection = calculate_projection(
        actual=actual,
        trend=trend,
        n1=n1,
        n2=n2,
        b=b,
    )

    prediction = calculate_prediction(
        projection=projection,
        trend=trend,
        w=w,
        b=b,
    )

else:

    projection = np.zeros(96)

    prediction = trend.copy()


df["Projection"] = projection
df["Prediction"] = prediction


# ==========================================================
# IMPORTANT TIME BLOCKS
# ==========================================================

lookup_blocks = [
    n1,
    n2,
    n1 + 3,
    n2 - 3,
]


lookup_names = [
    "Parabolic Power Generation Starting Block",
    "Parabolic Power Generation Ending Block",
    "Actual Generation Available Block (Lower Limit)",
    "Actual Generation Effective Block (Upper Limit)",
]


lookup_df = pd.DataFrame(
    {
        "Parameter": lookup_names,
        "Block": lookup_blocks,
    }
)


time_map = (
    df
    .set_index("Blocks")["Time-Blocks"]
)


lookup_df["Time Block"] = (
    lookup_df["Block"]
    .map(time_map)
    .fillna("—")
)


with st.expander(
    "📅 Important Time Blocks"
):

    st.dataframe(
        lookup_df,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# GRAPH
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
        x=df["Blocks"],
        y=df["Projection"],
        name="Projection",
        mode="lines",
        line=dict(
            color="#00c6ff",
            width=3,
        ),
    )
)


# ----------------------------------------------------------
# Actual
# ----------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=df["Blocks"],
        y=df["Actual"],
        name="Actual",
        mode="lines",
        line=dict(
            color="#ef4444",
            width=3,
        ),
    )
)


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=df["Blocks"],
        y=df["Prediction"],
        name="Prediction",
        mode="lines",
        line=dict(
            color="#22c55e",
            width=2.5,
            dash="dash",
        ),
    )
)


# ----------------------------------------------------------
# Peak block
# ----------------------------------------------------------

if valid_parameters:

    fig.add_vline(
        x=b,
        line_width=1,
        line_dash="dot",
        annotation_text=f"Peak {b}",
        annotation_position="top",
    )


fig.update_layout(
    height=550,
    template="streamlit",
    hovermode="x unified",
    margin=dict(
        l=20,
        r=20,
        t=60,
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
    yaxis_title="Power",
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

with st.expander(
    "📊 View RT Calculation Data"
):

    output_df = df[
        [
            "Blocks",
            "Time-Blocks",
            "Actual",
            "Trend",
            "Projection",
            "Prediction",
        ]
    ].copy()

    st.dataframe(
        output_df,
        use_container_width=True,
        hide_index=True,
    )
