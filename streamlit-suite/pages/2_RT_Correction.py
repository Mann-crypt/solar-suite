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

if "rt_input" not in st.session_state:

    st.session_state.rt_input = pd.DataFrame({
        "Actual": np.zeros(96),
        "Trend": np.zeros(96)
    })


if "rt_params" not in st.session_state:

    st.session_state.rt_params = {
        "w": 0.3,
        "n1": 29,
        "n2": 72,
        "b": 39
    }


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "Guruji ne kaha tha RT Correct kardo bhyii🛐!!"
)


# ==========================================================
# INPUT DATA
# ==========================================================

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
            format="%.2f"
        ),
        "Trend": st.column_config.NumberColumn(
            "Trend",
            format="%.2f"
        ),
    }
)


edited_df = (
    edited_df
    .iloc[:96]
    .reset_index(drop=True)
)


edited_df["Actual"] = pd.to_numeric(
    edited_df["Actual"],
    errors="coerce"
).fillna(0)


edited_df["Trend"] = pd.to_numeric(
    edited_df["Trend"],
    errors="coerce"
).fillna(0)


# Save immediately
st.session_state.rt_input = edited_df.copy()

df = edited_df.copy()


# ==========================================================
# TIME BLOCKS
# ==========================================================

start = datetime.strptime(
    "00:00",
    "%H:%M"
)


df["Time-Blocks"] = [
    (
        f"{(start + timedelta(minutes=15*i)).strftime('%H:%M')}"
        f" - "
        f"{(start + timedelta(minutes=15*(i+1))).strftime('%H:%M')}"
    )
    for i in range(96)
]


df["Blocks"] = np.arange(
    1,
    97
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


mask = actual > 0.5


# ==========================================================
# OBJECTIVE
# ==========================================================

def objective(x):

    w, n1, n2, b = x

    n1 = int(round(n1))
    n2 = int(round(n2))
    b = int(round(b))

    if not (n1 < b < n2):

        return 1e6


    # ------------------------------------------------------
    # Peak reference
    # ------------------------------------------------------

    p = df.loc[
        df["Blocks"].isin(
            [b-1, b, b+1]
        ),
        "Actual"
    ].mean()


    # ------------------------------------------------------
    # Parabolic projection
    # ------------------------------------------------------

    calc = p * (
        (
            (n1 - blocks)
            *
            (n2 - blocks)
        )
        /
        (
            (n1 - b)
            *
            (n2 - b)
        )
    )


    projection = np.where(
        calc < 0,
        0,
        calc
    )


    # ------------------------------------------------------
    # Prediction
    #
    # Kept exactly as reference.
    # ------------------------------------------------------

    prediction = np.where(
        blocks > b,
        w * projection
        + (1 - w) * trend,
        trend
    )


    # ------------------------------------------------------
    # Evaluation
    #
    # Kept exactly as reference.
    # ------------------------------------------------------

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
        /
        act.max()
    )


    peak_error = (
        abs(
            act.max()
            -
            pred.max()
        )
        /
        act.max()
    )


    energy_error = (
        abs(
            act.sum()
            -
            pred.sum()
        )
        /
        act.sum()
    )


    return (
        0.80 * block_error
        +
        0.10 * peak_error
        +
        0.10 * energy_error
    )


# ==========================================================
# OPTIMIZE
# ==========================================================

if st.button(
    "🚀 Dabaiye na!!",
    use_container_width=True,
    type="primary"
):

    with st.spinner(
        "Optimizing..."
    ):

        result = differential_evolution(
            objective,

            bounds=[
                (0.3, 0.3),
                (5, 40),
                (55, 95),
                (35, 40)
            ],

            popsize=20,
            maxiter=100,
            polish=True,
            seed=42
        )


    w, n1, n2, b = result.x


    st.session_state.rt_params = {
        "w": float(w),
        "n1": int(round(n1)),
        "n2": int(round(n2)),
        "b": int(round(b))
    }


    st.rerun()


# ==========================================================
# PARAMETERS
# ==========================================================

st.subheader(
    "Parameters"
)


p = st.session_state.rt_params


col1, col2 = st.columns(2)


with col1:

    w = st.number_input(
        "Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(p["w"]),
        step=0.01,
        format="%.2f",
        key="rt_weight"
    )


    n2 = st.number_input(
        "n2",
        min_value=1,
        max_value=96,
        value=int(p["n2"]),
        step=1,
        key="rt_n2"
    )


with col2:

    n1 = st.number_input(
        "n1",
        min_value=1,
        max_value=96,
        value=int(p["n1"]),
        step=1,
        key="rt_n1"
    )


    b = st.number_input(
        "Peak Block",
        min_value=1,
        max_value=96,
        value=int(p["b"]),
        step=1,
        key="rt_b"
    )


# Save current values
st.session_state.rt_params = {
    "w": float(w),
    "n1": int(n1),
    "n2": int(n2),
    "b": int(b)
}


# ==========================================================
# VALIDATE PARAMETERS
# ==========================================================

if not (n1 < b < n2):

    st.warning(
        "Please maintain: n1 < Peak Block < n2"
    )

    st.stop()


# ==========================================================
# FINAL CALCULATION
# ==========================================================

p = df.loc[
    df["Blocks"].isin(
        [b-1, b, b+1]
    ),
    "Actual"
].mean()


calc = p * (
    (
        (n1 - blocks)
        *
        (n2 - blocks)
    )
    /
    (
        (n1 - b)
        *
        (n2 - b)
    )
)


projection = np.where(
    calc < 0,
    0,
    calc
)


# Keep prediction calculation from reference
prediction = np.where(
    blocks > b,
    w * projection
    +
    (1 - w) * trend,
    trend
)


df["Projection"] = projection


# ==========================================================
# IMPORTANT TIME BLOCKS
# ==========================================================

lookup_blocks = [
    n1,
    n2,
    n1 + 3,
    n2 - 3
]


lookup_names = [
    "Parabolic Power Generation Starting Block",
    "Parabolic Power Generation Ending Block",
    "Actual Generation Available Block (Lower Limit)",
    "Actual Generation Effective Block (Upper Limit)"
]


lookup_df = pd.DataFrame({
    "Parameter": lookup_names,
    "Block": lookup_blocks
})


lookup_df["Time Block"] = lookup_df[
    "Block"
].map(
    df.set_index(
        "Blocks"
    )["Time-Blocks"]
)


# ==========================================================
# IMPORTANT TIME BLOCKS OUTPUT
# ==========================================================

st.subheader(
    "Important Time Blocks"
)


st.dataframe(
    lookup_df,
    use_container_width=True,
    hide_index=True
)


# ==========================================================
# GRAPH
# ==========================================================

st.subheader(
    "RT Correction Curve"
)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=df["Blocks"],
        y=df["Projection"],
        name="Projection",
        mode="lines",
        line=dict(
            color="#00c6ff",
            width=3
        )
    )
)


fig.add_trace(
    go.Scatter(
        x=df["Blocks"],
        y=df["Actual"],
        name="Actual",
        mode="lines",
        line=dict(
            color="#ef4444",
            width=3
        )
    )
)


fig.update_layout(
    height=550,
    template="streamlit",
    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    ),

    xaxis_title="Block",
    yaxis_title="Generation"
)


st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)
