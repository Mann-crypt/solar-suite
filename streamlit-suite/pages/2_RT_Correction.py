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
# CACHED HEAVY COMPUTATION
# All inputs are plain tuples so Streamlit can hash them.
# The DE runs once; param adjustments use cached_rt_curve.
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_rt_optimize(actual_tuple, trend_tuple):
    actual = np.array(actual_tuple, dtype=float)
    trend  = np.array(trend_tuple,  dtype=float)
    blocks = np.arange(1, 97, dtype=float)
    mask   = actual > 0.5

    def objective(x):
        w, n1, n2, b = x
        n1, n2, b = int(round(n1)), int(round(n2)), int(round(b))
        if not (n1 < b < n2):
            return 1e6
        p    = actual[np.isin(blocks, [b-1, b, b+1])].mean()
        calc = p * (((n1-blocks)*(n2-blocks)) / ((n1-b)*(n2-b)))
        proj = np.where(calc < 0, 0, calc)
        pred = proj[mask]
        act  = actual[mask]
        if act.max() == 0:
            return 1e6
        return (0.80 * np.mean(np.abs(act-pred)) / act.max()
                + 0.10 * abs(act.max()-pred.max()) / act.max()
                + 0.10 * abs(act.sum()-pred.sum()) / act.sum())

    result = differential_evolution(
        objective,
        bounds=[(0.3, 0.3), (5, 40), (55, 95), (35, 40)],
        popsize=20, maxiter=100, polish=True, seed=42,
    )
    w, n1, n2, b = result.x
    return {
        "w":  float(w),
        "n1": int(round(n1)),
        "n2": int(round(n2)),
        "b":  int(round(b)),
    }


@st.cache_data(show_spinner=False)
def cached_rt_curve(actual_tuple, trend_tuple, w, n1, n2, b):
    """Final parabolic curve — cached so param tweaks are instant."""
    actual = np.array(actual_tuple, dtype=float)
    trend  = np.array(trend_tuple,  dtype=float)
    blocks = np.arange(1, 97, dtype=float)

    mask   = np.isin(blocks, [b-1, b, b+1])
    p      = actual[mask].mean()
    calc   = p * (((n1-blocks)*(n2-blocks)) / ((n1-b)*(n2-b)))
    proj   = np.where(calc < 0, 0, calc)

    return proj.tolist()


@st.cache_data(show_spinner=False)
def time_block_lookup(n1, n2):
    start = datetime.strptime("00:00", "%H:%M")
    tb    = [
        f"{(start+timedelta(minutes=15*i)).strftime('%H:%M')} - "
        f"{(start+timedelta(minutes=15*(i+1))).strftime('%H:%M')}"
        for i in range(96)
    ]
    rows = [
        ("Parabolic Power Generation Starting Block",        n1,     tb[n1-1]  if 0<n1<=96  else "—"),
        ("Parabolic Power Generation Ending Block",          n2,     tb[n2-1]  if 0<n2<=96  else "—"),
        ("Actual Generation Available Block (Lower Limit)",  n1+3,   tb[n1+2]  if n1+3<=96  else "—"),
        ("Actual Generation Effective Block (Upper Limit)",  n2-3,   tb[n2-4]  if n2-3>=1   else "—"),
    ]
    return pd.DataFrame(rows, columns=["Parameter","Block","Time Block"])


# ==========================================================
# PAGE
# ==========================================================

st.title("Guruji ne kaha tha RT Correct kardo bhayi 🛐!!")

# ── Session defaults ──────────────────────────────────────
if "rt_actual" not in st.session_state:
    st.session_state.rt_actual = [0.0] * 96
if "rt_trend" not in st.session_state:
    st.session_state.rt_trend  = [0.0] * 96
if "rt_params" not in st.session_state:
    st.session_state.rt_params = {"w":0.3,"n1":29,"n2":72,"b":39}
if "rt_optimized" not in st.session_state:
    st.session_state.rt_optimized = False

# ── Data editor ───────────────────────────────────────────
input_df = pd.DataFrame({
    "Actual": st.session_state.rt_actual,
    "Trend":  st.session_state.rt_trend,
})

edited_df = st.data_editor(
    input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="rt_editor",
)
edited_df = edited_df.iloc[:96].reset_index(drop=True)
edited_df["Actual"] = pd.to_numeric(edited_df["Actual"], errors="coerce").fillna(0)
edited_df["Trend"]  = pd.to_numeric(edited_df["Trend"],  errors="coerce").fillna(0)

# Detect edits — clear optimized result so it reruns with new data
actual_list = edited_df["Actual"].tolist()
trend_list  = edited_df["Trend"].tolist()

if (actual_list != st.session_state.rt_actual
        or trend_list != st.session_state.rt_trend):
    changed_n = sum(
        a != b for a, b in
        zip(actual_list, st.session_state.rt_actual)
    ) + sum(
        a != b for a, b in
        zip(trend_list, st.session_state.rt_trend)
    )
    st.toast(f"✨ {changed_n} cells updated!", icon="✅")
    st.session_state.rt_actual    = actual_list
    st.session_state.rt_trend     = trend_list
    st.session_state.rt_optimized = False   # force re-optimize on next run

# ── Optimize button ───────────────────────────────────────
if st.button("🚀 Dabaiye na!!", use_container_width=True, type="primary"):
    with st.spinner("Optimizing..."):
        params = cached_rt_optimize(
            tuple(actual_list),
            tuple(trend_list),
        )
    st.session_state.rt_params    = params
    st.session_state.rt_optimized = True
    st.rerun()

# ── Parameter inputs inside a form — zero reruns while typing ──
st.subheader("Parameters")
st.caption("Adjust values then click Recalculate — chart updates instantly.")

p = st.session_state.rt_params

with st.form("rt_params_form"):
    col1, col2 = st.columns(2)
    with col1:
        w  = st.number_input("Weight", 0.0, 1.0, value=float(p["w"]), step=0.01)
        n2 = st.number_input("n2",            value=int(p["n2"]),     step=1)
    with col2:
        n1 = st.number_input("n1",            value=int(p["n1"]),     step=1)
        b  = st.number_input("Peak Block",    value=int(p["b"]),      step=1)
    recalc = st.form_submit_button("🔄 Recalculate", use_container_width=True, type="primary")

if recalc:
    st.session_state.rt_params = {"w": w, "n1": n1, "n2": n2, "b": b}
    p = st.session_state.rt_params

# ── Final curve — fully cached, instant ──────────────────
proj = cached_rt_curve(
    tuple(actual_list), tuple(trend_list),
    p["w"], p["n1"], p["n2"], p["b"],
)

# Time block lookup
with st.expander("📅 Important Time Blocks"):
    st.dataframe(
        time_block_lookup(p["n1"], p["n2"]),
        use_container_width=True,
        hide_index=True,
    )

# ── Chart ─────────────────────────────────────────────────
blocks = list(range(1, 97))
fig = go.Figure()
fig.add_trace(go.Scatter(x=blocks, y=proj,        name="Projection",
                          line=dict(color="#00c6ff", width=3)))
fig.add_trace(go.Scatter(x=blocks, y=actual_list, name="Actual",
                          line=dict(color="#ef4444", width=3)))
fig.update_layout(
    height=550, hovermode="x unified", template="streamlit",
    legend=dict(orientation="h", y=1.08, x=0),
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig, use_container_width=True)
