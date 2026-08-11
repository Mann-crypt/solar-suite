import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution

st.set_page_config(page_title="RT Correction — Solar Suite", layout="wide")

st.sidebar.markdown("""
<h1 style='text-align:center;
background:linear-gradient(90deg,#00c6ff,#0072ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
font-size:40px;font-weight:800;'>
⚡ Solar Suite
</h1>
<p style='text-align:center;color:gray;font-size:14px;'>Forecast Correction Platform</p>
""", unsafe_allow_html=True)
st.divider()

st.markdown("""
<style>
div[data-testid="metric-container"]{
    background:#111827;border:1px solid #1f2937;border-radius:10px;padding:12px 20px;
}
</style>""", unsafe_allow_html=True)

st.title("Guruji ne kaha tha RT Correct kardo bhayi 🛐!!")

# ── Initialise session defaults ───────────────────────────────────────────────
if "rt_input" not in st.session_state:
    st.session_state.rt_input = pd.DataFrame({"Actual": np.zeros(96), "Trend": np.zeros(96)})

if "rt_params" not in st.session_state:
    st.session_state.rt_params = {"w": 0.3, "n1": 29, "n2": 72, "b": 39}

# ── Data editor ───────────────────────────────────────────────────────────────
edited_df = st.data_editor(
    st.session_state.rt_input,
    key="rt_editor",
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
)
edited_df = edited_df.iloc[:96].reset_index(drop=True)

changed = (edited_df != st.session_state.rt_input).any(axis=1)
if changed.any():
    st.toast(f"✨ {changed.sum()} rows updated successfully!", icon="✅")

st.session_state.rt_input = edited_df.copy()
df = edited_df.copy()

# ── Time block labels ─────────────────────────────────────────────────────────
start_time = datetime.strptime("00:00", "%H:%M")
df["Time-Blocks"] = [
    f"{(start_time + timedelta(minutes=15*i)).strftime('%H:%M')} - "
    f"{(start_time + timedelta(minutes=15*(i+1))).strftime('%H:%M')}"
    for i in range(96)
]
df["Blocks"] = np.arange(1, 97)

actual = df["Actual"].to_numpy(dtype=float)
trend  = df["Trend"].to_numpy(dtype=float)
blocks = df["Blocks"].to_numpy(dtype=float)
mask   = actual > 0.5

# ── Objective ─────────────────────────────────────────────────────────────────
def objective(x):
    w, n1, n2, b = x
    n1, n2, b = int(round(n1)), int(round(n2)), int(round(b))
    if not (n1 < b < n2):
        return 1e6
    p = df.loc[df["Blocks"].isin([b-1, b, b+1]), "Actual"].mean()
    calc = p * (((n1 - blocks) * (n2 - blocks)) / ((n1 - b) * (n2 - b)))
    proj = np.where(calc < 0, 0, calc)
    pred = proj[mask]; act = actual[mask]
    if act.max() == 0: return 1e6
    return (0.80 * np.mean(np.abs(act - pred)) / act.max()
            + 0.10 * abs(act.max() - pred.max()) / act.max()
            + 0.10 * abs(act.sum() - pred.sum()) / act.sum())

# ── Optimize button ───────────────────────────────────────────────────────────
if st.button("🚀 Dabaiye na!!", use_container_width=True, type="primary"):
    with st.spinner("Optimizing..."):
        result = differential_evolution(
            objective,
            bounds=[(0.3, 0.3), (5, 40), (55, 95), (35, 40)],
            popsize=20, maxiter=100, polish=True, seed=42,
        )
    w, n1, n2, b = result.x
    st.session_state.rt_params = {
        "w": float(w), "n1": int(round(n1)),
        "n2": int(round(n2)), "b": int(round(b)),
    }
    st.rerun()

# ── Parameter inputs ──────────────────────────────────────────────────────────
p = st.session_state.rt_params
col1, col2 = st.columns(2)
with col1:
    w  = st.number_input("Weight",      0.0, 1.0, value=float(p["w"]),  step=0.01)
    n2 = st.number_input("n2",          value=int(p["n2"]),              step=1)
with col2:
    n1 = st.number_input("n1",          value=int(p["n1"]),              step=1)
    b  = st.number_input("Peak Block",  value=int(p["b"]),               step=1)

# ── Final calculation ─────────────────────────────────────────────────────────
p_val = df.loc[df["Blocks"].isin([b-1, b, b+1]), "Actual"].mean()
calc  = p_val * (((n1 - blocks) * (n2 - blocks)) / ((n1 - b) * (n2 - b)))
proj  = np.where(calc < 0, 0, calc)
pred  = np.where(blocks > b, w * proj + (1 - w) * trend, trend)
df["Projection"] = proj

# ── Time block lookup ─────────────────────────────────────────────────────────
lookup_df = pd.DataFrame({
    "Parameter": [
        "Parabolic Power Generation Starting Block",
        "Parabolic Power Generation Ending Block",
        "Actual Generation Available Block (Lower Limit)",
        "Actual Generation Effective Block (Upper Limit)",
    ],
    "Block": [n1, n2, n1 + 3, n2 - 3],
})
lookup_df["Time Block"] = lookup_df["Block"].map(df.set_index("Blocks")["Time-Blocks"])
with st.expander("📅 Important Time Blocks"):
    st.dataframe(lookup_df, use_container_width=True, hide_index=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Blocks"], y=df["Projection"], name="Projection",
                          line=dict(color="#00c6ff", width=3)))
fig.add_trace(go.Scatter(x=df["Blocks"], y=df["Actual"], name="Actual",
                          line=dict(color="#ef4444", width=3)))
fig.update_layout(height=550, hovermode="x unified", template="plotly_dark",
                  paper_bgcolor="#111827", plot_bgcolor="#111827",
                  legend=dict(orientation="h", y=1.08, x=0),
                  margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(fig, use_container_width=True)
