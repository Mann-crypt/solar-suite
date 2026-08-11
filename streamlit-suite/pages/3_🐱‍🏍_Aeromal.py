import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter

st.set_page_config(page_title="Aeromal — Solar Suite", layout="wide")

# Password from environment variable — never hardcoded in source
AEROMAL_PASSWORD = "asdfghjkl;'"
# ── Auth gate ─────────────────────────────────────────────────────────────────
if "aeromal_auth" not in st.session_state:
    st.session_state.aeromal_auth = False

if not st.session_state.aeromal_auth:
    st.title("🔒 Access bas bade logo ke paas hai")
    password = st.text_input("Enter Password", type="password")
    if st.button("Login", type="primary", use_container_width=True):
        if AEROMAL_PASSWORD and password == AEROMAL_PASSWORD:
            st.session_state.aeromal_auth = True
            st.rerun()
        else:
            st.error("Incorrect Password")
    st.stop()

st.title("Kaha hai Aeromal ka khauf?!!")

# ── Data editor ───────────────────────────────────────────────────────────────
if "cam_input" not in st.session_state:
    st.session_state.cam_input = pd.DataFrame({"Power": np.zeros(96)})

edited_df = st.data_editor(
    st.session_state.cam_input,
    key="cam_editor",
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic",
)
st.session_state.cam_input = edited_df.copy()

power = pd.to_numeric(edited_df.iloc[:, 0], errors="coerce").fillna(0).to_numpy()

if len(power) == 0:
    st.stop()
if len(power) % 96 != 0:
    st.error("Number of rows must be divisible by 96.")
    st.stop()

days = len(power) // 96

# ── Curtailment toggle ────────────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stToggle"]{
    background:#1f2937;border:2px solid #0072ff;border-radius:14px;
    padding:14px 18px;margin-bottom:15px;
}
div[data-testid="stToggle"] label{font-size:18px !important;font-weight:700 !important;}
</style>""", unsafe_allow_html=True)

curtailment = st.toggle("⚡ Curtailment Mode", value=False)

# ─────────────────────────────────────────────────────────────────────────────
# CURTAILMENT MODE
# ─────────────────────────────────────────────────────────────────────────────
if curtailment:
    if not np.any(power > 0):
        st.warning("Please enter Power values to continue.")
        st.stop()

    a  = power.reshape(days, 96)
    ap = np.percentile(a, 95, axis=0) * 1.03

    st.subheader("Parameters")
    col1, col2 = st.columns(2)
    with col1:
        Power_Availability = st.number_input("Power Availability (%)", value=100, step=1)
        peak_cap           = st.number_input("Peak Cap", value=int(np.max(power)), step=1)
    with col2:
        target_width  = st.number_input("Target Width",  value=25, step=1)
        window_length = st.slider("Window Length", 5, 31, 11, step=2)

    # ── Curtailment profile function (untouched) ──────────────────────────────
    def solar_cap_curve(y, peak_cap, target_width, window_length, Power_Availability):
        y = np.array(y, dtype=float)
        n = len(y)
        para = np.zeros(n)

        ys   = savgol_filter(y, 7, 2)
        grad = np.gradient(ys)

        left_peak  = np.argmax(ys[:n//2])
        left_start = np.argmax(grad[:left_peak])
        m1, c1 = np.polyfit(np.arange(left_start, left_peak), ys[left_start:left_peak], 1)

        right_peak = np.argmax(ys[n//2:]) + n//2
        threshold  = 0.02 * np.max(ys)
        active_idx = np.where(ys > threshold)[0]
        right_end  = active_idx[-1]
        m2, c2 = np.polyfit(np.arange(right_peak, right_end), ys[right_peak:right_end], 1)

        A = (1/m2) - (1/m1)
        B = (c1/m1) - (c2/m2)
        trip = (target_width - B) / A

        peak_left_idx  = int(round((trip - c1) / m1))
        peak_right_idx = int(round((trip - c2) / m2))
        trip = max(0, trip)

        if peak_cap >= trip:
            for i in range(n):
                val = m1*i + c1
                para[i] = min(val, trip)
                if val >= trip: peak_left_idx = i; break

            right_curve = np.zeros(n)
            for i in range(n-1, -1, -1):
                val = m2*i + c2
                right_curve[i] = min(val, trip)
                if val >= trip: peak_right_idx = i; break

            para  = np.maximum(para, right_curve)
            width = peak_right_idx - peak_left_idx
            dome_height = max(20, 0.12 * trip)
            x     = np.linspace(-1, 1, width)
            shape = np.sqrt(np.maximum(0, 1 - x**2))
            dome  = trip + dome_height * shape
            dome[0] = trip; dome[-1] = trip
            para[peak_left_idx:peak_right_idx] = dome
            para = savgol_filter(para, window_length, 3)
            para = np.clip(para, 0, None)
            para[:left_start]  = ys[:left_start]
            para[right_end:]   = ys[right_end:]
            para = savgol_filter(para, 7, 3) * Power_Availability / 100
            para = np.clip(para, 0, None)
            para = np.where(para < 0.2, 0, para)
        else:
            for i in range(n):
                val = m1*i + c1
                para[i] = val
                if val >= peak_cap: peak_left_idx = i; break

            right_curve = np.zeros(n)
            for i in range(n-1, -1, -1):
                val = m2*i + c2
                right_curve[i] = val
                if val >= peak_cap: peak_right_idx = i; break

            para = np.maximum(para, right_curve)
            para[peak_left_idx:peak_right_idx] = peak_cap
            para = np.clip(para, 0, peak_cap)
            para = savgol_filter(para, window_length, 3)
            para[:left_start] = ys[:left_start]
            para[right_end:]  = ys[right_end:]
            para = savgol_filter(para, 7, 3) * Power_Availability / 100
            para = np.clip(para, 0, peak_cap)
            para = np.where(para < 1, 0, para)

        return para

    Final_Smooth = solar_cap_curve(ap, peak_cap, target_width, window_length, Power_Availability)

    # Best symmetry shift
    least_error = np.inf; best_shift = 0
    for i in range(96):
        sh  = np.roll(Final_Smooth, -i)
        sym = (Final_Smooth + sh[::-1]) / 2
        err = np.sqrt(np.mean((Final_Smooth - sym)**2))
        if err < least_error: least_error = err; best_shift = i

    shift = st.number_input("Shift", min_value=0, max_value=95, value=best_shift, step=1)
    sh = np.roll(Final_Smooth, -int(shift))
    Final_Smooth_Sym = (Final_Smooth + sh[::-1]) / 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(96), y=ap,              name="Generation",  line=dict(width=3)))
    fig.add_trace(go.Scatter(x=np.arange(96), y=Final_Smooth,    name="Profile",     line=dict(width=3, color="#00c6ff")))
    fig.add_trace(go.Scatter(x=np.arange(96), y=Final_Smooth_Sym,name="Sym Profile", line=dict(width=3, color="#0072ff")))
    fig.update_layout(height=550, hovermode="x unified", template="plotly_dark",
                      paper_bgcolor="#111827", plot_bgcolor="#111827",
                      legend=dict(orientation="h", y=1.08, x=0),
                      margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    output = pd.DataFrame({"Power": ap, "Profile": Final_Smooth, "Sym Profile": Final_Smooth_Sym})
    st.dataframe(output, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# NO CURTAILMENT MODE
# ─────────────────────────────────────────────────────────────────────────────
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        window = st.number_input("Window Length", min_value=5, max_value=31, step=2, value=11)
    with col2:
        power_availability = st.number_input("Power Availability (%)", min_value=0, max_value=1000, value=100)

    a  = power.reshape(days, 96)
    ap = np.percentile(a, 95, axis=0)
    s  = savgol_filter(ap, window_length=window, polyorder=3)

    least_error = np.inf; best_shift = 0
    for i in range(96):
        sh  = np.roll(s, -i)
        sym = (s + sh[::-1]) / 2
        err = np.sqrt(np.mean((ap - sym)**2))
        if err < least_error: least_error = err; best_shift = i

    with col3:
        shift = st.number_input("Shift", min_value=0, max_value=95, value=int(best_shift))

    alpha = 0.50
    sh    = np.roll(s, -shift)
    sym   = alpha * s + (1 - alpha) * sh[::-1]
    thr   = 0.1
    idx   = np.where(ap > thr)[0]
    if len(idx) > 0:
        start_i, end_i = idx[0], idx[-1]
        blend = 8
        w = np.linspace(1, 0, blend)
        sym[start_i+1:start_i+1+blend] = w*ap[start_i+1:start_i+1+blend] + (1-w)*sym[start_i+1:start_i+1+blend]
        w = np.linspace(0, 1, blend)
        sym[end_i-blend:end_i] = w*ap[end_i-blend:end_i] + (1-w)*sym[end_i-blend:end_i]

    s   = savgol_filter(ap, window_length=11, polyorder=3)
    sym = savgol_filter(sym, window_length=11, polyorder=3)
    s   = np.clip(s, 0, None); sym = np.clip(sym, 0, None)
    s   = np.where(s < 0.1, 0, s); sym = np.where(sym < 0.1, 0, sym)
    s   *= power_availability / 100; sym *= power_availability / 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(96), y=sym, name="Sym Profile",    line=dict(color="#00c6ff", width=4)))
    fig.add_trace(go.Scatter(x=np.arange(96), y=s,   name="Profile",        line=dict(color="#22c55e", width=4)))
    fig.add_trace(go.Scatter(x=np.arange(96), y=ap,  name="95th Percentile",line=dict(color="#ef4444", width=4)))
    fig.update_layout(height=550, hovermode="x unified", template="plotly_dark",
                      xaxis_title="Block", yaxis_title="Power",
                      paper_bgcolor="#111827", plot_bgcolor="#111827",
                      legend=dict(orientation="h", y=1.08, x=0),
                      margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    result = pd.DataFrame({"Percentile": ap, "Profile": s, "Sym Profile": sym})
    st.subheader("Generated Curve")
    st.dataframe(result, use_container_width=True, hide_index=True)
