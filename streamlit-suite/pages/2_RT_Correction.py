import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datetime import datetime, timedelta
from scipy.optimize import differential_evolution




st.title("Guruji ne kaha tha RT Correct kardo bhyii🛐!!")

    if "rt_input" not in st.session_state:
        st.session_state.rt_input = pd.DataFrame({
            "Actual": np.zeros(96),
            "Trend": np.zeros(96)
        })
    
    edited_df = st.data_editor(
        st.session_state.rt_input,
        key="rt_editor",
        use_container_width=True,
        hide_index=True,
        num_rows="fixed"
    )
    
    edited_df = edited_df.iloc[:96].reset_index(drop=True)
    
    # Detect changes
    changed_rows = (edited_df != st.session_state.rt_input).any(axis=1)
    
    if changed_rows.any():
    
        st.toast(
            f"✨ {changed_rows.sum()} rows updated successfully!",
            icon="✅"
        )
    
    # Update session state
    st.session_state.rt_input = edited_df.copy()
    
    st.session_state.rt_input = edited_df.copy()
    df = edited_df.copy()
    
    # ---------------- Time Blocks ----------------
    
    start = datetime.strptime("00:00", "%H:%M")
    
    df["Time-Blocks"] = [
        f"{(start+timedelta(minutes=15*i)).strftime('%H:%M')} - {(start+timedelta(minutes=15*(i+1))).strftime('%H:%M')}"
        for i in range(96)
    ]
    
    df["Blocks"] = np.arange(1,97)
    # ---------------- Default Parameters ----------------

    if "rt_params" not in st.session_state:
        st.session_state.rt_params = {
            "w": 0.3,
            "n1": 29,
            "n2": 72,
            "b": 39
        }
    
    actual = df["Actual"].to_numpy(dtype=float)
    trend = df["Trend"].to_numpy(dtype=float)
    blocks = df["Blocks"].to_numpy(dtype=float)
    
    mask = actual > 0.5
    
    # ---------------- Objective ----------------
    
    def objective(x):
    
        w, n1, n2, b = x
    
        n1 = int(round(n1))
        n2 = int(round(n2))
        b = int(round(b))
    
        if not (n1 < b < n2):
            return 1e6
    
        p = df.loc[
            df["Blocks"].isin([b-1, b, b+1]),
            "Actual"
        ].mean()
    
        calc = p * (
            ((n1 - blocks) * (n2 - blocks))
            /
            ((n1 - b) * (n2 - b))
        )
    
        projection = np.where(calc < 0, 0, calc)
    
        prediction = np.where(
            blocks > b,
            w * projection + (1 - w) * trend,
            trend
        )
    
        pred = projection[mask]
        act = actual[mask]
    
        block_error = np.mean(np.abs(act - pred)) / act.max()
    
        peak_error = abs(act.max() - pred.max()) / act.max()
    
        energy_error = abs(act.sum() - pred.sum()) / act.sum()
    
        return (
            0.80 * block_error +
            0.10 * peak_error +
            0.10 * energy_error
        )
    
    
    # ---------------- Optimize ----------------
    
    if st.button("🚀 Dabaiye na!!", use_container_width=True, type="primary"):
    
        with st.spinner("Optimizing..."):
    
            result = differential_evolution(
                objective,
                bounds=[
                    (0.3, 0.3),      # same as notebook
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
    
    
    # ---------------- User Inputs ----------------
    
    col1, col2 = st.columns(2)
    
    with col1:
    
        w = st.number_input(
            "Weight",
            0.0,
            1.0,
            value=float(st.session_state.rt_params["w"]),
            step=0.01
        )
    
        n2 = st.number_input(
            "n2",
            value=int(st.session_state.rt_params["n2"]),
            step=1
        )
    
    with col2:
    
        n1 = st.number_input(
            "n1",
            value=int(st.session_state.rt_params["n1"]),
            step=1
        )
    
        b = st.number_input(
            "Peak Block",
            value=int(st.session_state.rt_params["b"]),
            step=1
        )
    
    
    # ---------------- Final Calculation ----------------
    
    p = df.loc[
        df["Blocks"].isin([b-1, b, b+1]),
        "Actual"
    ].mean()
    
    calc = p * (
        ((n1 - blocks) * (n2 - blocks))
        /
        ((n1 - b) * (n2 - b))
    )
    
    projection = np.where(calc < 0, 0, calc)
    
    prediction = np.where(
        blocks > b,
        w * projection + (1 - w) * trend,
        trend
    )
    
    df["Projection"] = projection

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
        "Actual Generation Effective Block (Upper Limit)"
    ]
    
    lookup_df = pd.DataFrame({
        "Parameter": lookup_names,
        "Block": lookup_blocks
    })
    
    lookup_df["Time Block"] = lookup_df["Block"].map(
        df.set_index("Blocks")["Time-Blocks"]
    )
    
    with st.expander("📅 Important Time Blocks"):
        st.dataframe(
            lookup_df,
            use_container_width=True,
            hide_index=True
        )
    
    
    # ---------------- Graph ----------------
    
    fig=go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df["Blocks"],
            y=df["Projection"],
            name="Projection"
        )
    )
    
    
    fig.add_trace(
        go.Scatter(
            x=df["Blocks"],
            y=df["Actual"],
            name="Actual"
        )
    )
    
    fig.update_layout(
        height=550,
        hovermode="x unified"
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True
    )
