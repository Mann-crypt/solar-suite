# ============================================================
# IMPORTS
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Forecast Correction",
    page_icon="☀️",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 4px;
}

.sub-title {
    color: #666;
    font-size: 15px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 20px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 10px;
}

.metric-card {
    background: #f7f9fc;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}

.metric-label {
    color: #6b7280;
    font-size: 13px;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Fixed / Tracking plant forecast correction and optimization'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.segmented_control(
    "Plant Type",
    ["Fixed", "Tracking"],
    default="Tracking"
)


# ============================================================
# FILE UPLOADER
# ============================================================

st.markdown(
    '<div class="section-title">📁 Input File</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Excel file",
    type=["xlsx", "xls"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.info("Upload the Excel file to continue.")
    st.stop()


# ============================================================
# READ EXCEL
# ============================================================

@st.cache_data
def read_excel_file(file_bytes):

    return pd.ExcelFile(
        io.BytesIO(file_bytes)
    )


excel_file = read_excel_file(
    uploaded_file.getvalue()
)


# ============================================================
# READ AREA & EFFICIENCY
# ============================================================

df = pd.read_excel(
    io.BytesIO(uploaded_file.getvalue()),
    sheet_name="Area & Efficiency",
    header=1,
    usecols=range(12)
)

df.columns = (
    df.columns
    .astype(str)
    .str.replace("*", "", regex=False)
    .str.strip()
)

if "S.No." in df.columns:

    valid_rows = df["S.No."].notna()

    if valid_rows.any():
        df = df.loc[:valid_rows[valid_rows].index[-1]].copy()


# ============================================================
# CALCULATE TOTAL AREA
# ============================================================

if (
    "No of Module" in df.columns
    and "Area of 1 Module (m2)" in df.columns
):

    df["Total area (m2)"] = (
        pd.to_numeric(
            df["No of Module"],
            errors="coerce"
        ).fillna(0)
        *
        pd.to_numeric(
            df["Area of 1 Module (m2)"],
            errors="coerce"
        ).fillna(0)
    )


# ============================================================
# READ CLUSTER WEIGHTS
# ============================================================

df_w = pd.read_excel(
    io.BytesIO(uploaded_file.getvalue()),
    sheet_name="Area & Efficiency",
    header=1,
    usecols=[14, 15]
)

df_w.columns = df_w.columns.str.strip()

if "Clusters" in df_w.columns:

    valid_rows = df_w["Clusters"].notna()

    if valid_rows.any():
        df_w = df_w.loc[
            :valid_rows[valid_rows].index[-1]
        ].copy()


# ============================================================
# USER ERROR %
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Correction Parameters</div>',
    unsafe_allow_html=True
)

error_col = st.columns(3)

with error_col[0]:

    error_min = st.number_input(
        "Error % Minimum",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.1
    )

with error_col[1]:

    error_max = st.number_input(
        "Error % Maximum",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.1
    )

with error_col[2]:

    error_step = st.number_input(
        "Error % Step",
        min_value=0.01,
        max_value=5.0,
        value=0.1,
        step=0.01
    )


# ============================================================
# BASE EFFICIENCY
# ============================================================

df["Standard PV Efficiency (%)"] = pd.to_numeric(
    df["Standard PV Efficiency (%)"],
    errors="coerce"
).fillna(0)


# ============================================================
# USER INPUT DATA
# GHI + ACTUAL
# ============================================================

st.markdown(
    '<div class="section-title">📊 Forecast Input Data</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter or paste the 15-minute GHI forecast and Actual power. "
    "The table is editable."
)


# ============================================================
# DEFAULT INPUT DATA
# ============================================================

ghi_columns = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15"
]

input_columns = ghi_columns + ["Actual"]


# ============================================================
# READ DEFAULT GHI FROM RESULT SHEET
# ONLY USED AS INITIAL INPUT
# ============================================================

try:

    default_ghi = pd.read_excel(
        io.BytesIO(uploaded_file.getvalue()),
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5]
    )

    default_ghi.columns = [
        "Block",
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15"
    ]

except Exception:

    default_ghi = pd.DataFrame({
        "Block": np.arange(1, 97),
        "GHI C11": np.zeros(96),
        "GHI C12": np.zeros(96),
        "GHI C13": np.zeros(96),
        "GHI C14": np.zeros(96),
        "GHI C15": np.zeros(96)
    })


# ============================================================
# READ DEFAULT ACTUAL
# ============================================================

try:

    default_actual_df = pd.read_excel(
        io.BytesIO(uploaded_file.getvalue()),
        sheet_name="Fixed-C11",
        header=1
    )

    default_actual_df.columns = (
        default_actual_df.columns
        .astype(str)
        .str.strip()
    )

    default_actual = pd.to_numeric(
        default_actual_df["Actual"],
        errors="coerce"
    ).fillna(0)

except Exception:

    default_actual = pd.Series(
        np.zeros(len(default_ghi))
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

n_rows = max(
    len(default_ghi),
    len(default_actual)
)

input_df = pd.DataFrame(
    np.zeros((n_rows, len(input_columns))),
    columns=input_columns
)

for col in ghi_columns:

    if col in default_ghi.columns:

        values = pd.to_numeric(
            default_ghi[col],
            errors="coerce"
        ).fillna(0).to_numpy()

        input_df.loc[
            :min(len(values), n_rows) - 1,
            col
        ] = values[:n_rows]


actual_values = pd.to_numeric(
    default_actual,
    errors="coerce"
).fillna(0).to_numpy()

input_df.loc[
    :min(len(actual_values), n_rows) - 1,
    "Actual"
] = actual_values[:n_rows]


# ============================================================
# EDITABLE INPUT DATAFRAME
# ============================================================

edited_input = st.data_editor(
    input_df,
    use_container_width=True,
    height=420,
    num_rows="fixed",
    column_config={
        "GHI C11": st.column_config.NumberColumn(
            "GHI C11",
            min_value=0,
            format="%.3f"
        ),

        "GHI C12": st.column_config.NumberColumn(
            "GHI C12",
            min_value=0,
            format="%.3f"
        ),

        "GHI C13": st.column_config.NumberColumn(
            "GHI C13",
            min_value=0,
            format="%.3f"
        ),

        "GHI C14": st.column_config.NumberColumn(
            "GHI C14",
            min_value=0,
            format="%.3f"
        ),

        "GHI C15": st.column_config.NumberColumn(
            "GHI C15",
            min_value=0,
            format="%.3f"
        ),

        "Actual": st.column_config.NumberColumn(
            "Actual Power",
            min_value=0,
            format="%.3f"
        )
    }
)


# ============================================================
# BUILD df_ghi
# ============================================================

df_ghi = edited_input[
    ghi_columns
].copy()

df_ghi = df_ghi.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)


# ============================================================
# BUILD df_fix
# ============================================================

df_fix = pd.DataFrame({
    "Actual": pd.to_numeric(
        edited_input["Actual"],
        errors="coerce"
    ).fillna(0)
})


# ============================================================
# VALIDATE INPUT
# ============================================================

if len(df_ghi) != len(df_fix):

    st.error(
        "GHI and Actual data must contain the same number of rows."
    )

    st.stop()


# ============================================================
# BLOCK NUMBERS
# ============================================================

blocks = np.arange(
    1,
    len(df_ghi) + 1
)


# ============================================================
# TRACKING
# ============================================================

if plant_type == "Tracking":

    # ========================================================
    # READ BACKEND CALCULATIONS
    # ========================================================

    backend_list = []

    for cluster in ["C11", "C12", "C13", "C14", "C15"]:

        backend = pd.read_excel(
            io.BytesIO(uploaded_file.getvalue()),
            sheet_name=f"Backend Cal {cluster}"
        )

        backend_list.append(backend)


    # ========================================================
    # READ TRACKING SHEET
    # ========================================================

    df_trac = pd.read_excel(
        io.BytesIO(uploaded_file.getvalue()),
        sheet_name="Tracking",
        header=1
    )

    df_trac.columns = (
        df_trac.columns
        .astype(str)
        .str.strip()
    )


    # ========================================================
    # CLUSTER EFFECTIVE AREA
    #
    # IMPORTANT:
    #
    # Error % is applied HERE ONLY.
    #
    # It must NOT be applied again inside the
    # Tracking DNI / power calculation.
    # ========================================================

    def calculate_cluster_areas(error):

        temp_df = df.copy()

        temp_df["Error %"] = error

        temp_df["Net Efficiency (%)"] = (
            temp_df["Standard PV Efficiency (%)"]
            - error
        )

        temp_df["Eff Area"] = (
            temp_df["Net Efficiency (%)"]
            *
            temp_df["Total area (m2)"]
            / 100
        )

        cluster_sums = (
            temp_df
            .groupby("Clusters")["Eff Area"]
            .sum()
        )

        cluster_area = (
            df_w["Clusters"]
            .map(cluster_sums)
            .fillna(0)
            .to_numpy(dtype=float)
        )

        if len(cluster_area) < 5:

            cluster_area = np.pad(
                cluster_area,
                (0, 5 - len(cluster_area))
            )

        return cluster_area[:5]


    # ========================================================
    # INITIAL ACTUAL DATA
    # ========================================================

    actual_full = (
        df_fix["Actual"]
        .to_numpy(dtype=float)
    )

    actual_full = np.nan_to_num(
        actual_full,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    mask = actual_full != 0

    if not mask.any():

        st.error(
            "No non-zero Actual values found. "
            "Please enter Actual power data."
        )

        st.stop()

    actual = actual_full[mask]

    actual_max = actual.max()
    actual_sum = actual.sum()


    # ========================================================
    # TRACKING PARAMETERS
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Tracking Parameters</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        DHI = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=1,
            step=1
        )

        GHI_Starting_Block = st.number_input(
            "GHI Starting Block",
            min_value=1,
            max_value=len(df_ghi),
            value=min(30, len(df_ghi)),
            step=1
        )

    with p2:

        GHI_Ending_Block = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=len(df_ghi),
            value=min(79, len(df_ghi)),
            step=1
        )

        GHI_Max_Block = st.number_input(
            "GHI Max Block",
            min_value=1,
            max_value=len(df_ghi),
            value=min(53, len(df_ghi)),
            step=1
        )

    with p3:

        Tracking_angle_lim_E = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            value=11,
            step=1
        )

        Tracking_angle_lim_W = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            value=23,
            step=1
        )


    # ========================================================
    # GHI MATRIX
    # ========================================================

    ghi_matrix = df_ghi[
        ghi_columns
    ].to_numpy(dtype=float)


    # ========================================================
    # OBJECTIVE FUNCTION
    # ========================================================

    def tracking_forecast(
        DHI,
        start_block,
        end_block,
        max_block,
        east_limit,
        west_limit,
        cluster_area
    ):

        if not (
            start_block
            < max_block
            < end_block
        ):

            return None, None, None


        # ----------------------------------------------------
        # SAME FORMULA AS JUPYTER
        # ----------------------------------------------------

        m1 = 90 / (
            start_block
            - 1
            - max_block
        )

        m2 = 90 / (
            end_block
            + 1
            - max_block
        )


        # ----------------------------------------------------
        # ZENITH
        # ----------------------------------------------------

        zenith = np.where(

            blocks <= max_block,

            np.minimum(
                89,
                m1 * (
                    blocks
                    - max_block
                )
            ),

            np.minimum(
                89,
                m2 * (
                    blocks
                    - max_block
                )
            )
        )


        # ----------------------------------------------------
        # PANEL ANGLE
        # ----------------------------------------------------

        panel = np.where(

            blocks < max_block,

            np.minimum(
                zenith,
                abs(east_limit)
            ),

            np.where(

                (
                    (blocks > max_block)
                    &
                    (zenith > west_limit)
                ),

                west_limit,

                zenith
            )
        )


        # ----------------------------------------------------
        # COSINE
        # ----------------------------------------------------

        cos_alpha = np.cos(
            np.radians(panel)
        )

        cos_alpha = np.clip(
            cos_alpha,
            1e-6,
            None
        )


        # ----------------------------------------------------
        # DHI
        # ----------------------------------------------------

        dhi = (
            ghi_matrix
            *
            DHI
            / 100
        )


        # ----------------------------------------------------
        # DNI
        # ----------------------------------------------------

        dni = (
            ghi_matrix
            - dhi
        ) / cos_alpha[:, None]


        # ----------------------------------------------------
        # IMPORTANT
        #
        # Error % IS NOT applied here.
        #
        # cluster_area already contains:
        #
        # Standard Efficiency - Error %
        #
        # Therefore this is ONLY:
        #
        # DNI × Effective Area
        # ----------------------------------------------------

        forecast = (
            dni @ cluster_area
        ) / 1_000_000


        return forecast, zenith, panel


    # ========================================================
    # OPTIMIZATION
    # ========================================================

    if st.button(
        "🚀 Run Tracking Optimization",
        use_container_width=True
    ):

        error_values = np.arange(
            error_min,
            error_max + error_step / 2,
            error_step
        )


        best_result = None

        best_score = np.inf


        # ====================================================
        # ERROR % LOOP
        #
        # Error is applied exactly once here.
        # ====================================================

        for error in error_values:

            cluster_area = calculate_cluster_areas(
                error
            )


            def objective(x):

                d = int(round(x[0]))
                start = int(round(x[1]))
                end = int(round(x[2]))
                max_b = int(round(x[3]))
                east = int(round(x[4]))
                west = int(round(x[5]))


                if not (
                    start
                    < max_b
                    < end
                ):

                    return 1e9


                forecast, _, _ = tracking_forecast(
                    d,
                    start,
                    end,
                    max_b,
                    east,
                    west,
                    cluster_area
                )


                if forecast is None:

                    return 1e9


                if (
                    np.isnan(forecast).any()
                    or np.isinf(forecast).any()
                ):

                    return 1e9


                prediction = forecast[mask]


                if len(prediction) == 0:

                    return 1e9


                block_error = (
                    np.mean(
                        np.abs(
                            actual
                            - prediction
                        )
                    )
                    / actual_max
                )


                peak_error = (
                    abs(
                        actual_max
                        - prediction.max()
                    )
                    / actual_max
                )


                energy_error = (
                    abs(
                        actual_sum
                        - prediction.sum()
                    )
                    / actual_sum
                )


                score = (
                    0.80 * block_error
                    +
                    0.10 * peak_error
                    +
                    0.10 * energy_error
                )


                return score


            bounds = [

                (0, 10),

                (
                    10,
                    min(30, len(df_ghi))
                ),

                (
                    min(65, len(df_ghi) - 1),
                    min(80, len(df_ghi))
                ),

                (
                    min(47, len(df_ghi) - 2),
                    min(53, len(df_ghi) - 1)
                ),

                (10, 70),

                (10, 70)
            ]


            result = differential_evolution(

                objective,

                bounds=bounds,

                strategy="best1bin",

                maxiter=40,

                popsize=15,

                tol=0.001,

                mutation=(0.5, 1),

                recombination=0.7,

                seed=42,

                polish=True,

                workers=1
            )


            if result.fun < best_score:

                best_score = result.fun

                best_result = (
                    error,
                    result
                )


        # ====================================================
        # BEST ERROR + PARAMETERS
        # ====================================================

        best_error = best_result[0]

        result = best_result[1]


        best = np.round(
            result.x
        ).astype(int)


        DHI = best[0]

        GHI_Starting_Block = best[1]

        GHI_Ending_Block = best[2]

        GHI_Max_Block = best[3]

        Tracking_angle_lim_E = best[4]

        Tracking_angle_lim_W = best[5]


        # ====================================================
        # FINAL EFFECTIVE AREA
        # ====================================================

        cluster_area = calculate_cluster_areas(
            best_error
        )


        # ====================================================
        # FINAL FORECAST
        # ====================================================

        forecast, zenith, panel = tracking_forecast(

            DHI,

            GHI_Starting_Block,

            GHI_Ending_Block,

            GHI_Max_Block,

            Tracking_angle_lim_E,

            Tracking_angle_lim_W,

            cluster_area
        )


        # ====================================================
        # FINAL METRICS
        # ====================================================

        calculated_peak = forecast.max()

        actual_peak = actual_full.max()

        peak_error = abs(
            calculated_peak
            - actual_peak
        )

        peak_error_pct = (
            peak_error
            / actual_peak
            * 100
            if actual_peak != 0
            else np.nan
        )


        # ====================================================
        # SAVE TRACKING RESULTS
        # ====================================================

        df_trac = df_trac.copy()

        df_trac["Zenith Angle"] = zenith

        df_trac["Panel Angle"] = panel

        df_trac["Fixed Power=I*Ƞ*A"] = forecast


        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            '<div class="section-title">✅ Optimization Result</div>',
            unsafe_allow_html=True
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Best Error %",
                f"{best_error:.2f}%"
            )


        with c2:

            st.metric(
                "Calculated Peak",
                f"{calculated_peak:.3f}"
            )


        with c3:

            st.metric(
                "Actual Peak",
                f"{actual_peak:.3f}"
            )


        with c4:

            st.metric(
                "Peak Error %",
                f"{peak_error_pct:.2f}%"
            )


        # ====================================================
        # PARAMETERS
        # ====================================================

        st.markdown(
            '<div class="section-title">🎯 Optimized Parameters</div>',
            unsafe_allow_html=True
        )


        p1, p2, p3, p4, p5, p6 = st.columns(6)


        with p1:
            st.metric("DHI", DHI)

        with p2:
            st.metric(
                "GHI Start",
                GHI_Starting_Block
            )

        with p3:
            st.metric(
                "GHI End",
                GHI_Ending_Block
            )

        with p4:
            st.metric(
                "GHI Max",
                GHI_Max_Block
            )

        with p5:
            st.metric(
                "East Limit",
                Tracking_angle_lim_E
            )

        with p6:
            st.metric(
                "West Limit",
                Tracking_angle_lim_W
            )


        # ====================================================
        # GRAPH
        # ====================================================

        st.markdown(
            '<div class="section-title">📈 Forecast vs Actual</div>',
            unsafe_allow_html=True
        )


        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=np.arange(
                    len(forecast)
                ),
                y=forecast,
                mode="lines",
                name="Forecast",
                line=dict(
                    width=3
                )
            )
        )


        fig.add_trace(
            go.Scatter(
                x=np.arange(
                    len(actual_full)
                ),
                y=actual_full,
                mode="lines",
                name="Actual",
                line=dict(
                    width=3
                )
            )
        )


        fig.update_layout(

            height=450,

            xaxis_title="15-Minute Block",

            yaxis_title="Power",

            hovermode="x unified",

            template="plotly_white",

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # TRACKING ANGLES
        # ====================================================

        with st.expander(
            "View Tracking Angles"
        ):

            angle_df = pd.DataFrame({

                "Block": blocks,

                "Zenith Angle": zenith,

                "Panel Angle": panel,

                "Forecast": forecast,

                "Actual": actual_full

            })


            st.dataframe(
                angle_df,
                use_container_width=True,
                height=400
            )


# ============================================================
# FIXED
# ============================================================

elif plant_type == "Fixed":

    st.info(
        "The same editable GHI and Actual input is now available "
        "for Fixed calculation as well."
    )

    st.write(
        "Use the same df_ghi and df_fix objects created above "
        "for the Fixed calculation."
    )
