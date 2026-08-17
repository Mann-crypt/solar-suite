# ============================================================
# STREAMLIT PAGE
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
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

st.markdown(
    """
    <style>

    .main-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .sub-title {
        color: #666;
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Fixed and Tracking Plant Forecast Calibration</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Configuration")

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx", "xls"]
    )

    plant_type = st.radio(
        "Plant Type",
        ["Fixed", "Tracking"],
        horizontal=True
    )

    st.divider()

    st.subheader("Error %")

    error_mode = st.radio(
        "Error Calculation",
        [
            "Automatic",
            "Manual"
        ]
    )

    if error_mode == "Manual":

        manual_error = st.number_input(
            "Error %",
            min_value=0.0,
            max_value=30.0,
            value=2.0,
            step=0.1
        )

    else:

        col1, col2 = st.columns(2)

        with col1:
            error_min = st.number_input(
                "Min %",
                min_value=0.0,
                max_value=30.0,
                value=0.0,
                step=0.1
            )

        with col2:
            error_max = st.number_input(
                "Max %",
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=0.1
            )

        error_step = st.number_input(
            "Step %",
            min_value=0.01,
            max_value=2.0,
            value=0.1,
            step=0.01
        )

    st.divider()


# ============================================================
# STOP IF NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload the Solar Forecast Excel file from the sidebar to begin."
    )

    st.stop()


# ============================================================
# LOAD EXCEL
# ============================================================

@st.cache_data
def load_excel(file_bytes):

    return io.BytesIO(file_bytes)


file_bytes = uploaded_file.getvalue()

excel_file = load_excel(file_bytes)


# ============================================================
# COMMON DATA PREPARATION
# ============================================================

@st.cache_data
def prepare_common_data(file_bytes):

    file_obj = io.BytesIO(file_bytes)

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    df = pd.read_excel(
        file_obj,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12)
    )

    df = df.copy()

    if "S.No." in df.columns:

        null_indices = df[df["S.No."].isna()].index

        if len(null_indices) > 0:

            first_null_pos = df.index.get_loc(
                null_indices[0]
            )

            df = df.iloc[:first_null_pos]

    df.columns = (
        df.columns
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    # --------------------------------------------------------
    # TOTAL AREA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CLUSTER WEIGHTS
    # --------------------------------------------------------

    df_w = pd.read_excel(
        file_obj,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15]
    )

    df_w = df_w.copy()

    if "Clusters" in df_w.columns:

        null_indices = df_w[
            df_w["Clusters"].isna()
        ].index

        if len(null_indices) > 0:

            first_null_pos = df_w.index.get_loc(
                null_indices[0]
            )

            df_w = df_w.iloc[:first_null_pos]

    # --------------------------------------------------------
    # FORECAST CONFIG
    # --------------------------------------------------------

    df_st = pd.read_excel(
        file_obj,
        sheet_name="Forecast Config",
        header=8
    )

    lat = float(
        pd.to_numeric(
            df_st.loc[0, "Lat"],
            errors="coerce"
        )
    )

    # --------------------------------------------------------
    # TILT ANGLE
    # --------------------------------------------------------

    df_tilt = pd.read_excel(
        file_obj,
        sheet_name="Config Tilt Angle",
        header=7
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" in df_tilt.columns:

        null_indices = df_tilt[
            df_tilt["Fixed"].isna()
        ].index

        if len(null_indices) > 0:

            first_null_pos = df_tilt.index.get_loc(
                null_indices[0]
            )

            df_tilt = df_tilt.iloc[:first_null_pos]

    df_tilt = df_tilt.dropna(
        how="all",
        axis=1
    )

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month"
        }
    )

    month_lookup = {}

    if (
        "Month" in df_tilt.columns
        and "Fixed" in df_tilt.columns
    ):

        month_lookup = (
            df_tilt
            .set_index("Month")["Fixed"]
            .to_dict()
        )

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    df_ghi = pd.read_excel(
        file_obj,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5]
    )

    df_ghi = df_ghi.fillna(0)

    # --------------------------------------------------------
    # FIXED DATA
    # --------------------------------------------------------

    df_fix = pd.read_excel(
        file_obj,
        sheet_name="Fixed-C11",
        header=1
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    if "Date" in df_fix.columns:

        null_indices = df_fix[
            df_fix["Date"].isna()
        ].index

        if len(null_indices) > 0:

            first_null_pos = df_fix.index.get_loc(
                null_indices[0]
            )

            df_fix = df_fix.iloc[
                :first_null_pos
            ]

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df_fix["Date"] = pd.Timestamp.today()

    first_date = (
        pd.Timestamp.today()
        .replace(
            month=1,
            day=1
        )
        .normalize()
    )

    # --------------------------------------------------------
    # DECLINATION
    # --------------------------------------------------------

    day_number = (
        df_fix["Date"] - first_date
    ).dt.days + 1

    df_fix["Declination Angle ∆"] = (
        23.45
        *
        np.sin(
            np.radians(
                360
                *
                (284 + day_number)
                / 365
            )
        )
    )

    # --------------------------------------------------------
    # ELEVATION
    # --------------------------------------------------------

    df_fix["Elevation angle a"] = (
        90
        - lat
        + df_fix["Declination Angle ∆"]
    )

    # --------------------------------------------------------
    # TILT
    # --------------------------------------------------------

    df_fix["Tilt Angle b"] = (
        df_fix["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

    df_fix["Tilt Angle b"] = (
        pd.to_numeric(
            df_fix["Tilt Angle b"],
            errors="coerce"
        ).fillna(0)
    )

    df_fix["a+b"] = (
        df_fix["Elevation angle a"]
        +
        df_fix["Tilt Angle b"]
    )

    df_fix["SIN(a+b)"] = np.sin(
        np.radians(
            df_fix["a+b"]
        )
    )

    df_fix["Sin(a)"] = np.sin(
        np.radians(
            df_fix["Elevation angle a"]
        )
    )

    sin_a = df_fix["Sin(a)"].replace(
        0,
        np.nan
    )

    # --------------------------------------------------------
    # POA FOR EACH CLUSTER
    # --------------------------------------------------------

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15"
    ]

    poa_columns = []

    for i, ghi_col in enumerate(ghi_cols, start=1):

        poa_col = (
            "POA fixed"
            if i == 1
            else f"POA Fixed-C{i}"
        )

        df_fix[poa_col] = (
            df_ghi[ghi_col].to_numpy()
            *
            df_fix["SIN(a+b)"]
            /
            sin_a
        )

        df_fix[poa_col] = (
            df_fix[poa_col]
            .replace(
                [np.inf, -np.inf],
                0
            )
            .fillna(0)
        )

        poa_columns.append(
            poa_col
        )

    # --------------------------------------------------------
    # BACKEND CAL
    # --------------------------------------------------------

    backend_list = []

    for cluster in range(11, 16):

        backend_list.append(
            pd.read_excel(
                file_obj,
                sheet_name=f"Backend Cal C{cluster}"
            )
        )

    # --------------------------------------------------------
    # TRACKING DATA
    # --------------------------------------------------------

    df_trac = pd.read_excel(
        file_obj,
        sheet_name="Tracking",
        header=1
    )

    return (
        df,
        df_w,
        df_ghi,
        df_fix,
        df_trac,
        backend_list,
        ghi_cols,
        lat
    )


(
    df,
    df_w,
    df_ghi,
    df_fix,
    df_trac,
    backend_list,
    ghi_cols,
    lat
) = prepare_common_data(file_bytes)


# ============================================================
# ERROR CALCULATION
# ============================================================

def calculate_cluster_area(
    df,
    df_w,
    error
):

    work_df = df.copy()

    work_df["Error %"] = error

    work_df["Net Efficiency (%)"] = (
        pd.to_numeric(
            work_df["Standard PV Efficiency (%)"],
            errors="coerce"
        ).fillna(0)
        -
        error
    )

    work_df["Total area (m2)"] = (
        pd.to_numeric(
            work_df["No of Module"],
            errors="coerce"
        ).fillna(0)
        *
        pd.to_numeric(
            work_df["Area of 1 Module (m2)"],
            errors="coerce"
        ).fillna(0)
    )

    work_df["Eff Area"] = (
        work_df["Net Efficiency (%)"]
        *
        work_df["Total area (m2)"]
        / 100
    )

    cluster_sums = (
        work_df
        .groupby("Clusters")["Eff Area"]
        .sum()
    )

    weights = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0)
        .to_numpy(dtype=float)
    )

    return work_df, weights


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df,
    df_w,
    df_fix,
    error
):

    work_df, weights = calculate_cluster_area(
        df,
        df_w,
        error
    )

    poa_cols = [
        "POA fixed",
        "POA Fixed-C12",
        "POA Fixed-C13",
        "POA Fixed-C14",
        "POA Fixed-C15"
    ]

    forecast_components = []

    for poa_col, weight in zip(
        poa_cols,
        weights
    ):

        component = (
            df_fix[poa_col].to_numpy(
                dtype=float
            )
            *
            weight
            /
            1_000_000
        )

        forecast_components.append(
            component
        )

    forecast = np.sum(
        forecast_components,
        axis=0
    )

    return (
        forecast,
        work_df,
        weights
    )


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    df,
    df_w,
    df_ghi,
    df_fix,
    backend_list,
    error,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit
):

    _, weights = calculate_cluster_area(
        df,
        df_w,
        error
    )

    blocks = (
        backend_list[0]["Block No."]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not (
        start_block
        < max_block
        < end_block
    ):
        raise ValueError(
            "GHI blocks must satisfy: "
            "Starting < Maximum < Ending"
        )

    # --------------------------------------------------------
    # SLOPES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ZENITH
    # --------------------------------------------------------

    zenith = np.where(
        blocks <= max_block,

        np.minimum(
            89,
            m1
            *
            (
                blocks
                - max_block
            )
        ),

        np.minimum(
            89,
            m2
            *
            (
                blocks
                - max_block
            )
        )
    )

    # --------------------------------------------------------
    # PANEL ANGLE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # COS
    # --------------------------------------------------------

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None
    )

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    ghi_matrix = np.column_stack(
        [
            df_ghi[col]
            .to_numpy(dtype=float)
            for col in ghi_cols
        ]
    )

    # --------------------------------------------------------
    # DHI
    # --------------------------------------------------------

    dhi = (
        ghi_matrix
        *
        DHI
        / 100
    )

    # --------------------------------------------------------
    # DNI
    # --------------------------------------------------------

    dni = (
        ghi_matrix
        -
        dhi
    ) / cos_alpha[:, None]

    dni = np.nan_to_num(
        dni,
        nan=0,
        posinf=0,
        neginf=0
    )

    # --------------------------------------------------------
    # POWER
    # --------------------------------------------------------

    forecast = (
        dni @ weights
    ) / 1_000_000

    forecast = np.nan_to_num(
        forecast,
        nan=0,
        posinf=0,
        neginf=0
    )

    return (
        forecast,
        zenith,
        panel,
        weights
    )


# ============================================================
# PEAK ERROR
# ============================================================

def peak_metrics(
    forecast,
    actual
):

    forecast = np.asarray(
        forecast,
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float
    )

    actual_peak = np.nanmax(
        actual
    )

    calculated_peak = np.nanmax(
        forecast
    )

    peak_error = abs(
        calculated_peak
        -
        actual_peak
    )

    peak_error_pct = (
        peak_error
        /
        actual_peak
        *
        100
        if actual_peak != 0
        else 0
    )

    return (
        calculated_peak,
        actual_peak,
        peak_error,
        peak_error_pct
    )


# ============================================================
# AUTO ERROR OPTIMIZATION
# ============================================================

def optimize_error_fixed(
    df,
    df_w,
    df_fix,
    error_min,
    error_max,
    error_step
):

    actual = (
        df_fix["Actual"]
        .to_numpy(dtype=float)
    )

    errors = np.arange(
        error_min,
        error_max + error_step / 2,
        error_step
    )

    results = []

    for error in errors:

        forecast, _, _ = (
            calculate_fixed_forecast(
                df,
                df_w,
                df_fix,
                error
            )
        )

        (
            calculated_peak,
            actual_peak,
            peak_error,
            peak_error_pct
        ) = peak_metrics(
            forecast,
            actual
        )

        results.append(
            {
                "Error %": error,
                "Calculated Peak": calculated_peak,
                "Actual Peak": actual_peak,
                "Peak Error": peak_error,
                "Peak Error %": peak_error_pct
            }
        )

    result_df = pd.DataFrame(
        results
    )

    best_row = result_df.loc[
        result_df["Peak Error"].idxmin()
    ]

    return (
        float(best_row["Error %"]),
        result_df
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    df,
    df_w,
    df_ghi,
    df_fix,
    backend_list,
    error,
    bounds
):

    actual_full = (
        df_fix["Actual"]
        .to_numpy(dtype=float)
    )

    mask = actual_full != 0

    actual = actual_full[mask]

    actual_max = (
        actual.max()
        if len(actual)
        else 1
    )

    actual_sum = (
        actual.sum()
        if len(actual)
        else 1
    )

    blocks = (
        backend_list[0]["Block No."]
        .to_numpy(dtype=float)
    )

    ghi_matrix = np.column_stack(
        [
            df_ghi[col]
            .to_numpy(dtype=float)
            for col in ghi_cols
        ]
    )

    _, weights = calculate_cluster_area(
        df,
        df_w,
        error
    )

    def objective(x):

        DHI = int(round(x[0]))
        start_block = int(round(x[1]))
        end_block = int(round(x[2]))
        max_block = int(round(x[3]))
        east_limit = int(round(x[4]))
        west_limit = int(round(x[5]))

        if not (
            start_block
            < max_block
            < end_block
        ):
            return 1e9

        try:

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

            zenith = np.where(
                blocks <= max_block,

                np.minimum(
                    89,
                    m1
                    *
                    (
                        blocks
                        - max_block
                    )
                ),

                np.minimum(
                    89,
                    m2
                    *
                    (
                        blocks
                        - max_block
                    )
                )
            )

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

            cos_alpha = np.cos(
                np.radians(panel)
            )

            cos_alpha = np.clip(
                cos_alpha,
                1e-6,
                None
            )

            dhi = (
                ghi_matrix
                *
                DHI
                / 100
            )

            dni = (
                ghi_matrix
                -
                dhi
            ) / cos_alpha[:, None]

            prediction_full = (
                dni @ weights
            ) / 1_000_000

            prediction_full = np.nan_to_num(
                prediction_full,
                nan=0,
                posinf=0,
                neginf=0
            )

            prediction = (
                prediction_full[mask]
            )

            if len(prediction) == 0:
                return 1e9

            block_error = (
                np.mean(
                    np.abs(
                        actual
                        -
                        prediction
                    )
                )
                /
                actual_max
            )

            peak_error = (
                abs(
                    actual_max
                    -
                    prediction.max()
                )
                /
                actual_max
            )

            energy_error = (
                abs(
                    actual_sum
                    -
                    prediction.sum()
                )
                /
                actual_sum
            )

            return (
                0.80 * block_error
                +
                0.10 * peak_error
                +
                0.10 * energy_error
            )

        except Exception:

            return 1e9

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

    best = (
        np.round(result.x)
        .astype(int)
    )

    return best, result.fun


# ============================================================
# RUN BUTTON
# ============================================================

run_calculation = st.sidebar.button(
    "🚀 Run Calculation",
    type="primary",
    use_container_width=True
)


if not run_calculation:

    st.info(
        "Configure the plant type and parameters in the sidebar, "
        "then click **Run Calculation**."
    )

    st.stop()


# ============================================================
# MAIN CALCULATION
# ============================================================

actual = (
    df_fix["Actual"]
    .to_numpy(dtype=float)
)


# ============================================================
# FIXED PLANT
# ============================================================

if plant_type == "Fixed":

    st.subheader("Fixed Plant Configuration")

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if error_mode == "Automatic":

        with st.spinner(
            "Finding best Error %..."
        ):

            best_error, error_results = (
                optimize_error_fixed(
                    df,
                    df_w,
                    df_fix,
                    error_min,
                    error_max,
                    error_step
                )
            )

    else:

        best_error = manual_error

        error_results = pd.DataFrame()

    # --------------------------------------------------------
    # FINAL FORECAST
    # --------------------------------------------------------

    forecast, final_df, weights = (
        calculate_fixed_forecast(
            df,
            df_w,
            df_fix,
            best_error
        )
    )

    (
        calculated_peak,
        actual_peak,
        peak_error,
        peak_error_pct
    ) = peak_metrics(
        forecast,
        actual
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Best Error %",
        f"{best_error:.2f}%"
    )

    c2.metric(
        "Forecast Peak",
        f"{calculated_peak:.3f}"
    )

    c3.metric(
        "Actual Peak",
        f"{actual_peak:.3f}"
    )

    c4.metric(
        "Peak Error",
        f"{peak_error_pct:.2f}%"
    )

    # --------------------------------------------------------
    # ERROR SEARCH TABLE
    # --------------------------------------------------------

    if error_mode == "Automatic":

        with st.expander(
            "View Error % Optimization"
        ):

            st.dataframe(
                error_results.style.format(
                    {
                        "Error %": "{:.2f}",
                        "Calculated Peak": "{:.3f}",
                        "Actual Peak": "{:.3f}",
                        "Peak Error": "{:.3f}",
                        "Peak Error %": "{:.2f}%"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(forecast)),
            y=forecast,
            mode="lines",
            name="Forecast"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(actual)),
            y=actual,
            mode="lines",
            name="Actual"
        )
    )

    fig.update_layout(
        title="Fixed Plant: Forecast vs Actual",
        xaxis_title="Block",
        yaxis_title="Power",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # EFFECTIVE AREA
    # --------------------------------------------------------

    st.subheader(
        "Cluster Effective Area"
    )

    area_display = pd.DataFrame(
        {
            "Cluster": df_w["Clusters"],
            "Effective Area (m²)": weights
        }
    )

    st.dataframe(
        area_display.style.format(
            {
                "Effective Area (m²)": "{:,.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TRACKING PLANT
# ============================================================

else:

    st.subheader(
        "Tracking Plant Configuration"
    )

    # --------------------------------------------------------
    # TRACKING PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        "### Tracking Parameters"
    )

    param_col1, param_col2 = st.columns(2)

    with param_col1:

        DHI = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

        GHI_Starting_Block = st.number_input(
            "GHI Starting Block",
            min_value=1,
            max_value=95,
            value=10,
            step=1
        )

        GHI_Ending_Block = st.number_input(
            "GHI Ending Block",
            min_value=2,
            max_value=96,
            value=80,
            step=1
        )

    with param_col2:

        GHI_Max_Block = st.number_input(
            "GHI Max Block",
            min_value=2,
            max_value=95,
            value=50,
            step=1
        )

        Tracking_angle_lim_E = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            value=45,
            step=1
        )

        Tracking_angle_lim_W = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            value=45,
            step=1
        )

    # --------------------------------------------------------
    # AUTO OPTIMIZATION
    # --------------------------------------------------------

    auto_tracking = st.checkbox(
        "Automatically optimize tracking parameters",
        value=True
    )

    if auto_tracking:

        st.markdown(
            "### Optimization Bounds"
        )

        b1, b2, b3 = st.columns(3)

        with b1:

            dhi_min = st.number_input(
                "DHI Min",
                0,
                50,
                0
            )

            dhi_max = st.number_input(
                "DHI Max",
                1,
                50,
                10
            )

        with b2:

            start_min = st.number_input(
                "Start Min",
                1,
                50,
                10
            )

            start_max = st.number_input(
                "Start Max",
                2,
                60,
                30
            )

        with b3:

            end_min = st.number_input(
                "End Min",
                40,
                95,
                65
            )

            end_max = st.number_input(
                "End Max",
                41,
                96,
                80
            )

        b4, b5, b6 = st.columns(3)

        with b4:

            max_min = st.number_input(
                "Max Block Min",
                1,
                80,
                47
            )

            max_max = st.number_input(
                "Max Block Max",
                2,
                95,
                53
            )

        with b5:

            east_min = st.number_input(
                "East Limit Min",
                0,
                90,
                10
            )

            east_max = st.number_input(
                "East Limit Max",
                1,
                90,
                70
            )

        with b6:

            west_min = st.number_input(
                "West Limit Min",
                0,
                90,
                10
            )

            west_max = st.number_input(
                "West Limit Max",
                1,
                90,
                70
            )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if error_mode == "Automatic":

        with st.spinner(
            "Finding best Error %..."
        ):

            best_error, error_results = (
                optimize_error_fixed(
                    df,
                    df_w,
                    df_fix,
                    error_min,
                    error_max,
                    error_step
                )
            )

    else:

        best_error = manual_error
        error_results = pd.DataFrame()

    st.info(
        f"Using Error % = **{best_error:.2f}%**"
    )

    # --------------------------------------------------------
    # TRACKING OPTIMIZATION
    # --------------------------------------------------------

    if auto_tracking:

        bounds = [
            (dhi_min, dhi_max),
            (start_min, start_max),
            (end_min, end_max),
            (max_min, max_max),
            (east_min, east_max),
            (west_min, west_max)
        ]

        with st.spinner(
            "Optimizing tracking parameters..."
        ):

            best_params, optimization_score = (
                optimize_tracking(
                    df,
                    df_w,
                    df_ghi,
                    df_fix,
                    backend_list,
                    best_error,
                    bounds
                )
            )

        DHI = int(best_params[0])
        GHI_Starting_Block = int(best_params[1])
        GHI_Ending_Block = int(best_params[2])
        GHI_Max_Block = int(best_params[3])
        Tracking_angle_lim_E = int(best_params[4])
        Tracking_angle_lim_W = int(best_params[5])

        st.success(
            "Tracking optimization completed."
        )

    # --------------------------------------------------------
    # FINAL TRACKING FORECAST
    # --------------------------------------------------------

    try:

        (
            forecast,
            zenith,
            panel,
            weights
        ) = calculate_tracking_forecast(
            df,
            df_w,
            df_ghi,
            df_fix,
            backend_list,
            best_error,
            DHI,
            GHI_Starting_Block,
            GHI_Ending_Block,
            GHI_Max_Block,
            Tracking_angle_lim_E,
            Tracking_angle_lim_W
        )

    except Exception as e:

        st.error(
            f"Tracking calculation failed: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    (
        calculated_peak,
        actual_peak,
        peak_error,
        peak_error_pct
    ) = peak_metrics(
        forecast,
        actual
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Error %",
        f"{best_error:.2f}%"
    )

    c2.metric(
        "Forecast Peak",
        f"{calculated_peak:.3f}"
    )

    c3.metric(
        "Actual Peak",
        f"{actual_peak:.3f}"
    )

    c4.metric(
        "Peak Error",
        f"{peak_error_pct:.2f}%"
    )

    # --------------------------------------------------------
    # PARAMETERS RESULT
    # --------------------------------------------------------

    st.subheader(
        "Final Tracking Parameters"
    )

    parameter_df = pd.DataFrame(
        {
            "Parameter": [
                "DHI (%)",
                "GHI Starting Block",
                "GHI Ending Block",
                "GHI Max Block",
                "Tracking East Limit",
                "Tracking West Limit"
            ],
            "Value": [
                DHI,
                GHI_Starting_Block,
                GHI_Ending_Block,
                GHI_Max_Block,
                Tracking_angle_lim_E,
                Tracking_angle_lim_W
            ]
        }
    )

    st.dataframe(
        parameter_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(forecast)),
            y=forecast,
            mode="lines",
            name="Forecast"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(actual)),
            y=actual,
            mode="lines",
            name="Actual"
        )
    )

    fig.update_layout(
        title="Tracking Plant: Forecast vs Actual",
        xaxis_title="Block",
        yaxis_title="Power",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TRACKING ANGLES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig_angle = go.Figure()

        fig_angle.add_trace(
            go.Scatter(
                x=np.arange(len(zenith)),
                y=zenith,
                mode="lines",
                name="Zenith Angle"
            )
        )

        fig_angle.add_trace(
            go.Scatter(
                x=np.arange(len(panel)),
                y=panel,
                mode="lines",
                name="Panel Angle"
            )
        )

        fig_angle.update_layout(
            title="Tracking Angles",
            xaxis_title="Block",
            yaxis_title="Angle (°)",
            height=400
        )

        st.plotly_chart(
            fig_angle,
            use_container_width=True
        )

    with col2:

        area_display = pd.DataFrame(
            {
                "Cluster": df_w["Clusters"],
                "Effective Area (m²)": weights
            }
        )

        st.markdown(
            "### Cluster Effective Area"
        )

        st.dataframe(
            area_display.style.format(
                {
                    "Effective Area (m²)": "{:,.2f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.divider()

st.subheader(
    "Download Results"
)


@st.cache_data
def create_output_excel(
    forecast,
    actual,
    plant_type,
    best_error,
    df_w,
    df_fix,
    zenith=None,
    panel=None
):

    output = io.BytesIO()

    result_df = pd.DataFrame(
        {
            "Block": np.arange(
                len(forecast)
            ),
            "Forecast": forecast,
            "Actual": actual
        }
    )

    result_df["Error"] = (
        result_df["Forecast"]
        -
        result_df["Actual"]
    )

    result_df["Absolute Error"] = (
        result_df["Error"]
        .abs()
    )

    if plant_type == "Tracking":

        result_df["Zenith Angle"] = zenith
        result_df["Panel Angle"] = panel

    summary_df = pd.DataFrame(
        {
            "Parameter": [
                "Plant Type",
                "Error %",
                "Forecast Peak",
                "Actual Peak",
                "Peak Error",
                "Peak Error %"
            ],
            "Value": [
                plant_type,
                best_error,
                forecast.max(),
                actual.max(),
                abs(
                    forecast.max()
                    -
                    actual.max()
                ),
                (
                    abs(
                        forecast.max()
                        -
                        actual.max()
                    )
                    /
                    actual.max()
                    *
                    100
                    if actual.max() != 0
                    else 0
                )
            ]
        }
    )

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        result_df.to_excel(
            writer,
            sheet_name="Forecast",
            index=False
        )

        df_w.to_excel(
            writer,
            sheet_name="Cluster Area",
            index=False
        )

        df_fix.to_excel(
            writer,
            sheet_name="Calculation",
            index=False
        )

    output.seek(0)

    return output.getvalue()


if plant_type == "Tracking":

    excel_output = create_output_excel(
        forecast,
        actual,
        plant_type,
        best_error,
        df_w,
        df_fix,
        zenith,
        panel
    )

else:

    excel_output = create_output_excel(
        forecast,
        actual,
        plant_type,
        best_error,
        df_w,
        df_fix
    )


st.download_button(
    label="⬇️ Download Excel Report",
    data=excel_output,
    file_name=(
        f"Solar_Forecast_"
        f"{plant_type}_Correction.xlsx"
    ),
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    use_container_width=True
)
