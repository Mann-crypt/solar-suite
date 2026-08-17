# ============================================================
# STREAMLIT APP
# FIXED / TRACKING LOSS CORRECTION MODEL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Loss Correction Model",
    page_icon="☀️",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

CLUSTERS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

MAX_OPT_ITER = 40
OPT_POPSIZE = 15

TRACKING_BOUNDS = [
    (0, 10),       # DHI
    (10, 30),      # Starting Block
    (65, 80),      # Ending Block
    (47, 53),      # Max Block
    (10, 70),      # East Limit
    (10, 70),      # West Limit
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #8b949e;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.04);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "input_df" not in st.session_state:
    st.session_state.input_df = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# HELPERS
# ============================================================

def validate_columns(df, required, name="Data"):

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} is missing: "
            f"{', '.join(missing)}"
        )


def get_sheet_names(uploaded_file):

    uploaded_file.seek(0)

    return pd.ExcelFile(
        uploaded_file
    ).sheet_names


# ============================================================
# READ AREA & EFFICIENCY
# ============================================================

def read_area_efficiency(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False
        )
        .str.strip()
    )

    validate_columns(
        df,
        [
            "S.No.",
            "Standard PV Efficiency (%)",
            "No of Module",
            "Area of 1 Module (m2)",
        ],
        "Area & Efficiency",
    )

    df = df[
        df["S.No."].notna()
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# ============================================================
# EFFECTIVE AREAS
# ============================================================

def read_effective_areas(uploaded_file):

    uploaded_file.seek(0)

    area_df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=None,
    )

    fixed_weights = (
        pd.to_numeric(
            area_df.iloc[2:7, 15],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    tracking_weights = (
        pd.to_numeric(
            area_df.iloc[28:33, 15],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    if len(fixed_weights) != 5:

        raise ValueError(
            "Could not read 5 fixed effective areas."
        )

    if len(tracking_weights) != 5:

        raise ValueError(
            "Could not read 5 tracking effective areas."
        )

    return (
        fixed_weights,
        tracking_weights,
    )


# ============================================================
# LATITUDE
# ============================================================

def read_latitude(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        ["Lat"],
        "Forecast Config",
    )

    return float(
        df["Lat"].iloc[0]
    )


# ============================================================
# TILT
# ============================================================

def read_tilt_lookup(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    df = df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    validate_columns(
        df,
        [
            "Month_Num",
            "Fixed",
        ],
        "Config Tilt Angle",
    )

    df["Month_Num"] = pd.to_numeric(
        df["Month_Num"],
        errors="coerce",
    )

    df["Fixed"] = pd.to_numeric(
        df["Fixed"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Month_Num"]
    )

    return (
        df
        .set_index("Month_Num")["Fixed"]
        .to_dict()
    )


# ============================================================
# READ GHI
# ============================================================

def read_ghi(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=range(6),
    )

    df.columns = [
        "Block",
        *GHI_COLS,
    ]

    df = df[
        pd.to_numeric(
            df["Block"],
            errors="coerce",
        ).notna()
    ].copy()

    df["Block"] = pd.to_numeric(
        df["Block"],
        errors="coerce",
    )

    for col in GHI_COLS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

    return df.reset_index(
        drop=True
    )


# ============================================================
# READ FIXED-C11
# ============================================================

def read_fixed_data(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-C11",
        header=1,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        [
            "Date",
            "Actual",
        ],
        "Fixed-C11",
    )

    date_valid = df[
        "Date"
    ].notna()

    if not date_valid.any():

        raise ValueError(
            "No valid Date rows found."
        )

    first_blank = np.where(
        ~date_valid.to_numpy()
    )[0]

    if len(first_blank):

        df = df.iloc[
            :first_blank[0]
        ].copy()

    else:

        df = df.loc[
            date_valid
        ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    return df


# ============================================================
# READ TRACKING SHEET
# ============================================================

def read_tracking_sheet(uploaded_file, n):

    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        sheet_name="Tracking",
        header=1,
    )

    df = df.iloc[
        :n
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# ============================================================
# READ BACKEND BLOCKS
# ============================================================

def read_backend_blocks(uploaded_file, cluster):

    uploaded_file.seek(0)

    sheet = (
        f"Backend Cal {cluster}"
    )

    df = pd.read_excel(
        uploaded_file,
        sheet_name=sheet,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        ["Block No."],
        sheet,
    )

    blocks = pd.to_numeric(
        df["Block No."],
        errors="coerce",
    )

    return blocks.to_numpy(
        dtype=float
    )


# ============================================================
# ALIGN DATA
# ============================================================

def align_data(
    fixed_df,
    ghi_df,
):

    n = min(
        len(fixed_df),
        len(ghi_df),
    )

    if n == 0:

        raise ValueError(
            "No valid rows available."
        )

    fixed_df = fixed_df.iloc[
        :n
    ].copy()

    ghi_df = ghi_df.iloc[
        :n
    ].copy()

    return (
        fixed_df,
        ghi_df,
        n,
    )


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def calculate_solar_geometry(
    dates,
    lat,
    tilt_lookup,
):

    dates = pd.to_datetime(
        dates
    )

    first_date = pd.Timestamp(
        year=2025,
        month=1,
        day=1,
    )

    day_offset = (
        dates
        - first_date
    ).dt.days.to_numpy(
        dtype=float
    )

    declination = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_offset
                    + 1
                )
                / 365
            )
        )
    )

    elevation = (
        90
        - lat
        + declination
    )

    months = dates.dt.month.to_numpy()

    tilt = np.array([
        tilt_lookup.get(
            float(month),
            0
        )
        for month in months
    ])

    a_plus_b = (
        elevation
        + tilt
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(a_plus_b)
    )

    sin_a_safe = np.where(
        np.abs(sin_a) < 1e-8,
        1e-8,
        sin_a,
    )

    return (
        declination,
        elevation,
        tilt,
        sin_a_safe,
        sin_ab,
    )


# ============================================================
# CALCULATE EFFICIENCY LOSS
# ============================================================

def optimize_fixed_efficiency_loss(
    standard_efficiency,
    fixed_weights,
    poa_matrix,
    actual,
):

    valid_mask = (
        np.isfinite(actual)
        & (actual != 0)
    )

    if not valid_mask.any():

        raise ValueError(
            "Actual power contains no valid non-zero values."
        )

    actual_day = actual[
        valid_mask
    ]

    actual_peak = np.max(
        actual_day
    )

    if actual_peak <= 0:

        raise ValueError(
            "Actual peak must be greater than zero."
        )

    max_loss = np.min(
        standard_efficiency
    )

    results = []

    loss_values = np.arange(
        0,
        max_loss + 0.0001,
        0.1,
    )

    for loss in loss_values:

        net_efficiency = (
            standard_efficiency
            - loss
        )

        net_efficiency = np.maximum(
            net_efficiency,
            0,
        )

        efficiency_factor = np.divide(
            net_efficiency,
            standard_efficiency,
            out=np.zeros_like(
                net_efficiency
            ),
            where=(
                standard_efficiency != 0
            ),
        )

        adjusted_weights = (
            fixed_weights
            * efficiency_factor
        )

        power_matrix = (
            poa_matrix
            * adjusted_weights[None, :]
            / 1_000_000
        )

        predicted = (
            power_matrix.sum(
                axis=1
            )
        )

        predicted_day = predicted[
            valid_mask
        ]

        if len(predicted_day) == 0:
            continue

        predicted_peak = np.max(
            predicted_day
        )

        peak_error = abs(
            actual_peak
            - predicted_peak
        )

        peak_error_percent = (
            peak_error
            / actual_peak
            * 100
        )

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    - predicted_day
                )
            )
            / actual_peak
        )

        actual_energy = np.sum(
            actual_day
        )

        predicted_energy = np.sum(
            predicted_day
        )

        energy_error = abs(
            actual_energy
            - predicted_energy
        ) / actual_energy

        score = (
            0.80 * block_error
            + 0.10 * (
                peak_error
                / actual_peak
            )
            + 0.10 * energy_error
        )

        results.append({

            "Error %": loss,

            "Actual Peak": actual_peak,

            "Predicted Peak": predicted_peak,

            "Peak Error": peak_error,

            "Peak Error (%)":
                peak_error_percent,

            "Block Error":
                block_error,

            "Energy Error":
                energy_error,

            "Overall Score":
                score,
        })

    results_df = pd.DataFrame(
        results
    )

    best_row = results_df.loc[
        results_df[
            "Peak Error"
        ].idxmin()
    ]

    return (
        float(best_row["Error %"]),
        results_df,
        valid_mask,
        actual_peak,
    )


# ============================================================
# FIXED MODEL
# ============================================================

def run_fixed_model(
    standard_efficiency,
    fixed_weights,
    ghi_matrix,
    sin_a,
    sin_ab,
    actual,
):

    fixed_poa = (
        ghi_matrix
        * sin_ab[:, None]
        / sin_a[:, None]
    )

    (
        best_loss,
        loss_results,
        valid_mask,
        actual_peak,
    ) = optimize_fixed_efficiency_loss(
        standard_efficiency,
        fixed_weights,
        fixed_poa,
        actual,
    )

    net_efficiency = (
        standard_efficiency
        - best_loss
    )

    net_efficiency = np.maximum(
        net_efficiency,
        0,
    )

    efficiency_factor = np.divide(
        net_efficiency,
        standard_efficiency,
        out=np.zeros_like(
            standard_efficiency
        ),
        where=(
            standard_efficiency != 0
        ),
    )

    final_weights = (
        fixed_weights
        * efficiency_factor
    )

    power_matrix = (
        fixed_poa
        * final_weights[None, :]
        / 1_000_000
    )

    forecast = (
        power_matrix.sum(
            axis=1
        )
    )

    actual_day = actual[
        valid_mask
    ]

    forecast_day = forecast[
        valid_mask
    ]

    block_error = (
        np.mean(
            np.abs(
                actual_day
                - forecast_day
            )
        )
        / actual_peak
    )

    peak_error = (
        abs(
            actual_peak
            - forecast_day.max()
        )
        / actual_peak
    )

    energy_error = (
        abs(
            actual_day.sum()
            - forecast_day.sum()
        )
        / actual_day.sum()
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return {
        "forecast": forecast,
        "power_matrix": power_matrix,
        "best_loss": best_loss,
        "loss_results": loss_results,
        "block_error": block_error,
        "peak_error": peak_error,
        "energy_error": energy_error,
        "score": score,
        "peak": forecast_day.max(),
        "net_efficiency": net_efficiency,
        "final_weights": final_weights,
    }


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
    blocks,
    ghi_matrix,
    tracking_weights,
):

    if not (
        start_block
        < max_block
        < end_block
    ):

        return None

    denominator_1 = (
        start_block
        - 1
        - max_block
    )

    denominator_2 = (
        end_block
        + 1
        - max_block
    )

    if (
        denominator_1 == 0
        or denominator_2 == 0
    ):

        return None

    m1 = (
        90
        / denominator_1
    )

    m2 = (
        90
        / denominator_2
    )

    zenith = np.where(

        blocks <= max_block,

        np.minimum(
            89,
            m1
            * (
                blocks
                - max_block
            ),
        ),

        np.minimum(
            89,
            m2
            * (
                blocks
                - max_block
            ),
        ),
    )

    panel = np.where(

        blocks < max_block,

        np.minimum(
            zenith,
            abs(east_limit),
        ),

        np.where(
            (
                (blocks > max_block)
                & (zenith > west_limit)
            ),
            west_limit,
            zenith,
        ),
    )

    cos_alpha = np.clip(
        np.cos(
            np.radians(panel)
        ),
        1e-6,
        None,
    )

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    power_matrix = (
        dni
        * tracking_weights[None, :]
        / 1_000_000
    )

    forecast = (
        power_matrix.sum(
            axis=1
        )
    )

    return (
        forecast,
        power_matrix,
        zenith,
        panel,
        dni,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    blocks,
    ghi_matrix,
    tracking_weights,
    actual,
):

    valid_mask = (
        np.isfinite(actual)
        & (actual != 0)
    )

    actual_day = actual[
        valid_mask
    ]

    actual_peak = np.max(
        actual_day
    )

    def objective(x):

        DHI = int(
            round(x[0])
        )

        start_block = int(
            round(x[1])
        )

        end_block = int(
            round(x[2])
        )

        max_block = int(
            round(x[3])
        )

        east_limit = int(
            round(x[4])
        )

        west_limit = int(
            round(x[5])
        )

        result = calculate_tracking(

            DHI,
            start_block,
            end_block,
            max_block,
            east_limit,
            west_limit,
            blocks,
            ghi_matrix,
            tracking_weights,
        )

        if result is None:
            return 1e9

        prediction = result[0]

        if not np.all(
            np.isfinite(prediction)
        ):
            return 1e9

        prediction_day = prediction[
            valid_mask
        ]

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    - prediction_day
                )
            )
            / actual_peak
        )

        peak_error = (
            abs(
                actual_peak
                - prediction_day.max()
            )
            / actual_peak
        )

        energy_error = (
            abs(
                actual_day.sum()
                - prediction_day.sum()
            )
            / actual_day.sum()
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(
        objective,
        bounds=TRACKING_BOUNDS,
        strategy="best1bin",
        maxiter=MAX_OPT_ITER,
        popsize=OPT_POPSIZE,
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    best = np.rint(
        result.x
    ).astype(int)

    params = {
        "DHI": best[0],
        "start": best[1],
        "end": best[2],
        "max": best[3],
        "east": best[4],
        "west": best[5],
    }

    return params


# ============================================================
# FINAL TRACKING MODEL
# ============================================================

def run_tracking_model(
    blocks,
    ghi_matrix,
    tracking_weights,
    actual,
    fixed_loss,
):

    with st.spinner(
        "🔄 Optimizing tracking parameters..."
    ):

        params = optimize_tracking(
            blocks,
            ghi_matrix,
            tracking_weights,
            actual,
        )

    result = calculate_tracking(
        params["DHI"],
        params["start"],
        params["end"],
        params["max"],
        params["east"],
        params["west"],
        blocks,
        ghi_matrix,
        tracking_weights,
    )

    (
        forecast,
        power_matrix,
        zenith,
        panel,
        dni,
    ) = result

    valid_mask = (
        np.isfinite(actual)
        & (actual != 0)
    )

    actual_day = actual[
        valid_mask
    ]

    forecast_day = forecast[
        valid_mask
    ]

    actual_peak = actual_day.max()

    block_error = (
        np.mean(
            np.abs(
                actual_day
                - forecast_day
            )
        )
        / actual_peak
    )

    peak_error = (
        abs(
            actual_peak
            - forecast_day.max()
        )
        / actual_peak
    )

    energy_error = (
        abs(
            actual_day.sum()
            - forecast_day.sum()
        )
        / actual_day.sum()
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return {
        "forecast": forecast,
        "power_matrix": power_matrix,
        "zenith": zenith,
        "panel": panel,
        "dni": dni,
        "params": params,
        "fixed_loss": fixed_loss,
        "block_error": block_error,
        "peak_error": peak_error,
        "energy_error": energy_error,
        "score": score,
        "peak": forecast_day.max(),
    }


# ============================================================
# CHART
# ============================================================

def show_forecast_chart(
    actual,
    forecast,
    title,
):

    n = min(
        len(actual),
        len(forecast),
    )

    x = np.arange(
        1,
        n + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual[:n],
            mode="lines",
            name="Actual",
            line=dict(
                color="#EF4444",
                width=2.5,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast[:n],
            mode="lines",
            name="Forecast",
            line=dict(
                color="#3B82F6",
                width=2.5,
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=500,
        hovermode="x unified",
        template="plotly_white",
        xaxis_title="15 Minute Block",
        yaxis_title="Power (MW)",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# METRICS
# ============================================================

def show_metrics(
    actual,
    model,
):

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Actual Peak",
        f"{np.max(actual):.3f} MW",
    )

    c2.metric(
        "Forecast Peak",
        f"{model['peak']:.3f} MW",
    )

    c3.metric(
        "Block Error",
        f"{model['block_error'] * 100:.2f}%",
    )

    c4.metric(
        "Peak Error",
        f"{model['peak_error'] * 100:.2f}%",
    )

    c5.metric(
        "Overall Score",
        f"{model['score']:.5f}",
    )


# ============================================================
# CLUSTER RESULT TABLE
# ============================================================

def cluster_power_table(
    power_matrix,
    mode,
):

    data = {}

    for i, cluster in enumerate(CLUSTERS):

        data[
            f"{cluster} {mode} Power (MW)"
        ] = power_matrix[:, i]

    return pd.DataFrame(data)


# ============================================================
# FIXED OUTPUT
# ============================================================

def show_fixed_output(
    actual,
    fixed,
):

    st.markdown(
        '<div class="section-title">'
        '🏗️ Fixed Model Results'
        '</div>',
        unsafe_allow_html=True,
    )

    show_metrics(
        actual,
        fixed,
    )

    st.markdown(
        '<div class="section-title">'
        '📈 Fixed Forecast vs Actual'
        '</div>',
        unsafe_allow_html=True,
    )

    show_forecast_chart(
        actual,
        fixed["forecast"],
        "Fixed Forecast vs Actual",
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            '<div class="section-title">'
            '⚡ Fixed Parameters'
            '</div>',
            unsafe_allow_html=True,
        )

        st.metric(
            "Optimized Efficiency Loss",
            f"{fixed['best_loss']:.2f}%",
        )

    with c2:

        st.markdown(
            '<div class="section-title">'
            '📊 Error Metrics'
            '</div>',
            unsafe_allow_html=True,
        )

        st.write(
            {
                "Block Error":
                    f"{fixed['block_error'] * 100:.4f}%",
                "Peak Error":
                    f"{fixed['peak_error'] * 100:.4f}%",
                "Energy Error":
                    f"{fixed['energy_error'] * 100:.4f}%",
                "Overall Score":
                    f"{fixed['score']:.6f}",
            }
        )

    st.markdown(
        '<div class="section-title">'
        '🏭 Cluster-wise Fixed Power'
        '</div>',
        unsafe_allow_html=True,
    )

    fixed_table = cluster_power_table(
        fixed["power_matrix"],
        "Fixed",
    )

    st.dataframe(
        fixed_table.round(4),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "📉 View Efficiency Loss Optimization Results"
    ):

        display = fixed[
            "loss_results"
        ].copy()

        st.dataframe(
            display.round(6),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TRACKING OUTPUT
# ============================================================

def show_tracking_output(
    actual,
    tracking,
):

    st.markdown(
        '<div class="section-title">'
        '🔄 Tracking Model Results'
        '</div>',
        unsafe_allow_html=True,
    )

    show_metrics(
        actual,
        tracking,
    )

    st.markdown(
        '<div class="section-title">'
        '📈 Tracking Forecast vs Actual'
        '</div>',
        unsafe_allow_html=True,
    )

    show_forecast_chart(
        actual,
        tracking["forecast"],
        "Tracking Forecast vs Actual",
    )

    st.markdown(
        '<div class="section-title">'
        '⚙️ Optimized Tracking Parameters'
        '</div>',
        unsafe_allow_html=True,
    )

    params = tracking[
        "params"
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "DHI",
        f"{params['DHI']}%",
    )

    c2.metric(
        "Starting Block",
        params["start"],
    )

    c3.metric(
        "Ending Block",
        params["end"],
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Max Block",
        params["max"],
    )

    c2.metric(
        "East Limit",
        f"{params['east']}°",
    )

    c3.metric(
        "West Limit",
        f"{params['west']}°",
    )

    st.markdown(
        '<div class="section-title">'
        '🏭 Cluster-wise Tracking Power'
        '</div>',
        unsafe_allow_html=True,
    )

    tracking_table = cluster_power_table(
        tracking["power_matrix"],
        "Tracking",
    )

    st.dataframe(
        tracking_table.round(4),
        use_container_width=True,
        hide_index=True,
    )

    angle_df = pd.DataFrame({

        "Block": np.arange(
            1,
            len(tracking["zenith"]) + 1,
        ),

        "Zenith Angle":
            tracking["zenith"],

        "Panel Angle":
            tracking["panel"],

    })

    with st.expander(
        "📐 View Tracking Angles"
    ):

        st.dataframe(
            angle_df.round(3),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# DOWNLOAD EXCEL
# ============================================================

def create_output_excel(
    fixed,
    tracking,
    actual,
):

    from io import BytesIO

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        summary = pd.DataFrame({

            "Metric": [

                "Efficiency Loss (%)",
                "Block Error (%)",
                "Peak Error (%)",
                "Energy Error (%)",
                "Overall Score",
                "Peak Power (MW)",
            ],

            "Fixed": [

                fixed["best_loss"],
                fixed["block_error"] * 100,
                fixed["peak_error"] * 100,
                fixed["energy_error"] * 100,
                fixed["score"],
                fixed["peak"],
            ],

            "Tracking": [

                fixed["best_loss"],
                tracking["block_error"] * 100,
                tracking["peak_error"] * 100,
                tracking["energy_error"] * 100,
                tracking["score"],
                tracking["peak"],
            ],
        })

        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        fixed_df = pd.DataFrame({

            "Actual": actual,
            "Fixed Forecast": fixed["forecast"],
        })

        for i, cluster in enumerate(
            CLUSTERS
        ):

            fixed_df[
                f"{cluster} Fixed Power"
            ] = fixed[
                "power_matrix"
            ][:, i]

        fixed_df.to_excel(
            writer,
            sheet_name="Fixed",
            index=False,
        )

        tracking_df = pd.DataFrame({

            "Actual": actual,
            "Tracking Forecast":
                tracking["forecast"],
            "Zenith Angle":
                tracking["zenith"],
            "Panel Angle":
                tracking["panel"],
        })

        for i, cluster in enumerate(
            CLUSTERS
        ):

            tracking_df[
                f"{cluster} Tracking Power"
            ] = tracking[
                "power_matrix"
            ][:, i]

        tracking_df.to_excel(
            writer,
            sheet_name="Tracking",
            index=False,
        )

    output.seek(0)

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    st.markdown(
        '<div class="main-title">'
        '☀️ Solar Loss Correction Model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Fixed and Tracking cluster-based power forecasting'
        '</div>',
        unsafe_allow_html=True,
    )


    # ========================================================
    # FILE UPLOAD
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📁 Input Workbook'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel Workbook",
        type=["xlsx", "xls"],
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload the Excel workbook to begin."
        )

        return


    # ========================================================
    # LOAD WORKBOOK
    # ========================================================

    try:

        sheets = get_sheet_names(
            uploaded_file
        )

        required_sheets = [
            "Area & Efficiency",
            "Forecast Config",
            "Config Tilt Angle",
            "Result",
            "Fixed-C11",
            "Tracking",
            "Backend Cal C11",
            "Backend Cal C12",
            "Backend Cal C13",
            "Backend Cal C14",
            "Backend Cal C15",
        ]

        missing = [
            s
            for s in required_sheets
            if s not in sheets
        ]

        if missing:

            st.error(
                "Missing sheets: "
                + ", ".join(missing)
            )

            return

    except Exception as e:

        st.error(
            f"Unable to read workbook: {e}"
        )

        return


    # ========================================================
    # READ DATA
    # ========================================================

    try:

        area_df = read_area_efficiency(
            uploaded_file
        )

        fixed_weights, tracking_weights = (
            read_effective_areas(
                uploaded_file
            )
        )

        lat = read_latitude(
            uploaded_file
        )

        tilt_lookup = read_tilt_lookup(
            uploaded_file
        )

        ghi_df = read_ghi(
            uploaded_file
        )

        fixed_df = read_fixed_data(
            uploaded_file
        )

        fixed_df, ghi_df, n = align_data(
            fixed_df,
            ghi_df,
        )

    except Exception as e:

        st.error(
            f"Unable to load workbook: {e}"
        )

        st.exception(e)

        return


    # ========================================================
    # DATA PREPARATION
    # ========================================================

    actual = fixed_df[
        "Actual"
    ].to_numpy(
        dtype=float
    )

    dates = fixed_df[
        "Date"
    ]

    ghi_matrix = np.column_stack([

        ghi_df[col].to_numpy(
            dtype=float
        )

        for col in GHI_COLS

    ])

    blocks = ghi_df[
        "Block"
    ].to_numpy(
        dtype=float
    )


    standard_efficiency = pd.to_numeric(

        area_df[
            "Standard PV Efficiency (%)"
        ],

        errors="coerce",

    ).dropna().to_numpy(
        dtype=float
    )

    standard_efficiency = (
        standard_efficiency[:5]
    )


    if len(standard_efficiency) != 5:

        st.error(
            "Exactly 5 standard efficiency values "
            "are required."
        )

        return


    # ========================================================
    # SOLAR GEOMETRY
    # ========================================================

    (
        declination,
        elevation,
        tilt,
        sin_a,
        sin_ab,
    ) = calculate_solar_geometry(
        dates,
        lat,
        tilt_lookup,
    )


    # ========================================================
    # FIXED MODEL
    # ========================================================

    with st.spinner(
        "🏗️ Calculating Fixed model..."
    ):

        fixed = run_fixed_model(
            standard_efficiency,
            fixed_weights,
            ghi_matrix,
            sin_a,
            sin_ab,
            actual,
        )


    # ========================================================
    # TRACKING MODEL
    # ========================================================

    tracking_weights_adjusted = (
        tracking_weights
        * (
            fixed["net_efficiency"]
            / standard_efficiency
        )
    )


    try:

        backend_blocks = read_backend_blocks(
            uploaded_file,
            "C11",
        )

        backend_blocks = backend_blocks[
            :n
        ]

        if len(backend_blocks) != n:

            backend_blocks = blocks.copy()

    except Exception:

        backend_blocks = blocks.copy()


    with st.spinner(
        "🔄 Optimizing Tracking model..."
    ):

        tracking = run_tracking_model(
            backend_blocks,
            ghi_matrix,
            tracking_weights_adjusted,
            actual,
            fixed["best_loss"],
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    st.session_state.results = {
        "fixed": fixed,
        "tracking": tracking,
        "actual": actual,
    }


    # ========================================================
    # MODEL SELECTOR
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🎛️ Select Model Output'
        '</div>',
        unsafe_allow_html=True,
    )

    selected_model = st.segmented_control(
        "Model",
        options=[
            "🏗️ Fixed",
            "🔄 Tracking",
        ],
        default="🏗️ Fixed",
        selection_mode="single",
        key="model_output_selector",
        label_visibility="collapsed",
        width="stretch",
    )


    # ========================================================
    # DISPLAY SELECTED MODEL
    # ========================================================

    if selected_model == "🏗️ Fixed":

        show_fixed_output(
            actual,
            fixed,
        )

    else:

        show_tracking_output(
            actual,
            tracking,
        )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📥 Download Results'
        '</div>',
        unsafe_allow_html=True,
    )

    output_excel = create_output_excel(
        fixed,
        tracking,
        actual,
    )

    st.download_button(
        label="📥 Download Fixed & Tracking Results",
        data=output_excel,
        file_name="Solar_Loss_Correction_Results.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
