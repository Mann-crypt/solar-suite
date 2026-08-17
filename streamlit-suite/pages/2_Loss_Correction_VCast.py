# ============================================================
# STREAMLIT APP
# SOLAR FORECAST CORRECTION
#
# Fixed + Tracking
# Optimized calculation
#
# Calculation flow:
# 1. Upload Excel
# 2. Input GHI + Actual data
# 3. Select Plant Type
# 4. Automatic calculation
# 5. Editable optimization parameters
# 6. Recalculate
# 7. Graph + results
#
# IMPORTANT:
# Tracking Error % is applied ONLY through Effective Area.
# It is NOT applied again inside Tracking optimization.
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
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-weight: 700;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.05);
        text-align: center;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

CLUSTER_POWER_COLS = [
    "CL1_Fixed Power=I*Ƞ*A",
    "CL2_Fixed Power=I*Ƞ*A",
    "CL3_Fixed Power=I*Ƞ*A",
    "CL4_Fixed Power=I*Ƞ*A",
    "CL5_Fixed Power=I*Ƞ*A",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def trim_at_first_null(df, column):
    """
    Trim dataframe at first null value in a specified column.
    If no null exists, return complete dataframe.
    """
    if column not in df.columns:
        return df.copy()

    null_idx = df[df[column].isna()].index

    if len(null_idx) == 0:
        return df.copy()

    first_pos = df.index.get_loc(null_idx[0])

    return df.iloc[:first_pos].copy()


def clean_columns(df):
    """Clean Excel column names."""
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    return df


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


# ============================================================
# READ EXCEL FILE
# ============================================================

@st.cache_data(show_spinner=False)
def load_excel(uploaded_file):

    # Read workbook once
    excel = pd.ExcelFile(uploaded_file)

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    df = pd.read_excel(
        excel,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df = clean_columns(df)

    df = trim_at_first_null(df, "S.No.")

    # --------------------------------------------------------
    # Preserve original values
    # --------------------------------------------------------

    if "Error %" not in df.columns:
        df["Error %"] = 0.0

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    if (
        "No of Module" in df.columns
        and "Area of 1 Module (m2)" in df.columns
    ):
        df["Total area (m2)"] = (
            safe_numeric(df["No of Module"])
            * safe_numeric(df["Area of 1 Module (m2)"])
        )

    # --------------------------------------------------------
    # Cluster table
    # --------------------------------------------------------

    df_w = pd.read_excel(
        excel,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df_w.columns = [
        str(c).strip()
        for c in df_w.columns
    ]

    if "Clusters" in df_w.columns:

        null_idx = df_w[
            df_w["Clusters"].isna()
        ].index

        if len(null_idx) > 0:

            first_pos = df_w.index.get_loc(
                null_idx[0]
            )

            df_w = df_w.iloc[:first_pos].copy()

    # --------------------------------------------------------
    # Forecast Config
    # --------------------------------------------------------

    df_st = pd.read_excel(
        excel,
        sheet_name="Forecast Config",
        header=8,
    )

    lat = float(
        pd.to_numeric(
            df_st.loc[0, "Lat"],
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    df_tilt = pd.read_excel(
        excel,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    df_tilt = trim_at_first_null(
        df_tilt,
        "Fixed",
    )

    df_tilt = df_tilt.dropna(
        how="all",
        axis=1,
    )

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    month_lookup = (
        df_tilt
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # Result sheet
    # --------------------------------------------------------

    df_ghi = pd.read_excel(
        excel,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df_ghi = df_ghi.fillna(0)

    # --------------------------------------------------------
    # Fixed C11
    # --------------------------------------------------------

    df_fix = pd.read_excel(
        excel,
        sheet_name="Fixed-C11",
        header=1,
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    df_fix = trim_at_first_null(
        df_fix,
        "Date",
    )

    # --------------------------------------------------------
    # Backend tracking sheets
    # --------------------------------------------------------

    backend = []

    for cluster in ["C11", "C12", "C13", "C14", "C15"]:

        backend.append(
            pd.read_excel(
                excel,
                sheet_name=f"Backend Cal {cluster}",
            )
        )

    # --------------------------------------------------------
    # Tracking sheet
    # --------------------------------------------------------

    df_trac = pd.read_excel(
        excel,
        sheet_name="Tracking",
        header=1,
    )

    return (
        df,
        df_w,
        df_st,
        df_tilt,
        df_ghi,
        df_fix,
        backend,
        df_trac,
        lat,
        month_lookup,
    )


# ============================================================
# PREPARE INPUT DATA
# ============================================================

def prepare_input_data(
    df_ghi,
    df_fix,
):
    """
    Prepare user supplied GHI and Actual data.

    User input is expected to provide:
        GHI C11 ... GHI C15
        Actual
    """

    ghi = df_ghi.copy()
    actual_df = df_fix.copy()

    # --------------------------------------------------------
    # Ensure GHI columns exist
    # --------------------------------------------------------

    for col in GHI_COLS:

        if col not in ghi.columns:
            ghi[col] = 0.0

        ghi[col] = pd.to_numeric(
            ghi[col],
            errors="coerce",
        ).fillna(0.0)

    # --------------------------------------------------------
    # Actual
    # --------------------------------------------------------

    if "Actual" not in actual_df.columns:
        actual_df["Actual"] = 0.0

    actual_df["Actual"] = pd.to_numeric(
        actual_df["Actual"],
        errors="coerce",
    ).fillna(0.0)

    # --------------------------------------------------------
    # Align length
    # --------------------------------------------------------

    n = min(
        len(ghi),
        len(actual_df),
    )

    ghi = ghi.iloc[:n].reset_index(drop=True)

    actual_df = (
        actual_df
        .iloc[:n]
        .reset_index(drop=True)
    )

    return ghi, actual_df


# ============================================================
# CALCULATE EFFECTIVE AREAS
# ============================================================

def calculate_effective_areas(
    df_original,
    df_w_original,
    error_pct,
):
    """
    Error % is applied exactly once here.

    Net Efficiency =
        Standard PV Efficiency - Error %

    Effective Area =
        Net Efficiency * Total Area / 100
    """

    df = df_original.copy()

    df_w = df_w_original.copy()

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    df["Error %"] = error_pct

    # --------------------------------------------------------
    # Total area
    # --------------------------------------------------------

    df["Total area (m2)"] = (
        safe_numeric(
            df["No of Module"]
        )
        * safe_numeric(
            df["Area of 1 Module (m2)"]
        )
    )

    # --------------------------------------------------------
    # Net efficiency
    # --------------------------------------------------------

    df["Net Efficiency (%)"] = (
        safe_numeric(
            df["Standard PV Efficiency (%)"]
        )
        - error_pct
    )

    # --------------------------------------------------------
    # Effective area
    # --------------------------------------------------------

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100.0
    )

    # --------------------------------------------------------
    # Cluster effective area
    # --------------------------------------------------------

    cluster_sums = (
        df.groupby("Clusters")["Eff Area"]
        .sum()
    )

    df_w["Eff Area(m2)"] = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0.0)
    )

    return df, df_w


# ============================================================
# FIXED PREPARATION
# ============================================================

def prepare_fixed_base(
    df_ghi,
    df_fix,
    lat,
    month_lookup,
):
    """
    Prepare all Fixed calculations which do not depend on
    Error %.
    """

    result = df_fix.copy()

    # --------------------------------------------------------
    # Original code uses today's date
    # --------------------------------------------------------

    today = pd.Timestamp.today().normalize()

    result["Date"] = today

    first_date = today.replace(
        month=1,
        day=1,
    )

    # --------------------------------------------------------
    # Declination
    # --------------------------------------------------------

    day_number = (
        result["Date"]
        - first_date
    ).dt.days

    result["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_number
                    + 1
                )
                / 365
            )
        )
    )

    # --------------------------------------------------------
    # Elevation
    # --------------------------------------------------------

    result["Elevation angle a"] = (
        90
        - lat
        + result["Declination Angle ∆"]
    )

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    result["Tilt Angle b"] = (
        result["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

    result["a+b"] = (
        result["Elevation angle a"]
        + result["Tilt Angle b"]
    )

    result["SIN(a+b)"] = np.sin(
        np.radians(result["a+b"])
    )

    result["Sin(a)"] = np.sin(
        np.radians(
            result["Elevation angle a"]
        )
    )

    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    sin_a = np.where(
        np.abs(
            result["Sin(a)"].to_numpy()
        ) < 1e-9,
        1e-9,
        result["Sin(a)"].to_numpy(),
    )

    # --------------------------------------------------------
    # POA for each cluster
    # --------------------------------------------------------

    poa = []

    for col in GHI_COLS:

        ghi = (
            pd.to_numeric(
                df_ghi[col],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(
                dtype=float
            )
        )

        poa_value = (
            ghi
            * result["SIN(a+b)"].to_numpy()
            / sin_a
        )

        poa.append(poa_value)

    return result, np.column_stack(poa)


# ============================================================
# FIXED POWER
# ============================================================

def calculate_fixed_power(
    poa_matrix,
    effective_areas,
):
    """
    Power = POA × Effective Area / 1,000,000
    """

    forecast = (
        poa_matrix
        @ effective_areas
    ) / 1_000_000

    return forecast


# ============================================================
# FIXED OPTIMIZATION
# ============================================================

def optimize_fixed(
    df_original,
    df_w_original,
    df_ghi,
    df_fix,
    lat,
    month_lookup,
    error_min=0.0,
    error_max=10.0,
    error_step=0.1,
):
    """
    Find Error % which minimizes peak error.

    This is equivalent to the original loop but avoids
    repeatedly rebuilding the complete calculation.
    """

    base_fix, poa_matrix = prepare_fixed_base(
        df_ghi,
        df_fix,
        lat,
        month_lookup,
    )

    actual = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    actual_peak = (
        np.max(actual)
        if len(actual)
        else 0
    )

    errors = np.round(
        np.arange(
            error_min,
            error_max + error_step / 2,
            error_step,
        ),
        10,
    )

    results = []

    best_error = None
    best_peak_error = np.inf
    best_forecast = None
    best_areas = None
    best_df = None
    best_df_w = None

    for error in errors:

        temp_df, temp_df_w = calculate_effective_areas(
            df_original,
            df_w_original,
            error,
        )

        areas = (
            pd.to_numeric(
                temp_df_w["Eff Area(m2)"],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(dtype=float)
        )

        # Always exactly five clusters
        areas = np.pad(
            areas[:5],
            (0, max(0, 5 - len(areas))),
        )

        forecast = calculate_fixed_power(
            poa_matrix,
            areas,
        )

        calculated_peak = (
            np.max(forecast)
            if len(forecast)
            else 0
        )

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

        results.append(
            {
                "Error %": error,
                "Calculated Peak": calculated_peak,
                "Actual Peak": actual_peak,
                "Peak Error": peak_error,
                "Peak Error %": peak_error_pct,
            }
        )

        if peak_error < best_peak_error:

            best_peak_error = peak_error
            best_error = float(error)
            best_forecast = forecast
            best_areas = areas
            best_df = temp_df
            best_df_w = temp_df_w

    results_df = pd.DataFrame(results)

    return {
        "best_error": best_error,
        "forecast": best_forecast,
        "effective_areas": best_areas,
        "df": best_df,
        "df_w": best_df_w,
        "df_fix": base_fix,
        "results": results_df,
        "poa_matrix": poa_matrix,
    }


# ============================================================
# TRACKING PREPARATION
# ============================================================

def prepare_tracking_base(
    df_ghi,
    df_fix,
    backend_list,
    effective_areas,
):
    """
    Prepare NumPy arrays for Tracking optimization.

    IMPORTANT:
    Error % has already been applied to effective_areas.

    Therefore Tracking does NOT apply Error % again.
    """

    ghi_matrix = np.column_stack(
        [
            pd.to_numeric(
                df_ghi[col],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(dtype=float)
            for col in GHI_COLS
        ]
    )

    blocks = pd.to_numeric(
        backend_list[0]["Block No."],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Align all arrays
    # --------------------------------------------------------

    n = min(
        len(ghi_matrix),
        len(blocks),
        len(df_fix),
    )

    ghi_matrix = ghi_matrix[:n]
    blocks = blocks[:n]

    actual = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )[:n]

    areas = np.asarray(
        effective_areas,
        dtype=float,
    )

    if len(areas) < 5:
        areas = np.pad(
            areas,
            (0, 5 - len(areas)),
        )

    areas = areas[:5]

    return (
        ghi_matrix,
        blocks,
        actual,
        areas,
    )


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def make_tracking_objective(
    ghi_matrix,
    blocks,
    actual,
    effective_areas,
):
    """
    Creates a fast NumPy objective function.

    No pandas operations occur inside the optimizer.
    """

    # --------------------------------------------------------
    # Actual mask
    # --------------------------------------------------------

    mask = actual != 0

    if not np.any(mask):
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual_valid = actual[mask]

    actual_max = np.max(
        actual_valid
    )

    actual_sum = np.sum(
        actual_valid
    )

    def objective(x):

        DHI = int(round(x[0]))
        GHI_Starting_Block = int(round(x[1]))
        GHI_Ending_Block = int(round(x[2]))
        GHI_Max_Block = int(round(x[3]))
        Tracking_angle_lim_E = int(round(x[4]))
        Tracking_angle_lim_W = int(round(x[5]))

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if not (
            GHI_Starting_Block
            < GHI_Max_Block
            < GHI_Ending_Block
        ):
            return 1e9

        # ----------------------------------------------------
        # Same equations as original Jupyter calculation
        # ----------------------------------------------------

        denominator_1 = (
            GHI_Starting_Block
            - 1
            - GHI_Max_Block
        )

        denominator_2 = (
            GHI_Ending_Block
            + 1
            - GHI_Max_Block
        )

        if denominator_1 == 0 or denominator_2 == 0:
            return 1e9

        m1 = 90 / denominator_1
        m2 = 90 / denominator_2

        # ----------------------------------------------------
        # Zenith
        # ----------------------------------------------------

        zenith = np.where(
            blocks <= GHI_Max_Block,

            np.minimum(
                89,
                m1
                * (
                    blocks
                    - GHI_Max_Block
                ),
            ),

            np.minimum(
                89,
                m2
                * (
                    blocks
                    - GHI_Max_Block
                ),
            ),
        )

        # ----------------------------------------------------
        # Panel
        # ----------------------------------------------------

        panel = np.where(
            blocks < GHI_Max_Block,

            np.minimum(
                zenith,
                abs(
                    Tracking_angle_lim_E
                ),
            ),

            np.where(
                (
                    (blocks > GHI_Max_Block)
                    &
                    (
                        zenith
                        > Tracking_angle_lim_W
                    )
                ),

                Tracking_angle_lim_W,

                zenith,
            ),
        )

        # ----------------------------------------------------
        # Cosine
        # ----------------------------------------------------

        cos_alpha = np.cos(
            np.radians(panel)
        )

        cos_alpha = np.clip(
            cos_alpha,
            1e-6,
            None,
        )

        # ----------------------------------------------------
        # DHI
        # ----------------------------------------------------

        dhi = (
            ghi_matrix
            * DHI
            / 100.0
        )

        # ----------------------------------------------------
        # DNI
        # ----------------------------------------------------

        dni = (
            ghi_matrix
            - dhi
        ) / cos_alpha[:, None]

        # ----------------------------------------------------
        # Forecast
        #
        # IMPORTANT:
        # effective_areas already contain Error %
        #
        # NO second Error % multiplication here.
        # ----------------------------------------------------

        prediction_full = (
            dni @ effective_areas
        ) / 1_000_000

        if (
            np.isnan(
                prediction_full
            ).any()
            or np.isinf(
                prediction_full
            ).any()
        ):
            return 1e9

        prediction = (
            prediction_full[mask]
        )

        if len(prediction) == 0:
            return 1e9

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        block_error = (
            np.mean(
                np.abs(
                    actual_valid
                    - prediction
                )
            )
            / actual_max
        )

        peak_error = (
            abs(
                actual_max
                - np.max(prediction)
            )
            / actual_max
        )

        energy_error = (
            abs(
                actual_sum
                - np.sum(prediction)
            )
            / actual_sum
        )

        # ----------------------------------------------------
        # Original weighting
        # ----------------------------------------------------

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return score

    return objective


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    ghi_matrix,
    blocks,
    effective_areas,
    params,
):
    """
    Calculate final Tracking forecast using optimized params.
    """

    (
        DHI,
        GHI_Starting_Block,
        GHI_Ending_Block,
        GHI_Max_Block,
        Tracking_angle_lim_E,
        Tracking_angle_lim_W,
    ) = [
        int(round(v))
        for v in params
    ]

    m1 = 90 / (
        GHI_Starting_Block
        - 1
        - GHI_Max_Block
    )

    m2 = 90 / (
        GHI_Ending_Block
        + 1
        - GHI_Max_Block
    )

    zenith = np.where(
        blocks <= GHI_Max_Block,

        np.minimum(
            89,
            m1
            * (
                blocks
                - GHI_Max_Block
            ),
        ),

        np.minimum(
            89,
            m2
            * (
                blocks
                - GHI_Max_Block
            ),
        ),
    )

    panel = np.where(
        blocks < GHI_Max_Block,

        np.minimum(
            zenith,
            abs(
                Tracking_angle_lim_E
            ),
        ),

        np.where(
            (
                (blocks > GHI_Max_Block)
                &
                (
                    zenith
                    > Tracking_angle_lim_W
                )
            ),

            Tracking_angle_lim_W,

            zenith,
        ),
    )

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None,
    )

    dhi = (
        ghi_matrix
        * DHI
        / 100.0
    )

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    forecast = (
        dni @ effective_areas
    ) / 1_000_000

    return (
        forecast,
        zenith,
        panel,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    df_ghi,
    df_fix,
    backend_list,
    effective_areas,
    bounds,
    maxiter=40,
    popsize=15,
):

    (
        ghi_matrix,
        blocks,
        actual,
        areas,
    ) = prepare_tracking_base(
        df_ghi,
        df_fix,
        backend_list,
        effective_areas,
    )

    objective = make_tracking_objective(
        ghi_matrix,
        blocks,
        actual,
        areas,
    )

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    best = np.round(
        result.x
    ).astype(int)

    forecast, zenith, panel = (
        calculate_tracking_forecast(
            ghi_matrix,
            blocks,
            areas,
            best,
        )
    )

    return {
        "result": result,
        "best": best,
        "forecast": forecast,
        "zenith": zenith,
        "panel": panel,
        "blocks": blocks,
        "actual": actual,
        "ghi_matrix": ghi_matrix,
        "effective_areas": areas,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    forecast,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    n = min(
        len(actual),
        len(forecast),
    )

    actual = actual[:n]
    forecast = forecast[:n]

    actual_peak = (
        np.max(actual)
        if len(actual)
        else 0
    )

    forecast_peak = (
        np.max(forecast)
        if len(forecast)
        else 0
    )

    peak_error = abs(
        forecast_peak
        - actual_peak
    )

    peak_error_pct = (
        peak_error
        / actual_peak
        * 100
        if actual_peak != 0
        else np.nan
    )

    actual_energy = np.sum(
        actual
    )

    forecast_energy = np.sum(
        forecast
    )

    energy_error_pct = (
        abs(
            forecast_energy
            - actual_energy
        )
        / actual_energy
        * 100
        if actual_energy != 0
        else np.nan
    )

    mae = np.mean(
        np.abs(
            actual
            - forecast
        )
    )

    return {
        "Actual Peak": actual_peak,
        "Forecast Peak": forecast_peak,
        "Peak Error": peak_error,
        "Peak Error %": peak_error_pct,
        "Actual Energy": actual_energy,
        "Forecast Energy": forecast_energy,
        "Energy Error %": energy_error_pct,
        "MAE": mae,
    }


# ============================================================
# GRAPH
# ============================================================

def plot_forecast(
    actual,
    forecast,
    title,
):

    n = min(
        len(actual),
        len(forecast),
    )

    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual[:n],
            mode="lines",
            name="Actual",
            line=dict(
                width=2,
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
                width=2,
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="15-Minute Block",
        yaxis_title="Power (MW)",
        hovermode="x unified",
        height=480,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.title("☀️ Solar Forecast Correction")

st.caption(
    "Automatic optimization of forecast correction parameters "
    "for Fixed and Tracking solar plants."
)


# ============================================================
# FILE UPLOADER
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload Excel File</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload your Excel workbook",
    type=["xlsx", "xls"],
)


if uploaded_file is None:

    st.info(
        "Upload the Excel workbook to continue."
    )

    st.stop()


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    (
        df_area_original,
        df_w_original,
        df_st,
        df_tilt,
        df_ghi_excel,
        df_fix_excel,
        backend_list,
        df_trac_excel,
        lat,
        month_lookup,
    ) = load_excel(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


# ============================================================
# INPUT DATA
# ============================================================

st.markdown(
    '<div class="section-title">2. Input GHI and Actual Data</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Enter or paste the GHI forecast and Actual Power data used "
    "for the calculation."
)


# ------------------------------------------------------------
# GHI INPUT
# ------------------------------------------------------------

st.markdown("**GHI Forecast**")

default_ghi = df_ghi_excel[
    GHI_COLS
].copy()

input_ghi = st.data_editor(
    default_ghi,
    num_rows="dynamic",
    use_container_width=True,
    height=300,
    key="ghi_editor",
)


# ------------------------------------------------------------
# ACTUAL INPUT
# ------------------------------------------------------------

st.markdown("**Actual Power**")

default_actual = pd.DataFrame(
    {
        "Actual": pd.to_numeric(
            df_fix_excel["Actual"],
            errors="coerce",
        ).fillna(0)
    }
)

input_actual = st.data_editor(
    default_actual,
    num_rows="dynamic",
    use_container_width=True,
    height=250,
    key="actual_editor",
)


# ============================================================
# PREPARE INPUT
# ============================================================

df_ghi, df_fix = prepare_input_data(
    input_ghi,
    input_actual,
)


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">3. Plant Type</div>',
    unsafe_allow_html=True,
)

plant_type = st.segmented_control(
    "Select Plant Type",
    options=[
        "Fixed",
        "Tracking",
    ],
    default="Fixed",
)


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">4. Optimization Parameters</div>',
    unsafe_allow_html=True,
)


if plant_type == "Fixed":

    col1, col2, col3 = st.columns(3)

    with col1:

        error_min = st.number_input(
            "Minimum Error %",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=0.1,
        )

    with col2:

        error_max = st.number_input(
            "Maximum Error %",
            min_value=-50.0,
            max_value=50.0,
            value=10.0,
            step=0.1,
        )

    with col3:

        error_step = st.number_input(
            "Error % Step",
            min_value=0.01,
            max_value=5.0,
            value=0.1,
            step=0.01,
        )

else:

    # --------------------------------------------------------
    # Tracking parameters
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        dhi_min = st.number_input(
            "DHI Min",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )

        dhi_max = st.number_input(
            "DHI Max",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
        )

    with col2:

        start_min = st.number_input(
            "GHI Start Min",
            min_value=1,
            max_value=95,
            value=10,
            step=1,
        )

        start_max = st.number_input(
            "GHI Start Max",
            min_value=1,
            max_value=95,
            value=30,
            step=1,
        )

    with col3:

        end_min = st.number_input(
            "GHI End Min",
            min_value=1,
            max_value=96,
            value=65,
            step=1,
        )

        end_max = st.number_input(
            "GHI End Max",
            min_value=1,
            max_value=96,
            value=80,
            step=1,
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        max_min = st.number_input(
            "GHI Max Min",
            min_value=1,
            max_value=95,
            value=47,
            step=1,
        )

        max_max = st.number_input(
            "GHI Max Max",
            min_value=1,
            max_value=95,
            value=53,
            step=1,
        )

    with col2:

        east_min = st.number_input(
            "East Limit Min",
            min_value=0,
            max_value=90,
            value=10,
            step=1,
        )

        east_max = st.number_input(
            "East Limit Max",
            min_value=0,
            max_value=90,
            value=70,
            step=1,
        )

    with col3:

        west_min = st.number_input(
            "West Limit Min",
            min_value=0,
            max_value=90,
            value=10,
            step=1,
        )

        west_max = st.number_input(
            "West Limit Max",
            min_value=0,
            max_value=90,
            value=70,
            step=1,
        )

    maxiter = st.number_input(
        "Optimization Iterations",
        min_value=1,
        max_value=200,
        value=40,
        step=1,
    )

    popsize = st.number_input(
        "Population Size",
        min_value=5,
        max_value=50,
        value=15,
        step=1,
    )


# ============================================================
# RUN BUTTON
# ============================================================

st.markdown("")


run = st.button(
    "🚀 Run Optimization",
    type="primary",
    use_container_width=True,
)


if not run:

    st.info(
        "Set the parameters and click **Run Optimization**."
    )

    st.stop()


# ============================================================
# FIXED
# ============================================================

if plant_type == "Fixed":

    if error_max < error_min:

        st.error(
            "Maximum Error % must be greater than Minimum Error %."
        )

        st.stop()

    with st.spinner(
        "Running Fixed optimization..."
    ):

        fixed_result = optimize_fixed(
            df_area_original,
            df_w_original,
            df_ghi,
            df_fix,
            lat,
            month_lookup,
            error_min,
            error_max,
            error_step,
        )

    best_error = fixed_result[
        "best_error"
    ]

    forecast = fixed_result[
        "forecast"
    ]

    actual = df_fix[
        "Actual"
    ].to_numpy(
        dtype=float
    )

    metrics = calculate_metrics(
        actual,
        forecast,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.success(
        f"Optimization completed. Best Error % = "
        f"{best_error:.2f}%"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Best Error %",
            f"{best_error:.2f}%",
        )

    with c2:
        st.metric(
            "Actual Peak",
            f"{metrics['Actual Peak']:.3f}",
        )

    with c3:
        st.metric(
            "Forecast Peak",
            f"{metrics['Forecast Peak']:.3f}",
        )

    with c4:
        st.metric(
            "Peak Error %",
            f"{metrics['Peak Error %']:.2f}%",
        )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    st.plotly_chart(
        plot_forecast(
            actual,
            forecast,
            "Fixed Plant: Actual vs Forecast",
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    st.subheader(
        "Optimization Results"
    )

    st.dataframe(
        fixed_result["results"],
        use_container_width=True,
    )


# ============================================================
# TRACKING
# ============================================================

else:

    # --------------------------------------------------------
    # Validate bounds
    # --------------------------------------------------------

    bounds = [
        (
            dhi_min,
            dhi_max,
        ),
        (
            start_min,
            start_max,
        ),
        (
            end_min,
            end_max,
        ),
        (
            max_min,
            max_max,
        ),
        (
            east_min,
            east_max,
        ),
        (
            west_min,
            west_max,
        ),
    ]

    if any(
        low > high
        for low, high in bounds
    ):

        st.error(
            "Every minimum value must be less than or equal to "
            "its maximum value."
        )

        st.stop()

    # --------------------------------------------------------
    # Actual validation
    # --------------------------------------------------------

    actual_values = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    if not np.any(
        actual_values != 0
    ):

        st.error(
            "No non-zero Actual values found for Tracking."
        )

        st.stop()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # First optimize Error %
    #
    # This ensures Tracking receives effective areas
    # calculated with Error % exactly once.
    # --------------------------------------------------------

    with st.spinner(
        "Step 1/2: Calculating optimal Error %..."
    ):

        fixed_result = optimize_fixed(
            df_area_original,
            df_w_original,
            df_ghi,
            df_fix,
            lat,
            month_lookup,
            error_min,
            error_max,
            error_step,
        )

    best_error = fixed_result[
        "best_error"
    ]

    effective_areas = fixed_result[
        "effective_areas"
    ]

    # --------------------------------------------------------
    # Tracking optimization
    # --------------------------------------------------------

    with st.spinner(
        "Step 2/2: Running Tracking optimization..."
    ):

        tracking_result = optimize_tracking(
            df_ghi,
            df_fix,
            backend_list,
            effective_areas,
            bounds,
            maxiter,
            popsize,
        )

    best = tracking_result[
        "best"
    ]

    forecast = tracking_result[
        "forecast"
    ]

    actual = tracking_result[
        "actual"
    ]

    metrics = calculate_metrics(
        actual,
        forecast,
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    (
        DHI,
        GHI_Starting_Block,
        GHI_Ending_Block,
        GHI_Max_Block,
        Tracking_angle_lim_E,
        Tracking_angle_lim_W,
    ) = best

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.success(
        "Tracking optimization completed."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Error %",
            f"{best_error:.2f}%",
        )

    with c2:

        st.metric(
            "Actual Peak",
            f"{metrics['Actual Peak']:.3f}",
        )

    with c3:

        st.metric(
            "Forecast Peak",
            f"{metrics['Forecast Peak']:.3f}",
        )

    with c4:

        st.metric(
            "Peak Error %",
            f"{metrics['Peak Error %']:.2f}%",
        )

    # --------------------------------------------------------
    # Optimized parameters
    # --------------------------------------------------------

    st.subheader(
        "Optimized Tracking Parameters"
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "DHI",
            DHI,
        )

        st.metric(
            "GHI Starting Block",
            GHI_Starting_Block,
        )

    with p2:

        st.metric(
            "GHI Max Block",
            GHI_Max_Block,
        )

        st.metric(
            "GHI Ending Block",
            GHI_Ending_Block,
        )

    with p3:

        st.metric(
            "East Limit",
            Tracking_angle_lim_E,
        )

        st.metric(
            "West Limit",
            Tracking_angle_lim_W,
        )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    st.plotly_chart(
        plot_forecast(
            actual,
            forecast,
            "Tracking Plant: Actual vs Forecast",
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Tracking angle graph
    # --------------------------------------------------------

    st.subheader(
        "Tracking Angles"
    )

    angle_fig = go.Figure()

    angle_fig.add_trace(
        go.Scatter(
            x=np.arange(
                len(
                    tracking_result["zenith"]
                )
            ),
            y=tracking_result["zenith"],
            mode="lines",
            name="Zenith Angle",
        )
    )

    angle_fig.add_trace(
        go.Scatter(
            x=np.arange(
                len(
                    tracking_result["panel"]
                )
            ),
            y=tracking_result["panel"],
            mode="lines",
            name="Panel Angle",
        )
    )

    angle_fig.update_layout(
        xaxis_title="15-Minute Block",
        yaxis_title="Angle (°)",
        hovermode="x unified",
        height=400,
    )

    st.plotly_chart(
        angle_fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Effective areas
    # --------------------------------------------------------

    st.subheader(
        "Cluster Effective Areas"
    )

    area_df = pd.DataFrame(
        {
            "Cluster": [
                "C11",
                "C12",
                "C13",
                "C14",
                "C15",
            ],
            "Effective Area (m²)": effective_areas,
        }
    )

    st.dataframe(
        area_df,
        use_container_width=True,
        hide_index=True,
    )
