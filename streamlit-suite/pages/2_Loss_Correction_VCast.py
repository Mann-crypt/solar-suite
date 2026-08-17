# ============================================================
# STREAMLIT APP
# SOLAR FORECAST CORRECTION
# FIXED + TRACKING
#
# CALCULATION LOGIC PRESERVED FROM ORIGINAL CODE
# ONLY UI / PLOTTING CHANGED
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
    initial_sidebar_state="collapsed"
)


# ============================================================
# DEFAULT FILE
# ============================================================

DEFAULT_FILE = (
    r"C:\Users\Manjot Singh\Desktop"
    r"\Chouraniya Juniper_FixedUserStory_MAL Solar Cluster_Fixed-Tracking ECM10.xlsx"
)


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

DEFAULT_FIXED_START = 0.0
DEFAULT_FIXED_END = 10.0
DEFAULT_FIXED_STEP = 0.1

DEFAULT_DHI_MIN = 0
DEFAULT_DHI_MAX = 10

DEFAULT_START_MIN = 10
DEFAULT_START_MAX = 30

DEFAULT_END_MIN = 65
DEFAULT_END_MAX = 80

DEFAULT_MAX_MIN = 47
DEFAULT_MAX_MAX = 53

DEFAULT_EAST_MIN = 10
DEFAULT_EAST_MAX = 70

DEFAULT_WEST_MIN = 10
DEFAULT_WEST_MAX = 70

DEFAULT_MAXITER = 40
DEFAULT_POPSIZE = 15
DEFAULT_TOL = 0.001
DEFAULT_MUTATION_LOW = 0.5
DEFAULT_MUTATION_HIGH = 1.0
DEFAULT_RECOMBINATION = 0.7
DEFAULT_SEED = 42


# ============================================================
# SESSION STATE
# ============================================================

if "calculated" not in st.session_state:
    st.session_state.calculated = False

if "calculation_result" not in st.session_state:
    st.session_state.calculation_result = None

if "file_bytes" not in st.session_state:
    st.session_state.file_bytes = None


# ============================================================
# HEADER
# ============================================================

st.title("☀️ Solar Forecast Correction")

st.caption(
    "Fixed / Tracking forecast correction with automatic calculation "
    "and editable optimization parameters."
)


# ============================================================
# FILE INPUT
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"],
    key="excel_uploader"
)

if uploaded_file is not None:
    st.session_state.file_bytes = uploaded_file.getvalue()


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.segmented_control(
    "Plant Type",
    options=["Fixed", "Tracking"],
    default="Fixed",
    key="plant_type"
)


# ============================================================
# FILE LOADER
# ============================================================

def get_excel_source():

    if st.session_state.file_bytes is not None:
        return io.BytesIO(st.session_state.file_bytes)

    return DEFAULT_FILE


# ============================================================
# SAFE NUMERIC CONVERSION
# ============================================================

def numeric_series(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


# ============================================================
# READ COMMON DATA
# ============================================================

def read_common_data(file_source):

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    df = pd.read_excel(
        file_source,
        sheet_name="Area & Efficiency",
        header=[1],
        usecols=range(12)
    )

    null_indices = df[df["S.No."].isna()].index

    if len(null_indices) > 0:
        first_null_pos = df.index.get_loc(null_indices[0])
        df = df.iloc[:first_null_pos]

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_cols = [
        "Standard PV Efficiency (%)",
        "Error %",
        "No of Module",
        "Area of 1 Module (m2)",
        "Total area (m2)"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    if "Total area (m2)" not in df.columns:
        df["Total area (m2)"] = (
            df["No of Module"]
            * df["Area of 1 Module (m2)"]
        )

    if "Error %" not in df.columns:
        df["Error %"] = 0.0

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - df["Error %"]
    )

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

    # --------------------------------------------------------
    # CLUSTER AREA
    # --------------------------------------------------------

    df_w = pd.read_excel(
        file_source,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15]
    )

    null_indices = df_w[df_w["Clusters"].isna()].index

    if len(null_indices) > 0:
        first_null_pos = df_w.index.get_loc(null_indices[0])
        df_w = df_w.iloc[:first_null_pos]

    cluster_sums = (
        df.groupby("Clusters")["Eff Area"]
        .sum()
    )

    df_w["Eff Area(m2)"] = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0)
    )

    return df, df_w


# ============================================================
# READ GHI
# ============================================================

def read_ghi(file_source):

    df_ghi = pd.read_excel(
        file_source,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5]
    )

    df_ghi = df_ghi.fillna(0)

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15"
    ]

    for col in ghi_cols:
        if col not in df_ghi.columns:
            df_ghi[col] = 0.0

        df_ghi[col] = numeric_series(
            df_ghi[col]
        )

    return df_ghi


# ============================================================
# READ FIXED DATA
# ============================================================

def read_fixed_data(file_source):

    # --------------------------------------------------------
    # FORECAST CONFIG
    # --------------------------------------------------------

    df_st = pd.read_excel(
        file_source,
        sheet_name="Forecast Config",
        header=[8]
    )

    lat = float(
        pd.to_numeric(
            df_st.loc[0, "Lat"],
            errors="coerce"
        )
    )

    # --------------------------------------------------------
    # TILT
    # --------------------------------------------------------

    df_tilt = pd.read_excel(
        file_source,
        sheet_name="Config Tilt Angle",
        header=[7]
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
            first_null_pos = (
                df_tilt.index.get_loc(
                    null_indices[0]
                )
            )

            df_tilt = df_tilt.iloc[
                :first_null_pos
            ]

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

    month_lookup = (
        df_tilt
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # FIXED-C11
    # --------------------------------------------------------

    df_fix = pd.read_excel(
        file_source,
        sheet_name="Fixed-C11",
        header=[1]
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    null_indices = df_fix[
        df_fix["Date"].isna()
    ].index

    if len(null_indices) > 0:

        first_null_pos = (
            df_fix.index.get_loc(
                null_indices[0]
            )
        )

        df_fix = df_fix.iloc[
            :first_null_pos
        ]

    # --------------------------------------------------------
    # ACTUAL
    # --------------------------------------------------------

    df_fix["Actual"] = numeric_series(
        df_fix["Actual"]
    )

    # --------------------------------------------------------
    # ORIGINAL CODE USES TODAY'S DATE
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

    df_fix["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + (
                        df_fix["Date"]
                        - first_date
                    ).dt.days
                    + 1
                )
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

    # --------------------------------------------------------
    # ANGLES
    # --------------------------------------------------------

    df_fix["a+b"] = (
        df_fix["Elevation angle a"]
        + df_fix["Tilt Angle b"]
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

    return df_fix, lat


# ============================================================
# CALCULATE FIXED POA
# ============================================================

def calculate_fixed_poa(
    df_fix,
    df_ghi
):

    ghi_mapping = {
        "C11": "GHI C11",
        "C12": "GHI C12",
        "C13": "GHI C13",
        "C14": "GHI C14",
        "C15": "GHI C15"
    }

    # --------------------------------------------------------
    # C11
    # --------------------------------------------------------

    df_fix["GHI*sin(a)"] = (
        df_ghi["GHI C11"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)"] = (
        df_ghi["GHI C11"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA fixed"] = (
        df_fix["GHI*sin(a+b)"]
        / df_fix["Sin(a)"].replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # C12-C15
    # --------------------------------------------------------

    for cluster_num in range(2, 6):

        cluster = f"C{cluster_num}"
        ghi_col = ghi_mapping[cluster]

        df_fix[
            f"GHI*sin(a)-CL{cluster_num}"
        ] = (
            df_ghi[ghi_col]
            * df_fix["Sin(a)"]
        )

        df_fix[
            f"GHI*sin(a+b)-CL{cluster_num}"
        ] = (
            df_ghi[ghi_col]
            * df_fix["SIN(a+b)"]
        )

        df_fix[
            f"POA Fixed-C{cluster_num}"
        ] = (
            df_fix[
                f"GHI*sin(a+b)-CL{cluster_num}"
            ]
            / df_fix["Sin(a)"].replace(
                0,
                np.nan
            )
        )

    return df_fix


# ============================================================
# FIXED POWER CALCULATION
# ============================================================

def calculate_fixed_power(
    df,
    df_w,
    df_fix,
    error
):

    df = df.copy()
    df_w = df_w.copy()
    df_fix = df_fix.copy()

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    df["Error %"] = error

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - df["Error %"]
    )

    # --------------------------------------------------------
    # TOTAL AREA
    # --------------------------------------------------------

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    # --------------------------------------------------------
    # EFFECTIVE AREA
    # --------------------------------------------------------

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

    # --------------------------------------------------------
    # CLUSTER EFFECTIVE AREA
    # --------------------------------------------------------

    cluster_sums = (
        df.groupby("Clusters")["Eff Area"]
        .sum()
    )

    df_w["Eff Area(m2)"] = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0)
    )

    # --------------------------------------------------------
    # FIXED CLUSTER POWER
    # --------------------------------------------------------

    poa_cols = [
        "POA fixed",
        "POA Fixed-C12",
        "POA Fixed-C13",
        "POA Fixed-C14",
        "POA Fixed-C15"
    ]

    power_cols = []

    for i in range(5):

        power_col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        df_fix[power_col] = (
            df_fix[poa_cols[i]]
            * df_w.iloc[i]["Eff Area(m2)"]
        ) / 1_000_000

        power_cols.append(power_col)

    # --------------------------------------------------------
    # TOTAL POWER
    # --------------------------------------------------------

    df_fix[
        "Total Power (CL1+CL2+…)"
    ] = (
        df_fix[power_cols]
        .sum(axis=1)
    )

    return df, df_w, df_fix


# ============================================================
# FIND BEST FIXED ERROR
# ============================================================

def optimize_fixed(
    df,
    df_w,
    df_fix,
    start_error,
    end_error,
    step_error
):

    actual_peak = df_fix["Actual"].max()

    if not np.isfinite(actual_peak) or actual_peak == 0:

        raise ValueError(
            "No valid non-zero Actual peak found for Fixed."
        )

    results = []

    errors = np.arange(
        start_error,
        end_error + step_error / 2,
        step_error
    )

    for error in errors:

        _, _, temp_fix = calculate_fixed_power(
            df,
            df_w,
            df_fix,
            error
        )

        calculated_peak = (
            temp_fix[
                "Total Power (CL1+CL2+…)"
            ].max()
        )

        peak_error = abs(
            calculated_peak
            - actual_peak
        )

        peak_error_pct = (
            peak_error
            / actual_peak
            * 100
        )

        results.append({
            "Error %": error,
            "Calculated Peak": calculated_peak,
            "Actual Peak": actual_peak,
            "Peak Error": peak_error,
            "Peak Error %": peak_error_pct
        })

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["Peak Error"].idxmin()
    ]

    best_error = float(
        best_row["Error %"]
    )

    # --------------------------------------------------------
    # FINAL CALCULATION
    # --------------------------------------------------------

    final_df, final_df_w, final_df_fix = (
        calculate_fixed_power(
            df,
            df_w,
            df_fix,
            best_error
        )
    )

    final_calculated_peak = (
        final_df_fix[
            "Total Power (CL1+CL2+…)"
        ].max()
    )

    final_actual_peak = (
        final_df_fix["Actual"].max()
    )

    final_peak_error = abs(
        final_calculated_peak
        - final_actual_peak
    )

    final_peak_error_pct = (
        final_peak_error
        / final_actual_peak
        * 100
        if final_actual_peak != 0
        else np.nan
    )

    return {
        "best_error": best_error,
        "df": final_df,
        "df_w": final_df_w,
        "df_fix": final_df_fix,
        "results": results_df,
        "calculated_peak": final_calculated_peak,
        "actual_peak": final_actual_peak,
        "peak_error": final_peak_error,
        "peak_error_pct": final_peak_error_pct
    }


# ============================================================
# READ TRACKING DATA
# ============================================================

def read_tracking_data(file_source):

    df_bcal1 = pd.read_excel(
        file_source,
        sheet_name="Backend Cal C11"
    )

    df_bcal2 = pd.read_excel(
        file_source,
        sheet_name="Backend Cal C12"
    )

    df_bcal3 = pd.read_excel(
        file_source,
        sheet_name="Backend Cal C13"
    )

    df_bcal4 = pd.read_excel(
        file_source,
        sheet_name="Backend Cal C14"
    )

    df_bcal5 = pd.read_excel(
        file_source,
        sheet_name="Backend Cal C15"
    )

    df_trac = pd.read_excel(
        file_source,
        sheet_name="Tracking",
        header=1
    )

    backend_list = [
        df_bcal1,
        df_bcal2,
        df_bcal3,
        df_bcal4,
        df_bcal5
    ]

    return backend_list, df_trac


# ============================================================
# TRACKING FORECAST CALCULATION
# ============================================================

def tracking_forecast(
    x,
    blocks,
    ghi_matrix,
    cl_weights
):

    DHI = int(round(x[0]))

    GHI_Starting_Block = int(
        round(x[1])
    )

    GHI_Ending_Block = int(
        round(x[2])
    )

    GHI_Max_Block = int(
        round(x[3])
    )

    Tracking_angle_lim_E = int(
        round(x[4])
    )

    Tracking_angle_lim_W = int(
        round(x[5])
    )

    # --------------------------------------------------------
    # VALIDATE BLOCK POSITIONS
    # --------------------------------------------------------

    if not (
        GHI_Starting_Block
        < GHI_Max_Block
        < GHI_Ending_Block
    ):
        return None

    # --------------------------------------------------------
    # SLOPES
    # --------------------------------------------------------

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
        return None

    m1 = 90 / denominator_1

    m2 = 90 / denominator_2

    # --------------------------------------------------------
    # ZENITH
    # --------------------------------------------------------

    zenith = np.where(
        blocks <= GHI_Max_Block,

        np.minimum(
            89,
            m1 * (
                blocks
                - GHI_Max_Block
            )
        ),

        np.minimum(
            89,
            m2 * (
                blocks
                - GHI_Max_Block
            )
        )
    )

    # --------------------------------------------------------
    # PANEL
    # --------------------------------------------------------

    panel = np.where(

        blocks < GHI_Max_Block,

        np.minimum(
            zenith,
            abs(
                Tracking_angle_lim_E
            )
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

            zenith
        )
    )

    # --------------------------------------------------------
    # COSINE
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
    # DHI
    # --------------------------------------------------------

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    # --------------------------------------------------------
    # DNI
    # --------------------------------------------------------

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    # --------------------------------------------------------
    # POWER
    # --------------------------------------------------------

    prediction = (
        dni @ cl_weights
    ) / 1_000_000

    if (
        np.isnan(prediction).any()
        or np.isinf(prediction).any()
    ):
        return None

    return {
        "DHI": DHI,
        "GHI Starting Block": GHI_Starting_Block,
        "GHI Ending Block": GHI_Ending_Block,
        "GHI Max Block": GHI_Max_Block,
        "Tracking East Limit": Tracking_angle_lim_E,
        "Tracking West Limit": Tracking_angle_lim_W,
        "zenith": zenith,
        "panel": panel,
        "forecast": prediction
    }


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    df_fix,
    df_w,
    df_ghi,
    backend_list,
    bounds,
    maxiter,
    popsize,
    tol,
    mutation_low,
    mutation_high,
    recombination,
    seed
):

    # --------------------------------------------------------
    # GHI MATRIX
    # --------------------------------------------------------

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15"
    ]

    ghi_matrix = np.column_stack([
        numeric_series(
            df_ghi[col]
        ).to_numpy(dtype=float)
        for col in ghi_cols
    ])

    # --------------------------------------------------------
    # CLUSTER WEIGHTS
    # --------------------------------------------------------

    cl_weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # BLOCKS
    # --------------------------------------------------------

    blocks = (
        pd.to_numeric(
            backend_list[0]["Block No."],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # ACTUAL
    #
    # IMPORTANT:
    # SAME SOURCE AS ORIGINAL CODE
    # df_fix["Actual"]
    # --------------------------------------------------------

    actual_full = (
        pd.to_numeric(
            df_fix["Actual"],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # ORIGINAL MASK
    # --------------------------------------------------------

    mask = actual_full != 0

    if not mask.any():

        raise ValueError(
            "No non-zero Actual values found for Tracking. "
            "Tracking uses Fixed-C11 -> Actual exactly as in "
            "the original calculation."
        )

    actual = actual_full[mask]

    actual_max = actual.max()
    actual_sum = actual.sum()

    if actual_max == 0 or actual_sum == 0:

        raise ValueError(
            "Actual data contains no usable non-zero values "
            "for Tracking optimization."
        )

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    def objective(x):

        result = tracking_forecast(
            x,
            blocks,
            ghi_matrix,
            cl_weights
        )

        if result is None:
            return 1e9

        prediction_full = result["forecast"]

        prediction = (
            prediction_full[mask]
        )

        # ----------------------------------------------------
        # BLOCK ERROR
        # ----------------------------------------------------

        block_error = (
            np.mean(
                np.abs(
                    actual
                    - prediction
                )
            )
            / actual_max
        )

        # ----------------------------------------------------
        # PEAK ERROR
        # ----------------------------------------------------

        peak_error = (
            abs(
                actual_max
                - prediction.max()
            )
            / actual_max
        )

        # ----------------------------------------------------
        # ENERGY ERROR
        # ----------------------------------------------------

        energy_error = (
            abs(
                actual_sum
                - prediction.sum()
            )
            / actual_sum
        )

        # ----------------------------------------------------
        # ORIGINAL SCORE
        # ----------------------------------------------------

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return score

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    result = differential_evolution(

        objective,

        bounds=bounds,

        strategy="best1bin",

        maxiter=int(maxiter),

        popsize=int(popsize),

        tol=float(tol),

        mutation=(
            float(mutation_low),
            float(mutation_high)
        ),

        recombination=float(
            recombination
        ),

        seed=int(seed),

        polish=True,

        workers=1
    )

    # --------------------------------------------------------
    # ORIGINAL ROUNDING
    # --------------------------------------------------------

    best = np.round(
        result.x
    ).astype(int)

    # --------------------------------------------------------
    # FINAL FORECAST USING ROUNDED PARAMETERS
    # --------------------------------------------------------

    final = tracking_forecast(
        best,
        blocks,
        ghi_matrix,
        cl_weights
    )

    if final is None:

        raise ValueError(
            "Tracking optimization produced "
            "an invalid final parameter combination."
        )

    forecast = final["forecast"]

    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

    prediction = forecast[mask]

    calculated_peak = prediction.max()

    peak_error = abs(
        actual_max
        - calculated_peak
    )

    peak_error_pct = (
        peak_error
        / actual_max
        * 100
    )

    calculated_energy = prediction.sum()

    energy_error_pct = (
        abs(
            actual_sum
            - calculated_energy
        )
        / actual_sum
        * 100
    )

    block_error_pct = (
        np.mean(
            np.abs(
                actual
                - prediction
            )
        )
        / actual_max
        * 100
    )

    return {
        "result": result,
        "best": best,
        "DHI": final["DHI"],
        "GHI Starting Block": final[
            "GHI Starting Block"
        ],
        "GHI Ending Block": final[
            "GHI Ending Block"
        ],
        "GHI Max Block": final[
            "GHI Max Block"
        ],
        "Tracking East Limit": final[
            "Tracking East Limit"
        ],
        "Tracking West Limit": final[
            "Tracking West Limit"
        ],
        "zenith": final["zenith"],
        "panel": final["panel"],
        "forecast": forecast,
        "actual": actual_full,
        "blocks": blocks,
        "objective": result.fun,
        "calculated_peak": calculated_peak,
        "actual_peak": actual_max,
        "peak_error": peak_error,
        "peak_error_pct": peak_error_pct,
        "energy_error_pct": energy_error_pct,
        "block_error_pct": block_error_pct
    }


# ============================================================
# PLOTLY FORECAST GRAPH
# ============================================================

def forecast_plot(
    forecast,
    actual,
    title
):

    n = min(
        len(forecast),
        len(actual)
    )

    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(forecast)[:n],
            mode="lines",
            name="Forecast",
            line=dict(
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(actual)[:n],
            mode="lines",
            name="Actual",
            line=dict(
                width=2
            )
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Block",
        yaxis_title="Power",
        hovermode="x unified",
        height=520,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    fig.update_xaxes(
        range=[
            0,
            max(95, n - 1)
        ]
    )

    fig.update_yaxes(
        rangemode="tozero"
    )

    return fig


# ============================================================
# RUN AUTOMATIC CALCULATION
# ============================================================

def run_calculation(
    plant_type,
    file_source
):

    # --------------------------------------------------------
    # COMMON
    # --------------------------------------------------------

    df, df_w = read_common_data(
        file_source
    )

    df_ghi = read_ghi(
        file_source
    )

    df_fix, lat = read_fixed_data(
        file_source
    )

    df_fix = calculate_fixed_poa(
        df_fix,
        df_ghi
    )

    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    fixed_result = optimize_fixed(
        df=df,
        df_w=df_w,
        df_fix=df_fix,
        start_error=DEFAULT_FIXED_START,
        end_error=DEFAULT_FIXED_END,
        step_error=DEFAULT_FIXED_STEP
    )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    tracking_result = None

    if plant_type == "Tracking":

        backend_list, df_trac = (
            read_tracking_data(
                file_source
            )
        )

        tracking_bounds = [

            (
                DEFAULT_DHI_MIN,
                DEFAULT_DHI_MAX
            ),

            (
                DEFAULT_START_MIN,
                DEFAULT_START_MAX
            ),

            (
                DEFAULT_END_MIN,
                DEFAULT_END_MAX
            ),

            (
                DEFAULT_MAX_MIN,
                DEFAULT_MAX_MAX
            ),

            (
                DEFAULT_EAST_MIN,
                DEFAULT_EAST_MAX
            ),

            (
                DEFAULT_WEST_MIN,
                DEFAULT_WEST_MAX
            )
        ]

        tracking_result = optimize_tracking(

            df_fix=fixed_result["df_fix"],

            df_w=fixed_result["df_w"],

            df_ghi=df_ghi,

            backend_list=backend_list,

            bounds=tracking_bounds,

            maxiter=DEFAULT_MAXITER,

            popsize=DEFAULT_POPSIZE,

            tol=DEFAULT_TOL,

            mutation_low=DEFAULT_MUTATION_LOW,

            mutation_high=DEFAULT_MUTATION_HIGH,

            recombination=DEFAULT_RECOMBINATION,

            seed=DEFAULT_SEED
        )

        df_trac["Zenith Angle"] = (
            tracking_result["zenith"]
        )

        df_trac["Panel Angle"] = (
            tracking_result["panel"]
        )

        # ----------------------------------------------------
        # ORIGINAL COLUMN NAME PRESERVED
        # ----------------------------------------------------

        df_trac[
            "Fixed Power=I*Ƞ*A"
        ] = tracking_result[
            "forecast"
        ]

        tracking_result[
            "df_trac"
        ] = df_trac

    return {
        "plant_type": plant_type,
        "lat": lat,
        "fixed": fixed_result,
        "tracking": tracking_result,
        "df_ghi": df_ghi
    }


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

if (
    not st.session_state.calculated
    or st.session_state.get(
        "last_plant_type"
    ) != plant_type
    or (
        uploaded_file is not None
        and st.session_state.get(
            "last_uploaded_name"
        ) != uploaded_file.name
    )
):

    try:

        with st.spinner(
            f"Running {plant_type} calculation..."
        ):

            source = get_excel_source()

            calculation = run_calculation(
                plant_type,
                source
            )

            st.session_state.calculation_result = (
                calculation
            )

            st.session_state.calculated = True

            st.session_state.last_plant_type = (
                plant_type
            )

            if uploaded_file is not None:

                st.session_state.last_uploaded_name = (
                    uploaded_file.name
                )

    except Exception as e:

        st.session_state.calculated = False

        st.error(
            f"{plant_type} calculation failed: {e}"
        )

        st.stop()


# ============================================================
# GET CALCULATION
# ============================================================

calculation = (
    st.session_state.calculation_result
)


# ============================================================
# FIXED UI
# ============================================================

if plant_type == "Fixed":

    fixed = calculation["fixed"]

    st.subheader("Fixed Plant")

    # --------------------------------------------------------
    # AUTOMATIC RESULT
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Automatic Error %",
        f"{fixed['best_error']:.2f}%"
    )

    c2.metric(
        "Forecast Peak",
        f"{fixed['calculated_peak']:.4f}"
    )

    c3.metric(
        "Actual Peak",
        f"{fixed['actual_peak']:.4f}"
    )

    c4.metric(
        "Peak Error %",
        f"{fixed['peak_error_pct']:.4f}%"
    )

    st.plotly_chart(
        forecast_plot(
            fixed["df_fix"][
                "Total Power (CL1+CL2+…)"
            ].to_numpy(),
            fixed["df_fix"][
                "Actual"
            ].to_numpy(),
            "Fixed Plant Forecast vs Actual"
        ),
        use_container_width=True
    )

    # --------------------------------------------------------
    # EDITABLE PARAMETERS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Edit Fixed Calculation Parameters"
    )

    f1, f2, f3 = st.columns(3)

    fixed_start = f1.number_input(
        "Error % Start",
        min_value=0.0,
        max_value=100.0,
        value=float(
            DEFAULT_FIXED_START
        ),
        step=0.1,
        key="fixed_start"
    )

    fixed_end = f2.number_input(
        "Error % End",
        min_value=0.0,
        max_value=100.0,
        value=float(
            DEFAULT_FIXED_END
        ),
        step=0.1,
        key="fixed_end"
    )

    fixed_step = f3.number_input(
        "Error % Step",
        min_value=0.001,
        max_value=10.0,
        value=float(
            DEFAULT_FIXED_STEP
        ),
        step=0.1,
        key="fixed_step"
    )

    if fixed_start >= fixed_end:

        st.error(
            "Error % Start must be smaller than Error % End."
        )

    else:

        if st.button(
            "Apply Fixed Parameters",
            type="primary",
            use_container_width=True
        ):

            try:

                source = get_excel_source()

                df, df_w = read_common_data(
                    source
                )

                df_ghi = read_ghi(
                    source
                )

                df_fix, lat = read_fixed_data(
                    source
                )

                df_fix = calculate_fixed_poa(
                    df_fix,
                    df_ghi
                )

                fixed_result = optimize_fixed(
                    df=df,
                    df_w=df_w,
                    df_fix=df_fix,
                    start_error=fixed_start,
                    end_error=fixed_end,
                    step_error=fixed_step
                )

                calculation["fixed"] = (
                    fixed_result
                )

                st.session_state.calculation_result = (
                    calculation
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Fixed recalculation failed: {e}"
                )


# ============================================================
# TRACKING UI
# ============================================================

if plant_type == "Tracking":

    tracking = calculation["tracking"]

    st.subheader("Tracking Plant")

    # --------------------------------------------------------
    # AUTOMATIC RESULT
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "DHI",
        tracking["DHI"]
    )

    c2.metric(
        "GHI Max Block",
        tracking[
            "GHI Max Block"
        ]
    )

    c3.metric(
        "Forecast Peak",
        f"{tracking['calculated_peak']:.4f}"
    )

    c4.metric(
        "Peak Error %",
        f"{tracking['peak_error_pct']:.4f}%"
    )

    st.plotly_chart(
        forecast_plot(
            tracking["forecast"],
            tracking["actual"],
            "Tracking Plant Forecast vs Actual"
        ),
        use_container_width=True
    )

    # --------------------------------------------------------
    # EDITABLE OPTIMIZATION PARAMETERS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Edit Tracking Optimization Parameters"
    )

    # ========================================================
    # PARAMETER BOUNDS
    # ========================================================

    st.markdown("### Parameter Bounds")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**DHI (%)**")

        dhi_min = st.number_input(
            "DHI Min",
            value=float(
                DEFAULT_DHI_MIN
            ),
            key="dhi_min"
        )

        dhi_max = st.number_input(
            "DHI Max",
            value=float(
                DEFAULT_DHI_MAX
            ),
            key="dhi_max"
        )

        st.markdown(
            "**GHI Starting Block**"
        )

        start_min = st.number_input(
            "Starting Block Min",
            value=float(
                DEFAULT_START_MIN
            ),
            key="start_min"
        )

        start_max = st.number_input(
            "Starting Block Max",
            value=float(
                DEFAULT_START_MAX
            ),
            key="start_max"
        )

        st.markdown(
            "**GHI Ending Block**"
        )

        end_min = st.number_input(
            "Ending Block Min",
            value=float(
                DEFAULT_END_MIN
            ),
            key="end_min"
        )

        end_max = st.number_input(
            "Ending Block Max",
            value=float(
                DEFAULT_END_MAX
            ),
            key="end_max"
        )

    with col2:

        st.markdown(
            "**GHI Max Block**"
        )

        max_min = st.number_input(
            "Max Block Min",
            value=float(
                DEFAULT_MAX_MIN
            ),
            key="max_min"
        )

        max_max = st.number_input(
            "Max Block Max",
            value=float(
                DEFAULT_MAX_MAX
            ),
            key="max_max"
        )

        st.markdown(
            "**Tracking East Limit**"
        )

        east_min = st.number_input(
            "East Limit Min",
            value=float(
                DEFAULT_EAST_MIN
            ),
            key="east_min"
        )

        east_max = st.number_input(
            "East Limit Max",
            value=float(
                DEFAULT_EAST_MAX
            ),
            key="east_max"
        )

        st.markdown(
            "**Tracking West Limit**"
        )

        west_min = st.number_input(
            "West Limit Min",
            value=float(
                DEFAULT_WEST_MIN
            ),
            key="west_min"
        )

        west_max = st.number_input(
            "West Limit Max",
            value=float(
                DEFAULT_WEST_MAX
            ),
            key="west_max"
        )

    # ========================================================
    # OPTIMIZER SETTINGS
    # ========================================================

    st.markdown("### Differential Evolution Settings")

    o1, o2, o3 = st.columns(3)

    maxiter = o1.number_input(
        "Max Iterations",
        min_value=1,
        max_value=1000,
        value=DEFAULT_MAXITER,
        step=1,
        key="maxiter"
    )

    popsize = o2.number_input(
        "Population Size",
        min_value=1,
        max_value=500,
        value=DEFAULT_POPSIZE,
        step=1,
        key="popsize"
    )

    tol = o3.number_input(
        "Tolerance",
        min_value=0.000001,
        max_value=1.0,
        value=DEFAULT_TOL,
        format="%.6f",
        key="tol"
    )

    o4, o5, o6 = st.columns(3)

    mutation_low = o4.number_input(
        "Mutation Min",
        min_value=0.0,
        max_value=2.0,
        value=DEFAULT_MUTATION_LOW,
        step=0.1,
        key="mutation_low"
    )

    mutation_high = o5.number_input(
        "Mutation Max",
        min_value=0.0,
        max_value=2.0,
        value=DEFAULT_MUTATION_HIGH,
        step=0.1,
        key="mutation_high"
    )

    recombination = o6.number_input(
        "Recombination",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_RECOMBINATION,
        step=0.1,
        key="recombination"
    )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=DEFAULT_SEED,
        step=1,
        key="seed"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    valid_parameters = True

    if dhi_min >= dhi_max:
        st.error(
            "DHI Min must be smaller than DHI Max."
        )
        valid_parameters = False

    if start_min >= start_max:
        st.error(
            "GHI Starting Block Min must be smaller than Max."
        )
        valid_parameters = False

    if end_min >= end_max:
        st.error(
            "GHI Ending Block Min must be smaller than Max."
        )
        valid_parameters = False

    if max_min >= max_max:
        st.error(
            "GHI Max Block Min must be smaller than Max."
        )
        valid_parameters = False

    if east_min >= east_max:
        st.error(
            "East Limit Min must be smaller than Max."
        )
        valid_parameters = False

    if west_min >= west_max:
        st.error(
            "West Limit Min must be smaller than Max."
        )
        valid_parameters = False

    if mutation_low >= mutation_high:
        st.error(
            "Mutation Min must be smaller than Mutation Max."
        )
        valid_parameters = False

    if valid_parameters:

        if st.button(
            "Apply Tracking Parameters",
            type="primary",
            use_container_width=True
        ):

            try:

                # ------------------------------------------------
                # SAME DATA AS AUTOMATIC CALCULATION
                # ------------------------------------------------

                source = get_excel_source()

                df, df_w = read_common_data(
                    source
                )

                df_ghi = read_ghi(
                    source
                )

                df_fix, lat = read_fixed_data(
                    source
                )

                df_fix = calculate_fixed_poa(
                    df_fix,
                    df_ghi
                )

                # ------------------------------------------------
                # FIRST APPLY FIXED ERROR OPTIMIZATION
                #
                # This is IMPORTANT because tracking uses
                # df_w effective areas generated from the
                # optimized Error %.
                # ------------------------------------------------

                fixed_result = optimize_fixed(
                    df=df,
                    df_w=df_w,
                    df_fix=df_fix,
                    start_error=DEFAULT_FIXED_START,
                    end_error=DEFAULT_FIXED_END,
                    step_error=DEFAULT_FIXED_STEP
                )

                # ------------------------------------------------
                # TRACKING DATA
                # ------------------------------------------------

                backend_list, df_trac = (
                    read_tracking_data(
                        source
                    )
                )

                tracking_bounds = [

                    (
                        dhi_min,
                        dhi_max
                    ),

                    (
                        start_min,
                        start_max
                    ),

                    (
                        end_min,
                        end_max
                    ),

                    (
                        max_min,
                        max_max
                    ),

                    (
                        east_min,
                        east_max
                    ),

                    (
                        west_min,
                        west_max
                    )
                ]

                tracking_result = (
                    optimize_tracking(

                        df_fix=fixed_result[
                            "df_fix"
                        ],

                        df_w=fixed_result[
                            "df_w"
                        ],

                        df_ghi=df_ghi,

                        backend_list=backend_list,

                        bounds=tracking_bounds,

                        maxiter=maxiter,

                        popsize=popsize,

                        tol=tol,

                        mutation_low=mutation_low,

                        mutation_high=mutation_high,

                        recombination=recombination,

                        seed=seed
                    )
                )

                df_trac[
                    "Zenith Angle"
                ] = tracking_result[
                    "zenith"
                ]

                df_trac[
                    "Panel Angle"
                ] = tracking_result[
                    "panel"
                ]

                df_trac[
                    "Fixed Power=I*Ƞ*A"
                ] = tracking_result[
                    "forecast"
                ]

                tracking_result[
                    "df_trac"
                ] = df_trac

                calculation["fixed"] = (
                    fixed_result
                )

                calculation["tracking"] = (
                    tracking_result
                )

                st.session_state.calculation_result = (
                    calculation
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Tracking recalculation failed: {e}"
                )
