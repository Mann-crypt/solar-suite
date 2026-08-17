# ============================================================
# STREAMLIT APP
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
#
# IMPORTANT:
# Error % is applied EXACTLY ONCE.
#
# Flow:
# Excel
#   ↓
# Error %
#   ↓
# Net Efficiency
#   ↓
# Effective Area
#   ↓
# Cluster Effective Area
#   ↓
# Tracking optimizer
#   ↓
# Final Forecast
#
# Tracking optimizer NEVER applies Error % again.
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
# SESSION STATE
# ============================================================

DEFAULTS = {
    "plant_type": "Fixed",

    # Automatically calculated error
    "best_error": 0.0,

    # Tracking parameters
    "DHI": 0,
    "GHI_Starting_Block": 30,
    "GHI_Ending_Block": 79,
    "GHI_Max_Block": 53,
    "Tracking_angle_lim_E": 11,
    "Tracking_angle_lim_W": 23,

    # Calculation state
    "calculated": False,
    "auto_calculated": False,
    "calculation_signature": None,
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }

    div[data-testid="stNumberInput"] input {
        font-size: 15px;
    }

    div[data-testid="stFileUploader"] {
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

POWER_COLS = [
    "CL1_Fixed Power=I*Ƞ*A",
    "CL2_Fixed Power=I*Ƞ*A",
    "CL3_Fixed Power=I*Ƞ*A",
    "CL4_Fixed Power=I*Ƞ*A",
    "CL5_Fixed Power=I*Ƞ*A",
]


# ============================================================
# SAFE EXCEL READER
# ============================================================

def read_excel(uploaded_file, sheet_name, **kwargs):

    uploaded_file.seek(0)

    return pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name,
        **kwargs
    )


# ============================================================
# REMOVE DATA AFTER FIRST NULL ROW
# ============================================================

def trim_at_first_null(df, column):

    df = df.copy()

    if column not in df.columns:
        return df

    null_indices = df[df[column].isna()].index

    if len(null_indices) > 0:

        first_null_pos = df.index.get_loc(
            null_indices[0]
        )

        df = df.iloc[:first_null_pos]

    return df.reset_index(drop=True)


# ============================================================
# LOAD AREA & EFFICIENCY
# ============================================================

def load_area_efficiency(uploaded_file):

    df = read_excel(
        uploaded_file,
        "Area & Efficiency",
        header=1,
        usecols=range(12)
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    df = trim_at_first_null(
        df,
        "S.No."
    )

    return df


# ============================================================
# LOAD CLUSTER AREA TABLE
# ============================================================

def load_cluster_table(uploaded_file):

    df_w = read_excel(
        uploaded_file,
        "Area & Efficiency",
        header=1,
        usecols=[14, 15]
    )

    df_w.columns = (
        df_w.columns
        .astype(str)
        .str.strip()
    )

    df_w = trim_at_first_null(
        df_w,
        "Clusters"
    )

    return df_w.reset_index(drop=True)


# ============================================================
# CALCULATE EFFECTIVE AREA
#
# THIS IS THE ONLY PLACE WHERE ERROR % IS APPLIED.
# ============================================================

def calculate_effective_area(
    area_df,
    cluster_df,
    error_percent
):

    df = area_df.copy()
    df_w = cluster_df.copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    standard_eff = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
        errors="coerce"
    )

    no_modules = pd.to_numeric(
        df["No of Module"],
        errors="coerce"
    )

    module_area = pd.to_numeric(
        df["Area of 1 Module (m2)"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # ERROR % APPLIED ONCE
    # --------------------------------------------------------

    df["Error %"] = float(error_percent)

    df["Net Efficiency (%)"] = (
        standard_eff - float(error_percent)
    )

    # --------------------------------------------------------
    # TOTAL AREA
    # --------------------------------------------------------

    df["Total area (m2)"] = (
        no_modules * module_area
    )

    # --------------------------------------------------------
    # EFFECTIVE AREA
    # --------------------------------------------------------

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100.0
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
        .fillna(0.0)
    )

    return df, df_w


# ============================================================
# LOAD FORECAST CONFIG
# ============================================================

def load_latitude(uploaded_file):

    df_st = read_excel(
        uploaded_file,
        "Forecast Config",
        header=8
    )

    lat = float(
        pd.to_numeric(
            df_st.loc[0, "Lat"],
            errors="coerce"
        )
    )

    return lat


# ============================================================
# LOAD TILT ANGLES
# ============================================================

def load_tilt_lookup(uploaded_file):

    df_tilt = read_excel(
        uploaded_file,
        "Config Tilt Angle",
        header=7
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    df_tilt = trim_at_first_null(
        df_tilt,
        "Fixed"
    )

    df_tilt = df_tilt.dropna(
        how="all",
        axis=1
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

    return month_lookup


# ============================================================
# LOAD GHI
# ============================================================

def load_ghi(uploaded_file):

    df_ghi = read_excel(
        uploaded_file,
        "Result",
        usecols=[0, 1, 2, 3, 4, 5]
    )

    df_ghi = df_ghi.fillna(0)

    for col in GHI_COLS:
        df_ghi[col] = pd.to_numeric(
            df_ghi[col],
            errors="coerce"
        ).fillna(0)

    return df_ghi


# ============================================================
# LOAD FIXED DATA
# ============================================================

def load_fixed_data(uploaded_file):

    df_fix = read_excel(
        uploaded_file,
        "Fixed-C11",
        header=1
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    df_fix = trim_at_first_null(
        df_fix,
        "Date"
    )

    return df_fix.reset_index(drop=True)


# ============================================================
# PREPARE FIXED CALCULATION
# ============================================================

def prepare_fixed_calculation(
    df_fix,
    df_ghi,
    lat,
    month_lookup
):

    df = df_fix.copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # Preserve the Jupyter logic.
    # --------------------------------------------------------

    today = pd.Timestamp.today()

    df["Date"] = today

    first_date = (
        today
        .replace(
            month=1,
            day=1
        )
        .normalize()
    )

    # --------------------------------------------------------
    # DECLINATION
    # --------------------------------------------------------

    df["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + (
                        df["Date"]
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

    df["Elevation angle a"] = (
        90
        - lat
        + df["Declination Angle ∆"]
    )

    # --------------------------------------------------------
    # TILT
    # --------------------------------------------------------

    df["Tilt Angle b"] = (
        df["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

    # --------------------------------------------------------
    # ANGLES
    # --------------------------------------------------------

    df["a+b"] = (
        df["Elevation angle a"]
        + df["Tilt Angle b"]
    )

    df["SIN(a+b)"] = np.sin(
        np.radians(df["a+b"])
    )

    df["Sin(a)"] = np.sin(
        np.radians(df["Elevation angle a"])
    )

    # --------------------------------------------------------
    # SAFE SIN(A)
    # --------------------------------------------------------

    sin_a = df["Sin(a)"].replace(
        0,
        np.nan
    )

    # --------------------------------------------------------
    # CLUSTER POA
    # --------------------------------------------------------

    for i, ghi_col in enumerate(GHI_COLS):

        cluster_number = i + 1

        if cluster_number == 1:

            ghi_sina = "GHI*sin(a)"
            ghi_sinab = "GHI*sin(a+b)"
            poa_col = "POA fixed"

        else:

            ghi_sina = (
                f"GHI*sin(a)-CL{cluster_number}"
            )

            ghi_sinab = (
                f"GHI*sin(a+b)-CL{cluster_number}"
            )

            poa_col = (
                f"POA Fixed-C{cluster_number}"
            )

        df[ghi_sina] = (
            df_ghi[ghi_col].to_numpy()
            * df["Sin(a)"].to_numpy()
        )

        df[ghi_sinab] = (
            df_ghi[ghi_col].to_numpy()
            * df["SIN(a+b)"].to_numpy()
        )

        df[poa_col] = (
            df[ghi_sinab]
            / sin_a
        )

    return df


# ============================================================
# CALCULATE FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df_fix,
    df_w
):

    df = df_fix.copy()

    for i in range(5):

        cluster_number = i + 1

        if cluster_number == 1:
            poa_col = "POA fixed"
        else:
            poa_col = (
                f"POA Fixed-C{cluster_number}"
            )

        power_col = (
            f"CL{cluster_number}_Fixed Power=I*Ƞ*A"
        )

        effective_area = float(
            pd.to_numeric(
                df_w.iloc[i]["Eff Area(m2)"],
                errors="coerce"
            )
            if pd.notna(
                df_w.iloc[i]["Eff Area(m2)"]
            )
            else 0.0
        )

        df[power_col] = (
            pd.to_numeric(
                df[poa_col],
                errors="coerce"
            ).fillna(0)
            * effective_area
            / 1_000_000
        )

    df["Total Power (CL1+CL2+…)"] = (
        df[POWER_COLS]
        .sum(axis=1)
    )

    return df


# ============================================================
# FIXED PEAK ERROR
# ============================================================

def fixed_peak_error(
    df_base,
    df_w_base,
    df_fix_base,
    error_percent
):

    # --------------------------------------------------------
    # Apply Error % ONCE
    # --------------------------------------------------------

    _, df_w = calculate_effective_area(
        df_base,
        df_w_base,
        error_percent
    )

    df_calc = calculate_fixed_forecast(
        df_fix_base,
        df_w
    )

    actual = pd.to_numeric(
        df_calc["Actual"],
        errors="coerce"
    )

    forecast = pd.to_numeric(
        df_calc["Total Power (CL1+CL2+…)"],
        errors="coerce"
    )

    actual_peak = actual.max()
    calculated_peak = forecast.max()

    if not np.isfinite(actual_peak):
        return np.inf

    if actual_peak == 0:
        return np.inf

    return abs(
        calculated_peak - actual_peak
    ) / actual_peak


# ============================================================
# FIND BEST ERROR %
#
# SAME 0.1% LOOP AS JUPYTER
# ============================================================

def find_best_error(
    df_base,
    df_w_base,
    df_fix_base
):

    results = []

    for error in np.arange(
        0,
        10.01,
        0.1
    ):

        error_value = round(
            float(error),
            10
        )

        error_score = fixed_peak_error(
            df_base,
            df_w_base,
            df_fix_base,
            error_value
        )

        results.append({
            "Error %": error_value,
            "Score": error_score
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result_df = result_df.dropna(
        subset=["Score"]
    )

    if result_df.empty:
        return 0.0, result_df

    best_row = result_df.loc[
        result_df["Score"].idxmin()
    ]

    best_error = float(
        best_row["Error %"]
    )

    return best_error, result_df


# ============================================================
# LOAD TRACKING BACKEND
# ============================================================

def load_tracking_data(uploaded_file):

    backend_list = []

    for cluster in CLUSTERS:

        backend = read_excel(
            uploaded_file,
            f"Backend Cal {cluster}"
        )

        backend_list.append(
            backend
        )

    df_trac = read_excel(
        uploaded_file,
        "Tracking",
        header=1
    )

    df_trac.columns = (
        df_trac.columns
        .astype(str)
        .str.strip()
    )

    return backend_list, df_trac


# ============================================================
# PREPARE TRACKING INPUT
# ============================================================

def prepare_tracking_inputs(
    df_w,
    df_ghi,
    df_fix,
    backend_list
):

    # --------------------------------------------------------
    # CLUSTER EFFECTIVE AREAS
    #
    # CRITICAL:
    # These values already contain Error %.
    #
    # DO NOT APPLY ERROR % AGAIN.
    # --------------------------------------------------------

    cl_weights = (
        pd.to_numeric(
            df_w["Eff Area(m2)"],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # GHI MATRIX
    # --------------------------------------------------------

    ghi_matrix = np.column_stack(
        [
            pd.to_numeric(
                df_ghi[col],
                errors="coerce"
            )
            .fillna(0)
            .to_numpy(dtype=float)
            for col in GHI_COLS
        ]
    )

    # --------------------------------------------------------
    # BLOCKS
    # --------------------------------------------------------

    blocks = pd.to_numeric(
        backend_list[0]["Block No."],
        errors="coerce"
    ).to_numpy(dtype=float)

    # --------------------------------------------------------
    # ACTUAL
    # --------------------------------------------------------

    actual_full = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce"
    ).fillna(0).to_numpy(dtype=float)

    # --------------------------------------------------------
    # LENGTH ALIGNMENT
    # --------------------------------------------------------

    n = min(
        len(blocks),
        len(actual_full),
        len(ghi_matrix)
    )

    blocks = blocks[:n]
    actual_full = actual_full[:n]
    ghi_matrix = ghi_matrix[:n]

    mask = (
        np.isfinite(actual_full)
        & (actual_full != 0)
    )

    actual = actual_full[mask]

    if len(actual) == 0:
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual_max = actual.max()
    actual_sum = actual.sum()

    if actual_max == 0:
        raise ValueError(
            "Tracking Actual peak is zero."
        )

    if actual_sum == 0:
        raise ValueError(
            "Tracking Actual energy is zero."
        )

    return (
        cl_weights,
        ghi_matrix,
        blocks,
        actual_full,
        mask,
        actual,
        actual_max,
        actual_sum
    )


# ============================================================
# TRACKING FORECAST FROM PARAMETERS
#
# NO ERROR % HERE.
# ============================================================

def tracking_forecast(
    x,
    blocks,
    ghi_matrix,
    cl_weights
):

    DHI = int(round(x[0]))
    GHI_Starting_Block = int(round(x[1]))
    GHI_Ending_Block = int(round(x[2]))
    GHI_Max_Block = int(round(x[3]))
    Tracking_angle_lim_E = int(round(x[4]))
    Tracking_angle_lim_W = int(round(x[5]))

    # --------------------------------------------------------
    # VALIDATE BLOCK POSITIONS
    # --------------------------------------------------------

    if not (
        GHI_Starting_Block
        < GHI_Max_Block
        < GHI_Ending_Block
    ):
        return None, None, None

    # --------------------------------------------------------
    # SAME JUPYTER FORMULAS
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

    if denominator_1 == 0:
        return None, None, None

    if denominator_2 == 0:
        return None, None, None

    m1 = (
        90
        / denominator_1
    )

    m2 = (
        90
        / denominator_2
    )

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
    #
    # cl_weights ALREADY CONTAIN EFFECTIVE AREA
    # AFTER THE SINGLE Error % APPLICATION.
    # --------------------------------------------------------

    prediction_full = (
        dni @ cl_weights
    ) / 1_000_000

    if (
        np.isnan(prediction_full).any()
        or np.isinf(prediction_full).any()
    ):
        return None, None, None

    return (
        prediction_full,
        zenith,
        panel
    )


# ============================================================
# TRACKING OBJECTIVE
#
# ERROR % IS NOT USED HERE.
# ============================================================

def make_tracking_objective(
    blocks,
    ghi_matrix,
    cl_weights,
    mask,
    actual,
    actual_max,
    actual_sum
):

    def objective(x):

        prediction_full, _, _ = (
            tracking_forecast(
                x,
                blocks,
                ghi_matrix,
                cl_weights
            )
        )

        if prediction_full is None:
            return 1e9

        prediction = (
            prediction_full[mask]
        )

        if len(prediction) == 0:
            return 1e9

        if (
            np.isnan(prediction).any()
            or np.isinf(prediction).any()
        ):
            return 1e9

        prediction_max = prediction.max()

        prediction_sum = prediction.sum()

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
                - prediction_max
            )
            / actual_max
        )

        # ----------------------------------------------------
        # ENERGY ERROR
        # ----------------------------------------------------

        energy_error = (
            abs(
                actual_sum
                - prediction_sum
            )
            / actual_sum
        )

        # ----------------------------------------------------
        # SAME JUPYTER SCORE
        # ----------------------------------------------------

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return score

    return objective


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    blocks,
    ghi_matrix,
    cl_weights,
    mask,
    actual,
    actual_max,
    actual_sum
):

    objective = make_tracking_objective(
        blocks,
        ghi_matrix,
        cl_weights,
        mask,
        actual,
        actual_max,
        actual_sum
    )

    # --------------------------------------------------------
    # SAME JUPYTER BOUNDS
    # --------------------------------------------------------

    bounds = [
        (0, 10),       # DHI
        (10, 30),      # GHI Starting Block
        (65, 80),      # GHI Ending Block
        (47, 53),      # GHI Max Block
        (10, 70),      # Tracking East Limit
        (10, 70),      # Tracking West Limit
    ]

    # --------------------------------------------------------
    # SAME OPTIMIZER SETTINGS
    # --------------------------------------------------------

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

    if not result.success:
        # Keep result anyway if it has a valid solution.
        pass

    # --------------------------------------------------------
    # SAME ROUNDING AS JUPYTER
    # --------------------------------------------------------

    best = np.round(
        result.x
    ).astype(int)

    return (
        best,
        float(result.fun),
        result
    )


# ============================================================
# FINAL TRACKING CALCULATION
# ============================================================

def calculate_final_tracking(
    best,
    blocks,
    ghi_matrix,
    cl_weights
):

    forecast, zenith, panel = (
        tracking_forecast(
            best,
            blocks,
            ghi_matrix,
            cl_weights
        )
    )

    if forecast is None:
        raise ValueError(
            "Unable to calculate final Tracking forecast."
        )

    return (
        forecast,
        zenith,
        panel
    )


# ============================================================
# AUTO CALCULATION
# ============================================================

def automatic_calculation(
    uploaded_file,
    plant_type
):

    # ========================================================
    # READ COMMON DATA
    # ========================================================

    df_area = load_area_efficiency(
        uploaded_file
    )

    df_w_base = load_cluster_table(
        uploaded_file
    )

    df_ghi = load_ghi(
        uploaded_file
    )

    df_fix_base = load_fixed_data(
        uploaded_file
    )

    lat = load_latitude(
        uploaded_file
    )

    month_lookup = load_tilt_lookup(
        uploaded_file
    )

    # ========================================================
    # PREPARE FIXED GEOMETRY
    # ========================================================

    df_fix_geometry = (
        prepare_fixed_calculation(
            df_fix_base,
            df_ghi,
            lat,
            month_lookup
        )
    )

    # ========================================================
    # FIND ERROR %
    #
    # Error % is calculated ONCE.
    # ========================================================

    best_error, error_results = (
        find_best_error(
            df_area,
            df_w_base,
            df_fix_geometry
        )
    )

    # ========================================================
    # APPLY BEST ERROR %
    #
    # THIS IS THE ONLY APPLICATION.
    # ========================================================

    df_final_area, df_w_final = (
        calculate_effective_area(
            df_area,
            df_w_base,
            best_error
        )
    )

    # ========================================================
    # FIXED
    # ========================================================

    if plant_type == "Fixed":

        df_final = calculate_fixed_forecast(
            df_fix_geometry,
            df_w_final
        )

        actual = pd.to_numeric(
            df_final["Actual"],
            errors="coerce"
        )

        forecast = pd.to_numeric(
            df_final[
                "Total Power (CL1+CL2+…)"
            ],
            errors="coerce"
        )

        return {
            "plant_type": "Fixed",
            "area_df": df_final_area,
            "cluster_df": df_w_final,
            "forecast_df": df_final,
            "best_error": best_error,
            "error_results": error_results,
            "forecast": forecast.to_numpy(),
            "actual": actual.to_numpy(),
        }

    # ========================================================
    # TRACKING
    # ========================================================

    backend_list, df_trac = (
        load_tracking_data(
            uploaded_file
        )
    )

    (
        cl_weights,
        ghi_matrix,
        blocks,
        actual_full,
        mask,
        actual,
        actual_max,
        actual_sum
    ) = prepare_tracking_inputs(
        df_w_final,
        df_ghi,
        df_fix_base,
        backend_list
    )

    # ========================================================
    # IMPORTANT:
    #
    # cl_weights already contain:
    #
    # Standard Efficiency
    #       -
    # Error %
    #       ↓
    # Net Efficiency
    #       ↓
    # Effective Area
    #
    # THERE IS NO ERROR % CALCULATION BELOW.
    # ========================================================

    (
        best_tracking,
        tracking_score,
        optimizer_result
    ) = optimize_tracking(
        blocks,
        ghi_matrix,
        cl_weights,
        mask,
        actual,
        actual_max,
        actual_sum
    )

    (
        forecast,
        zenith,
        panel
    ) = calculate_final_tracking(
        best_tracking,
        blocks,
        ghi_matrix,
        cl_weights
    )

    # ========================================================
    # SAVE FINAL TRACKING DATA
    # ========================================================

    df_trac = df_trac.copy()

    n = min(
        len(df_trac),
        len(zenith),
        len(panel),
        len(forecast)
    )

    df_trac = df_trac.iloc[:n].copy()

    df_trac["Zenith Angle"] = (
        zenith[:n]
    )

    df_trac["Panel Angle"] = (
        panel[:n]
    )

    df_trac["Fixed Power=I*Ƞ*A"] = (
        forecast[:n]
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "plant_type": "Tracking",

        "area_df": df_final_area,

        # IMPORTANT:
        # df_w_final already has Error % applied ONCE.
        "cluster_df": df_w_final,

        "tracking_df": df_trac,

        "best_error": best_error,

        "error_results": error_results,

        "tracking_score": tracking_score,

        "tracking_result": optimizer_result,

        "tracking_parameters": {
            "DHI": int(best_tracking[0]),
            "GHI Starting Block": int(best_tracking[1]),
            "GHI Ending Block": int(best_tracking[2]),
            "GHI Max Block": int(best_tracking[3]),
            "Tracking East Limit": int(best_tracking[4]),
            "Tracking West Limit": int(best_tracking[5]),
        },

        "forecast": forecast,

        "actual": actual_full,

        "blocks": blocks,

        "zenith": zenith,

        "panel": panel,

        "cl_weights": cl_weights,
    }


# ============================================================
# RE-RUN CALCULATION WITH USER PARAMETERS
# ============================================================

def recalculate_with_user_parameters(
    uploaded_file,
    plant_type,
    error_percent,
    tracking_parameters
):

    # ========================================================
    # COMMON DATA
    # ========================================================

    df_area = load_area_efficiency(
        uploaded_file
    )

    df_w_base = load_cluster_table(
        uploaded_file
    )

    df_ghi = load_ghi(
        uploaded_file
    )

    df_fix_base = load_fixed_data(
        uploaded_file
    )

    lat = load_latitude(
        uploaded_file
    )

    month_lookup = load_tilt_lookup(
        uploaded_file
    )

    df_fix_geometry = (
        prepare_fixed_calculation(
            df_fix_base,
            df_ghi,
            lat,
            month_lookup
        )
    )

    # ========================================================
    # APPLY USER ERROR %
    #
    # AGAIN:
    # EXACTLY ONCE.
    # ========================================================

    df_final_area, df_w_final = (
        calculate_effective_area(
            df_area,
            df_w_base,
            float(error_percent)
        )
    )

    # ========================================================
    # FIXED
    # ========================================================

    if plant_type == "Fixed":

        df_final = calculate_fixed_forecast(
            df_fix_geometry,
            df_w_final
        )

        return {
            "plant_type": "Fixed",
            "area_df": df_final_area,
            "cluster_df": df_w_final,
            "forecast_df": df_final,
            "forecast": df_final[
                "Total Power (CL1+CL2+…)"
            ].to_numpy(),
            "actual": pd.to_numeric(
                df_final["Actual"],
                errors="coerce"
            ).to_numpy(),
        }

    # ========================================================
    # TRACKING
    # ========================================================

    backend_list, df_trac = (
        load_tracking_data(
            uploaded_file
        )
    )

    (
        cl_weights,
        ghi_matrix,
        blocks,
        actual_full,
        mask,
        actual,
        actual_max,
        actual_sum
    ) = prepare_tracking_inputs(
        df_w_final,
        df_ghi,
        df_fix_base,
        backend_list
    )

    # ========================================================
    # USER PARAMETERS
    #
    # NO ERROR % USED HERE.
    # ========================================================

    x = np.array(
        [
            tracking_parameters[
                "DHI"
            ],

            tracking_parameters[
                "GHI Starting Block"
            ],

            tracking_parameters[
                "GHI Ending Block"
            ],

            tracking_parameters[
                "GHI Max Block"
            ],

            tracking_parameters[
                "Tracking East Limit"
            ],

            tracking_parameters[
                "Tracking West Limit"
            ],
        ],
        dtype=float
    )

    forecast, zenith, panel = (
        tracking_forecast(
            x,
            blocks,
            ghi_matrix,
            cl_weights
        )
    )

    if forecast is None:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    df_trac = df_trac.copy()

    n = min(
        len(df_trac),
        len(forecast),
        len(zenith),
        len(panel)
    )

    df_trac = df_trac.iloc[:n].copy()

    df_trac["Zenith Angle"] = (
        zenith[:n]
    )

    df_trac["Panel Angle"] = (
        panel[:n]
    )

    df_trac["Fixed Power=I*Ƞ*A"] = (
        forecast[:n]
    )

    return {
        "plant_type": "Tracking",
        "area_df": df_final_area,
        "cluster_df": df_w_final,
        "tracking_df": df_trac,
        "forecast": forecast,
        "actual": actual_full,
        "blocks": blocks,
        "zenith": zenith,
        "panel": panel,
        "cl_weights": cl_weights,
    }


# ============================================================
# UI
# ============================================================

st.title("Solar Forecast Correction")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"]
)


# ============================================================
# STOP UNTIL FILE IS UPLOADED
# ============================================================

if uploaded_file is None:

    st.stop()


# ============================================================
# FILE SIGNATURE
# ============================================================

file_signature = (
    uploaded_file.name,
    uploaded_file.size
)


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.segmented_control(
    "Plant Type",
    options=[
        "Fixed",
        "Tracking"
    ],
    default=st.session_state.plant_type
)

if plant_type is None:
    plant_type = "Fixed"

st.session_state.plant_type = plant_type


# ============================================================
# AUTOMATIC CALCULATION
#
# Run only when a new file / plant type is selected.
# ============================================================

current_signature = (
    file_signature,
    plant_type
)

if (
    st.session_state.calculation_signature
    != current_signature
):

    with st.spinner("Calculating..."):

        try:

            auto_result = automatic_calculation(
                uploaded_file,
                plant_type
            )

            st.session_state.auto_result = (
                auto_result
            )

            st.session_state.best_error = (
                auto_result["best_error"]
            )

            # ------------------------------------------------
            # Load automatically calculated Tracking values
            # ------------------------------------------------

            if plant_type == "Tracking":

                params = (
                    auto_result[
                        "tracking_parameters"
                    ]
                )

                st.session_state.DHI = (
                    params["DHI"]
                )

                st.session_state.GHI_Starting_Block = (
                    params[
                        "GHI Starting Block"
                    ]
                )

                st.session_state.GHI_Ending_Block = (
                    params[
                        "GHI Ending Block"
                    ]
                )

                st.session_state.GHI_Max_Block = (
                    params[
                        "GHI Max Block"
                    ]
                )

                st.session_state.Tracking_angle_lim_E = (
                    params[
                        "Tracking East Limit"
                    ]
                )

                st.session_state.Tracking_angle_lim_W = (
                    params[
                        "Tracking West Limit"
                    ]
                )

            st.session_state.calculation_signature = (
                current_signature
            )

            st.session_state.auto_calculated = True

        except Exception as e:

            st.error(
                f"Calculation failed: {e}"
            )

            st.stop()


# ============================================================
# EDITABLE ERROR %
# ============================================================

error_percent = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=10.0,
    value=float(
        st.session_state.best_error
    ),
    step=0.1,
    format="%.1f"
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

tracking_parameters = {}

if plant_type == "Tracking":

    col1, col2, col3 = st.columns(3)

    with col1:

        DHI = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            step=1,
            value=int(
                st.session_state.DHI
            )
        )

        GHI_Starting_Block = st.number_input(
            "GHI Starting Block",
            min_value=1,
            max_value=64,
            step=1,
            value=int(
                st.session_state.GHI_Starting_Block
            )
        )

    with col2:

        GHI_Ending_Block = st.number_input(
            "GHI Ending Block",
            min_value=2,
            max_value=95,
            step=1,
            value=int(
                st.session_state.GHI_Ending_Block
            )
        )

        GHI_Max_Block = st.number_input(
            "GHI Max Block",
            min_value=2,
            max_value=94,
            step=1,
            value=int(
                st.session_state.GHI_Max_Block
            )
        )

    with col3:

        Tracking_angle_lim_E = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            step=1,
            value=int(
                st.session_state.Tracking_angle_lim_E
            )
        )

        Tracking_angle_lim_W = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            step=1,
            value=int(
                st.session_state.Tracking_angle_lim_W
            )
        )

    tracking_parameters = {
        "DHI": DHI,
        "GHI Starting Block": GHI_Starting_Block,
        "GHI Ending Block": GHI_Ending_Block,
        "GHI Max Block": GHI_Max_Block,
        "Tracking East Limit": Tracking_angle_lim_E,
        "Tracking West Limit": Tracking_angle_lim_W,
    }


# ============================================================
# RECALCULATE WHEN USER EDITS PARAMETERS
# ============================================================

if plant_type == "Tracking":

    current_values = (
        float(error_percent),
        int(DHI),
        int(GHI_Starting_Block),
        int(GHI_Ending_Block),
        int(GHI_Max_Block),
        int(Tracking_angle_lim_E),
        int(Tracking_angle_lim_W),
    )

else:

    current_values = (
        float(error_percent),
    )


if (
    st.session_state.get(
        "last_user_values"
    ) != current_values
):

    try:

        with st.spinner("Updating calculation..."):

            final_result = (
                recalculate_with_user_parameters(
                    uploaded_file,
                    plant_type,
                    error_percent,
                    tracking_parameters
                )
            )

            st.session_state.final_result = (
                final_result
            )

            st.session_state.last_user_values = (
                current_values
            )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )
