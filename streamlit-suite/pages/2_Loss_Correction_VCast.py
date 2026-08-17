# ============================================================
# STREAMLIT APP
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
# CONSTANTS
# ============================================================

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15"
]

FIXED_POWER_COLS = [
    "CL1_Fixed Power=I*Ƞ*A",
    "CL2_Fixed Power=I*Ƞ*A",
    "CL3_Fixed Power=I*Ƞ*A",
    "CL4_Fixed Power=I*Ƞ*A",
    "CL5_Fixed Power=I*Ƞ*A"
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 10px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_excel(uploaded_file, sheet_name, **kwargs):
    """
    Read uploaded Excel file using BytesIO.
    """
    uploaded_file.seek(0)
    return pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name,
        **kwargs
    )


def clean_columns(df):
    """
    Remove * and surrounding spaces from column names.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    return df


def trim_at_first_null(df, column):
    """
    Same logic as original code:
    stop dataframe at first NaN in selected column.
    """
    df = df.copy()

    if column not in df.columns:
        return df

    null_indices = df[df[column].isna()].index

    if len(null_indices) == 0:
        return df

    first_null_pos = df.index.get_loc(null_indices[0])

    return df.iloc[:first_null_pos].copy()


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# READ COMMON DATA
# ============================================================

def load_common_data(uploaded_file):

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    df = read_excel(
        uploaded_file,
        "Area & Efficiency",
        header=[1],
        usecols=range(12)
    )

    df = clean_columns(df)

    df = trim_at_first_null(
        df,
        "S.No."
    )

    # --------------------------------------------------------
    # Cluster effective-area mapping
    # --------------------------------------------------------

    df_w = read_excel(
        uploaded_file,
        "Area & Efficiency",
        header=1,
        usecols=[14, 15]
    )

    df_w = trim_at_first_null(
        df_w,
        "Clusters"
    )

    # --------------------------------------------------------
    # Forecast Config
    # --------------------------------------------------------

    df_st = read_excel(
        uploaded_file,
        "Forecast Config",
        header=[8]
    )

    df_st.columns = df_st.columns.astype(str).str.strip()

    if "Lat" not in df_st.columns:
        raise ValueError(
            "Column 'Lat' was not found in Forecast Config."
        )

    lat = float(df_st.loc[0, "Lat"])

    # --------------------------------------------------------
    # Tilt angle
    # --------------------------------------------------------

    df_tilt = read_excel(
        uploaded_file,
        "Config Tilt Angle",
        header=[7]
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" not in df_tilt.columns:
        raise ValueError(
            "Column 'Fixed' was not found in Config Tilt Angle."
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

    # --------------------------------------------------------
    # GHI Result
    # --------------------------------------------------------

    df_ghi = read_excel(
        uploaded_file,
        "Result",
        usecols=[0, 1, 2, 3, 4, 5]
    )

    df_ghi = df_ghi.fillna(0)

    return {
        "df": df,
        "df_w": df_w,
        "lat": lat,
        "month_lookup": month_lookup,
        "df_ghi": df_ghi
    }


# ============================================================
# PREPARE EFFICIENCY
# ============================================================

def calculate_effective_area(
    df_original,
    df_w_original,
    error_percent
):

    df = df_original.copy()
    df_w = df_w_original.copy()

    # --------------------------------------------------------
    # Same calculation as original code
    # --------------------------------------------------------

    df["Error %"] = error_percent

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - df["Error %"]
    )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

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
# PREPARE FIXED DATA
# ============================================================

def prepare_fixed_data(
    uploaded_file,
    common_data
):

    df = common_data["df"].copy()
    df_w = common_data["df_w"].copy()
    lat = common_data["lat"]
    month_lookup = common_data["month_lookup"]
    df_ghi = common_data["df_ghi"].copy()

    # --------------------------------------------------------
    # Fixed-C11
    # --------------------------------------------------------

    df_fix = read_excel(
        uploaded_file,
        "Fixed-C11",
        header=[1]
    )

    df_fix = clean_columns(df_fix)

    df_fix = trim_at_first_null(
        df_fix,
        "Date"
    )

    # --------------------------------------------------------
    # Original code replaces Date with today
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
    # Declination
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
    # Elevation
    # --------------------------------------------------------

    df_fix["Elevation angle a"] = (
        90
        - lat
        + df_fix["Declination Angle ∆"]
    )

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    df_fix["Tilt Angle b"] = (
        df_fix["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

    # --------------------------------------------------------
    # a+b
    # --------------------------------------------------------

    df_fix["a+b"] = (
        df_fix["Elevation angle a"]
        + df_fix["Tilt Angle b"]
    )

    # --------------------------------------------------------
    # Trigonometry
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Cluster POA
    # --------------------------------------------------------

    for i, cluster in enumerate(CLUSTERS):

        ghi_col = f"GHI {cluster}"

        if ghi_col not in df_ghi.columns:
            raise ValueError(
                f"{ghi_col} not found in Result sheet."
            )

        if i == 0:
            suffix = ""
            poa_name = "POA fixed"
        else:
            suffix = f"-CL{i + 1}"
            poa_name = f"POA Fixed-C{str(i + 1).zfill(2)}"

        # Preserve original column naming
        if i == 0:

            df_fix["GHI*sin(a)"] = (
                df_ghi[ghi_col]
                * df_fix["Sin(a)"]
            )

            df_fix["GHI*sin(a+b)"] = (
                df_ghi[ghi_col]
                * df_fix["SIN(a+b)"]
            )

            df_fix["POA fixed"] = (
                df_fix["GHI*sin(a+b)"]
                / df_fix["Sin(a)"]
            )

        else:

            df_fix[
                f"GHI*sin(a)-CL{i + 1}"
            ] = (
                df_ghi[ghi_col]
                * df_fix["Sin(a)"]
            )

            df_fix[
                f"GHI*sin(a+b)-CL{i + 1}"
            ] = (
                df_ghi[ghi_col]
                * df_fix["SIN(a+b)"]
            )

            df_fix[poa_name] = (
                df_fix[
                    f"GHI*sin(a+b)-CL{i + 1}"
                ]
                / df_fix["Sin(a)"]
            )

    return df, df_w, df_fix


# ============================================================
# FIXED POWER CALCULATION
# ============================================================

def calculate_fixed_power(
    df_original,
    df_w_original,
    df_fix_original,
    error_percent
):

    df, df_w = calculate_effective_area(
        df_original,
        df_w_original,
        error_percent
    )

    df_fix = df_fix_original.copy()

    poa_columns = [
        "POA fixed",
        "POA Fixed-C12",
        "POA Fixed-C13",
        "POA Fixed-C14",
        "POA Fixed-C15"
    ]

    # --------------------------------------------------------
    # Cluster power
    # --------------------------------------------------------

    for i in range(5):

        power_col = FIXED_POWER_COLS[i]

        df_fix[power_col] = (
            df_fix[poa_columns[i]]
            * df_w.iloc[i]["Eff Area(m2)"]
        ) / 1_000_000

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    df_fix["Total Power (CL1+CL2+…)"] = (
        df_fix[FIXED_POWER_COLS]
        .sum(axis=1)
    )

    return df, df_w, df_fix


# ============================================================
# FIXED OPTIMIZATION
# ============================================================

def optimize_fixed(
    df,
    df_w,
    df_fix,
    error_start,
    error_end,
    error_step
):

    actual = safe_numeric(
        df_fix["Actual"]
    ).to_numpy()

    actual = actual[
        np.isfinite(actual)
    ]

    if len(actual) == 0:
        raise ValueError(
            "No valid Actual values found for Fixed."
        )

    actual_peak = np.max(actual)

    if actual_peak == 0:
        raise ValueError(
            "Actual peak is zero for Fixed."
        )

    results = []

    errors = np.arange(
        error_start,
        error_end + error_step / 2,
        error_step
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

    # --------------------------------------------------------
    # THIS WAS MISSING IN ORIGINAL CODE
    # --------------------------------------------------------

    best_index = results_df[
        "Peak Error"
    ].idxmin()

    best_error = float(
        results_df.loc[
            best_index,
            "Error %"
        ]
    )

    # --------------------------------------------------------
    # Final calculation
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
        safe_numeric(
            final_df_fix["Actual"]
        ).max()
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
        "results": results_df,
        "df": final_df,
        "df_w": final_df_w,
        "df_fix": final_df_fix,
        "calculated_peak": final_calculated_peak,
        "actual_peak": final_actual_peak,
        "peak_error": final_peak_error,
        "peak_error_pct": final_peak_error_pct
    }


# ============================================================
# READ TRACKING DATA
# ============================================================

def prepare_tracking_data(
    uploaded_file,
    df_w
):

    backend_list = []

    for cluster in CLUSTERS:

        sheet = f"Backend Cal {cluster}"

        backend = read_excel(
            uploaded_file,
            sheet_name=sheet
        )

        backend.columns = (
            backend.columns
            .astype(str)
            .str.strip()
        )

        backend_list.append(
            backend
        )

    # --------------------------------------------------------
    # Tracking sheet
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fixed-C11 is still read because this is how your original
    # tracking calculation was structured.
    # --------------------------------------------------------

    df_fix = read_excel(
        uploaded_file,
        "Fixed-C11",
        header=1
    )

    df_fix = clean_columns(df_fix)

    df_fix = trim_at_first_null(
        df_fix,
        "Date"
    )

    return (
        backend_list,
        df_trac,
        df_fix
    )


# ============================================================
# FIND TRACKING ACTUAL
# ============================================================

def get_tracking_actual(
    df_trac,
    df_fix
):

    # --------------------------------------------------------
    # Preferred source:
    # Tracking sheet if it has Actual and valid data.
    #
    # This prevents the previous:
    # "No non-zero Actual values found"
    #
    # If Tracking does not contain Actual, the original
    # Fixed-C11 Actual source is used.
    # --------------------------------------------------------

    if "Actual" in df_trac.columns:

        tracking_actual = pd.to_numeric(
            df_trac["Actual"],
            errors="coerce"
        ).fillna(0).to_numpy(
            dtype=float
        )

        if np.any(tracking_actual != 0):
            return tracking_actual

    if "Actual" in df_fix.columns:

        fixed_actual = pd.to_numeric(
            df_fix["Actual"],
            errors="coerce"
        ).fillna(0).to_numpy(
            dtype=float
        )

        if np.any(fixed_actual != 0):
            return fixed_actual

    raise ValueError(
        "No non-zero Actual values found for Tracking. "
        "Neither Tracking nor Fixed-C11 contains usable Actual data."
    )


# ============================================================
# TRACKING FORECAST FUNCTION
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
    # Validation
    # --------------------------------------------------------

    if not (
        GHI_Starting_Block
        < GHI_Max_Block
        < GHI_Ending_Block
    ):
        return None

    # --------------------------------------------------------
    # Slopes
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
    # Zenith
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
    # Panel
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
    # Cosine
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
    # Power
    # --------------------------------------------------------

    prediction = (
        dni @ cl_weights
    ) / 1_000_000

    if (
        np.isnan(prediction).any()
        or np.isinf(prediction).any()
    ):
        return None

    return (
        prediction,
        zenith,
        panel
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    df_w,
    backend_list,
    df_trac,
    df_fix,
    dhi_min,
    dhi_max,
    start_min,
    start_max,
    end_min,
    end_max,
    max_min,
    max_max,
    east_min,
    east_max,
    west_min,
    west_max,
    maxiter,
    popsize,
    tolerance
):

    # --------------------------------------------------------
    # Cluster weights
    # --------------------------------------------------------

    cl_weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    # Result sheet must be read separately by caller.
    #
    # We use the original data attached to df_fix if possible.
    # This function expects GHI matrix to be passed through
    # df_fix attributes only in the wrapper below.
    # --------------------------------------------------------

    raise RuntimeError(
        "Internal tracking optimizer setup error."
    )


# ============================================================
# TRACKING OPTIMIZER
# ============================================================

def run_tracking_optimization(
    df_w,
    df_ghi,
    backend_list,
    df_trac,
    df_fix,
    dhi_min,
    dhi_max,
    start_min,
    start_max,
    end_min,
    end_max,
    max_min,
    max_max,
    east_min,
    east_max,
    west_min,
    west_max,
    maxiter,
    popsize,
    tolerance
):

    # --------------------------------------------------------
    # Cluster weights
    # --------------------------------------------------------

    cl_weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # GHI matrix
    # --------------------------------------------------------

    ghi_matrix = np.column_stack([
        pd.to_numeric(
            df_ghi[col],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
        for col in GHI_COLS
    ])

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    blocks = pd.to_numeric(
        backend_list[0]["Block No."],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Match lengths
    # --------------------------------------------------------

    n = min(
        len(blocks),
        len(ghi_matrix)
    )

    blocks = blocks[:n]
    ghi_matrix = ghi_matrix[:n]

    # --------------------------------------------------------
    # Actual
    # --------------------------------------------------------

    actual_full = get_tracking_actual(
        df_trac,
        df_fix
    )

    actual_full = actual_full[:n]

    # --------------------------------------------------------
    # Mask
    #
    # SAME logic as original:
    # actual != 0
    # --------------------------------------------------------

    mask = (
        np.isfinite(actual_full)
        & (actual_full != 0)
    )

    if not np.any(mask):
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual = actual_full[mask]

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

    # --------------------------------------------------------
    # Objective
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

        prediction_full = result[0]

        prediction = (
            prediction_full[mask]
        )

        if len(prediction) == 0:
            return 1e9

        # ----------------------------------------------------
        # Block error
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
        # Peak error
        # ----------------------------------------------------

        peak_error = (
            abs(
                actual_max
                - prediction.max()
            )
            / actual_max
        )

        # ----------------------------------------------------
        # Energy error
        # ----------------------------------------------------

        energy_error = (
            abs(
                actual_sum
                - prediction.sum()
            )
            / actual_sum
        )

        # ----------------------------------------------------
        # SAME SCORE
        # ----------------------------------------------------

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return score

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    bounds = [
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

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=int(maxiter),
        popsize=int(popsize),
        tol=float(tolerance),
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1
    )

    # --------------------------------------------------------
    # Round exactly as original
    # --------------------------------------------------------

    best = np.round(
        result.x
    ).astype(int)

    # --------------------------------------------------------
    # Final forecast
    # --------------------------------------------------------

    final = tracking_forecast(
        best,
        blocks,
        ghi_matrix,
        cl_weights
    )

    if final is None:
        raise ValueError(
            "Unable to generate final Tracking forecast."
        )

    forecast, zenith, panel = final

    # --------------------------------------------------------
    # Final metrics
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

    energy_error = abs(
        actual_sum
        - prediction.sum()
    )

    energy_error_pct = (
        energy_error
        / actual_sum
        * 100
    )

    block_error = (
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
        "best": best,
        "forecast": forecast,
        "zenith": zenith,
        "panel": panel,
        "blocks": blocks,
        "actual": actual_full,
        "score": result.fun,
        "block_error_pct": block_error,
        "peak_error": peak_error,
        "peak_error_pct": peak_error_pct,
        "energy_error": energy_error,
        "energy_error_pct": energy_error_pct,
        "actual_peak": actual_max,
        "calculated_peak": calculated_peak,
        "df_trac": df_trac
    }


# ============================================================
# PLOTLY FORECAST GRAPH
# ============================================================

def make_forecast_graph(
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
            y=forecast[:n],
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
            y=actual[:n],
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
        height=500,
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

    fig.update_xaxes(
        range=[
            0,
            max(96, n - 1)
        ]
    )

    return fig


# ============================================================
# UI HEADER
# ============================================================

st.title(
    "☀️ Solar Forecast Correction"
)

st.caption(
    "Fixed / Tracking optimization using the uploaded Excel workbook"
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
    default="Fixed"
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"],
    help="Upload the Excel workbook containing the required sheets."
)


if uploaded_file is None:

    st.info(
        "Upload the Excel workbook to start automatic calculation."
    )

    st.stop()


# ============================================================
# LOAD COMMON DATA
# ============================================================

try:

    common = load_common_data(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


df_original = common["df"]
df_w_original = common["df_w"]
df_ghi = common["df_ghi"]


# ============================================================
# SESSION STATE
# ============================================================

if "calculation_done" not in st.session_state:
    st.session_state.calculation_done = False

if "fixed_result" not in st.session_state:
    st.session_state.fixed_result = None

if "tracking_result" not in st.session_state:
    st.session_state.tracking_result = None


# ============================================================
# FIXED UI
# ============================================================

if plant_type == "Fixed":

    st.markdown(
        '<div class="section-title">Fixed Plant Parameters</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        error_start = st.number_input(
            "Error % Start",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=0.1
        )

    with col2:

        error_end = st.number_input(
            "Error % End",
            min_value=-100.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        )

    with col3:

        error_step = st.number_input(
            "Error % Step",
            min_value=0.001,
            max_value=10.0,
            value=0.1,
            step=0.1
        )

    calculate_fixed_button = st.button(
        "Calculate Fixed Forecast",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # AUTOMATIC FIRST CALCULATION
    # --------------------------------------------------------

    if (
        not st.session_state.calculation_done
        or calculate_fixed_button
    ):

        try:

            with st.spinner(
                "Calculating Fixed forecast..."
            ):

                (
                    df_base,
                    df_w_base,
                    df_fix_base
                ) = prepare_fixed_data(
                    uploaded_file,
                    common
                )

                fixed_result = optimize_fixed(
                    df_base,
                    df_w_base,
                    df_fix_base,
                    error_start,
                    error_end,
                    error_step
                )

                st.session_state.fixed_result = (
                    fixed_result
                )

                st.session_state.calculation_done = True

        except Exception as e:

            st.error(
                f"Fixed calculation failed: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # AUTO OPTIMIZED VALUE
    # --------------------------------------------------------

    result = (
        st.session_state.fixed_result
    )

    if result is not None:

        automatic_error = result[
            "best_error"
        ]

        st.markdown(
            '<div class="section-title">Optimized Parameter</div>',
            unsafe_allow_html=True
        )

        editable_error = st.number_input(
            "Error %",
            min_value=-100.0,
            max_value=100.0,
            value=float(
                automatic_error
            ),
            step=0.1,
            key="fixed_editable_error"
        )

        recalculate_fixed = st.button(
            "Recalculate with Editable Error %",
            use_container_width=True
        )

        # ----------------------------------------------------
        # USER-EDITED RECALCULATION
        # ----------------------------------------------------

        if recalculate_fixed:

            try:

                with st.spinner(
                    "Recalculating..."
                ):

                    df_base, df_w_base, df_fix_base = (
                        prepare_fixed_data(
                            uploaded_file,
                            common
                        )
                    )

                    (
                        final_df,
                        final_df_w,
                        final_df_fix
                    ) = calculate_fixed_power(
                        df_base,
                        df_w_base,
                        df_fix_base,
                        editable_error
                    )

                    actual_peak = (
                        pd.to_numeric(
                            final_df_fix["Actual"],
                            errors="coerce"
                        ).max()
                    )

                    calculated_peak = (
                        final_df_fix[
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
                        if actual_peak != 0
                        else np.nan
                    )

                    result["best_error"] = (
                        editable_error
                    )

                    result["df"] = final_df
                    result["df_w"] = final_df_w
                    result["df_fix"] = final_df_fix

                    result["actual_peak"] = (
                        actual_peak
                    )

                    result["calculated_peak"] = (
                        calculated_peak
                    )

                    result["peak_error"] = (
                        peak_error
                    )

                    result["peak_error_pct"] = (
                        peak_error_pct
                    )

                    st.session_state.fixed_result = (
                        result
                    )

            except Exception as e:

                st.error(
                    f"Recalculation failed: {e}"
                )

        # ----------------------------------------------------
        # GRAPH ONLY
        # ----------------------------------------------------

        final_result = (
            st.session_state.fixed_result
        )

        if final_result is not None:

            final_fix = final_result[
                "df_fix"
            ]

            forecast = final_fix[
                "Total Power (CL1+CL2+…)"
            ].to_numpy()

            actual = pd.to_numeric(
                final_fix["Actual"],
                errors="coerce"
            ).fillna(0).to_numpy()

            st.plotly_chart(
                make_forecast_graph(
                    forecast,
                    actual,
                    "Fixed Forecast vs Actual"
                ),
                use_container_width=True
            )


# ============================================================
# TRACKING UI
# ============================================================

if plant_type == "Tracking":

    st.markdown(
        '<div class="section-title">Tracking Optimization Parameters</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DHI
    # --------------------------------------------------------

    st.markdown("**DHI (%)**")

    c1, c2 = st.columns(2)

    with c1:

        dhi_min = st.number_input(
            "DHI Minimum",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )

    with c2:

        dhi_max = st.number_input(
            "DHI Maximum",
            min_value=0,
            max_value=100,
            value=10,
            step=1
        )

    # --------------------------------------------------------
    # GHI START
    # --------------------------------------------------------

    st.markdown("**GHI Starting Block**")

    c1, c2 = st.columns(2)

    with c1:

        start_min = st.number_input(
            "Starting Block Minimum",
            min_value=0,
            max_value=95,
            value=10,
            step=1
        )

    with c2:

        start_max = st.number_input(
            "Starting Block Maximum",
            min_value=0,
            max_value=95,
            value=30,
            step=1
        )

    # --------------------------------------------------------
    # GHI END
    # --------------------------------------------------------

    st.markdown("**GHI Ending Block**")

    c1, c2 = st.columns(2)

    with c1:

        end_min = st.number_input(
            "Ending Block Minimum",
            min_value=0,
            max_value=95,
            value=65,
            step=1
        )

    with c2:

        end_max = st.number_input(
            "Ending Block Maximum",
            min_value=0,
            max_value=95,
            value=80,
            step=1
        )

    # --------------------------------------------------------
    # GHI MAX
    # --------------------------------------------------------

    st.markdown("**GHI Max Block**")

    c1, c2 = st.columns(2)

    with c1:

        max_min = st.number_input(
            "Max Block Minimum",
            min_value=0,
            max_value=95,
            value=47,
            step=1
        )

    with c2:

        max_max = st.number_input(
            "Max Block Maximum",
            min_value=0,
            max_value=95,
            value=53,
            step=1
        )

    # --------------------------------------------------------
    # EAST
    # --------------------------------------------------------

    st.markdown("**Tracking East Limit**")

    c1, c2 = st.columns(2)

    with c1:

        east_min = st.number_input(
            "East Limit Minimum",
            min_value=0,
            max_value=90,
            value=10,
            step=1
        )

    with c2:

        east_max = st.number_input(
            "East Limit Maximum",
            min_value=0,
            max_value=90,
            value=70,
            step=1
        )

    # --------------------------------------------------------
    # WEST
    # --------------------------------------------------------

    st.markdown("**Tracking West Limit**")

    c1, c2 = st.columns(2)

    with c1:

        west_min = st.number_input(
            "West Limit Minimum",
            min_value=0,
            max_value=90,
            value=10,
            step=1
        )

    with c2:

        west_max = st.number_input(
            "West Limit Maximum",
            min_value=0,
            max_value=90,
            value=70,
            step=1
        )

    # --------------------------------------------------------
    # OPTIMIZER PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        "**Optimizer Parameters**"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        maxiter = st.number_input(
            "Max Iterations",
            min_value=1,
            max_value=500,
            value=40,
            step=1
        )

    with c2:

        popsize = st.number_input(
            "Population Size",
            min_value=1,
            max_value=100,
            value=15,
            step=1
        )

    with c3:

        tolerance = st.number_input(
            "Tolerance",
            min_value=0.000001,
            max_value=1.0,
            value=0.001,
            step=0.001,
            format="%.6f"
        )

    calculate_tracking_button = st.button(
        "Calculate Tracking Forecast",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # AUTOMATIC CALCULATION
    # --------------------------------------------------------

    if (
        not st.session_state.calculation_done
        or calculate_tracking_button
    ):

        try:

            with st.spinner(
                "Optimizing Tracking parameters..."
            ):

                (
                    backend_list,
                    df_trac,
                    df_fix
                ) = prepare_tracking_data(
                    uploaded_file,
                    df_w_original
                )

                tracking_result = (
                    run_tracking_optimization(
                        df_w_original,
                        df_ghi,
                        backend_list,
                        df_trac,
                        df_fix,

                        dhi_min,
                        dhi_max,

                        start_min,
                        start_max,

                        end_min,
                        end_max,

                        max_min,
                        max_max,

                        east_min,
                        east_max,

                        west_min,
                        west_max,

                        maxiter,
                        popsize,
                        tolerance
                    )
                )

                st.session_state.tracking_result = (
                    tracking_result
                )

                st.session_state.calculation_done = True

        except Exception as e:

            st.error(
                f"Tracking optimization failed: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # EDITABLE OPTIMIZED VALUES
    # --------------------------------------------------------

    result = (
        st.session_state.tracking_result
    )

    if result is not None:

        best = result["best"]

        st.markdown(
            '<div class="section-title">Optimized Values</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            edit_dhi = st.number_input(
                "DHI (%)",
                min_value=0,
                max_value=100,
                value=int(best[0]),
                step=1,
                key="edit_dhi"
            )

        with c2:

            edit_start = st.number_input(
                "GHI Starting Block",
                min_value=0,
                max_value=95,
                value=int(best[1]),
                step=1,
                key="edit_start"
            )

        with c3:

            edit_end = st.number_input(
                "GHI Ending Block",
                min_value=0,
                max_value=95,
                value=int(best[2]),
                step=1,
                key="edit_end"
            )

        c1, c2, c3 = st.columns(3)

        with c1:

            edit_max = st.number_input(
                "GHI Max Block",
                min_value=0,
                max_value=95,
                value=int(best[3]),
                step=1,
                key="edit_max"
            )

        with c2:

            edit_east = st.number_input(
                "Tracking East Limit",
                min_value=0,
                max_value=90,
                value=int(best[4]),
                step=1,
                key="edit_east"
            )

        with c3:

            edit_west = st.number_input(
                "Tracking West Limit",
                min_value=0,
                max_value=90,
                value=int(best[5]),
                step=1,
                key="edit_west"
            )

        # ----------------------------------------------------
        # MANUAL RECALCULATION
        # ----------------------------------------------------

        recalculate_tracking = st.button(
            "Recalculate with Edited Parameters",
            use_container_width=True
        )

        if recalculate_tracking:

            try:

                # ------------------------------------------------
                # Re-read source data
                # ------------------------------------------------

                (
                    backend_list,
                    df_trac,
                    df_fix
                ) = prepare_tracking_data(
                    uploaded_file,
                    df_w_original
                )

                # ------------------------------------------------
                # Same calculation as optimizer
                # ------------------------------------------------

                cl_weights = (
                    pd.to_numeric(
                        df_w_original.iloc[:5, 1],
                        errors="coerce"
                    )
                    .fillna(0)
                    .to_numpy(
                        dtype=float
                    )
                )

                ghi_matrix = np.column_stack([
                    pd.to_numeric(
                        df_ghi[col],
                        errors="coerce"
                    )
                    .fillna(0)
                    .to_numpy(
                        dtype=float
                    )
                    for col in GHI_COLS
                ])

                blocks = pd.to_numeric(
                    backend_list[0]["Block No."],
                    errors="coerce"
                ).to_numpy(
                    dtype=float
                )

                n = min(
                    len(blocks),
                    len(ghi_matrix)
                )

                blocks = blocks[:n]

                ghi_matrix = (
                    ghi_matrix[:n]
                )

                actual_full = (
                    get_tracking_actual(
                        df_trac,
                        df_fix
                    )
                )

                actual_full = (
                    actual_full[:n]
                )

                manual_parameters = np.array([
                    edit_dhi,
                    edit_start,
                    edit_end,
                    edit_max,
                    edit_east,
                    edit_west
                ])

                final = tracking_forecast(
                    manual_parameters,
                    blocks,
                    ghi_matrix,
                    cl_weights
                )

                if final is None:

                    raise ValueError(
                        "Invalid tracking parameters. "
                        "Make sure Starting Block < Max Block < Ending Block."
                    )

                forecast, zenith, panel = final

                mask = (
                    np.isfinite(actual_full)
                    & (actual_full != 0)
                )

                if not np.any(mask):

                    raise ValueError(
                        "No non-zero Actual values found."
                    )

                actual = (
                    actual_full[mask]
                )

                prediction = (
                    forecast[mask]
                )

                actual_peak = (
                    actual.max()
                )

                calculated_peak = (
                    prediction.max()
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

                result["best"] = (
                    manual_parameters
                )

                result["forecast"] = (
                    forecast
                )

                result["zenith"] = (
                    zenith
                )

                result["panel"] = (
                    panel
                )

                result["actual"] = (
                    actual_full
                )

                result["actual_peak"] = (
                    actual_peak
                )

                result["calculated_peak"] = (
                    calculated_peak
                )

                result["peak_error"] = (
                    peak_error
                )

                result["peak_error_pct"] = (
                    peak_error_pct
                )

                st.session_state.tracking_result = (
                    result
                )

            except Exception as e:

                st.error(
                    f"Tracking recalculation failed: {e}"
                )

        # ----------------------------------------------------
        # FINAL GRAPH
        # ----------------------------------------------------

        final_result = (
            st.session_state.tracking_result
        )

        if final_result is not None:

            forecast = (
                final_result["forecast"]
            )

            actual = (
                final_result["actual"]
            )

            st.plotly_chart(
                make_forecast_graph(
                    forecast,
                    actual,
                    "Tracking Forecast vs Actual"
                ),
                use_container_width=True
            )
