# ============================================================
# STREAMLIT APP
# SOLAR LOSS CORRECTION
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
    page_title="Solar Loss Correction",
    page_icon="☀️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("☀️ Solar Loss Correction")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_excel_sheet(file_bytes, sheet_name, **kwargs):
    """
    Read Excel sheet from uploaded file bytes.
    No Streamlit caching is used here because returning
    openpyxl workbook objects causes serialization problems.
    """
    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
        **kwargs
    )


def clean_columns(df):
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
    Keep rows until first null value in the specified column.
    If no null exists, return the complete dataframe.
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
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


def safe_divide(a, b):
    b = np.asarray(b, dtype=float)

    return np.divide(
        np.asarray(a, dtype=float),
        b,
        out=np.zeros_like(
            np.asarray(a, dtype=float),
            dtype=float
        ),
        where=np.abs(b) > 1e-9
    )


# ============================================================
# LOAD BASIC DATA
# ============================================================

def load_common_data(file_bytes):

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    df = read_excel_sheet(
        file_bytes,
        "Area & Efficiency",
        header=1,
        usecols=range(12)
    )

    df = clean_columns(df)

    df = trim_at_first_null(
        df,
        "S.No."
    )

    # --------------------------------------------------------
    # Make sure required columns are numeric
    # --------------------------------------------------------

    numeric_columns = [
        "Standard PV Efficiency (%)",
        "Error %",
        "Total area (m2)",
        "No of Module",
        "Area of 1 Module (m2)"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    # --------------------------------------------------------
    # Total area
    # --------------------------------------------------------

    if (
        "No of Module" in df.columns
        and "Area of 1 Module (m2)" in df.columns
    ):
        df["Total area (m2)"] = (
            df["No of Module"]
            * df["Area of 1 Module (m2)"]
        )

    # --------------------------------------------------------
    # Default Error %
    # --------------------------------------------------------

    if "Error %" not in df.columns:
        df["Error %"] = 0.0

    # --------------------------------------------------------
    # Net efficiency
    # --------------------------------------------------------

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - df["Error %"]
    )

    # --------------------------------------------------------
    # Effective area
    # --------------------------------------------------------

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

    # --------------------------------------------------------
    # Cluster effective area
    # --------------------------------------------------------

    df_w = read_excel_sheet(
        file_bytes,
        "Area & Efficiency",
        header=1,
        usecols=[14, 15]
    )

    df_w = clean_columns(df_w)

    df_w = trim_at_first_null(
        df_w,
        "Clusters"
    )

    if "Clusters" not in df_w.columns:
        raise ValueError(
            "Column 'Clusters' was not found in Area & Efficiency."
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

    # --------------------------------------------------------
    # Latitude
    # --------------------------------------------------------

    df_st = read_excel_sheet(
        file_bytes,
        "Forecast Config",
        header=8
    )

    df_st.columns = (
        df_st.columns
        .astype(str)
        .str.strip()
    )

    if "Lat" not in df_st.columns:
        raise ValueError(
            "Column 'Lat' was not found in Forecast Config."
        )

    lat_values = pd.to_numeric(
        df_st["Lat"],
        errors="coerce"
    ).dropna()

    if len(lat_values) == 0:
        raise ValueError(
            "Latitude could not be read from Forecast Config."
        )

    lat = float(lat_values.iloc[0])

    # --------------------------------------------------------
    # Tilt angle
    # --------------------------------------------------------

    df_tilt = read_excel_sheet(
        file_bytes,
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
            "Unnamed: 3": "Month"
        }
    )

    if (
        "Month" not in df_tilt.columns
        or "Fixed" not in df_tilt.columns
    ):
        raise ValueError(
            "Month/Fixed columns were not found in Config Tilt Angle."
        )

    df_tilt["Month"] = (
        df_tilt["Month"]
        .astype(str)
        .str.strip()
    )

    df_tilt["Fixed"] = pd.to_numeric(
        df_tilt["Fixed"],
        errors="coerce"
    )

    month_lookup = (
        df_tilt
        .dropna(subset=["Month"])
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # GHI Result
    # --------------------------------------------------------

    df_ghi = read_excel_sheet(
        file_bytes,
        "Result",
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
            raise ValueError(
                f"Column '{col}' was not found in Result sheet."
            )

        df_ghi[col] = pd.to_numeric(
            df_ghi[col],
            errors="coerce"
        ).fillna(0)

    # --------------------------------------------------------
    # Fixed C11
    # --------------------------------------------------------

    df_fix = read_excel_sheet(
        file_bytes,
        "Fixed-C11",
        header=1
    )

    df_fix = clean_columns(df_fix)

    df_fix = trim_at_first_null(
        df_fix,
        "Date"
    )

    if "Actual" not in df_fix.columns:
        raise ValueError(
            "Column 'Actual' was not found in Fixed-C11."
        )

    df_fix["Actual"] = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # Keep common length consistent
    # --------------------------------------------------------

    n = min(
        len(df_fix),
        len(df_ghi)
    )

    df_fix = df_fix.iloc[:n].copy()
    df_ghi = df_ghi.iloc[:n].copy()

    return {
        "df": df,
        "df_w": df_w,
        "df_st": df_st,
        "df_tilt": df_tilt,
        "month_lookup": month_lookup,
        "df_ghi": df_ghi,
        "df_fix": df_fix,
        "lat": lat
    }


# ============================================================
# SOLAR ANGLE CALCULATION
# ============================================================

def prepare_fixed_calculation(
    df_fix,
    df_ghi,
    lat,
    month_lookup
):

    df_fix = df_fix.copy()

    n = min(
        len(df_fix),
        len(df_ghi)
    )

    df_fix = df_fix.iloc[:n].copy()
    df_ghi = df_ghi.iloc[:n].copy()

    # --------------------------------------------------------
    # Date
    #
    # Same logic as your Jupyter code:
    # every block uses today's date
    # --------------------------------------------------------

    today = pd.Timestamp.today().normalize()

    df_fix["Date"] = today

    first_date = today.replace(
        month=1,
        day=1
    )

    # --------------------------------------------------------
    # Declination Angle
    # --------------------------------------------------------

    day_number = (
        df_fix["Date"] - first_date
    ).dt.days

    df_fix["Declination Angle ∆"] = (
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

    df_fix["Tilt Angle b"] = pd.to_numeric(
        df_fix["Tilt Angle b"],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # a + b
    # --------------------------------------------------------

    df_fix["a+b"] = (
        df_fix["Elevation angle a"]
        + df_fix["Tilt Angle b"]
    )

    # --------------------------------------------------------
    # Sine
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
    # Fixed C11
    # --------------------------------------------------------

    df_fix["GHI*sin(a)"] = (
        df_ghi["GHI C11"].to_numpy()
        * df_fix["Sin(a)"].to_numpy()
    )

    df_fix["GHI*sin(a+b)"] = (
        df_ghi["GHI C11"].to_numpy()
        * df_fix["SIN(a+b)"].to_numpy()
    )

    df_fix["POA fixed"] = safe_divide(
        df_fix["GHI*sin(a+b)"],
        df_fix["Sin(a)"]
    )

    # --------------------------------------------------------
    # C12
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL2"] = (
        df_ghi["GHI C12"].to_numpy()
        * df_fix["Sin(a)"].to_numpy()
    )

    df_fix["GHI*sin(a+b)-CL2"] = (
        df_ghi["GHI C12"].to_numpy()
        * df_fix["SIN(a+b)"].to_numpy()
    )

    df_fix["POA Fixed-C12"] = safe_divide(
        df_fix["GHI*sin(a+b)-CL2"],
        df_fix["Sin(a)"]
    )

    # --------------------------------------------------------
    # C13
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL3"] = (
        df_ghi["GHI C13"].to_numpy()
        * df_fix["Sin(a)"].to_numpy()
    )

    df_fix["GHI*sin(a+b)-CL3"] = (
        df_ghi["GHI C13"].to_numpy()
        * df_fix["SIN(a+b)"].to_numpy()
    )

    df_fix["POA Fixed-C13"] = safe_divide(
        df_fix["GHI*sin(a+b)-CL3"],
        df_fix["Sin(a)"]
    )

    # --------------------------------------------------------
    # C14
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL4"] = (
        df_ghi["GHI C14"].to_numpy()
        * df_fix["Sin(a)"].to_numpy()
    )

    df_fix["GHI*sin(a+b)-CL4"] = (
        df_ghi["GHI C14"].to_numpy()
        * df_fix["SIN(a+b)"].to_numpy()
    )

    df_fix["POA Fixed-C14"] = safe_divide(
        df_fix["GHI*sin(a+b)-CL4"],
        df_fix["Sin(a)"]
    )

    # --------------------------------------------------------
    # C15
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL5"] = (
        df_ghi["GHI C15"].to_numpy()
        * df_fix["Sin(a)"].to_numpy()
    )

    df_fix["GHI*sin(a+b)-CL5"] = (
        df_ghi["GHI C15"].to_numpy()
        * df_fix["SIN(a+b)"].to_numpy()
    )

    df_fix["POA Fixed-C15"] = safe_divide(
        df_fix["GHI*sin(a+b)-CL5"],
        df_fix["Sin(a)"]
    )

    return df_fix


# ============================================================
# CALCULATE EFFECTIVE AREA
# ============================================================

def calculate_effective_area(
    df_original,
    df_w,
    error
):

    df = df_original.copy()
    df_w = df_w.copy()

    # --------------------------------------------------------
    # Error %
    # --------------------------------------------------------

    df["Error %"] = error

    # --------------------------------------------------------
    # Net efficiency
    # --------------------------------------------------------

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - df["Error %"]
    )

    # --------------------------------------------------------
    # Total area
    # --------------------------------------------------------

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    # --------------------------------------------------------
    # Effective area
    # --------------------------------------------------------

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

    # --------------------------------------------------------
    # Cluster area
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

    return df, df_w


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df_fix,
    df_w
):

    df_fix = df_fix.copy()

    # --------------------------------------------------------
    # Number of clusters
    # --------------------------------------------------------

    weights = (
        pd.to_numeric(
            df_w["Eff Area(m2)"],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    weights = np.pad(
        weights,
        (0, max(0, 5 - len(weights))),
        constant_values=0
    )[:5]

    # --------------------------------------------------------
    # Cluster powers
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

        cluster_no = i + 1

        power_col = (
            f"CL{cluster_no}_Fixed Power=I*Ƞ*A"
        )

        df_fix[power_col] = (
            df_fix[poa_cols[i]]
            * weights[i]
            / 1_000_000
        )

        power_cols.append(power_col)

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    df_fix["Total Power (CL1+CL2+…)"] = (
        df_fix[power_cols]
        .sum(axis=1)
    )

    return df_fix


# ============================================================
# FIND BEST FIXED ERROR %
# ============================================================

def optimize_fixed_error(
    df_original,
    df_w,
    df_fix_base
):

    actual = pd.to_numeric(
        df_fix_base["Actual"],
        errors="coerce"
    ).fillna(0).to_numpy()

    actual_peak = (
        np.max(actual)
        if len(actual) > 0
        else 0
    )

    results = []

    best_error = 0.0
    best_peak_error_pct = np.inf

    # --------------------------------------------------------
    # EXACT RANGE FROM JUPYTER
    # --------------------------------------------------------

    for error in np.arange(
        0,
        10.01,
        0.1
    ):

        df, df_w_temp = calculate_effective_area(
            df_original,
            df_w,
            error
        )

        df_result = calculate_fixed_forecast(
            df_fix_base,
            df_w_temp
        )

        calculated_peak = (
            df_result[
                "Total Power (CL1+CL2+…)"
            ].max()
        )

        peak_error = abs(
            calculated_peak
            - actual_peak
        )

        if actual_peak != 0:

            peak_error_pct = (
                peak_error
                / actual_peak
                * 100
            )

        else:

            peak_error_pct = np.nan

        results.append({
            "Error %": error,
            "Calculated Peak": calculated_peak,
            "Actual Peak": actual_peak,
            "Peak Error": peak_error,
            "Peak Error %": peak_error_pct
        })

        if (
            np.isfinite(peak_error_pct)
            and peak_error_pct
            < best_peak_error_pct
        ):

            best_peak_error_pct = (
                peak_error_pct
            )

            best_error = float(
                error
            )

    results_df = pd.DataFrame(
        results
    )

    return (
        best_error,
        results_df
    )


# ============================================================
# FINAL FIXED CALCULATION
# ============================================================

def run_fixed(
    data
):

    df_original = data["df"]
    df_w = data["df_w"]
    df_fix_base = data["df_fix"]

    # --------------------------------------------------------
    # Optimize Error %
    # --------------------------------------------------------

    best_error, error_results = (
        optimize_fixed_error(
            df_original,
            df_w,
            df_fix_base
        )
    )

    # --------------------------------------------------------
    # Apply best Error %
    # --------------------------------------------------------

    df, df_w_final = (
        calculate_effective_area(
            df_original,
            df_w,
            best_error
        )
    )

    # --------------------------------------------------------
    # Final forecast
    # --------------------------------------------------------

    df_fix = calculate_fixed_forecast(
        data["df_fix"],
        df_w_final
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    calculated_peak = (
        df_fix[
            "Total Power (CL1+CL2+…)"
        ].max()
    )

    actual_peak = (
        df_fix["Actual"].max()
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

    return {
        "df": df,
        "df_w": df_w_final,
        "df_fix": df_fix,
        "best_error": best_error,
        "error_results": error_results,
        "calculated_peak": calculated_peak,
        "actual_peak": actual_peak,
        "peak_error": peak_error,
        "peak_error_pct": peak_error_pct
    }


# ============================================================
# LOAD TRACKING BACKEND DATA
# ============================================================

def load_tracking_data(file_bytes):

    backend_list = []

    for cluster in [
        "C11",
        "C12",
        "C13",
        "C14",
        "C15"
    ]:

        df_backend = read_excel_sheet(
            file_bytes,
            f"Backend Cal {cluster}"
        )

        backend_list.append(
            df_backend
        )

    df_trac = read_excel_sheet(
        file_bytes,
        "Tracking",
        header=1
    )

    df_trac = clean_columns(
        df_trac
    )

    return {
        "backend_list": backend_list,
        "df_trac": df_trac
    }


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def run_tracking(
    data
):

    df_original = data["df"]
    df_w = data["df_w"]
    df_ghi = data["df_ghi"]
    df_fix = data["df_fix"]

    tracking_data = data["tracking"]

    backend_list = tracking_data[
        "backend_list"
    ]

    df_trac = tracking_data[
        "df_trac"
    ].copy()

    # --------------------------------------------------------
    # Cluster weights
    #
    # Exact logic from Jupyter:
    # df_w.iloc[:5, 1]
    # --------------------------------------------------------

    cl_weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    cl_weights = np.pad(
        cl_weights,
        (0, max(0, 5 - len(cl_weights))),
        constant_values=0
    )[:5]

    # --------------------------------------------------------
    # GHI matrix
    # --------------------------------------------------------

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15"
    ]

    n = min(
        len(df_ghi),
        len(df_fix),
        len(backend_list[0])
    )

    ghi_matrix = np.column_stack([
        pd.to_numeric(
            df_ghi[col].iloc[:n],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)

        for col in ghi_cols
    ])

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    blocks = pd.to_numeric(
        backend_list[0][
            "Block No."
        ].iloc[:n],
        errors="coerce"
    ).fillna(0).to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Actual
    # --------------------------------------------------------

    actual_full = pd.to_numeric(
        df_fix["Actual"].iloc[:n],
        errors="coerce"
    ).fillna(0).to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Exact mask from Jupyter
    # --------------------------------------------------------

    mask = actual_full != 0

    actual = actual_full[mask]

    if len(actual) == 0:

        raise ValueError(
            "Actual power contains no non-zero values."
        )

    actual_max = actual.max()
    actual_sum = actual.sum()

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    def objective(x):

        DHI = int(
            round(x[0])
        )

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

        # ----------------------------------------------------
        # Validate block positions
        # ----------------------------------------------------

        if not (
            GHI_Starting_Block
            < GHI_Max_Block
            < GHI_Ending_Block
        ):

            return 1e9

        # ----------------------------------------------------
        # Slopes
        #
        # EXACT JUPYTER FORMULA
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

        if (
            denominator_1 == 0
            or denominator_2 == 0
        ):

            return 1e9

        m1 = 90 / denominator_1

        m2 = 90 / denominator_2

        # ----------------------------------------------------
        # Zenith
        #
        # EXACT JUPYTER LOGIC
        # ----------------------------------------------------

        zenith = np.where(
            blocks <= GHI_Max_Block,

            np.minimum(
                89,
                m1
                * (
                    blocks
                    - GHI_Max_Block
                )
            ),

            np.minimum(
                89,
                m2
                * (
                    blocks
                    - GHI_Max_Block
                )
            )
        )

        # ----------------------------------------------------
        # Panel angle
        #
        # EXACT JUPYTER LOGIC
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Cosine
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
            * DHI
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
        # Prediction
        #
        # EXACT JUPYTER FORMULA
        # ----------------------------------------------------

        prediction_full = (
            dni @ cl_weights
        ) / 1_000_000

        # ----------------------------------------------------
        # Invalid
        # ----------------------------------------------------

        if (
            np.isnan(
                prediction_full
            ).any()

            or np.isinf(
                prediction_full
            ).any()
        ):

            return 1e9

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

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
        # Final score
        #
        # EXACT JUPYTER WEIGHTS
        # ----------------------------------------------------

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return score

    # ========================================================
    # PARAMETER BOUNDS
    # ========================================================

    bounds = [
        (0, 10),
        (10, 30),
        (65, 80),
        (47, 53),
        (10, 70),
        (10, 70)
    ]

    # ========================================================
    # OPTIMIZATION
    # ========================================================

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

    # ========================================================
    # BEST PARAMETERS
    # ========================================================

    best = np.round(
        result.x
    ).astype(int)

    DHI = best[0]

    GHI_Starting_Block = best[1]

    GHI_Ending_Block = best[2]

    GHI_Max_Block = best[3]

    Tracking_angle_lim_E = best[4]

    Tracking_angle_lim_W = best[5]

    # ========================================================
    # FINAL TRACKING ANGLES
    # ========================================================

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
            )
        ),

        np.minimum(
            89,
            m2
            * (
                blocks
                - GHI_Max_Block
            )
        )
    )

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

    # ========================================================
    # COS
    # ========================================================

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None
    )

    # ========================================================
    # FINAL DHI
    # ========================================================

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    # ========================================================
    # FINAL DNI
    # ========================================================

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    # ========================================================
    # FINAL FORECAST
    #
    # EXACT JUPYTER FORMULA
    # ========================================================

    forecast = (
        dni @ cl_weights
    ) / 1_000_000

    # ========================================================
    # SAVE TRACKING DATA
    # ========================================================

    df_trac = df_trac.iloc[
        :len(forecast)
    ].copy()

    df_trac["Zenith Angle"] = (
        zenith
    )

    df_trac["Panel Angle"] = (
        panel
    )

    df_trac["Fixed Power=I*Ƞ*A"] = (
        forecast
    )

    # ========================================================
    # TRACKING METRICS
    # ========================================================

    actual = actual_full[
        :len(forecast)
    ]

    calculated_peak = (
        np.max(forecast)
    )

    actual_peak = (
        np.max(actual)
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

    actual_sum = np.sum(
        actual
    )

    forecast_sum = np.sum(
        forecast
    )

    energy_error_pct = (
        abs(
            forecast_sum
            - actual_sum
        )
        / actual_sum
        * 100
        if actual_sum != 0
        else np.nan
    )

    mask_final = actual != 0

    if mask_final.any():

        block_error_pct = (
            np.mean(
                np.abs(
                    actual[mask_final]
                    - forecast[mask_final]
                )
            )
            / actual_peak
            * 100
        )

    else:

        block_error_pct = np.nan

    return {
        "df_trac": df_trac,
        "forecast": forecast,
        "zenith": zenith,
        "panel": panel,
        "DHI": DHI,
        "GHI_Starting_Block": GHI_Starting_Block,
        "GHI_Ending_Block": GHI_Ending_Block,
        "GHI_Max_Block": GHI_Max_Block,
        "Tracking_angle_lim_E": Tracking_angle_lim_E,
        "Tracking_angle_lim_W": Tracking_angle_lim_W,
        "optimization_score": result.fun,
        "calculated_peak": calculated_peak,
        "actual_peak": actual_peak,
        "peak_error": peak_error,
        "peak_error_pct": peak_error_pct,
        "energy_error_pct": energy_error_pct,
        "block_error_pct": block_error_pct
    }


# ============================================================
# PLOT
# ============================================================

def create_power_plot(
    forecast,
    actual,
    title
):

    n = min(
        len(forecast),
        len(actual)
    )

    forecast = np.asarray(
        forecast[:n],
        dtype=float
    )

    actual = np.asarray(
        actual[:n],
        dtype=float
    )

    blocks = np.arange(
        n
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=blocks,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=blocks,
            y=actual,
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
        hovermode="x unified"
    )

    return fig


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("Upload Excel File")

uploaded_file = st.file_uploader(
    "Upload your solar calculation Excel file",
    type=["xlsx", "xls"]
)


# ============================================================
# STOP UNTIL FILE IS UPLOADED
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload the Excel file to continue."
    )

    st.stop()


# ============================================================
# READ FILE BYTES
# ============================================================

file_bytes = uploaded_file.getvalue()


# ============================================================
# PLANT TYPE
# ============================================================

st.subheader("Plant Type")

plant_type = st.radio(
    "Select plant type",
    [
        "Fixed",
        "Tracking"
    ],
    horizontal=True
)


# ============================================================
# LOAD COMMON DATA
# ============================================================

try:

    with st.spinner(
        "Reading Excel file..."
    ):

        common_data = (
            load_common_data(
                file_bytes
            )
        )

except Exception as e:

    st.error(
        f"Error while reading Excel file: {e}"
    )

    st.stop()


# ============================================================
# SHOW BASIC INFORMATION
# ============================================================

st.subheader("Plant Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Plant Type",
        plant_type
    )

with col2:

    st.metric(
        "Latitude",
        f"{common_data['lat']:.4f}"
    )

with col3:

    st.metric(
        "GHI Blocks",
        len(common_data["df_ghi"])
    )


# ============================================================
# FIXED
# ============================================================

if plant_type == "Fixed":

    st.subheader(
        "Fixed Plant Loss Correction"
    )

    # --------------------------------------------------------
    # Run calculation
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Calculating Fixed plant..."
        ):

            fixed_result = run_fixed(
                common_data
            )

    except Exception as e:

        st.error(
            f"Fixed calculation failed: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Best Error %",
            f"{fixed_result['best_error']:.2f}%"
        )

    with col2:

        st.metric(
            "Calculated Peak",
            f"{fixed_result['calculated_peak']:.4f}"
        )

    with col3:

        st.metric(
            "Actual Peak",
            f"{fixed_result['actual_peak']:.4f}"
        )

    with col4:

        st.metric(
            "Peak Error %",
            f"{fixed_result['peak_error_pct']:.2f}%"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig = create_power_plot(
        fixed_result["df_fix"][
            "Total Power (CL1+CL2+…)"
        ].to_numpy(),

        fixed_result["df_fix"][
            "Actual"
        ].to_numpy(),

        "Fixed Plant Forecast vs Actual"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Error optimization table
    # --------------------------------------------------------

    with st.expander(
        "Error % Optimization Results"
    ):

        st.dataframe(
            fixed_result[
                "error_results"
            ],
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Effective area
    # --------------------------------------------------------

    with st.expander(
        "Cluster Effective Area"
    ):

        st.dataframe(
            fixed_result["df_w"],
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Final calculation
    # --------------------------------------------------------

    with st.expander(
        "Final Fixed Calculation"
    ):

        st.dataframe(
            fixed_result["df_fix"],
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        fixed_result["df"].to_excel(
            writer,
            sheet_name="Area & Efficiency",
            index=False
        )

        fixed_result["df_w"].to_excel(
            writer,
            sheet_name="Cluster Area",
            index=False
        )

        fixed_result["df_fix"].to_excel(
            writer,
            sheet_name="Fixed Result",
            index=False
        )

        fixed_result[
            "error_results"
        ].to_excel(
            writer,
            sheet_name="Error Optimization",
            index=False
        )

    output.seek(0)

    st.download_button(
        label="Download Fixed Result",
        data=output.getvalue(),
        file_name="Solar_Fixed_Loss_Correction.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# ============================================================
# TRACKING
# ============================================================

else:

    st.subheader(
        "Tracking Plant Loss Correction"
    )

    # --------------------------------------------------------
    # Load Tracking sheets
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Reading Tracking sheets..."
        ):

            tracking_data = (
                load_tracking_data(
                    file_bytes
                )
            )

            common_data["tracking"] = (
                tracking_data
            )

    except Exception as e:

        st.error(
            f"Tracking sheets could not be loaded: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Run Tracking
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Optimizing Tracking parameters..."
        ):

            tracking_result = run_tracking(
                common_data
            )

    except Exception as e:

        st.error(
            f"Tracking calculation failed: {e}"
        )

        st.stop()

    # ========================================================
    # PARAMETERS
    # ========================================================

    st.subheader(
        "Optimized Tracking Parameters"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "DHI %",
            tracking_result["DHI"]
        )

    with col2:

        st.metric(
            "GHI Starting Block",
            tracking_result[
                "GHI_Starting_Block"
            ]
        )

    with col3:

        st.metric(
            "GHI Ending Block",
            tracking_result[
                "GHI_Ending_Block"
            ]
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "GHI Max Block",
            tracking_result[
                "GHI_Max_Block"
            ]
        )

    with col5:

        st.metric(
            "Tracking East Limit",
            tracking_result[
                "Tracking_angle_lim_E"
            ]
        )

    with col6:

        st.metric(
            "Tracking West Limit",
            tracking_result[
                "Tracking_angle_lim_W"
            ]
        )

    # ========================================================
    # ACCURACY
    # ========================================================

    st.subheader(
        "Tracking Accuracy"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Calculated Peak",
            f"{tracking_result['calculated_peak']:.4f}"
        )

    with col2:

        st.metric(
            "Actual Peak",
            f"{tracking_result['actual_peak']:.4f}"
        )

    with col3:

        st.metric(
            "Peak Error %",
            f"{tracking_result['peak_error_pct']:.2f}%"
        )

    with col4:

        st.metric(
            "Energy Error %",
            f"{tracking_result['energy_error_pct']:.2f}%"
        )

    # ========================================================
    # FORECAST VS ACTUAL
    # ========================================================

    fig = create_power_plot(
        tracking_result["forecast"],
        common_data["df_fix"][
            "Actual"
        ].to_numpy(),
        "Tracking Forecast vs Actual"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # TRACKING ANGLES
    # ========================================================

    st.subheader(
        "Tracking Angles"
    )

    angle_df = pd.DataFrame({
        "Block": np.arange(
            len(
                tracking_result[
                    "forecast"
                ]
            )
        ),

        "Zenith Angle":
            tracking_result[
                "zenith"
            ],

        "Panel Angle":
            tracking_result[
                "panel"
            ]
    })

    st.dataframe(
        angle_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # TRACKING RESULT
    # ========================================================

    with st.expander(
        "Final Tracking Calculation"
    ):

        st.dataframe(
            tracking_result[
                "df_trac"
            ],
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # CLUSTER AREAS
    # ========================================================

    with st.expander(
        "Cluster Effective Areas"
    ):

        st.dataframe(
            common_data["df_w"],
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        common_data["df"].to_excel(
            writer,
            sheet_name="Area & Efficiency",
            index=False
        )

        common_data["df_w"].to_excel(
            writer,
            sheet_name="Cluster Area",
            index=False
        )

        tracking_result[
            "df_trac"
        ].to_excel(
            writer,
            sheet_name="Tracking Result",
            index=False
        )

        pd.DataFrame({
            "Parameter": [
                "DHI %",
                "GHI Starting Block",
                "GHI Ending Block",
                "GHI Max Block",
                "Tracking East Limit",
                "Tracking West Limit",
                "Optimization Score",
                "Calculated Peak",
                "Actual Peak",
                "Peak Error %",
                "Energy Error %",
                "Block Error %"
            ],

            "Value": [
                tracking_result["DHI"],
                tracking_result[
                    "GHI_Starting_Block"
                ],
                tracking_result[
                    "GHI_Ending_Block"
                ],
                tracking_result[
                    "GHI_Max_Block"
                ],
                tracking_result[
                    "Tracking_angle_lim_E"
                ],
                tracking_result[
                    "Tracking_angle_lim_W"
                ],
                tracking_result[
                    "optimization_score"
                ],
                tracking_result[
                    "calculated_peak"
                ],
                tracking_result[
                    "actual_peak"
                ],
                tracking_result[
                    "peak_error_pct"
                ],
                tracking_result[
                    "energy_error_pct"
                ],
                tracking_result[
                    "block_error_pct"
                ]
            ]
        }).to_excel(
            writer,
            sheet_name="Tracking Parameters",
            index=False
        )

    output.seek(0)

    st.download_button(
        label="Download Tracking Result",
        data=output.getvalue(),
        file_name="Solar_Tracking_Loss_Correction.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )
