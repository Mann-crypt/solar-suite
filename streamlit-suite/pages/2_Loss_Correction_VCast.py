# ============================================================
# STREAMLIT APP
# SOLAR FORECAST CORRECTION
# JUPYTER-CONSISTENT CALCULATION
# FIXED + TRACKING
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

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.20);
        padding: 12px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

GHI_COLUMNS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

POWER_COLUMNS = [
    "CL1_Fixed Power=I*Ƞ*A",
    "CL2_Fixed Power=I*Ƞ*A",
    "CL3_Fixed Power=I*Ƞ*A",
    "CL4_Fixed Power=I*Ƞ*A",
    "CL5_Fixed Power=I*Ƞ*A",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

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
    df = df.copy()

    if column not in df.columns:
        return df

    null_indices = df[df[column].isna()].index

    if len(null_indices) == 0:
        return df

    first_null_pos = df.index.get_loc(null_indices[0])

    return df.iloc[:first_null_pos].copy()


def numeric_series(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


# ============================================================
# READ AREA / EFFICIENCY
# EXACT JUPYTER LOGIC
# ============================================================

def read_area_efficiency(excel_bytes):

    df = pd.read_excel(
        excel_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12)
    )

    df = clean_columns(df)

    df = trim_at_first_null(df, "S.No.")

    # --------------------------------------------------------
    # Original Jupyter initialization
    # --------------------------------------------------------

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
    # Cluster table
    # Exact same columns as Jupyter
    # --------------------------------------------------------

    df_w = pd.read_excel(
        excel_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15]
    )

    df_w = trim_at_first_null(df_w, "Clusters")

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
# EXACT JUPYTER LOGIC
# ============================================================

def read_ghi(excel_bytes):

    df_ghi = pd.read_excel(
        excel_bytes,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5]
    )

    df_ghi = df_ghi.fillna(0)

    for col in GHI_COLUMNS:
        if col not in df_ghi.columns:
            raise ValueError(
                f"Missing required column '{col}' in Result sheet."
            )

        df_ghi[col] = pd.to_numeric(
            df_ghi[col],
            errors="coerce"
        ).fillna(0)

    return df_ghi


# ============================================================
# READ FIXED-C11
# EXACT JUPYTER LOGIC
# ============================================================

def read_fixed_c11(excel_bytes):

    df_fix = pd.read_excel(
        excel_bytes,
        sheet_name="Fixed-C11",
        header=1
    )

    df_fix = clean_columns(df_fix)

    df_fix = trim_at_first_null(df_fix, "Date")

    if "Actual" not in df_fix.columns:
        raise ValueError(
            "Column 'Actual' was not found in Fixed-C11."
        )

    df_fix["Actual"] = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce"
    ).fillna(0)

    return df_fix


# ============================================================
# READ LATITUDE
# EXACT JUPYTER LOGIC
# ============================================================

def read_latitude(excel_bytes):

    df_st = pd.read_excel(
        excel_bytes,
        sheet_name="Forecast Config",
        header=8
    )

    lat = float(df_st.loc[0, "Lat"])

    return lat


# ============================================================
# READ TILT
# EXACT JUPYTER LOGIC
# ============================================================

def read_tilt(excel_bytes):

    df_tilt = pd.read_excel(
        excel_bytes,
        sheet_name="Config Tilt Angle",
        header=7
    )

    df_tilt.columns = df_tilt.columns.astype(str).str.strip()

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
# PREPARE FIXED SOLAR GEOMETRY
# EXACT JUPYTER LOGIC
# ============================================================

def prepare_fixed_geometry(
    df_fix,
    df_ghi,
    lat,
    month_lookup
):

    df_fix = df_fix.copy()

    # --------------------------------------------------------
    # EXACT JUPYTER BEHAVIOR
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

    df_fix["Elevation angle a"] = (
        90
        - lat
        + df_fix["Declination Angle ∆"]
    )

    df_fix["Tilt Angle b"] = (
        df_fix["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

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

    # --------------------------------------------------------
    # C11
    # --------------------------------------------------------

    df_fix["GHI*sin(a)"] = (
        df_ghi["GHI C11"].to_numpy()
        * df_fix["Sin(a)"].to_numpy()
    )

    df_fix["GHI*sin(a+b)"] = (
        df_ghi["GHI C11"].to_numpy()
        * df_fix["SIN(a+b)"].to_numpy()
    )

    df_fix["POA fixed"] = (
        df_fix["GHI*sin(a+b)"]
        / df_fix["Sin(a)"]
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

    df_fix["POA Fixed-C12"] = (
        df_fix["GHI*sin(a+b)-CL2"]
        / df_fix["Sin(a)"]
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

    df_fix["POA Fixed-C13"] = (
        df_fix["GHI*sin(a+b)-CL3"]
        / df_fix["Sin(a)"]
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

    df_fix["POA Fixed-C14"] = (
        df_fix["GHI*sin(a+b)-CL4"]
        / df_fix["Sin(a)"]
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

    df_fix["POA Fixed-C15"] = (
        df_fix["GHI*sin(a+b)-CL5"]
        / df_fix["Sin(a)"]
    )

    return df_fix


# ============================================================
# FIXED CALCULATION
# EXACT JUPYTER LOGIC
# ============================================================

def calculate_fixed(
    df_original,
    df_w_original,
    df_fix_original,
    error_start=0.0,
    error_end=10.0,
    error_step=0.1
):

    df = df_original.copy()
    df_w = df_w_original.copy()
    df_fix = df_fix_original.copy()

    actual_peak = df_fix["Actual"].max()

    results = []

    # --------------------------------------------------------
    # EXACT ERROR LOOP
    # --------------------------------------------------------

    errors = np.arange(
        error_start,
        error_end + error_step * 0.1,
        error_step
    )

    for error in errors:

        # ----------------------------------------------------
        # 1. Apply Error %
        # ----------------------------------------------------

        df["Error %"] = error

        df["Net Efficiency (%)"] = (
            df["Standard PV Efficiency (%)"]
            - df["Error %"]
        )

        # ----------------------------------------------------
        # 2. Calculate Effective Area
        # ----------------------------------------------------

        df["Total area (m2)"] = (
            df["No of Module"]
            * df["Area of 1 Module (m2)"]
        )

        df["Eff Area"] = (
            df["Net Efficiency (%)"]
            * df["Total area (m2)"]
            / 100
        )

        # ----------------------------------------------------
        # 3. Cluster Effective Areas
        # ----------------------------------------------------

        cluster_sums = (
            df.groupby("Clusters")["Eff Area"]
            .sum()
        )

        df_w["Eff Area(m2)"] = (
            df_w["Clusters"]
            .map(cluster_sums)
            .fillna(0)
        )

        # ----------------------------------------------------
        # 4. Cluster Power
        # ----------------------------------------------------

        df_fix[
            "CL1_Fixed Power=I*Ƞ*A"
        ] = (
            df_fix["POA fixed"]
            * df_w.iloc[0]["Eff Area(m2)"]
        ) / 1000000

        df_fix[
            "CL2_Fixed Power=I*Ƞ*A"
        ] = (
            df_fix["POA Fixed-C12"]
            * df_w.iloc[1]["Eff Area(m2)"]
        ) / 1000000

        df_fix[
            "CL3_Fixed Power=I*Ƞ*A"
        ] = (
            df_fix["POA Fixed-C13"]
            * df_w.iloc[2]["Eff Area(m2)"]
        ) / 1000000

        df_fix[
            "CL4_Fixed Power=I*Ƞ*A"
        ] = (
            df_fix["POA Fixed-C14"]
            * df_w.iloc[3]["Eff Area(m2)"]
        ) / 1000000

        df_fix[
            "CL5_Fixed Power=I*Ƞ*A"
        ] = (
            df_fix["POA Fixed-C15"]
            * df_w.iloc[4]["Eff Area(m2)"]
        ) / 1000000

        # ----------------------------------------------------
        # 5. Total Power
        # ----------------------------------------------------

        df_fix[
            "Total Power (CL1+CL2+…)"
        ] = (
            df_fix[
                "CL1_Fixed Power=I*Ƞ*A"
            ]
            + df_fix[
                "CL2_Fixed Power=I*Ƞ*A"
            ]
            + df_fix[
                "CL3_Fixed Power=I*Ƞ*A"
            ]
            + df_fix[
                "CL4_Fixed Power=I*Ƞ*A"
            ]
            + df_fix[
                "CL5_Fixed Power=I*Ƞ*A"
            ]
        )

        # ----------------------------------------------------
        # 6. Peak Error
        # ----------------------------------------------------

        calculated_peak = (
            df_fix[
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

        results.append(
            {
                "Error %": error,
                "Calculated Peak": calculated_peak,
                "Actual Peak": actual_peak,
                "Peak Error": peak_error,
                "Peak Error %": peak_error_pct,
            }
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # THIS WAS MISSING FROM YOUR JUPYTER CODE
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    best_error = (
        results_df.loc[
            results_df["Peak Error %"].idxmin(),
            "Error %"
        ]
    )

    # --------------------------------------------------------
    # FINAL RECALCULATION
    # --------------------------------------------------------

    df["Error %"] = best_error

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

    # --------------------------------------------------------
    # FINAL POWER
    # --------------------------------------------------------

    df_fix[
        "CL1_Fixed Power=I*Ƞ*A"
    ] = (
        df_fix["POA fixed"]
        * df_w.iloc[0]["Eff Area(m2)"]
    ) / 1000000

    df_fix[
        "CL2_Fixed Power=I*Ƞ*A"
    ] = (
        df_fix["POA Fixed-C12"]
        * df_w.iloc[1]["Eff Area(m2)"]
    ) / 1000000

    df_fix[
        "CL3_Fixed Power=I*Ƞ*A"
    ] = (
        df_fix["POA Fixed-C13"]
        * df_w.iloc[2]["Eff Area(m2)"]
    ) / 1000000

    df_fix[
        "CL4_Fixed Power=I*Ƞ*A"
    ] = (
        df_fix["POA Fixed-C14"]
        * df_w.iloc[3]["Eff Area(m2)"]
    ) / 1000000

    df_fix[
        "CL5_Fixed Power=I*Ƞ*A"
    ] = (
        df_fix["POA Fixed-C15"]
        * df_w.iloc[4]["Eff Area(m2)"]
    ) / 1000000

    df_fix[
        "Total Power (CL1+CL2+…)"
    ] = df_fix[POWER_COLUMNS].sum(axis=1)

    final_calculated_peak = (
        df_fix[
            "Total Power (CL1+CL2+…)"
        ].max()
    )

    final_actual_peak = (
        df_fix["Actual"].max()
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
        "df": df,
        "df_w": df_w,
        "df_fix": df_fix,
        "results_df": results_df,
        "best_error": float(best_error),
        "calculated_peak": float(final_calculated_peak),
        "actual_peak": float(final_actual_peak),
        "peak_error": float(final_peak_error),
        "peak_error_pct": float(final_peak_error_pct),
    }


# ============================================================
# READ TRACKING BACKEND
# EXACT JUPYTER LOGIC
# ============================================================

def read_tracking_backend(excel_bytes):

    backend_list = []

    for cluster in ["C11", "C12", "C13", "C14", "C15"]:

        backend_list.append(
            pd.read_excel(
                excel_bytes,
                sheet_name=f"Backend Cal {cluster}"
            )
        )

    df_trac = pd.read_excel(
        excel_bytes,
        sheet_name="Tracking",
        header=1
    )

    return backend_list, df_trac


# ============================================================
# TRACKING PREPARATION
# ============================================================

def prepare_tracking_data(
    df_w,
    df_ghi,
    df_fix,
    backend_list
):

    # --------------------------------------------------------
    # EXACT JUPYTER WEIGHTS
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
    # EXACT GHI MATRIX
    # --------------------------------------------------------

    ghi_matrix = np.column_stack(
        [
            df_ghi[col]
            .to_numpy(dtype=float)
            for col in GHI_COLUMNS
        ]
    )

    # --------------------------------------------------------
    # EXACT BLOCK SOURCE
    # --------------------------------------------------------

    blocks = (
        backend_list[0]["Block No."]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # EXACT ACTUAL SOURCE
    #
    # IMPORTANT:
    # This is df_fix["Actual"], NOT Tracking["Actual"]
    # --------------------------------------------------------

    actual_full = (
        df_fix["Actual"]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Align lengths exactly
    # --------------------------------------------------------

    n = min(
        len(blocks),
        len(ghi_matrix),
        len(actual_full)
    )

    blocks = blocks[:n]
    ghi_matrix = ghi_matrix[:n]
    actual_full = actual_full[:n]

    # --------------------------------------------------------
    # EXACT MASK
    # --------------------------------------------------------

    mask = actual_full != 0

    actual = actual_full[mask]

    if len(actual) == 0:
        raise ValueError(
            "No non-zero Actual values found for Tracking. "
            "Tracking uses Actual from Fixed-C11, exactly like "
            "the Jupyter calculation."
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
        actual_sum,
    )


# ============================================================
# TRACKING FORECAST FUNCTION
# EXACT JUPYTER MATHEMATICS
# ============================================================

def tracking_forecast(
    DHI,
    GHI_Starting_Block,
    GHI_Ending_Block,
    GHI_Max_Block,
    Tracking_angle_lim_E,
    Tracking_angle_lim_W,
    blocks,
    ghi_matrix,
    cl_weights
):

    DHI = int(round(DHI))

    GHI_Starting_Block = int(
        round(GHI_Starting_Block)
    )

    GHI_Ending_Block = int(
        round(GHI_Ending_Block)
    )

    GHI_Max_Block = int(
        round(GHI_Max_Block)
    )

    Tracking_angle_lim_E = int(
        round(Tracking_angle_lim_E)
    )

    Tracking_angle_lim_W = int(
        round(Tracking_angle_lim_W)
    )

    # --------------------------------------------------------
    # EXACT VALIDATION
    # --------------------------------------------------------

    if not (
        GHI_Starting_Block
        < GHI_Max_Block
        < GHI_Ending_Block
    ):
        raise ValueError(
            "GHI Starting Block < GHI Max Block < "
            "GHI Ending Block is required."
        )

    # --------------------------------------------------------
    # EXACT SLOPES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EXACT ZENITH
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
    # EXACT PANEL ANGLE
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
    # EXACT COSINE
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
    # EXACT DHI
    # --------------------------------------------------------

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    # --------------------------------------------------------
    # EXACT DNI
    # --------------------------------------------------------

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    # --------------------------------------------------------
    # EXACT POWER
    # --------------------------------------------------------

    forecast = (
        dni @ cl_weights
    ) / 1_000_000

    if (
        np.isnan(forecast).any()
        or np.isinf(forecast).any()
    ):
        raise ValueError(
            "Tracking calculation produced invalid values."
        )

    return (
        forecast,
        zenith,
        panel,
        cos_alpha,
        dni
    )


# ============================================================
# TRACKING OBJECTIVE
# EXACT JUPYTER OBJECTIVE
# ============================================================

def make_tracking_objective(
    blocks,
    ghi_matrix,
    cl_weights,
    actual_full,
    mask,
    actual,
    actual_max,
    actual_sum
):

    def objective(x):

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

        # ----------------------------------------------------
        # EXACT VALIDATION
        # ----------------------------------------------------

        if not (
            GHI_Starting_Block
            < GHI_Max_Block
            < GHI_Ending_Block
        ):
            return 1e9

        # ----------------------------------------------------
        # EXACT SLOPES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # EXACT ZENITH
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # EXACT PANEL
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
        # EXACT COS
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
        # EXACT DHI
        # ----------------------------------------------------

        dhi = (
            ghi_matrix
            * DHI
            / 100
        )

        # ----------------------------------------------------
        # EXACT DNI
        # ----------------------------------------------------

        dni = (
            ghi_matrix
            - dhi
        ) / cos_alpha[:, None]

        # ----------------------------------------------------
        # EXACT PREDICTION
        # ----------------------------------------------------

        prediction_full = (
            dni @ cl_weights
        ) / 1_000_000

        # ----------------------------------------------------
        # INVALID
        # ----------------------------------------------------

        if (
            np.isnan(prediction_full).any()
            or np.isinf(prediction_full).any()
        ):
            return 1e9

        # ----------------------------------------------------
        # EXACT MASK
        # ----------------------------------------------------

        prediction = (
            prediction_full[mask]
        )

        if len(prediction) == 0:
            return 1e9

        # ----------------------------------------------------
        # EXACT BLOCK ERROR
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
        # EXACT PEAK ERROR
        # ----------------------------------------------------

        peak_error = (
            abs(
                actual_max
                - prediction.max()
            )
            / actual_max
        )

        # ----------------------------------------------------
        # EXACT ENERGY ERROR
        # ----------------------------------------------------

        energy_error = (
            abs(
                actual_sum
                - prediction.sum()
            )
            / actual_sum
        )

        # ----------------------------------------------------
        # EXACT SCORE
        # ----------------------------------------------------

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return score

    return objective


# ============================================================
# RUN TRACKING OPTIMIZATION
# EXACT JUPYTER SETTINGS
# ============================================================

def optimize_tracking(
    blocks,
    ghi_matrix,
    cl_weights,
    actual_full,
    mask,
    actual,
    actual_max,
    actual_sum,
    bounds,
    maxiter=40,
    popsize=15,
    tol=0.001,
    mutation_low=0.5,
    mutation_high=1.0,
    recombination=0.7,
    seed=42,
    polish=True
):

    objective = make_tracking_objective(
        blocks,
        ghi_matrix,
        cl_weights,
        actual_full,
        mask,
        actual,
        actual_max,
        actual_sum
    )

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=(
            mutation_low,
            mutation_high
        ),
        recombination=recombination,
        seed=seed,
        polish=polish,
        workers=1
    )

    best = np.round(
        result.x
    ).astype(int)

    return result, best


# ============================================================
# FINAL TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    best,
    blocks,
    ghi_matrix,
    cl_weights,
    actual_full,
    mask,
    actual_max,
    actual_sum
):

    (
        forecast,
        zenith,
        panel,
        cos_alpha,
        dni
    ) = tracking_forecast(
        best[0],
        best[1],
        best[2],
        best[3],
        best[4],
        best[5],
        blocks,
        ghi_matrix,
        cl_weights
    )

    prediction = forecast[mask]

    block_error = (
        np.mean(
            np.abs(
                actual_full[mask]
                - prediction
            )
        )
        / actual_max
    )

    peak_error = (
        abs(
            actual_max
            - forecast.max()
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
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return {
        "forecast": forecast,
        "zenith": zenith,
        "panel": panel,
        "dni": dni,
        "block_error": block_error,
        "peak_error": peak_error,
        "energy_error": energy_error,
        "score": score,
        "actual": actual_full,
    }


# ============================================================
# PLOTLY GRAPH
# ============================================================

def create_plot(
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
        hovermode="x unified",
        template="plotly_white",
        height=500,
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
            max(95, n - 1)
        ]
    )

    return fig


# ============================================================
# SESSION STATE
# ============================================================

if "calculated" not in st.session_state:
    st.session_state.calculated = False

if "auto_result" not in st.session_state:
    st.session_state.auto_result = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# ============================================================
# HEADER
# ============================================================

st.title("Solar Forecast Correction")


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"],
    key="solar_excel"
)


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.segmented_control(
    "Plant Type",
    options=["Fixed", "Tracking"],
    default="Fixed"
)


# ============================================================
# NOTHING ELSE UNTIL FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload the Excel workbook to start automatic calculation."
    )

    st.stop()


# ============================================================
# FILE BYTES
# ============================================================

file_bytes = uploaded_file.getvalue()


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

file_changed = (
    st.session_state.uploaded_file_name
    != uploaded_file.name
)


if (
    not st.session_state.calculated
    or file_changed
    or st.session_state.get(
        "last_plant_type"
    ) != plant_type
):

    try:

        with st.spinner(
            f"Running automatic {plant_type} calculation..."
        ):

            # ------------------------------------------------
            # COMMON DATA
            # ------------------------------------------------

            df, df_w = read_area_efficiency(
                io.BytesIO(file_bytes)
            )

            df_ghi = read_ghi(
                io.BytesIO(file_bytes)
            )

            df_fix = read_fixed_c11(
                io.BytesIO(file_bytes)
            )

            # ------------------------------------------------
            # INITIAL ERROR VALUE
            #
            # Same assumption as original workbook flow
            # ------------------------------------------------

            if "Error %" not in df.columns:
                df["Error %"] = 0.0

            df["Error %"] = pd.to_numeric(
                df["Error %"],
                errors="coerce"
            ).fillna(0)

            # ------------------------------------------------
            # FIXED
            # ------------------------------------------------

            if plant_type == "Fixed":

                lat = read_latitude(
                    io.BytesIO(file_bytes)
                )

                month_lookup = read_tilt(
                    io.BytesIO(file_bytes)
                )

                df_fix_geometry = (
                    prepare_fixed_geometry(
                        df_fix,
                        df_ghi,
                        lat,
                        month_lookup
                    )
                )

                auto_result = calculate_fixed(
                    df,
                    df_w,
                    df_fix_geometry
                )

            # ------------------------------------------------
            # TRACKING
            # ------------------------------------------------

            else:

                backend_list, df_trac = (
                    read_tracking_backend(
                        io.BytesIO(file_bytes)
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
                    actual_sum,
                ) = prepare_tracking_data(
                    df_w,
                    df_ghi,
                    df_fix,
                    backend_list
                )

                bounds = [
                    (0, 10),
                    (10, 30),
                    (65, 80),
                    (47, 53),
                    (10, 70),
                    (10, 70),
                ]

                result, best = optimize_tracking(
                    blocks,
                    ghi_matrix,
                    cl_weights,
                    actual_full,
                    mask,
                    actual,
                    actual_max,
                    actual_sum,
                    bounds=bounds,
                    maxiter=40,
                    popsize=15,
                    tol=0.001,
                    mutation_low=0.5,
                    mutation_high=1.0,
                    recombination=0.7,
                    seed=42,
                    polish=True
                )

                tracking_result = (
                    calculate_tracking(
                        best,
                        blocks,
                        ghi_matrix,
                        cl_weights,
                        actual_full,
                        mask,
                        actual_max,
                        actual_sum
                    )
                )

                auto_result = {
                    "best": best,
                    "optimizer_result": result,
                    "tracking": tracking_result,
                    "blocks": blocks,
                    "actual_full": actual_full,
                    "ghi_matrix": ghi_matrix,
                    "cl_weights": cl_weights,
                    "df_trac": df_trac,
                }

        st.session_state.auto_result = (
            auto_result
        )

        st.session_state.calculated = True

        st.session_state.uploaded_file_name = (
            uploaded_file.name
        )

        st.session_state.last_plant_type = (
            plant_type
        )

    except Exception as e:

        st.session_state.calculated = False
        st.session_state.auto_result = None

        st.error(
            f"{plant_type} calculation failed: {e}"
        )

        st.stop()


# ============================================================
# GET AUTOMATIC RESULT
# ============================================================

auto_result = (
    st.session_state.auto_result
)


# ============================================================
# FIXED UI
# ============================================================

if plant_type == "Fixed":

    best_error_auto = (
        auto_result["best_error"]
    )

    st.subheader(
        "Fixed Parameters"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        error_start = st.number_input(
            "Error % Start",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            format="%.2f"
        )

    with col2:

        error_end = st.number_input(
            "Error % End",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1,
            format="%.2f"
        )

    with col3:

        error_step = st.number_input(
            "Error % Step",
            min_value=0.01,
            max_value=10.0,
            value=0.1,
            step=0.01,
            format="%.2f"
        )

    # --------------------------------------------------------
    # Automatic result
    # --------------------------------------------------------

    st.subheader(
        "Automatic Result"
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Best Error %",
            f"{best_error_auto:.2f}%"
        )

    with m2:
        st.metric(
            "Calculated Peak",
            f"{auto_result['calculated_peak']:.4f}"
        )

    with m3:
        st.metric(
            "Actual Peak",
            f"{auto_result['actual_peak']:.4f}"
        )

    with m4:
        st.metric(
            "Peak Error %",
            f"{auto_result['peak_error_pct']:.4f}%"
        )

    # --------------------------------------------------------
    # Editable final Error %
    # --------------------------------------------------------

    st.subheader(
        "Editable Final Parameter"
    )

    selected_error = st.number_input(
        "Error %",
        min_value=0.0,
        max_value=100.0,
        value=float(best_error_auto),
        step=0.1,
        format="%.2f"
    )

    # --------------------------------------------------------
    # Recalculate using user parameter
    # --------------------------------------------------------

    if st.button(
        "Recalculate Fixed",
        type="primary"
    ):

        try:

            with st.spinner(
                "Recalculating..."
            ):

                df = auto_result["df"].copy()
                df_w = auto_result["df_w"].copy()
                df_fix = auto_result["df_fix"].copy()

                df["Error %"] = (
                    selected_error
                )

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
                    df.groupby(
                        "Clusters"
                    )["Eff Area"].sum()
                )

                df_w["Eff Area(m2)"] = (
                    df_w["Clusters"]
                    .map(cluster_sums)
                    .fillna(0)
                )

                df_fix[
                    "CL1_Fixed Power=I*Ƞ*A"
                ] = (
                    df_fix["POA fixed"]
                    * df_w.iloc[0]["Eff Area(m2)"]
                ) / 1000000

                df_fix[
                    "CL2_Fixed Power=I*Ƞ*A"
                ] = (
                    df_fix["POA Fixed-C12"]
                    * df_w.iloc[1]["Eff Area(m2)"]
                ) / 1000000

                df_fix[
                    "CL3_Fixed Power=I*Ƞ*A"
                ] = (
                    df_fix["POA Fixed-C13"]
                    * df_w.iloc[2]["Eff Area(m2)"]
                ) / 1000000

                df_fix[
                    "CL4_Fixed Power=I*Ƞ*A"
                ] = (
                    df_fix["POA Fixed-C14"]
                    * df_w.iloc[3]["Eff Area(m2)"]
                ) / 1000000

                df_fix[
                    "CL5_Fixed Power=I*Ƞ*A"
                ] = (
                    df_fix["POA Fixed-C15"]
                    * df_w.iloc[4]["Eff Area(m2)"]
                ) / 1000000

                df_fix[
                    "Total Power (CL1+CL2+…)"
                ] = df_fix[
                    POWER_COLUMNS
                ].sum(axis=1)

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

            st.subheader(
                "Final Result"
            )

            a, b, c, d = st.columns(4)

            with a:
                st.metric(
                    "Error %",
                    f"{selected_error:.2f}%"
                )

            with b:
                st.metric(
                    "Forecast Peak",
                    f"{calculated_peak:.4f}"
                )

            with c:
                st.metric(
                    "Actual Peak",
                    f"{actual_peak:.4f}"
                )

            with d:
                st.metric(
                    "Peak Error %",
                    f"{peak_error_pct:.4f}%"
                )

            fig = create_plot(
                df_fix[
                    "Total Power (CL1+CL2+…)"
                ].to_numpy(),
                df_fix[
                    "Actual"
                ].to_numpy(),
                "Fixed Forecast vs Actual"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Fixed recalculation failed: {e}"
            )


# ============================================================
# TRACKING UI
# ============================================================

else:

    best = (
        auto_result["best"]
    )

    blocks = (
        auto_result["blocks"]
    )

    actual_full = (
        auto_result["actual_full"]
    )

    ghi_matrix = (
        auto_result["ghi_matrix"]
    )

    cl_weights = (
        auto_result["cl_weights"]
    )

    # --------------------------------------------------------
    # Automatic optimizer result
    # --------------------------------------------------------

    st.subheader(
        "Automatic Tracking Optimization"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        DHI = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(best[0]),
            step=1
        )

        GHI_Starting_Block = st.number_input(
            "GHI Starting Block",
            min_value=0,
            max_value=95,
            value=int(best[1]),
            step=1
        )

    with col2:

        GHI_Ending_Block = st.number_input(
            "GHI Ending Block",
            min_value=0,
            max_value=95,
            value=int(best[2]),
            step=1
        )

        GHI_Max_Block = st.number_input(
            "GHI Max Block",
            min_value=0,
            max_value=95,
            value=int(best[3]),
            step=1
        )

    with col3:

        Tracking_angle_lim_E = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=180,
            value=int(best[4]),
            step=1
        )

        Tracking_angle_lim_W = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=180,
            value=int(best[5]),
            step=1
        )

    # --------------------------------------------------------
    # Optimizer controls
    # --------------------------------------------------------

    with st.expander(
        "Optimization Settings"
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            maxiter = st.number_input(
                "Max Iterations",
                min_value=1,
                max_value=1000,
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
                step=0.0001,
                format="%.6f"
            )

        c4, c5, c6 = st.columns(3)

        with c4:

            mutation_low = st.number_input(
                "Mutation Minimum",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1
            )

        with c5:

            mutation_high = st.number_input(
                "Mutation Maximum",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1
            )

        with c6:

            recombination = st.number_input(
                "Recombination",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1
            )

        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1
        )

        polish = st.checkbox(
            "Polish",
            value=True
        )

    # --------------------------------------------------------
    # Automatic result metrics
    # --------------------------------------------------------

    auto_tracking = (
        auto_result["tracking"]
    )

    auto_score = (
        auto_tracking["score"]
    )

    auto_peak_error = (
        auto_tracking["peak_error"]
        * 100
    )

    auto_energy_error = (
        auto_tracking["energy_error"]
        * 100
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Optimization Score",
            f"{auto_score:.6f}"
        )

    with b:

        st.metric(
            "Peak Error %",
            f"{auto_peak_error:.4f}%"
        )

    with c:

        st.metric(
            "Energy Error %",
            f"{auto_energy_error:.4f}%"
        )

    with d:

        st.metric(
            "DHI",
            f"{int(best[0])}%"
        )

    # --------------------------------------------------------
    # Recalculate with editable parameters
    # --------------------------------------------------------

    if st.button(
        "Recalculate Tracking",
        type="primary"
    ):

        try:

            if not (
                GHI_Starting_Block
                < GHI_Max_Block
                < GHI_Ending_Block
            ):

                st.error(
                    "GHI Starting Block must be less than "
                    "GHI Max Block, and GHI Max Block must "
                    "be less than GHI Ending Block."
                )

                st.stop()

            edited_best = np.array(
                [
                    DHI,
                    GHI_Starting_Block,
                    GHI_Ending_Block,
                    GHI_Max_Block,
                    Tracking_angle_lim_E,
                    Tracking_angle_lim_W,
                ],
                dtype=int
            )

            edited_result = calculate_tracking(
                edited_best,
                blocks,
                ghi_matrix,
                cl_weights,
                actual_full,
                actual_full != 0,
                actual_full[
                    actual_full != 0
                ].max(),
                actual_full[
                    actual_full != 0
                ].sum()
            )

            forecast = (
                edited_result["forecast"]
            )

            st.subheader(
                "Final Tracking Result"
            )

            m1, m2, m3 = st.columns(3)

            with m1:

                st.metric(
                    "Optimization Score",
                    f"{edited_result['score']:.6f}"
                )

            with m2:

                st.metric(
                    "Peak Error %",
                    f"{edited_result['peak_error'] * 100:.4f}%"
                )

            with m3:

                st.metric(
                    "Energy Error %",
                    f"{edited_result['energy_error'] * 100:.4f}%"
                )

            fig = create_plot(
                forecast,
                actual_full,
                "Tracking Forecast vs Actual"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Tracking recalculation failed: {e}"
            )
