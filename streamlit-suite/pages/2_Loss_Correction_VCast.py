# ============================================================
# STREAMLIT APP
# SOLAR LOSS CORRECTION
# FIXED / TRACKING
# JUPYTER CALCULATION PRESERVED
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

st.title("Solar Loss Correction")
st.write("Upload the Excel file first, then select the plant type.")


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xlsm"]
)

if uploaded_file is None:
    st.info("Please upload the Excel file to continue.")
    st.stop()


# ============================================================
# READ UPLOADED FILE INTO MEMORY
# ============================================================

file_bytes = uploaded_file.getvalue()


def read_excel_sheet(sheet_name, **kwargs):
    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
        **kwargs
    )


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.segmented_control(
    "Plant Type",
    options=["Fixed", "Tracking"],
    default="Fixed"
)


if plant_type is None:
    st.stop()


# ============================================================
# COMMON DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_common_data(file_data):

    def read_sheet(sheet_name, **kwargs):
        return pd.read_excel(
            io.BytesIO(file_data),
            sheet_name=sheet_name,
            **kwargs
        )

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    df = read_sheet(
        "Area & Efficiency",
        header=[1],
        usecols=range(12)
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    if "S.No." in df.columns:

        null_indices = df[df["S.No."].isna()].index

        if len(null_indices) > 0:
            first_null_pos = df.index.get_loc(null_indices[0])
            df = df.iloc[:first_null_pos]

    # --------------------------------------------------------
    # AREA & EFFICIENCY CLUSTER TABLE
    # --------------------------------------------------------

    df_w = read_sheet(
        "Area & Efficiency",
        header=1,
        usecols=[14, 15]
    )

    df_w.columns = df_w.columns.astype(str).str.strip()

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

    df_st = read_sheet(
        "Forecast Config",
        header=[8]
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

    df_tilt = read_sheet(
        "Config Tilt Angle",
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

    month_lookup = (
        df_tilt
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # RESULT / GHI
    # --------------------------------------------------------

    df_ghi = read_sheet(
        "Result",
        usecols=[0, 1, 2, 3, 4, 5]
    )

    df_ghi = df_ghi.fillna(0)

    # --------------------------------------------------------
    # FIXED-C11
    # --------------------------------------------------------

    df_fix = read_sheet(
        "Fixed-C11",
        header=[1]
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

            df_fix = df_fix.iloc[:first_null_pos]

    # --------------------------------------------------------
    # IMPORTANT
    # DO NOT CHANGE THE LENGTH OF THE DATA
    # --------------------------------------------------------

    n = min(
        len(df_fix),
        len(df_ghi)
    )

    df_fix = df_fix.iloc[:n].copy()
    df_ghi = df_ghi.iloc[:n].copy()

    return (
        df,
        df_w,
        df_ghi,
        df_fix,
        lat,
        month_lookup
    )


(
    df_base,
    df_w_base,
    df_ghi,
    df_fix_base,
    lat,
    month_lookup
) = load_common_data(file_bytes)


# ============================================================
# COMMON GHI COLUMNS
# ============================================================

ghi_cols = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15"
]


# ============================================================
# PREPARE AREA DATA
# ============================================================

df_area = df_base.copy()

df_area["Standard PV Efficiency (%)"] = pd.to_numeric(
    df_area["Standard PV Efficiency (%)"],
    errors="coerce"
)

df_area["Error %"] = pd.to_numeric(
    df_area["Error %"],
    errors="coerce"
).fillna(0)

df_area["No of Module"] = pd.to_numeric(
    df_area["No of Module"],
    errors="coerce"
)

df_area["Area of 1 Module (m2)"] = pd.to_numeric(
    df_area["Area of 1 Module (m2)"],
    errors="coerce"
)

df_area["Total area (m2)"] = (
    df_area["No of Module"] *
    df_area["Area of 1 Module (m2)"]
)


# ============================================================
# FUNCTION: PREPARE SOLAR ANGLES
# ============================================================

def prepare_solar_angles(
    df_fix,
    df_ghi,
    lat,
    month_lookup
):

    df_fix = df_fix.copy()

    # --------------------------------------------------------
    # Same date logic as Jupyter
    # --------------------------------------------------------

    today = pd.Timestamp.today()

    df_fix["Date"] = today

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

    day_number = (
        df_fix["Date"] - first_date
    ).dt.days + 1

    df_fix["Declination Angle ∆"] = (
        23.45 *
        np.sin(
            np.radians(
                360 *
                (
                    284 +
                    day_number
                ) /
                365
            )
        )
    )

    # --------------------------------------------------------
    # ELEVATION
    # --------------------------------------------------------

    df_fix["Elevation angle a"] = (
        90 -
        lat +
        df_fix["Declination Angle ∆"]
    )

    # --------------------------------------------------------
    # TILT
    # --------------------------------------------------------

    df_fix["Tilt Angle b"] = (
        df_fix["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

    df_fix["a+b"] = (
        df_fix["Elevation angle a"] +
        df_fix["Tilt Angle b"]
    )

    # --------------------------------------------------------
    # SIN
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
    # CLUSTER POA
    # --------------------------------------------------------

    for i, col in enumerate(ghi_cols):

        suffix = f"CL{i + 1}"

        ghi = pd.to_numeric(
            df_ghi[col],
            errors="coerce"
        ).fillna(0).to_numpy()

        df_fix[
            f"GHI*sin(a)-{suffix}"
        ] = (
            ghi *
            df_fix["Sin(a)"].to_numpy()
        )

        df_fix[
            f"GHI*sin(a+b)-{suffix}"
        ] = (
            ghi *
            df_fix["SIN(a+b)"].to_numpy()
        )

        df_fix[
            f"POA Fixed-{suffix}"
        ] = (
            df_fix[
                f"GHI*sin(a+b)-{suffix}"
            ]
            /
            df_fix["Sin(a)"].replace(
                0,
                np.nan
            )
        )

    # --------------------------------------------------------
    # Keep original C11 name
    # --------------------------------------------------------

    df_fix["GHI*sin(a)"] = (
        pd.to_numeric(
            df_ghi["GHI C11"],
            errors="coerce"
        ).fillna(0).to_numpy()
        *
        df_fix["Sin(a)"].to_numpy()
    )

    df_fix["GHI*sin(a+b)"] = (
        pd.to_numeric(
            df_ghi["GHI C11"],
            errors="coerce"
        ).fillna(0).to_numpy()
        *
        df_fix["SIN(a+b)"].to_numpy()
    )

    df_fix["POA fixed"] = (
        df_fix["GHI*sin(a+b)"]
        /
        df_fix["Sin(a)"].replace(
            0,
            np.nan
        )
    )

    return df_fix


# ============================================================
# FUNCTION: CALCULATE EFFECTIVE AREAS
# ============================================================

def calculate_effective_areas(
    df_source,
    df_w_source,
    error
):

    df = df_source.copy()
    df_w = df_w_source.copy()

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    df["Error %"] = error

    # --------------------------------------------------------
    # NET EFFICIENCY
    # --------------------------------------------------------

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        -
        df["Error %"]
    )

    # --------------------------------------------------------
    # TOTAL AREA
    # --------------------------------------------------------

    df["Total area (m2)"] = (
        df["No of Module"]
        *
        df["Area of 1 Module (m2)"]
    )

    # --------------------------------------------------------
    # EFFECTIVE AREA
    # --------------------------------------------------------

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        *
        df["Total area (m2)"]
        /
        100
    )

    # --------------------------------------------------------
    # CLUSTER SUM
    # --------------------------------------------------------

    cluster_sums = (
        df
        .groupby("Clusters")["Eff Area"]
        .sum()
    )

    df_w["Eff Area(m2)"] = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0)
    )

    return df, df_w


# ============================================================
# FIND BEST ERROR %
# ============================================================

def find_best_error(
    df_source,
    df_w_source,
    df_fix_source,
    df_ghi,
    plant_type
):

    actual = pd.to_numeric(
        df_fix_source["Actual"],
        errors="coerce"
    ).fillna(0).to_numpy()

    actual_peak = actual.max()

    results = []

    for error in np.arange(
        0,
        10.01,
        0.1
    ):

        df, df_w = calculate_effective_areas(
            df_source,
            df_w_source,
            error
        )

        weights = (
            pd.to_numeric(
                df_w.iloc[:5, 1],
                errors="coerce"
            )
            .fillna(0)
            .to_numpy(dtype=float)
        )

        if plant_type == "Fixed":

            df_calc = df_fix_source.copy()

            power = np.zeros(
                len(df_calc),
                dtype=float
            )

            poa_cols = [
                "POA fixed",
                "POA Fixed-CL2",
                "POA Fixed-CL3",
                "POA Fixed-CL4",
                "POA Fixed-CL5"
            ]

            for i, poa_col in enumerate(
                poa_cols
            ):

                poa = pd.to_numeric(
                    df_calc[poa_col],
                    errors="coerce"
                ).fillna(0).to_numpy()

                power += (
                    poa *
                    weights[i]
                ) /
                1_000_000

        else:

            power = calculate_tracking_forecast(
                df_ghi=df_ghi,
                df_fix=df_fix_source,
                weights=weights,
                DHI=0,
                GHI_Starting_Block=10,
                GHI_Ending_Block=70,
                GHI_Max_Block=50,
                Tracking_angle_lim_E=45,
                Tracking_angle_lim_W=45
            )

        calculated_peak = (
            np.nanmax(power)
        )

        peak_error = abs(
            calculated_peak -
            actual_peak
        )

        peak_error_pct = (
            peak_error /
            actual_peak *
            100
            if actual_peak != 0
            else np.nan
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

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["Peak Error"].idxmin()
    ]

    return (
        float(best_row["Error %"]),
        results_df
    )


# ============================================================
# FIXED CALCULATION
# ============================================================

def calculate_fixed(
    df_source,
    df_w_source,
    df_fix_source,
    best_error
):

    df, df_w = calculate_effective_areas(
        df_source,
        df_w_source,
        best_error
    )

    df_fix = df_fix_source.copy()

    weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    poa_cols = [
        "POA fixed",
        "POA Fixed-CL2",
        "POA Fixed-CL3",
        "POA Fixed-CL4",
        "POA Fixed-CL5"
    ]

    power_cols = []

    for i, poa_col in enumerate(
        poa_cols
    ):

        power_col = (
            f"CL{i + 1}_Fixed "
            f"Power=I*Ƞ*A"
        )

        df_fix[power_col] = (
            pd.to_numeric(
                df_fix[poa_col],
                errors="coerce"
            )
            .fillna(0)
            *
            weights[i]
            /
            1_000_000
        )

        power_cols.append(
            power_col
        )

    df_fix[
        "Total Power (CL1+CL2+…)"
    ] = (
        df_fix[power_cols]
        .sum(axis=1)
    )

    actual = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce"
    ).fillna(0)

    calculated_peak = (
        df_fix[
            "Total Power (CL1+CL2+…)"
        ].max()
    )

    actual_peak = actual.max()

    peak_error = abs(
        calculated_peak -
        actual_peak
    )

    peak_error_pct = (
        peak_error /
        actual_peak *
        100
        if actual_peak != 0
        else np.nan
    )

    return (
        df,
        df_w,
        df_fix,
        calculated_peak,
        actual_peak,
        peak_error,
        peak_error_pct
    )


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking_forecast(
    df_ghi,
    df_fix,
    weights,
    DHI,
    GHI_Starting_Block,
    GHI_Ending_Block,
    GHI_Max_Block,
    Tracking_angle_lim_E,
    Tracking_angle_lim_W
):

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
            .to_numpy(
                dtype=float
            )
            for col in ghi_cols
        ]
    )

    # --------------------------------------------------------
    # BLOCKS
    # --------------------------------------------------------

    if "Block No." in df_fix.columns:

        blocks = pd.to_numeric(
            df_fix["Block No."],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

    else:

        blocks = np.arange(
            len(df_fix),
            dtype=float
        )

    # --------------------------------------------------------
    # SAME JUPYTER FORMULA
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
    # ZENITH
    # --------------------------------------------------------

    zenith = np.where(
        blocks <= GHI_Max_Block,

        np.minimum(
            89,
            m1 *
            (
                blocks -
                GHI_Max_Block
            )
        ),

        np.minimum(
            89,
            m2 *
            (
                blocks -
                GHI_Max_Block
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
                    zenith >
                    Tracking_angle_lim_W
                )
            ),

            Tracking_angle_lim_W,

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
    # DHI
    # --------------------------------------------------------

    dhi = (
        ghi_matrix *
        DHI /
        100
    )

    # --------------------------------------------------------
    # DNI
    # --------------------------------------------------------

    dni = (
        ghi_matrix -
        dhi
    ) / cos_alpha[:, None]

    # --------------------------------------------------------
    # POWER
    # --------------------------------------------------------

    forecast = (
        dni @ weights
    ) / 1_000_000

    return (
        forecast,
        zenith,
        panel,
        cos_alpha,
        dni
    )


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def optimize_tracking(
    df_ghi,
    df_fix,
    weights
):

    ghi_matrix = np.column_stack(
        [
            pd.to_numeric(
                df_ghi[col],
                errors="coerce"
            )
            .fillna(0)
            .to_numpy(
                dtype=float
            )
            for col in ghi_cols
        ]
    )

    if "Block No." in df_fix.columns:

        blocks = pd.to_numeric(
            df_fix["Block No."],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

    else:

        blocks = np.arange(
            len(df_fix),
            dtype=float
        )

    actual_full = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce"
    ).fillna(0).to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # SAME MASK AS JUPYTER
    # --------------------------------------------------------

    mask = actual_full != 0

    actual = actual_full[mask]

    if len(actual) == 0:

        raise ValueError(
            "No non-zero Actual power values found."
        )

    actual_max = actual.max()
    actual_sum = actual.sum()

    # --------------------------------------------------------
    # OBJECTIVE
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
        # VALIDATION
        # ----------------------------------------------------

        if not (
            GHI_Starting_Block
            <
            GHI_Max_Block
            <
            GHI_Ending_Block
        ):

            return 1e9

        # ----------------------------------------------------
        # SLOPES
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
        # ZENITH
        # ----------------------------------------------------

        zenith = np.where(
            blocks <= GHI_Max_Block,

            np.minimum(
                89,
                m1 *
                (
                    blocks -
                    GHI_Max_Block
                )
            ),

            np.minimum(
                89,
                m2 *
                (
                    blocks -
                    GHI_Max_Block
                )
            )
        )

        # ----------------------------------------------------
        # PANEL
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
                        zenith >
                        Tracking_angle_lim_W
                    )
                ),

                Tracking_angle_lim_W,

                zenith
            )
        )

        # ----------------------------------------------------
        # COS
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
            ghi_matrix *
            DHI /
            100
        )

        # ----------------------------------------------------
        # DNI
        # ----------------------------------------------------

        dni = (
            ghi_matrix -
            dhi
        ) / cos_alpha[:, None]

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        prediction_full = (
            dni @ weights
        ) / 1_000_000

        if (
            np.isnan(
                prediction_full
            ).any()
            or
            np.isinf(
                prediction_full
            ).any()
        ):

            return 1e9

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        prediction = (
            prediction_full[mask]
        )

        if len(prediction) == 0:
            return 1e9

        # ----------------------------------------------------
        # BLOCK ERROR
        # ----------------------------------------------------

        block_error = (
            np.mean(
                np.abs(
                    actual -
                    prediction
                )
            )
            /
            actual_max
        )

        # ----------------------------------------------------
        # PEAK ERROR
        # ----------------------------------------------------

        peak_error = (
            abs(
                actual_max -
                prediction.max()
            )
            /
            actual_max
        )

        # ----------------------------------------------------
        # ENERGY ERROR
        # ----------------------------------------------------

        energy_error = (
            abs(
                actual_sum -
                prediction.sum()
            )
            /
            actual_sum
        )

        # ----------------------------------------------------
        # SAME SCORE
        # ----------------------------------------------------

        score = (
            0.80 *
            block_error
            +
            0.10 *
            peak_error
            +
            0.10 *
            energy_error
        )

        return score

    # --------------------------------------------------------
    # SAME BOUNDS
    # --------------------------------------------------------

    bounds = [
        (0, 10),
        (10, 30),
        (65, 80),
        (47, 53),
        (10, 70),
        (10, 70)
    ]

    # --------------------------------------------------------
    # SAME OPTIMIZER
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

    # --------------------------------------------------------
    # SAME INTEGER ROUNDING
    # --------------------------------------------------------

    best = np.round(
        result.x
    ).astype(int)

    parameters = {
        "DHI": best[0],
        "GHI Starting Block": best[1],
        "GHI Ending Block": best[2],
        "GHI Max Block": best[3],
        "Tracking East Limit": best[4],
        "Tracking West Limit": best[5]
    }

    return (
        parameters,
        result.fun
    )


# ============================================================
# RUN CALCULATION
# ============================================================

st.divider()

run_calculation = st.button(
    "Run Calculation",
    type="primary",
    use_container_width=True
)

if not run_calculation:
    st.info(
        f"File uploaded successfully. "
        f"Plant type selected: {plant_type}. "
        f"Click Run Calculation."
    )
    st.stop()


# ============================================================
# BEST ERROR %
# ============================================================

with st.spinner("Finding best Error %..."):

    (
        best_error,
        error_results
    ) = find_best_error(
        df_area,
        df_w_base,
        df_fix_base,
        df_ghi,
        plant_type
    )


# ============================================================
# FIXED
# ============================================================

if plant_type == "Fixed":

    df_fix = prepare_solar_angles(
        df_fix_base,
        df_ghi,
        lat,
        month_lookup
    )

    (
        df_final,
        df_w_final,
        df_fix_final,
        calculated_peak,
        actual_peak,
        peak_error,
        peak_error_pct
    ) = calculate_fixed(
        df_area,
        df_w_base,
        df_fix,
        best_error
    )

    st.success(
        f"Best Error %: {best_error:.2f}%"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Best Error %",
        f"{best_error:.2f}%"
    )

    col2.metric(
        "Calculated Peak",
        f"{calculated_peak:.3f}"
    )

    col3.metric(
        "Actual Peak",
        f"{actual_peak:.3f}"
    )

    col4.metric(
        "Peak Error %",
        f"{peak_error_pct:.3f}%"
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=df_fix_final[
                "Total Power (CL1+CL2+…)"
            ],
            mode="lines",
            name="Forecast"
        )
    )

    fig.add_trace(
        go.Scatter(
            y=df_fix_final["Actual"],
            mode="lines",
            name="Actual"
        )
    )

    fig.update_layout(
        title="Fixed Plant Forecast vs Actual",
        xaxis_title="Block",
        yaxis_title="Power",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    with st.expander(
        "Fixed Calculation Data"
    ):

        st.dataframe(
            df_fix_final,
            use_container_width=True
        )

    with st.expander(
        "Error % Optimization"
    ):

        st.dataframe(
            error_results,
            use_container_width=True
        )


# ============================================================
# TRACKING
# ============================================================

else:

    # --------------------------------------------------------
    # GET EFFECTIVE AREAS USING BEST ERROR
    # --------------------------------------------------------

    (
        df_tracking_area,
        df_w_tracking
    ) = calculate_effective_areas(
        df_area,
        df_w_base,
        best_error
    )

    # --------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------

    cl_weights = (
        pd.to_numeric(
            df_w_tracking.iloc[:5, 1],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # BACKEND CAL SHEETS
    # --------------------------------------------------------

    with st.spinner(
        "Reading Tracking backend data..."
    ):

        df_bcal1 = read_excel_sheet(
            "Backend Cal C11"
        )

        df_bcal2 = read_excel_sheet(
            "Backend Cal C12"
        )

        df_bcal3 = read_excel_sheet(
            "Backend Cal C13"
        )

        df_bcal4 = read_excel_sheet(
            "Backend Cal C14"
        )

        df_bcal5 = read_excel_sheet(
            "Backend Cal C15"
        )

        backend_list = [
            df_bcal1,
            df_bcal2,
            df_bcal3,
            df_bcal4,
            df_bcal5
        ]

        df_trac = read_excel_sheet(
            "Tracking",
            header=1
        )

        df_trac.columns = (
            df_trac.columns
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # USE BACKEND BLOCK NUMBERS EXACTLY LIKE JUPYTER
    # --------------------------------------------------------

    if "Block No." not in backend_list[0].columns:

        st.error(
            "Backend Cal C11 does not contain "
            "'Block No.'."
        )

        st.stop()

    # --------------------------------------------------------
    # ALIGN LENGTHS
    # --------------------------------------------------------

    n = min(
        len(df_trac),
        len(df_ghi),
        len(backend_list[0]),
        len(df_fix_base)
    )

    df_trac = (
        df_trac
        .iloc[:n]
        .copy()
    )

    df_ghi_tracking = (
        df_ghi
        .iloc[:n]
        .copy()
    )

    df_actual_tracking = (
        df_fix_base["Actual"]
        .iloc[:n]
        .copy()
    )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    with st.spinner(
        "Optimizing Tracking parameters..."
    ):

        (
            tracking_parameters,
            tracking_score
        ) = optimize_tracking(
            df_ghi_tracking,
            df_fix_base.iloc[:n].copy(),
            cl_weights
        )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    DHI = tracking_parameters[
        "DHI"
    ]

    GHI_Starting_Block = (
        tracking_parameters[
            "GHI Starting Block"
        ]
    )

    GHI_Ending_Block = (
        tracking_parameters[
            "GHI Ending Block"
        ]
    )

    GHI_Max_Block = (
        tracking_parameters[
            "GHI Max Block"
        ]
    )

    Tracking_angle_lim_E = (
        tracking_parameters[
            "Tracking East Limit"
        ]
    )

    Tracking_angle_lim_W = (
        tracking_parameters[
            "Tracking West Limit"
        ]
    )

    # --------------------------------------------------------
    # FINAL TRACKING CALCULATION
    # --------------------------------------------------------

    (
        forecast,
        zenith,
        panel,
        cos_alpha,
        dni
    ) = calculate_tracking_forecast(
        df_ghi_tracking,
        df_fix_base.iloc[:n].copy(),
        cl_weights,
        DHI,
        GHI_Starting_Block,
        GHI_Ending_Block,
        GHI_Max_Block,
        Tracking_angle_lim_E,
        Tracking_angle_lim_W
    )

    # --------------------------------------------------------
    # SAVE TO TRACKING DATAFRAME
    # --------------------------------------------------------

    df_trac[
        "Zenith Angle"
    ] = zenith

    df_trac[
        "Panel Angle"
    ] = panel

    df_trac[
        "Fixed Power=I*Ƞ*A"
    ] = forecast

    df_trac[
        "DHI (%)"
    ] = DHI

    df_trac[
        "DNI"
    ] = np.nanmean(
        dni,
        axis=1
    )

    df_trac[
        "Forecast"
    ] = forecast

    df_trac[
        "Actual"
    ] = df_actual_tracking.to_numpy()

    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

    actual = (
        df_actual_tracking
        .to_numpy(
            dtype=float
        )
    )

    mask = actual != 0

    actual_valid = actual[mask]
    forecast_valid = forecast[mask]

    actual_peak = (
        actual_valid.max()
        if len(actual_valid)
        else 0
    )

    forecast_peak = (
        forecast_valid.max()
        if len(forecast_valid)
        else 0
    )

    peak_error = abs(
        actual_peak -
        forecast_peak
    )

    peak_error_pct = (
        peak_error /
        actual_peak *
        100
        if actual_peak != 0
        else np.nan
    )

    energy_error_pct = (
        abs(
            actual_valid.sum() -
            forecast_valid.sum()
        )
        /
        actual_valid.sum()
        *
        100
        if actual_valid.sum() != 0
        else np.nan
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    st.success(
        "Tracking calculation completed."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Best Error %",
        f"{best_error:.2f}%"
    )

    col2.metric(
        "Forecast Peak",
        f"{forecast_peak:.3f}"
    )

    col3.metric(
        "Actual Peak",
        f"{actual_peak:.3f}"
    )

    col4.metric(
        "Peak Error %",
        f"{peak_error_pct:.3f}%"
    )

    # --------------------------------------------------------
    # TRACKING PARAMETERS
    # --------------------------------------------------------

    st.subheader(
        "Optimized Tracking Parameters"
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "DHI",
        DHI
    )

    p2.metric(
        "GHI Starting Block",
        GHI_Starting_Block
    )

    p3.metric(
        "GHI Ending Block",
        GHI_Ending_Block
    )

    p4, p5, p6 = st.columns(3)

    p4.metric(
        "GHI Max Block",
        GHI_Max_Block
    )

    p5.metric(
        "Tracking East Limit",
        Tracking_angle_lim_E
    )

    p6.metric(
        "Tracking West Limit",
        Tracking_angle_lim_W
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    st.write(
        f"Optimization Score: "
        f"{tracking_score:.8f}"
    )

    st.write(
        f"Energy Error %: "
        f"{energy_error_pct:.3f}%"
    )

    # --------------------------------------------------------
    # FORECAST GRAPH
    # --------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=forecast,
            mode="lines",
            name="Forecast"
        )
    )

    fig.add_trace(
        go.Scatter(
            y=actual,
            mode="lines",
            name="Actual"
        )
    )

    fig.update_layout(
        title="Tracking Forecast vs Actual",
        xaxis_title="Block",
        yaxis_title="Power",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # ANGLES GRAPH
    # --------------------------------------------------------

    fig_angles = go.Figure()

    fig_angles.add_trace(
        go.Scatter(
            y=zenith,
            mode="lines",
            name="Zenith Angle"
        )
    )

    fig_angles.add_trace(
        go.Scatter(
            y=panel,
            mode="lines",
            name="Panel Angle"
        )
    )

    fig_angles.update_layout(
        title="Tracking Zenith and Panel Angles",
        xaxis_title="Block",
        yaxis_title="Angle (°)",
        height=450
    )

    st.plotly_chart(
        fig_angles,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TRACKING DATA
    # --------------------------------------------------------

    with st.expander(
        "Tracking Calculation Data"
    ):

        st.dataframe(
            df_trac,
            use_container_width=True
        )

    # --------------------------------------------------------
    # ERROR OPTIMIZATION
    # --------------------------------------------------------

    with st.expander(
        "Error % Optimization"
    ):

        st.dataframe(
            error_results,
            use_container_width=True
        )
