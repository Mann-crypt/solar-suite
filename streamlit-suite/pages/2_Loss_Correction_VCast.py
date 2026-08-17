# ============================================================
# STREAMLIT PAGE
# VCAST LOSS CORRECTION
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
    page_title="VCast Loss Correction",
    page_icon="☀️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("☀️ VCast Loss Correction")

st.caption(
    "Fixed and Tracking forecast correction with automatic "
    "Error % calculation and editable optimization parameters."
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
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload VCast Excel Workbook",
    type=["xlsx", "xls"]
)


if uploaded_file is None:

    st.info(
        "Upload the VCast Excel workbook to start the calculation."
    )

    st.stop()


# ============================================================
# COMMON CONSTANTS
# ============================================================

CLUSTERS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15"
]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15"
]


# ============================================================
# LOAD EXCEL
# ============================================================

@st.cache_data(show_spinner=False)
def load_workbook(file_bytes):

    return pd.ExcelFile(
        io.BytesIO(file_bytes)
    )


excel = load_workbook(
    uploaded_file.getvalue()
)


sheet_names = excel.sheet_names


# ============================================================
# WORKBOOK TYPE DETECTION
# ============================================================

if "Fixed-C11" in sheet_names:

    workbook_type = "VCast"

elif "Fixed-CL1" in sheet_names:

    workbook_type = "Cluster"

elif "Fixed" in sheet_names:

    workbook_type = "Non-Cluster"

else:

    workbook_type = "Unknown"


st.success(
    f"Workbook detected: **{workbook_type}**"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_sheets = [
    "Area & Efficiency",
    "Forecast Config",
    "Config Tilt Angle",
    "Result"
]

if plant_type == "Fixed":

    required_sheets.append(
        "Fixed-C11"
    )

else:

    required_sheets.extend([
        "Fixed-C11",
        "Tracking",
        "Backend Cal C11",
        "Backend Cal C12",
        "Backend Cal C13",
        "Backend Cal C14",
        "Backend Cal C15"
    ])


missing_sheets = [
    sheet
    for sheet in required_sheets
    if sheet not in sheet_names
]


if missing_sheets:

    st.error(
        "Missing required sheets: "
        + ", ".join(missing_sheets)
    )

    st.stop()


# ============================================================
# READ AREA & EFFICIENCY
# ============================================================

@st.cache_data(show_spinner=False)
def read_area_efficiency(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12)
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

    if "S.No." in df.columns:

        df = df[
            df["S.No."].notna()
        ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


df_area = read_area_efficiency(
    uploaded_file.getvalue()
)


# ============================================================
# CALCULATE TOTAL AREA
# ============================================================

df_area["No of Module"] = pd.to_numeric(
    df_area["No of Module"],
    errors="coerce"
)

df_area["Area of 1 Module (m2)"] = pd.to_numeric(
    df_area["Area of 1 Module (m2)"],
    errors="coerce"
)

df_area["Standard PV Efficiency (%)"] = pd.to_numeric(
    df_area["Standard PV Efficiency (%)"],
    errors="coerce"
)


df_area["Total area (m2)"] = (
    df_area["No of Module"]
    *
    df_area["Area of 1 Module (m2)"]
)


# ============================================================
# READ CLUSTER INFORMATION
# ============================================================

df_weights = pd.read_excel(
    io.BytesIO(
        uploaded_file.getvalue()
    ),
    sheet_name="Area & Efficiency",
    header=1,
    usecols=[14, 15]
)


df_weights.columns = [
    str(col).strip()
    for col in df_weights.columns
]


if len(df_weights.columns) >= 2:

    cluster_column = df_weights.columns[0]
    area_column = df_weights.columns[1]

else:

    st.error(
        "Could not read cluster effective-area columns."
    )

    st.stop()


if cluster_column != "Clusters":

    df_weights = df_weights.rename(
        columns={
            cluster_column: "Clusters"
        }
    )


df_weights = df_weights[
    df_weights["Clusters"].notna()
].copy()


# ============================================================
# READ LATITUDE
# ============================================================

df_config = pd.read_excel(
    io.BytesIO(
        uploaded_file.getvalue()
    ),
    sheet_name="Forecast Config",
    header=8
)


if "Lat" not in df_config.columns:

    st.error(
        "Lat column not found in Forecast Config."
    )

    st.stop()


lat = pd.to_numeric(
    df_config.loc[0, "Lat"],
    errors="coerce"
)


if pd.isna(lat):

    st.error(
        "Could not read latitude."
    )

    st.stop()


lat = float(lat)


# ============================================================
# READ TILT
# ============================================================

df_tilt = pd.read_excel(
    io.BytesIO(
        uploaded_file.getvalue()
    ),
    sheet_name="Config Tilt Angle",
    header=7
)


df_tilt.columns = (
    df_tilt.columns
    .astype(str)
    .str.strip()
)


if "Fixed" not in df_tilt.columns:

    st.error(
        "Fixed tilt column not found."
    )

    st.stop()


if "Unnamed: 2" in df_tilt.columns:

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num"
        }
    )


if "Unnamed: 3" in df_tilt.columns:

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 3": "Month"
        }
    )


df_tilt = df_tilt[
    df_tilt["Fixed"].notna()
].copy()


df_tilt["Month_Num"] = pd.to_numeric(
    df_tilt["Month_Num"],
    errors="coerce"
)


df_tilt["Fixed"] = pd.to_numeric(
    df_tilt["Fixed"],
    errors="coerce"
)


month_lookup = (
    df_tilt
    .dropna(
        subset=["Month_Num"]
    )
    .set_index("Month_Num")["Fixed"]
    .to_dict()
)


# ============================================================
# READ RESULT / GHI
# ============================================================

df_ghi = pd.read_excel(
    io.BytesIO(
        uploaded_file.getvalue()
    ),
    sheet_name="Result",
    usecols=range(6)
)


df_ghi.columns = [
    "Block",
    *GHI_COLS
]


df_ghi["Block"] = pd.to_numeric(
    df_ghi["Block"],
    errors="coerce"
)


df_ghi = df_ghi[
    df_ghi["Block"].notna()
].copy()


for col in GHI_COLS:

    df_ghi[col] = pd.to_numeric(
        df_ghi[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# READ FIXED-C11
# ============================================================

df_fix = pd.read_excel(
    io.BytesIO(
        uploaded_file.getvalue()
    ),
    sheet_name="Fixed-C11",
    header=1
)


df_fix.columns = (
    df_fix.columns
    .astype(str)
    .str.strip()
)


if "Date" not in df_fix.columns:

    st.error(
        "Date column not found in Fixed-C11."
    )

    st.stop()


date_valid = (
    df_fix["Date"].notna()
)


if not date_valid.any():

    st.error(
        "No valid Date rows found."
    )

    st.stop()


first_blank = np.where(
    ~date_valid.to_numpy()
)[0]


if len(first_blank) > 0:

    df_fix = df_fix.iloc[
        :first_blank[0]
    ].copy()

else:

    df_fix = df_fix.loc[
        date_valid
    ].copy()


df_fix.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# ACTUAL POWER
# ============================================================

if "Actual" not in df_fix.columns:

    st.error(
        "Actual column not found in Fixed-C11."
    )

    st.stop()


df_fix["Actual"] = pd.to_numeric(
    df_fix["Actual"],
    errors="coerce"
).fillna(0)


# ============================================================
# ALIGN DATA
# ============================================================

n = min(
    len(df_fix),
    len(df_ghi)
)


if n == 0:

    st.error(
        "No valid forecast rows found."
    )

    st.stop()


df_fix = df_fix.iloc[
    :n
].copy()


df_ghi = df_ghi.iloc[
    :n
].copy()


dates = pd.to_datetime(
    df_fix["Date"],
    errors="coerce"
)


if dates.isna().any():

    st.error(
        "Invalid dates found in Fixed-C11."
    )

    st.stop()


actual_full = (
    df_fix["Actual"]
    .to_numpy(dtype=float)
)


actual_mask = (
    np.isfinite(actual_full)
    &
    (actual_full != 0)
)


if not actual_mask.any():

    st.error(
        "Actual power contains no valid non-zero values."
    )

    st.stop()


actual_day = (
    actual_full[
        actual_mask
    ]
)


actual_peak = (
    np.max(actual_day)
)


actual_energy = (
    np.sum(actual_day)
)


# ============================================================
# GHI MATRIX
# ============================================================

ghi_matrix = np.column_stack([
    df_ghi[col]
    .to_numpy(dtype=float)
    for col in GHI_COLS
])


blocks = (
    df_ghi["Block"]
    .to_numpy(dtype=float)
)


# ============================================================
# SOLAR GEOMETRY
# ============================================================

first_date = pd.Timestamp(
    year=2025,
    month=1,
    day=1
)


day_offset = (
    dates - first_date
).dt.days.to_numpy(
    dtype=float
)


declination = (
    23.45
    *
    np.sin(
        np.radians(
            360
            *
            (
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


months = (
    dates.dt.month
    .to_numpy()
)


tilt = np.array([
    month_lookup.get(
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
    np.radians(
        elevation
    )
)


sin_ab = np.sin(
    np.radians(
        a_plus_b
    )
)


sin_a_safe = np.where(
    np.abs(sin_a) < 1e-8,
    1e-8,
    sin_a
)


# ============================================================
# FIXED POA
# ============================================================

fixed_poa = (
    ghi_matrix
    *
    sin_ab[:, None]
    /
    sin_a_safe[:, None]
)


# ============================================================
# EFFECTIVE AREA FUNCTION
# ============================================================

def calculate_effective_areas(
    error_percent
):

    temp = df_area.copy()


    temp["Error %"] = (
        error_percent
    )


    temp["Net Efficiency (%)"] = (
        temp["Standard PV Efficiency (%)"]
        -
        error_percent
    )


    temp["Net Efficiency (%)"] = (
        temp["Net Efficiency (%)"]
        .clip(lower=0)
    )


    temp["Eff Area"] = (
        temp["Net Efficiency (%)"]
        *
        temp["Total area (m2)"]
        /
        100
    )


    cluster_sums = (
        temp
        .groupby("Clusters")["Eff Area"]
        .sum()
    )


    weights = (
        df_weights["Clusters"]
        .map(cluster_sums)
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )


    return (
        temp,
        weights
    )


# ============================================================
# FIXED FORECAST FUNCTION
# ============================================================

def calculate_fixed(
    error_percent
):

    temp,
    weights = calculate_effective_areas(
        error_percent
    )


    power_matrix = (
        fixed_poa
        *
        weights[None, :]
        /
        1_000_000
    )


    forecast = (
        power_matrix
        .sum(axis=1)
    )


    return (
        temp,
        weights,
        power_matrix,
        forecast
    )


# ============================================================
# FIND FIXED ERROR %
# ============================================================

@st.cache_data(show_spinner=False)
def optimize_fixed_error(
    file_bytes
):

    # This function uses the same workbook data
    # already loaded above.

    errors = np.arange(
        0,
        10.01,
        0.1
    )


    test_results = []


    for error in errors:

        temp = df_area.copy()


        temp["Error %"] = (
            error
        )


        temp["Net Efficiency (%)"] = (
            temp["Standard PV Efficiency (%)"]
            -
            error
        )


        temp["Net Efficiency (%)"] = (
            temp["Net Efficiency (%)"]
            .clip(lower=0)
        )


        temp["Eff Area"] = (
            temp["Net Efficiency (%)"]
            *
            temp["Total area (m2)"]
            /
            100
        )


        cluster_sums = (
            temp
            .groupby("Clusters")["Eff Area"]
            .sum()
        )


        weights = (
            df_weights["Clusters"]
            .map(cluster_sums)
            .fillna(0)
            .to_numpy(
                dtype=float
            )
        )


        power = (
            fixed_poa
            *
            weights[None, :]
            /
            1_000_000
        )


        forecast = (
            power.sum(
                axis=1
            )
        )


        forecast_day = (
            forecast[
                actual_mask
            ]
        )


        calculated_peak = (
            np.max(
                forecast_day
            )
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
        )


        test_results.append({

            "Error %": error,

            "Calculated Peak":
                calculated_peak,

            "Actual Peak":
                actual_peak,

            "Peak Error":
                peak_error,

            "Peak Error %":
                peak_error_pct

        })


    result_df = pd.DataFrame(
        test_results
    )


    best = result_df.loc[
        result_df[
            "Peak Error"
        ].idxmin()
    ]


    return (
        float(best["Error %"]),
        result_df
    )


# ============================================================
# FIXED SECTION
# ============================================================

if plant_type == "Fixed":

    st.header(
        "Fixed Plant Loss Correction"
    )


    # --------------------------------------------------------
    # AUTOMATIC ERROR
    # --------------------------------------------------------

    with st.spinner(
        "Calculating Fixed Error %..."
    ):

        auto_fixed_error, fixed_test_results = (
            optimize_fixed_error(
                uploaded_file.getvalue()
            )
        )


    st.success(
        f"Automatically calculated Error %: "
        f"**{auto_fixed_error:.2f}%**"
    )


    # --------------------------------------------------------
    # USER EDITABLE ERROR
    # --------------------------------------------------------

    fixed_error = st.number_input(
        "Fixed Error %",
        min_value=0.0,
        max_value=20.0,
        value=float(auto_fixed_error),
        step=0.1,
        format="%.2f"
    )


    # --------------------------------------------------------
    # FINAL FIXED CALCULATION
    # --------------------------------------------------------

    (
        df_final,
        fixed_weights,
        fixed_power_matrix,
        fixed_forecast
    ) = calculate_fixed(
        fixed_error
    )


    fixed_day = (
        fixed_forecast[
            actual_mask
        ]
    )


    fixed_peak = (
        np.max(
            fixed_day
        )
    )


    fixed_peak_error = abs(
        fixed_peak
        -
        actual_peak
    )


    fixed_peak_error_pct = (
        fixed_peak_error
        /
        actual_peak
        *
        100
    )


    fixed_energy = (
        np.sum(
            fixed_day
        )
    )


    fixed_energy_error = abs(
        actual_energy
        -
        fixed_energy
    ) / actual_energy


    fixed_block_error = (
        np.mean(
            np.abs(
                actual_day
                -
                fixed_day
            )
        )
        /
        actual_peak
    )


    fixed_score = (
        0.80
        *
        fixed_block_error
        +
        0.10
        *
        (
            fixed_peak_error
            /
            actual_peak
        )
        +
        0.10
        *
        fixed_energy_error
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Actual Peak",
        f"{actual_peak:.4f} MW"
    )


    c2.metric(
        "Fixed Peak",
        f"{fixed_peak:.4f} MW"
    )


    c3.metric(
        "Peak Error",
        f"{fixed_peak_error_pct:.3f}%"
    )


    c4.metric(
        "Overall Score",
        f"{fixed_score:.5f}"
    )


    # --------------------------------------------------------
    # EFFICIENCY TABLE
    # --------------------------------------------------------

    efficiency_view = df_final[
        [
            "Clusters",
            "Standard PV Efficiency (%)",
            "Error %",
            "Net Efficiency (%)",
            "Total area (m2)",
            "Eff Area"
        ]
    ].copy()


    st.subheader(
        "Efficiency & Effective Area"
    )


    st.dataframe(
        efficiency_view,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # CLUSTER EFFECTIVE AREA
    # --------------------------------------------------------

    cluster_view = pd.DataFrame({

        "Cluster":
            CLUSTERS,

        "Effective Area (m²)":
            fixed_weights

    })


    st.subheader(
        "Cluster Effective Areas"
    )


    st.dataframe(
        cluster_view,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # ERROR TEST RESULTS
    # --------------------------------------------------------

    with st.expander(
        "Fixed Error % Optimization Results"
    ):

        st.dataframe(
            fixed_test_results,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual_full,
            mode="lines",
            name="Actual"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=fixed_forecast,
            mode="lines",
            name="Fixed Forecast"
        )
    )


    fig.update_layout(
        title="Actual vs Fixed Forecast",
        xaxis_title="Block",
        yaxis_title="Power (MW)",
        hovermode="x unified",
        height=500
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    output = io.BytesIO()


    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df_final.to_excel(
            writer,
            sheet_name="Area & Efficiency",
            index=False
        )


        df_output = df_fix.copy()


        for i, cluster in enumerate(
            CLUSTERS
        ):

            df_output[
                f"{cluster}_Fixed Power=I*Ƞ*A"
            ] = fixed_power_matrix[
                :, i
            ]


        df_output[
            "Total Power (CL1+CL2+…)"
        ] = fixed_forecast


        df_output.to_excel(
            writer,
            sheet_name="Fixed-C11",
            index=False
        )


        pd.DataFrame({

            "Metric": [

                "Error %",

                "Actual Peak",

                "Fixed Peak",

                "Peak Error",

                "Peak Error %",

                "Block Error",

                "Energy Error",

                "Overall Score"

            ],

            "Value": [

                fixed_error,

                actual_peak,

                fixed_peak,

                fixed_peak_error,

                fixed_peak_error_pct,

                fixed_block_error,

                fixed_energy_error,

                fixed_score

            ]

        }).to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )


    st.download_button(
        "⬇️ Download Fixed Results",
        data=output.getvalue(),
        file_name="VCast_Fixed_Loss_Correction.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# ============================================================
# TRACKING SECTION
# ============================================================

else:

    st.header(
        "Tracking Plant Loss Correction"
    )


    # ========================================================
    # TRACKING SHEETS
    # ========================================================

    backend_list = []


    for cluster in CLUSTERS:

        backend = pd.read_excel(
            io.BytesIO(
                uploaded_file.getvalue()
            ),
            sheet_name=f"Backend Cal {cluster}"
        )

        backend_list.append(
            backend
        )


    df_tracking = pd.read_excel(
        io.BytesIO(
            uploaded_file.getvalue()
        ),
        sheet_name="Tracking",
        header=1
    )


    df_tracking = df_tracking.iloc[
        :n
    ].copy()


    df_tracking.reset_index(
        drop=True,
        inplace=True
    )


    # ========================================================
    # TRACKING BLOCKS
    # ========================================================

    block_column = None


    possible_block_columns = [
        "Block No.",
        "Block",
        "Block No"
    ]


    for col in possible_block_columns:

        if col in backend_list[0].columns:

            block_column = col

            break


    if block_column is None:

        st.error(
            "Block No. column not found in Backend Cal C11."
        )

        st.stop()


    blocks_tracking = (
        pd.to_numeric(
            backend_list[0][
                block_column
            ],
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )


    blocks_tracking = (
        blocks_tracking[
            :n
        ]
    )


    # ========================================================
    # TRACKING ERROR OPTIMIZATION
    # ========================================================

    # IMPORTANT:
    #
    # Tracking Error % is calculated independently.
    #
    # First optimize Error % using the tracking model
    # with the default tracking parameters.
    #
    # Then the user can edit Error % and all tracking
    # parameters.


    default_dhi = 5

    default_start = 20

    default_end = 72

    default_max = 50

    default_east = 45

    default_west = 45


    # ========================================================
    # TRACKING CALCULATION
    # ========================================================

    def calculate_tracking_model(
        error_percent,
        dhi_percent,
        start_block,
        end_block,
        max_block,
        east_limit,
        west_limit
    ):

        # ----------------------------------------------------
        # Effective areas
        # ----------------------------------------------------

        temp, tracking_weights = (
            calculate_effective_areas(
                error_percent
            )
        )


        # ----------------------------------------------------
        # Validate blocks
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Same Jupyter calculation
        # ----------------------------------------------------

        m1 = (
            90
            /
            denominator_1
        )


        m2 = (
            90
            /
            denominator_2
        )


        zenith = np.where(

            blocks_tracking
            <= max_block,

            np.minimum(
                89,
                m1
                *
                (
                    blocks_tracking
                    -
                    max_block
                )
            ),

            np.minimum(
                89,
                m2
                *
                (
                    blocks_tracking
                    -
                    max_block
                )
            )
        )


        # ----------------------------------------------------
        # Panel
        # ----------------------------------------------------

        panel = np.where(

            blocks_tracking
            < max_block,

            np.minimum(
                zenith,
                abs(east_limit)
            ),

            np.where(

                (
                    (
                        blocks_tracking
                        > max_block
                    )
                    &
                    (
                        zenith
                        > west_limit
                    )
                ),

                west_limit,

                zenith
            )
        )


        # ----------------------------------------------------
        # Cosine
        # ----------------------------------------------------

        cos_alpha = np.cos(
            np.radians(
                panel
            )
        )


        cos_alpha = np.clip(
            cos_alpha,
            1e-6,
            None
        )


        # ----------------------------------------------------
        # DHI
        # ----------------------------------------------------

        dhi_matrix = (
            ghi_matrix
            *
            dhi_percent
            /
            100
        )


        # ----------------------------------------------------
        # DNI
        # ----------------------------------------------------

        dni = (
            ghi_matrix
            -
            dhi_matrix
        ) / cos_alpha[:, None]


        # ----------------------------------------------------
        # EXACT JUPYTER LOGIC
        #
        # DNI @ cluster effective areas
        # ----------------------------------------------------

        forecast = (
            dni
            @ tracking_weights
        ) / 1_000_000


        return {

            "forecast":
                forecast,

            "tracking_weights":
                tracking_weights,

            "zenith":
                zenith,

            "panel":
                panel,

            "dni":
                dni,

            "temp":
                temp

        }


    # ========================================================
    # TRACKING ERROR %
    # ========================================================

    tracking_error_results = []


    for error in np.arange(
        0,
        10.01,
        0.1
    ):

        model = calculate_tracking_model(

            error,

            default_dhi,

            default_start,

            default_end,

            default_max,

            default_east,

            default_west

        )


        if model is None:

            continue


        forecast = (
            model["forecast"]
        )


        forecast_day = (
            forecast[
                actual_mask
            ]
        )


        if len(forecast_day) == 0:

            continue


        calculated_peak = (
            np.max(
                forecast_day
            )
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
        )


        tracking_error_results.append({

            "Error %":
                error,

            "Calculated Peak":
                calculated_peak,

            "Actual Peak":
                actual_peak,

            "Peak Error":
                peak_error,

            "Peak Error %":
                peak_error_pct

        })


    tracking_error_df = pd.DataFrame(
        tracking_error_results
    )


    if tracking_error_df.empty:

        st.error(
            "Tracking Error % optimization "
            "did not produce valid results."
        )

        st.stop()


    tracking_best_error = float(
        tracking_error_df.loc[
            tracking_error_df[
                "Peak Error"
            ].idxmin(),
            "Error %"
        ]
    )


    # ========================================================
    # TRACKING USER PARAMETERS
    # ========================================================

    st.success(
        f"Automatically calculated Tracking Error %: "
        f"**{tracking_best_error:.2f}%**"
    )


    st.subheader(
        "Tracking Parameters"
    )


    # --------------------------------------------------------
    # ERROR %
    # --------------------------------------------------------

    t1, t2, t3 = st.columns(3)


    with t1:

        tracking_error = st.number_input(
            "Tracking Error %",
            min_value=0.0,
            max_value=20.0,
            value=float(
                tracking_best_error
            ),
            step=0.1,
            format="%.2f"
        )


    with t2:

        DHI = st.number_input(
            "DHI (%)",
            min_value=0.0,
            max_value=10.0,
            value=float(
                default_dhi
            ),
            step=1.0
        )


    with t3:

        GHI_start = st.number_input(
            "GHI Starting Block",
            min_value=1,
            max_value=50,
            value=int(
                default_start
            ),
            step=1
        )


    t4, t5, t6 = st.columns(3)


    with t4:

        GHI_end = st.number_input(
            "GHI Ending Block",
            min_value=51,
            max_value=95,
            value=int(
                default_end
            ),
            step=1
        )


    with t5:

        GHI_max = st.number_input(
            "GHI Max Block",
            min_value=1,
            max_value=95,
            value=int(
                default_max
            ),
            step=1
        )


    with t6:

        east_limit = st.number_input(
            "Tracking East Limit (°)",
            min_value=0,
            max_value=70,
            value=int(
                default_east
            ),
            step=1
        )


    west_limit = st.number_input(
        "Tracking West Limit (°)",
        min_value=0,
        max_value=70,
        value=int(
            default_west
        ),
        step=1
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if not (
        GHI_start
        < GHI_max
        < GHI_end
    ):

        st.error(
            "Tracking block sequence must satisfy: "
            "Starting Block < Max Block < Ending Block"
        )

        st.stop()


    # ========================================================
    # FINAL TRACKING MODEL
    # ========================================================

    tracking_model = (
        calculate_tracking_model(

            tracking_error,

            DHI,

            GHI_start,

            GHI_end,

            GHI_max,

            east_limit,

            west_limit

        )
    )


    if tracking_model is None:

        st.error(
            "Unable to calculate Tracking forecast "
            "with the selected parameters."
        )

        st.stop()


    tracking_forecast = (
        tracking_model["forecast"]
    )


    tracking_weights = (
        tracking_model[
            "tracking_weights"
        ]
    )


    zenith = (
        tracking_model[
            "zenith"
        ]
    )


    panel = (
        tracking_model[
            "panel"
        ]
    )


    dni = (
        tracking_model[
            "dni"
        ]
    )


    # ========================================================
    # FINAL TRACKING METRICS
    # ========================================================

    tracking_day = (
        tracking_forecast[
            actual_mask
        ]
    )


    tracking_peak = (
        np.max(
            tracking_day
        )
    )


    tracking_peak_error = abs(
        tracking_peak
        -
        actual_peak
    )


    tracking_peak_error_pct = (
        tracking_peak_error
        /
        actual_peak
        *
        100
    )


    tracking_energy = (
        np.sum(
            tracking_day
        )
    )


    tracking_energy_error = abs(
        actual_energy
        -
        tracking_energy
    ) / actual_energy


    tracking_block_error = (
        np.mean(
            np.abs(
                actual_day
                -
                tracking_day
            )
        )
        /
        actual_peak
    )


    tracking_score = (
        0.80
        *
        tracking_block_error
        +
        0.10
        *
        (
            tracking_peak_error
            /
            actual_peak
        )
        +
        0.10
        *
        tracking_energy_error
    )


    # ========================================================
    # METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Actual Peak",
        f"{actual_peak:.4f} MW"
    )


    c2.metric(
        "Tracking Peak",
        f"{tracking_peak:.4f} MW"
    )


    c3.metric(
        "Peak Error",
        f"{tracking_peak_error_pct:.3f}%"
    )


    c4.metric(
        "Overall Score",
        f"{tracking_score:.5f}"
    )


    # ========================================================
    # PARAMETER SUMMARY
    # ========================================================

    parameter_summary = pd.DataFrame({

        "Parameter": [

            "Error %",

            "DHI (%)",

            "GHI Starting Block",

            "GHI Ending Block",

            "GHI Max Block",

            "East Tracking Limit",

            "West Tracking Limit"

        ],

        "Automatic / Current Value": [

            tracking_error,

            DHI,

            GHI_start,

            GHI_end,

            GHI_max,

            east_limit,

            west_limit

        ]

    })


    st.subheader(
        "Tracking Parameters Summary"
    )


    st.dataframe(
        parameter_summary,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CLUSTER EFFECTIVE AREAS
    # ========================================================

    cluster_view = pd.DataFrame({

        "Cluster":
            CLUSTERS,

        "Tracking Effective Area (m²)":
            tracking_weights

    })


    st.subheader(
        "Tracking Effective Areas"
    )


    st.dataframe(
        cluster_view,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # TRACKING DATA
    # ========================================================

    df_tracking["Zenith Angle"] = (
        zenith
    )


    df_tracking["Panel Angle"] = (
        panel
    )


    # Exact Jupyter output
    df_tracking[
        "Fixed Power=I*Ƞ*A"
    ] = tracking_forecast


    st.subheader(
        "Tracking Forecast Data"
    )


    preview_cols = [
        col
        for col in [
            "Block No.",
            "Zenith Angle",
            "Panel Angle",
            "Fixed Power=I*Ƞ*A"
        ]
        if col in df_tracking.columns
    ]


    st.dataframe(
        df_tracking[
            preview_cols
        ],
        use_container_width=True,
        height=400
    )


    # ========================================================
    # ERROR TEST RESULTS
    # ========================================================

    with st.expander(
        "Tracking Error % Optimization Results"
    ):

        st.dataframe(
            tracking_error_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # FORECAST GRAPH
    # ========================================================

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual_full,
            mode="lines",
            name="Actual"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=tracking_forecast,
            mode="lines",
            name="Tracking Forecast"
        )
    )


    fig.update_layout(
        title="Actual vs Tracking Forecast",
        xaxis_title="Block",
        yaxis_title="Power (MW)",
        hovermode="x unified",
        height=500
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    output = io.BytesIO()


    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df_tracking.to_excel(
            writer,
            sheet_name="Tracking",
            index=False
        )


        pd.DataFrame({

            "Metric": [

                "Error %",

                "DHI (%)",

                "GHI Starting Block",

                "GHI Ending Block",

                "GHI Max Block",

                "East Tracking Limit",

                "West Tracking Limit",

                "Actual Peak",

                "Tracking Peak",

                "Peak Error",

                "Peak Error %",

                "Block Error",

                "Energy Error",

                "Overall Score"

            ],

            "Value": [

                tracking_error,

                DHI,

                GHI_start,

                GHI_end,

                GHI_max,

                east_limit,

                west_limit,

                actual_peak,

                tracking_peak,

                tracking_peak_error,

                tracking_peak_error_pct,

                tracking_block_error,

                tracking_energy_error,

                tracking_score

            ]

        }).to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )


    st.download_button(
        "⬇️ Download Tracking Results",
        data=output.getvalue(),
        file_name="VCast_Tracking_Loss_Correction.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )
