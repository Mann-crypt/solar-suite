# ============================================================
# VCAST LOSS CORRECTION
# FIXED / TRACKING
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


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
    "Fixed / Tracking forecast correction using peak-error based "
    "efficiency loss calculation."
)


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.segmented_control(
    "Plant Type",
    ["Fixed", "Tracking"],
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
        "Upload the VCast workbook to start the calculation."
    )

    st.stop()


# ============================================================
# CHECK VCAST WORKBOOK
# ============================================================

try:

    excel_file = pd.ExcelFile(
        uploaded_file
    )

    sheet_names = excel_file.sheet_names

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


if "Fixed-C11" not in sheet_names:

    st.error(
        "This workbook does not appear to be a VCast workbook. "
        "Required sheet 'Fixed-C11' was not found."
    )

    st.stop()


# ============================================================
# CONSTANTS
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
# HELPER
# ============================================================

def first_blank_position(df, column):

    valid = df[column].notna()

    if not valid.any():

        return len(df)

    blank_indices = np.where(
        ~valid.to_numpy()
    )[0]

    if len(blank_indices) == 0:

        return len(df)

    return blank_indices[0]


# ============================================================
# READ AREA & EFFICIENCY
# ============================================================

try:

    df_area = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12)
    )

except Exception as e:

    st.error(
        f"Unable to read Area & Efficiency sheet: {e}"
    )

    st.stop()


df_area.columns = (
    df_area.columns
    .astype(str)
    .str.replace(
        "*",
        "",
        regex=False
    )
    .str.strip()
)


if "S.No." not in df_area.columns:

    st.error(
        "Column 'S.No.' was not found in Area & Efficiency."
    )

    st.stop()


area_end = first_blank_position(
    df_area,
    "S.No."
)

df_area = df_area.iloc[
    :area_end
].copy()


# ============================================================
# NUMERIC COLUMNS
# ============================================================

for col in [
    "No of Module",
    "Area of 1 Module (m2)",
    "Standard PV Efficiency (%)"
]:

    if col in df_area.columns:

        df_area[col] = pd.to_numeric(
            df_area[col],
            errors="coerce"
        )


# ============================================================
# TOTAL AREA
# ============================================================

df_area["Total area (m2)"] = (
    df_area["No of Module"]
    *
    df_area["Area of 1 Module (m2)"]
)


# ============================================================
# READ CLUSTER AREA TABLE
# ============================================================

try:

    df_w = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15]
    )

except Exception as e:

    st.error(
        f"Unable to read cluster effective-area information: {e}"
    )

    st.stop()


df_w.columns = [
    "Clusters",
    "Original Eff Area"
]


cluster_end = first_blank_position(
    df_w,
    "Clusters"
)

df_w = df_w.iloc[
    :cluster_end
].copy()


df_w["Original Eff Area"] = pd.to_numeric(
    df_w["Original Eff Area"],
    errors="coerce"
).fillna(0)


# ============================================================
# READ FORECAST CONFIG
# ============================================================

try:

    df_config = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8
    )

    lat = float(
        df_config.loc[0, "Lat"]
    )

except Exception as e:

    st.error(
        f"Unable to read latitude from Forecast Config: {e}"
    )

    st.stop()


# ============================================================
# READ TILT
# ============================================================

try:

    df_tilt = pd.read_excel(
        uploaded_file,
        sheet_name="Config Tilt Angle",
        header=7
    )

except Exception as e:

    st.error(
        f"Unable to read Config Tilt Angle: {e}"
    )

    st.stop()


df_tilt.columns = (
    df_tilt.columns
    .astype(str)
    .str.strip()
)


if "Fixed" not in df_tilt.columns:

    st.error(
        "Fixed tilt information was not found."
    )

    st.stop()


tilt_end = first_blank_position(
    df_tilt,
    "Fixed"
)

df_tilt = df_tilt.iloc[
    :tilt_end
].copy()


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

try:

    df_ghi = pd.read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5]
    )

except Exception as e:

    st.error(
        f"Unable to read Result sheet: {e}"
    )

    st.stop()


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
# READ VCAST FIXED-C11
# ============================================================

try:

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name="Fixed-C11",
        header=1
    )

except Exception as e:

    st.error(
        f"Unable to read Fixed-C11: {e}"
    )

    st.stop()


df_fix.columns = (
    df_fix.columns
    .astype(str)
    .str.strip()
)


if "Date" not in df_fix.columns:

    st.error(
        "Column 'Date' was not found in Fixed-C11."
    )

    st.stop()


fix_end = first_blank_position(
    df_fix,
    "Date"
)

df_fix = df_fix.iloc[
    :fix_end
].copy()


if "Actual" not in df_fix.columns:

    st.error(
        "Column 'Actual' was not found in Fixed-C11."
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


df_fix = df_fix.iloc[
    :n
].copy()

df_ghi = df_ghi.iloc[
    :n
].copy()


if n == 0:

    st.error(
        "No valid VCast forecast rows found."
    )

    st.stop()


# ============================================================
# DATES
# ============================================================

dates = pd.to_datetime(
    df_fix["Date"],
    errors="coerce"
)


# ============================================================
# IMPORTANT:
# Your supplied calculation used today's date.
#
# We keep that same calculation logic here.
# ============================================================

calculation_date = (
    pd.Timestamp.today()
    .normalize()
)


df_fix["Date"] = (
    calculation_date
)


first_date = (
    calculation_date
    .replace(
        month=1,
        day=1
    )
    .normalize()
)


# ============================================================
# DECLINATION
# ============================================================

day_offset = (
    df_fix["Date"]
    - first_date
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


# ============================================================
# ELEVATION
# ============================================================

elevation = (
    90
    - lat
    + declination
)


# ============================================================
# TILT
# ============================================================

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


# ============================================================
# POA
# ============================================================

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


ghi_matrix = np.column_stack([
    df_ghi[col].to_numpy(
        dtype=float
    )
    for col in GHI_COLS
])


fixed_poa = (
    ghi_matrix
    *
    sin_ab[:, None]
    /
    sin_a_safe[:, None]
)


# ============================================================
# ACTUAL
# ============================================================

actual_full = (
    df_fix["Actual"]
    .to_numpy(
        dtype=float
    )
)


valid_mask = (
    np.isfinite(actual_full)
    &
    (actual_full != 0)
)


if not valid_mask.any():

    st.error(
        "Actual power contains no valid non-zero values."
    )

    st.stop()


actual_day = (
    actual_full[
        valid_mask
    ]
)


actual_peak = (
    actual_day.max()
)


if actual_peak <= 0:

    st.error(
        "Actual peak must be greater than zero."
    )

    st.stop()


# ============================================================
# FUNCTION:
# CALCULATE FIXED FORECAST
# ============================================================

def calculate_fixed(
    error_percent
):

    temp_df = df_area.copy()


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    temp_df["Error %"] = (
        error_percent
    )


    # --------------------------------------------------------
    # Net efficiency
    # --------------------------------------------------------

    temp_df[
        "Net Efficiency (%)"
    ] = (
        temp_df[
            "Standard PV Efficiency (%)"
        ]
        -
        temp_df["Error %"]
    )


    temp_df[
        "Net Efficiency (%)"
    ] = np.maximum(
        temp_df[
            "Net Efficiency (%)"
        ],
        0
    )


    # --------------------------------------------------------
    # Effective area
    # --------------------------------------------------------

    temp_df["Eff Area"] = (
        temp_df[
            "Net Efficiency (%)"
        ]
        *
        temp_df[
            "Total area (m2)"
        ]
        /
        100
    )


    # --------------------------------------------------------
    # Cluster effective areas
    # --------------------------------------------------------

    cluster_sums = (
        temp_df
        .groupby(
            "Clusters"
        )["Eff Area"]
        .sum()
    )


    weights = (
        df_w["Clusters"]
        .map(
            cluster_sums
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )


    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    power_matrix = (
        fixed_poa
        *
        weights[None, :]
        /
        1_000_000
    )


    forecast = (
        power_matrix.sum(
            axis=1
        )
    )


    return (
        forecast,
        power_matrix,
        temp_df,
        weights
    )


# ============================================================
# AUTOMATIC FIXED ERROR %
# ============================================================

fixed_test_results = []


for error in np.arange(
    0,
    10.01,
    0.1
):

    (
        test_forecast,
        _,
        _,
        _
    ) = calculate_fixed(
        error
    )


    test_day = (
        test_forecast[
            valid_mask
        ]
    )


    calculated_peak = (
        test_day.max()
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


    fixed_test_results.append({

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


fixed_results_df = pd.DataFrame(
    fixed_test_results
)


best_fixed_row = (
    fixed_results_df.loc[
        fixed_results_df[
            "Peak Error"
        ].idxmin()
    ]
)


automatic_fixed_error = float(
    best_fixed_row[
        "Error %"
    ]
)


# ============================================================
# TRACKING DATA
# ============================================================

tracking_available = all(
    sheet in sheet_names
    for sheet in [
        "Backend Cal C11",
        "Backend Cal C12",
        "Backend Cal C13",
        "Backend Cal C14",
        "Backend Cal C15",
        "Tracking"
    ]
)


# ============================================================
# TRACKING ERROR CALCULATION
#
# IMPORTANT:
# Error % is calculated FIRST.
# Then tracking parameters are optimized.
# ============================================================

def calculate_tracking_error(
    error_percent
):

    temp_df = df_area.copy()


    temp_df["Error %"] = (
        error_percent
    )


    temp_df[
        "Net Efficiency (%)"
    ] = (
        temp_df[
            "Standard PV Efficiency (%)"
        ]
        -
        error_percent
    )


    temp_df[
        "Net Efficiency (%)"
    ] = np.maximum(
        temp_df[
            "Net Efficiency (%)"
        ],
        0
    )


    temp_df["Eff Area"] = (
        temp_df[
            "Net Efficiency (%)"
        ]
        *
        temp_df[
            "Total area (m2)"
        ]
        /
        100
    )


    cluster_sums = (
        temp_df
        .groupby(
            "Clusters"
        )["Eff Area"]
        .sum()
    )


    weights = (
        df_w["Clusters"]
        .map(
            cluster_sums
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )


    return (
        temp_df,
        weights
    )


# ============================================================
# TRACKING RAW DATA
# ============================================================

if tracking_available:

    df_trac = pd.read_excel(
        uploaded_file,
        sheet_name="Tracking",
        header=1
    )


    df_trac = df_trac.iloc[
        :n
    ].copy()


    backend_list = []

    for cl in CLUSTERS:

        backend = pd.read_excel(
            uploaded_file,
            sheet_name=f"Backend Cal {cl}"
        )

        backend_list.append(
            backend
        )


    blocks = (
        backend_list[0][
            "Block No."
        ]
        .to_numpy(
            dtype=float
        )
    )

else:

    df_trac = None
    backend_list = []
    blocks = None


# ============================================================
# TRACKING FORECAST FUNCTION
# ============================================================

def calculate_tracking(
    error_percent,
    dhi_percent,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit
):

    (
        temp_df,
        weights
    ) = calculate_tracking_error(
        error_percent
    )


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
        /
        denominator_1
    )


    m2 = (
        90
        /
        denominator_2
    )


    zenith = np.where(

        blocks <= max_block,

        np.minimum(
            89,
            m1
            *
            (
                blocks
                -
                max_block
            )
        ),

        np.minimum(
            89,
            m2
            *
            (
                blocks
                -
                max_block
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
        np.radians(
            panel
        )
    )


    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None
    )


    dhi = (
        ghi_matrix
        *
        dhi_percent
        /
        100
    )


    dni = (
        ghi_matrix
        -
        dhi
    ) / cos_alpha[:, None]


    power_matrix = (
        dni
        *
        weights[None, :]
        /
        1_000_000
    )


    forecast = (
        power_matrix.sum(
            axis=1
        )
    )


    return {
        "forecast": forecast,
        "power_matrix": power_matrix,
        "zenith": zenith,
        "panel": panel,
        "dni": dni,
        "weights": weights,
        "area_df": temp_df
    }


# ============================================================
# AUTOMATIC TRACKING ERROR
#
# First determine the Error % by peak error.
# For this stage use default tracking parameters.
# ============================================================

if (
    plant_type == "Tracking"
    and tracking_available
):

    default_dhi = 5
    default_start = 20
    default_end = 72
    default_max = 50
    default_east = 60
    default_west = 60


    tracking_error_results = []


    for error in np.arange(
        0,
        10.01,
        0.1
    ):

        result_tracking = calculate_tracking(
            error,
            default_dhi,
            default_start,
            default_end,
            default_max,
            default_east,
            default_west
        )


        if result_tracking is None:

            continue


        prediction = (
            result_tracking[
                "forecast"
            ]
        )


        prediction_day = (
            prediction[
                valid_mask
            ]
        )


        if len(
            prediction_day
        ) == 0:

            continue


        calculated_peak = (
            prediction_day.max()
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


    tracking_error_df = pd.DataFrame(
        tracking_error_results
    )


    if not tracking_error_df.empty:

        best_tracking_error_row = (
            tracking_error_df.loc[
                tracking_error_df[
                    "Peak Error"
                ].idxmin()
            ]
        )


        automatic_tracking_error = float(
            best_tracking_error_row[
                "Error %"
            ]
        )

    else:

        automatic_tracking_error = (
            automatic_fixed_error
        )

else:

    tracking_error_df = pd.DataFrame()

    automatic_tracking_error = (
        automatic_fixed_error
    )


# ============================================================
# SESSION STATE
# ============================================================

if "vcast_fixed_error" not in st.session_state:

    st.session_state[
        "vcast_fixed_error"
    ] = automatic_fixed_error


if "vcast_tracking_error" not in st.session_state:

    st.session_state[
        "vcast_tracking_error"
    ] = automatic_tracking_error


# ============================================================
# PARAMETERS
# ============================================================

st.divider()

st.subheader(
    "Model Parameters"
)


if plant_type == "Fixed":

    col1, col2, col3 = st.columns(3)


    with col1:

        fixed_error = st.number_input(
            "Error %",
            min_value=0.0,
            max_value=20.0,
            value=float(
                st.session_state[
                    "vcast_fixed_error"
                ]
            ),
            step=0.1,
            key="fixed_error_input"
        )


    with col2:

        st.metric(
            "Automatic Error %",
            f"{automatic_fixed_error:.2f}%"
        )


    with col3:

        st.metric(
            "Actual Peak",
            f"{actual_peak:.4f} MW"
        )


    current_error = (
        fixed_error
    )


else:

    col1, col2, col3 = st.columns(3)


    with col1:

        tracking_error = st.number_input(
            "Error %",
            min_value=0.0,
            max_value=20.0,
            value=float(
                st.session_state[
                    "vcast_tracking_error"
                ]
            ),
            step=0.1,
            key="tracking_error_input"
        )


    with col2:

        st.metric(
            "Automatic Error %",
            f"{automatic_tracking_error:.2f}%"
        )


    with col3:

        st.metric(
            "Actual Peak",
            f"{actual_peak:.4f} MW"
        )


    current_error = (
        tracking_error
    )


# ============================================================
# FIXED MODEL
# ============================================================

if plant_type == "Fixed":

    (
        fixed_forecast,
        fixed_power_matrix,
        final_area_df,
        fixed_weights
    ) = calculate_fixed(
        current_error
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    fixed_day = (
        fixed_forecast[
            valid_mask
        ]
    )


    fixed_peak = (
        fixed_day.max()
    )


    peak_error = abs(
        fixed_peak
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


    block_error = (
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


    energy_error = (
        abs(
            actual_day.sum()
            -
            fixed_day.sum()
        )
        /
        actual_day.sum()
    )


    # --------------------------------------------------------
    # Metrics UI
    # --------------------------------------------------------

    st.subheader(
        "Fixed Model"
    )


    m1, m2, m3, m4 = st.columns(4)


    m1.metric(
        "Peak Power",
        f"{fixed_peak:.4f} MW"
    )


    m2.metric(
        "Peak Error",
        f"{peak_error:.4f} MW"
    )


    m3.metric(
        "Peak Error %",
        f"{peak_error_pct:.3f}%"
    )


    m4.metric(
        "Energy Error",
        f"{energy_error * 100:.3f}%"
    )


    # --------------------------------------------------------
    # Area table
    # --------------------------------------------------------

    st.subheader(
        "Cluster Effective Areas"
    )


    area_display = pd.DataFrame({

        "Cluster":
            df_w["Clusters"].to_numpy(),

        "Effective Area (m²)":
            fixed_weights

    })


    st.dataframe(
        area_display,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Efficiency table
    # --------------------------------------------------------

    efficiency_display = final_area_df[
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
        "Efficiency & Area"
    )


    st.dataframe(
        efficiency_display,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Chart
    # --------------------------------------------------------

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual_full,
            name="Actual",
            mode="lines"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=fixed_forecast,
            name="Fixed Forecast",
            mode="lines"
        )
    )


    fig.update_layout(
        title="Actual vs Fixed Forecast",
        xaxis_title="Block",
        yaxis_title="Power (MW)",
        height=450,
        hovermode="x unified"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Error test
    # --------------------------------------------------------

    with st.expander(
        "Error % Test Results"
    ):

        st.dataframe(
            fixed_results_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TRACKING MODEL
# ============================================================

else:

    if not tracking_available:

        st.error(
            "Tracking sheets were not found in this VCast workbook."
        )

        st.stop()


    st.subheader(
        "Tracking Parameters"
    )


    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    default_dhi = 5
    default_start = 20
    default_end = 72
    default_max = 50
    default_east = 60
    default_west = 60


    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if "tracking_dhi" not in st.session_state:

        st.session_state[
            "tracking_dhi"
        ] = default_dhi


    if "tracking_start" not in st.session_state:

        st.session_state[
            "tracking_start"
        ] = default_start


    if "tracking_end" not in st.session_state:

        st.session_state[
            "tracking_end"
        ] = default_end


    if "tracking_max" not in st.session_state:

        st.session_state[
            "tracking_max"
        ] = default_max


    if "tracking_east" not in st.session_state:

        st.session_state[
            "tracking_east"
        ] = default_east


    if "tracking_west" not in st.session_state:

        st.session_state[
            "tracking_west"
        ] = default_west


    # --------------------------------------------------------
    # EDITABLE PARAMETERS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)


    with c1:

        dhi_value = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            value=int(
                st.session_state[
                    "tracking_dhi"
                ]
            ),
            step=1,
            key="tracking_dhi_input"
        )


        start_value = st.number_input(
            "GHI Starting Block",
            min_value=1,
            max_value=96,
            value=int(
                st.session_state[
                    "tracking_start"
                ]
            ),
            step=1,
            key="tracking_start_input"
        )


    with c2:

        end_value = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                st.session_state[
                    "tracking_end"
                ]
            ),
            step=1,
            key="tracking_end_input"
        )


        max_value = st.number_input(
            "GHI Max Block",
            min_value=1,
            max_value=96,
            value=int(
                st.session_state[
                    "tracking_max"
                ]
            ),
            step=1,
            key="tracking_max_input"
        )


    with c3:

        east_value = st.number_input(
            "East Tracking Limit (°)",
            min_value=0,
            max_value=90,
            value=int(
                st.session_state[
                    "tracking_east"
                ]
            ),
            step=1,
            key="tracking_east_input"
        )


        west_value = st.number_input(
            "West Tracking Limit (°)",
            min_value=0,
            max_value=90,
            value=int(
                st.session_state[
                    "tracking_west"
                ]
            ),
            step=1,
            key="tracking_west_input"
        )


    # --------------------------------------------------------
    # PARAMETER VALIDATION
    # --------------------------------------------------------

    if not (
        start_value
        < max_value
        < end_value
    ):

        st.error(
            "GHI Starting Block must be less than "
            "GHI Max Block, and GHI Max Block must be "
            "less than GHI Ending Block."
        )

        st.stop()


    # --------------------------------------------------------
    # CALCULATE TRACKING
    # --------------------------------------------------------

    tracking_result = calculate_tracking(

        current_error,

        dhi_value,

        start_value,

        end_value,

        max_value,

        east_value,

        west_value

    )


    if tracking_result is None:

        st.error(
            "Unable to calculate Tracking forecast "
            "with the selected parameters."
        )

        st.stop()


    tracking_forecast = (
        tracking_result[
            "forecast"
        ]
    )


    tracking_power_matrix = (
        tracking_result[
            "power_matrix"
        ]
    )


    zenith = (
        tracking_result[
            "zenith"
        ]
    )


    panel = (
        tracking_result[
            "panel"
        ]
    )


    tracking_area_df = (
        tracking_result[
            "area_df"
        ]
    )


    tracking_day = (
        tracking_forecast[
            valid_mask
        ]
    )


    tracking_peak = (
        tracking_day.max()
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


    tracking_energy_error = (
        abs(
            actual_day.sum()
            -
            tracking_day.sum()
        )
        /
        actual_day.sum()
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    st.subheader(
        "Tracking Model"
    )


    m1, m2, m3, m4 = st.columns(4)


    m1.metric(
        "Peak Power",
        f"{tracking_peak:.4f} MW"
    )


    m2.metric(
        "Peak Error",
        f"{tracking_peak_error:.4f} MW"
    )


    m3.metric(
        "Peak Error %",
        f"{tracking_peak_error_pct:.3f}%"
    )


    m4.metric(
        "Energy Error",
        f"{tracking_energy_error * 100:.3f}%"
    )


    # --------------------------------------------------------
    # Automatic Error
    # --------------------------------------------------------

    st.info(
        f"Automatic Error % based on minimum Peak Error: "
        f"{automatic_tracking_error:.2f}%"
    )


    # --------------------------------------------------------
    # Tracking calculation table
    # --------------------------------------------------------

    tracking_display = pd.DataFrame({

        "Parameter": [

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

            "Peak Error (%)",

            "Block Error",

            "Energy Error"

        ],

        "Value": [

            current_error,

            dhi_value,

            start_value,

            end_value,

            max_value,

            east_value,

            west_value,

            actual_peak,

            tracking_peak,

            tracking_peak_error,

            tracking_peak_error_pct,

            tracking_block_error,

            tracking_energy_error

        ]

    })


    st.dataframe(
        tracking_display,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Zenith / Panel data
    # --------------------------------------------------------

    angle_display = pd.DataFrame({

        "Block":
            blocks,

        "Zenith Angle":
            zenith,

        "Panel Angle":
            panel

    })


    with st.expander(
        "Tracking Angle Calculation"
    ):

        st.dataframe(
            angle_display,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # Efficiency / Area
    # --------------------------------------------------------

    tracking_efficiency_display = tracking_area_df[
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
        "Tracking Efficiency & Area"
    )


    st.dataframe(
        tracking_efficiency_display,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Chart
    # --------------------------------------------------------

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual_full,
            name="Actual",
            mode="lines"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=tracking_forecast,
            name="Tracking Forecast",
            mode="lines"
        )
    )


    fig.update_layout(
        title="Actual vs Tracking Forecast",
        xaxis_title="Block",
        yaxis_title="Power (MW)",
        height=450,
        hovermode="x unified"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Tracking Power By Cluster
    # --------------------------------------------------------

    tracking_power_display = pd.DataFrame({

        "Cluster": CLUSTERS,

        "Effective Area (m²)":
            tracking_result[
                "weights"
            ]

    })


    st.subheader(
        "Tracking Effective Area"
    )


    st.dataframe(
        tracking_power_display,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Error Test
    # --------------------------------------------------------

    if not tracking_error_df.empty:

        with st.expander(
            "Tracking Error % Test Results"
        ):

            st.dataframe(
                tracking_error_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VCast calculation: Error % is selected using minimum peak "
    "error. All displayed model parameters remain editable."
)
