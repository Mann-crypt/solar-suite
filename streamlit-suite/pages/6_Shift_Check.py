# ============================================================
# STREAMLIT APP
# FORECAST SHIFT MAPE ANALYZER
#
# Features:
#   - CSV / XLSX upload
#   - Actual column selection
#   - Forecast column selection
#   - Left / Right forecast shifting
#   - Automatic MAPE calculation for every shift
#   - Best shift detection
#   - Interactive charts
#   - Zero-actual handling
#   - Lightweight / freeze-safe design
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Forecast Shift MAPE Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .sub-title {
        font-size: 15px;
        color: #666;
        margin-bottom: 20px;
    }

    .metric-card {
        padding: 16px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        background-color: rgba(128,128,128,0.04);
        text-align: center;
    }

    .metric-label {
        font-size: 13px;
        color: #666;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
        margin-top: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 Forecast Shift MAPE Analyzer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
    Compare any Actual Data column against any Forecast column and
    automatically determine how forecast shifting affects MAPE.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes, file_name):
    """
    Read uploaded CSV/XLSX into a DataFrame.

    Cached using file bytes and file name.
    This prevents unnecessary file parsing during reruns.
    """

    if not file_bytes:
        return None

    name = str(file_name).lower()

    try:

        if name.endswith(".csv"):

            # Try UTF-8 first
            try:
                df = pd.read_csv(io.BytesIO(file_bytes))
            except UnicodeDecodeError:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding="latin1"
                )

        elif name.endswith((".xlsx", ".xls")):

            df = pd.read_excel(
                io.BytesIO(file_bytes)
            )

        else:
            raise ValueError(
                "Unsupported file format. Please upload CSV or Excel."
            )

        return df

    except Exception as e:

        raise ValueError(
            f"Could not read file: {str(e)}"
        )


def clean_numeric_series(series):
    """
    Convert a column safely to numeric.
    Invalid values become NaN.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def calculate_mape(actual, forecast):
    """
    Calculate MAPE.

    Rows where:
        - Actual is NaN
        - Forecast is NaN
        - Actual == 0

    are excluded.

    Returns:
        mape
        valid_points
    """

    actual = pd.to_numeric(
        actual,
        errors="coerce"
    )

    forecast = pd.to_numeric(
        forecast,
        errors="coerce"
    )

    valid = (
        actual.notna()
        & forecast.notna()
        & (actual != 0)
    )

    if not valid.any():
        return np.nan, 0

    actual_valid = actual.loc[valid].astype(float)
    forecast_valid = forecast.loc[valid].astype(float)

    ape = (
        np.abs(
            (actual_valid - forecast_valid)
            / actual_valid
        )
        * 100
    )

    # Remove any unexpected infinite values
    ape = ape.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if ape.empty:
        return np.nan, 0

    return float(ape.mean()), int(len(ape))


def shift_forecast(series, direction, shift_blocks):
    """
    Shift forecast.

    Left:
        shift(-N)

    Right:
        shift(+N)

    None:
        no shift
    """

    if direction == "Left":

        return series.shift(
            -int(shift_blocks)
        )

    elif direction == "Right":

        return series.shift(
            int(shift_blocks)
        )

    return series.copy()


@st.cache_data(show_spinner=False)
def calculate_shift_analysis(
    actual_values,
    forecast_values,
    direction,
    max_shift
):
    """
    Calculate MAPE for every shift.

    This function is cached so repeated UI interactions
    do not unnecessarily recalculate the same result.
    """

    actual = pd.Series(
        actual_values,
        dtype="float64"
    )

    forecast = pd.Series(
        forecast_values,
        dtype="float64"
    )

    results = []

    # Always calculate zero shift
    shifts = range(
        0,
        int(max_shift) + 1
    )

    for shift in shifts:

        shifted = shift_forecast(
            forecast,
            direction,
            shift
        )

        mape, valid_points = calculate_mape(
            actual,
            shifted
        )

        results.append(
            {
                "Shift": shift,
                "Direction": (
                    "None"
                    if shift == 0
                    else direction
                ),
                "MAPE (%)": mape,
                "Valid Points": valid_points,
            }
        )

    result_df = pd.DataFrame(results)

    return result_df


def format_time_shift(
    shift,
    block_minutes,
    direction
):
    """
    Convert blocks into human-readable time.
    """

    total_minutes = int(
        shift * block_minutes
    )

    if total_minutes == 0:
        return "0 min"

    hours = total_minutes // 60
    minutes = total_minutes % 60

    if hours > 0 and minutes > 0:
        text = f"{hours}h {minutes}m"

    elif hours > 0:
        text = f"{hours}h"

    else:
        text = f"{minutes}m"

    if direction == "Left":
        return f"{text} Left"

    if direction == "Right":
        return f"{text} Right"

    return text


def detect_datetime_column(df):
    """
    Try to identify a datetime column automatically.
    """

    priority_names = [
        "timestamp",
        "datetime",
        "date time",
        "date_time",
        "date",
        "time",
    ]

    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in priority_names:

        if name in lower_map:
            return lower_map[name]

    # Secondary detection
    for col in df.columns:

        try:

            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio >= 0.80:
                return col

        except Exception:
            continue

    return None


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("1. Upload Data")

uploaded_file = st.file_uploader(
    "Upload your CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="The file should contain your Actual and Forecast columns.",
)


# ============================================================
# NO FILE STATE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel file to start the MAPE shift analysis."
    )

    st.markdown(
        """
        ### Expected Data

        Your DataFrame can contain columns such as:

        - Timestamp
        - Green Gen Meter
        - Green Gen SCADA
        - SEMS
        - Actual GHI
        - Forecast
        - ECM10
        - ECM11
        - Any other forecast columns

        The app does not assume which column is Actual or Forecast.
        You select them yourself.
        """
    )

    st.stop()


# ============================================================
# READ FILE
# ============================================================

try:

    file_bytes = uploaded_file.getvalue()

    df = read_uploaded_file(
        file_bytes,
        uploaded_file.name
    )

except Exception as e:

    st.error(
        f"❌ File reading failed: {str(e)}"
    )

    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

if df is None or df.empty:

    st.error(
        "The uploaded file contains no data."
    )

    st.stop()


# Remove completely empty columns
df = df.dropna(
    axis=1,
    how="all"
)


if df.empty:

    st.error(
        "The uploaded file contains no usable columns."
    )

    st.stop()


# ============================================================
# DATA SUMMARY
# ============================================================

st.subheader("2. Data Information")

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:
    st.metric(
        "Rows",
        f"{len(df):,}"
    )

with info_col2:
    st.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

with info_col3:

    numeric_count = df.select_dtypes(
        include=np.number
    ).shape[1]

    st.metric(
        "Numeric Columns",
        f"{numeric_count:,}"
    )

with info_col4:

    datetime_col = detect_datetime_column(df)

    st.metric(
        "Datetime Column",
        str(datetime_col)
        if datetime_col is not None
        else "Not detected"
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "Preview DataFrame",
    expanded=False
):

    st.dataframe(
        df.head(100),
        width="stretch",
        height=350,
    )


# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_columns = []

for col in df.columns:

    numeric_series = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    if numeric_series.notna().any():
        numeric_columns.append(col)


if len(numeric_columns) < 2:

    st.error(
        "At least two numeric columns are required: "
        "one Actual column and one Forecast column."
    )

    st.stop()


# ============================================================
# COLUMN SELECTION
# ============================================================

st.subheader("3. Select Actual and Forecast")

selection_col1, selection_col2 = st.columns(2)


# ------------------------------------------------------------
# Automatic Actual Column Suggestions
# ------------------------------------------------------------

actual_priority = [
    "SEMS",
    "Green Gen Meter",
    "Green Gen SCADA",
    "Actual GHI",
]


def find_matching_column(
    columns,
    target
):

    target_lower = target.lower()

    # Exact
    for col in columns:

        if str(col).strip().lower() == target_lower:
            return col

    # Contains
    for col in columns:

        if target_lower in str(col).strip().lower():
            return col

    return None


default_actual = None

for candidate in actual_priority:

    found = find_matching_column(
        numeric_columns,
        candidate
    )

    if found is not None:

        default_actual = found
        break


if default_actual is None:
    default_actual = numeric_columns[0]


# ------------------------------------------------------------
# Forecast suggestion
# ------------------------------------------------------------

forecast_keywords = [
    "forecast",
    "forecaster",
    "ecm",
    "pred",
    "prediction",
]


default_forecast = None

for col in numeric_columns:

    if col == default_actual:
        continue

    col_lower = str(col).lower()

    if any(
        keyword in col_lower
        for keyword in forecast_keywords
    ):

        default_forecast = col
        break


if default_forecast is None:

    for col in numeric_columns:

        if col != default_actual:
            default_forecast = col
            break


# ------------------------------------------------------------
# Actual selector
# ------------------------------------------------------------

with selection_col1:

    actual_column = st.selectbox(
        "Actual Data Column",
        options=numeric_columns,
        index=numeric_columns.index(
            default_actual
        ),
        help=(
            "Select the column that represents "
            "the actual measured data."
        ),
    )


# ------------------------------------------------------------
# Forecast selector
# ------------------------------------------------------------

with selection_col2:

    forecast_column = st.selectbox(
        "Forecast Data Column",
        options=numeric_columns,
        index=numeric_columns.index(
            default_forecast
        ),
        help=(
            "Select the forecast column that "
            "you want to evaluate."
        ),
    )


# ============================================================
# SAME COLUMN CHECK
# ============================================================

if actual_column == forecast_column:

    st.warning(
        "Actual and Forecast columns are the same. "
        "Please select two different columns."
    )

    st.stop()


# ============================================================
# SHIFT SETTINGS
# ============================================================

st.subheader("4. Forecast Shift Settings")

settings_col1, settings_col2, settings_col3 = st.columns(3)


with settings_col1:

    direction = st.selectbox(
        "Shift Forecaster",
        options=[
            "Left",
            "Right",
        ],
        index=0,
        help=(
            "Left moves the forecast earlier. "
            "Right moves the forecast later."
        ),
    )


with settings_col2:

    max_shift = st.number_input(
        "Maximum Shift (Blocks)",
        min_value=0,
        max_value=1000,
        value=8,
        step=1,
        help=(
            "The app will calculate MAPE from "
            "0 up to this number of blocks."
        ),
    )


with settings_col3:

    block_minutes = st.number_input(
        "Block Duration (Minutes)",
        min_value=1,
        max_value=1440,
        value=15,
        step=1,
        help=(
            "For 15-minute data use 15. "
            "For 5-minute data use 5, etc."
        ),
    )


# ============================================================
# EXPLANATION
# ============================================================

if direction == "Left":

    st.info(
        f"Left shift: forecast is moved earlier. "
        f"1 block = {block_minutes} minutes."
    )

else:

    st.info(
        f"Right shift: forecast is moved later. "
        f"1 block = {block_minutes} minutes."
    )


# ============================================================
# PREPARE SERIES
# ============================================================

actual_series = clean_numeric_series(
    df[actual_column]
)

forecast_series = clean_numeric_series(
    df[forecast_column]
)


# ============================================================
# ORIGINAL MAPE
# ============================================================

original_mape, original_valid_points = calculate_mape(
    actual_series,
    forecast_series
)


# ============================================================
# CALCULATE SHIFT ANALYSIS
# ============================================================

calculate_button = st.button(
    "🔍 Calculate Shift MAPE",
    type="primary",
    width="stretch",
)


if calculate_button:

    with st.spinner(
        "Calculating MAPE for all shifts..."
    ):

        results = calculate_shift_analysis(
            tuple(actual_series.tolist()),
            tuple(forecast_series.tolist()),
            direction,
            int(max_shift),
        )

    # Store results
    st.session_state["shift_results"] = results
    st.session_state["analysis_actual"] = actual_column
    st.session_state["analysis_forecast"] = forecast_column
    st.session_state["analysis_direction"] = direction
    st.session_state["analysis_block_minutes"] = block_minutes


# ============================================================
# DISPLAY RESULTS ONLY AFTER CALCULATION
# ============================================================

if "shift_results" not in st.session_state:

    st.stop()


results = st.session_state["shift_results"]


# ============================================================
# BEST SHIFT
# ============================================================

valid_results = results[
    results["MAPE (%)"].notna()
].copy()


if valid_results.empty:

    st.error(
        "MAPE could not be calculated. "
        "Check that Actual and Forecast contain valid numeric data "
        "and that Actual is not zero for all rows."
    )

    st.stop()


best_index = valid_results[
    "MAPE (%)"
].idxmin()


best_row = valid_results.loc[
    best_index
]


best_shift = int(
    best_row["Shift"]
)

best_mape = float(
    best_row["MAPE (%)"]
)

best_direction = str(
    best_row["Direction"]
)

best_valid_points = int(
    best_row["Valid Points"]
)


# ============================================================
# IMPROVEMENT
# ============================================================

if pd.notna(original_mape):

    improvement = (
        float(original_mape)
        - best_mape
    )

else:

    improvement = np.nan


if pd.notna(original_mape) and original_mape != 0:

    improvement_percent = (
        improvement
        / float(original_mape)
        * 100
    )

else:

    improvement_percent = np.nan


best_time = format_time_shift(
    best_shift,
    block_minutes,
    best_direction,
)


# ============================================================
# RESULTS HEADER
# ============================================================

st.subheader("5. MAPE Results")


# ============================================================
# METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    value = (
        f"{original_mape:.2f}%"
        if pd.notna(original_mape)
        else "N/A"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Original MAPE</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with metric2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Best MAPE</div>
            <div class="metric-value">{best_mape:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with metric3:

    if pd.notna(improvement):

        improvement_text = (
            f"{improvement:.2f} pp"
        )

    else:

        improvement_text = "N/A"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">MAPE Improvement</div>
            <div class="metric-value">{improvement_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with metric4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Best Shift</div>
            <div class="metric-value">{best_time}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# COMPARISON INFORMATION
# ============================================================

st.markdown("### Selected Comparison")

comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

with comparison_col1:

    st.write(
        f"**Actual:** `{actual_column}`"
    )

with comparison_col2:

    st.write(
        f"**Forecast:** `{forecast_column}`"
    )

with comparison_col3:

    st.write(
        f"**Direction:** `{direction}`"
    )


# ============================================================
# BEST SHIFT MESSAGE
# ============================================================

if best_shift == 0:

    st.success(
        f"Best result is the original forecast. "
        f"No shift is required. MAPE = {best_mape:.2f}%."
    )

else:

    if improvement > 0:

        st.success(
            f"Best shift: **{best_direction} {best_shift} blocks "
            f"({best_time})**. "
            f"MAPE improves from **{original_mape:.2f}%** "
            f"to **{best_mape:.2f}%**."
        )

    elif improvement < 0:

        st.warning(
            f"The best tested shift is **{best_direction} "
            f"{best_shift} blocks**, but it is worse than the "
            f"original forecast. Original MAPE = "
            f"**{original_mape:.2f}%**, shifted MAPE = "
            f"**{best_mape:.2f}%**."
        )

    else:

        st.info(
            "The best shift produces the same MAPE as the "
            "original forecast."
        )


# ============================================================
# MAPE VS SHIFT CHART
# ============================================================

st.subheader("6. MAPE vs Forecast Shift")


chart_df = results.copy()

chart_df["Time Shift (min)"] = (
    chart_df["Shift"]
    * int(block_minutes)
)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=chart_df["Shift"],
        y=chart_df["MAPE (%)"],
        mode="lines+markers",
        name="MAPE",
        hovertemplate=(
            "Shift: %{x} blocks"
            "<br>MAPE: %{y:.2f}%"
            "<extra></extra>"
        ),
    )
)


# Best point
fig.add_trace(
    go.Scatter(
        x=[best_shift],
        y=[best_mape],
        mode="markers",
        marker=dict(
            size=12,
        ),
        name="Best Shift",
        hovertemplate=(
            f"Best Shift: {best_shift} blocks"
            f"<br>MAPE: {best_mape:.2f}%"
            "<extra></extra>"
        ),
    )
)


# Original MAPE reference
if pd.notna(original_mape):

    fig.add_hline(
        y=float(original_mape),
        line_dash="dash",
        annotation_text=(
            f"Original MAPE: {original_mape:.2f}%"
        ),
        annotation_position="top right",
    )


fig.update_layout(
    xaxis_title="Shift (Blocks)",
    yaxis_title="MAPE (%)",
    hovermode="x unified",
    height=500,
    margin=dict(
        l=40,
        r=40,
        t=50,
        b=40,
    ),
)


st.plotly_chart(
    fig,
    width="stretch",
)


# ============================================================
# SHIFT ANALYSIS TABLE
# ============================================================

st.subheader("7. Shift Analysis Table")


display_df = results.copy()

display_df["Time Shift"] = display_df.apply(
    lambda row: format_time_shift(
        int(row["Shift"]),
        int(block_minutes),
        str(row["Direction"]),
    ),
    axis=1,
)


display_df["MAPE (%)"] = display_df[
    "MAPE (%)"
].round(4)


display_df = display_df[
    [
        "Shift",
        "Direction",
        "Time Shift",
        "MAPE (%)",
        "Valid Points",
    ]
]


# Highlight best row
def highlight_best(row):

    if int(row["Shift"]) == best_shift:

        return [
            "font-weight: bold"
            for _ in row
        ]

    return [
        ""
        for _ in row
    ]


styled_df = display_df.style.apply(
    highlight_best,
    axis=1
)


st.dataframe(
    styled_df,
    width="stretch",
    height=400,
)


# ============================================================
# SHIFTED FORECAST
# ============================================================

st.subheader("8. Actual vs Forecast")


shifted_forecast = shift_forecast(
    forecast_series,
    direction,
    best_shift,
)


# ============================================================
# X AXIS
# ============================================================

datetime_col = detect_datetime_column(df)


if datetime_col is not None:

    x_values = pd.to_datetime(
        df[datetime_col],
        errors="coerce"
    )

    x_title = str(
        datetime_col
    )

else:

    x_values = np.arange(
        len(df)
    )

    x_title = "Data Point"


# ============================================================
# ACTUAL VS ORIGINAL FORECAST
# ============================================================

plot_df = pd.DataFrame(
    {
        "X": x_values,
        "Actual": actual_series,
        "Original Forecast": forecast_series,
        "Shifted Forecast": shifted_forecast,
    }
)


# Limit chart rows for browser performance
# while retaining the full calculations above.

MAX_CHART_POINTS = 20000

if len(plot_df) > MAX_CHART_POINTS:

    step = int(
        np.ceil(
            len(plot_df)
            / MAX_CHART_POINTS
        )
    )

    chart_plot_df = plot_df.iloc[
        ::step
    ].copy()

else:

    chart_plot_df = plot_df


fig2 = go.Figure()


fig2.add_trace(
    go.Scatter(
        x=chart_plot_df["X"],
        y=chart_plot_df["Actual"],
        mode="lines",
        name="Actual",
        connectgaps=False,
    )
)


fig2.add_trace(
    go.Scatter(
        x=chart_plot_df["X"],
        y=chart_plot_df["Original Forecast"],
        mode="lines",
        name="Original Forecast",
        connectgaps=False,
    )
)


fig2.add_trace(
    go.Scatter(
        x=chart_plot_df["X"],
        y=chart_plot_df["Shifted Forecast"],
        mode="lines",
        name=f"Shifted Forecast ({best_time})",
        connectgaps=False,
    )
)


fig2.update_layout(
    xaxis_title=x_title,
    yaxis_title="Value",
    hovermode="x unified",
    height=550,
    margin=dict(
        l=40,
        r=40,
        t=50,
        b=40,
    ),
)


st.plotly_chart(
    fig2,
    width="stretch",
)


# ============================================================
# CALCULATION DETAILS
# ============================================================

with st.expander(
    "MAPE Calculation Details",
    expanded=False,
):

    st.markdown(
        """
        ### MAPE Formula

        For every valid data point:

        **APE = |Actual - Forecast| / |Actual| × 100**

        Then:

        **MAPE = Mean(APE)**

        Rows are excluded when:

        - Actual is missing
        - Forecast is missing
        - Actual equals zero
        - Calculated APE is invalid/infinite

        ### Shift Logic

        **Left**

        The forecast is moved earlier:

        `forecast.shift(-N)`

        **Right**

        The forecast is moved later:

        `forecast.shift(+N)`

        The Actual Data column is never shifted.
        """
    )


# ============================================================
# DATA QUALITY
# ============================================================

with st.expander(
    "Data Quality Information",
    expanded=False,
):

    actual_missing = int(
        actual_series.isna().sum()
    )

    forecast_missing = int(
        forecast_series.isna().sum()
    )

    actual_zero = int(
        (actual_series == 0).sum()
    )

    st.write(
        f"**Actual column:** `{actual_column}`"
    )

    st.write(
        f"Missing Actual values: **{actual_missing:,}**"
    )

    st.write(
        f"Zero Actual values: **{actual_zero:,}**"
    )

    st.write(
        f"**Forecast column:** `{forecast_column}`"
    )

    st.write(
        f"Missing Forecast values: **{forecast_missing:,}**"
    )

    st.write(
        f"Original valid MAPE points: "
        f"**{original_valid_points:,}**"
    )

    st.write(
        f"Best-shift valid MAPE points: "
        f"**{best_valid_points:,}**"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Forecast Shift MAPE Analyzer"
)
