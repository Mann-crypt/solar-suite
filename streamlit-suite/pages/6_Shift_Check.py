# ============================================================
# STREAMLIT APP
# FORECAST SHIFT MAPE ANALYZER
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
    page_title="Forecast Shift MAPE",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 30px;
            font-weight: 700;
            margin-bottom: 2px;
        }

        .sub-title {
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }

        .result-box {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 10px;
            padding: 18px;
            text-align: center;
        }

        .result-label {
            font-size: 13px;
            color: #666;
        }

        .result-value {
            font-size: 28px;
            font-weight: 700;
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
    '<div class="sub-title">'
    'Compare Actual Data with a Forecast and interactively shift '
    'the forecast left or right.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# FUNCTIONS
# ============================================================

def read_uploaded_file(uploaded_file):
    """
    Read CSV/XLSX safely.
    """

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if file_name.endswith(".csv"):

        try:
            return pd.read_csv(
                io.BytesIO(file_bytes)
            )

        except UnicodeDecodeError:

            return pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="latin1"
            )

    if file_name.endswith((".xlsx", ".xls")):

        return pd.read_excel(
            io.BytesIO(file_bytes)
        )

    raise ValueError(
        "Unsupported file type. Please upload CSV or Excel."
    )


def make_unique_columns(df):
    """
    Make duplicate column names unique.
    """

    df = df.copy()

    new_columns = []
    seen = {}

    for col in df.columns:

        name = str(col).strip()

        if not name:
            name = "Column"

        if name not in seen:

            seen[name] = 0
            new_columns.append(name)

        else:

            seen[name] += 1
            new_columns.append(
                f"{name}_{seen[name]}"
            )

    df.columns = new_columns

    return df


def numeric_columns(df):
    """
    Return columns containing at least one numeric value.
    """

    cols = []

    for col in df.columns:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.notna().any():
            cols.append(col)

    return cols


def calculate_mape(actual, forecast):
    """
    Calculate MAPE.

    Excludes:
        - missing Actual
        - missing Forecast
        - Actual == 0
        - infinite values
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

    ape = ape.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if ape.empty:
        return np.nan, 0

    return float(ape.mean()), len(ape)


def shift_series(series, direction, shift):
    """
    Shift only the Forecast series.

    Left:
        shift(-N)

    Right:
        shift(+N)
    """

    shift = int(shift)

    if shift == 0:
        return series.copy()

    if direction == "Left":
        return series.shift(-shift)

    return series.shift(shift)


def find_datetime_column(df):
    """
    Try to identify a datetime column.
    """

    preferred = [
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

    for name in preferred:

        if name in lower_map:
            return lower_map[name]

    return None


def get_default_actual(columns):
    """
    Prefer common Actual columns.
    """

    priorities = [
        "SEMS",
        "Green Gen Meter",
        "Green Gen SCADA",
        "Actual GHI",
    ]

    for priority in priorities:

        priority_lower = priority.lower()

        # Exact match
        for col in columns:

            if str(col).strip().lower() == priority_lower:
                return col

        # Partial match
        for col in columns:

            if priority_lower in str(col).lower():
                return col

    return columns[0] if columns else None


def get_default_forecast(columns, actual_column):
    """
    Prefer columns that look like Forecast.
    """

    keywords = [
        "forecast",
        "forecaster",
        "ecm",
        "prediction",
        "pred",
    ]

    for col in columns:

        if col == actual_column:
            continue

        text = str(col).lower()

        if any(
            keyword in text
            for keyword in keywords
        ):
            return col

    for col in columns:

        if col != actual_column:
            return col

    return None


def downsample_for_plot(df, max_points=12000):
    """
    Downsample only for visual rendering.

    MAPE calculations always use the complete dataset.
    """

    if len(df) <= max_points:
        return df

    step = int(
        np.ceil(
            len(df) / max_points
        )
    )

    return df.iloc[::step].copy()


# ============================================================
# INPUT METHOD
# ============================================================

st.subheader("1. Data Input")

input_method = st.radio(
    "Select input method",
    [
        "Upload File",
        "Manual Entry",
    ],
    horizontal=True,
)


# ============================================================
# UPLOAD FILE
# ============================================================

if input_method == "Upload File":

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV or Excel file to begin."
        )

        st.stop()

    try:

        df = read_uploaded_file(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to read the file: {e}"
        )

        st.stop()


# ============================================================
# MANUAL ENTRY
# ============================================================

else:

    st.caption(
        "Paste or type your data directly into the table. "
        "You can add/remove rows and columns."
    )

    default_manual_data = pd.DataFrame(
        {
            "Timestamp": ["", "", "", "", ""],
            "SEMS": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "Green Gen Meter": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "Green Gen SCADA": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "Actual GHI": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "Forecast": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )

    df = st.data_editor(
        default_manual_data,
        num_rows="dynamic",
        width="stretch",
        height=350,
        key="manual_dataframe",
    )


# ============================================================
# BASIC DATA CLEANING
# ============================================================

if df is None:

    st.stop()


df = df.copy()

df = make_unique_columns(df)

# Remove completely empty columns
df = df.dropna(
    axis=1,
    how="all"
)


if df.empty:

    st.warning(
        "There is no usable data yet."
    )

    st.stop()


# ============================================================
# DATAFRAME INFORMATION
# ============================================================

st.subheader("2. Data")

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Rows",
        f"{len(df):,}"
    )

with info2:

    st.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

with info3:

    numeric_count = len(
        numeric_columns(df)
    )

    st.metric(
        "Numeric Columns",
        f"{numeric_count:,}"
    )


# ============================================================
# PREVIEW FOR UPLOADED FILE
# ============================================================

if input_method == "Upload File":

    with st.expander(
        "View DataFrame",
        expanded=False,
    ):

        st.dataframe(
            df.head(100),
            width="stretch",
            height=350,
        )


# ============================================================
# NUMERIC COLUMNS
# ============================================================

num_cols = numeric_columns(df)


if len(num_cols) < 2:

    st.error(
        "At least two numeric columns are required "
        "for Actual and Forecast."
    )

    st.stop()


# ============================================================
# COLUMN SELECTION
# ============================================================

st.subheader("3. Select Data")

actual_default = get_default_actual(
    num_cols
)

forecast_default = get_default_forecast(
    num_cols,
    actual_default
)


select1, select2 = st.columns(2)


with select1:

    actual_column = st.selectbox(
        "Actual Data",
        options=num_cols,
        index=(
            num_cols.index(actual_default)
            if actual_default in num_cols
            else 0
        ),
    )


with select2:

    forecast_options = [
        col
        for col in num_cols
        if col != actual_column
    ]

    if not forecast_options:

        st.error(
            "No other numeric column is available "
            "for Forecast."
        )

        st.stop()

    if forecast_default not in forecast_options:

        forecast_default = forecast_options[0]

    forecast_column = st.selectbox(
        "Forecast Data",
        options=forecast_options,
        index=forecast_options.index(
            forecast_default
        ),
    )


# ============================================================
# SHIFT CONTROL
# ============================================================

st.subheader("4. Forecast Shift")

shift_col1, shift_col2 = st.columns(2)


with shift_col1:

    direction = st.radio(
        "Shift Forecaster",
        [
            "Left",
            "Right",
        ],
        horizontal=True,
        help=(
            "Left moves the forecast earlier. "
            "Right moves the forecast later."
        ),
    )


with shift_col2:

    shift = st.number_input(
        "Shift (Blocks)",
        min_value=0,
        max_value=10000,
        value=0,
        step=1,
        help=(
            "Enter the number of blocks by which "
            "the forecast should be shifted."
        ),
    )


# ============================================================
# BLOCK DURATION
# ============================================================

block_minutes = st.number_input(
    "Block Duration (minutes)",
    min_value=1,
    max_value=1440,
    value=15,
    step=1,
    help=(
        "Example: 15 for 15-minute solar forecast data."
    ),
)


# ============================================================
# SERIES
# ============================================================

actual = pd.to_numeric(
    df[actual_column],
    errors="coerce"
)

forecast = pd.to_numeric(
    df[forecast_column],
    errors="coerce"
)


# ============================================================
# SHIFT FORECAST
# ============================================================

shifted_forecast = shift_series(
    forecast,
    direction,
    int(shift),
)


# ============================================================
# MAPE
# ============================================================

mape, valid_points = calculate_mape(
    actual,
    shifted_forecast,
)


# ============================================================
# TIME SHIFT TEXT
# ============================================================

total_minutes = int(
    shift * block_minutes
)

if total_minutes == 0:

    shift_text = "No Shift"

else:

    hours = total_minutes // 60
    minutes = total_minutes % 60

    if hours and minutes:

        duration_text = (
            f"{hours}h {minutes}m"
        )

    elif hours:

        duration_text = (
            f"{hours}h"
        )

    else:

        duration_text = (
            f"{minutes}m"
        )

    shift_text = (
        f"{duration_text} {direction}"
    )


# ============================================================
# RESULT
# ============================================================

st.subheader("5. Result")

result1, result2, result3 = st.columns(3)


with result1:

    if pd.isna(mape):

        mape_text = "N/A"

    else:

        mape_text = f"{mape:.2f}%"

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">MAPE</div>
            <div class="result-value">{mape_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with result2:

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Forecast Shift</div>
            <div class="result-value">{shift_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with result3:

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Valid Points</div>
            <div class="result-value">{valid_points:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAPE WARNING
# ============================================================

if pd.isna(mape):

    st.warning(
        "MAPE cannot be calculated. Make sure both columns "
        "contain numeric values and Actual Data contains "
        "at least some non-zero values."
    )


# ============================================================
# DATA QUALITY
# ============================================================

zero_actual = int(
    (actual == 0).sum()
)

missing_actual = int(
    actual.isna().sum()
)

missing_forecast = int(
    forecast.isna().sum()
)


with st.expander(
    "Data Quality",
    expanded=False,
):

    q1, q2, q3 = st.columns(3)

    with q1:

        st.write(
            f"Actual missing: **{missing_actual:,}**"
        )

    with q2:

        st.write(
            f"Forecast missing: **{missing_forecast:,}**"
        )

    with q3:

        st.write(
            f"Actual = 0: **{zero_actual:,}**"
        )

    st.caption(
        "Rows where Actual = 0 are excluded from MAPE."
    )


# ============================================================
# GRAPH
# ============================================================

st.subheader("6. Actual vs Forecast")

datetime_column = find_datetime_column(
    df
)


if datetime_column is not None:

    x_values = pd.to_datetime(
        df[datetime_column],
        errors="coerce"
    )

    x_title = datetime_column

else:

    x_values = np.arange(
        len(df)
    )

    x_title = "Data Point"


plot_df = pd.DataFrame(
    {
        "X": x_values,
        "Actual": actual,
        "Original Forecast": forecast,
        "Shifted Forecast": shifted_forecast,
    }
)


plot_df = downsample_for_plot(
    plot_df
)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=plot_df["X"],
        y=plot_df["Actual"],
        mode="lines",
        name="Actual",
        connectgaps=False,
    )
)


fig.add_trace(
    go.Scatter(
        x=plot_df["X"],
        y=plot_df["Original Forecast"],
        mode="lines",
        name="Original Forecast",
        connectgaps=False,
    )
)


fig.add_trace(
    go.Scatter(
        x=plot_df["X"],
        y=plot_df["Shifted Forecast"],
        mode="lines",
        name="Shifted Forecast",
        connectgaps=False,
    )
)


fig.update_layout(
    height=550,
    hovermode="x unified",
    xaxis_title=x_title,
    yaxis_title="Value",
    margin=dict(
        l=40,
        r=30,
        t=40,
        b=40,
    ),
)


st.plotly_chart(
    fig,
    width="stretch",
)


# ============================================================
# CURRENT SETTINGS
# ============================================================

with st.expander(
    "Current Analysis Settings",
    expanded=False,
):

    st.write(
        f"**Actual:** `{actual_column}`"
    )

    st.write(
        f"**Forecast:** `{forecast_column}`"
    )

    st.write(
        f"**Direction:** `{direction}`"
    )

    st.write(
        f"**Shift:** `{shift}` blocks"
    )

    st.write(
        f"**Block duration:** `{block_minutes}` minutes"
    )

    st.write(
        f"**Total shift:** `{shift_text}`"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MAPE is calculated on the complete dataset. "
    "Only the displayed graph is downsampled for browser performance."
)
