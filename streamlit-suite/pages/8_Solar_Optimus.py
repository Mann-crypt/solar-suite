import streamlit as st
import pandas as pd
import numpy as np
import io

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="96-Block Percentile Calculator",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
            background-color: #fafafa;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e5e5e5;
            padding: 12px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 96-Block Percentile Calculator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Upload time-series data, reshape it into daily 96-block profiles, "
    "and calculate a percentile for every 15-minute time block."
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================

BLOCKS_PER_DAY = 96
MIN_VALUES_FOR_CALCULATION = 96

# ============================================================
# HELPER FUNCTIONS
# ============================================================


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes, file_name):
    """
    Read CSV or Excel file from bytes.
    """

    try:
        file_lower = file_name.lower()

        if file_lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))

        elif file_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))

        else:
            raise ValueError(
                "Unsupported file format. Please upload CSV or Excel."
            )

        return df

    except Exception as exc:
        raise RuntimeError(f"Unable to read the file: {exc}") from exc


def get_numeric_columns(df):
    """
    Return columns that contain at least one numeric value.
    """

    numeric_columns = []

    for column in df.columns:
        numeric_series = pd.to_numeric(df[column], errors="coerce")

        if numeric_series.notna().any():
            numeric_columns.append(column)

    return numeric_columns


def prepare_numeric_series(df, selected_column):
    """
    Convert selected column to numeric and remove invalid values.
    """

    numeric = pd.to_numeric(
        df[selected_column],
        errors="coerce",
    )

    valid_values = numeric.dropna().reset_index(drop=True)

    return valid_values


def calculate_percentile_profile(values, percentile):
    """
    Reshape values into (days, 96) and calculate percentile
    for each of the 96 time blocks.
    """

    total_values = len(values)

    complete_days = total_values // BLOCKS_PER_DAY

    usable_values = complete_days * BLOCKS_PER_DAY

    if complete_days == 0:
        return None, None, 0

    trimmed = values.iloc[:usable_values].to_numpy(dtype=float)

    reshaped = trimmed.reshape(
        complete_days,
        BLOCKS_PER_DAY,
    )

    percentile_values = np.percentile(
        reshaped,
        percentile,
        axis=0,
    )

    return reshaped, percentile_values, complete_days


def create_result_dataframe(percentile_values, percentile):
    """
    Create the final 96-block result dataframe.
    """

    block_numbers = np.arange(1, BLOCKS_PER_DAY + 1)

    time_labels = pd.date_range(
        start="00:00",
        periods=BLOCKS_PER_DAY,
        freq="15min",
    ).strftime("%H:%M")

    result = pd.DataFrame(
        {
            "Block": block_numbers,
            "Time": time_labels,
            f"P{percentile:g}": percentile_values,
        }
    )

    return result


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("1. Upload Data")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="The file should contain a column with 15-minute time-series data.",
)

if uploaded_file is None:
    st.info(
        "Upload a CSV or Excel file to start the percentile calculation."
    )

    st.stop()

# ============================================================
# READ FILE
# ============================================================

try:
    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        st.error("The uploaded file is empty.")
        st.stop()

    df = read_uploaded_file(
        file_bytes,
        uploaded_file.name,
    )

except Exception as exc:
    st.error(str(exc))
    st.stop()

# ============================================================
# BASIC FILE VALIDATION
# ============================================================

if df is None or df.empty:
    st.error("The uploaded file contains no data.")
    st.stop()

if len(df.columns) == 0:
    st.error("No columns were found in the uploaded file.")
    st.stop()

# Remove completely empty columns
df = df.dropna(axis=1, how="all")

if df.empty or len(df.columns) == 0:
    st.error("The file contains only empty columns.")
    st.stop()

# ============================================================
# FILE INFORMATION
# ============================================================

st.success(
    f"File loaded successfully: **{uploaded_file.name}**"
)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric(
        "Rows",
        f"{len(df):,}",
    )

with info_col2:
    st.metric(
        "Columns",
        f"{len(df.columns):,}",
    )

with info_col3:
    st.metric(
        "Potential numeric columns",
        f"{len(get_numeric_columns(df)):,}",
    )

# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# COLUMN SELECTION
# ============================================================

st.subheader("2. Select Data Column")

numeric_columns = get_numeric_columns(df)

if not numeric_columns:
    st.error(
        "No numeric data columns were detected. "
        "Please upload a file containing numeric time-series data."
    )
    st.stop()

selected_column = st.selectbox(
    "Select the column for percentile calculation",
    options=numeric_columns,
    index=0,
)

# ============================================================
# PREPARE DATA
# ============================================================

values = prepare_numeric_series(
    df,
    selected_column,
)

if values.empty:
    st.error(
        f"The selected column **{selected_column}** contains no valid numeric values."
    )
    st.stop()

total_valid_values = len(values)

# ============================================================
# DATA VALIDATION
# ============================================================

st.subheader("3. Data Validation")

complete_days = total_valid_values // BLOCKS_PER_DAY
remainder = total_valid_values % BLOCKS_PER_DAY

validation_col1, validation_col2, validation_col3 = st.columns(3)

with validation_col1:
    st.metric(
        "Valid values",
        f"{total_valid_values:,}",
    )

with validation_col2:
    st.metric(
        "Complete days",
        f"{complete_days:,}",
    )

with validation_col3:
    st.metric(
        "Values/day",
        f"{BLOCKS_PER_DAY}",
    )

if total_valid_values < MIN_VALUES_FOR_CALCULATION:
    st.error(
        f"At least {BLOCKS_PER_DAY} valid values are required "
        "to create one complete day."
    )
    st.stop()

if remainder != 0:

    discarded = remainder

    st.warning(
        f"The selected column contains **{total_valid_values:,}** valid values. "
        f"This represents **{complete_days} complete day(s)** plus "
        f"**{discarded} incomplete value(s)**."
    )

    st.info(
        f"The last incomplete day will be excluded. "
        f"Only the first **{complete_days * BLOCKS_PER_DAY:,}** values "
        f"will be used for the percentile calculation."
    )

else:

    st.success(
        f"Validation passed. "
        f"{complete_days:,} complete day(s) × {BLOCKS_PER_DAY} blocks/day."
    )

# ============================================================
# PERCENTILE CONTROL
# ============================================================

st.subheader("4. Percentile Selection")

percentile_col1, percentile_col2 = st.columns([2, 1])

with percentile_col1:

    percentile = st.slider(
        "Select percentile",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=0.5,
        help=(
            "The selected percentile is calculated independently "
            "for each of the 96 time blocks."
        ),
    )

with percentile_col2:

    st.metric(
        "Selected percentile",
        f"P{percentile:g}",
    )

# ============================================================
# CALCULATION
# ============================================================

reshaped_data, percentile_values, used_days = (
    calculate_percentile_profile(
        values,
        percentile,
    )
)

if reshaped_data is None or percentile_values is None:
    st.error(
        "Unable to create complete 96-block days from the selected data."
    )
    st.stop()

# ============================================================
# RESULT
# ============================================================

result_df = create_result_dataframe(
    percentile_values,
    percentile,
)

st.subheader("5. Percentile Result")

result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:
    st.metric(
        "Days used",
        f"{used_days:,}",
    )

with result_col2:
    st.metric(
        "Time blocks",
        "96",
    )

with result_col3:
    st.metric(
        "Percentile",
        f"P{percentile:g}",
    )

# ============================================================
# RESULT TABLE
# ============================================================

st.markdown(
    f"### P{percentile:g} Profile"
)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True,
    height=500,
)

# ============================================================
# CHART
# ============================================================

st.subheader("6. 96-Block Percentile Profile")

chart_df = result_df.set_index("Time")

st.line_chart(
    chart_df[f"P{percentile:g}"],
    use_container_width=True,
)

# ============================================================
# OPTIONAL DAILY MATRIX
# ============================================================

with st.expander(
    "View reshaped daily data (Days × 96 blocks)",
    expanded=False,
):

    block_columns = [
        f"Block_{i:02d}"
        for i in range(1, BLOCKS_PER_DAY + 1)
    ]

    daily_matrix = pd.DataFrame(
        reshaped_data,
        columns=block_columns,
    )

    daily_matrix.insert(
        0,
        "Day",
        np.arange(1, used_days + 1),
    )

    st.caption(
        f"Shape: {daily_matrix.shape[0]} days × "
        f"{daily_matrix.shape[1] - 1} time blocks"
    )

    st.dataframe(
        daily_matrix,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

# ============================================================
# DOWNLOAD
# ============================================================

st.subheader("7. Download Result")

download_df = result_df.copy()

csv_data = download_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label=f"⬇️ Download P{percentile:g} Profile CSV",
    data=csv_data,
    file_name=f"percentile_P{percentile:g}_96_blocks.csv",
    mime="text/csv",
    use_container_width=False,
)

# ============================================================
# METHODOLOGY
# ============================================================

with st.expander("Calculation methodology", expanded=False):

    st.markdown(
        """
        **Calculation logic**

        1. The selected column is converted to numeric values.
        2. Invalid/non-numeric values are removed.
        3. The valid values are grouped into complete days.
        4. Each day contains exactly **96 blocks**.
        5. Data is reshaped as:

        `Days × 96`

        6. The selected percentile is calculated independently for
        each of the 96 columns:

        `np.percentile(daily_matrix, percentile, axis=0)`

        7. The final output contains exactly **96 percentile values**,
        corresponding to:

        `00:00, 00:15, 00:30, ... , 23:45`

        If the number of valid values is not divisible by 96,
        the incomplete final day is excluded from the calculation.
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "96-block percentile calculation • 15-minute resolution • "
    "Incomplete days are automatically excluded"
)
