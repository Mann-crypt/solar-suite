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
# MULTIPLE COLUMN SELECTION
# ============================================================

st.subheader("2. Select Data Columns")

numeric_columns = get_numeric_columns(df)

if not numeric_columns:
    st.error(
        "No numeric data columns were detected. "
        "Please upload a file containing numeric time-series data."
    )
    st.stop()

selected_columns = st.multiselect(
    "Select one or more columns for percentile calculation",
    options=numeric_columns,
    default=[],
    help=(
        "You can select multiple numeric columns. "
        "A separate 96-block percentile profile will be "
        "calculated for every selected column."
    ),
)

if not selected_columns:
    st.info(
        "Please select at least one column to continue."
    )
    st.stop()


# ============================================================
# PERCENTILE CONTROL
# ============================================================

st.subheader("3. Percentile Selection")

percentile_col1, percentile_col2 = st.columns([3, 1])

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
# PROCESS ALL SELECTED COLUMNS
# ============================================================

st.subheader("4. Data Validation")

processed_data = {}
validation_rows = []

for column in selected_columns:

    # Convert to numeric
    numeric_values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    # Remove invalid values
    valid_values = numeric_values.dropna().reset_index(drop=True)

    total_values = len(valid_values)

    complete_days = total_values // BLOCKS_PER_DAY

    remainder = total_values % BLOCKS_PER_DAY

    usable_values = complete_days * BLOCKS_PER_DAY

    validation_rows.append(
        {
            "Column": column,
            "Valid Values": total_values,
            "Complete Days": complete_days,
            "Incomplete Values": remainder,
            "Usable Values": usable_values,
            "Status": (
                "Valid"
                if complete_days >= 1
                else "Insufficient data"
            ),
        }
    )

    if complete_days >= 1:

        trimmed_values = valid_values.iloc[
            :usable_values
        ].to_numpy(dtype=float)

        reshaped = trimmed_values.reshape(
            complete_days,
            BLOCKS_PER_DAY,
        )

        processed_data[column] = {
            "values": valid_values,
            "reshaped": reshaped,
            "days": complete_days,
            "remainder": remainder,
        }


# ============================================================
# VALIDATION TABLE
# ============================================================

validation_df = pd.DataFrame(validation_rows)

st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CHECK WHETHER ANY COLUMN CAN BE PROCESSED
# ============================================================

valid_columns = [
    column
    for column in selected_columns
    if column in processed_data
]

invalid_columns = [
    column
    for column in selected_columns
    if column not in processed_data
]


if invalid_columns:

    st.warning(
        "The following columns do not contain enough valid data "
        "for one complete 96-block day and will be excluded: "
        + ", ".join(invalid_columns)
    )


if not valid_columns:

    st.error(
        "None of the selected columns contains enough data "
        "to create a complete 96-block day."
    )

    st.stop()


# ============================================================
# INCOMPLETE DAY INFORMATION
# ============================================================

for column in valid_columns:

    info = processed_data[column]

    if info["remainder"] > 0:

        st.warning(
            f"**{column}** has {info['remainder']} incomplete "
            f"value(s) after {info['days']} complete day(s). "
            f"The incomplete portion will be excluded."
        )


# ============================================================
# CALCULATE PERCENTILE FOR EACH COLUMN
# ============================================================

result_data = {
    "Block": np.arange(
        1,
        BLOCKS_PER_DAY + 1,
    ),
    "Time": pd.date_range(
        start="00:00",
        periods=BLOCKS_PER_DAY,
        freq="15min",
    ).strftime("%H:%M"),
}


for column in valid_columns:

    reshaped = processed_data[column]["reshaped"]

    percentile_values = np.percentile(
        reshaped,
        percentile,
        axis=0,
    )

    result_data[
        f"{column} P{percentile:g}"
    ] = percentile_values


result_df = pd.DataFrame(result_data)


# ============================================================
# RESULT SUMMARY
# ============================================================

st.subheader("5. Percentile Result")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric(
        "Columns processed",
        len(valid_columns),
    )

with summary_col2:
    st.metric(
        "Time blocks",
        BLOCKS_PER_DAY,
    )

with summary_col3:
    st.metric(
        "Percentile",
        f"P{percentile:g}",
    )


# ============================================================
# RESULT TABLE
# ============================================================

st.markdown(
    f"### 96-Block P{percentile:g} Profile"
)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True,
    height=550,
)


# ============================================================
# GRAPH
# ============================================================

st.subheader("6. Percentile Profiles")

chart_data = result_df.set_index("Time")

profile_columns = [
    column
    for column in result_df.columns
    if column not in ["Block", "Time"]
]

st.line_chart(
    chart_data[profile_columns],
    use_container_width=True,
)


# ============================================================
# INDIVIDUAL COLUMN GRAPHS
# ============================================================

with st.expander(
    "View individual percentile profiles",
    expanded=False,
):

    for column in valid_columns:

        result_column = f"{column} P{percentile:g}"

        st.markdown(
            f"**{column} - P{percentile:g}**"
        )

        individual_chart = result_df.set_index(
            "Time"
        )[[result_column]]

        st.line_chart(
            individual_chart,
            use_container_width=True,
        )


# ============================================================
# DAILY MATRIX
# ============================================================

with st.expander(
    "View reshaped daily data",
    expanded=False,
):

    matrix_column = st.selectbox(
        "Select column to view its Days × 96 matrix",
        options=valid_columns,
        key="daily_matrix_column",
    )

    selected_matrix = processed_data[
        matrix_column
    ]["reshaped"]

    block_columns = [
        f"Block_{i:02d}"
        for i in range(1, BLOCKS_PER_DAY + 1)
    ]

    daily_matrix = pd.DataFrame(
        selected_matrix,
        columns=block_columns,
    )

    daily_matrix.insert(
        0,
        "Day",
        np.arange(
            1,
            len(daily_matrix) + 1,
        ),
    )

    st.caption(
        f"{matrix_column} shape: "
        f"{daily_matrix.shape[0]} days × "
        f"{BLOCKS_PER_DAY} blocks"
    )

    st.dataframe(
        daily_matrix,
        use_container_width=True,
        hide_index=True,
        height=500,
    )


# ============================================================
# DOWNLOAD RESULT
# ============================================================

st.subheader("7. Download Result")

download_csv = result_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label=f"⬇️ Download P{percentile:g} Result CSV",
    data=download_csv,
    file_name=(
        f"percentile_P{percentile:g}_96_block_profiles.csv"
    ),
    mime="text/csv",
)


# ============================================================
# CALCULATION DETAILS
# ============================================================

with st.expander(
    "Calculation methodology",
    expanded=False,
):

    st.markdown(
        f"""
        ### Calculation

        Each selected column is processed independently.

        For every column:

        **1.** Convert the column to numeric.

        **2.** Remove invalid/non-numeric values.

        **3.** Calculate the number of complete days:

        `Complete Days = Valid Values // 96`

        **4.** Remove any incomplete final day.

        **5.** Reshape the data:

        `Days × 96`

        **6.** Calculate the selected percentile across days:

        `np.percentile(data.reshape(-1, 96), {percentile:g}, axis=0)`

        **7.** Return 96 values representing:

        `00:00 → 00:15 → 00:30 → ... → 23:45`

        The percentile can be changed using the slider above,
        and the table, charts and downloaded CSV automatically
        update to the new percentile.
        """
    )
