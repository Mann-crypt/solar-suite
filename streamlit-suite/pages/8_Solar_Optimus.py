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
# CONSTANTS
# ============================================================

BLOCKS_PER_DAY = 96


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            color: #666;
            font-size: 1rem;
            margin-bottom: 25px;
        }

        .result-info {
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
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
    "Upload your time-series file and calculate the selected percentile "
    "for all 96 daily 15-minute time blocks."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


@st.cache_data(show_spinner=False)
def read_file(file_bytes, file_name):
    """
    Read CSV or Excel file.
    """

    try:
        file_name_lower = file_name.lower()

        if file_name_lower.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes))

        if file_name_lower.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(file_bytes))

        if file_name_lower.endswith(".xls"):
            return pd.read_excel(io.BytesIO(file_bytes))

        raise ValueError(
            "Unsupported file type. Please upload CSV, XLSX or XLS."
        )

    except Exception as exc:
        raise RuntimeError(
            f"Unable to read uploaded file: {exc}"
        ) from exc


def numeric_columns(df):
    """
    Detect columns that contain numeric values.
    """

    columns = []

    for column in df.columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if converted.notna().any():
            columns.append(column)

    return columns


def calculate_column_percentile(
    series,
    percentile,
):
    """
    Convert one column into complete Days x 96 matrix
    and calculate percentile for every time block.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric.dropna().reset_index(drop=True)

    total_values = len(numeric)

    complete_days = total_values // BLOCKS_PER_DAY

    if complete_days == 0:
        return None, 0, total_values, 0

    usable_values = (
        complete_days * BLOCKS_PER_DAY
    )

    trimmed = numeric.iloc[
        :usable_values
    ].to_numpy(
        dtype=float
    )

    matrix = trimmed.reshape(
        complete_days,
        BLOCKS_PER_DAY,
    )

    percentile_result = np.percentile(
        matrix,
        percentile,
        axis=0,
    )

    remainder = (
        total_values
        - usable_values
    )

    return (
        percentile_result,
        complete_days,
        total_values,
        remainder,
    )


def create_time_labels():
    """
    Create 96 labels from 00:00 to 23:45.
    """

    return pd.date_range(
        start="00:00",
        periods=BLOCKS_PER_DAY,
        freq="15min",
    ).strftime("%H:%M").tolist()


# ============================================================
# UPLOAD
# ============================================================

st.subheader("1. Upload File")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
    help=(
        "Every numeric column will be processed independently. "
        "Non-numeric columns will be ignored."
    ),
)


if uploaded_file is None:

    st.info(
        "Please upload a CSV or Excel file to continue."
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

    df = read_file(
        file_bytes,
        uploaded_file.name,
    )

except Exception as exc:

    st.error(str(exc))
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
    how="all",
)


if df.empty:

    st.error(
        "The uploaded file contains only empty columns."
    )

    st.stop()


# ============================================================
# FILE SUMMARY
# ============================================================

st.success(
    f"File loaded successfully: **{uploaded_file.name}**"
)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:

    st.metric(
        "Rows",
        f"{len(df):,}",
    )

with summary_col2:

    st.metric(
        "Total columns",
        f"{len(df.columns):,}",
    )

detected_numeric_columns = numeric_columns(df)

with summary_col3:

    st.metric(
        "Numeric columns",
        f"{len(detected_numeric_columns):,}",
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "Preview uploaded data",
    expanded=False,
):

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DETECT COLUMNS
# ============================================================

st.subheader(
    "2. Columns Detected"
)

if not detected_numeric_columns:

    st.error(
        "No numeric columns were found in the uploaded file."
    )

    st.info(
        "Make sure your data columns contain numeric values."
    )

    st.stop()


st.write(
    f"**{len(detected_numeric_columns)} numeric columns** "
    "will be processed:"
)

st.write(
    ", ".join(
        str(col)
        for col in detected_numeric_columns
    )
)


non_numeric_columns = [
    column
    for column in df.columns
    if column not in detected_numeric_columns
]

if non_numeric_columns:

    st.caption(
        "Non-numeric columns will be ignored: "
        + ", ".join(
            str(col)
            for col in non_numeric_columns
        )
    )


# ============================================================
# PERCENTILE CONTROL
# ============================================================

st.subheader(
    "3. Select Percentile"
)

percentile_col1, percentile_col2 = st.columns(
    [3, 1]
)

with percentile_col1:

    percentile = st.slider(
        "Percentile",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=0.5,
        help=(
            "The selected percentile is calculated "
            "independently for every 15-minute block "
            "of every numeric column."
        ),
    )

with percentile_col2:

    st.metric(
        "Selected",
        f"P{percentile:g}",
    )


# ============================================================
# CALCULATION
# ============================================================

st.subheader(
    "4. Processing"
)

results = {}

validation_results = []

progress_bar = st.progress(
    0
)

status_text = st.empty()

total_columns = len(
    detected_numeric_columns
)

for index, column in enumerate(
    detected_numeric_columns
):

    status_text.text(
        f"Processing: {column}"
    )

    (
        percentile_values,
        complete_days,
        total_values,
        remainder,
    ) = calculate_column_percentile(
        df[column],
        percentile,
    )

    validation_results.append(
        {
            "Column": column,
            "Valid Values": total_values,
            "Complete Days": complete_days,
            "Incomplete Values": remainder,
            "Status": (
                "OK"
                if complete_days > 0
                else "Insufficient data"
            ),
        }
    )

    if percentile_values is not None:

        results[column] = percentile_values

    progress_bar.progress(
        (index + 1) / total_columns
    )


status_text.empty()
progress_bar.empty()


# ============================================================
# VALIDATION TABLE
# ============================================================

st.subheader(
    "5. Validation"
)

validation_df = pd.DataFrame(
    validation_results
)

st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CHECK IF ANY COLUMN CAN BE PROCESSED
# ============================================================

if not results:

    st.error(
        "None of the numeric columns contains enough "
        "data for at least one complete 96-block day."
    )

    st.stop()


# ============================================================
# WARN ABOUT INCOMPLETE DATA
# ============================================================

incomplete_columns = validation_df[
    validation_df["Incomplete Values"] > 0
]

if not incomplete_columns.empty:

    st.warning(
        f"{len(incomplete_columns)} column(s) contain "
        "incomplete final days. The incomplete values "
        "are excluded automatically."
    )


# ============================================================
# CREATE FINAL RESULT
# ============================================================

time_labels = create_time_labels()

result_df = pd.DataFrame(
    {
        "Block": np.arange(
            1,
            BLOCKS_PER_DAY + 1,
        ),
        "Time": time_labels,
    }
)


for column in detected_numeric_columns:

    if column in results:

        result_df[
            f"{column} | P{percentile:g}"
        ] = results[column]


# ============================================================
# RESULT SUMMARY
# ============================================================

st.subheader(
    "6. 96-Block Percentile Result"
)

result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:

    st.metric(
        "Processed columns",
        f"{len(results):,}",
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

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True,
    height=600,
)


# ============================================================
# CHART
# ============================================================

st.subheader(
    f"7. P{percentile:g} 96-Block Profile"
)

chart_columns = [
    column
    for column in result_df.columns
    if column not in ["Block", "Time"]
]

if chart_columns:

    chart_data = result_df.set_index(
        "Time"
    )[chart_columns]

    st.line_chart(
        chart_data,
        use_container_width=True,
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.subheader(
    "8. Download Result"
)

csv_data = result_df.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label=(
        f"⬇️ Download P{percentile:g} "
        "96-Block Result"
    ),
    data=csv_data,
    file_name=(
        f"96_block_percentile_P"
        f"{percentile:g}.csv"
    ),
    mime="text/csv",
)


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "Calculation methodology",
    expanded=False,
):

    st.markdown(
        f"""
        ### Calculation

        Every numeric column is processed independently.

        For each column:

        1. Convert values to numeric.
        2. Remove invalid/blank values.
        3. Divide the data into complete days.
        4. Each day contains exactly **96 blocks**.
        5. Reshape the data into:

        **(Days, 96)**

        6. Calculate **P{percentile:g}** along the day axis.

        ```text
        Daily Matrix

        Day 1    Block 1   Block 2   Block 3   ... Block 96
        Day 2    Block 1   Block 2   Block 3   ... Block 96
        Day 3    Block 1   Block 2   Block 3   ... Block 96
        ...
        Day N    Block 1   Block 2   Block 3   ... Block 96


        P{percentile:g}

        Block 1   → percentile of all Day values
        Block 2   → percentile of all Day values
        Block 3   → percentile of all Day values
        ...
        Block 96  → percentile of all Day values
        ```

        If a column does not contain a multiple of 96 valid
        values, the incomplete final day is excluded.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "96-block percentile calculator • "
    "15-minute resolution • "
    "All numeric columns processed independently"
)
