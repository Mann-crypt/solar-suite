````python
import io

import numpy as np
import pandas as pd
import streamlit as st


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
    '<div class="main-title">'
    "📊 96-Block Percentile Calculator"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Upload time-series data, select multiple columns, "
    "reshape the data into Days × 96 blocks, and calculate "
    "a selectable percentile for every 15-minute time block."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


@st.cache_data(show_spinner=False)
def read_uploaded_file(
    file_bytes,
    file_name,
):
    """
    Read CSV or Excel file from uploaded bytes.
    """

    try:

        file_lower = file_name.lower()

        if file_lower.endswith(".csv"):

            df = pd.read_csv(
                io.BytesIO(file_bytes)
            )

        elif file_lower.endswith(".xlsx"):

            df = pd.read_excel(
                io.BytesIO(file_bytes)
            )

        elif file_lower.endswith(".xls"):

            df = pd.read_excel(
                io.BytesIO(file_bytes)
            )

        else:

            raise ValueError(
                "Unsupported file format. "
                "Please upload CSV, XLSX, or XLS."
            )

    except Exception as exc:

        raise RuntimeError(
            f"Unable to read the file: {exc}"
        ) from exc

    if df is None:

        raise ValueError(
            "The file could not be read."
        )

    if df.empty:

        raise ValueError(
            "The uploaded file contains no rows."
        )

    return df


def make_unique_columns(df):
    """
    Make duplicate column names unique.

    Example:
        Power, Power, Power

    becomes:
        Power, Power_2, Power_3
    """

    new_columns = []
    counts = {}

    for column in df.columns:

        column = str(column).strip()

        if column not in counts:

            counts[column] = 1
            new_columns.append(column)

        else:

            counts[column] += 1

            new_columns.append(
                f"{column}_{counts[column]}"
            )

    result = df.copy()

    result.columns = new_columns

    return result


def get_numeric_columns(df):
    """
    Return columns that contain at least one value
    that can be interpreted as numeric.
    """

    numeric_columns = []

    for column in df.columns:

        numeric_series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric_series.notna().any():

            numeric_columns.append(column)

    return numeric_columns


def process_column(
    series,
    percentile,
):
    """
    Process one selected column.

    Steps:

    1. Convert to numeric.
    2. Remaining non-numeric values become NaN.
    3. Replace those values with zero.
    4. Reshape into Days × 96.
    5. Calculate percentile for each block.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    # Treat any non-numeric value as zero.
    numeric = numeric.fillna(0)

    values = numeric.to_numpy(
        dtype=float
    )

    total_values = len(values)

    complete_days = (
        total_values // BLOCKS_PER_DAY
    )

    incomplete_values = (
        total_values % BLOCKS_PER_DAY
    )

    usable_values = (
        complete_days * BLOCKS_PER_DAY
    )

    if complete_days == 0:

        return None

    trimmed_values = values[
        :usable_values
    ]

    reshaped = trimmed_values.reshape(
        complete_days,
        BLOCKS_PER_DAY,
    )

    percentile_values = np.percentile(
        reshaped,
        percentile,
        axis=0,
    )

    return {
        "total_values": total_values,
        "complete_days": complete_days,
        "incomplete_values": incomplete_values,
        "usable_values": usable_values,
        "reshaped": reshaped,
        "percentile_values": percentile_values,
    }


def create_result_dataframe(
    processed_data,
    percentile,
):
    """
    Create the final 96-block result dataframe.
    """

    time_labels = pd.date_range(
        start="00:00",
        periods=BLOCKS_PER_DAY,
        freq="15min",
    ).strftime("%H:%M")

    result = pd.DataFrame(
        {
            "Block": np.arange(
                1,
                BLOCKS_PER_DAY + 1,
            ),

            "Time": time_labels,
        }
    )

    for column, data in processed_data.items():

        result[
            f"{column} P{percentile:g}"
        ] = data["percentile_values"]

    return result


def create_excel_file(
    result_df,
):
    """
    Convert result dataframe to Excel bytes.
    """

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        result_df.to_excel(
            writer,
            index=False,
            sheet_name="Percentile",
        )

    return output.getvalue()


# ============================================================
# 1. FILE UPLOAD
# ============================================================

st.subheader("1. Upload Data")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
    help=(
        "Upload a time-series file containing your "
        "15-minute data."
    ),
)


if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel file to start."
    )

    st.stop()


# ============================================================
# 2. READ FILE
# ============================================================

try:

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:

        st.error(
            "The uploaded file is empty."
        )

        st.stop()

    df = read_uploaded_file(
        file_bytes,
        uploaded_file.name,
    )

except Exception as exc:

    st.error(
        f"❌ {exc}"
    )

    st.stop()


# ============================================================
# 3. BASIC DATA CLEANING
# ============================================================

if df is None or df.empty:

    st.error(
        "The uploaded file contains no data."
    )

    st.stop()


# Convert column names to strings
df.columns = [
    str(column).strip()
    for column in df.columns
]


# Make duplicate columns unique
df = make_unique_columns(df)


# Remove completely empty columns
df = df.dropna(
    axis=1,
    how="all",
)


if df.empty or len(df.columns) == 0:

    st.error(
        "The file contains no usable columns."
    )

    st.stop()


# ============================================================
# 4. FILL NULL VALUES WITH ZERO
# ============================================================

# Zero is a valid value in this calculation.
#
# Every blank / NaN cell is therefore converted to 0
# before column selection and percentile calculation.

null_count_before = int(
    df.isna().sum().sum()
)

df = df.fillna(0)


# ============================================================
# FILE LOADED MESSAGE
# ============================================================

st.success(
    f"✅ File loaded successfully: "
    f"**{uploaded_file.name}**"
)


if null_count_before > 0:

    st.info(
        f"{null_count_before:,} blank/null value(s) "
        "were replaced with 0."
    )


# ============================================================
# 5. FILE INFORMATION
# ============================================================

numeric_columns = get_numeric_columns(df)


info_col1, info_col2, info_col3, info_col4 = st.columns(4)


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
        "Numeric Columns",
        f"{len(numeric_columns):,}",
    )


with info_col4:

    complete_days_overall = (
        len(df) // BLOCKS_PER_DAY
    )

    st.metric(
        "Complete 96-Block Days",
        f"{complete_days_overall:,}",
    )


# ============================================================
# 6. DATA PREVIEW
# ============================================================

with st.expander(
    "Preview uploaded data",
    expanded=False,
):

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 7. NUMERIC COLUMN CHECK
# ============================================================

if not numeric_columns:

    st.error(
        "❌ No numeric columns were detected."
    )

    st.write(
        "Please upload a file containing numeric "
        "time-series data."
    )

    st.stop()


# ============================================================
# 8. MULTIPLE COLUMN SELECTION
# ============================================================

st.subheader("2. Select Data Columns")

selected_columns = st.multiselect(
    "Select one or more columns",
    options=numeric_columns,
    default=[],
    help=(
        "Select multiple columns to calculate a separate "
        "96-block percentile profile for each column."
    ),
)


if not selected_columns:

    st.info(
        "Select at least one column to continue."
    )

    st.stop()


st.write(
    f"**{len(selected_columns)} column(s) selected:**"
)

st.caption(
    ", ".join(selected_columns)
)


# ============================================================
# 9. PERCENTILE CONTROL
# ============================================================

st.subheader("3. Percentile Selection")

percentile_col1, percentile_col2 = st.columns(
    [4, 1]
)


with percentile_col1:

    percentile = st.slider(
        "Select percentile",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=0.5,
        help=(
            "The selected percentile is calculated "
            "independently for each of the 96 blocks."
        ),
    )


with percentile_col2:

    st.metric(
        "Selected Percentile",
        f"P{percentile:g}",
    )


# ============================================================
# 10. PROCESS SELECTED COLUMNS
# ============================================================

st.subheader("4. Data Validation")

processed_data = {}

validation_rows = []


for column in selected_columns:

    result = process_column(
        df[column],
        percentile,
    )


    if result is None:

        validation_rows.append(
            {
                "Column": column,
                "Rows": len(df),
                "Complete Days": 0,
                "Usable Values": 0,
                "Incomplete Values": len(df),
                "Status": "Insufficient data",
            }
        )

        continue


    processed_data[column] = result


    validation_rows.append(
        {
            "Column": column,
            "Rows": result["total_values"],
            "Complete Days": result["complete_days"],
            "Usable Values": result["usable_values"],
            "Incomplete Values": result["incomplete_values"],
            "Status": "Valid",
        }
    )


validation_df = pd.DataFrame(
    validation_rows
)


# ============================================================
# 11. VALIDATION TABLE
# ============================================================

st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 12. INCOMPLETE DATA WARNING
# ============================================================

has_incomplete_data = False


for column, data in processed_data.items():

    if data["incomplete_values"] > 0:

        has_incomplete_data = True

        st.warning(
            f"**{column}:** "
            f"{data['incomplete_values']} row(s) remain "
            f"after the last complete 96-block day. "
            "These rows are excluded from the percentile calculation."
        )


if not has_incomplete_data:

    st.success(
        "All selected columns contain complete 96-block days."
    )


# ============================================================
# 13. STOP IF NO COLUMN CAN BE PROCESSED
# ============================================================

if not processed_data:

    st.error(
        "None of the selected columns contains at least "
        "96 rows."
    )

    st.stop()


# ============================================================
# 14. PROCESSING SUMMARY
# ============================================================

summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)


with summary_col1:

    st.metric(
        "Columns Processed",
        len(processed_data),
    )


with summary_col2:

    st.metric(
        "Blocks / Day",
        96,
    )


with summary_col3:

    minimum_days = min(
        data["complete_days"]
        for data in processed_data.values()
    )

    st.metric(
        "Complete Days",
        f"{minimum_days:,}",
    )


with summary_col4:

    st.metric(
        "Percentile",
        f"P{percentile:g}",
    )


# ============================================================
# 15. CREATE RESULT
# ============================================================

result_df = create_result_dataframe(
    processed_data,
    percentile,
)


# ============================================================
# 16. RESULT TABLE
# ============================================================

st.subheader("5. 96-Block Percentile Result")

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True,
    height=550,
)


# ============================================================
# 17. GRAPH
# ============================================================

st.subheader("6. Percentile Profile")

chart_df = result_df.set_index(
    "Time"
)


chart_columns = [
    column
    for column in result_df.columns
    if column not in [
        "Block",
        "Time",
    ]
]


if chart_columns:

    st.line_chart(
        chart_df[chart_columns],
        use_container_width=True,
    )


# ============================================================
# 18. INDIVIDUAL GRAPHS
# ============================================================

with st.expander(
    "View individual column profiles",
    expanded=False,
):

    for column in processed_data:

        result_column = (
            f"{column} P{percentile:g}"
        )


        st.markdown(
            f"#### {column}"
        )


        individual_df = (
            result_df
            .set_index("Time")
            [[result_column]]
        )


        st.line_chart(
            individual_df,
            use_container_width=True,
        )


# ============================================================
# 19. DAYS × 96 MATRIX
# ============================================================

st.subheader("7. Reshaped Days × 96 Data")


matrix_column = st.selectbox(
    "Select column to view its Days × 96 matrix",
    options=list(
        processed_data.keys()
    ),
)


matrix = processed_data[
    matrix_column
]["reshaped"]


st.write(
    f"**Selected column:** {matrix_column}"
)


st.caption(
    f"Matrix shape: "
    f"{matrix.shape[0]:,} days × "
    f"{matrix.shape[1]} blocks"
)


block_columns = [
    f"Block_{i:02d}"
    for i in range(
        1,
        BLOCKS_PER_DAY + 1,
    )
]


matrix_df = pd.DataFrame(
    matrix,
    columns=block_columns,
)


matrix_df.insert(
    0,
    "Day",
    np.arange(
        1,
        len(matrix_df) + 1,
    ),
)


st.dataframe(
    matrix_df,
    use_container_width=True,
    hide_index=True,
    height=500,
)


# ============================================================
# 20. DOWNLOAD
# ============================================================

st.subheader("8. Download Result")


download_col1, download_col2 = st.columns(2)


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

csv_data = result_df.to_csv(
    index=False
).encode("utf-8")


with download_col1:

    st.download_button(
        label=f"⬇️ Download P{percentile:g} CSV",
        data=csv_data,
        file_name=(
            f"96_block_percentile_P"
            f"{percentile:g}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ------------------------------------------------------------
# EXCEL
# ------------------------------------------------------------

try:

    excel_data = create_excel_file(
        result_df
    )


    with download_col2:

        st.download_button(
            label=f"⬇️ Download P{percentile:g} Excel",
            data=excel_data,
            file_name=(
                f"96_block_percentile_P"
                f"{percentile:g}.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

except Exception as exc:

    st.warning(
        f"Excel download unavailable: {exc}"
    )


# ============================================================
# 21. CALCULATION METHODOLOGY
# ============================================================

with st.expander(
    "Calculation methodology",
    expanded=False,
):

    st.markdown(
        """
        ### Data processing

        **Step 1: Upload**

        CSV, XLSX and XLS files are supported.

        **Step 2: Null handling**

        All blank / NaN values are replaced with **0**.

        Zero is treated as a valid value and is included in
        the percentile calculation.

        **Step 3: Column selection**

        One or more numeric columns can be selected.

        **Step 4: Reshape**

        Each selected column is reshaped into:

        ```
        Days × 96
        ```

        Each row represents one day and each column represents
        one 15-minute block.

        **Step 5: Percentile**

        The percentile is calculated independently for each
        15-minute block:

        ```
        np.percentile(
            daily_matrix,
            percentile,
            axis=0
        )
        ```

        **Step 6: Output**

        The result always contains 96 blocks:

        ```
        Block 1  → 00:00
        Block 2  → 00:15
        Block 3  → 00:30
        ...
        Block 95 → 23:30
        Block 96 → 23:45
        ```
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "96-Block Percentile Calculator | "
    "15-minute resolution | "
    "Multiple columns | "
    "Null values treated as zero"
)
````
