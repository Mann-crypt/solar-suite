```python
import io

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="96 Block Percentile",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

BLOCKS_PER_DAY = 96

TIME_LABELS = pd.date_range(
    start="00:00",
    periods=96,
    freq="15min",
).strftime("%H:%M").tolist()


# ============================================================
# FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_file(file_bytes, file_name):
    """Read CSV or Excel file."""

    extension = file_name.lower().split(".")[-1]

    if extension == "csv":
        df = pd.read_csv(
            io.BytesIO(file_bytes)
        )

    elif extension in ["xlsx", "xls"]:
        df = pd.read_excel(
            io.BytesIO(file_bytes)
        )

    else:
        raise ValueError(
            "Unsupported file type. Please upload CSV, XLSX or XLS."
        )

    if df is None or df.empty:
        raise ValueError(
            "The uploaded file is empty."
        )

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all",
    )

    if df.empty:
        raise ValueError(
            "The uploaded file contains no usable columns."
        )

    # Convert column names to strings
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # Make duplicate column names unique
    new_columns = []
    column_count = {}

    for col in df.columns:

        if col not in column_count:
            column_count[col] = 1
            new_columns.append(col)

        else:
            column_count[col] += 1

            new_columns.append(
                f"{col}_{column_count[col]}"
            )

    df.columns = new_columns

    return df


def get_numeric_columns(df):
    """Find columns containing numeric data."""

    numeric_columns = []

    for col in df.columns:

        numeric_data = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if numeric_data.notna().any():
            numeric_columns.append(col)

    return numeric_columns


def process_column(series, percentile):
    """
    Convert one column to numeric,
    reshape into Days x 96,
    and calculate percentile for each block.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid_values = (
        numeric
        .dropna()
        .reset_index(drop=True)
    )

    valid_count = len(valid_values)

    complete_days = (
        valid_count // BLOCKS_PER_DAY
    )

    remainder = (
        valid_count % BLOCKS_PER_DAY
    )

    usable_count = (
        complete_days * BLOCKS_PER_DAY
    )

    if complete_days < 1:
        return None

    values = valid_values.iloc[
        :usable_count
    ].to_numpy(
        dtype=float
    )

    daily_matrix = values.reshape(
        complete_days,
        BLOCKS_PER_DAY,
    )

    percentile_values = np.percentile(
        daily_matrix,
        percentile,
        axis=0,
    )

    return {
        "valid_count": valid_count,
        "complete_days": complete_days,
        "remainder": remainder,
        "usable_count": usable_count,
        "daily_matrix": daily_matrix,
        "percentile_values": percentile_values,
    }


def make_result_table(processed_data, percentile):
    """Create final 96-block result."""

    result = pd.DataFrame(
        {
            "Block": range(1, 97),
            "Time": TIME_LABELS,
        }
    )

    for column, data in processed_data.items():

        result[
            f"{column} P{percentile:g}"
        ] = data["percentile_values"]

    return result


def dataframe_to_excel(df):
    """Create Excel file in memory."""

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Percentile",
        )

    return output.getvalue()


# ============================================================
# HEADER
# ============================================================

st.title("📊 96-Block Percentile Calculator")

st.write(
    "Upload your time-series file, select one or more columns, "
    "and calculate a percentile profile for each of the 96 "
    "15-minute blocks."
)


# ============================================================
# STEP 1: UPLOAD FILE
# ============================================================

st.header("1. Upload File")

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


# ============================================================
# READ FILE
# ============================================================

try:

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        st.error(
            "The uploaded file is empty."
        )
        st.stop()

    df = load_file(
        file_bytes,
        uploaded_file.name,
    )

except Exception as error:

    st.error(
        f"Unable to read the file: {error}"
    )

    st.stop()


# ============================================================
# FILE SUCCESS
# ============================================================

st.success(
    f"File loaded successfully: {uploaded_file.name}"
)


# ============================================================
# FILE SUMMARY
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        f"{len(df):,}",
    )

with col2:
    st.metric(
        "Columns",
        f"{len(df.columns):,}",
    )

numeric_columns = get_numeric_columns(df)

with col3:
    st.metric(
        "Numeric columns",
        f"{len(numeric_columns):,}",
    )


# ============================================================
# PREVIEW
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
# CHECK NUMERIC COLUMNS
# ============================================================

if len(numeric_columns) == 0:

    st.error(
        "No numeric columns were found in the uploaded file."
    )

    st.stop()


# ============================================================
# STEP 2: SELECT COLUMNS
# ============================================================

st.header("2. Select Columns")

selected_columns = st.multiselect(
    "Select one or more columns for percentile calculation",
    options=numeric_columns,
)

if len(selected_columns) == 0:

    st.info(
        "Select at least one column."
    )

    st.stop()


st.write(
    f"Selected columns: **{len(selected_columns)}**"
)

st.caption(
    ", ".join(selected_columns)
)


# ============================================================
# STEP 3: PERCENTILE
# ============================================================

st.header("3. Select Percentile")

percentile_col1, percentile_col2 = st.columns(
    [4, 1]
)

with percentile_col1:

    percentile = st.slider(
        "Percentile",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=0.5,
    )

with percentile_col2:

    st.metric(
        "Selected",
        f"P{percentile:g}",
    )


# ============================================================
# STEP 4: PROCESS DATA
# ============================================================

st.header("4. Data Validation")

processed_data = {}

validation_results = []


for column in selected_columns:

    result = process_column(
        df[column],
        percentile,
    )

    # Count valid values independently
    numeric_series = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    valid_count = int(
        numeric_series.notna().sum()
    )

    complete_days = (
        valid_count // BLOCKS_PER_DAY
    )

    remainder = (
        valid_count % BLOCKS_PER_DAY
    )

    usable_count = (
        complete_days * BLOCKS_PER_DAY
    )

    if result is not None:

        status = "Valid"

        processed_data[column] = result

    else:

        status = "Insufficient data"

    validation_results.append(
        {
            "Column": column,
            "Valid Values": valid_count,
            "Complete Days": complete_days,
            "Usable Values": usable_count,
            "Incomplete Values": remainder,
            "Status": status,
        }
    )


validation_df = pd.DataFrame(
    validation_results
)


# ============================================================
# VALIDATION TABLE
# ============================================================

st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# VALIDATION WARNINGS
# ============================================================

invalid_columns = [
    col
    for col in selected_columns
    if col not in processed_data
]


if invalid_columns:

    st.warning(
        "These columns have fewer than 96 valid values "
        "and will not be processed:"
    )

    for column in invalid_columns:
        st.write(
            f"- {column}"
        )


incomplete_columns = []

for column, data in processed_data.items():

    if data["remainder"] > 0:

        incomplete_columns.append(
            column
        )


if incomplete_columns:

    st.warning(
        "Some columns contain an incomplete final day. "
        "The incomplete values are excluded from the calculation."
    )


# ============================================================
# STOP IF NOTHING CAN BE PROCESSED
# ============================================================

if not processed_data:

    st.error(
        "None of the selected columns contains enough data "
        "for one complete 96-block day."
    )

    st.stop()


# ============================================================
# PROCESSING SUMMARY
# ============================================================

summary1, summary2, summary3 = st.columns(3)

with summary1:

    st.metric(
        "Columns processed",
        len(processed_data),
    )

with summary2:

    total_days = min(
        data["complete_days"]
        for data in processed_data.values()
    )

    st.metric(
        "96-block days",
        f"{total_days:,}",
    )

with summary3:

    st.metric(
        "Percentile",
        f"P{percentile:g}",
    )


# ============================================================
# STEP 5: RESULT
# ============================================================

st.header("5. Percentile Result")


result_df = make_result_table(
    processed_data,
    percentile,
)


st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True,
    height=550,
)


# ============================================================
# STEP 6: GRAPH
# ============================================================

st.header("6. Percentile Profile")

graph_df = result_df.set_index(
    "Time"
)

graph_columns = [
    col
    for col in result_df.columns
    if col not in [
        "Block",
        "Time",
    ]
]

if graph_columns:

    st.line_chart(
        graph_df[graph_columns],
        use_container_width=True,
    )


# ============================================================
# INDIVIDUAL GRAPHS
# ============================================================

with st.expander(
    "View individual column profiles",
    expanded=False,
):

    for column in processed_data:

        result_column = (
            f"{column} P{percentile:g}"
        )

        st.subheader(
            column
        )

        individual_df = result_df.set_index(
            "Time"
        )[[result_column]]

        st.line_chart(
            individual_df,
            use_container_width=True,
        )


# ============================================================
# STEP 7: DAYS x 96 MATRIX
# ============================================================

st.header("7. Reshaped Days × 96 Data")

matrix_column = st.selectbox(
    "Select a column to view its reshaped matrix",
    options=list(
        processed_data.keys()
    ),
)


selected_matrix = processed_data[
    matrix_column
]["daily_matrix"]


st.write(
    f"**{matrix_column}**"
)

st.caption(
    f"Matrix shape: "
    f"{selected_matrix.shape[0]} days × "
    f"{selected_matrix.shape[1]} blocks"
)


block_names = [
    f"Block_{i:02d}"
    for i in range(1, 97)
]


matrix_df = pd.DataFrame(
    selected_matrix,
    columns=block_names,
)


matrix_df.insert(
    0,
    "Day",
    range(
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
# STEP 8: DOWNLOAD
# ============================================================

st.header("8. Download Result")


download_col1, download_col2 = st.columns(2)


# ------------------------------------------------------------
# CSV DOWNLOAD
# ------------------------------------------------------------

csv_data = result_df.to_csv(
    index=False
).encode("utf-8")


with download_col1:

    st.download_button(
        label="⬇️ Download CSV",
        data=csv_data,
        file_name=(
            f"96_block_percentile_P"
            f"{percentile:g}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ------------------------------------------------------------
# EXCEL DOWNLOAD
# ------------------------------------------------------------

try:

    excel_data = dataframe_to_excel(
        result_df
    )

    with download_col2:

        st.download_button(
            label="⬇️ Download Excel",
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

except Exception as error:

    st.warning(
        f"Excel download could not be created: {error}"
    )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "Calculation methodology",
    expanded=False,
):

    st.write(
        "Each selected column is processed independently."
    )

    st.code(
        """
Valid values
     ↓
Complete days
     ↓
reshape(-1, 96)
     ↓
Days × 96 matrix
     ↓
np.percentile(matrix, percentile, axis=0)
     ↓
96 percentile values
        """,
        language="text",
    )

    st.write(
        "For example, if there are 30 complete days:"
    )

    st.code(
        """
30 × 96

Day 1   → 96 blocks
Day 2   → 96 blocks
Day 3   → 96 blocks
...
Day 30  → 96 blocks

Result:

30 rows × 96 columns
        """,
        language="text",
    )

    st.write(
        "The percentile is calculated vertically across "
        "the days for each block."
    )

    st.code(
        "np.percentile(daily_matrix, percentile, axis=0)",
        language="python",
    )

    st.write(
        "Therefore the output always contains exactly "
        "96 time blocks from 00:00 to 23:45."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "96-Block Percentile Calculator | "
    "15-minute resolution | "
    "Multiple columns supported"
)
```
