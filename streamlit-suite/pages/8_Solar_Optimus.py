```python
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
MAX_PREVIEW_DAYS = 100
MAX_DATA_PREVIEW_ROWS = 20


# ============================================================
# TIME LABELS
# ============================================================

TIME_LABELS = [
    f"{hour:02d}:{minute:02d}"
    for hour in range(24)
    for minute in (0, 15, 30, 45)
]


# ============================================================
# CSS
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
    '<div class="main-title">📊 96-Block Percentile Calculator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Upload time-series data, select multiple columns, "
    "reshape the data into Days × 96 blocks, and calculate "
    "a percentile for every 15-minute block."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# READ FILE
# ============================================================

@st.cache_data(show_spinner=False, max_entries=5)
def read_file(file_bytes, file_name):
    """Read CSV or Excel file."""

    name = str(file_name).lower()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(
                io.BytesIO(file_bytes),
                low_memory=False,
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="latin1",
                low_memory=False,
            )

    if name.endswith(".xlsx"):
        return pd.read_excel(
            io.BytesIO(file_bytes)
        )

    if name.endswith(".xls"):
        return pd.read_excel(
            io.BytesIO(file_bytes)
        )

    raise ValueError(
        "Unsupported file type. Please upload CSV, XLSX or XLS."
    )


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):
    """Convert column names to strings and make them unique."""

    result = df.copy()

    names = []
    used = {}

    for column in result.columns:

        name = str(column).strip()

        if not name:
            name = "Unnamed"

        if name not in used:
            used[name] = 1
            names.append(name)
        else:
            used[name] += 1
            names.append(
                f"{name}_{used[name]}"
            )

    result.columns = names

    return result


# ============================================================
# FIND NUMERIC COLUMNS
# ============================================================

def find_numeric_columns(df):
    """Return columns containing at least one numeric value."""

    numeric_columns = []

    for column in df.columns:

        try:
            numeric = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if numeric.notna().any():
                numeric_columns.append(column)

        except Exception:
            continue

    return numeric_columns


# ============================================================
# PREPARE COLUMN
# ============================================================

@st.cache_data(show_spinner=False, max_entries=50)
def prepare_column(series):
    """
    Convert selected column to numeric.

    Null and non-numeric values become zero.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric.fillna(0)

    return numeric.to_numpy(
        dtype=np.float64
    )


# ============================================================
# CALCULATE PROFILE
# ============================================================

@st.cache_data(show_spinner=False, max_entries=100)
def calculate_profile(values, percentile):
    """
    Reshape values into Days × 96 and calculate
    percentile across days for each block.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    total_values = len(values)

    complete_days = (
        total_values // BLOCKS_PER_DAY
    )

    incomplete_values = (
        total_values % BLOCKS_PER_DAY
    )

    if complete_days < 1:
        return None

    usable_values = (
        complete_days * BLOCKS_PER_DAY
    )

    trimmed = values[
        :usable_values
    ]

    daily_matrix = trimmed.reshape(
        complete_days,
        BLOCKS_PER_DAY,
    )

    percentile_values = np.percentile(
        daily_matrix,
        percentile,
        axis=0,
    )

    return {
        "matrix": daily_matrix,
        "percentile": percentile_values,
        "days": complete_days,
        "total_values": total_values,
        "usable_values": usable_values,
        "incomplete_values": incomplete_values,
    }


# ============================================================
# CREATE RESULT
# ============================================================

def create_result_dataframe(
    profiles,
    percentile,
):
    """Create 96-row percentile result."""

    result = pd.DataFrame(
        {
            "Block": np.arange(
                1,
                BLOCKS_PER_DAY + 1,
            ),
            "Time": TIME_LABELS,
        }
    )

    for column, values in profiles.items():

        result[
            f"{column} P{percentile:g}"
        ] = values

    return result


# ============================================================
# EXCEL EXPORT
# ============================================================

def create_excel(result_df):
    """Create Excel file in memory."""

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
# UPLOAD
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
        "Upload your 15-minute time-series data."
    ),
)


if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel file to begin."
    )

    st.stop()


# ============================================================
# READ UPLOADED FILE
# ============================================================

try:

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:

        st.error(
            "The uploaded file is empty."
        )

        st.stop()

    df = read_file(
        file_bytes,
        uploaded_file.name,
    )

except Exception as error:

    st.error(
        f"Unable to read the file: {error}"
    )

    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

if df is None:

    st.error(
        "No dataframe was created from the uploaded file."
    )

    st.stop()


if df.empty:

    st.error(
        "The uploaded file contains no rows."
    )

    st.stop()


if len(df.columns) == 0:

    st.error(
        "The uploaded file contains no columns."
    )

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df = clean_column_names(df)


# ============================================================
# REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

empty_columns = [
    column
    for column in df.columns
    if df[column].isna().all()
]


if empty_columns:

    df = df.drop(
        columns=empty_columns
    )


if df.empty or len(df.columns) == 0:

    st.error(
        "The file contains only empty columns."
    )

    st.stop()


# ============================================================
# FILL NULL VALUES WITH ZERO
# ============================================================

null_count = int(
    df.isna().sum().sum()
)

df = df.fillna(0)


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    f"File loaded successfully: {uploaded_file.name}"
)


if null_count > 0:

    st.info(
        f"{null_count:,} null/blank value(s) "
        "were replaced with 0."
    )


# ============================================================
# FILE INFORMATION
# ============================================================

numeric_columns = find_numeric_columns(df)

complete_days = (
    len(df) // BLOCKS_PER_DAY
)


info1, info2, info3, info4 = st.columns(4)


with info1:
    st.metric(
        "Rows",
        f"{len(df):,}",
    )


with info2:
    st.metric(
        "Columns",
        f"{len(df.columns):,}",
    )


with info3:
    st.metric(
        "Numeric Columns",
        f"{len(numeric_columns):,}",
    )


with info4:
    st.metric(
        "Complete Days",
        f"{complete_days:,}",
    )


# ============================================================
# PREVIEW
# ============================================================

with st.expander(
    "Preview uploaded data",
    expanded=False,
):

    st.dataframe(
        df.head(MAX_DATA_PREVIEW_ROWS),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# NUMERIC COLUMN VALIDATION
# ============================================================

if not numeric_columns:

    st.error(
        "No numeric columns were detected."
    )

    st.stop()


# ============================================================
# COLUMN SELECTION
# ============================================================

st.subheader("2. Select Data Columns")

selected_columns = st.multiselect(
    "Select one or more columns",
    options=numeric_columns,
    default=[],
    help=(
        "You can select multiple columns. "
        "Each selected column gets its own "
        "96-block percentile profile."
    ),
)


if not selected_columns:

    st.info(
        "Please select at least one column."
    )

    st.stop()


st.caption(
    f"{len(selected_columns)} column(s) selected: "
    + ", ".join(selected_columns)
)


# ============================================================
# PERCENTILE
# ============================================================

st.subheader("3. Percentile Selection")

percentile_left, percentile_right = st.columns(
    [4, 1]
)


with percentile_left:

    percentile = st.slider(
        "Select percentile",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=0.5,
        help=(
            "Percentile is calculated independently "
            "for every one of the 96 blocks."
        ),
    )


with percentile_right:

    st.metric(
        "Selected",
        f"P{percentile:g}",
    )


# ============================================================
# PROCESS COLUMNS
# ============================================================

st.subheader("4. Data Validation")

processed = {}
validation_rows = []


for column in selected_columns:

    try:

        values = prepare_column(
            df[column]
        )

        result = calculate_profile(
            values,
            percentile,
        )

    except Exception as error:

        validation_rows.append(
            {
                "Column": column,
                "Rows": len(df),
                "Complete Days": 0,
                "Usable Values": 0,
                "Incomplete Values": len(df),
                "Status": f"Error: {error}",
            }
        )

        continue


    if result is None:

        validation_rows.append(
            {
                "Column": column,
                "Rows": len(values),
                "Complete Days": 0,
                "Usable Values": 0,
                "Incomplete Values": len(values),
                "Status": "Insufficient data",
            }
        )

        continue


    processed[column] = result


    validation_rows.append(
        {
            "Column": column,
            "Rows": result["total_values"],
            "Complete Days": result["days"],
            "Usable Values": result["usable_values"],
            "Incomplete Values": result["incomplete_values"],
            "Status": "Valid",
        }
    )


# ============================================================
# VALIDATION TABLE
# ============================================================

validation_df = pd.DataFrame(
    validation_rows
)


st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# NO VALID COLUMNS
# ============================================================

if not processed:

    st.error(
        "None of the selected columns contains "
        "at least 96 values."
    )

    st.stop()


# ============================================================
# INCOMPLETE DAY WARNING
# ============================================================

incomplete_found = False


for column, data in processed.items():

    remainder = data["incomplete_values"]

    if remainder > 0:

        incomplete_found = True

        st.warning(
            f"{column}: {remainder} value(s) remain "
            "after the last complete 96-block day. "
            "Those values are excluded."
        )


if not incomplete_found:

    st.success(
        "All selected columns contain complete 96-block days."
    )


# ============================================================
# SUMMARY
# ============================================================

summary1, summary2, summary3, summary4 = (
    st.columns(4)
)


with summary1:

    st.metric(
        "Columns Processed",
        len(processed),
    )


with summary2:

    st.metric(
        "Blocks Per Day",
        BLOCKS_PER_DAY,
    )


with summary3:

    minimum_days = min(
        data["days"]
        for data in processed.values()
    )

    st.metric(
        "Complete Days",
        f"{minimum_days:,}",
    )


with summary4:

    st.metric(
        "Percentile",
        f"P{percentile:g}",
    )


# ============================================================
# RESULT
# ============================================================

profiles = {
    column: data["percentile"]
    for column, data in processed.items()
}


result_df = create_result_dataframe(
    profiles,
    percentile,
)


# ============================================================
# RESULT TABLE
# ============================================================

st.subheader(
    "5. 96-Block Percentile Result"
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

st.subheader(
    "6. Percentile Profile"
)


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
# INDIVIDUAL COLUMN GRAPH
# ============================================================

with st.expander(
    "View individual percentile profile",
    expanded=False,
):

    graph_column = st.selectbox(
        "Select column",
        options=list(processed.keys()),
        key="graph_column",
    )


    graph_name = (
        f"{graph_column} P{percentile:g}"
    )


    individual_df = (
        result_df
        .set_index("Time")
        [[graph_name]]
    )


    st.line_chart(
        individual_df,
        use_container_width=True,
    )


# ============================================================
# DAYS × 96 MATRIX
# ============================================================

st.subheader(
    "7. Reshaped Days × 96 Data"
)


matrix_column = st.selectbox(
    "Select column",
    options=list(processed.keys()),
    key="matrix_column",
)


matrix = processed[
    matrix_column
]["matrix"]


total_days = matrix.shape[0]


st.caption(
    f"Full matrix shape: "
    f"{total_days:,} days × 96 blocks"
)


# ============================================================
# MATRIX PREVIEW
# ============================================================

display_days = min(
    total_days,
    MAX_PREVIEW_DAYS,
)


matrix_preview = matrix[
    :display_days
]


block_columns = [
    f"Block_{i:02d}"
    for i in range(
        1,
        BLOCKS_PER_DAY + 1,
    )
]


matrix_df = pd.DataFrame(
    matrix_preview,
    columns=block_columns,
)


matrix_df.insert(
    0,
    "Day",
    np.arange(
        1,
        display_days + 1,
    ),
)


st.dataframe(
    matrix_df,
    use_container_width=True,
    hide_index=True,
    height=500,
)


if total_days > MAX_PREVIEW_DAYS:

    st.info(
        f"Only the first {MAX_PREVIEW_DAYS} days "
        "are displayed. The percentile calculation "
        f"still uses all {total_days:,} complete days."
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.subheader(
    "8. Download Result"
)


download1, download2 = st.columns(2)


# ============================================================
# CSV DOWNLOAD
# ============================================================

csv_bytes = result_df.to_csv(
    index=False
).encode("utf-8")


with download1:

    st.download_button(
        label=f"⬇️ Download P{percentile:g} CSV",
        data=csv_bytes,
        file_name=(
            f"96_block_percentile_P"
            f"{percentile:g}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# EXCEL DOWNLOAD
# ============================================================

with download2:

    try:

        excel_bytes = create_excel(
            result_df
        )

        st.download_button(
            label=f"⬇️ Download P{percentile:g} Excel",
            data=excel_bytes,
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
            f"Excel export could not be created: {error}"
        )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "Calculation methodology",
    expanded=False,
):

    st.markdown(
        """
        ### Calculation Logic

        **1. Upload**

        CSV, XLSX and XLS files are supported.

        **2. Null handling**

        All blank / NaN values are converted to **0**.

        Existing zero values remain zero.

        **3. Column selection**

        Multiple numeric columns can be selected.

        **4. Reshape**

        Each selected column is reshaped into:

        `Days × 96`

        Each row represents one day.

        Each column represents one 15-minute block.

        **5. Incomplete final day**

        If the number of values is not divisible by 96,
        the incomplete final portion is excluded.

        **6. Percentile**

        The selected percentile is calculated independently
        for each of the 96 blocks using:

        `np.percentile(daily_matrix, percentile, axis=0)`

        **7. Result**

        The output always contains exactly 96 blocks:

        `00:00, 00:15, 00:30, ... 23:45`
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
```
