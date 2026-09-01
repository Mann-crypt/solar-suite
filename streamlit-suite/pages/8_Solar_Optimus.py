import io
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="96 Block Percentile Calculator",
    page_icon="📊",
    layout="wide"
)


BLOCKS = 96


st.title("📊 96 Block Percentile Calculator")
st.write(
    "Upload your time-series file, select one or more columns, "
    "choose a percentile, and calculate the percentile profile "
    "for all 96 daily time blocks."
)


# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------

st.header("1. Upload File")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)


if uploaded_file is None:
    st.info("Please upload a CSV or Excel file.")
    st.stop()


# ------------------------------------------------------------
# READ FILE
# ------------------------------------------------------------

try:
    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) == 0:
        st.error("The uploaded file is empty.")
        st.stop()

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                low_memory=False
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="latin1",
                low_memory=False
            )

    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(
            io.BytesIO(file_bytes)
        )

    elif file_name.endswith(".xls"):
        df = pd.read_excel(
            io.BytesIO(file_bytes)
        )

    else:
        st.error(
            "Unsupported file format."
        )
        st.stop()

except Exception as e:
    st.error(
        "Could not read the uploaded file."
    )
    st.exception(e)
    st.stop()


# ------------------------------------------------------------
# BASIC VALIDATION
# ------------------------------------------------------------

if df is None:
    st.error("No data was loaded.")
    st.stop()


if df.empty:
    st.error("The uploaded file contains no rows.")
    st.stop()


if len(df.columns) == 0:
    st.error("The uploaded file contains no columns.")
    st.stop()


# ------------------------------------------------------------
# CLEAN COLUMN NAMES
# ------------------------------------------------------------

new_columns = []
column_counter = {}

for column in df.columns:

    column_name = str(column).strip()

    if column_name == "":
        column_name = "Unnamed"

    if column_name in column_counter:
        column_counter[column_name] += 1
        column_name = (
            column_name
            + "_"
            + str(column_counter[column_name])
        )
    else:
        column_counter[column_name] = 1

    new_columns.append(column_name)


df.columns = new_columns


# ------------------------------------------------------------
# NULL VALUES -> ZERO
# ------------------------------------------------------------

null_count = int(
    df.isna().sum().sum()
)

df = df.fillna(0)


st.success(
    "File loaded successfully."
)


if null_count > 0:
    st.info(
        str(null_count)
        + " blank/null value(s) were replaced with 0."
    )


# ------------------------------------------------------------
# FILE INFORMATION
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

with col3:
    st.metric(
        "Possible Complete Days",
        f"{len(df) // BLOCKS:,}"
    )


# ------------------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------------------

with st.expander(
    "Preview uploaded data"
):

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------------------
# FIND NUMERIC COLUMNS
# ------------------------------------------------------------

numeric_columns = []

for column in df.columns:

    try:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.notna().any():
            numeric_columns.append(column)

    except Exception:
        pass


if len(numeric_columns) == 0:

    st.error(
        "No numeric columns were detected."
    )

    st.stop()


# ------------------------------------------------------------
# COLUMN SELECTION
# ------------------------------------------------------------

st.header("2. Select Columns")

selected_columns = st.multiselect(
    "Select one or more columns for percentile calculation",
    options=numeric_columns
)


if len(selected_columns) == 0:

    st.info(
        "Select at least one column."
    )

    st.stop()


st.write(
    "Selected columns:",
    ", ".join(selected_columns)
)


# ------------------------------------------------------------
# PERCENTILE
# ------------------------------------------------------------

st.header("3. Select Percentile")

percentile = st.slider(
    "Percentile",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=0.5
)


st.metric(
    "Selected Percentile",
    f"P{percentile:g}"
)


# ------------------------------------------------------------
# PROCESS DATA
# ------------------------------------------------------------

st.header("4. Data Validation")

processed = {}
validation = []


for column in selected_columns:

    try:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Null and invalid values become zero.
        values = values.fillna(0)

        values = values.to_numpy(
            dtype=np.float64
        )

        total_values = len(values)

        complete_days = (
            total_values // BLOCKS
        )

        remainder = (
            total_values % BLOCKS
        )

        usable_values = (
            complete_days * BLOCKS
        )

        if complete_days == 0:

            validation.append(
                {
                    "Column": column,
                    "Values": total_values,
                    "Complete Days": 0,
                    "Usable Values": 0,
                    "Incomplete Values": remainder,
                    "Status": "Insufficient data"
                }
            )

            continue


        # Only complete days are used.
        trimmed = values[
            :usable_values
        ]


        # Reshape into Days x 96.
        daily_matrix = trimmed.reshape(
            complete_days,
            BLOCKS
        )


        # Percentile across days.
        percentile_values = np.percentile(
            daily_matrix,
            percentile,
            axis=0
        )


        processed[column] = {
            "matrix": daily_matrix,
            "percentile": percentile_values,
            "days": complete_days,
            "remainder": remainder
        }


        validation.append(
            {
                "Column": column,
                "Values": total_values,
                "Complete Days": complete_days,
                "Usable Values": usable_values,
                "Incomplete Values": remainder,
                "Status": "Valid"
            }
        )

    except Exception as e:

        validation.append(
            {
                "Column": column,
                "Values": 0,
                "Complete Days": 0,
                "Usable Values": 0,
                "Incomplete Values": 0,
                "Status": "Error: " + str(e)
            }
        )


# ------------------------------------------------------------
# VALIDATION TABLE
# ------------------------------------------------------------

validation_df = pd.DataFrame(
    validation
)


st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# VALID COLUMNS
# ------------------------------------------------------------

valid_columns = list(
    processed.keys()
)


if len(valid_columns) == 0:

    st.error(
        "None of the selected columns contains at least "
        "96 values."
    )

    st.stop()


# ------------------------------------------------------------
# INCOMPLETE DATA WARNING
# ------------------------------------------------------------

for column in valid_columns:

    remainder = processed[
        column
    ]["remainder"]

    if remainder > 0:

        st.warning(
            column
            + " has "
            + str(remainder)
            + " value(s) after the last complete day. "
            + "Those values are excluded."
        )


# ------------------------------------------------------------
# RESULT DATAFRAME
# ------------------------------------------------------------

st.header("5. 96 Block Percentile Result")


result = pd.DataFrame(
    {
        "Block": np.arange(
            1,
            BLOCKS + 1
        ),
        "Time": [
            f"{hour:02d}:{minute:02d}"
            for hour in range(24)
            for minute in [0, 15, 30, 45]
        ]
    }
)


for column in valid_columns:

    result[
        column + f" P{percentile:g}"
    ] = processed[
        column
    ]["percentile"]


st.dataframe(
    result,
    use_container_width=True,
    hide_index=True,
    height=550
)


# ------------------------------------------------------------
# GRAPH
# ------------------------------------------------------------

st.header("6. Percentile Graph")


graph_data = result.set_index(
    "Time"
)


graph_columns = [
    column
    for column in result.columns
    if column not in ["Block", "Time"]
]


st.line_chart(
    graph_data[graph_columns],
    use_container_width=True
)


# ------------------------------------------------------------
# INDIVIDUAL GRAPH
# ------------------------------------------------------------

with st.expander(
    "View individual column"
):

    graph_column = st.selectbox(
        "Select column",
        valid_columns
    )

    graph_name = (
        graph_column
        + f" P{percentile:g}"
    )

    st.line_chart(
        result.set_index("Time")[
            [graph_name]
        ],
        use_container_width=True
    )


# ------------------------------------------------------------
# DAYS x 96 MATRIX
# ------------------------------------------------------------

st.header("7. Days × 96 Matrix")


matrix_column = st.selectbox(
    "Select column to view",
    valid_columns
)


matrix = processed[
    matrix_column
]["matrix"]


total_days = matrix.shape[0]


st.write(
    "Matrix shape:",
    f"{total_days:,} days × 96 blocks"
)


# ------------------------------------------------------------
# MATRIX PREVIEW
# ------------------------------------------------------------

# Do not render thousands of rows.
# This keeps Streamlit responsive.

display_days = min(
    total_days,
    100
)


matrix_preview = matrix[
    :display_days
]


block_names = [
    f"Block_{i:02d}"
    for i in range(
        1,
        BLOCKS + 1
    )
]


matrix_df = pd.DataFrame(
    matrix_preview,
    columns=block_names
)


matrix_df.insert(
    0,
    "Day",
    np.arange(
        1,
        display_days + 1
    )
)


st.dataframe(
    matrix_df,
    use_container_width=True,
    hide_index=True,
    height=500
)


if total_days > 100:

    st.info(
        "Only the first 100 days are displayed. "
        "The percentile calculation uses all complete days."
    )


# ------------------------------------------------------------
# DOWNLOAD CSV
# ------------------------------------------------------------

st.header("8. Download Result")


csv_data = result.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label=f"⬇️ Download P{percentile:g} CSV",
    data=csv_data,
    file_name=(
        f"96_block_percentile_P{percentile:g}.csv"
    ),
    mime="text/csv",
    use_container_width=True
)


# ------------------------------------------------------------
# DOWNLOAD EXCEL
# ------------------------------------------------------------

try:

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        result.to_excel(
            writer,
            index=False,
            sheet_name="Percentile"
        )

    st.download_button(
        label=f"⬇️ Download P{percentile:g} Excel",
        data=excel_buffer.getvalue(),
        file_name=(
            f"96_block_percentile_P{percentile:g}.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )

except Exception as e:

    st.warning(
        "Excel download is unavailable: "
        + str(e)
    )


# ------------------------------------------------------------
# CALCULATION DETAILS
# ------------------------------------------------------------

with st.expander(
    "Calculation methodology"
):

    st.write(
        "Null values are converted to zero."
    )

    st.write(
        "Selected columns are converted to numeric values."
    )

    st.write(
        "Each column is reshaped as:"
    )

    st.code(
        "Days × 96"
    )

    st.write(
        "The percentile is calculated independently "
        "for every one of the 96 blocks."
    )

    st.code(
        "np.percentile(daily_matrix, percentile, axis=0)"
    )

    st.write(
        "If the final day contains fewer than 96 values, "
        "that incomplete portion is excluded."
    )
```
