import io
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
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

    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df = pd.read_excel(
            io.BytesIO(file_bytes)
        )

    else:
        st.error("Unsupported file format.")
        st.stop()

except Exception as e:
    st.error("Could not read the uploaded file.")
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
        column_name = f"{column_name}_{column_counter[column_name]}"
    else:
        column_counter[column_name] = 1

    new_columns.append(column_name)

df.columns = new_columns

# ------------------------------------------------------------
# NULL VALUES -> ZERO
# ------------------------------------------------------------
null_count = int(df.isna().sum().sum())
df = df.fillna(0)

st.success("File loaded successfully.")

if null_count > 0:
    st.info(f"{null_count:,} blank/null value(s) were replaced with 0.")

# ------------------------------------------------------------
# FILE INFORMATION
# ------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", f"{len(df):,}")

with col2:
    st.metric("Columns", f"{len(df.columns):,}")

with col3:
    st.metric("Possible Complete Days", f"{len(df) // BLOCKS:,}")

# ------------------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------------------
with st.expander("Preview uploaded data"):
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
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            numeric_columns.append(column)
    except Exception:
        pass

if len(numeric_columns) == 0:
    st.error("No numeric columns were detected.")
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
    st.info("Select at least one column.")
    st.stop()

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

st.metric("Selected Percentile", f"P{percentile:g}")

# ------------------------------------------------------------
# PROCESS DATA
# ------------------------------------------------------------
st.header("4. Data Validation")

processed = {}
validation = []

for column in selected_columns:
    try:
        values = pd.to_numeric(df[column], errors="coerce")
        values = values.fillna(0)
        values = values.to_numpy(dtype=np.float64)

        total_values = len(values)
        complete_days = total_values // BLOCKS
        remainder = total_values % BLOCKS
        usable_values = complete_days * BLOCKS

        if complete_days == 0:
            validation.append({
                "Column": column,
                "Values": total_values,
                "Complete Days": 0,
                "Usable Values": 0,
                "Incomplete Values": remainder,
                "Status": "Insufficient data"
            })
            continue

        # Slice data to whole days only
        trimmed = values[:usable_values]

        # Reshape to 2D array matrix [Days, 96 Blocks]
        daily_matrix = trimmed.reshape(complete_days, BLOCKS)

        # Apply percentile mapping vertically across the days
        percentile_values = np.percentile(daily_matrix, percentile, axis=0)

        processed[column] = {
            "matrix": daily_matrix,
            "percentile": percentile_values,
            "days": complete_days,
            "remainder": remainder
        }

        validation.append({
            "Column": column,
            "Values": total_values,
            "Complete Days": complete_days,
            "Usable Values": usable_values,
            "Incomplete Values": remainder,
            "Status": "Valid"
        })

    except Exception as e:
        validation.append({
            "Column": column,
            "Values": 0,
            "Complete Days": 0,
            "Usable Values": 0,
            "Incomplete Values": 0,
            "Status": f"Error: {str(e)}"
        })

# Display validation logs summary
validation_df = pd.DataFrame(validation)
st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True
)

valid_columns = [col for col in processed.keys()]

if len(valid_columns) == 0:
    st.error("No valid columns with sufficient data are available to calculate percentiles.")
    st.stop()

# ------------------------------------------------------------
# RESULTS & EXPORT
# ------------------------------------------------------------
st.header(f"5. Calculated Percentile Profile (P{percentile:g})")

# Generate standard 1 to 96 timeline base index
result_data = {"Block": np.arange(1, BLOCKS + 1)}

for column in valid_columns:
    result_data[column] = processed[column]["percentile"]

results_df = pd.DataFrame(result_data)

# Splitting UI workspace: Left is Data, Right is Data Visualization
result_col1, result_col2 = st.columns(2)

with result_col1:
    st.subheader("Data Summary")
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Generate CSV stream for standard download
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    
    st.download_button(
        label=f"📥 Download P{percentile:g} Profile as CSV",
        data=csv_bytes,
        file_name=f"96_block_profile_p{percentile:g}.csv",
        mime="text/csv",
        use_container_width=True
    )

with result_col2:
    st.subheader("Visual Profile")
    chart_df = results_df.set_index("Block")
    st.line_chart(chart_df, y=valid_columns)
