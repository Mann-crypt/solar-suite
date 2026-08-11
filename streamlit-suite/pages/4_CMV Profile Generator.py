import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from io import BytesIO
from scipy.signal import savgol_filter


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="CMV Curve Generator",
    page_icon="📈",
    layout="wide",
)


# ==========================================================
# FUNCTIONS
# ==========================================================

def find_date_column(df):
    """Find a likely Date column."""

    possible_names = [
        "Date",
        "Datetime",
        "Date Time",
        "Timestamp",
        "Time",
    ]

    normalized = {
        str(col).strip().lower().replace("_", " "): col
        for col in df.columns
    }

    for name in possible_names:
        key = name.lower()

        if key in normalized:
            return normalized[key]

    return None


def prepare_data(
    df,
    min_data_requirement,
):
    """
    Clean raw CMV data.

    min_data_requirement is a percentage from 0 to 100.
    """

    df = df.copy()

    # ------------------------------------------------------
    # Date handling
    # ------------------------------------------------------

    date_col = find_date_column(df)

    if date_col is not None:

        parsed_date = pd.to_datetime(
            df[date_col],
            errors="coerce",
        )

        # Keep Date as index only if useful values exist
        if parsed_date.notna().any():

            df[date_col] = parsed_date
            df = df.set_index(date_col)

    # ------------------------------------------------------
    # Convert numeric columns where possible
    # ------------------------------------------------------

    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):

            converted = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            # Convert only when there is meaningful
            # numeric data
            if converted.notna().sum() > 0:

                df[col] = converted

    # ------------------------------------------------------
    # Keep only numeric columns
    # ------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.empty:

        raise ValueError(
            "No numeric columns were found "
            "in the workbook."
        )

    # ------------------------------------------------------
    # Minimum data requirement
    # ------------------------------------------------------

    min_data_threshold = int(
        np.ceil(
            len(numeric_df)
            * min_data_requirement
            / 100
        )
    )

    numeric_df = numeric_df.dropna(
        axis=1,
        thresh=min_data_threshold,
    )

    if numeric_df.empty:

        raise ValueError(
            "No columns satisfy the selected "
            f"{min_data_requirement}% minimum "
            "data requirement."
        )

    # ------------------------------------------------------
    # Replace missing values
    # ------------------------------------------------------

    numeric_df = numeric_df.fillna(0)

    # ------------------------------------------------------
    # Remove abnormal columns
    #
    # Original logic:
    # > 1200 OR maximum < 800
    # ------------------------------------------------------

    columns_to_drop = []

    for col in numeric_df.columns:

        values = numeric_df[col].to_numpy(
            dtype=float
        )

        if len(values) == 0:
            columns_to_drop.append(col)
            continue

        maximum = np.nanmax(values)

        if maximum > 1200 or maximum < 800:

            columns_to_drop.append(col)

    numeric_df = numeric_df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    # ------------------------------------------------------
    # Remove completely zero columns
    # ------------------------------------------------------

    zero_columns = numeric_df.columns[
        (numeric_df == 0).all()
    ]

    numeric_df = numeric_df.drop(
        columns=zero_columns,
        errors="ignore",
    )

    if numeric_df.empty:

        raise ValueError(
            "No usable generation columns remain "
            "after cleaning."
        )

    return numeric_df


def calculate_percentiles(
    df,
    blocks=96,
):
    """
    Reshape data into days x 96 blocks and calculate
    95th percentile for each block.
    """

    usable_rows = (
        len(df) // blocks
    ) * blocks

    if usable_rows < blocks:

        raise ValueError(
            f"At least {blocks} rows are required."
        )

    df = df.iloc[
        :usable_rows
    ].copy()

    days = usable_rows // blocks

    percentile_data = {}

    for col in df.columns:

        values = df[col].to_numpy(
            dtype=float
        )

        reshaped = values.reshape(
            days,
            blocks,
        )

        percentile_data[col] = (
            np.percentile(
                reshaped,
                95,
                axis=0,
            )
        )

    return pd.DataFrame(
        percentile_data
    )


def remove_constant_blocks(
    percentile_df,
):
    """
    Replace consecutive equal percentile values
    with zero, following the original logic.
    """

    result = percentile_df.copy()

    for col in result.columns:

        series = result[col]

        result[col] = series.where(
            series.diff() != 0,
            0,
        )

    return result


def generate_smooth_profile(
    average,
):
    """
    Generate smooth CMV profile.
    """

    values = np.asarray(
        average,
        dtype=float,
    )

    # First smoothing
    smooth = savgol_filter(
        values,
        window_length=15,
        polyorder=3,
    )

    smooth = np.clip(
        smooth,
        0,
        None,
    )

    # Remove very small generation
    smooth = np.where(
        smooth < 4.9,
        0,
        smooth,
    )

    # Second smoothing
    smooth = savgol_filter(
        smooth,
        window_length=7,
        polyorder=3,
    )

    smooth = np.clip(
        smooth,
        0,
        None,
    )

    return smooth


def make_percentile_chart(
    percentile_df,
    selected_columns,
):
    """
    Plot 95th percentile profile of all selected columns.
    """

    fig = go.Figure()

    x = np.arange(
        1,
        len(percentile_df) + 1,
    )

    for col in percentile_df.columns:

        is_selected = (
            col in selected_columns
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=percentile_df[col],
                name=str(col),
                mode="lines",
                line=dict(
                    width=3
                    if is_selected
                    else 1.5
                ),
                opacity=1
                if is_selected
                else 0.45,
                legendgroup=str(col),
            )
        )

    fig.update_layout(
        height=550,
        hovermode="x unified",
        template="streamlit",
        xaxis_title="15 Minute Block",
        yaxis_title="95th Percentile Power",
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


def make_final_chart(
    percentile_df,
    average,
    smooth,
):
    """
    Plot percentile columns, average and final profile.
    """

    fig = go.Figure()

    x = np.arange(
        1,
        len(average) + 1,
    )

    # ------------------------------------------------------
    # Individual percentile profiles
    # ------------------------------------------------------

    for col in percentile_df.columns:

        fig.add_trace(
            go.Scatter(
                x=x,
                y=percentile_df[col],
                name=str(col),
                mode="lines",
                line=dict(
                    width=1.2
                ),
                opacity=0.35,
            )
        )

    # ------------------------------------------------------
    # Average
    # ------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=x,
            y=average,
            name="95th Percentile Average",
            mode="lines",
            line=dict(
                width=3,
            ),
        )
    )

    # ------------------------------------------------------
    # Smooth profile
    # ------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=x,
            y=smooth,
            name="Smooth Profile",
            mode="lines",
            line=dict(
                width=4,
                color="#00c6ff",
            ),
        )
    )

    fig.update_layout(
        height=550,
        hovermode="x unified",
        template="streamlit",
        xaxis_title="15 Minute Block",
        yaxis_title="Power",
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# ==========================================================
# HEADER
# ==========================================================

st.title("📈 CMV Curve Generator")

st.caption(
    "Generate a 95th-percentile CMV generation profile "
    "from historical generation data."
)

st.divider()


# ==========================================================
# EXCEL IMPORT
# ==========================================================

st.subheader("1. Import Excel Workbook")

uploaded_file = st.file_uploader(
    "Upload Excel Workbook",
    type=["xlsx"],
    key="cmv_excel_upload",
)

if uploaded_file is None:

    st.info(
        "Upload an Excel workbook to begin."
    )

    st.stop()


# ==========================================================
# SHEET SELECTION
# ==========================================================

try:

    excel_file = pd.ExcelFile(
        uploaded_file
    )

    sheet_names = excel_file.sheet_names

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


selected_sheet = st.selectbox(
    "Select CMV source sheet",
    sheet_names,
)


# ==========================================================
# READ SOURCE SHEET
# ==========================================================

try:

    raw_df = pd.read_excel(
        uploaded_file,
        sheet_name=selected_sheet,
    )

except Exception as e:

    st.error(
        f"Unable to read selected sheet: {e}"
    )

    st.stop()


if raw_df.empty:

    st.error(
        "The selected sheet is empty."
    )

    st.stop()


st.success(
    f"Loaded {selected_sheet}: "
    f"{raw_df.shape[0]:,} rows × "
    f"{raw_df.shape[1]:,} columns."
)


# ==========================================================
# MINIMUM DATA REQUIREMENT
# ==========================================================

st.subheader(
    "2. Cleaning Settings"
)

min_data_requirement = st.slider(
    "Minimum Data Requirement",
    min_value=0,
    max_value=100,
    value=30,
    step=5,
    format="%d%%",
    help=(
        "Minimum percentage of non-empty values "
        "required for a column to be retained."
    ),
)

min_data_threshold = int(
    np.ceil(
        len(raw_df)
        * min_data_requirement
        / 100
    )
)

st.caption(
    f"Current setting: **{min_data_requirement}%** "
    f"→ at least **{min_data_threshold:,} "
    f"of {len(raw_df):,} rows** must contain data."
)


# ==========================================================
# CLEAN DATA
# ==========================================================

try:

    clean_df = prepare_data(
        raw_df,
        min_data_requirement,
    )

except Exception as e:

    st.error(
        f"Cleaning failed: {e}"
    )

    st.stop()


# ==========================================================
# CLEANING SUMMARY
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Original Columns",
    raw_df.shape[1],
)

col2.metric(
    "Usable Columns",
    clean_df.shape[1],
)

col3.metric(
    "Rows",
    len(clean_df),
)

col4.metric(
    "Minimum Requirement",
    f"{min_data_requirement}%",
)


# ==========================================================
# DATE INFORMATION
# ==========================================================

if isinstance(
    clean_df.index,
    pd.DatetimeIndex,
):

    st.success(
        "Date column detected and used as the DataFrame index."
    )


# ==========================================================
# PREVIEW CLEAN DATA
# ==========================================================

with st.expander(
    "View Cleaned Data"
):

    st.dataframe(
        clean_df.head(20),
        use_container_width=True,
    )


# ==========================================================
# PERCENTILE CALCULATION
# ==========================================================

try:

    percentile_df = calculate_percentiles(
        clean_df,
        blocks=96,
    )

    percentile_df = (
        remove_constant_blocks(
            percentile_df
        )
    )

except Exception as e:

    st.error(
        f"Unable to calculate 95th percentile: {e}"
    )

    st.stop()


# ==========================================================
# COLUMN SELECTION
# ==========================================================

st.subheader(
    "3. Select Columns for Average"
)

all_columns = list(
    percentile_df.columns
)

selected_columns = st.multiselect(
    "Columns included in final average",
    options=all_columns,
    default=all_columns,
    help=(
        "Select the generation columns that should "
        "contribute to the final CMV average."
    ),
)

if not selected_columns:

    st.warning(
        "Select at least one column."
    )

    st.stop()


# ==========================================================
# PERCENTILE GRAPH
# ==========================================================

st.subheader(
    "4. 95th Percentile Profiles"
)

st.caption(
    "Use this chart to decide which columns "
    "should contribute to the final average."
)

fig_percentile = make_percentile_chart(
    percentile_df,
    selected_columns,
)

st.plotly_chart(
    fig_percentile,
    use_container_width=True,
)


# ==========================================================
# SELECTED DATA
# ==========================================================

selected_percentile_df = (
    percentile_df[
        selected_columns
    ]
    .copy()
)


# ==========================================================
# FINAL AVERAGE
# ==========================================================

average = (
    selected_percentile_df
    .mean(axis=1)
    .to_numpy()
)


# ==========================================================
# SMOOTH PROFILE
# ==========================================================

if len(average) < 15:

    st.error(
        "At least 15 blocks are required "
        "for smoothing."
    )

    st.stop()


smooth = generate_smooth_profile(
    average
)


# ==========================================================
# FINAL RESULT
# ==========================================================

st.subheader(
    "5. Generated CMV Curve"
)

final_output = pd.DataFrame({
    "Block": np.arange(
        1,
        len(smooth) + 1,
    ),
    "95th Percentile Average": average,
    "Smooth Profile": smooth,
})

fig_final = make_final_chart(
    selected_percentile_df,
    average,
    smooth,
)

st.plotly_chart(
    fig_final,
    use_container_width=True,
)


# ==========================================================
# RESULT TABLE
# ==========================================================

with st.expander(
    "View Generated Curve Data"
):

    st.dataframe(
        final_output,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# COLUMN SELECTION TABLE
# ==========================================================

selection_df = pd.DataFrame({
    "Column": all_columns,
    "Included in Average": [
        col in selected_columns
        for col in all_columns
    ],
})


with st.expander(
    "View Column Selection"
):

    st.dataframe(
        selection_df,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# EXCEL OUTPUT
# ==========================================================

st.subheader(
    "6. Update Original Workbook"
)

st.caption(
    "The selected source sheet is preserved. "
    "The generated CMV Curve, percentile data and "
    "column selection are added as separate sheets."
)


if st.button(
    "💾 Add CMV Curve & Download Updated Workbook",
    type="primary",
    use_container_width=True,
):

    try:

        # --------------------------------------------------
        # IMPORTANT:
        # Start from the ORIGINAL uploaded Excel bytes.
        # --------------------------------------------------

        output = BytesIO(
            uploaded_file.getvalue()
        )

        # --------------------------------------------------
        # Create output tables
        # --------------------------------------------------

        cmv_output = pd.DataFrame({
            "Block": np.arange(
                1,
                len(smooth) + 1,
            ),
            "95th Percentile Average": average,
            "Smooth Profile": smooth,
        })

        percentile_output = (
            percentile_df.copy()
        )

        percentile_output.insert(
            0,
            "Block",
            np.arange(
                1,
                len(percentile_output) + 1,
            ),
        )

        selection_output = pd.DataFrame({
            "Column": all_columns,
            "Included in Average": [
                col in selected_columns
                for col in all_columns
            ],
        })

        settings_output = pd.DataFrame({
            "Setting": [
                "Source Sheet",
                "Minimum Data Requirement",
                "Rows",
                "Original Columns",
                "Usable Columns",
            ],
            "Value": [
                selected_sheet,
                f"{min_data_requirement}%",
                len(raw_df),
                raw_df.shape[1],
                clean_df.shape[1],
            ],
        })

        # --------------------------------------------------
        # Update ORIGINAL workbook
        # --------------------------------------------------

        with pd.ExcelWriter(
            output,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:

            cmv_output.to_excel(
                writer,
                sheet_name="CMV_Curve",
                index=False,
            )

            percentile_output.to_excel(
                writer,
                sheet_name="Percentile_Data",
                index=False,
            )

            selection_output.to_excel(
                writer,
                sheet_name="Column_Selection",
                index=False,
            )

            settings_output.to_excel(
                writer,
                sheet_name="CMV_Settings",
                index=False,
            )

        output.seek(0)

        # --------------------------------------------------
        # Download name
        # --------------------------------------------------

        original_name = uploaded_file.name

        if original_name.lower().endswith(
            ".xlsx"
        ):

            download_name = (
                original_name[:-5]
                + "_Updated.xlsx"
            )

        else:

            download_name = (
                original_name
                + "_Updated.xlsx"
            )

        # --------------------------------------------------
        # Single download button
        # --------------------------------------------------

        st.success(
            "CMV Curve added successfully. "
            "All existing workbook sheets were preserved."
        )

        st.download_button(
            "⬇️ Download Updated Workbook",
            data=output.getvalue(),
            file_name=download_name,
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    except Exception as e:

        st.error(
            f"Unable to update workbook: {e}"
        )
