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
    """Find a likely date/datetime column."""

    possible_names = [
        "date",
        "datetime",
        "date time",
        "timestamp",
        "time",
    ]

    normalized = {
        str(col).strip().lower().replace("_", " "): col
        for col in df.columns
    }

    for name in possible_names:
        if name in normalized:
            return normalized[name]

    return None


def prepare_data(df, min_data_requirement):
    """Clean raw CMV data."""

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

        if parsed_date.notna().any():
            df[date_col] = parsed_date
            df = df.set_index(date_col)

    # ------------------------------------------------------
    # Convert possible numeric columns
    # ------------------------------------------------------

    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(df[col]):

            converted = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            if converted.notna().sum() > 0:
                df[col] = converted

    # ------------------------------------------------------
    # Keep numeric columns
    # ------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.empty:
        raise ValueError(
            "No numeric columns were found in the workbook."
        )

    # ------------------------------------------------------
    # Minimum data requirement
    # ------------------------------------------------------

    threshold = int(
        np.ceil(
            len(numeric_df)
            * min_data_requirement
            / 100
        )
    )

    numeric_df = numeric_df.dropna(
        axis=1,
        thresh=threshold,
    )

    if numeric_df.empty:
        raise ValueError(
            f"No columns satisfy the selected "
            f"{min_data_requirement}% minimum data requirement."
        )

    # ------------------------------------------------------
    # Fill missing values
    # ------------------------------------------------------

    numeric_df = numeric_df.fillna(0)

    # ------------------------------------------------------
    # Remove abnormal columns
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
            "No usable generation columns remain after cleaning."
        )

    return numeric_df


def calculate_percentiles(df, blocks=96):
    """Calculate 95th percentile for each 15-minute block."""

    usable_rows = (
        len(df) // blocks
    ) * blocks

    if usable_rows < blocks:
        raise ValueError(
            f"At least {blocks} rows are required."
        )

    df = df.iloc[:usable_rows].copy()

    days = usable_rows // blocks

    result = {}

    for col in df.columns:

        values = df[col].to_numpy(
            dtype=float
        )

        reshaped = values.reshape(
            days,
            blocks,
        )

        result[col] = np.percentile(
            reshaped,
            95,
            axis=0,
        )

    return pd.DataFrame(result)


def remove_constant_blocks(percentile_df):
    """Replace consecutive equal percentile values with zero."""

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
    window_length=15,
    polyorder=3,
):
    """Generate smooth CMV profile."""

    values = np.asarray(
        average,
        dtype=float,
    )

    # Make sure window is valid
    max_window = len(values)

    if max_window % 2 == 0:
        max_window -= 1

    window_length = min(
        window_length,
        max_window,
    )

    if window_length <= polyorder:
        window_length = polyorder + 2

        if window_length % 2 == 0:
            window_length += 1

    if window_length > max_window:
        window_length = max_window

    # First smoothing
    smooth = savgol_filter(
        values,
        window_length=window_length,
        polyorder=polyorder,
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
    second_window = min(
        7,
        max_window,
    )

    if second_window % 2 == 0:
        second_window -= 1

    if second_window > polyorder:

        smooth = savgol_filter(
            smooth,
            window_length=second_window,
            polyorder=polyorder,
        )

    return np.clip(
        smooth,
        0,
        None,
    )


def make_percentile_chart(
    percentile_df,
    selected_columns,
):
    """Chart used for deciding which columns to include."""

    fig = go.Figure()

    x = np.arange(
        1,
        len(percentile_df) + 1,
    )

    for col in percentile_df.columns:

        selected = col in selected_columns

        fig.add_trace(
            go.Scatter(
                x=x,
                y=percentile_df[col],
                name=str(col),
                mode="lines",
                line=dict(
                    width=3 if selected else 1.5
                ),
                opacity=1 if selected else 0.35,
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
            y=-0.18,
            x=0,
        ),
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=100,
        ),
        hoverlabel=dict(
            namelength=-1,
        ),
    )

    return fig


def make_final_chart(
    average,
    smooth,
):
    """Final CMV chart. Only average and smooth profile."""

    fig = go.Figure()

    x = np.arange(
        1,
        len(average) + 1,
    )

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
            y=-0.18,
            x=0,
        ),
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=100,
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
    st.info("Upload an Excel workbook to begin.")
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
# CLEANING SETTINGS
# ==========================================================

st.subheader("2. Cleaning Settings")

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

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Original Columns",
    raw_df.shape[1],
)

c2.metric(
    "Usable Columns",
    clean_df.shape[1],
)

c3.metric(
    "Rows",
    len(clean_df),
)

c4.metric(
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
# CLEAN DATA PREVIEW
# ==========================================================

with st.expander("View Cleaned Data"):

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

    percentile_df = remove_constant_blocks(
        percentile_df
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
        "Select the generation columns that "
        "should contribute to the final CMV average."
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
    "Hover over a legend item to highlight that "
    "column's profile."
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

selected_percentile_df = percentile_df[
    selected_columns
].copy()


# ==========================================================
# FINAL AVERAGE
# ==========================================================

average = (
    selected_percentile_df
    .mean(axis=1)
    .to_numpy()
)


# ==========================================================
# SMOOTHING SETTINGS
# ==========================================================

st.subheader(
    "5. Smoothing Settings"
)

if "cmv_window" not in st.session_state:
    st.session_state.cmv_window = 15


def decrease_window():

    st.session_state.cmv_window = max(
        5,
        st.session_state.cmv_window - 2,
    )


def increase_window():

    st.session_state.cmv_window += 2


s1, s2, s3 = st.columns(
    [1, 2, 1]
)

with s1:

    st.button(
        "−",
        key="window_minus",
        on_click=decrease_window,
        use_container_width=True,
    )

with s2:

    st.number_input(
        "Window Length",
        min_value=5,
        max_value=95,
        step=2,
        key="cmv_window",
        label_visibility="collapsed",
    )

with s3:

    st.button(
        "+",
        key="window_plus",
        on_click=increase_window,
        use_container_width=True,
    )


window_length = st.session_state.cmv_window

# Make sure window does not exceed available blocks
max_window = len(average)

if max_window % 2 == 0:
    max_window -= 1

window_length = min(
    window_length,
    max_window,
)

if window_length <= 3:

    st.error(
        "Window length is too small for smoothing."
    )

    st.stop()


st.caption(
    f"Current Savitzky-Golay window: "
    f"**{window_length} blocks** "
    f"({window_length * 15} minutes)"
)


# ==========================================================
# GENERATE SMOOTH PROFILE
# ==========================================================

try:

    smooth = generate_smooth_profile(
        average,
        window_length=window_length,
        polyorder=3,
    )

except Exception as e:

    st.error(
        f"Smoothing failed: {e}"
    )

    st.stop()


# ==========================================================
# GENERATED CMV CURVE
# ==========================================================

st.subheader(
    "6. Generated CMV Curve"
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
    "7. Update Original Workbook"
)

st.caption(
    "Adds/replaces CMV_Curve, Percentile_Data, "
    "Column_Selection and CMV_Settings sheets "
    "while preserving the existing workbook."
)


if st.button(
    "💾 Add CMV Curve & Download Updated Workbook",
    type="primary",
    use_container_width=True,
):

    try:

        # --------------------------------------------------
        # ORIGINAL WORKBOOK
        # --------------------------------------------------

        output = BytesIO(
            uploaded_file.getvalue()
        )

        # --------------------------------------------------
        # CMV OUTPUT
        # --------------------------------------------------

        cmv_output = pd.DataFrame({
            "Block": np.arange(
                1,
                len(smooth) + 1,
            ),
            "95th Percentile Average": average,
            "Smooth Profile": smooth,
        })

        # --------------------------------------------------
        # PERCENTILE OUTPUT
        # --------------------------------------------------

        percentile_output = percentile_df.copy()

        percentile_output.insert(
            0,
            "Block",
            np.arange(
                1,
                len(percentile_output) + 1,
            ),
        )

        # --------------------------------------------------
        # COLUMN SELECTION
        # --------------------------------------------------

        selection_output = pd.DataFrame({
            "Column": all_columns,
            "Included in Average": [
                col in selected_columns
                for col in all_columns
            ],
        })

        # --------------------------------------------------
        # SETTINGS
        # --------------------------------------------------

        settings_output = pd.DataFrame({
            "Setting": [
                "Source Sheet",
                "Minimum Data Requirement",
                "Smoothing Window Length",
                "Smoothing Window Minutes",
                "Polynomial Order",
                "Rows",
                "Original Columns",
                "Usable Columns",
                "Selected Columns",
            ],
            "Value": [
                selected_sheet,
                f"{min_data_requirement}%",
                window_length,
                window_length * 15,
                3,
                len(raw_df),
                raw_df.shape[1],
                clean_df.shape[1],
                len(selected_columns),
            ],
        })

        # --------------------------------------------------
        # UPDATE ORIGINAL WORKBOOK
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
        # DOWNLOAD NAME
        # --------------------------------------------------

        original_name = uploaded_file.name

        if original_name.lower().endswith(".xlsx"):

            download_name = (
                original_name[:-5]
                + "_Updated.xlsx"
            )

        else:

            download_name = (
                original_name
                + "_Updated.xlsx"
            )

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
