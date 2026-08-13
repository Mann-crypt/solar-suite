import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from io import BytesIO
from scipy.signal import savgol_filter


# ==========================================================
# PAGE
# ==========================================================

st.set_page_config(
    page_title="CMV Curve Generator",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 12px;
}

.section {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 18px 0 8px;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 650;
    min-height: 42px;
}

div[data-testid="stDownloadButton"] > button {
    border-radius: 10px;
    font-weight: 650;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# HELPERS
# ==========================================================

def find_date_column(df):

    names = {
        "date",
        "datetime",
        "date time",
        "timestamp",
        "time",
    }

    for col in df.columns:

        key = str(col).strip().lower().replace("_", " ")

        if key in names:
            return col

    return None


# ==========================================================
# CACHED EXCEL READING
# ==========================================================

@st.cache_data(show_spinner=False, max_entries=10)
def read_excel_sheet(file_bytes, sheet_name):

    return pd.read_excel(
        BytesIO(file_bytes),
        sheet_name=sheet_name,
    )


@st.cache_data(show_spinner=False, max_entries=10)
def get_sheet_names(file_bytes):

    return pd.ExcelFile(
        BytesIO(file_bytes)
    ).sheet_names


# ==========================================================
# CACHED CLEANING
# ==========================================================

@st.cache_data(show_spinner=False, max_entries=20)
def prepare_data(
    raw_df,
    min_data_requirement,
    min_cap,
    max_cap,
):

    df = raw_df.copy()

    # Date
    date_col = find_date_column(df)

    if date_col:

        parsed = pd.to_datetime(
            df[date_col],
            errors="coerce",
        )

        if parsed.notna().any():

            df[date_col] = parsed
            df = df.set_index(date_col)

    # Convert numeric
    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(df[col]):

            converted = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            if converted.notna().sum() > 0:
                df[col] = converted

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric.empty:
        raise ValueError(
            "No numeric columns found."
        )

    threshold = int(
        np.ceil(
            len(numeric)
            * min_data_requirement
            / 100
        )
    )

    numeric = numeric.dropna(
        axis=1,
        thresh=threshold,
    )

    if numeric.empty:
        raise ValueError(
            "No columns satisfy the minimum data requirement."
        )

    numeric = numeric.fillna(0)

    # Generation cap
    max_values = numeric.max()

    numeric = numeric.loc[
        :,
        (max_values >= min_cap)
        & (max_values <= max_cap),
    ]

    # Remove zero columns
    numeric = numeric.loc[
        :,
        (numeric != 0).any(),
    ]

    if numeric.empty:
        raise ValueError(
            "No usable columns remain after cleaning."
        )

    return numeric


# ==========================================================
# CACHED PERCENTILE
# ==========================================================

@st.cache_data(show_spinner=False, max_entries=20)
def calculate_percentiles(
    df,
    blocks=96,
):

    usable = (
        len(df) // blocks
    ) * blocks

    if usable < blocks:
        raise ValueError(
            f"At least {blocks} rows are required."
        )

    arr = df.iloc[:usable].to_numpy(
        dtype=float
    )

    days = usable // blocks

    reshaped = arr.reshape(
        days,
        blocks,
        -1,
    )

    result = np.percentile(
        reshaped,
        95,
        axis=0,
    )

    return pd.DataFrame(
        result,
        columns=df.columns,
    )


# ==========================================================
# CONSTANT BLOCK REMOVAL
# ==========================================================

@st.cache_data(show_spinner=False, max_entries=20)
def remove_constant_blocks(df):

    result = df.copy()

    for col in result.columns:

        s = result[col]

        groups = s.ne(
            s.shift()
        ).cumsum()

        sizes = groups.map(
            groups.value_counts()
        )

        result[col] = s.mask(
            (sizes > 2) & (s != 0),
            0,
        )

    return result


# ==========================================================
# WINDOW VALIDATION
# ==========================================================

def valid_window(window, length):

    maximum = min(length, 51)

    if maximum % 2 == 0:
        maximum -= 1

    maximum = max(3, maximum)

    window = int(window)

    if window % 2 == 0:
        window -= 1

    return max(
        3,
        min(window, maximum),
    )


# ==========================================================
# CACHED SMOOTHING
# ==========================================================

@st.cache_data(show_spinner=False, max_entries=30)
def generate_smooth(
    average,
    window1,
    poly1,
    second,
    window2,
    poly2,
    threshold=4.9,
):

    values = np.asarray(
        average,
        dtype=float,
    )

    window1 = valid_window(
        window1,
        len(values),
    )

    poly1 = min(
        int(poly1),
        window1 - 1,
    )

    smooth = savgol_filter(
        values,
        window_length=window1,
        polyorder=poly1,
    )

    smooth = np.clip(
        smooth,
        0,
        None,
    )

    smooth[smooth < threshold] = 0

    if second:

        window2 = valid_window(
            window2,
            len(values),
        )

        poly2 = min(
            int(poly2),
            window2 - 1,
        )

        smooth = savgol_filter(
            smooth,
            window_length=window2,
            polyorder=poly2,
        )

        smooth = np.clip(
            smooth,
            0,
            None,
        )

    return smooth


# ==========================================================
# CHARTS
# ==========================================================

def percentile_chart(
    df,
    selected,
):

    fig = go.Figure()

    x = np.arange(
        1,
        len(df) + 1,
    )

    for col in df.columns:

        active = col in selected

        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col],
                name=str(col),
                mode="lines",
                line=dict(
                    width=3 if active else 1,
                ),
                opacity=1 if active else 0.25,
            )
        )

    fig.update_layout(
        height=500,
        template="streamlit",
        xaxis_title="15 Minute Block",
        yaxis_title="95th Percentile Power",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20,
        ),
    )

    return fig


def final_chart(
    average,
    smooth,
):

    x = np.arange(
        1,
        len(average) + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=average,
            name="95th Percentile Average",
            mode="lines",
            line=dict(width=3),
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
        height=500,
        template="streamlit",
        xaxis_title="15 Minute Block",
        yaxis_title="Power",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20,
        ),
    )

    return fig


# ==========================================================
# HEADER
# ==========================================================

st.title("📡 CMV Curve Generator")

st.caption(
    "Generate a 95th-percentile CMV generation profile "
    "from historical generation data."
)


# ==========================================================
# UPLOAD
# ==========================================================

st.markdown(
    '<div class="section">📁 Import Workbook</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload Excel Workbook",
    type=["xlsx"],
    key="cmv_upload",
)

if uploaded is None:

    st.info(
        "Upload an Excel workbook to begin."
    )

    st.stop()


file_bytes = uploaded.getvalue()


# ==========================================================
# SHEET
# ==========================================================

try:

    sheets = get_sheet_names(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


selected_sheet = st.selectbox(
    "CMV Source Sheet",
    sheets,
)


# ==========================================================
# READ RAW DATA
# ==========================================================

try:

    raw_df = read_excel_sheet(
        file_bytes,
        selected_sheet,
    )

except Exception as e:

    st.error(
        f"Unable to read sheet: {e}"
    )

    st.stop()


if raw_df.empty:

    st.error(
        "Selected sheet is empty."
    )

    st.stop()


# ==========================================================
# SETTINGS FORM
# ==========================================================

st.markdown(
    '<div class="section">⚙️ Processing Settings</div>',
    unsafe_allow_html=True,
)

with st.form("cmv_settings"):

    c1, c2, c3 = st.columns(3)

    with c1:

        min_data = st.slider(
            "Minimum Data",
            0,
            100,
            30,
            5,
            format="%d%%",
        )

    with c2:

        min_cap = st.number_input(
            "Minimum Generation Cap",
            min_value=0.0,
            value=800.0,
            step=10.0,
        )

    with c3:

        max_cap = st.number_input(
            "Maximum Generation Cap",
            min_value=0.0,
            value=1200.0,
            step=10.0,
        )

    st.markdown("")

    c1, c2, c3 = st.columns(3)

    with c1:

        first_window = st.number_input(
            "1st Window",
            min_value=3,
            max_value=51,
            value=15,
            step=2,
        )

    with c2:

        first_poly = st.number_input(
            "1st Polynomial",
            min_value=1,
            max_value=10,
            value=3,
        )

    with c3:

        second_enabled = st.checkbox(
            "Apply Second Smoothing",
            value=True,
        )

    if second_enabled:

        c1, c2 = st.columns(2)

        with c1:

            second_window = st.number_input(
                "2nd Window",
                min_value=3,
                max_value=51,
                value=7,
                step=2,
            )

        with c2:

            second_poly = st.number_input(
                "2nd Polynomial",
                min_value=1,
                max_value=10,
                value=3,
            )

    else:

        second_window = 7
        second_poly = 3

    generate = st.form_submit_button(
        "🚀 Generate CMV Curve",
        type="primary",
        use_container_width=True,
    )


# ==========================================================
# GENERATE ONLY AFTER BUTTON
# ==========================================================

if not generate:

    st.info(
        "Configure the settings above and click "
        "**Generate CMV Curve**."
    )

    st.stop()


# ==========================================================
# VALIDATION
# ==========================================================

if min_cap >= max_cap:

    st.error(
        "Minimum Generation Cap must be less than Maximum Generation Cap."
    )

    st.stop()


# ==========================================================
# CLEAN
# ==========================================================

with st.spinner("Cleaning generation data..."):

    try:

        clean_df = prepare_data(
            raw_df,
            min_data,
            min_cap,
            max_cap,
        )

    except Exception as e:

        st.error(
            f"Cleaning failed: {e}"
        )

        st.stop()


# ==========================================================
# PERCENTILE
# ==========================================================

with st.spinner("Calculating 95th percentile..."):

    try:

        percentile_df = calculate_percentiles(
            clean_df,
            96,
        )

        percentile_df = (
            remove_constant_blocks(
                percentile_df
            )
        )

    except Exception as e:

        st.error(
            f"Percentile calculation failed: {e}"
        )

        st.stop()


# ==========================================================
# SUMMARY
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
    "Data Requirement",
    f"{min_data}%",
)


# ==========================================================
# COLUMN SELECTION
# ==========================================================

st.markdown(
    '<div class="section">📊 Column Selection</div>',
    unsafe_allow_html=True,
)

all_columns = list(
    percentile_df.columns
)

selected_columns = st.multiselect(
    "Columns included in average",
    all_columns,
    default=all_columns,
)

if not selected_columns:

    st.warning(
        "Select at least one column."
    )

    st.stop()


# ==========================================================
# OPTIONAL PERCENTILE CHART
# ==========================================================

with st.expander(
    "📈 View 95th Percentile Profiles"
):

    st.caption(
        "Selected columns are highlighted."
    )

    st.plotly_chart(
        percentile_chart(
            percentile_df,
            selected_columns,
        ),
        use_container_width=True,
    )


# ==========================================================
# AVERAGE
# ==========================================================

average = (
    percentile_df[
        selected_columns
    ]
    .mean(axis=1)
    .to_numpy()
)


# ==========================================================
# SMOOTH
# ==========================================================

with st.spinner("Generating smooth CMV profile..."):

    smooth = generate_smooth(
        tuple(average),
        first_window,
        first_poly,
        second_enabled,
        second_window,
        second_poly,
    )


# ==========================================================
# FINAL CHART
# ==========================================================

st.markdown(
    '<div class="section">☀️ Generated CMV Curve</div>',
    unsafe_allow_html=True,
)

st.plotly_chart(
    final_chart(
        average,
        smooth,
    ),
    use_container_width=True,
)


# ==========================================================
# RESULT TABLE
# ==========================================================

final_output = pd.DataFrame({
    "Block": np.arange(
        1,
        len(smooth) + 1,
    ),
    "95th Percentile Average": average,
    "Smooth Profile": smooth,
})

with st.expander(
    "🔍 View Generated Curve Data"
):

    st.dataframe(
        final_output.round(3),
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# SETTINGS OUTPUT
# ==========================================================

settings_output = pd.DataFrame({
    "Setting": [
        "Source Sheet",
        "Minimum Data Requirement",
        "Minimum Generation Cap",
        "Maximum Generation Cap",
        "First Window Length",
        "First Polynomial Order",
        "Second Smoothing",
        "Second Window Length",
        "Second Polynomial Order",
        "Rows",
        "Original Columns",
        "Usable Columns",
    ],
    "Value": [
        selected_sheet,
        f"{min_data}%",
        min_cap,
        max_cap,
        first_window,
        first_poly,
        "Enabled" if second_enabled else "Disabled",
        second_window if second_enabled else "",
        second_poly if second_enabled else "",
        len(raw_df),
        raw_df.shape[1],
        clean_df.shape[1],
    ],
})


# ==========================================================
# EXCEL DOWNLOAD
# ==========================================================

st.markdown(
    '<div class="section">💾 Export</div>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=10)
def create_output_excel(
    original_bytes,
    cmv_output,
    percentile_output,
    selection_output,
    settings_output,
):

    output = BytesIO(
        original_bytes
    )

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

    return output.getvalue()


cmv_output = final_output.copy()

percentile_output = percentile_df.copy()

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
        c in selected_columns
        for c in all_columns
    ],
})


excel_output = create_output_excel(
    file_bytes,
    cmv_output,
    percentile_output,
    selection_output,
    settings_output,
)

download_name = (
    uploaded.name.rsplit(".", 1)[0]
    + "_Updated.xlsx"
)

st.download_button(
    "⬇️ Download Updated Workbook",
    data=excel_output,
    file_name=download_name,
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    use_container_width=True,
)
