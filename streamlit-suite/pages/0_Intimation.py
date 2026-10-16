
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------
st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Intimation Dashboard")
st.caption(
    "Filter intimation records by POC1, State and PSS."
)


# -----------------------------------------
# BACKEND EXCEL FILE
# Excel is stored in the repository root
# -----------------------------------------
from pathlib import Path
import zipfile
import streamlit as st

EXCEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "intimation.xlsx"
)

if not EXCEL_PATH.exists():
    st.error(f"File not found: {EXCEL_PATH}")
    st.stop()

with open(EXCEL_PATH, "rb") as f:
    header = f.read(32)

st.write("File size:", EXCEL_PATH.stat().st_size, "bytes")
st.write("File signature:", repr(header[:16]))
st.write("Valid XLSX ZIP container:", zipfile.is_zipfile(EXCEL_PATH))



# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data(ttl=300)
def load_data(file_path, modified_time):
    df = pd.read_excel(file_path, engine="openpyxl")

    # Clean header spacing
    df.columns = df.columns.astype(str).str.strip()

    # Remove accidental spaces from filter fields
    for col in ["POC1", "State", "PSS"]:
        if col not in df.columns:
            raise ValueError(
                f"Required column not found: {col}"
            )

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


if not EXCEL_PATH.exists():
    st.error(
        "intimation.xlsx was not found in the "
        "repository root."
    )
    st.stop()

try:
    df = load_data(
        str(EXCEL_PATH),
        EXCEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Unable to load Excel file: {exc}")
    st.stop()


# -----------------------------------------
# FILTER OPTIONS
# -----------------------------------------
poc_options = sorted(
    value
    for value in df["POC1"].unique()
    if value
)

state_options = sorted(
    value
    for value in df["State"].unique()
    if value
)


# -----------------------------------------
# FILTER UI
# -----------------------------------------
st.subheader("Filters")

col1, col2 = st.columns(2)

with col1:
    selected_pocs = st.multiselect(
        "POC1 / User",
        options=["All"] + poc_options,
        default=["All"],
        key="intimation_poc_filter",
    )

with col2:
    selected_states = st.multiselect(
        "State",
        options=["All"] + state_options,
        default=["All"],
        key="intimation_state_filter",
    )


search_pss = st.text_input(
    "Search PSS",
    placeholder="Enter PSS name or identifier...",
    key="intimation_pss_search",
)


# -----------------------------------------
# APPLY FILTERS
# -----------------------------------------
filtered_df = df.copy()

if "All" not in selected_pocs:
    if selected_pocs:
        filtered_df = filtered_df[
            filtered_df["POC1"].isin(selected_pocs)
        ]
    else:
        filtered_df = filtered_df.iloc[0:0]

if "All" not in selected_states:
    if selected_states:
        filtered_df = filtered_df[
            filtered_df["State"].isin(selected_states)
        ]
    else:
        filtered_df = filtered_df.iloc[0:0]

if search_pss.strip():
    filtered_df = filtered_df[
        filtered_df["PSS"].str.contains(
            search_pss.strip(),
            case=False,
            regex=False,
            na=False,
        )
    ]


# -----------------------------------------
# RESULTS SUMMARY
# -----------------------------------------
st.divider()

metric1, metric2 = st.columns(2)

metric1.metric(
    "Matching Records",
    f"{len(filtered_df):,}",
)

capacity = pd.to_numeric(
    filtered_df.get(
        "Capacity (MW)",
        pd.Series(dtype=float),
    ),
    errors="coerce",
).sum()

metric2.metric(
    "Total Capacity (MW)",
    f"{capacity:,.2f}",
)


# -----------------------------------------
# DISPLAY RESULTS
# -----------------------------------------
st.subheader("Intimation Records")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
    height=550,
)

# -----------------------------------------
# DOWNLOAD FILTERED DATA
# -----------------------------------------
csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="Download Filtered Results",
    data=csv_data,
    file_name="filtered_intimation.csv",
    mime="text/csv",
)
