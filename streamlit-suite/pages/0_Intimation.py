import streamlit as st
import pandas as pd
from pathlib import Path

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Intimation Dashboard")
st.caption("Filter intimation records by POC1, State and PSS.")

# -------------------------------
# Backend Excel file
# Keep app.py and intimation.xlsx
# in the same project folder
# -------------------------------
EXCEL_PATH = Path(__file__).parent / "intimation.xlsx"

REQUIRED_COLUMNS = [
    "Lat",
    "Long",
    "S.No",
    "PSS",
    "State",
    "Capacity (MW)",
    "Priority as per Amresh Sir   28.07.2023",
    "Priority as per   003 sheet",
    "POC1",
    "POC 2",
    "Check Status",
]

@st.cache_data
def load_data(file_path, modified_time):
    df = pd.read_excel(file_path)

    # Remove accidental spaces in column headers
    df.columns = df.columns.astype(str).str.strip()

    # Validate required columns
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Excel columns: {missing}"
        )

    # Normalize filter/search fields
    for col in ["POC1", "State", "PSS"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


if not EXCEL_PATH.exists():
    st.error(
        f"Excel file not found: {EXCEL_PATH.name}. "
        "Place it in the same folder as app.py."
    )
    st.stop()

try:
    df = load_data(
        str(EXCEL_PATH),
        EXCEL_PATH.stat().st_mtime
    )
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

# -------------------------------
# Filters
# -------------------------------
st.subheader("Filters")

col1, col2 = st.columns(2)

poc_options = sorted(
    x for x in df["POC1"].unique() if x
)
state_options = sorted(
    x for x in df["State"].unique() if x
)

with col1:
    selected_pocs = st.multiselect(
        "POC1 / User",
        options=["All"] + poc_options,
        default=["All"],
        help="Select All or one or more users."
    )

with col2:
    selected_states = st.multiselect(
        "State",
        options=["All"] + state_options,
        default=["All"],
        help="Select All or one or more states."
    )

# -------------------------------
# PSS search
# -------------------------------
search_pss = st.text_input(
    "Search PSS",
    placeholder="Type PSS name or identifier..."
)

# -------------------------------
# Apply filters
# -------------------------------
filtered_df = df.copy()

if selected_pocs and "All" not in selected_pocs:
    filtered_df = filtered_df[
        filtered_df["POC1"].isin(selected_pocs)
    ]
elif not selected_pocs:
    filtered_df = filtered_df.iloc[0:0]

if selected_states and "All" not in selected_states:
    filtered_df = filtered_df[
        filtered_df["State"].isin(selected_states)
    ]
elif not selected_states:
    filtered_df = filtered_df.iloc[0:0]

if search_pss.strip():
    filtered_df = filtered_df[
        filtered_df["PSS"].str.contains(
            search_pss.strip(),
            case=False,
            na=False,
            regex=False
        )
    ]

# -------------------------------
# Results
# -------------------------------
st.divider()

m1, m2 = st.columns(2)

m1.metric("Matching Records", len(filtered_df))
m2.metric(
    "Total Capacity (MW)",
    round(
        pd.to_numeric(
            filtered_df["Capacity (MW)"],
            errors="coerce"
        ).sum(),
        2
    )
)

st.subheader("Intimation Records")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
    height=550
)

# -------------------------------
# Export filtered results
# -------------------------------
output = filtered_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="Download Filtered Results (CSV)",
    data=output,
    file_name="filtered_intimation.csv",
    mime="text/csv"
)
