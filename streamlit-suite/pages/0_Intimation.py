from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="📊",
    layout="wide",
)

EXCEL_PATH = Path(__file__).resolve().parent.parent / "intimation.xlsx"


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .dashboard-header {
        padding: 25px 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #123456 0%, #176B87 100%);
        color: white;
        margin-bottom: 25px;
    }

    .dashboard-header h1 {
        color: white;
        font-size: 2.1rem;
        margin: 0;
        font-weight: 750;
    }

    .dashboard-header p {
        color: #E2F2F8;
        margin: 8px 0 0 0;
        font-size: 1rem;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 18px 20px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.20);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }

    .section-heading {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 8px 0 14px 0;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    div.stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="dashboard-header">
        <h1>📊 Intimation Dashboard</h1>
        <p>Search, filter and analyse intimation records in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD EXCEL DATA
# --------------------------------------------------
@st.cache_data(ttl=300)
def load_data(file_path, modified_time):
    df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip()

    required_columns = ["POC1", "State", "PSS"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required Excel columns: " + ", ".join(missing)
        )

    for col in required_columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


if not EXCEL_PATH.exists():
    st.error("❌ intimation.xlsx was not found in the repository root.")
    st.info("Please upload the correct Excel workbook to your GitHub repository.")
    st.stop()

try:
    df = load_data(str(EXCEL_PATH), EXCEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Unable to load Excel file: {exc}")
    st.info(
        "Check that intimation.xlsx is a valid Excel workbook "
        "and not an empty or incorrectly uploaded file."
    )
    st.stop()


# --------------------------------------------------
# FILTERS
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">🔎 Find Intimation Records</div>',
    unsafe_allow_html=True,
)

filter_col1, filter_col2 = st.columns([1, 1])

poc_options = sorted(
    value for value in df["POC1"].unique() if value
)

state_options = sorted(
    value for value in df["State"].unique() if value
)

with filter_col1:
    selected_poc = st.selectbox(
        "👤 Select User (POC1)",
        options=["All Users"] + poc_options,
        index=0,
        help="Choose one user or view records for all users.",
        key="dashboard_poc",
    )

with filter_col2:
    selected_states = st.multiselect(
        "📍 Select State",
        options=state_options,
        default=state_options,
        placeholder="Choose one or more states",
        help="Leave all states selected to view all records.",
        key="dashboard_states",
    )

search_col, reset_col = st.columns([5, 1])

with search_col:
    search_pss = st.text_input(
        "⚡ Search PSS",
        placeholder="Type a PSS name or identifier...",
        key="dashboard_pss_search",
    )

with reset_col:
    st.write("")
    st.write("")
    if st.button("↻ Reset Filters", use_container_width=True):
        st.session_state["dashboard_poc"] = "All Users"
        st.session_state["dashboard_states"] = state_options
        st.session_state["dashboard_pss_search"] = ""
        st.rerun()


# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------
filtered_df = df.copy()

if selected_poc != "All Users":
    filtered_df = filtered_df[filtered_df["POC1"] == selected_poc]

if len(selected_states) != len(state_options):
    filtered_df = filtered_df[
        filtered_df["State"].isin(selected_states)
    ]

if search_pss.strip():
    filtered_df = filtered_df[
        filtered_df["PSS"].str.contains(
            search_pss.strip(),
            case=False,
            regex=False,
            na=False,
        )
    ]


# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------
capacity_col = "Capacity (MW)"

if capacity_col in filtered_df.columns:
    capacity_values = pd.to_numeric(
        filtered_df[capacity_col]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )
    total_capacity = capacity_values.sum()
else:
    total_capacity = 0

unique_pss = filtered_df["PSS"].replace("", pd.NA).nunique()
unique_users = filtered_df["POC1"].replace("", pd.NA).nunique()

st.markdown("---")
st.markdown(
    '<div class="section-heading">📈 Overview</div>',
    unsafe_allow_html=True,
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Matching Records",
    f"{len(filtered_df):,}",
)

metric2.metric(
    "Total Capacity",
    f"{total_capacity:,.2f} MW",
)

metric3.metric(
    "Unique PSS",
    f"{unique_pss:,}",
)

metric4.metric(
    "Users in Results",
    f"{unique_users:,}",
)


# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------
st.markdown("---")

result_col1, result_col2 = st.columns([3, 1])

with result_col1:
    st.markdown(
        '<div class="section-heading">📋 Intimation Records</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} total records"
    )

with result_col2:
    st.write("")
    st.download_button(
        label="⬇️ Download CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_intimation.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=filtered_df.empty,
    )

if filtered_df.empty:
    st.warning(
        "No records found. Try changing the selected user, state, or PSS search."
    )
else:
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        height=560,
    )

st.caption(
    "Data source: intimation.xlsx · Filters apply to the displayed results."
)
