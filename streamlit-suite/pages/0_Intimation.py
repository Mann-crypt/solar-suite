from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="📊",
    layout="wide",
)

EXCEL_PATH = Path(__file__).resolve().parent.parent / "intimation.xlsx"


# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    .dashboard-header {
        background: linear-gradient(120deg, #123456, #167D9A);
        padding: 24px 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 22px;
    }

    .dashboard-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 750;
    }

    .dashboard-header p {
        color: #E4F5FA;
        margin: 7px 0 0 0;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 9px;
    }

    div[data-testid="stSelectbox"] > div > div {
        border-radius: 9px;
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
        <p>Explore and filter intimation records by user, PSS and state.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data(ttl=300)
def load_data(file_path, modified_time):
    data = pd.read_excel(file_path, engine="openpyxl")
    data.columns = data.columns.astype(str).str.strip()

    required = [
        "Lat",
        "Long",
        "PSS",
        "State",
        "Capacity (MW)",
        "POC1",
        "POC 2",
        "POC 3",
        "Check Status",
    ]

    # Match the two priority columns in the source workbook.
    priority_columns = [
        col for col in data.columns
        if col.lower().startswith("priority")
    ]

    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(
            "Required Excel columns missing: " + ", ".join(missing)
        )

    for col in ["PSS", "State", "POC1", "POC 2", "POC 3", "Check Status"]:
        data[col] = data[col].fillna("").astype(str).str.strip()

    # Use the first available priority column for the display.
    # If your workbook has both priority columns, choose the intended one below.
    if priority_columns:
        data["Priority"] = data[priority_columns[0]]
    else:
        data["Priority"] = ""

    return data


if not EXCEL_PATH.exists():
    st.error("intimation.xlsx was not found in the repository root.")
    st.stop()

try:
    df = load_data(str(EXCEL_PATH), EXCEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Unable to load Excel file: {exc}")
    st.stop()


# --------------------------------------------------
# FILTERS
# --------------------------------------------------
st.subheader("🔎 Filters")

filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.5])

user_options = ["All Users"] + sorted(
    x for x in df["POC1"].unique() if x
)

with filter_col1:
    selected_user = st.selectbox(
        "POC1 / User",
        user_options,
        key="intimation_user",
    )

with filter_col3:
    pss_search = st.text_input(
        "Search PSS",
        placeholder="Enter PSS name...",
        key="intimation_pss",
    )


# Apply user and PSS first so the State choices stay relevant.
base_df = df.copy()

if selected_user != "All Users":
    base_df = base_df[base_df["POC1"] == selected_user]

if pss_search.strip():
    base_df = base_df[
        base_df["PSS"].str.contains(
            pss_search.strip(),
            case=False,
            regex=False,
            na=False,
        )
    ]

available_states = sorted(
    x for x in base_df["State"].unique() if x
)

with filter_col2:
    selected_states = st.multiselect(
        "State",
        options=available_states,
        default=available_states,
        placeholder="Select states",
        key="intimation_states",
    )


# Apply state filter to the already filtered results.
filtered_df = base_df[
    base_df["State"].isin(selected_states)
].copy()


# --------------------------------------------------
# DISPLAY SELECTED COLUMNS ONLY
# --------------------------------------------------
display_columns = [
    "Lat",
    "Long",
    "PSS",
    "State",
    "Capacity (MW)",
    "Priority",
    "POC1",
    "POC 2",
    "POC 3",
    "Check Status",
]

display_df = filtered_df[display_columns].copy()

st.write("Total Excel records:", len(df))
st.write("Records after filters:", len(filtered_df))
st.write("Records for display:", len(display_df))
st.write("Display columns:", list(display_df.columns))
# --------------------------------------------------
# COMPACT INTIMATION CARDS
# --------------------------------------------------
st.markdown("---")
st.subheader("📋 Intimation Records")

st.caption(
    f"{len(display_df):,} matching records out of {len(df):,} total records"
)

# Card styling
st.markdown(
    """
    <style>
    .record-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
        background: var(--secondary-background-color);
        min-height: 210px;
    }

    .record-title {
        font-size: 1.05rem;
        font-weight: 750;
        overflow-wrap: anywhere;
        margin-bottom: 5px;
    }

    .record-state {
        font-size: 0.83rem;
        opacity: 0.75;
        margin-bottom: 12px;
    }

    .record-status {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        background: rgba(100, 116, 139, 0.15);
        overflow-wrap: anywhere;
        max-width: 100%;
    }

    .record-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 12px;
    }

    .record-label {
        font-size: 0.72rem;
        opacity: 0.7;
        margin-bottom: 3px;
    }

    .record-value {
        font-size: 0.88rem;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .record-coordinates {
        font-size: 0.78rem;
        opacity: 0.8;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_text(value):
    if pd.isna(value):
        return "—"
    value = str(value).strip()
    return value if value else "—"

# --------------------------------------------------
# OVERVIEW AT THE BOTTOM
# --------------------------------------------------
st.markdown("---")
st.subheader("📈 Overview")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Matching Records",
    f"{len(filtered_df):,}",
)

capacity = pd.to_numeric(
    filtered_df["Capacity (MW)"]
    .astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce",
).sum()

metric2.metric(
    "Total Capacity",
    f"{capacity:,.2f} MW",
)

metric3.metric(
    "Unique PSS",
    f"{filtered_df['PSS'].replace('', pd.NA).nunique():,}",
)
