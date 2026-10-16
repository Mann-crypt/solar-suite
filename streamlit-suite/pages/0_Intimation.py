
from pathlib import Path
from html import escape
from textwrap import dedent

import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXCEL_PATH = Path(__file__).resolve().parent.parent / "intimation.xlsx"


# =========================================================
# CUSTOM CSS
# =========================================================
st.html("""
<style>
.block-container {
    max-width: 1550px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.dashboard-header {
    background: linear-gradient(120deg, #102A43, #176B87 60%, #218CA5);
    padding: 22px 26px;
    border-radius: 15px;
    margin-bottom: 18px;
    box-shadow: 0 4px 15px rgba(15, 50, 75, 0.10);
}

.dashboard-header h1 {
    color: white !important;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
}

.dashboard-header p {
    color: #D9EEF5 !important;
    margin: 6px 0 0;
    font-size: 0.9rem;
}

.section-heading {
    font-size: 1.08rem;
    font-weight: 750;
    margin: 8px 0 4px;
}

.section-description {
    font-size: 0.82rem;
    opacity: 0.72;
    margin-bottom: 12px;
}

/* Overview metrics */
div[data-testid="stMetric"] {
    padding: 12px;
    border: 1px solid rgba(128, 128, 128, 0.20);
    border-radius: 11px;
    background: var(--secondary-background-color);
}

div[data-testid="stMetricLabel"] {
    font-weight: 650;
    font-size: 0.78rem;
}

div[data-testid="stMetricValue"] {
    font-size: 1.45rem;
}

/* Ultra-compact card */
.record-card {
    border: 1px solid rgba(128, 128, 128, 0.20);
    border-radius: 10px;
    padding: 9px 10px;
    margin-bottom: 4px;
    background: var(--secondary-background-color);
    overflow-wrap: anywhere;
    transition: border-color 0.2s ease;
}

.record-card:hover {
    border-color: #2693AD;
}

.record-top {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 7px;
}

.record-icon {
    flex-shrink: 0;
    width: 29px;
    height: 29px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(38, 147, 173, 0.13);
    font-size: 16px;
}

.record-title {
    font-size: 0.90rem;
    font-weight: 800;
    line-height: 1.3;
    margin: 0;
    overflow-wrap: anywhere;
}

.record-state {
    font-size: 0.73rem;
    opacity: 0.72;
    margin-top: 2px;
}

.record-primary {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
}

.record-primary-item {
    min-width: 0;
    padding: 6px 7px;
    border-radius: 6px;
    background: rgba(128, 128, 128, 0.07);
}

.record-label {
    font-size: 0.61rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.35px;
    opacity: 0.65;
    margin-bottom: 2px;
}

.record-value {
    font-size: 0.78rem;
    font-weight: 700;
    line-height: 1.3;
    overflow-wrap: anywhere;
    word-break: break-word;
}

/* Compact details dropdown */
div[data-testid="stExpander"] {
    border: none;
    margin-top: -3px;
    margin-bottom: 7px;
}

div[data-testid="stExpander"] details {
    border: none;
}

div[data-testid="stExpander"] summary {
    font-size: 0.76rem;
    font-weight: 650;
    padding: 5px 8px;
    border-radius: 6px;
    background: rgba(128, 128, 128, 0.07);
}

div[data-testid="stExpander"] summary:hover {
    color: #2693AD;
}

.detail-label {
    font-size: 0.74rem;
    font-weight: 700;
    opacity: 0.7;
    margin-top: 8px;
    margin-bottom: 2px;
}

.detail-value {
    font-size: 0.85rem;
    overflow-wrap: anywhere;
    word-break: break-word;
}

div[data-testid="stTextInput"] input {
    border-radius: 9px;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {
    border-radius: 9px;
}

@media (max-width: 600px) {
    .dashboard-header {
        padding: 17px;
    }

    .dashboard-header h1 {
        font-size: 1.5rem;
    }

    .record-card {
        padding: 8px;
    }

    .record-primary-item {
        padding: 5px;
    }
}
</style>
""")


# =========================================================
# HEADER
# =========================================================
st.html("""
<div class="dashboard-header">
    <h1>📊 Intimation Dashboard</h1>
    <p>Search, filter and review intimation records.</p>
</div>
""")


# =========================================================
# LOAD EXCEL DATA
# =========================================================
@st.cache_data(ttl=300)
def load_data(file_path, modified_time):
    data = pd.read_excel(file_path, engine="openpyxl")
    data.columns = data.columns.astype(str).str.strip()

    required_columns = [
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

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing Excel columns: " + ", ".join(missing_columns)
        )

    text_columns = [
        "PSS",
        "State",
        "POC1",
        "POC 2",
        "POC 3",
        "Check Status",
    ]

    for col in text_columns:
        data[col] = (
            data[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    priority_candidates = [
        "Priority as per 003 sheet",
        "Priority as per Amresh Sir 28.07.2023",
    ]

    priority_column = next(
        (
            col for col in priority_candidates
            if col in data.columns
        ),
        None,
    )

    if priority_column is None:
        priority_columns = [
            col for col in data.columns
            if col.lower().startswith("priority")
        ]
        priority_column = (
            priority_columns[0] if priority_columns else None
        )

    data["Priority"] = (
        data[priority_column]
        if priority_column
        else ""
    )

    return data


if not EXCEL_PATH.exists():
    st.error(
        "intimation.xlsx was not found in the repository root."
    )
    st.stop()

try:
    df = load_data(
        str(EXCEL_PATH),
        EXCEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Unable to load Excel data: {exc}")
    st.info(
        "Check that intimation.xlsx is a valid Excel workbook "
        "and contains the required columns."
    )
    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def safe_text(value):
    if pd.isna(value):
        return "—"

    value = str(value).strip()
    return value if value else "—"


def safe_number(value, decimals=2):
    number = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(number):
        return safe_text(value)

    return f"{number:,.{decimals}f}"


def show_detail(label, value):
    st.markdown(
        f'<div class="detail-label">{escape(label)}</div>'
        f'<div class="detail-value">{escape(safe_text(value))}</div>',
        unsafe_allow_html=True,
    )


def make_card(row):
    def val(column):
        value = row.get(column, "")

        if pd.isna(value) or str(value).strip() == "":
            return "—"

        return escape(str(value).strip())

    pss = val("PSS")
    state = val("State")
    poc1 = val("POC1")

    capacity = escape(
        safe_number(row.get("Capacity (MW)", ""), 2)
    )

    return dedent(f"""
    <div class="record-card">
        <div class="record-top">
            <div class="record-icon">⚡</div>
            <div>
                <div class="record-title">{pss}</div>
                <div class="record-state">📍 {state}</div>
            </div>
        </div>

        <div class="record-primary">
            <div class="record-primary-item">
                <div class="record-label">POC1</div>
                <div class="record-value">{poc1}</div>
            </div>

            <div class="record-primary-item">
                <div class="record-label">Capacity</div>
                <div class="record-value">{capacity} MW</div>
            </div>
        </div>
    </div>
    """).strip()


# =========================================================
# FILTER CALLBACKS
# =========================================================
def reset_dependent_filters():
    st.session_state.pop("intimation_states", None)
    st.session_state["intimation_card_page"] = 0


def reset_page():
    st.session_state["intimation_card_page"] = 0


# =========================================================
# FILTER OPTIONS
# =========================================================
user_options = ["All Users"] + sorted(
    value for value in df["POC1"].unique()
    if value
)

if st.session_state.get("intimation_user") not in user_options:
    st.session_state["intimation_user"] = "All Users"


# =========================================================
# FILTERS
# =========================================================
st.html("""
<div class="section-heading">🔎 Find Records</div>
<div class="section-description">
Choose a POC1 user, search for a PSS and filter by state.
</div>
""")

filter_col1, filter_col2, filter_col3 = st.columns(
    [1, 1.1, 1.5],
    gap="small",
)

with filter_col1:
    selected_user = st.selectbox(
        "👤 POC1 / User",
        options=user_options,
        key="intimation_user",
        on_change=reset_dependent_filters,
    )

with filter_col3:
    pss_search = st.text_input(
        "🔍 Search PSS",
        placeholder="Enter PSS name...",
        key="intimation_pss",
        on_change=reset_dependent_filters,
    )


# Filter by user and PSS before calculating state options.
base_df = df.copy()

if selected_user != "All Users":
    base_df = base_df[
        base_df["POC1"] == selected_user
    ]

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
    value for value in base_df["State"].unique()
    if value
)

# Keep only state selections that remain valid.
if "intimation_states" in st.session_state:
    st.session_state["intimation_states"] = [
        state
        for state in st.session_state["intimation_states"]
        if state in available_states
    ]

with filter_col2:
    selected_states = st.multiselect(
        "📍 State",
        options=available_states,
        default=available_states,
        key="intimation_states",
        placeholder="Select states",
        on_change=reset_page,
    )

filtered_df = base_df[
    base_df["State"].isin(selected_states)
].copy()


# =========================================================
# OVERVIEW — AT THE TOP, UPDATES WITH FILTERS
# =========================================================
capacity_values = pd.to_numeric(
    filtered_df["Capacity (MW)"]
    .astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce",
)

total_capacity = capacity_values.sum()
unique_pss = filtered_df["PSS"].replace("", pd.NA).nunique()

st.html("""
<div class="section-heading">📈 Overview</div>
<div class="section-description">
Live summary of the currently filtered records.
</div>
""")

metric1, metric2, metric3 = st.columns(3, gap="small")

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


# =========================================================
# INTIMATION CARDS
# =========================================================
st.divider()

st.html("""
<div class="section-heading">📋 Intimation Records</div>
""")

st.caption(
    f"Showing {len(filtered_df):,} matching records "
    f"out of {len(df):,} total records."
)

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

if display_df.empty:
    st.info(
        "No matching records found. Try changing the user, "
        "PSS search or selected states."
    )

else:
    cards_per_page = 20

    total_pages = (
        len(display_df) + cards_per_page - 1
    ) // cards_per_page

    current_page = st.session_state.get(
        "intimation_card_page", 0
    )
    current_page = max(
        0,
        min(current_page, total_pages - 1),
    )
    st.session_state["intimation_card_page"] = current_page

    page_col1, page_col2 = st.columns([3, 1])

    start = current_page * cards_per_page
    end = min(start + cards_per_page, len(display_df))

    with page_col1:
        st.caption(
            f"Records {start + 1}–{end} of {len(display_df):,}"
        )

    with page_col2:
        selected_page = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            index=current_page,
            key="intimation_card_page_selector",
            label_visibility="collapsed",
        )

    current_page = selected_page - 1
    st.session_state["intimation_card_page"] = current_page

    start = current_page * cards_per_page
    page_df = display_df.iloc[start:start + cards_per_page]
    records = list(page_df.iterrows())

    for index in range(0, len(records), 2):
        left_col, right_col = st.columns(2, gap="small")

        for offset, col in enumerate([left_col, right_col]):
            record_index = index + offset

            if record_index >= len(records):
                continue

            _, row = records[record_index]

            with col:
                # Compact summary card
                st.html(make_card(row))

                # Additional information stays hidden until clicked.
                with st.expander("▼ View details", expanded=False):
                    detail_col1, detail_col2 = st.columns(
                        2,
                        gap="small",
                    )

                    with detail_col1:
                        show_detail("Priority", row["Priority"])
                        show_detail("POC 2", row["POC 2"])
                        show_detail(
                            "Latitude",
                            safe_number(row["Lat"], 6),
                        )

                    with detail_col2:
                        show_detail("POC 3", row["POC 3"])
                        show_detail(
                            "Longitude",
                            safe_number(row["Long"], 6),
                        )
                        show_detail(
                            "Check Status",
                            row["Check Status"],
                        )


# =========================================================
# FOOTER
# =========================================================
st.caption(
    "Source: intimation.xlsx · Overview and records update "
    "automatically when filters change."
)
