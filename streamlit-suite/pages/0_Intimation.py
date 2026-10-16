
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
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
}

.dashboard-header {
    background: linear-gradient(120deg, #102A43, #176B87 60%, #218CA5);
    padding: 25px 28px;
    border-radius: 16px;
    margin-bottom: 22px;
    box-shadow: 0 5px 18px rgba(15, 50, 75, 0.12);
}

.dashboard-header h1 {
    color: white !important;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0;
}

.dashboard-header p {
    color: #D9EEF5 !important;
    margin: 7px 0 0;
    font-size: 0.92rem;
}

.section-heading {
    font-size: 1.15rem;
    font-weight: 750;
    margin: 12px 0 5px;
}

.section-description {
    font-size: 0.84rem;
    opacity: 0.72;
    margin-bottom: 14px;
}

/* Compact record card */
.record-card {
    border: 1px solid rgba(128, 128, 128, 0.24);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 8px;
    background: var(--secondary-background-color);
    box-shadow: 0 2px 7px rgba(0, 0, 0, 0.035);
    overflow-wrap: anywhere;
    transition: border-color 0.2s ease;
}

.record-card:hover {
    border-color: #2693AD;
}

.record-top {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}

.record-icon {
    flex-shrink: 0;
    width: 35px;
    height: 35px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(38, 147, 173, 0.13);
    font-size: 18px;
}

.record-title {
    font-size: 0.98rem;
    font-weight: 800;
    line-height: 1.35;
    margin: 0;
    overflow-wrap: anywhere;
}

.record-state {
    font-size: 0.78rem;
    opacity: 0.72;
    margin-top: 3px;
}

.record-primary {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.record-primary-item {
    min-width: 0;
    padding: 9px;
    border-radius: 8px;
    background: rgba(128, 128, 128, 0.07);
}

.record-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.45px;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 4px;
}

.record-value {
    font-size: 0.84rem;
    font-weight: 700;
    line-height: 1.4;
    overflow-wrap: anywhere;
    word-break: break-word;
}

.record-status-row {
    margin-top: 9px;
}

.record-status {
    display: inline-block;
    border-radius: 6px;
    padding: 4px 7px;
    font-size: 0.70rem;
    font-weight: 700;
    line-height: 1.35;
    overflow-wrap: anywhere;
}

.status-ok {
    color: #16794B;
    background: rgba(34, 197, 94, 0.13);
}

.status-pending {
    color: #A16207;
    background: rgba(234, 179, 8, 0.16);
}

.status-issue {
    color: #B42318;
    background: rgba(239, 68, 68, 0.12);
}

.status-neutral {
    color: var(--text-color);
    background: rgba(128, 128, 128, 0.13);
}

/* Details */
.details-heading {
    font-size: 0.82rem;
    font-weight: 750;
    margin: 10px 0 3px;
}

.details-value {
    font-size: 0.88rem;
    overflow-wrap: anywhere;
    word-break: break-word;
}

div[data-testid="stMetric"] {
    padding: 15px;
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    background: var(--secondary-background-color);
}

div[data-testid="stMetricLabel"] {
    font-weight: 650;
}

div[data-testid="stMetricValue"] {
    font-size: 1.6rem;
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
        padding: 19px;
    }

    .dashboard-header h1 {
        font-size: 1.5rem;
    }

    .record-card {
        padding: 10px;
    }

    .record-primary-item {
        padding: 8px;
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

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing Excel columns: " + ", ".join(missing)
        )

    text_columns = [
        "PSS", "State", "POC1",
        "POC 2", "POC 3", "Check Status",
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
        "Verify that intimation.xlsx is a valid Excel workbook "
        "and contains the required columns."
    )
    st.stop()


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
Select a POC1 user, search for a PSS and filter by state.
</div>
""")

filter_col1, filter_col2, filter_col3 = st.columns(
    [1, 1.1, 1.5],
    gap="medium",
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
        placeholder="Enter PSS name or identifier...",
        key="intimation_pss",
        on_change=reset_dependent_filters,
    )


# Filter by user and PSS first.
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

display_columns = [
    "Lat", "Long", "PSS", "State", "Capacity (MW)",
    "Priority", "POC1", "POC 2", "POC 3", "Check Status",
]

display_df = filtered_df[display_columns].copy()


# =========================================================
# DISPLAY HELPERS
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


def status_class_for(value):
    status = safe_text(value).lower()

    if any(word in status for word in [
        "pass", "checked", "complete",
        "completed", "verified", "ok",
    ]):
        return "status-ok"

    if any(word in status for word in [
        "pending", "progress", "waiting",
    ]):
        return "status-pending"

    if any(word in status for word in [
        "fail", "error", "issue",
        "mismatch", "reject",
    ]):
        return "status-issue"

    return "status-neutral"


def make_card(row):
    def val(column):
        value = row.get(column, "")

        if pd.isna(value) or str(value).strip() == "":
            return "—"

        return escape(str(value).strip())

    pss = val("PSS")
    state = val("State")
    poc1 = val("POC1")
    status = val("Check Status")

    capacity = escape(
        safe_number(row.get("Capacity (MW)", ""), 2)
    )

    status_class = status_class_for(
        row.get("Check Status", "")
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
                <div class="record-label">POC1 / User</div>
                <div class="record-value">{poc1}</div>
            </div>

            <div class="record-primary-item">
                <div class="record-label">Capacity</div>
                <div class="record-value">{capacity} MW</div>
            </div>
        </div>

        <div class="record-status-row">
            <span class="record-status {status_class}">
                {status}
            </span>
        </div>
    </div>
    """).strip()


def show_detail(label, value):
    st.markdown(
        f'<div class="details-heading">{escape(label)}</div>'
        f'<div class="details-value">{escape(safe_text(value))}</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# INTIMATION CARDS
# =========================================================
st.divider()

st.html("""
<div class="section-heading">📋 Intimation Records</div>
""")

st.caption(
    f"Showing {len(display_df):,} matching records "
    f"out of {len(df):,} total records."
)

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
    current_page = max(0, min(current_page, total_pages - 1))
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
        left_col, right_col = st.columns(2, gap="medium")

        for offset, col in enumerate([left_col, right_col]):
            record_index = index + offset

            if record_index >= len(records):
                continue

            row_index, row = records[record_index]

            with col:
                # Compact summary card
                st.html(make_card(row))

                # Click to reveal additional information
                with st.expander("＋ View details", expanded=False):
                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        show_detail("Priority", row["Priority"])
                        show_detail("POC 2", row["POC 2"])
                        show_detail("Latitude", safe_number(row["Lat"], 6))

                    with detail_col2:
                        show_detail("POC 3", row["POC 3"])
                        show_detail("Longitude", safe_number(row["Long"], 6))
                        show_detail("Check Status", row["Check Status"])


# =========================================================
# OVERVIEW
# =========================================================
st.divider()

st.html("""
<div class="section-heading">📈 Overview</div>
<div class="section-description">
Summary based on your current filters.
</div>
""")

capacity_values = pd.to_numeric(
    filtered_df["Capacity (MW)"]
    .astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce",
)

total_capacity = capacity_values.sum()

unique_pss = (
    filtered_df["PSS"]
    .replace("", pd.NA)
    .nunique()
)

metric1, metric2, metric3 = st.columns(3, gap="medium")

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

st.caption(
    "Source: intimation.xlsx · Overview updates automatically "
    "when filters change."
)
