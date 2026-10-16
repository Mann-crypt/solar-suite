
from pathlib import Path
from html import escape

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
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1550px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    .dashboard-header {
        background: linear-gradient(120deg, #102A43 0%, #176B87 60%, #218CA5 100%);
        padding: 27px 30px;
        border-radius: 18px;
        margin-bottom: 25px;
        box-shadow: 0 5px 18px rgba(15, 50, 75, 0.12);
    }

    .dashboard-header h1 {
        color: #FFFFFF !important;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .dashboard-header p {
        color: #D9EEF5 !important;
        margin: 8px 0 0 0;
        font-size: 0.95rem;
    }

    .section-heading {
        font-size: 1.18rem;
        font-weight: 750;
        margin: 12px 0 5px 0;
    }

    .section-description {
        font-size: 0.85rem;
        opacity: 0.72;
        margin-bottom: 15px;
    }

    .record-card {
        border: 1px solid rgba(128, 128, 128, 0.24);
        border-radius: 15px;
        padding: 17px;
        margin-bottom: 16px;
        background: var(--secondary-background-color);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.035);
        min-height: 260px;
        overflow-wrap: anywhere;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .record-card:hover {
        border-color: #2693AD;
        transform: translateY(-2px);
    }

    .record-top {
        display: flex;
        align-items: flex-start;
        gap: 11px;
        margin-bottom: 13px;
    }

    .record-icon {
        flex-shrink: 0;
        width: 39px;
        height: 39px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(38, 147, 173, 0.13);
        font-size: 20px;
    }

    .record-title {
        font-size: 1.03rem;
        font-weight: 800;
        line-height: 1.4;
        margin: 0;
        overflow-wrap: anywhere;
    }

    .record-state {
        font-size: 0.81rem;
        opacity: 0.72;
        margin-top: 4px;
    }

    .record-status-row {
        margin: 9px 0 15px 0;
    }

    .record-status {
        display: inline-block;
        border-radius: 7px;
        padding: 5px 9px;
        font-size: 0.73rem;
        font-weight: 750;
        line-height: 1.4;
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

    .record-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        column-gap: 13px;
        row-gap: 14px;
    }

    .record-field {
        min-width: 0;
    }

    .record-label {
        font-size: 0.69rem;
        font-weight: 650;
        letter-spacing: 0.65px;
        text-transform: uppercase;
        opacity: 0.65;
        margin-bottom: 4px;
    }

    .record-value {
        font-size: 0.87rem;
        font-weight: 650;
        line-height: 1.4;
        overflow-wrap: anywhere;
        word-break: break-word;
    }

    .record-coordinates {
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1.5;
        overflow-wrap: anywhere;
    }

    div[data-testid="stMetric"] {
        padding: 17px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 13px;
        background: var(--secondary-background-color);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 10px;
    }

    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stMultiSelect"] > div > div {
        border-radius: 10px;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 650;
    }

    @media (max-width: 600px) {
        .dashboard-header {
            padding: 21px;
        }

        .dashboard-header h1 {
            font-size: 1.55rem;
        }

        .record-card {
            padding: 14px;
            min-height: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="dashboard-header">
        <h1>📊 Intimation Dashboard</h1>
        <p>
            Search, filter and review intimation records
            with a clear, card-based view.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


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
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing Excel columns: "
            + ", ".join(missing_columns)
        )

    # Clean text columns without changing numeric coordinates.
    text_columns = [
        "PSS",
        "State",
        "POC1",
        "POC 2",
        "POC 3",
        "Check Status",
    ]

    for column in text_columns:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Select the preferred priority column.
    # Prefer the 003 sheet; otherwise use Amresh Sir's priority.
    priority_candidates = [
        "Priority as per 003 sheet",
        "Priority as per Amresh Sir 28.07.2023",
    ]

    priority_column = next(
        (
            column
            for column in priority_candidates
            if column in data.columns
        ),
        None,
    )

    if priority_column is None:
        priority_columns = [
            column
            for column in data.columns
            if column.lower().startswith("priority")
        ]
        priority_column = (
            priority_columns[0] if priority_columns else None
        )

    if priority_column:
        data["Priority"] = data[priority_column]
    else:
        data["Priority"] = ""

    return data


if not EXCEL_PATH.exists():
    st.error(
        "The file intimation.xlsx was not found in the repository root."
    )
    st.stop()

try:
    df = load_data(
        str(EXCEL_PATH),
        EXCEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Unable to load the Excel file: {exc}")
    st.info(
        "Check that intimation.xlsx is a valid Excel workbook "
        "and contains the required columns."
    )
    st.stop()


# =========================================================
# FILTER RESET CALLBACK
# =========================================================
def reset_state_filter():
    # When the user or PSS search changes, the state selection
    # is recreated from the newly matching records.
    st.session_state.pop("intimation_states", None)


# =========================================================
# FILTER OPTIONS
# =========================================================
user_options = ["All Users"] + sorted(
    value
    for value in df["POC1"].unique()
    if value
)

# Keep an existing valid user selection; otherwise use All Users.
if st.session_state.get("intimation_user") not in user_options:
    st.session_state["intimation_user"] = "All Users"


# =========================================================
# FILTERS
# =========================================================
st.markdown(
    '<div class="section-heading">🔎 Find Records</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Choose a user, search for a PSS, and narrow the results by state.'
    '</div>',
    unsafe_allow_html=True,
)

filter_col1, filter_col2, filter_col3 = st.columns(
    [1, 1.1, 1.5],
    gap="medium",
)

with filter_col1:
    selected_user = st.selectbox(
        "👤 POC1 / User",
        options=user_options,
        key="intimation_user",
        on_change=reset_state_filter,
    )

with filter_col3:
    pss_search = st.text_input(
        "🔍 Search PSS",
        placeholder="Enter PSS name or identifier...",
        key="intimation_pss",
        on_change=reset_state_filter,
    )


# Filter by user and PSS before creating the State options.
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
    value
    for value in base_df["State"].unique()
    if value
)

# Remove state selections that no longer exist in the filtered data.
if "intimation_states" in st.session_state:
    current_states = st.session_state["intimation_states"]
    st.session_state["intimation_states"] = [
        state
        for state in current_states
        if state in available_states
    ]

with filter_col2:
    selected_states = st.multiselect(
        "📍 State",
        options=available_states,
        default=available_states,
        key="intimation_states",
        placeholder="Select one or more states",
    )


# Apply all active filters.
filtered_df = base_df[
    base_df["State"].isin(selected_states)
].copy()


# =========================================================
# DISPLAY DATA
# =========================================================
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


# =========================================================
# CARD HELPERS
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


def make_card(row):
    def val(column):
        value = row.get(column, "")

        if pd.isna(value) or str(value).strip() == "":
            return "—"

        return escape(str(value).strip())

    pss = val("PSS")
    state = val("State")
    priority = val("Priority")
    poc1 = val("POC1")
    poc2 = val("POC 2")
    poc3 = val("POC 3")
    status = val("Check Status")

    capacity = safe_number(row.get("Capacity (MW)", ""), 2)
    lat = safe_number(row.get("Lat", ""), 6)
    long = safe_number(row.get("Long", ""), 6)

    # Assign a visual status class based on the text.
    status_text = safe_text(row.get("Check Status", "")).lower()

    if any(word in status_text for word in [
        "pass", "checked", "complete", "completed", "verified", "ok"
    ]):
        status_class = "status-ok"
    elif any(word in status_text for word in [
        "pending", "progress", "waiting"
    ]):
        status_class = "status-pending"
    elif any(word in status_text for word in [
        "fail", "error", "issue", "mismatch", "reject"
    ]):
        status_class = "status-issue"
    else:
        status_class = "status-neutral"

    return f"""
    <div class="record-card">

        <div class="record-top">
            <div class="record-icon">⚡</div>
            <div>
                <div class="record-title">{pss}</div>
                <div class="record-state">📍 {state}</div>
            </div>
        </div>

        <div class="record-status-row">
            <span class="record-status {status_class}">
                {status}
            </span>
        </div>

        <div class="record-grid">

            <div class="record-field">
                <div class="record-label">Capacity</div>
                <div class="record-value">{capacity} MW</div>
            </div>

            <div class="record-field">
                <div class="record-label">Priority</div>
                <div class="record-value">{priority}</div>
            </div>

            <div class="record-field">
                <div class="record-label">POC 1</div>
                <div class="record-value">{poc1}</div>
            </div>

            <div class="record-field">
                <div class="record-label">POC 2</div>
                <div class="record-value">{poc2}</div>
            </div>

            <div class="record-field">
                <div class="record-label">POC 3</div>
                <div class="record-value">{poc3}</div>
            </div>

            <div class="record-field">
                <div class="record-label">Latitude / Longitude</div>
                <div class="record-coordinates">
                    {lat}<br>{long}
                </div>
            </div>

        </div>
    </div>
    """


# =========================================================
# INTIMATION CARDS
# =========================================================
st.markdown("---")

st.markdown(
    '<div class="section-heading">📋 Intimation Records</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="section-description">'
    f'Showing {len(display_df):,} matching records out of '
    f'{len(df):,} total records.'
    f'</div>',
    unsafe_allow_html=True,
)

if display_df.empty:
    st.info(
        "No matching records found. Try changing the selected user, "
        "PSS search, or state."
    )

else:
    cards_per_page = 20

    total_pages = (
        len(display_df) + cards_per_page - 1
    ) // cards_per_page

    # Keep the current page valid when filters change.
    if "intimation_card_page" not in st.session_state:
        st.session_state["intimation_card_page"] = 0

    st.session_state["intimation_card_page"] = max(
        0,
        min(
            st.session_state["intimation_card_page"],
            total_pages - 1,
        ),
    )

    page_col1, page_col2 = st.columns([3, 1])

    with page_col1:
        start = (
            st.session_state["intimation_card_page"]
            * cards_per_page
        )
        end = min(
            start + cards_per_page,
            len(display_df),
        )

        st.caption(
            f"Displaying records {start + 1}–{end} "
            f"of {len(display_df):,}"
        )

    with page_col2:
        selected_page = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            index=st.session_state["intimation_card_page"],
            key="intimation_card_page_selector",
            label_visibility="collapsed",
        )

        st.session_state["intimation_card_page"] = (
            selected_page - 1
        )

    start = (
        st.session_state["intimation_card_page"]
        * cards_per_page
    )

    page_df = display_df.iloc[
        start:start + cards_per_page
    ]

    records = list(page_df.iterrows())

    for index in range(0, len(records), 2):
        left_col, right_col = st.columns(
            2,
            gap="medium",
        )

        with left_col:
            _, row = records[index]

            st.markdown(
                make_card(row),
                unsafe_allow_html=True,
            )

        if index + 1 < len(records):
            with right_col:
                _, row = records[index + 1]

                st.markdown(
                    make_card(row),
                    unsafe_allow_html=True,
                )


# =========================================================
# OVERVIEW — AT THE BOTTOM
# =========================================================
st.markdown("---")

st.markdown(
    '<div class="section-heading">📈 Overview</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Summary based on your current filters.'
    '</div>',
    unsafe_allow_html=True,
)

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
