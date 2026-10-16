
from pathlib import Path
import html
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>
    :root {
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --text-main: #172033;
        --text-muted: #64748b;
        --border-color: #e2e8f0;
        --card-bg: #ffffff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%, rgba(59,130,246,.055), transparent 35%),
            #f5f7fb;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.1rem;
        padding-bottom: 1.5rem;
    }

    /* Main heading */
    .dashboard-header {
        position: relative;
        overflow: hidden;
        padding: 18px 22px;
        margin-bottom: 16px;
        border-radius: 16px;
        background: linear-gradient(120deg, #172554 0%, #1d4ed8 62%, #3b82f6 100%);
        box-shadow: 0 8px 22px rgba(37, 99, 235, .17);
        color: white;
    }

    .dashboard-header::after {
        content: "";
        position: absolute;
        width: 190px;
        height: 190px;
        right: -40px;
        top: -105px;
        border-radius: 50%;
        background: rgba(255,255,255,.09);
        pointer-events: none;
    }

    .dashboard-header h1 {
        color: #ffffff;
        font-size: clamp(23px, 2.2vw, 31px);
        font-weight: 800;
        line-height: 1.2;
        margin: 0;
        letter-spacing: -.6px;
    }

    .dashboard-header p {
        color: #dbeafe;
        font-size: 12px;
        margin: 6px 0 0 1px;
        letter-spacing: .25px;
    }

    .header-chip {
        display: inline-block;
        margin-bottom: 7px;
        padding: 4px 9px;
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 20px;
        background: rgba(255,255,255,.12);
        color: #eff6ff;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    /* Overview cards */
    .overview-card {
        position: relative;
        overflow: hidden;
        min-height: 94px;
        padding: 15px 17px;
        border: 1px solid rgba(255,255,255,.28);
        border-radius: 13px;
        color: #ffffff;
        box-shadow: 0 5px 14px rgba(15,23,42,.08);
    }

    .overview-card.records {
        background: linear-gradient(125deg, #1d4ed8, #3b82f6);
    }

    .overview-card.capacity {
        background: linear-gradient(125deg, #047857, #10b981);
    }

    .overview-card .metric-label {
        color: rgba(255,255,255,.86);
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .8px;
    }

    .overview-card .metric-value {
        color: #ffffff;
        font-size: clamp(23px, 2.4vw, 30px);
        font-weight: 800;
        line-height: 1.3;
        margin-top: 5px;
    }

    .overview-card .metric-icon {
        position: absolute;
        right: 15px;
        top: 15px;
        font-size: 24px;
        opacity: .72;
    }

    /* Section labels */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 9px;
        margin: 15px 0 9px;
        color: #1e293b;
        font-size: 15px;
        font-weight: 800;
    }

    .section-heading .section-icon {
        display: inline-flex;
        width: 27px;
        height: 27px;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 14px;
    }

    .results-caption {
        color: #64748b;
        font-size: 11px;
        margin: 2px 0 10px;
    }

    /* PSS card container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #dfe7f2 !important;
        border-radius: 13px !important;
        background: rgba(255,255,255,.97) !important;
        box-shadow: 0 3px 10px rgba(15,23,42,.035);
        transition: border-color .18s ease, box-shadow .18s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #a8c5f8 !important;
        box-shadow: 0 6px 16px rgba(37,99,235,.075);
    }

    /* Compact PSS identity */
    .pss-title-row {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 10px;
    }

    .pss-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 33px;
        width: 33px;
        height: 33px;
        border-radius: 10px;
        background: linear-gradient(135deg, #dbeafe, #eff6ff);
        border: 1px solid #dbeafe;
        font-size: 16px;
    }

    .pss-heading {
        color: #172554;
        font-size: 14px;
        font-weight: 800;
        line-height: 1.35;
        overflow-wrap: anywhere;
        margin-top: 1px;
    }

    .pss-subheading {
        margin-top: 3px;
        color: #94a3b8;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: .8px;
        text-transform: uppercase;
    }

    /* Card information tiles */
    .pss-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 7px;
        margin-bottom: 7px;
    }

    .pss-info {
        min-width: 0;
        padding: 8px 9px;
        border: 1px solid #edf1f7;
        border-radius: 9px;
        background: #f8fafc;
    }

    .pss-info-label {
        color: #94a3b8;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .65px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .pss-info-value {
        color: #334155;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .pss-info.capacity-tile {
        background: #eff6ff;
        border-color: #dbeafe;
    }

    .pss-info.capacity-tile .pss-info-value {
        color: #1d4ed8;
    }

    .pss-info.state-tile {
        background: #f0fdf4;
        border-color: #dcfce7;
    }

    .pss-info.state-tile .pss-info-value {
        color: #047857;
    }

    /* Copy button */
    div[data-testid="stVerticalBlock"] iframe {
        border: 0;
    }

    /* Expand details */
    div[data-testid="stExpander"] {
        border: 1px solid #e7edf6 !important;
        border-radius: 9px !important;
        background: #fbfdff !important;
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        padding-top: 7px !important;
        padding-bottom: 7px !important;
        color: #334155 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
    }

    .detail-tile {
        padding: 9px 10px;
        margin: 0 0 7px;
        border-radius: 8px;
        background: #f1f5f9;
        border: 1px solid #e7edf5;
        min-height: 53px;
    }

    .detail-label {
        color: #64748b;
        font-size: 9px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .6px;
        margin-bottom: 4px;
    }

    .detail-value {
        color: #1e293b;
        font-size: 12px;
        font-weight: 650;
        overflow-wrap: anywhere;
        line-height: 1.4;
    }

    .status-pill {
        display: inline-block;
        max-width: 100%;
        padding: 4px 8px;
        border-radius: 6px;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 10px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 650;
        transition: all .15s ease;
    }

    .stButton > button[kind="secondary"] {
        border-color: #dbe4f0;
        background: #ffffff;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: #93b4f5;
        color: #1d4ed8;
    }

    /* Filters */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 9px;
    }

    /* Pagination */
    .page-label {
        padding-top: 7px;
        color: #64748b;
        text-align: center;
        font-size: 12px;
        font-weight: 700;
    }

    .empty-state {
        padding: 25px 15px;
        text-align: center;
        border: 1px dashed #cbd5e1;
        border-radius: 13px;
        background: #ffffff;
        color: #64748b;
    }

    .empty-state strong {
        display: block;
        color: #334155;
        font-size: 15px;
        margin-bottom: 5px;
    }

    /* Reduce excess vertical gaps */
    div[data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }

        .dashboard-header {
            padding: 15px;
        }

        .overview-card {
            padding: 12px;
            min-height: 85px;
        }

        .pss-heading {
            font-size: 12px;
        }
    }
</style>
""")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column(value):
    """Normalize column names for flexible matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_column(columns, candidates=None, startswith=None):
    """Find a matching column using candidate names or prefix."""
    normalized = {
        normalize_column(column): column
        for column in columns
    }

    if candidates:
        for candidate in candidates:
            key = normalize_column(candidate)
            if key in normalized:
                return normalized[key]

    if startswith:
        prefix = normalize_column(startswith)
        for column in columns:
            if normalize_column(column).startswith(prefix):
                return column

    return None


def clean_text(value, default="—"):
    """Convert values to clean display text."""
    if pd.isna(value):
        return default

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "nat"}:
        return default

    return text


def numeric_value(value):
    """Convert a number or formatted capacity string to float."""
    if pd.isna(value):
        return 0.0

    try:
        text = str(value).strip()
        text = text.replace(",", "")
        text = re.sub(r"(?i)\s*mw\s*$", "", text)
        match = re.search(r"-?\d+(?:\.\d+)?", text)

        return float(match.group()) if match else 0.0
    except (TypeError, ValueError):
        return 0.0


def display_capacity(value):
    """Format capacity without unnecessary decimal zeros."""
    number = numeric_value(value)

    if number.is_integer():
        return f"{int(number):,} MW"

    return f"{number:,.2f}".rstrip("0").rstrip(".") + " MW"


def safe_html(value):
    return html.escape(clean_text(value))


def detail_item(label, value, status=False):
    """Render a single detail tile."""
    label_html = safe_html(label)
    value_html = safe_html(value)

    if status:
        value_html = (
            f'<span class="status-pill">{value_html}</span>'
        )

    st.markdown(
        f"""
        <div class="detail-tile">
            <div class="detail-label">{label_html}</div>
            <div class="detail-value">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_copy_button(pss_name, card_key):
    """
    Render an isolated copy button.
    It runs in its own iframe, so clicking it does not expand the card.
    """
    text_to_copy = json_string = (
        '"' + str(pss_name).replace("\\", "\\\\").replace('"', '\\"') + '"'
    )

    # The JSON string above is used as a safe JavaScript string literal.
    components.html(
        f"""
        <div style="
            display:flex;
            justify-content:flex-end;
            align-items:center;
            margin:0;
            padding:0;
            font-family:Arial,sans-serif;
        ">
            <button id="copy_{card_key}" title="Copy PSS name"
                style="
                    border:1px solid #dbeafe;
                    background:linear-gradient(135deg,#eff6ff,#ffffff);
                    color:#1d4ed8;
                    border-radius:8px;
                    height:30px;
                    min-width:38px;
                    cursor:pointer;
                    font-size:16px;
                    box-shadow:0 2px 5px rgba(37,99,235,.06);
                ">
                ⚡
            </button>
            <span id="msg_{card_key}" style="
                display:none;
                margin-left:5px;
                font-size:10px;
                color:#047857;
                font-weight:700;
            ">Copied!</span>
        </div>
        <script>
        const copyBtn = document.getElementById("copy_{card_key}");
        const message = document.getElementById("msg_{card_key}");
        const pssText = {json_string};

        copyBtn.addEventListener("click", async function() {{
            try {{
                if (navigator.clipboard && window.isSecureContext) {{
                    await navigator.clipboard.writeText(pssText);
                }} else {{
                    const area = document.createElement("textarea");
                    area.value = pssText;
                    area.style.position = "fixed";
                    area.style.left = "-9999px";
                    document.body.appendChild(area);
                    area.select();
                    const copied = document.execCommand("copy");
                    area.remove();

                    if (!copied) throw new Error("Copy failed");
                }}

                message.style.display = "inline";
                copyBtn.innerHTML = "✓";
                setTimeout(() => {{
                    message.style.display = "none";
                    copyBtn.innerHTML = "⚡";
                }}, 1200);
            }} catch (error) {{
                message.textContent = "Copy unavailable";
                message.style.display = "inline";
                setTimeout(() => {{
                    message.style.display = "none";
                    message.textContent = "Copied!";
                }}, 1500);
            }}
        }});
        </script>
        """,
        height=36,
        scrolling=False,
    )


def render_pss_card(row, columns, card_key):
    """Render one compact PSS card with a native expandable details section."""

    pss_name = clean_text(row.get(columns["pss"]))
    state = clean_text(row.get(columns["state"]))
    capacity = display_capacity(row.get(columns["capacity"]))
    poc1 = clean_text(row.get(columns["poc1"]))

    with st.container(border=True):
        # Header row
        heading_col, copy_col = st.columns(
            [5.6, 0.8],
            vertical_alignment="center",
            gap="small",
        )

        with heading_col:
            st.markdown(
                f"""
                <div class="pss-title-row">
                    <div class="pss-icon">⚡</div>
                    <div>
                        <div class="pss-heading">{safe_html(pss_name)}</div>
                        <div class="pss-subheading">Power Substation</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with copy_col:
            render_copy_button(pss_name, card_key)

        # Two compact information tiles
        st.markdown(
            f"""
            <div class="pss-info-grid">
                <div class="pss-info capacity-tile">
                    <div class="pss-info-label">Capacity</div>
                    <div class="pss-info-value">{safe_html(capacity)}</div>
                </div>
                <div class="pss-info state-tile">
                    <div class="pss-info-label">State</div>
                    <div class="pss-info-value">{safe_html(state)}</div>
                </div>
            </div>
            <div class="pss-info">
                <div class="pss-info-label">POC 1</div>
                <div class="pss-info-value">{safe_html(poc1)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Native Streamlit expander: layout grows when expanded
        with st.expander("View site details", expanded=False):
            left, right = st.columns(2, gap="small")

            with left:
                detail_item(
                    "Priority",
                    row.get(columns["priority"]) if columns["priority"] else "—",
                )
                detail_item(
                    "POC 2",
                    row.get(columns["poc2"]) if columns["poc2"] else "—",
                )
                detail_item(
                    "Latitude",
                    row.get(columns["lat"]) if columns["lat"] else "—",
                )

            with right:
                detail_item(
                    "POC 3",
                    row.get(columns["poc3"]) if columns["poc3"] else "—",
                )
                detail_item(
                    "Longitude",
                    row.get(columns["long"]) if columns["long"] else "—",
                )
                detail_item(
                    "Check Status",
                    row.get(columns["status"]) if columns["status"] else "—",
                    status=True,
                )


# ============================================================
# LOAD EXCEL FILE
# ============================================================

@st.cache_data(show_spinner="Loading intimation data...")
def load_data(file_path, modified_time):
    return pd.read_excel(file_path)


file_path = Path(__file__).resolve().parent.parent / "intimation.xlsx"

if not file_path.exists():
    st.error(
        f"Excel file not found: {file_path}. "
        "Please check that intimation.xlsx is in the project root."
    )
    st.stop()

try:
    df = load_data(str(file_path), file_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Unable to read intimation.xlsx: {exc}")
    st.stop()

if df.empty:
    st.warning("The Excel file is empty.")
    st.stop()


# ============================================================
# COLUMN MAPPING
# ============================================================

all_columns = list(df.columns)

columns = {
    "pss": find_column(
        all_columns,
        candidates=[
            "PSS",
            "PSS Name",
            "Name of PSS",
            "Substation",
            "Substation Name",
        ],
    ),
    "state": find_column(
        all_columns,
        candidates=["State", "PSS State", "State Name"],
    ),
    "capacity": find_column(
        all_columns,
        candidates=[
            "Capacity (MW)",
            "Capacity MW",
            "Capacity",
            "PSS Capacity (MW)",
        ],
    ),
    "poc1": find_column(
        all_columns,
        candidates=["POC1", "POC 1", "POC-1"],
    ),
    "poc2": find_column(
        all_columns,
        candidates=["POC 2", "POC2", "POC-2"],
    ),
    "poc3": find_column(
        all_columns,
        candidates=["POC 3", "POC3", "POC-3"],
    ),
    "lat": find_column(
        all_columns,
        candidates=["Lat", "Latitude"],
    ),
    "long": find_column(
        all_columns,
        candidates=["Long", "Longitude", "Lon"],
    ),
    "status": find_column(
        all_columns,
        candidates=[
            "Check Status",
            "Status",
            "Intimation Status",
        ],
    ),
}

# Priority preference:
# 1. Priority as per 003 sheet
# 2. Priority as per Amresh Sir 28.07.2023
# 3. First column beginning with Priority

columns["priority"] = find_column(
    all_columns,
    candidates=[
        "Priority as per 003 sheet",
        "Priority as per Amresh Sir 28.07.2023",
    ],
)

if columns["priority"] is None:
    columns["priority"] = find_column(
        all_columns,
        startswith="Priority",
    )

if columns["pss"] is None:
    st.error(
        "PSS column could not be identified. "
        "Please check the column name in intimation.xlsx."
    )
    st.write("Available columns:", all_columns)
    st.stop()

# Provide sensible defaults for optional fields if absent.
for field in ["state", "capacity", "poc1"]:
    if columns[field] is None:
        st.warning(
            f"Column for '{field}' was not found. "
            "The corresponding filter or value may be unavailable."
        )


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="header-chip">SOLAR SUITE · OPERATIONS</div>
        <h1>⚡ Intimation Dashboard</h1>
        <p>Power Substation Overview &nbsp;·&nbsp; Search, filter and inspect site details</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILTERS
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-icon">⌕</span>
        Search & Filters
    </div>
    """,
    unsafe_allow_html=True,
)

filter1, filter2, filter3 = st.columns(
    [2.2, 1.4, 1.4],
    gap="medium",
)

with filter1:
    pss_search = st.text_input(
        "Search PSS",
        placeholder="Type PSS name...",
        key="intimation_pss_search",
    )

# Search first; POC1 choices are based on the PSS search.
search_df = df.copy()

if pss_search.strip():
    search_df = search_df[
        search_df[columns["pss"]]
        .fillna("")
        .astype(str)
        .str.contains(
            pss_search.strip(),
            case=False,
            na=False,
            regex=False,
        )
    ]

# POC1 options depend on the PSS search.
with filter2:
    if columns["poc1"]:
        poc1_options = sorted(
            {
                clean_text(value)
                for value in search_df[columns["poc1"]].tolist()
                if clean_text(value) != "—"
            },
            key=str.casefold,
        )

        poc1_choice = st.selectbox(
            "POC 1",
            options=["All"] + poc1_options,
            key="intimation_poc1_filter",
        )
    else:
        poc1_choice = "All"
        st.selectbox(
            "POC 1",
            options=["All"],
            key="intimation_poc1_unavailable",
            disabled=True,
        )

# State choices depend on both the PSS search and selected POC1.
state_source = search_df.copy()

if columns["poc1"] and poc1_choice != "All":
    state_source = state_source[
        state_source[columns["poc1"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == poc1_choice
    ]

with filter3:
    if columns["state"]:
        state_options = sorted(
            {
                clean_text(value)
                for value in state_source[columns["state"]].tolist()
                if clean_text(value) != "—"
            },
            key=str.casefold,
        )

        state_choice = st.selectbox(
            "State",
            options=["All"] + state_options,
            key="intimation_state_filter",
        )
    else:
        state_choice = "All"
        st.selectbox(
            "State",
            options=["All"],
            key="intimation_state_unavailable",
            disabled=True,
        )

# Apply all filters.
filtered_df = search_df.copy()

if columns["poc1"] and poc1_choice != "All":
    filtered_df = filtered_df[
        filtered_df[columns["poc1"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == poc1_choice
    ]

if columns["state"] and state_choice != "All":
    filtered_df = filtered_df[
        filtered_df[columns["state"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == state_choice
    ]

filtered_df = filtered_df.reset_index(drop=True)


# ============================================================
# OVERVIEW CARDS
# ============================================================

record_count = len(filtered_df)

if columns["capacity"]:
    total_capacity = filtered_df[columns["capacity"]].apply(
        numeric_value
    ).sum()
else:
    total_capacity = 0.0

capacity_display = display_capacity(total_capacity)

overview1, overview2 = st.columns(2, gap="medium")

with overview1:
    st.markdown(
        f"""
        <div class="overview-card records">
            <div class="metric-icon">▦</div>
            <div class="metric-label">Matching Records</div>
            <div class="metric-value">{record_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview2:
    st.markdown(
        f"""
        <div class="overview-card capacity">
            <div class="metric-icon">⚡</div>
            <div class="metric-label">Total Capacity</div>
            <div class="metric-value">{safe_html(capacity_display)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGINATION
# ============================================================

PAGE_SIZE = 20

# Reset the current page when the filters change.
filter_signature = (
    pss_search.strip().lower(),
    poc1_choice,
    state_choice,
)

if st.session_state.get("intimation_filter_signature") != filter_signature:
    st.session_state["intimation_filter_signature"] = filter_signature
    st.session_state["intimation_page"] = 1

total_pages = max(1, (record_count + PAGE_SIZE - 1) // PAGE_SIZE)

current_page = st.session_state.get("intimation_page", 1)
current_page = min(max(1, current_page), total_pages)
st.session_state["intimation_page"] = current_page

start_index = (current_page - 1) * PAGE_SIZE
end_index = min(start_index + PAGE_SIZE, record_count)

page_df = filtered_df.iloc[start_index:end_index]


# ============================================================
# PSS CARDS SECTION
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-icon">▤</span>
        PSS Sites
    </div>
    """,
    unsafe_allow_html=True,
)

if record_count:
    st.markdown(
        f"""
        <div class="results-caption">
            Showing <b>{start_index + 1}–{end_index}</b>
            of <b>{record_count:,}</b> matching sites
            &nbsp; · &nbsp; {PAGE_SIZE} records per page
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = list(page_df.iterrows())

    # Two cards per row; each expander participates in native Streamlit
    # layout, so cards below move down when a card is expanded.
    for pair_start in range(0, len(records), 2):
        card_cols = st.columns(2, gap="medium")

        for offset in range(2):
            record_index = pair_start + offset

            if record_index >= len(records):
                continue

            original_index, row = records[record_index]

            with card_cols[offset]:
                render_pss_card(
                    row=row,
                    columns=columns,
                    card_key=f"pss_{current_page}_{original_index}",
                )

        # A little separation between each card row.
        st.markdown(
            '<div style="height:4px"></div>',
            unsafe_allow_html=True,
        )

    # Pagination controls
    if total_pages > 1:
        st.markdown("---")

        prev_col, page_col, next_col = st.columns(
            [1, 2, 1],
            vertical_alignment="center",
        )

        with prev_col:
            if st.button(
                "← Previous",
                use_container_width=True,
                disabled=current_page <= 1,
                key="intimation_prev_page",
            ):
                st.session_state["intimation_page"] = current_page - 1
                st.rerun()

        with page_col:
            st.markdown(
                f"""
                <div class="page-label">
                    Page {current_page} of {total_pages}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with next_col:
            if st.button(
                "Next →",
                use_container_width=True,
                disabled=current_page >= total_pages,
                key="intimation_next_page",
            ):
                st.session_state["intimation_page"] = current_page + 1
                st.rerun()

else:
    st.markdown(
        """
        <div class="empty-state">
            <strong>🔎 No matching PSS sites</strong>
            Try another PSS name or change the selected POC 1 / State filters.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:22px;
        padding-top:10px;
        border-top:1px solid #e2e8f0;
        text-align:center;
        color:#94a3b8;
        font-size:10px;
    ">
        SOLAR SUITE &nbsp;·&nbsp; INTIMATION DASHBOARD
    </div>
    """,
    unsafe_allow_html=True,
)
