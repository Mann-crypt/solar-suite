
from pathlib import Path
import html
import json
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM UI
# =========================================================

st.html("""
<style>
    .stApp {
        background: #f6f8fc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1rem;
        padding-bottom: 1.5rem;
    }

    /* Dashboard header */
    .dashboard-header {
        padding: 17px 21px;
        margin-bottom: 15px;
        border-radius: 15px;
        background: linear-gradient(115deg, #172554, #1d4ed8 72%, #3b82f6);
        color: white;
        box-shadow: 0 5px 15px rgba(37, 99, 235, .12);
    }

    .header-label {
        display: inline-block;
        padding: 3px 8px;
        margin-bottom: 7px;
        border: 1px solid rgba(255,255,255,.25);
        border-radius: 20px;
        background: rgba(255,255,255,.10);
        color: #dbeafe;
        font-size: 9px;
        font-weight: 750;
        letter-spacing: 1px;
    }

    .dashboard-header h1 {
        margin: 0;
        color: #fff;
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -.5px;
    }

    .dashboard-header p {
        margin: 5px 0 0;
        color: #dbeafe;
        font-size: 11px;
    }

    /* Section headings */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 16px 0 9px;
        color: #1e293b;
        font-size: 14px;
        font-weight: 800;
    }

    .section-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 27px;
        height: 27px;
        border-radius: 8px;
        background: #e8efff;
        color: #1d4ed8;
        font-size: 14px;
    }

    .filter-caption {
        margin: 0 0 8px;
        color: #64748b;
        font-size: 11px;
    }

    /* Overview */
    .metric-card {
        position: relative;
        overflow: hidden;
        min-height: 83px;
        padding: 13px 16px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 12px rgba(15,23,42,.065);
    }

    .metric-records {
        background: linear-gradient(120deg, #1d4ed8, #3b82f6);
    }

    .metric-capacity {
        background: linear-gradient(120deg, #047857, #10b981);
    }

    .metric-label {
        color: rgba(255,255,255,.88);
        font-size: 10px;
        font-weight: 750;
        letter-spacing: .7px;
        text-transform: uppercase;
    }

    .metric-value {
        margin-top: 5px;
        color: white;
        font-size: 25px;
        line-height: 1.25;
        font-weight: 800;
    }

    .metric-icon {
        position: absolute;
        top: 12px;
        right: 14px;
        font-size: 22px;
        opacity: .65;
    }

    /* PSS card */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e2e8f0 !important;
        border-radius: 12px !important;
        background: #fff !important;
        box-shadow: 0 2px 7px rgba(15,23,42,.035);
        transition: border-color .16s ease, box-shadow .16s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #b9cdf6 !important;
        box-shadow: 0 4px 12px rgba(37,99,235,.07);
    }

    /* Compact collapsed card title */
    .pss-title-row {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
        margin-bottom: 8px;
    }

    .pss-icon {
        display: flex;
        flex: 0 0 31px;
        align-items: center;
        justify-content: center;
        width: 31px;
        height: 31px;
        border-radius: 9px;
        border: 1px solid #dbeafe;
        background: linear-gradient(135deg, #dbeafe, #f0f7ff);
        font-size: 15px;
    }

    .pss-heading {
        color: #172554;
        font-size: 12px;
        font-weight: 800;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .pss-subheading {
        margin-top: 2px;
        color: #94a3b8;
        font-size: 8px;
        font-weight: 750;
        letter-spacing: .7px;
        text-transform: uppercase;
    }

    /* Compact information tiles */
    .pss-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        margin-bottom: 6px;
    }

    .pss-info {
        min-width: 0;
        padding: 7px 8px;
        border: 1px solid #edf1f7;
        border-radius: 8px;
        background: #f8fafc;
    }

    .pss-info-label {
        display: flex;
        align-items: center;
        gap: 4px;
        margin-bottom: 3px;
        color: #94a3b8;
        font-size: 8px;
        font-weight: 800;
        letter-spacing: .55px;
        text-transform: uppercase;
    }

    .pss-info-value {
        color: #334155;
        font-size: 11px;
        font-weight: 750;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .capacity-tile {
        border-color: #dbeafe;
        background: #eff6ff;
    }

    .capacity-tile .pss-info-value {
        color: #1d4ed8;
    }

    .location-tile {
        border-color: #d1fae5;
        background: #f0fdf4;
    }

    .location-tile .pss-info-value {
        color: #047857;
    }

    .poc-tile {
        padding: 7px 8px;
        border: 1px solid #edf1f7;
        border-radius: 8px;
        background: #fafbfe;
    }

    /* Keep copy button compact */
    div[data-testid="stVerticalBlock"] iframe {
        border: 0;
    }

    /* Expanded details remain comfortable */
    div[data-testid="stExpander"] {
        margin-top: 7px;
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
        font-weight: 750 !important;
    }

    .detail-tile {
        min-height: 48px;
        margin-bottom: 7px;
        padding: 9px;
        border: 1px solid #e7edf5;
        border-radius: 8px;
        background: #f8fafc;
    }

    .detail-label {
        margin-bottom: 4px;
        color: #64748b;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .5px;
        text-transform: uppercase;
    }

    .detail-value {
        color: #1e293b;
        font-size: 12px;
        font-weight: 650;
        line-height: 1.4;
        overflow-wrap: anywhere;
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

    /* Inputs and buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 650;
    }

    .results-caption {
        margin: 2px 0 9px;
        color: #64748b;
        font-size: 11px;
    }

    .page-label {
        padding-top: 7px;
        color: #64748b;
        text-align: center;
        font-size: 12px;
        font-weight: 700;
    }

    .empty-state {
        padding: 23px 14px;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        background: white;
        color: #64748b;
        text-align: center;
    }

    .empty-state strong {
        display: block;
        margin-bottom: 5px;
        color: #334155;
        font-size: 14px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: .65rem;
            padding-right: .65rem;
        }

        .dashboard-header {
            padding: 15px;
        }

        .dashboard-header h1 {
            font-size: 22px;
        }

        .metric-value {
            font-size: 22px;
        }

        .pss-heading {
            font-size: 11px;
        }
    }
</style>
""")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_column(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_column(columns, candidates=None, startswith=None):
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
    if pd.isna(value):
        return default

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "nat"}:
        return default

    return value


def safe_html(value):
    return html.escape(clean_text(value))


def numeric_value(value):
    if pd.isna(value):
        return 0.0

    try:
        value = str(value).strip().replace(",", "")
        value = re.sub(r"(?i)\s*mw\s*$", "", value)

        match = re.search(r"-?\d+(?:\.\d+)?", value)

        return float(match.group()) if match else 0.0

    except (TypeError, ValueError):
        return 0.0


def display_capacity(value):
    number = numeric_value(value)

    if number.is_integer():
        return f"{int(number):,} MW"

    return f"{number:,.2f}".rstrip("0").rstrip(".") + " MW"


def unique_values(series):
    values = {
        clean_text(value)
        for value in series.tolist()
        if clean_text(value) != "—"
    }

    return sorted(values, key=str.casefold)


def detail_item(label, value, status=False):
    value_html = safe_html(value)

    if status:
        value_html = (
            f'<span class="status-pill">{value_html}</span>'
        )

    st.markdown(
        f"""
        <div class="detail-tile">
            <div class="detail-label">{safe_html(label)}</div>
            <div class="detail-value">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_copy_button(pss_name, card_key):
    """Copy button runs in an isolated iframe."""

    js_text = json.dumps(str(pss_name))

    components.html(
        f"""
        <div style="
            display:flex;
            justify-content:flex-end;
            padding:0;
            margin:0;
            font-family:Arial,sans-serif;
        ">
            <button id="copy_{card_key}" title="Copy PSS name"
                style="
                    height:29px;
                    min-width:34px;
                    cursor:pointer;
                    border:1px solid #dbeafe;
                    border-radius:8px;
                    background:#eff6ff;
                    color:#1d4ed8;
                    font-size:15px;
                ">⚡</button>
            <span id="msg_{card_key}" style="
                display:none;
                margin:7px 0 0 4px;
                color:#047857;
                font-size:9px;
                font-weight:700;
            ">Copied</span>
        </div>

        <script>
        const btn = document.getElementById("copy_{card_key}");
        const msg = document.getElementById("msg_{card_key}");
        const pssName = {js_text};

        btn.addEventListener("click", async () => {{
            try {{
                if (navigator.clipboard && window.isSecureContext) {{
                    await navigator.clipboard.writeText(pssName);
                }} else {{
                    const el = document.createElement("textarea");
                    el.value = pssName;
                    el.style.position = "fixed";
                    el.style.left = "-9999px";
                    document.body.appendChild(el);
                    el.select();
                    const ok = document.execCommand("copy");
                    el.remove();

                    if (!ok) throw new Error("Copy failed");
                }}

                btn.textContent = "✓";
                msg.style.display = "inline";

                setTimeout(() => {{
                    btn.textContent = "⚡";
                    msg.style.display = "none";
                }}, 1200);
            }} catch (e) {{
                btn.textContent = "!";
                setTimeout(() => btn.textContent = "⚡", 1200);
            }}
        }});
        </script>
        """,
        height=34,
        scrolling=False,
    )


def render_pss_card(row, columns, card_key):
    pss_name = clean_text(row[columns["pss"]])

    state = (
        clean_text(row[columns["state"]])
        if columns["state"] else "—"
    )

    capacity = (
        display_capacity(row[columns["capacity"]])
        if columns["capacity"] else "—"
    )

    poc1 = (
        clean_text(row[columns["poc1"]])
        if columns["poc1"] else "—"
    )

    # Coordinates are displayed in the expanded details.
    # The location pin indicates the location/state field.
    with st.container(border=True):
        title_col, copy_col = st.columns(
            [5.7, 0.8],
            vertical_alignment="center",
            gap="small",
        )

        with title_col:
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

        st.markdown(
            f"""
            <div class="pss-info-grid">
                <div class="pss-info capacity-tile">
                    <div class="pss-info-label">⚡ Capacity</div>
                    <div class="pss-info-value">{safe_html(capacity)}</div>
                </div>

                <div class="pss-info location-tile">
                    <div class="pss-info-label">📍 Location</div>
                    <div class="pss-info-value">{safe_html(state)}</div>
                </div>
            </div>

            <div class="poc-tile">
                <div class="pss-info-label">👤 POC 1</div>
                <div class="pss-info-value">{safe_html(poc1)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View site details", expanded=False):
            left, right = st.columns(2, gap="small")

            with left:
                detail_item(
                    "Priority",
                    row[columns["priority"]]
                    if columns["priority"] else "—",
                )
                detail_item(
                    "POC 2",
                    row[columns["poc2"]]
                    if columns["poc2"] else "—",
                )
                detail_item(
                    "📍 Latitude",
                    row[columns["lat"]]
                    if columns["lat"] else "—",
                )

            with right:
                detail_item(
                    "POC 3",
                    row[columns["poc3"]]
                    if columns["poc3"] else "—",
                )
                detail_item(
                    "📍 Longitude",
                    row[columns["long"]]
                    if columns["long"] else "—",
                )
                detail_item(
                    "Check Status",
                    row[columns["status"]]
                    if columns["status"] else "—",
                    status=True,
                )


# =========================================================
# LOAD EXCEL FILE
# =========================================================

@st.cache_data(show_spinner="Loading intimation data...")
def load_data(file_path, modified_time):
    return pd.read_excel(file_path)


file_path = Path(__file__).resolve().parent.parent / "intimation.xlsx"

if not file_path.exists():
    st.error(
        f"Excel file not found: {file_path}. "
        "Ensure intimation.xlsx is present in the project root."
    )
    st.stop()

try:
    df = load_data(str(file_path), file_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Unable to read intimation.xlsx: {exc}")
    st.stop()

if df.empty:
    st.warning("The Excel file contains no records.")
    st.stop()


# =========================================================
# COLUMN MAPPING
# =========================================================

all_columns = list(df.columns)

columns = {
    "pss": find_column(
        all_columns,
        [
            "PSS",
            "PSS Name",
            "Name of PSS",
            "Substation",
            "Substation Name",
        ],
    ),
    "state": find_column(
        all_columns,
        ["State", "PSS State", "State Name"],
    ),
    "capacity": find_column(
        all_columns,
        [
            "Capacity (MW)",
            "Capacity MW",
            "Capacity",
            "PSS Capacity (MW)",
        ],
    ),
    "poc1": find_column(
        all_columns,
        ["POC1", "POC 1", "POC-1"],
    ),
    "poc2": find_column(
        all_columns,
        ["POC 2", "POC2", "POC-2"],
    ),
    "poc3": find_column(
        all_columns,
        ["POC 3", "POC3", "POC-3"],
    ),
    "lat": find_column(
        all_columns,
        ["Lat", "Latitude"],
    ),
    "long": find_column(
        all_columns,
        ["Long", "Longitude", "Lon"],
    ),
    "status": find_column(
        all_columns,
        ["Check Status", "Status", "Intimation Status"],
    ),
}

columns["priority"] = find_column(
    all_columns,
    [
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
    st.error("PSS column could not be identified in the Excel file.")
    st.write("Available columns:", all_columns)
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="header-label">SOLAR SUITE · OPERATIONS</div>
        <h1>⚡ Intimation Dashboard</h1>
        <p>Search PSS sites, filter records and inspect site details.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SEARCH AND DEPENDENT FILTERS
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-icon">⌕</span>
        Search & Filters
    </div>
    <div class="filter-caption">
        Type or select a PSS name. POC 1 and State options adjust
        to the selected filters.
    </div>
    """,
    unsafe_allow_html=True,
)

filter1, filter2, filter3 = st.columns(
    [2.0, 1.25, 1.25],
    gap="medium",
)

pss_values = unique_values(df[columns["pss"]])

with filter1:
    selected_pss = st.selectbox(
        "Search PSS",
        options=pss_values,
        index=None,
        placeholder="Type or select a PSS name...",
        key="intimation_pss_autocomplete",
    )

search_df = df.copy()

if selected_pss:
    search_df = search_df[
        search_df[columns["pss"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == selected_pss
    ]

with filter2:
    if columns["poc1"]:
        poc1_values = unique_values(search_df[columns["poc1"]])

        selected_poc1 = st.selectbox(
            "POC 1",
            options=poc1_values,
            index=None,
            placeholder="All POC 1",
            key="intimation_poc1_autocomplete",
        )
    else:
        selected_poc1 = None
        st.selectbox(
            "POC 1",
            options=[],
            placeholder="Column unavailable",
            disabled=True,
            key="intimation_poc1_unavailable",
        )

state_source = search_df.copy()

if selected_poc1 and columns["poc1"]:
    state_source = state_source[
        state_source[columns["poc1"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == selected_poc1
    ]

with filter3:
    if columns["state"]:
        state_values = unique_values(state_source[columns["state"]])

        selected_state = st.selectbox(
            "State",
            options=state_values,
            index=None,
            placeholder="All States",
            key="intimation_state_autocomplete",
        )
    else:
        selected_state = None
        st.selectbox(
            "State",
            options=[],
            placeholder="Column unavailable",
            disabled=True,
            key="intimation_state_unavailable",
        )

filtered_df = search_df.copy()

if selected_poc1 and columns["poc1"]:
    filtered_df = filtered_df[
        filtered_df[columns["poc1"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == selected_poc1
    ]

if selected_state and columns["state"]:
    filtered_df = filtered_df[
        filtered_df[columns["state"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == selected_state
    ]

filtered_df = filtered_df.reset_index(drop=True)


# =========================================================
# OVERVIEW METRICS
# =========================================================

record_count = len(filtered_df)

if columns["capacity"]:
    total_capacity = filtered_df[columns["capacity"]].apply(
        numeric_value
    ).sum()
    capacity_text = display_capacity(total_capacity)
else:
    capacity_text = "—"

metric1, metric2 = st.columns(2, gap="medium")

with metric1:
    st.markdown(
        f"""
        <div class="metric-card metric-records">
            <div class="metric-icon">▦</div>
            <div class="metric-label">Matching Records</div>
            <div class="metric-value">{record_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        f"""
        <div class="metric-card metric-capacity">
            <div class="metric-icon">⚡</div>
            <div class="metric-label">Total Capacity</div>
            <div class="metric-value">{safe_html(capacity_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PAGINATION
# =========================================================

PAGE_SIZE = 20

filter_signature = (
    selected_pss or "",
    selected_poc1 or "",
    selected_state or "",
)

if (
    st.session_state.get("intimation_filter_signature")
    != filter_signature
):
    st.session_state["intimation_filter_signature"] = filter_signature
    st.session_state["intimation_page"] = 1

total_pages = max(1, (record_count + PAGE_SIZE - 1) // PAGE_SIZE)

current_page = st.session_state.get("intimation_page", 1)
current_page = min(max(current_page, 1), total_pages)

st.session_state["intimation_page"] = current_page

start_index = (current_page - 1) * PAGE_SIZE
end_index = min(start_index + PAGE_SIZE, record_count)

page_df = filtered_df.iloc[start_index:end_index]


# =========================================================
# PSS CARDS
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-icon">▤</span>
        PSS Sites
    </div>
    """,
    unsafe_allow_html=True,
)

if record_count > 0:
    st.markdown(
        f"""
        <div class="results-caption">
            Showing <b>{start_index + 1}–{end_index}</b>
            of <b>{record_count:,}</b> matching sites
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = list(page_df.iterrows())

    for row_start in range(0, len(records), 2):
        card_columns = st.columns(2, gap="medium")

        for offset in range(2):
            position = row_start + offset

            if position >= len(records):
                continue

            original_index, row = records[position]

            with card_columns[offset]:
                render_pss_card(
                    row=row,
                    columns=columns,
                    card_key=f"site_{current_page}_{original_index}",
                )

        st.markdown(
            '<div style="height:4px"></div>',
            unsafe_allow_html=True,
        )

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
                disabled=current_page == 1,
                key="intimation_previous",
            ):
                st.session_state["intimation_page"] = current_page - 1
                st.rerun()

        with page_col:
            st.markdown(
                f'<div class="page-label">Page {current_page} of {total_pages}</div>',
                unsafe_allow_html=True,
            )

        with next_col:
            if st.button(
                "Next →",
                use_container_width=True,
                disabled=current_page == total_pages,
                key="intimation_next",
            ):
                st.session_state["intimation_page"] = current_page + 1
                st.rerun()

else:
    st.markdown(
        """
        <div class="empty-state">
            <strong>🔎 No matching PSS sites</strong>
            Try another PSS name or change the POC 1 or State filter.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        margin-top:20px;
        padding-top:11px;
        border-top:1px solid #e2e8f0;
        color:#94a3b8;
        text-align:center;
        font-size:10px;
    ">
        SOLAR SUITE · INTIMATION DASHBOARD
    </div>
    """,
    unsafe_allow_html=True,
)
